import os
import json
import logging
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Tuple

import fire
import numpy as np
import pandas as pd
import torch
import multiprocessing
from datasets import Dataset, load_dataset

from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftModelForSequenceClassification,
    PeftConfig,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from scipy.stats import spearmanr, pearsonr


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------------------------

def tokenize(
    data_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    model_name: Optional[str] = None,
    sequence_length: int = 8192,
    batch_size: int = 1000,
    max_batches: Optional[int] = None,
    num_proc: Optional[int] = None,
    task_type: str = "classification",
    # HF dataset options
    hf_dataset: Optional[str] = None,
    hf_config: Optional[str] = None,
    hf_split: str = "train",
    # Column names
    seq_column: str = "sequence",
    label_column: str = "label",
) -> None:
    """Tokenize sequences from a local TSV or a Hugging Face dataset and save to parquet.

    Inputs
    - local TSV: pass data_dir=path/to/file.tsv (expects columns `Sequence`, `Label`; case-insensitive)
    - HF dataset: pass hf_dataset='namespace/dataset' and optionally hf_config, hf_split

    Labels
    - multi_label: Label can be a string of 0/1 (e.g., '0101...') or a list[int]
    - classification/regression: Label should be numeric
    """
    if model_name is None:
        raise ValueError("model_name must be provided to load the tokenizer")
    if data_dir is None and hf_dataset is None:
        raise ValueError("Provide either data_dir (local TSV) or hf_dataset (Hugging Face)")

    if num_proc is None:
        num_proc = multiprocessing.cpu_count()

    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load dataset from source
    if hf_dataset is not None:
        logger.info(f"Loading HF dataset: {hf_dataset} (config={hf_config}, split={hf_split})")
        dataset = load_dataset(hf_dataset, hf_config, split=hf_split)
    else:
        if output_path is None:
            output_path = str(Path(data_dir).with_suffix(".parquet"))
        logger.info(f"Loading local TSV: {data_dir}")
        if task_type == "multi_label":
            dtype = {label_column: str, label_column.capitalize(): str, label_column.upper(): str}
            df = pd.read_csv(data_dir, sep="\t", dtype=dtype)
            dataset = Dataset.from_pandas(df, preserve_index=False)
        else:
            dataset = Dataset.from_csv(data_dir, sep="\t")

    # normalize column names to lowercase
    for c in list(dataset.column_names):
        new_name = c.lower()
        if c != new_name:
            dataset = dataset.rename_column(c, new_name)
    # also normalize provided column names
    seq_column = seq_column.lower()
    label_column = label_column.lower()

    # optionally subsample for quick tests
    if max_batches is not None:
        max_examples = max_batches * batch_size
        dataset = dataset.select(range(min(max_examples, len(dataset))))
        logger.info(f"Limited dataset to {max_examples} examples ({max_batches} batches)")

    # Validate that all sequences match the specified sequence_length
    logger.info(f"Validating that all sequences are exactly {sequence_length} characters long...")
    seq_lengths = set(len(str(seq)) for seq in dataset[seq_column])
    if seq_lengths != {sequence_length}:
        raise SystemExit(
            f"ERROR: Not all sequences are {sequence_length} characters long.\n"
            f"  Found sequence lengths: {sorted(seq_lengths)}\n"
            f"  Please specify the correct --sequence_length to match your data."
        )

    def _to_label_list(val):
        if isinstance(val, str):
            return [int(c) for c in val]
        if isinstance(val, (list, tuple, np.ndarray)):
            return [int(x) for x in val]
        # some datasets may store as bytes; try decode
        try:
            s = str(val)
            return [int(c) for c in s]
        except Exception:
            raise ValueError(f"Unsupported multi_label value type: {type(val)} -> {val}")

    def tokenize_batch(examples):
        if seq_column not in examples:
            raise KeyError(f"Missing sequence column '{seq_column}' in dataset; set --seq_column if different")
        tokenized = tokenizer(
            [str(seq) for seq in examples[seq_column]],
            padding="max_length",
            truncation=True,
            max_length=sequence_length,
            add_special_tokens=False,
        )

        # sanity check: all sequences should be equal length after tokenization
        lengths = set(map(len, tokenized["input_ids"]))
        if lengths != {sequence_length}:
            raise ValueError(
                f"All sequences must be of length {sequence_length}; found batch lengths {lengths}"
            )

        result = {"input_ids": tokenized["input_ids"]}
        if task_type == "multi_label":
            if label_column not in examples:
                raise KeyError(f"Missing label column '{label_column}' for multi_label tasks")
            label_list = [_to_label_list(label) for label in examples[label_column]]
            result["labels"] = label_list
            remove_cols = [seq_column, label_column]
        else:
            if label_column in examples:
                result["label"] = examples[label_column]
            remove_cols = [seq_column]
        return result

    logger.info(f"Tokenizing sequences in batches of {batch_size}")
    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=[c for c in [seq_column, label_column] if c in dataset.column_names],
        num_proc=num_proc,
    )

    if output_path is None:
        # when hf_dataset used and output not provided, build a default name
        if hf_dataset is not None:
            src_name = hf_dataset.replace("/", "_")
            cfg = f"_{hf_config}" if hf_config else ""
            output_path = f"{src_name}{cfg}_{hf_split}_tokenized.parquet"
        else:
            src_name = Path(data_dir).stem
            output_path = f"{src_name}_{hf_split}_tokenized.parquet"

    logger.info(f"Saving tokenized dataset to {output_path}")
    tokenized_dataset.to_parquet(output_path, compression="zstd")
    logger.info("Tokenization complete")


# ------------------------------------------------------------------------------
# Display model + LoRA
# ------------------------------------------------------------------------------

def display(
    model_name: str,
    task_type: str = "classification",
    num_labels: Optional[int] = None,
) -> None:
    """Display a LoRA-adapted model structure and parameter trainability.

    Parameters
    - model_name: HF model id or path
    - task_type: classification | regression | multi_label (default: classification)
    - num_labels: required for multi_label
    """
    if task_type not in {"classification", "regression", "multi_label"}:
        raise ValueError("task_type must be one of {'classification','regression','multi_label'}")
    if task_type == "multi_label" and (num_labels is None or num_labels <= 1):
        raise ValueError("For multi_label, please provide num_labels > 1")

    logger.info(f"Loading base model from {model_name} for task_type={task_type}")
    base_model = load_base_model(model_name=model_name, task_type=task_type, num_labels=num_labels if task_type == "multi_label" else num_labels)

    logger.info("Configuring LoRA adapter")
    model = create_peft_model(base_model)

    logger.info(f"Model structure:\n{model}")
    logger.info("Model parameters:")
    parameters = []
    for n, p in model.named_parameters():
        parameters.append(
            dict(
                name=n,
                is_trainable=p.requires_grad,
                shape=tuple(p.shape) if hasattr(p, "shape") else None,
                size=p.numel(),
            )
        )

    # pretty log
    col_widths = {
        "name": max(len("Name"), max(len(str(p["name"])) for p in parameters)),
        "is_trainable": max(len("Trainable"), max(len(str(p["is_trainable"])) for p in parameters)),
        "shape": max(len("Shape"), max(len(str(p["shape"])) for p in parameters)),
        "size": max(len("Size"), max(len(str(p["size"])) for p in parameters)),
    }
    col_widths = {k: v + 2 for k, v in col_widths.items()}
    header = (
        f"{'Name':<{col_widths['name']}} "
        f"{'Trainable':<{col_widths['is_trainable']}} "
        f"{'Shape':<{col_widths['shape']}} "
        f"{'Size':<{col_widths['size']}}"
    )
    logger.info(header)
    logger.info("-" * (sum(col_widths.values()) + len(col_widths) - 1))
    for param in parameters:
        row = (
            f"{param['name']:<{col_widths['name']}} "
            f"{str(param['is_trainable']):<{col_widths['is_trainable']}} "
            f"{str(param['shape']):<{col_widths['shape']}} "
            f"{param['size']:<{col_widths['size']}}"
        )
        logger.info(row)


# ------------------------------------------------------------------------------
# Train / Evaluate / Predict
# ------------------------------------------------------------------------------

def train(
    train_dir: str,
    valid_dir: str,
    output_dir: str = "/tmp/pcv2-ft",
    model_name: Optional[str] = None,
    task_type: str = "classification",  # classification | regression | multi_label
    num_labels: Optional[int] = None,    # required for multi_label
    train_batch_size: int = 43,
    eval_batch_size: int = 3,
    eval_num_samples: Optional[int] = 0,
    max_steps: int = 500,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "pcv2-ft",
    wandb_run_name: Optional[str] = None,
    learning_rate: float = 1e-3,
    warmup_steps: int = 50,
    lr_scheduler_type: str = "linear",
    gradient_accumulation_steps: int = 64,
    bf16: bool = True,
    num_train_epochs: int = 3,
    weight_decay: float = 0.01,
    eval_strategy: str = "steps",
    eval_steps: int = 25,
    save_strategy: str = "steps",
    save_steps: int = 100,
    logging_steps: int = 100,
    remove_unused_columns: bool = False,
    resume_from_checkpoint: Optional[str] = None,
) -> None:
    """Fine-tune PlantCAD2 with LoRA across multiple task types."""
    if model_name is None:
        raise ValueError("model_name is required")

    if task_type not in {"classification", "regression", "multi_label"}:
        raise ValueError("task_type must be one of {'classification','regression','multi_label'}")

    if task_type == "multi_label" and (num_labels is None or num_labels <= 1):
        raise ValueError("For multi_label, please provide num_labels > 1")

    logger.info(f"Loading base model from {model_name}")
    base_model = load_base_model(model_name=model_name, task_type=task_type, num_labels=num_labels)

    logger.info("Configuring LoRA adapter")
    model = create_peft_model(base_model)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    logger.info("Trainable parameters: %d", trainable_params)
    logger.info("All parameters: %d", all_params)
    logger.info("Percent trainable: %.2f%%", trainable_params / all_params * 100)

    # Load datasets
    logger.info("Loading datasets")
    train_dataset = Dataset.from_parquet(str(Path(train_dir)), keep_in_memory=False)
    eval_dataset = Dataset.from_parquet(str(Path(valid_dir)), keep_in_memory=False)

    # For multi_label: ensure labels are stored as 'labels' column with numeric lists.
    # Handles: string labels ("000110101"), list labels, or renamed columns.
    if task_type == "multi_label":
        for name, ds in [("train", train_dataset), ("eval", eval_dataset)]:
            # Find the label column
            if "labels" in ds.column_names:
                lbl_col = "labels"
            elif "label" in ds.column_names:
                lbl_col = "label"
            else:
                raise ValueError(f"No 'label' or 'labels' column found in {name} dataset")

            # Check actual data format from first sample
            sample_label = ds[0][lbl_col]
            if isinstance(sample_label, str):
                logger.info(f"Converting string labels to numeric lists in {name} dataset")
                def _convert_str(examples, col=lbl_col):
                    return {"labels": [[int(c) for c in str(v)] for v in examples[col]]}
                remove = [lbl_col] if lbl_col != "labels" else []
                ds = ds.map(_convert_str, batched=True, remove_columns=remove)
            elif isinstance(sample_label, (list, np.ndarray)):
                if lbl_col != "labels":
                    ds = ds.rename_column(lbl_col, "labels")
                logger.info(f"Labels in {name} dataset are already lists (len={len(sample_label)})")
            elif isinstance(sample_label, (int, np.integer)):
                if num_labels is None:
                    raise ValueError(
                        f"Labels in {name} are scalars (type=int) — likely multi-label strings "
                        f"parsed as integers. Provide --num_labels so we can zero-pad them back."
                    )
                logger.info(
                    f"Converting int labels to {num_labels}-digit zero-padded binary vectors "
                    f"in {name} dataset (e.g. int 0 -> '{'0' * num_labels}' -> [0]*{num_labels})"
                )
                def _convert_int(examples, col=lbl_col, nl=num_labels):
                    return {"labels": [[int(c) for c in str(v).zfill(nl)] for v in examples[col]]}
                remove = [lbl_col] if lbl_col != "labels" else []
                ds = ds.map(_convert_int, batched=True, remove_columns=remove)
            else:
                raise ValueError(
                    f"Labels in {name} dataset have unsupported type ({type(sample_label).__name__}). "
                    f"Expected string, list of ints, or int (zero-padded multi-label)."
                )

            # Verify label dimension matches num_labels
            verified_sample = ds[0]["labels"]
            sample_len = len(verified_sample) if isinstance(verified_sample, (list, np.ndarray)) else None
            if sample_len is not None and num_labels is not None and sample_len != num_labels:
                raise ValueError(
                    f"Label length mismatch in {name}: each label has {sample_len} elements "
                    f"but num_labels={num_labels}"
                )

            if name == "train":
                train_dataset = ds
            else:
                eval_dataset = ds

    # Remove metadata columns the collator can't handle
    for name, ds in [("train", train_dataset), ("eval", eval_dataset)]:
        extra_cols = [c for c in ds.column_names if c not in {"input_ids", "label", "labels", "attention_mask"}]
        if extra_cols:
            logger.info(f"Removing extra columns from {name} dataset: {extra_cols}")
            ds = ds.remove_columns(extra_cols)
            if name == "train":
                train_dataset = ds
            else:
                eval_dataset = ds


    logger.info(f"Train dataset: {train_dataset}")
    logger.info(f"Eval dataset: {eval_dataset}")

    if eval_num_samples:
        logger.info(f"Limiting eval dataset to {eval_num_samples} samples")
        eval_dataset = eval_dataset.select(range(min(eval_num_samples, len(eval_dataset))))

    if use_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        bf16=bf16,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        logging_steps=logging_steps,
        report_to="none" if not use_wandb else "wandb",
        run_name=wandb_run_name,
        remove_unused_columns=remove_unused_columns,
        save_total_limit=5,
        seed=seed,
    )

    compute = (
        compute_metrics_multilabel
        if task_type == "multi_label"
        else (compute_metrics_classification if task_type == "classification" else compute_metrics_regression)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute,
    )

    if resume_from_checkpoint is not None:
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()


def evaluate(
    checkpoint_dir: str,
    data_dir: str,
    output_dir: str = "/tmp/pcv2-ft-eval",
    model_name: Optional[str] = None,
    task_type: str = "classification",
    num_labels: Optional[int] = None,
    batch_size: int = 32,
    bf16: bool = True,
    sampling_rate: Optional[float] = None,
    seed: int = 42,
) -> None:
    """Evaluate a LoRA checkpoint (local dir or HF repo) on validation data."""

    if task_type == "multi_label" and (num_labels is None or num_labels <= 1):
        raise ValueError("For multi_label, please provide num_labels > 1")

    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_path.exists():
        if model_name is None:
            raise ValueError("model_name is required when loading a local checkpoint directory")
        logger.info(f"Loading local LoRA checkpoint from: {checkpoint_dir}")
        base_model = load_base_model(model_name=model_name, task_type=task_type, num_labels=num_labels)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    else:
        logger.info(f"Loading LoRA adapter from Hugging Face: {checkpoint_dir}")
        model = load_model_from_hf(checkpoint_dir, task_type=task_type, num_labels=num_labels)

    logger.info(f"Loading dataset from {data_dir}")
    eval_dataset = Dataset.from_parquet(data_dir, keep_in_memory=False)

    

    if sampling_rate:
        if sampling_rate > 1 or sampling_rate <= 0:
            raise ValueError("sampling_rate must be in (0, 1]")
        max_index = min(int(sampling_rate * len(eval_dataset)), len(eval_dataset))
        max_index = max(max_index, 1)
        eval_dataset = eval_dataset.shuffle(seed=seed).select(range(max_index))

    trainer_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
        bf16=bf16,
    )
    compute = (
        compute_metrics_multilabel
        if task_type == "multi_label"
        else (compute_metrics_classification if task_type == "classification" else compute_metrics_regression)
    )
    trainer = Trainer(model=model, args=trainer_args, compute_metrics=compute)

    logger.info("Evaluating model...")
    results = trainer.evaluate(eval_dataset=eval_dataset)
    logger.info("Evaluation complete")
    logger.info(f"Results:\n{results}")


def predict(
    checkpoint_dir: str,
    data_dir: str,
    output_file: str = "/tmp/predictions.csv",
    model_name: Optional[str] = None,
    task_type: str = "classification",
    bf16: bool = True,
    num_labels: Optional[int] = None,
    batch_size: int = 32,
    sampling_rate: Optional[float] = None,
    seed: int = 42,
) -> None:
    """Generate predictions and save to CSV.

    - classification: outputs probability of positive class in column 'probability_positive'.
    - regression: outputs predicted_value.
    - multi_label: outputs one probability column per class: class_0, class_1, ...
    """
    if task_type == "multi_label" and (num_labels is None or num_labels <= 1):
        raise ValueError("For multi_label, please provide num_labels > 1")

    # Determine whether checkpoint_dir is a local path or a HF repo id
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_path.exists():
        if model_name is None:
            raise ValueError("model_name is required when loading a local checkpoint directory")
        logger.info(f"Loading local LoRA checkpoint from: {checkpoint_dir}")
        base_model = load_base_model(model_name=model_name, task_type=task_type, num_labels=num_labels)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    else:
        logger.info(f"Loading LoRA adapter from Hugging Face: {checkpoint_dir}")
        model = load_model_from_hf(checkpoint_dir, task_type=task_type, num_labels=num_labels)

    logger.info(f"Loading dataset from {data_dir}")
    dataset = Dataset.from_parquet(data_dir, keep_in_memory=False)

    # For prediction, drop any label columns to avoid collator errors
    cols_to_keep = {"input_ids", "attention_mask"}
    cols_to_remove = [c for c in dataset.column_names if c not in cols_to_keep]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
    if "input_ids" not in dataset.column_names:
        raise ValueError(
            "Dataset must contain 'input_ids'. Tokenize your data first via `tokenize` and pass the resulting parquet."
        )

    

    if sampling_rate:
        if sampling_rate > 1 or sampling_rate <= 0:
            raise ValueError("sampling_rate must be in (0, 1]")
        max_index = min(int(sampling_rate * len(dataset)), len(dataset))
        max_index = max(max_index, 1)
        dataset = dataset.shuffle(seed=seed).select(range(max_index))

    trainer_args = TrainingArguments(
        output_dir="/tmp",
        per_device_eval_batch_size=batch_size,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
        bf16=bf16,
    )
    trainer = Trainer(model=model, args=trainer_args)

    logger.info("Generating predictions...")
    predictions = trainer.predict(test_dataset=dataset).predictions

    if task_type == "classification":
        probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=1).numpy()
        scores = probs[:, 1]
        df = pd.DataFrame({"probability_positive": scores})
    elif task_type == "regression":
        values = predictions.squeeze()
        df = pd.DataFrame({"predicted_value": values})
    else:  # multi_label
        probs = torch.sigmoid(torch.tensor(predictions)).numpy()
        df = pd.DataFrame(probs, columns=[f"class_{i}" for i in range(probs.shape[1])])

    output_path = Path(output_file)
    logger.info(f"Saving predictions to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info("Prediction scores saved successfully")


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def load_model_from_hf(hf_model_path: str, task_type: str, num_labels: Optional[int]):
    """Load a PEFT LoRA adapter hosted on Hugging Face by repo id or path.

    This resolves the correct base model from the adapter's PEFT config and
    constructs the base with the appropriate head for the specified task.
    """
    config = PeftConfig.from_pretrained(hf_model_path)
    base_model = load_base_model(
        model_name=config.base_model_name_or_path,
        task_type=task_type,
        num_labels=num_labels,
    )
    model = PeftModel.from_pretrained(base_model, hf_model_path)
    return model

def compute_metrics_classification(eval_pred):
    predictions, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=1)
    preds = np.argmax(predictions, axis=1)
    scores = probs[:, 1].numpy()

    balance = np.sum(labels) / len(labels)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
        "balance": balance,
    }


def compute_metrics_regression(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    mse = ((predictions - labels) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(predictions - labels).mean()
    ss_tot = ((labels - labels.mean()) ** 2).sum()
    ss_res = ((labels - predictions) ** 2).sum()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    pearson_corr, _ = pearsonr(predictions, labels)
    spearman_corr, _ = spearmanr(predictions, labels)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson_corr,
        "spearman_r": spearman_corr,
    }


def compute_metrics_multilabel(eval_pred):
    predictions, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(predictions)).numpy()
    preds = (probs > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="micro"),
        "roc_auc": roc_auc_score(labels, probs, average="micro"),
        "average_precision": average_precision_score(labels, probs, average="micro"),
    }


def resolve_base_checkpoint(model_name: str) -> str:
    """If `model_name` is a saved LoRA adapter checkpoint (not a full model),
    merge it into its base model and return a path to the merged checkpoint
    instead. This lets `model_name`/`--model_name` be pointed at either a
    plain base model or a previously-trained adapter interchangeably.

    Why merge rather than just re-attach the adapter as trainable: the
    adapter may have been trained for a different head/task (e.g. an MLM
    continual-pretrain adapter from HF_pre_train_lora.py, being reused as
    the base for a new classification/regression fine-tune here) -- merging
    bakes its weights into the backbone once, so a *fresh* LoRA adapter +
    task head can then be trained on top cleanly. Calling get_peft_model()
    directly on an already-PEFT-wrapped model instead stacks a second,
    un-merged adapter (PEFT warns "trying to modify a model with PEFT for a
    second time") and reproducibly diverges to NaN loss/grad_norm.
    """
    adapter_config_path = Path(model_name) / "adapter_config.json"
    if not adapter_config_path.exists():
        return model_name

    peft_config = PeftConfig.from_pretrained(model_name)
    base_path = peft_config.base_model_name_or_path
    # The base must be loaded with the *same* Auto* wrapper class the adapter
    # was originally trained against -- e.g. AutoModelForMaskedLM vs
    # AutoModelForSequenceClassification wrap the shared Caduceus backbone
    # under different module-path prefixes (`caduceus.backbone...` either
    # way, but via a different outer class), so PEFT's saved LoRA state dict
    # keys only line up against a matching wrapper. Using the wrong class
    # doesn't error -- PEFT just logs "missing adapter keys" and silently
    # merges in zero deltas, producing what looks like a normal but
    # untouched base model.
    adapter_cfg = json.loads(adapter_config_path.read_text())
    base_class_name = (adapter_cfg.get("auto_mapping") or {}).get("base_model_class")
    auto_cls = {
        "CaduceusForMaskedLM": AutoModelForMaskedLM,
        "CaduceusForSequenceClassification": AutoModelForSequenceClassification,
    }.get(base_class_name)
    if auto_cls is None:
        raise ValueError(
            f"Adapter at {model_name} has no recognized auto_mapping.base_model_class "
            f"(got {base_class_name!r}); don't know which model wrapper to merge it "
            f"against. Add a case for it in resolve_base_checkpoint()."
        )
    logger.info(f"Detected LoRA adapter checkpoint at {model_name} (base: {base_path}, "
                f"trained against {base_class_name}); merging")
    base = auto_cls.from_pretrained(base_path, trust_remote_code=True, torch_dtype=torch.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        peft_model = PeftModel.from_pretrained(base, model_name)
        if any("missing" in str(w.message).lower() and "key" in str(w.message).lower() for w in caught):
            raise RuntimeError(
                f"Loading the adapter at {model_name} reported missing keys against "
                f"{auto_cls.__name__} -- the merge would silently be a no-op. The "
                f"recorded base_model_class ({base_class_name}) may not match how "
                f"this adapter was actually trained."
            )
    merged = peft_model.merge_and_unload()

    merged_dir = tempfile.mkdtemp(prefix="merged_base_")
    # safe_serialization=False: Caduceus ties lm_head.weight to the input
    # embedding, which safetensors' shared-tensor check rejects (same reason
    # HF_pre_train_lora.py's Trainer uses save_safetensors=False).
    merged.save_pretrained(merged_dir, safe_serialization=False)
    AutoTokenizer.from_pretrained(base_path, trust_remote_code=True).save_pretrained(merged_dir)
    # trust_remote_code modeling files aren't always copied by save_pretrained;
    # make sure they're present so a later from_pretrained(merged_dir) works.
    for fname in ("configuration_caduceus.py", "modeling_caduceus.py", "modeling_rcps.py"):
        src, dst = Path(base_path) / fname, Path(merged_dir) / fname
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
    logger.info(f"Merged checkpoint written to {merged_dir} (not auto-deleted)")
    return merged_dir


def load_base_model(model_name: str, task_type: str, num_labels: Optional[int]) -> AutoModelForSequenceClassification:
    model_name = resolve_base_checkpoint(model_name)
    if task_type == "classification":
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1},
        )
    elif task_type == "regression":
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_labels=1,
            problem_type="regression",
        )
    else:  # multi_label
        if num_labels is None:
            raise ValueError("num_labels is required for multi_label classification")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )

    # Wrap forward to pass explicit kwargs HF expects
    original_forward = base_model.forward

    def forward_with_labels(*args, **kwargs):
        # ensure 'labels' key exists and typed correctly for multi-label
        labels = kwargs.get("labels", kwargs.get("label"))
        if labels is None:
            return original_forward(input_ids=kwargs["input_ids"])  # inference without labels
        if task_type == "multi_label":
            labels = labels.float()
            if labels.dim() == 1:
                raise ValueError(
                    f"Multi-label labels have wrong shape {tuple(labels.shape)}. "
                    f"Expected [batch_size, {num_labels}]. "
                    f"Ensure your dataset's label column contains lists of ints, not scalars."
                )
        return original_forward(input_ids=kwargs["input_ids"], labels=labels)

    base_model.forward = forward_with_labels
    return base_model


def create_peft_model(base_model: AutoModelForSequenceClassification) -> PeftModelForSequenceClassification:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["x_proj", "in_proj", "out_proj"],
    )
    return get_peft_model(base_model, peft_config)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(
        {
            "tokenize": tokenize,
            "train": train,
            "evaluate": evaluate,
            "predict": predict,
            "display": display,
        }
    )
