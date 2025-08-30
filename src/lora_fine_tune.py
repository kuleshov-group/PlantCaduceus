import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import fire
import numpy as np
import pandas as pd
import torch
import multiprocessing
from datasets import Dataset, load_dataset

from transformers import (
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


def load_base_model(model_name: str, task_type: str, num_labels: Optional[int]) -> AutoModelForSequenceClassification:
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
