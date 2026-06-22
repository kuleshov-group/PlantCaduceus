#!/usr/bin/env python
# coding=utf-8
"""
HF_pre_train_lora.py
--------------------

LoRA-enabled masked-language-model fine-tuning for PlantCAD / PlantCAD2.

This is a slimmed-down, standalone alternative to HF_pre_train.py for v2
of the lettuce fine-tune. The v1 script kept growing patches; rather
than poking at it again, this script is purpose-built around three
deltas that the v1 retro flagged:

  1. PEFT/LoRA wrap of AutoModelForMaskedLM, with the trainable-%
     printed at startup so we can tune r/alpha based on real numbers.
  2. TensorBoard logging out of the box (no text-log scraping).
  3. Outputs land in `model/<run_name>` not `output/...`, with a
     timestamp baked into the run name.

What is preserved from v1:
  - Soft-masked-loss weighting (lowercase = repetitive sequence, down-
    weighted in the loss). Implemented here as a custom Trainer
    subclass so we don't depend on the upstream Caduceus model's
    forward() signature.
  - Cosine LR schedule with warmup.
  - DDP via torchrun (no model-parallel — PlantCAD2-Small fits per-GPU).

CLI: see argparse block at the bottom. Designed to be called from
run_finetune_v2.sh which centralises paths/hparams.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("hf_pre_train_lora")


# ----------------------------------------------------------------------
# Data collator
#
# Subclasses the stock MLM collator so we can pass through a
# `loss_weights` column (one float per token, derived from soft-masking)
# alongside `input_ids` and `labels`. The collator itself doesn't apply
# the weights — that happens in WeightedMLMTrainer.compute_loss below.
# Keeping the two concerns separate keeps the collator dumb and
# debuggable.
# ----------------------------------------------------------------------
class CollatorMLMWithLossWeights(DataCollatorForLanguageModeling):
    """Stock MLM collator + carries `loss_weights` through to the model."""

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # Stack every column the dataset produced — input_ids,
        # loss_weights, (optionally) special_tokens_mask, etc.
        batch = {
            k: torch.stack([torch.tensor(ex[k]) for ex in examples], dim=0)
            for k in examples[0].keys()
        }

        # Pop special_tokens_mask before passing to torch_mask_tokens so
        # MLM masking respects it (same behaviour as HF default).
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        # Standard MLM masking: 15% of tokens get [MASK]/random/kept,
        # `labels` is -100 everywhere except those positions.
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        # Make loss_weights float (parquet/arrow sometimes round-trips
        # as float64; keep it float32 to match logits dtype).
        if "loss_weights" in batch:
            batch["loss_weights"] = batch["loss_weights"].float()
        return batch


# ----------------------------------------------------------------------
# Trainer subclass that applies per-token loss weights
#
# Stock Trainer assumes the model returns a scalar loss. We instead
# compute cross-entropy ourselves so we can multiply by the per-token
# weights coming from the dataset (lowercase positions = 0.1 weight).
# ----------------------------------------------------------------------
class WeightedMLMTrainer(Trainer):
    """Trainer that applies per-token loss weights for soft-masked DNA."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pull our extra column out before the model sees `inputs` —
        # the underlying model.forward() doesn't accept it.
        loss_weights = inputs.pop("loss_weights", None)
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.logits  # [B, T, V]

        # Standard token-level cross-entropy, NO reduction so we can
        # weight per position.
        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        flat_loss = loss_fct(
            logits.view(-1, logits.size(-1)),     # [B*T, V]
            labels.view(-1),                       # [B*T]
        ).view(labels.size())                      # back to [B, T]

        if loss_weights is not None:
            # `loss_weights` is [B, T]; masked positions get -100 in
            # labels, so they're zeroed by ignore_index already.
            flat_loss = flat_loss * loss_weights.to(flat_loss.dtype)

        # Average across the masked positions that actually contributed
        # — i.e. wherever labels != -100. Using mean over those keeps
        # the scalar comparable run-to-run regardless of mask count.
        denom = (labels != -100).float().sum().clamp(min=1.0)
        if loss_weights is not None:
            # Numerator = sum of (weighted) token losses on masked pos.
            num = flat_loss.sum()
        else:
            num = flat_loss.sum()
        loss = num / denom

        return (loss, outputs) if return_outputs else loss


# ----------------------------------------------------------------------
# Tokenisation
#
# Mirrors HF_pre_train.py: build `loss_weights` from the casing of the
# input `seq` string (lowercase = repeat-masked, gets down-weighted).
# ----------------------------------------------------------------------
def make_tokenize_fn(tokenizer, soft_masked_weight: float, max_seq_length: Optional[int] = None):
    """Build a tokenisation function.

    If `max_seq_length` is set and smaller than the input sequence length,
    we take a *random* window of that length from each 8192 bp chunk.
    This is the v2 memory-fit trick — PlantCAD2 + Mamba2 SSD Triton kernel
    blows up activations at 8192 bp in fp32 on Turing GPUs (sm_75), and
    Mamba2's fp16 kernel won't compile for sm_75 at all. Random-window
    truncation keeps full genome coverage across the dataset (each
    epoch sees different windows of the same chunks) while halving or
    quartering activation memory.
    """
    def fn(examples):
        seqs = examples["seq"]
        if max_seq_length is not None and max_seq_length > 0:
            # Random offset per row; using np.random keeps determinism
            # tied to the global numpy seed set elsewhere.
            new_seqs = []
            for s in seqs:
                if len(s) > max_seq_length:
                    off = int(np.random.randint(0, len(s) - max_seq_length + 1))
                    new_seqs.append(s[off:off + max_seq_length])
                else:
                    new_seqs.append(s)
            seqs = new_seqs
        res = tokenizer(
            seqs,
            return_special_tokens_mask=True,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        # Per-token weights: 1.0 for upper-case bases, `soft_masked_weight`
        # for lower-case (repeat-masked) bases. We rebuild from the
        # (possibly truncated) `seqs` list so the lengths line up.
        weights = np.ones_like(res["input_ids"], dtype=np.float32)
        is_lower = np.char.islower([list(x) for x in seqs])
        weights[is_lower] = soft_masked_weight
        res["loss_weights"] = weights.tolist()
        return res
    return fn


# ----------------------------------------------------------------------
# LoRA wrap
#
# Caduceus is a Mamba-family bidirectional SSM. The trainable projection
# matrices inside each Mamba block are usually named in_proj / out_proj
# / x_proj / dt_proj. PEFT's LoraConfig accepts a list of *suffix*
# strings — any nn.Linear whose name ends with one of these gets a LoRA
# adapter. If your specific checkpoint uses different names, override
# via --lora_target_modules on the CLI.
# ----------------------------------------------------------------------
DEFAULT_TARGET_MODULES = ["in_proj", "out_proj", "x_proj", "dt_proj"]


def wrap_with_lora(
    base_model: nn.Module,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: List[str],
):
    """Apply a LoRA adapter and log the trainable-parameter percentage."""
    lora_cfg = LoraConfig(
        # NOTE: task_type intentionally NOT set.
        # PEFT's task-typed wrappers (FEATURE_EXTRACTION, CAUSAL_LM, etc.)
        # all hard-code an `attention_mask=None` kwarg in their forward()
        # before delegating to the base model. Caduceus is a Mamba SSM
        # and its forward() does not accept `attention_mask` — so any
        # task_type produces `TypeError: got an unexpected keyword
        # argument 'attention_mask'`. Leaving task_type unset gives us
        # the generic PeftModel, which simply forwards *args, **kwargs
        # straight through. Loss computation lives in WeightedMLMTrainer
        # below, so we don't need PEFT's task-typed loss helpers anyway.
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)

    # `get_nb_trainable_parameters` returns (trainable, total) — print
    # the *percentage* per professor's feedback. This is the headline
    # number that justifies the LoRA choice.
    trainable, total = model.get_nb_trainable_parameters()
    pct = 100.0 * trainable / max(total, 1)
    log.info("LoRA wrapped.")
    log.info("  trainable params : %s", f"{trainable:,}")
    log.info("  total params     : %s", f"{total:,}")
    log.info("  trainable %%      : %.4f%%", pct)
    log.info("  target modules   : %s", target_modules)
    log.info("  r=%d, alpha=%d, dropout=%.3f", r, alpha, dropout)
    return model


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- Paths ---
    p.add_argument("--model_name_or_path", required=True,
                   help="Path to PlantCAD2-Small (or any AutoModelForMaskedLM)")
    p.add_argument("--dataset_path", required=True,
                   help="Path to a load_from_disk-compatible DatasetDict")
    p.add_argument("--output_root", default="model",
                   help="Parent dir for the run; final dir is <output_root>/<run_name>")
    p.add_argument("--run_name", default=None,
                   help="Run dir name; defaults to plantcad2_small_lettuce_<timestamp>")

    # --- Training hparams ---
    p.add_argument("--learning_rate", type=float, default=1e-3,
                   help="LoRA typically wants higher LR than full FT (1e-3 vs 5e-5).")
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Set explicitly OR use --num_train_epochs.")
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=32)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--soft_masked_loss_weights_train", type=float, default=0.1)
    p.add_argument("--soft_masked_loss_weights_evaluation", type=float, default=0.0)
    p.add_argument("--max_seq_length", type=int, default=None,
                   help="If set, truncate each input sequence to a random window of "
                        "this length. Needed on Turing GPUs where the Mamba2 SSD "
                        "kernel can't fit 8192 bp activations in 11 GB.")

    # --- LoRA ---
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", nargs="+",
                   default=DEFAULT_TARGET_MODULES,
                   help="nn.Linear name suffixes to LoRA-wrap.")

    # --- Eval / save / log cadence ---
    p.add_argument("--eval_strategy", default="steps")
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_strategy", default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=10)

    # --- Precision / DDP ---
    p.add_argument("--bf16", action="store_true",
                   help="Enable bf16 (DO NOT use on Turing/2080 Ti — they fake it and break).")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Caduceus historically does not support this; off by default.")
    p.add_argument("--ddp_find_unused_parameters", action="store_true",
                   help="Off by default; LoRA can introduce unused params if some target "
                        "modules don't exist in this model.")

    # --- Run mode ---
    p.add_argument("--do_train", action="store_true", default=True)
    p.add_argument("--do_eval", action="store_true", default=True)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Cap train rows for a sanity probe.")
    p.add_argument("--max_eval_samples", type=int, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # --- Resolve run name + output dir ---------------------------------
    # Timestamp the directory so re-runs never clobber each other and
    # `ls model/` reads as a tidy timeline.
    if args.run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"plantcad2_small_lettuce_{ts}"
    output_dir = Path(args.output_root) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run output dir: %s", output_dir)

    # --- Tokenizer + dataset ------------------------------------------
    log.info("Loading tokenizer from %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              trust_remote_code=True)

    log.info("Loading DatasetDict from %s", args.dataset_path)
    raw: DatasetDict = load_from_disk(args.dataset_path)
    if "train" not in raw or "validation" not in raw:
        raise SystemExit(f"DatasetDict at {args.dataset_path} must have "
                         f"'train' and 'validation' splits; got {list(raw.keys())}")

    if args.max_train_samples:
        raw["train"] = raw["train"].select(range(min(args.max_train_samples,
                                                     len(raw["train"]))))
    if args.max_eval_samples:
        raw["validation"] = raw["validation"].select(range(min(args.max_eval_samples,
                                                               len(raw["validation"]))))
    log.info("Final dataset sizes: train=%d, eval=%d",
             len(raw["train"]), len(raw["validation"]))

    # Columns to strip after tokenisation — anything that isn't
    # input_ids / loss_weights / special_tokens_mask.
    metadata_cols = [c for c in raw["train"].column_names if c != "seq"] + ["seq"]

    log.info("Tokenising (train weight=%.3f, eval weight=%.3f, max_seq_length=%s)",
             args.soft_masked_loss_weights_train,
             args.soft_masked_loss_weights_evaluation,
             args.max_seq_length)
    tok_train = raw["train"].map(
        make_tokenize_fn(tokenizer, args.soft_masked_loss_weights_train,
                         max_seq_length=args.max_seq_length),
        batched=True, remove_columns=metadata_cols,
    )
    tok_eval = raw["validation"].map(
        make_tokenize_fn(tokenizer, args.soft_masked_loss_weights_evaluation,
                         max_seq_length=args.max_seq_length),
        batched=True, remove_columns=metadata_cols,
    )

    # --- Model + LoRA -------------------------------------------------
    log.info("Loading base model from %s", args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    base_model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Turing-safe; bf16 silently broken on 2080 Ti.
    )
    model = wrap_with_lora(
        base_model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )

    # --- Collator -----------------------------------------------------
    collator = CollatorMLMWithLossWeights(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    # --- TrainingArguments -------------------------------------------
    # `report_to=["tensorboard"]` + `logging_dir` are the two knobs that
    # give the professor-friendly curves. Open with:
    #     tensorboard --logdir model/
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_dir=str(output_dir / "tensorboard"),
        report_to=["tensorboard"],
        bf16=args.bf16,
        fp16=args.fp16,
        tf32=False,
        seed=args.seed,
        save_safetensors=False,  # Caduceus tied-weight saving needs .bin
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        # `remove_unused_columns=False` is critical — without it, HF
        # silently strips our `loss_weights` column on the way into the
        # collator and we'd compute unweighted loss.
        remove_unused_columns=False,
        overwrite_output_dir=True,
    )

    # --- Trainer ------------------------------------------------------
    trainer = WeightedMLMTrainer(
        model=model,
        args=training_args,
        train_dataset=tok_train if args.do_train else None,
        eval_dataset=tok_eval if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # --- Train --------------------------------------------------------
    if args.do_train:
        log.info("Starting training")
        train_result = trainer.train()
        # Save the LoRA *adapter* (small — just the deltas). This is
        # what you'd ship for a downstream eval that re-loads the base
        # PlantCAD2-Small and overlays the adapter.
        trainer.save_model()
        # Also dump train metrics (loss curve summary, throughput, etc.)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # --- Eval ---------------------------------------------------------
    if args.do_eval:
        log.info("Running final eval")
        metrics = trainer.evaluate()
        # Perplexity = exp(eval_loss). With a 4-base alphabet, random
        # baseline is 4.0; lower is better.
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    log.info("Done. Artifacts in %s", output_dir)
    log.info("View TensorBoard:  tensorboard --logdir %s", args.output_root)


if __name__ == "__main__":
    sys.exit(main())
