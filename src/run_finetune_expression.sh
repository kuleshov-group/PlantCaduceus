#!/bin/bash
# ---------------------------------------------------------------------
# run_finetune_expression.sh
# ---------------------------------------------------------------------
# LoRA fine-tune of a PlantCAD2 checkpoint to predict gene expression
# counts from DNA sequence (regression head, num_labels=1).
#
# Reuses the existing, already-tested downstream-task pipeline in
# src/lora_fine_tune.py (the same script behind the released
# cross_species_leaf_absolute_expression_plantcad2_* models — see
# docs/PlantCAD2-overview.md) rather than a new training loop. This
# script only prepares paths/hyperparameters and calls it twice:
#   1. `lora_fine_tune.py tokenize` on the train/val TSVs
#   2. `lora_fine_tune.py train --task_type regression`
#
# How to run:
#   1. (one-time) Build the (sequence, label) TSVs. --counts-tsv accepts one or
#      more per-exon count TSVs sharing the same exon ID space (e.g. separate
#      studies); their experiment columns are merged and averaged together:
#        python src/scripts/build_expression_dataset.py \
#            --counts-tsv data/expression_7.tsv data/expression_72.tsv data/expression_85.tsv \
#            --gff data/GCF_002870075.5_Lsat_Salinas_v15_genomic.gff \
#            --fasta data/GCF_002870075.5_Lsat_Salinas_v15_genomic.fna \
#            --output-prefix data/expression_dataset
#      These count TSVs can be large; add --max-rows-per-tsv N for a fast
#      smoke-test build instead of processing the full files.
#
#   2. (recommended) Smoke-test the pipeline first:
#        SANITY=1 bash src/run_finetune_expression.sh
#
#   3. Full run:
#        bash src/run_finetune_expression.sh
#
#   4. To resume an interrupted run, set RESUME_FROM to a checkpoint path
#      and RUN_NAME to the same run directory it lives under (see the
#      "Resume from checkpoint" section below).
#
# Hardware knobs (env-var overridable, defaults below match this repo's
# 2080 Ti box). Labels are log1p(count) (see build_expression_dataset.py);
# lora_fine_tune.py predict's `predicted_value` column is therefore in
# log1p-space -- recover the integer count with:
#     round(max(0, expm1(predicted_value)))
# ---------------------------------------------------------------------

set -euo pipefail

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

# --- Paths --------------------------------------------------------------
# NOTE: these assume you cd into the plantcad/ repo before running.
MODEL_PATH="${MODEL_PATH:-./model/plantcad2_large_lettuce_20260719_010455}"
DATA_PREFIX="${DATA_PREFIX:-./data/expression_dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./model}"

# --- Resume from checkpoint -------------------------------------------
# Set RESUME_FROM to a checkpoint path to continue an interrupted run, and
# RUN_NAME to the *same* run directory name that checkpoint lives under
# (RUN_NAME defaults to a fresh timestamp otherwise, which would start a
# new run dir instead of continuing the old one). Example:
#   RESUME_FROM=./model/plantcad2_lettuce_expression_20260720_101500/checkpoint-500 \
#   RUN_NAME=plantcad2_lettuce_expression_20260720_101500 \
#   bash src/run_finetune_expression.sh
RESUME_FROM="${RESUME_FROM:-}"

RUN_NAME="${RUN_NAME:-plantcad2_lettuce_expression_$(date +%Y%m%d_%H%M%S)}"
echo "Run name: $RUN_NAME"
echo "Output dir: $OUTPUT_ROOT/$RUN_NAME"

# --- Hardware / hyperparameter knobs -------------------------------------
# Defaults below are known-safe on a single RTX 2080 Ti (11GB, sm_75/Turing):
# no bf16, fp32 only (src/INSTRUCTIONS.md ss7), sequences capped at 1024bp,
# batch size 1 (same value src/run_finetune_v2.sh found to be the max that
# fits PlantCAD2-Small + 1024bp + LoRA in fp32 on 11GB), effective batch 64
# via gradient accumulation.
#
# To run on a 4x H100 box instead (Hopper/sm_90, 80GB/GPU, native bf16),
# export these before invoking the script:
#   export WINDOW_SIZE=8192       # native PlantCAD2 context, fits easily
#   export BF16=True              # Hopper supports bf16 natively
#   export TRAIN_BS=32
#   export GRAD_ACCUM=2
#   export NPROC_PER_NODE=4       # true DDP across all 4 H100s
WINDOW_SIZE="${WINDOW_SIZE:-1024}"    # must match build_expression_dataset.py --window-size
BF16="${BF16:-False}"
TRAIN_BS="${TRAIN_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-64}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}" # >1 launches training via torchrun for DDP

# When running single-process (NPROC_PER_NODE=1), pin to exactly one GPU.
# Without this, HF Trainer sees every GPU the container can see and silently
# wraps the model in naive nn.DataParallel, which concentrates extra memory
# on GPU 0 (gradient/output gathering) and OOMs well before a true
# single-GPU run would. Multi-GPU training should go through the
# NPROC_PER_NODE>1 + torchrun (DDP) path below instead.
if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
SEED="${SEED:-42}"

# --- Known issue: intermittent divergence on this backbone/hardware -------
# src/lora_fine_tune.py's LoRA + regression-head wrap (AutoModelForSequenceClassification
# + PEFT) was found during verification to sometimes produce a NaN/inf forward
# pass on this exact checkpoint at fp32 -- reproducible by direct probing of
# load_base_model()/create_peft_model() outside this script entirely, so it is
# a property of that existing model/PEFT/Mamba2-Triton-on-Turing combination,
# not of the data pipeline here. Repeated identical-seed runs did not always
# reproduce the same outcome, so this looks like genuine kernel-level
# nondeterminism (Triton autotuning / fp32 Mamba2-SSD on sm_75) rather than
# something a fixed --seed reliably controls. In a sweep of 20 attempts, ~60%
# gave a stable finite forward pass and ~40% diverged to NaN/inf immediately
# (visible within the first couple of logged steps: eval_loss or grad_norm =
# inf/nan). If that happens:
#   - stop the run and just retry (same or different SEED) -- most attempts
#     are stable
#   - check the first 1-2 logged steps before trusting a longer run
#   - this class of Mamba2-SSD-Triton-kernel fragility on Turing (sm_75) is
#     already documented in src/INSTRUCTIONS.md ss7; it is expected to be far
#     less likely on an Ampere/Hopper GPU (e.g. the 4x H100 box), which has
#     much more mature Triton/Mamba2 kernel support.
echo "NOTE: watch the first couple of logged steps -- if eval_loss/grad_norm show"
echo "as inf or nan, kill this run and retry (see script comments -- known"
echo "intermittent Mamba2/Triton-on-Turing instability, ~60% of attempts are stable)."

# --- Sanity-probe mode ----------------------------------------------------
# Set SANITY=1 to smoke-test the full tokenize+train path on a small slice
# before committing to a real run.
SANITY="${SANITY:-0}"
if [[ "$SANITY" == "1" ]]; then
    echo "=== SANITY MODE: 200 examples/split, 30 steps ==="
    TRAIN_TSV="$(mktemp --suffix=.tsv)"
    VAL_TSV="$(mktemp --suffix=.tsv)"
    head -n 201 "${DATA_PREFIX}_train.tsv" > "$TRAIN_TSV"
    head -n 51  "${DATA_PREFIX}_val.tsv"   > "$VAL_TSV"
    MAX_STEPS_FLAG="--max_steps 30"
    EVAL_STEPS=15
    SAVE_STEPS=30
    LOGGING_STEPS=3
else
    echo "=== FULL RUN ==="
    TRAIN_TSV="${DATA_PREFIX}_train.tsv"
    VAL_TSV="${DATA_PREFIX}_val.tsv"
    MAX_STEPS_FLAG="--max_steps -1"
    EVAL_STEPS=100
    SAVE_STEPS=100
    LOGGING_STEPS=20
fi

TRAIN_PARQUET="$(mktemp --suffix=.parquet)"
VAL_PARQUET="$(mktemp --suffix=.parquet)"

# --- 1. Tokenize (fixed-length, regression labels) ------------------------
python src/lora_fine_tune.py tokenize \
    --data_dir "$TRAIN_TSV" --task_type regression \
    --seq_column sequence --label_column label \
    --sequence_length "$WINDOW_SIZE" --model_name "$MODEL_PATH" \
    --output_path "$TRAIN_PARQUET"

python src/lora_fine_tune.py tokenize \
    --data_dir "$VAL_TSV" --task_type regression \
    --seq_column sequence --label_column label \
    --sequence_length "$WINDOW_SIZE" --model_name "$MODEL_PATH" \
    --output_path "$VAL_PARQUET"

# --- 2. LoRA fine-tune with a regression head ------------------------------
LAUNCH=(python)
if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
    LAUNCH=(torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port=29501)
fi

"${LAUNCH[@]}" src/lora_fine_tune.py train \
    --model_name "$MODEL_PATH" --task_type regression \
    --train_dir "$TRAIN_PARQUET" --valid_dir "$VAL_PARQUET" \
    --output_dir "$OUTPUT_ROOT/$RUN_NAME" \
    --bf16 "$BF16" \
    --train_batch_size "$TRAIN_BS" --eval_batch_size "$TRAIN_BS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LEARNING_RATE" --warmup_steps 50 --lr_scheduler_type linear \
    --num_train_epochs "$NUM_EPOCHS" --weight_decay 0.01 \
    --eval_strategy steps --eval_steps "$EVAL_STEPS" \
    --save_strategy steps --save_steps "$SAVE_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --seed "$SEED" \
    $MAX_STEPS_FLAG \
    --remove_unused_columns False \
    ${RESUME_FROM:+--resume_from_checkpoint "$RESUME_FROM"}

echo ""
echo "Done. Run directory: $OUTPUT_ROOT/$RUN_NAME"
echo ""
echo "TensorBoard logs: $OUTPUT_ROOT/$RUN_NAME/tensorboard/"
echo "  View with:  tensorboard --logdir $OUTPUT_ROOT/"
echo "  (requires 'pip install tensorboard' once per container, per src/INSTRUCTIONS.md)"
echo ""
echo "To predict on new tokenized data and recover integer counts:"
echo "  python src/lora_fine_tune.py predict --checkpoint_dir $OUTPUT_ROOT/$RUN_NAME \\"
echo "      --model_name $MODEL_PATH --data_dir <tokenized>.parquet --task_type regression \\"
echo "      --output_file predictions.csv"
echo "  # then per row: predicted_count = round(max(0, expm1(predicted_value)))"
