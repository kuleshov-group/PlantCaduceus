#!/bin/bash
# ---------------------------------------------------------------------
# run_finetune_v2.sh
# ---------------------------------------------------------------------
# Lettuce LoRA fine-tune of PlantCAD2-Small, v2 pipeline.
#
# What changed vs v1:
#   - Base model:    PlantCaduceus_l20      -> PlantCAD2-Small  (8192 bp ctx)
#   - Data:          raw FASTA windows      -> pre-curated 50k random chunks TSV
#   - Method:        full fine-tune         -> LoRA via PEFT
#   - Output dir:    output/...             -> model/plantcad2_small_lettuce_<ts>/
#   - Logging:       text logs              -> TensorBoard
#
# How to run:
#   1. (one-time) Convert the TSV to a HuggingFace DatasetDict:
#        python src/prepare_dataset_from_tsv.py \
#            --tsv data/GCF_002870075.5_Lsat_Salinas_v15_genomic_rand_chunk_8192_samples_50k.tsv \
#            --output_dir data/lettuce_hf_dataset_rand8192_50k
#
#   2. (optional but recommended) Sanity-probe the memory budget first:
#        SANITY=1 bash src/run_finetune_v2.sh
#      That caps to 200 steps and 1k samples — enough to confirm the
#      PlantCAD2-Small + 8192 bp + LoRA + fp32 setup fits on 11 GB.
#
#   3. Full ~1-epoch run:
#        bash src/run_finetune_v2.sh
#
#   4. Watch the curves:
#        tensorboard --logdir model/
# ---------------------------------------------------------------------

set -euo pipefail

# --- Environment ------------------------------------------------------
# Offline mode prevents the HF hub from being contacted at runtime
# (saves time on the cluster and avoids "no internet" hangs).
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Reduce memory fragmentation — useful when we're tight on the 11 GB
# 2080 Ti's. Has no downside on bigger GPUs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL stability knobs for consumer-grade GPUs that lack proper P2P
# and InfiniBand. Same flags that finally made v1 stop crashing.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Disable W&B (TensorBoard is our logging target now).
export WANDB_DISABLED=true

# --- Paths ------------------------------------------------------------
# NOTE: these assume you cd into the plantcad/ repo before running.
# Use the exact folder name as it sits in model/ — the suffix encodes
# the architecture (24 layers, hidden dim 768).
MODEL_PATH="./model/PlantCAD2-Small-l24-d0768"
DATASET_PATH="./data/lettuce_hf_dataset_rand8192_50k"

# Output root for all runs.
OUTPUT_ROOT="./model"

# Compute the run name ONCE in bash, before launching torchrun. This
# way every DDP rank inherits the same RUN_NAME env var and they all
# write to the same `model/<run_name>/` directory. Previously each
# rank called datetime.now() independently inside Python, which under
# 8-way parallelism could land on different seconds and create up to
# 8 sibling directories per launch — confusing and wasteful.
RUN_NAME="plantcad2_small_lettuce_$(date +%Y%m%d_%H%M%S)"
echo "Run name: $RUN_NAME"
echo "Output dir: $OUTPUT_ROOT/$RUN_NAME"

# --- Hyperparameters --------------------------------------------------
# LoRA wants a higher LR than full fine-tune; 1e-3 is the canonical
# starting point in the PEFT examples.
LEARNING_RATE=1e-3
WARMUP_STEPS=100
LR_SCHEDULER="cosine"

# Memory: PlantCAD2-Small @ 8192 bp on a 2080 Ti is tight. We use
# bs=1 per device and let gradient accumulation reach an effective
# batch of 64 (1 * 8 GPUs * 8 accum).
PER_DEVICE_BS=1
GRAD_ACCUM=8

# Precision: fp32. We tried --fp16, but Mamba2's SSD Triton kernel
# (mamba_split_conv1d_scan_combined) requires tensor-core instructions
# only available on sm_80+ (Ampere+) and fails to compile on the 2080
# Ti's sm_75 (Turing). fp32 path through the same kernel DOES compile
# on sm_75; we just can't afford 8192 bp activations in 11 GB. We deal
# with that by truncating sequences (see MAX_SEQ_LENGTH below).
PRECISION_FLAG=""

# Sequence-length cap. Dataset chunks are 8192 bp, but at fp32 on a
# 2080 Ti the full length OOMs. We worked down: 8192 OOM, 4096 OOM,
# 2048 forward succeeded but Triton's first-time-autotune scratch
# buffer (256 MiB) then OOM'd in the backward layer-norm kernel.
# 1024 gives ~4.9 GiB forward + room for the autotuner. Still 2x
# the context PlantCAD v1 ever saw (512 bp). Tokeniser picks a
# random window each epoch, so we cover all 8192 bp of each chunk
# across multiple passes.
MAX_SEQ_LENGTH=1024

# LoRA config — sensible defaults (per professor "use sensible defaults").
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Soft-masked loss weighting — keep v1's choice (lowercase = repeats,
# count for only 10% of the loss during training, ignored at eval).
SOFT_MASKED_TRAIN=0.1
SOFT_MASKED_EVAL=0.0

# --- Sanity-probe mode ------------------------------------------------
# Set SANITY=1 to do a fast memory + plumbing probe instead of the
# real run. Saves you from waiting 6 hours just to find out it OOMs.
SANITY="${SANITY:-0}"
if [[ "$SANITY" == "1" ]]; then
    # 30 steps at ~275 s/step on 2080 Ti = ~2.3 hr. Enough to see a
    # real loss curve and prove the pipeline end-to-end. Eval at
    # step 15, save once at the end, log every 3 steps so the
    # TensorBoard curve has resolution.
    echo "=== SANITY MODE: 30 steps, 1k samples (~2.3 hr on 2080 Ti) ==="
    MAX_STEPS_FLAG="--max_steps 30"
    NUM_EPOCHS_FLAG="--num_train_epochs 1"
    SAMPLE_CAPS="--max_train_samples 1000 --max_eval_samples 200"
    EVAL_STEPS=15
    SAVE_STEPS=30
    LOGGING_STEPS=3
else
    echo "=== FULL RUN: ~1 epoch over 50k samples ==="
    MAX_STEPS_FLAG=""                       # let num_train_epochs decide
    NUM_EPOCHS_FLAG="--num_train_epochs 1"
    SAMPLE_CAPS=""
    EVAL_STEPS=100
    SAVE_STEPS=250
    LOGGING_STEPS=20
fi

# --- Launch -----------------------------------------------------------
# torchrun for DDP across all 8 GPUs. Each GPU runs a full replica of
# PlantCAD2-Small (88M params fit easily under LoRA); they differ in
# the data batch they're processing, and gradients are averaged. No
# model parallelism needed for 88M params.
torchrun --nproc_per_node=8 --master_port=29501 \
    src/HF_pre_train_lora.py \
    --ddp_find_unused_parameters \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_root "$OUTPUT_ROOT" \
    --run_name "$RUN_NAME" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_steps "$WARMUP_STEPS" \
    --lr_scheduler_type "$LR_SCHEDULER" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --per_device_eval_batch_size "$PER_DEVICE_BS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --soft_masked_loss_weights_train "$SOFT_MASKED_TRAIN" \
    --soft_masked_loss_weights_evaluation "$SOFT_MASKED_EVAL" \
    --eval_strategy steps --eval_steps "$EVAL_STEPS" \
    --save_strategy steps --save_steps "$SAVE_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --seed 32 \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    $PRECISION_FLAG \
    $MAX_STEPS_FLAG \
    $NUM_EPOCHS_FLAG \
    $SAMPLE_CAPS

echo ""
echo "Done. Run directory: ${OUTPUT_ROOT}/${RUN_NAME}"
echo ""
echo "Open TensorBoard with:  tensorboard --logdir ${OUTPUT_ROOT}/"
