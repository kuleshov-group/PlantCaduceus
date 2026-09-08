# PlantCAD2 Lettuce Finetune — Instructions (v2 pipeline)

End-to-end guide to running the LoRA finetune of PlantCAD2-Small on the
lettuce reference genome and viewing the results.

All commands assume:
- You are SSH'd into `paw.idav.ucdavis.edu`
- Repo lives at `/mnt/data1/shared/plantcad/`
- You run training inside the Docker container (`docker compose run --rm plantcad`)
- The container's working directory is `/workspace` which maps to the repo

---

## 0. One-time environment setup

```bash
# Start the container (drops you into /workspace as root)
cd /mnt/data1/shared/plantcad
docker compose run --rm plantcad

# Inside the container, install the missing deps (only needed once per
# container build):
pip install peft tensorboard
```

---

## 1. Files in this pipeline

| File | What it does |
|---|---|
| `src/prepare_dataset_from_tsv.py` | One-time TSV → HuggingFace DatasetDict converter |
| `src/HF_pre_train_lora.py` | Main training script (LoRA-wrapped MLM with soft-mask weighting) |
| `src/run_finetune_v2.sh` | Bash orchestration — every hyperparameter lives here |
| `src/plot_loss_curve.py` | Parses `trainer_state.json` → publication-quality PNG |

Do **not** touch the v1 pipeline (`HF_pre_train.py`, `prepare_dataset.py`,
`run_finetune.sh`) — those still work and are useful as a fallback.

---

## 2. Data sources used

| File | Description |
|---|---|
| `data/GCF_002870075.5_Lsat_Salinas_v15_genomic_rand_chunk_8192_samples_50k.tsv` | Pre-sliced 50k random 8192 bp chunks. **This is what we train on.** |
| `data/lettuce_hf_dataset_rand8192_50k/` | HF DatasetDict built from the TSV above by `prepare_dataset_from_tsv.py`. 47,500 train / 2,500 validation. |
| `model/PlantCAD2-Small-l24-d0768/` | Base pretrained model (Kuleshov Group) — 88M params, 8192 bp native context |
| `model/plantcad2_small_lettuce_<timestamp>/` | Output: LoRA adapter + tokenizer + checkpoints + tensorboard logs |

Other TSVs in `data/` (`...rand_chunk_512_samples_100k.tsv`,
`...random_1024.tsv`, etc.) are alternates from earlier experiments;
ignore unless you're switching contexts.

---

## 3. End-to-end workflow

### 3a. Build the HF dataset (one-time per TSV)

```bash
python src/prepare_dataset_from_tsv.py \
    --tsv data/GCF_002870075.5_Lsat_Salinas_v15_genomic_rand_chunk_8192_samples_50k.tsv \
    --output_dir data/lettuce_hf_dataset_rand8192_50k
```

Takes ~1 minute. Should report `train=47500, validation=2500`. Skip if
`data/lettuce_hf_dataset_rand8192_50k/` already exists.

### 3b. (Recommended) Sanity probe (~5 minutes)

```bash
SANITY=1 bash src/run_finetune_v2.sh
```

30 optimizer steps on 1,000 samples. Verifies the pipeline works
end-to-end before committing to the full run. Should print
`trainable %: 2.6743%` near the top, then 10 loss lines, then a final
eval.

### 3c. Full ~1-epoch run (~2 hours)

```bash
bash src/run_finetune_v2.sh
```

743 optimizer steps over all 47,500 training samples. Writes to
`model/plantcad2_small_lettuce_<YYYYMMDD_HHMMSS>/`.

### 3d. Plot the loss curve

```bash
python src/plot_loss_curve.py model/plantcad2_small_lettuce_<YYYYMMDD_HHMMSS>
```

Drops `loss_curve.png` inside the run directory.

### 3e. View live TensorBoard (from outside the container)

```bash
# On the host (NOT inside docker), in /mnt/data1/shared/plantcad:
tensorboard --logdir model/ --port 6006
```

Then from your laptop in a separate terminal:

```bash
ssh -L 6006:localhost:6006 sshokeen@paw.idav.ucdavis.edu
```

Open `http://localhost:6006` in your browser.

---

## 4. Where outputs live after a successful run

```
model/plantcad2_small_lettuce_<YYYYMMDD_HHMMSS>/
├── adapter_config.json          # LoRA config — needed to reload adapter
├── adapter_model.bin            # The LoRA weights (~20 MB) — the deliverable
├── tokenizer*.json              # Tokenizer copies
├── trainer_state.json           # Full loss curve in JSON
├── train_results.json           # Final train metrics
├── eval_results.json            # Final eval metrics + perplexity
├── checkpoint-250/, checkpoint-500/, checkpoint-750/   # Intermediate saves
└── tensorboard/                 # TB event files
```

---

## 5. Hyperparameters — what to change and where

All knobs live in `src/run_finetune_v2.sh`. The headline ones:

| Variable | Default | Bump up if... |
|---|---|---|
| `LEARNING_RATE` | `1e-3` | Loss plateaus immediately after warmup |
| `LORA_R` | `16` | Adapter capacity feels too small (low grad_norm, low improvement) |
| `LORA_ALPHA` | `32` | (Always set to ~2× rank; PEFT convention) |
| `MAX_SEQ_LENGTH` | `1024` | You have a bigger GPU (Ampere+); bump to 4096 or 8192 |
| `PER_DEVICE_BS` | `1` | You have a bigger GPU |
| `GRAD_ACCUM` | `8` | You changed batch size and want to preserve effective batch=64 |
| `SOFT_MASKED_TRAIN` | `0.1` | Repetitive regions should count more in the loss |
| `NUM_EPOCHS_FLAG` (full branch) | `--num_train_epochs 1` | You want a longer run |

To change a default: edit `src/run_finetune_v2.sh`, save, re-run.

---

## 6. Loading the trained model in code

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel
import torch

BASE = "./model/PlantCAD2-Small-l24-d0768"
ADAPTER = "./model/plantcad2_small_lettuce_<YYYYMMDD_HHMMSS>"

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
base = AutoModelForMaskedLM.from_pretrained(BASE, trust_remote_code=True,
                                            torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval().cuda()

# For downstream eval where you want a standalone HF model (no PEFT
# dependency), merge the adapter into the base weights:
merged = model.merge_and_unload()
merged.save_pretrained("./model/plantcad2_small_lettuce_MERGED")
```

The model accepts 8,192 bp inputs at inference even though we trained
at 1,024 bp — LoRA is sequence-length-agnostic.

---

## 7. Hardware constraints (read before debugging memory errors)

The 8× 2080 Ti (sm_75, Turing) has three hard limits that shape this
pipeline:

1. **No bf16** — silently broken on Turing. Stick to fp32 or fp16.
2. **Mamba2 SSD Triton kernel won't compile in fp16 on sm_75.** This is
   why we run in fp32. Symptom: a screen-full of MLIR garbage ending in
   `PassManager::run failed`. Fix: revert `--fp16`.
3. **11 GB VRAM per card** — won't fit PlantCAD2-Small at 8,192 bp in
   fp32. We truncate to 1,024 bp at tokenization. Going to 2,048+ bp
   on this hardware requires gradient checkpointing (not yet tested
   with PlantCAD2) or moving to Ampere/Hopper GPUs.

If a run OOMs, in order of preference:

1. Lower `MAX_SEQ_LENGTH` in `run_finetune_v2.sh` (1024 → 512)
2. Lower `LORA_R` (16 → 8)
3. Reduce `LORA_TARGET_MODULES` to just `in_proj`

---

## 8. Reading the loss curve — quick reference

| Metric | Range | What it means |
|---|---|---|
| `train/loss` | 0 → 1.39 (ln 4) | Per-microbatch cross-entropy. Noisy. Lower = better. |
| `eval/loss` | same | Held-out CE. Smooth. **This is the honest signal.** |
| `train/grad_norm` | 0.01 → 10+ | Optimizer step magnitude. Stable < 1 = healthy. |
| `train/learning_rate` | configured | Cosine-decays from peak. |
| `eval/perplexity` | 1.0 → 4.0 | exp(eval_loss). "Effective # of bases the model is choosing between." |

Random baseline for DNA is **perplexity 4.0**. Anything < 2.0 means the
model is being meaningfully informative.

---

## 9. v1 vs v2 — what's different

| | v1 | v2 |
|---|---|---|
| Base model | PlantCaduceus_l20 (20M, 512 bp) | PlantCAD2-Small (88M, 8192 bp) |
| Method | Full finetune | LoRA r=16, α=32 (2.67 % trainable) |
| Training context | 512 bp | 1024 bp (8192 bp OOMs on 11 GB) |
| Train samples | 5k | 47.5k |
| Output dir | `output/...` | `model/plantcad2_small_lettuce_<ts>/` |
| Logging | Text logs | TensorBoard |

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: peft` | Container missing PEFT | `pip install peft` |
| `TensorBoardCallback requires tensorboard` | Container missing TB | `pip install tensorboard` |
| `unexpected keyword argument 'attention_mask'` | PEFT task_type set | Already fixed; ensure script has no `task_type=` in `LoraConfig` |
| `PassManager::run failed` + MLIR dump | Triton can't compile Mamba2 SSD kernel for sm_75 in fp16 | Remove `--fp16` from `run_finetune_v2.sh` |
| `CUDA out of memory` during forward | Sequence too long | Lower `MAX_SEQ_LENGTH` |
| `CUDA out of memory` during `do_bench` | Triton autotune scratch (256 MB) doesn't fit | Lower `MAX_SEQ_LENGTH` |
| `Expected to have finished reduction...` (DDP) | LoRA dropout zeros some adapters → unused params | Ensure `--ddp_find_unused_parameters` is in the torchrun line |
| Multiple sibling run dirs created in one launch | Each rank picked a different timestamp | Already fixed: `RUN_NAME=$(...)` is computed in bash once |
| `tensorboard: error: unrecognized arguments: --bind_all` | Old TB version on the host | Use `--host 0.0.0.0 --port 6006` or just `--port 6006` + SSH tunnel |
