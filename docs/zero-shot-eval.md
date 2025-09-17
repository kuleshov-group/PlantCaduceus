# Zero‑Shot Evaluation

The `src/zero-shot-eval.py` script provides a unified CLI interface (via Fire) for evaluating PlantCAD2 models on multiple zero‑shot tasks using Hugging Face datasets.

Subcommands: `cons` (aliases: `token_auroc`, `mid`), `motif_acc`, `core_noncore`, `sv_effect`.

## Common Arguments

All subcommands support these shared parameters:

- `--repo_id`: Hugging Face dataset repository (e.g., `plantcad/PlantCAD2_zero_shot_tasks`)
- `--task`: HF configuration or subset name (e.g., `tis_recovery`)
- `--split`: Dataset split to evaluate (e.g., `test_maize`, `valid`)
- `--model`: HF model identifier (e.g., `kuleshov-group/PlantCAD2-Small-l24-d0768`)
- `--device`: Compute device (`cuda:0` or `cpu`)
- `--batch_size`: Batch size for inference

## Available Subcommands

### 1. Evolutionary Conservation (`evo_cons`)

Evaluates evolutionary conservation by masking the single middle token and using masked token accuracy as the conservation score.

**Example:**
```bash
python src/zero-shot-eval.py evo_cons \
  --repo_id plantcad/PlantCAD2_zero_shot_tasks \
  --task conservation_within_poaceae_tis \
  --split test \
  --model kuleshov-group/PlantCAD2-Small-l24-d0768 \
  --device cuda:0 \
  --token_idx 4095 \
  --batch_size 16
```

Requirements: Dataset must contain `sequence` and `label` columns.

**Available Tasks**:
- conservation_within_poaceae_tis (split: test)
- conservation_within_andropogoneae (split: test)
- conservation_within_poaceae_non_tis (split: test)

Note: For 8,192bp sequences, the middle token index is 4095 (0-based).

### 2. Motif Recovery (`motif_acc`)
Masks multiple tokens to evaluate both token‑level and motif‑level accuracy for motif recovery tasks.

**Example:**
```bash
python src/zero-shot-eval.py motif_acc \
  --repo_id plantcad/PlantCAD2_zero_shot_tasks \
  --task tis_recovery \
  --split test_maize \
  --model kuleshov-group/PlantCAD2-Small-l24-d0768 \
  --device cuda:0 \
  --mask_idx 4094,4095,4096 \
  --motif_len 3 \
  --batch_size 16
```

Requirements: Dataset must contain a `sequence` column.

**Available Tasks** (splits: `test_maize`, `test_tomato`):
- `tis_recovery` (mask_idx: 4094,4095,4096; motif_len: 3)
- `acceptor_recovery` (mask_idx: typically 4095,4096; motif_len: 2)
- `donor_recovery` (mask_idx: typically 4095,4096; motif_len: 2)
- `tts_recovery` (mask_idx: typically 4094,4095,4096; motif_len: 3)

### 3. Core/Non‑Core Classification (`core_noncore`)
Calculates AUROC by averaging masked token probabilities across masked positions for core/non‑core classification tasks.

**Example:**

```bash
python src/zero-shot-eval.py core_noncore \
  --repo_id plantcad/PlantCAD2_zero_shot_tasks \
  --task tis_core_noncore_classification \
  --split test_maize \
  --model kuleshov-group/PlantCAD2-Small-l24-d0768 \
  --device cuda:0 \
  --mask_idx 4094,4095,4096 \
  --motif_len 3 \
  --batch_size 16
```

Requirements: Dataset must contain `sequence` and `label` columns.

**Available Tasks**: (splits: `test_maize`, `test_tomato`):
- `tis_core_noncore_classification` (mask_idx: 4094,4095,4096; motif_len: 3)
- `acceptor_core_noncore_classification` (mask_idx: typically 4095,4096; motif_len: 2)
- `donor_core_noncore_classification` (mask_idx: typically 4095,4096; motif_len: 2)
- `tts_core_noncore_classification` (mask_idx: typically 4094,4095,4096; motif_len: 3)

### 4. Structural Variant Effect (`sv_effect`)
Performs boundary‑based structural variant scoring using reference boundary windows and mutated central regions with flanking sequences. Evaluates performance using AUPRC.

**Example:**
```bash
python src/zero-shot-eval.py sv_effect \
  --repo_id plantcad/PlantCAD2_zero_shot_tasks \
  --task structural_variant_effect_prediction \
  --split test \
  --model kuleshov-group/PlantCAD2-Small-l24-d0768 \
  --device cuda:0 \
  --batch_size 64 \
  --flanking 5 \
  --output sv_effect_scored.tsv
```

Requirements: Dataset must contain `RefSeq`, `MutSeq`, `left`, `right`, and `label` columns.

**Available Tasks**:
- `structural_variant_effect_prediction` (split: test)
