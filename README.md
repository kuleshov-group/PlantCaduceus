![Static Badge](https://img.shields.io/badge/Linux-blue?logo=Linux&logoColor=white)
![GitHub Repo stars](https://img.shields.io/github/stars/kuleshov-group/PlantCaduceus)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/kuleshov-group/PlantCaduceus)
[![PlantCAD](https://zenodo.org/badge/DOI/10.1073/pnas.2421738122.svg)](https://doi.org/10.1073/pnas.2421738122)
![PlantCAD Downloads](https://img.shields.io/badge/dynamic/json?color=blue&label=PlantCAD+downloads&query=downloads&url=https://huggingface.co/api/models/kuleshov-group/PlantCaduceus_l32)
[![PlantCAD2](https://zenodo.org/badge/DOI/10.1101/2025.08.27.672609.svg)](https://doi.org/10.1101/2025.08.27.672609)
![PlantCAD2 Downloads](https://img.shields.io/badge/dynamic/json?color=blue&label=PlantCAD2+downloads&query=downloads&url=https://huggingface.co/api/models/kuleshov-group/PlantCAD2-Small-l24-d0768)
[![PlantCAD 🤗](https://img.shields.io/badge/🤗-PlantCAD-yellow.svg?style=flat)](https://huggingface.co/collections/kuleshov-group/plantcaduceus-512bp-len-665a229ee098db706a55e44a)
[![PlantCAD2 🤗](https://img.shields.io/badge/🤗-PlantCAD2-yellow.svg?style=flat)](https://huggingface.co/collections/kuleshov-group/plantcad2-67e437e241a382671371a572)

<p align="center">
  <img src="img/logo.jpg" alt="logo" width="20%">
</p>

## [🚀 PlantCAD2 Release!](https://www.biorxiv.org/content/10.1101/2025.08.27.672609v1)
We’re excited to announce [PlantCAD2](https://huggingface.co/collections/kuleshov-group/plantcad2-67e437e241a382671371a572) 🌱 — our new DNA foundation model for angiosperms.

In addition, we’re also releasing a collection of [LoRA fine-tuned models](https://huggingface.co/collections/plantcad/fine-tuned-plantcad2-models-68b316a57616134fa7a1b6b6) 🎯, tailored for key downstream tasks including accessible chromatin, gene expression, and protein translation.

- Explore the **fine-tuned** PlantCAD2 models [here](docs/PlantCAD2-overview.md)

- Explore the **zero-shot** evaluation of PlantCAD2 models [here](docs/zero-shot-eval.md)

- Explore the post-training or pre-training of PlantCAD2 [here](https://github.com/kuleshov-group/PlantCaduceus/issues/19)

## Table of Contents
- [PlantCAD overview](#plantcad-overview)
- [Quick Start](#quick-start)
- [Model summary](#model-summary)
- [Prerequisites and system requirements](#prerequisites-and-system-requirements)
- [Installation](#installation)
  - [Option 1: Google Colab (Recommended for beginners)](#option-1-google-colab-recommended-for-beginners)
  - [Option 2: Local installation](docs/local-install.md)
  - [Option 3: Using Docker](docker/README.md)
- [Basic Usage](#basic-usage)
  - [Exploring model inputs and outputs](#exploring-model-inputs-and-outputs)
  - [Zero-shot Scoring of SNPs and Genomic Regions](#zero-shot-scoring-of-snps-and-genomic-regions)
  - [Zero-shot Scoring of Structural Variants](#zero-shot-scoring-of-structural-variants)
  - [In-silico mutagenesis pipeline](#in-silico-mutagenesis-pipeline)
- [Advanced Usage](#advanced-usage)
  - [Training XGBoost classifiers](#training-xgboost-classifiers)
  - [Using pre-trained XGBoost classifiers](#using-pre-trained-xgboost-classifiers)
- [Development and Training](#development-and-training)
  - [Pre-training PlantCAD](#pre-training-plantcad)
- [Model Recommendations](#model-recommendations)
- [Citation](#citation)

## [PlantCAD overview](https://plantcaduceus.github.io/)

PlantCaduceus, with its short name of **PlantCAD**, is a plant DNA LM based on the [Caduceus](https://arxiv.org/abs/2403.03234) architecture, which extends the efficient [Mamba](https://arxiv.org/abs/2312.00752) linear-time sequence modeling framework to incorporate bi-directionality and reverse complement equivariance, specifically designed for DNA sequences. PlantCAD is pre-trained on a curated dataset of 16 Angiosperm genomes. PlantCAD showed state-of-the-art cross species performance in predicting TIS, TTS, Splice Donor and Splice Acceptor. The zero-shot of PlantCAD enables identifying genome-wide deleterious mutations and known causal variants in Arabidopsis, Sorghum and Maize.

## Quick Start

**New to PlantCAD?** Try our [Google Colab demo](https://colab.research.google.com/drive/1QW9Lgwra0vHQAOICE2hsIVcp6DKClyhO?usp=sharing) - no installation required!

**For local usage:** See installation instructions [here](docs/local-install.md), then use `notebooks/examples.ipynb` to get started.

## Model summary
Pre-trained models have been uploaded to **HuggingFace 🤗**: [PlantCAD](https://huggingface.co/collections/kuleshov-group/plantcaduceus-512bp-len-665a229ee098db706a55e44a) and [PlantCAD2](https://huggingface.co/collections/plantcad/fine-tuned-plantcad2-models-68b316a57616134fa7a1b6b6). 
Here is the comparison between PlantCAD (v1) and PlantCAD2 models:

| Model | Max Input Length | Model Size | Embedding Size |
| :--- | :--- | :--- | :--- |
| **PlantCAD** | | | |
| [PlantCaduceus_l20](https://huggingface.co/kuleshov-group/PlantCaduceus_l20) | 512bp | 20M | 384 |
| [PlantCaduceus_l24](https://huggingface.co/kuleshov-group/PlantCaduceus_l24) | 512bp | 40M | 512 |
| [PlantCaduceus_l28](https://huggingface.co/kuleshov-group/PlantCaduceus_l28) | 512bp | 128M | 768 |
| [PlantCaduceus_l32](https://huggingface.co/kuleshov-group/PlantCaduceus_l32) | 512bp | 225M | 1024 |
| **PlantCAD2** | | | |
| [PlantCAD2-Small](https://huggingface.co/kuleshov-group/PlantCAD2-Small-l24-d0768) | 8192bp | 88M | 768 |
| [PlantCAD2-Medium](https://huggingface.co/kuleshov-group/PlantCAD2-Medium-l48-d1024) | 8192bp | 311M | 1024 |
| [PlantCAD2-Large](https://huggingface.co/kuleshov-group/PlantCAD2-Large-l48-d1536) | 8192bp | 694M | 1536 |

> **⚠️ Important:** The "Max Input Length" is a hard limit — your input sequences **cannot** exceed this length. The `-contextSize` parameter (see [Zero-shot Scoring](#zero-shot-scoring-of-genomic-variants-and-regions)) must be set to match. Use `-contextSize 512` for PlantCAD models and up to `-contextSize 8192` for PlantCAD2 models.

## Prerequisites and System Requirements

**For Google Colab:** Just a Google account - GPU runtime recommended (free tier available)

**For Local Installation: NVIDIA GPU is required!**

## Installation

### Option 1: Google Colab (Recommended for beginners)
**No installation required!** Just open our [PlantCAD Google Colab notebook](https://colab.research.google.com/drive/1QW9Lgwra0vHQAOICE2hsIVcp6DKClyhO?usp=sharing) and start analyzing your data.

### Option 2: Local installation
See our [Local Installation Guide](docs/local-install.md).


## Basic Usage

### Exploring model inputs and outputs

The easiest way to start is with our example notebook: `notebooks/examples.ipynb`

**Quick example - Get sequence embeddings:**
```python
import torch
from mamba_ssm import Mamba
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd

device = 'cuda:0'
# Test PlantCAD model loading
tokenizer = AutoTokenizer.from_pretrained('kuleshov-group/PlantCaduceus_l32')
model = AutoModelForMaskedLM.from_pretrained('kuleshov-group/PlantCaduceus_l32', trust_remote_code=True)
model.to(device)
# Example plant DNA sequence (512bp max)
sequence = "CTTAATTAATATTGCCTTTGTAATAACGCGCGAAACACAAATCTTCTCTGCCTAATGCAGTAGTCATGTGTTGACTCCTTCAAAATTTCCAAGAAGTTAGTGGCTGGTGTGTCATTGTCTTCATCTTTTTTTTTTTTTTTTTAAAAATTGAATGCGACATGTACTCCTCAACGTATAAGCTCAATGCTTGTTACTGAAACATCTCTTGTCTGATTTTTTCAGGCTAAGTCTTACAGAAAGTGATTGGGCACTTCAATGGCTTTCACAAATGAAAAAGATGGATCTAAGGGATTTGTGAAGAGAGTGGCTTCATCTTTCTCCATGAGGAAGAAGAAGAATGCAACAAGTGAACCCAAGTTGCTTCCAAGATCGAAATCAACAGGTTCTGCTAACTTTGAATCCATGAGGCTACCTGCAACGAAGAAGATTTCAGATGTCACAAACAAAACAAGGATCAAACCATTAGGTGGTGTAGCACCAGCACAACCAAGAAGGGAAAAGATCGATGATCG"
device = 'cuda:0'
# Get embeddings
encoding = tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )

input_ids = encoding["input_ids"].to(device)
with torch.inference_mode():
    outputs = model(input_ids=input_ids, output_hidden_states=True)

embeddings = outputs.hidden_states[-1]
print(f"Embedding shape: {embeddings.shape}")  # [batch_size, seq_len, embedding_dim]

embeddings = embeddings.to(torch.float32).cpu().numpy()
# Given that PlantCaduceus has bi-directionality and reverse complement equivariance, so the first half of embedding is for forward sequences and the sencond half is for reverse complemented sequences, we need to average the embeddings before working on downstream classifier

hidden_size = embeddings.shape[-1] // 2
forward = embeddings[..., 0:hidden_size]
reverse = embeddings[..., hidden_size:]
reverse = reverse[..., ::-1]
averaged_embeddings = (forward + reverse) / 2
print(averaged_embeddings.shape)
```


### Zero-shot Scoring of SNPs and Genomic Regions

The `zero_shot_score.py` script scores single-nucleotide polymorphisms (SNPs) or entire genomic regions using PlantCAD's log-likelihood ratios. It supports two primary modes:

1.  **Variant Scoring (VCF Input):** Scores specific genetic variants provided in a VCF file.
2.  **Genome-Wide Region Scoring (BED Input):** Calculates log-likelihood ratios for all positions within specified genomic regions (BED file).

#### Choosing a model and context size

| Scenario | Recommended Model | `-contextSize` | GPU Memory* | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Limited GPU | `PlantCaduceus_l32` | `512` (fixed) | ~2–3 GB | Works well for both coding and noncoding variants. Fast and lightweight. |
| Higher noncoding sensitivity | `PlantCAD2-Medium` | `≥ 2048` | ~10–34 GB | Longer context improves sensitivity for noncoding regions (promoters, enhancers, etc.). |
| Best accuracy | `PlantCAD2-Large` | `≥ 2048` | ~15–51 GB | Highest accuracy overall. Use `4096` or `8192` if GPU memory allows. |

> **🔒 Context size rules:**
> - **PlantCaduceus (v1):** context is always fixed at **512**. The script will override any other value.
> - **PlantCAD2:** minimum context is **2048**. If you specify a smaller value, the script will automatically raise it to 2048 with a warning. You can set it higher (up to 8192) for better accuracy.
>
> **⏱️ Note on inference time:** Inference time scales approximately **linearly** with `-contextSize`. For example, doubling the context window roughly doubles the runtime. Choose a context size that balances your accuracy needs with your compute budget. See the [inference speed table](#inference-speed) for detailed benchmarks.

#### Parameter reference

| Parameter | Applies to | Default | Description |
| :--- | :--- | :--- | :--- |
| `-input-vcf` | VCF mode | — | Path to input VCF file (mutually exclusive with `-input-bed`) |
| `-input-bed` | BED mode | — | Path to BED file specifying genomic regions |
| `-input-fasta` | Both | — | Path to reference genome FASTA (required) |
| `-output` | Both | — | Path to output file |
| `-model` | Both | — | HuggingFace model name or local path |
| `-device` | Both | `cuda:0` | Compute device |
| `-batchSize` | Both | `128` | Batch size for inference |
| `-contextSize` | Both | `2048` | Context window size in bp. **Must ≤ model's max input length.** Auto-enforced: PlantCaduceus → 512; PlantCAD2 → min 2048. |
| `-step-size` | BED only | `1` | Positions scored per window. Larger = faster but less precise ([details](docs/step_size_genome_wide_llr.md)). |
| `-use-masking` | BED only | `False` | Mask the center position(s) during inference. Recommended only with `-step-size 1`. |
| `-aggregation` | BED only | `average` | How to aggregate alt-allele scores: `max`, `average`, or `all`. |
| `-output-raw-prob` | BED only | `False` | Include raw nucleotide probabilities in output. |

```bash
# Download example reference genome
wget https://download.maizegdb.org/Zm-B73-REFERENCE-NAM-5.0/Zm-B73-REFERENCE-NAM-5.0.fa.gz
gunzip Zm-B73-REFERENCE-NAM-5.0.fa.gz
```

```bash
# VCF Input Mode
# --- Example: Variant Scoring (VCF Input) ---
# Estimate impact of specific variants from a VCF file.
# Note: Only the first 8 columns of the VCF file (CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO) are strictly required.
# --- Using PlantCAD (v1) with default 512bp context ---
python src/zero_shot_score.py \
    -input-vcf examples/example_maize_snp.vcf \
    -input-fasta Zm-B73-REFERENCE-NAM-5.0.fa \
    -output scored_variants.vcf \
    -model 'kuleshov-group/PlantCaduceus_l32' \
    -device 'cuda:0'
    # -contextSize defaults to 512, matching PlantCAD's max input length

# --- Using PlantCAD2 with a larger context window ---
python src/zero_shot_score.py \
    -input-vcf examples/example_maize_snp.vcf \
    -input-fasta Zm-B73-REFERENCE-NAM-5.0.fa \
    -output scored_variants_cad2.vcf \
    -model 'kuleshov-group/PlantCAD2-Large-l48-d1536' \
    -contextSize 2048 \
    -device 'cuda:0'
    # PlantCAD2 supports up to 8192bp; larger context can improve accuracy

# Expected output for VCF mode:
# - A new VCF file with PlantCAD scores added to the INFO field.
# - Scores represent log-likelihood ratios between reference and alternative alleles.
#   Low negative scores indicate potentially more deleterious mutations.
```

```bash
# BED Input Mode
# --- Example: Genome-Wide Region Scoring (BED Input) ---
# Calculate log-likelihood ratios for all positions within specified BED regions.
# Note: You would need an example BED file for this.
# For demonstration, creating a dummy BED file:
echo -e "chr1\t1000\t1010\nchr1\t2000\t2015" > examples/example_regions.bed

# --- Using PlantCAD (v1) — default 512bp context ---
python src/zero_shot_score.py \
    -input-bed examples/example_regions.bed \
    -input-fasta Zm-B73-REFERENCE-NAM-5.0.fa \
    -output genome_wide_scores.tsv \
    -model 'kuleshov-group/PlantCaduceus_l32' \
    -device 'cuda:0' \
    -step-size 1 \
    -aggregation average \
    -use-masking \
    -output-raw-prob
    # -contextSize defaults to 512 (max for PlantCAD v1)

# --- Using PlantCAD2 with larger context ---
python src/zero_shot_score.py \
    -input-bed examples/example_regions.bed \
    -input-fasta Zm-B73-REFERENCE-NAM-5.0.fa \
    -output genome_wide_scores_cad2.tsv \
    -model 'kuleshov-group/PlantCAD2-Large-l48-d1536' \
    -contextSize 4096 \
    -device 'cuda:0' \
    -step-size 8 \
    -aggregation average
    # PlantCAD2 supports up to 8192bp context; step-size 8 gives ~8x speedup

# Expected output for BED mode:
# - A tab-separated file containing scores for each position.
# - Output includes chromosome, start, end, reference allele, aggregated score,
#   and optionally raw probabilities for all four nucleotides.
# See the parameter reference table above for details on each flag.
```

When analyzing the entire genome or large genomic regions, the `-step-size` parameter is very important for speeding up the analysis. For a detailed guide on this trade-off between speed and accuracy, see **[here](docs/step_size_genome_wide_llr.md)**.


### Zero-shot Scoring of Structural Variants

The `zero_shot_score_sv.py` script scores structural variants (deletions and insertions) by comparing the model's predicted nucleotide probabilities at the SV junction flanks between the reference and mutant sequences.

#### How it works

1. **Reference sequence:** A window of `-contextSize` bp is extracted centered on the SV.
2. **Mutant sequence:** The SV is applied (deletion removed or insertion added) and the same-size window is extracted around the new junction.
3. **Flank scoring:** The model predicts nucleotide probabilities at `-flank-size` positions on each side of the junction in both sequences.
4. **Score:** For each flank position, the log-ratio $\log(P_{\text{mut}} / P_{\text{ref}})$ is computed. The final score is the average across all flank positions. Negative scores indicate the SV is predicted to be disruptive.

#### Parameter reference

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `-input-vcf` | — | Input VCF file containing SVs (deletions and/or insertions). |
| `-input-fasta` | — | Reference genome FASTA file (required). |
| `-output` | — | Output VCF file with `PlantCAD_SV_Score` in the INFO field. |
| `-model` | — | HuggingFace model name or local path. |
| `-device` | `cuda:0` | Compute device. |
| `-batchSize` | `32` | Batch size for inference. |
| `-contextSize` | `8192` | Context window size in bp. Should match the model's max input length. |
| `-flank-size` | `5` | Number of bases on each side of the SV junction to score. |

#### Example usage

```bash
# Score deletions and insertions in a VCF file
python src/zero_shot_score_sv.py \
    -input-vcf my_structural_variants.vcf \
    -input-fasta reference_genome.fa \
    -output scored_sv.vcf \
    -model 'kuleshov-group/PlantCAD2-Large-l48-d1536' \
    -device 'cuda:0' \
    -contextSize 8192 \
    -flank-size 5

# Expected output:
# A VCF file with PlantCAD_SV_Score added to the INFO field of each scored SV.
# Negative scores suggest the SV may be functionally disruptive.
```

> **Note:** SVs whose length exceeds `contextSize / 2 - flank-size` are automatically skipped, as the context window cannot adequately capture both flanks. Use a larger `-contextSize` (up to `8192` for PlantCAD2) when scoring large SVs.


### In-silico mutagenesis pipeline

For large-scale simulation and analysis of genetic variants, we provide a comprehensive in-silico mutagenesis pipeline. See [pipelines/in-silico-mutagenesis/README.md](pipelines/in-silico-mutagenesis/README.md) for detailed instructions.


## Advanced Usage

### Training XGBoost classifiers

Train custom classifiers on top of PlantCAD embeddings for specific annotation tasks (e.g., TIS, TTS, splice sites).

**Purpose**: Fine-tune prediction performance for specific annotation tasks using supervised learning.

**Data format**: Training data should follow the format used in our [cross-species annotation dataset](https://huggingface.co/datasets/kuleshov-group/cross-species-single-nucleotide-annotation/tree/main/TIS).

```bash
python src/train_XGBoost.py \
    -train train.tsv \
    -valid valid.tsv \
    -test test_rice.tsv \
    -model 'kuleshov-group/PlantCaduceus_l20' \
    -output ./output \
    -device 'cuda:0'
```

**Expected outputs:**
- Trained XGBoost classifier (`.json` file)
- Performance metrics on validation/test sets
- Feature importance analysis

### Using pre-trained XGBoost classifiers

We provide pre-trained XGBoost classifiers for common annotation tasks in the [`classifiers`](classifiers) directory.

**Available classifiers:**
- TIS (Translation Initiation Sites)
- TTS (Translation Termination Sites)  
- Splice donor/acceptor sites

```bash
python src/predict_XGBoost.py \
    -test test_rice.tsv \
    -model 'kuleshov-group/PlantCaduceus_l20' \
    -classifier classifiers/PlantCaduceus_l20/TIS_XGBoost.json \
    -device 'cuda:0' \
    -output ./output
```

**Expected output**: Predictions with confidence scores for each sequence in your test data.

## Development and Training

### Pre-training PlantCAD

For advanced users who want to pre-train PlantCAD or PlantCAD2 models from scratch or fine-tune on custom datasets.

**Requirements:**
- Large computational resources (multi-GPU recommended)
- WandB account for experiment tracking
- Custom genomic dataset in HuggingFace format

**Basic pre-training command:**
```bash
WANDB_PROJECT=PlantCAD python src/HF_pre_train.py \
    --do_train \
    --report_to wandb \
    --prediction_loss_only True \
    --remove_unused_columns False \
    --dataset_name 'kuleshov-group/Angiosperm_16_genomes' \
    --soft_masked_loss_weights_train 0.1 \
    --soft_masked_loss_weights_evaluation 0.0 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --dataloader_num_workers 16 \
    --preprocessing_num_workers 16 \
    --seed 32 \
    --save_strategy steps \
    --save_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --logging_steps 10 \
    --max_steps 120000 \
    --warmup_steps 1000 \
    --save_total_limit 20 \
    --learning_rate 2E-4 \
    --lr_scheduler_type constant_with_warmup \
    --run_name test \
    --overwrite_output_dir \
    --output_dir "PlantCaduceus_train_1" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --tokenizer_name 'kuleshov-group/PlantCaduceus_l20' \
    --config_name 'kuleshov-group/PlantCaduceus_l20' \
    --log_level info
```

**Key parameters:**
- `dataset_name`: Your custom dataset or use our Angiosperm dataset
- `max_steps`: Total training steps (adjust based on dataset size)
- `learning_rate`: 2E-4 works well for most cases
- `Batch sizes`: Adjust based on your GPU memory

## Model Recommendations
### Inference speed

Here are the inference speed benchmark results for PlantCaduceus (v1) and PlantCAD2 models:

| Model | Seq Len | Batch | Peak memory (GB) | Seq/s | Tokens/s |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PlantCaduceus_l20 | 512 | 16 | 0.31 | 400.28 | 204,942 |
| PlantCaduceus_l20 | 512 | 32 | 0.56 | 640.86 | 328,118 |
| PlantCaduceus_l20 | 512 | 64 | 1.01 | 663.04 | 339,475 |
| PlantCaduceus_l24 | 512 | 16 | 0.43 | 335.88 | 171,970 |
| PlantCaduceus_l24 | 512 | 32 | 0.75 | 392.83 | 201,140 |
| PlantCaduceus_l24 | 512 | 64 | 1.37 | 407.38 | 208,577 |
| PlantCaduceus_l28 | 512 | 16 | 0.77 | 207.61 | 106,295 |
| PlantCaduceus_l28 | 512 | 32 | 1.27 | 213.99 | 109,563 |
| PlantCaduceus_l28 | 512 | 64 | 2.22 | 219.97 | 112,626 |
| PlantCaduceus_l32 | 512 | 16 | 1.1 | 130.56 | 66,848 |
| PlantCaduceus_l32 | 512 | 32 | 1.71 | 132.62 | 67,902 |
| PlantCaduceus_l32 | 512 | 64 | 2.97 | 135.05 | 69,144 |
| PlantCAD2-Small | 8192 | 16 | 6.56 | 19.61 | 160,653 |
| PlantCAD2-Small | 8192 | 32 | 12.76 | 19.26 | 157,767 |
| PlantCAD2-Small | 8192 | 64 | 24.89 | 19 | 155,649 |
| PlantCAD2-Medium | 8192 | 16 | 9.62 | 6.88 | 56,386 |
| PlantCAD2-Medium | 8192 | 32 | 17.62 | 6.76 | 55,342 |
| PlantCAD2-Medium | 8192 | 64 | 33.62 | 6.79 | 55,636 |
| PlantCAD2-Large | 8192 | 16 | 14.89 | 3.92 | 32,111 |
| PlantCAD2-Large | 8192 | 32 | 26.95 | 3.87 | 31,741 |
| PlantCAD2-Large | 8192 | 64 | 51.09 | 3.89 | 31,833 |


### Which model to use? 
#### Variant Effect Analysis (Zero-Shot Scoring)

| Scenario | Recommended Model | `-contextSize` | GPU Memory* | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Limited GPU | `PlantCaduceus_l32` | `512` (fixed) | ~2–3 GB | Works well for both coding and noncoding variants. Fast and lightweight. |
| Higher noncoding sensitivity | `PlantCAD2-Medium` | `≥ 2048` | ~10–34 GB | Longer context improves sensitivity for noncoding regions (promoters, enhancers, etc.). |
| Best accuracy | `PlantCAD2-Large` | `≥ 2048` | ~15–51 GB | Highest accuracy overall. Use `4096` or `8192` if GPU memory allows. |

\* Approximate peak memory at batch size 16–64. See the [inference speed table](#inference-speed) for details.

> **How to choose:**
> - `PlantCaduceus_l32` with `-contextSize 512` is a strong baseline that works well for both **coding and noncoding** variants, and is fast and lightweight.
> - If you want **higher sensitivity for noncoding regions** (e.g., promoters, enhancers, intergenic variants), use `PlantCAD2-Medium` or `PlantCAD2-Large` with `-contextSize` of at least `2048`. The longer context window helps capture longer-range regulatory signals.
> - For the **best overall accuracy**, use `PlantCAD2-Large` with `-contextSize` of `4096` or `8192` if your GPU memory allows.
>
> **Note:** The script automatically enforces context size constraints — PlantCaduceus is fixed at 512, and PlantCAD2 has a minimum of 2048. You do not need to worry about setting an invalid context size.

#### Other Downstream Tasks

- **Long Sequences:** For tasks requiring sequences longer than 512bp, we highly recommend fine-tuning **PlantCAD2** models. PlantCAD (v1) models are limited to a 512bp context window, whereas PlantCAD2 supports up to 8192bp.

## Citations

If you find PlantCAD useful for your research, please consider citing our paper:

- Zhai, J., Gokaslan, A., Schiff, Y., Berthel, A., Liu, Z. Y., Lai, W. L., Miller, Z. R., Scheben, A., Stitzer, M. C., Romay, M. C., Buckler, E. S., & Kuleshov, V. (2025). Cross-species modeling of plant genomes at single nucleotide resolution using a pretrained DNA language model. Proceedings of the National Academy of Sciences, 122(24), e2421738122. https://doi.org/10.1073/pnas.2421738122
- Zhai J., Gokaslan A., Hsu SK., Chen SP., Liu ZY., Marroquin E., Czech E., Cannon B., Berthel A., Romay MC., Pennell M., Kuleshov V.* Buckler ES*. PlantCAD2: A Long-Context DNA Language Model for Cross-Species Functional Annotation in Angiosperms. bioRxiv. 2025. Nov 19. doi: https://doi.org/10.1101/2025.08.27.672609