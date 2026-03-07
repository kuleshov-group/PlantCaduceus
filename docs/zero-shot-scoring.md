# Zero-shot Scoring of SNPs and Genomic Regions

[← Back to main README](../README.md)

The `zero_shot_score.py` script scores single-nucleotide polymorphisms (SNPs) or entire genomic regions using PlantCAD's log-likelihood ratios. It supports two primary modes:

1. **Variant Scoring (VCF Input):** Scores specific genetic variants provided in a VCF file.
2. **Genome-Wide Region Scoring (BED Input):** Calculates log-likelihood ratios for all positions within specified genomic regions (BED file).

## Choosing a model and context size

| Scenario                     | Recommended Model     | `-contextSize` | GPU Memory* | Notes                                                                                   |
| :--------------------------- | :-------------------- | :-------------- | :---------- | :-------------------------------------------------------------------------------------- |
| Limited GPU                  | `PlantCaduceus_l32` | `512` (fixed)  | ~2–3 GB    | Works well for both coding and noncoding variants. Fast and lightweight.                |
| Higher noncoding sensitivity | `PlantCAD2-Medium`  | `≥ 2048`      | ~10–34 GB  | Longer context improves sensitivity for noncoding regions (promoters, enhancers, etc.). |
| Best accuracy                | `PlantCAD2-Large`   | `≥ 2048`      | ~15–51 GB  | Highest accuracy overall. Use `4096` or `8192` if GPU memory allows.                |

> **🔒 Context size rules:**
>
> - **PlantCaduceus (v1):** context is always fixed at **512**. The script will override any other value.
> - **PlantCAD2:** minimum context is **2048**. If you specify a smaller value, the script will automatically raise it to 2048 with a warning. You can set it higher (up to 8192) for better accuracy.
>
> **⏱️ Note on inference time:** Inference time scales approximately **linearly** with `-contextSize`. For example, doubling the context window roughly doubles the runtime. Choose a context size that balances your accuracy needs with your compute budget. See the [inference speed table](model-recommendations.md#inference-speed) for detailed benchmarks.

## Parameter reference

| Parameter            | Applies to | Default     | Description                                                                                                                       |
| :------------------- | :--------- | :---------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| `-input-vcf`       | VCF mode   | —          | Path to input VCF file (mutually exclusive with `-input-bed`)                                                                   |
| `-input-bed`       | BED mode   | —          | Path to BED file specifying genomic regions                                                                                       |
| `-input-fasta`     | Both       | —          | Path to reference genome FASTA (required)                                                                                         |
| `-output`          | Both       | —          | Path to output file                                                                                                               |
| `-model`           | Both       | —          | HuggingFace model name or local path                                                                                              |
| `-device`          | Both       | `cuda:0`  | Compute device                                                                                                                    |
| `-batchSize`       | Both       | `128`     | Batch size for inference                                                                                                          |
| `-contextSize`     | Both       | `2048`    | Context window size in bp. **Must ≤ model's max input length.** Auto-enforced: PlantCaduceus → 512; PlantCAD2 → min 2048. |
| `-step-size`       | BED only   | `1`       | Positions scored per window. Larger = faster but less precise ([details](step_size_genome_wide_llr.md)).                             |
| `-use-masking`     | BED only   | `False`   | Mask the center position(s) during inference. Recommended only with `-step-size 1`.                                             |
| `-aggregation`     | BED only   | `average` | How to aggregate alt-allele scores: `max`, `average`, or `all`.                                                              |
| `-output-raw-prob` | BED only   | `False`   | Include raw nucleotide probabilities in output.                                                                                   |

## Examples

### Setup: Download reference genome

```bash
wget https://download.maizegdb.org/Zm-B73-REFERENCE-NAM-5.0/Zm-B73-REFERENCE-NAM-5.0.fa.gz
gunzip Zm-B73-REFERENCE-NAM-5.0.fa.gz
```

### VCF Input Mode

Estimate impact of specific variants from a VCF file. Only the first 8 columns of the VCF file (CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO) are strictly required.

```bash
# Using PlantCAD (v1), which requires a fixed 512bp context
python src/zero_shot_score.py \
    -input-vcf examples/example_maize_snp.vcf \
    -input-fasta Zm-B73-REFERENCE-NAM-5.0.fa \
    -output scored_variants.vcf \
    -model 'kuleshov-group/PlantCaduceus_l32' \
    -contextSize 512 \
    -device 'cuda:0'
    # PlantCaduceus (PlantCAD v1) forces contextSize=512 regardless of the default (2048)

# Using PlantCAD2 with a larger context window
python src/zero_shot_score.py \
    -input-vcf examples/example_maize_snp.vcf \
    -input-fasta Zm-B73-REFERENCE-NAM-5.0.fa \
    -output scored_variants_cad2.vcf \
    -model 'kuleshov-group/PlantCAD2-Large-l48-d1536' \
    -contextSize 2048 \
    -device 'cuda:0'
    # PlantCAD2 supports up to 8192bp; larger context can improve accuracy
```

**Expected output:**
- A new VCF file with PlantCAD scores added to the INFO field.
- Scores represent log-likelihood ratios between reference and alternative alleles. Low negative scores indicate potentially more deleterious mutations.

### BED Input Mode

Calculate log-likelihood ratios for all positions within specified genomic regions.

```bash
# Create example BED file
echo -e "chr1\t1000\t1010\nchr1\t2000\t2015" > examples/example_regions.bed

# Using PlantCAD (v1) — default 512bp context
python src/zero_shot_score.py \
    -input-bed examples/example_regions.bed \
    -input-fasta Zm-B73-REFERENCE-NAM-5.0.fa \
    -output genome_wide_scores.tsv \
    -model 'kuleshov-group/PlantCaduceus_l32' \
    -contextSize 512 \
    -device 'cuda:0' \
    -step-size 1 \
    -aggregation average \
    -use-masking \
    -output-raw-prob
    # Explicitly set -contextSize 512 (max for PlantCaduceus v1; script enforces this)

# Using PlantCAD2 with larger context
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
```

**Expected output:**
- A tab-separated file containing scores for each position.
- Output includes chromosome, start, end, reference allele, aggregated score, and optionally raw probabilities for all four nucleotides.

When analyzing the entire genome or large genomic regions, the `-step-size` parameter is very important for speeding up the analysis. For a detailed guide on this trade-off between speed and accuracy, see **[Step Size Guide](step_size_genome_wide_llr.md)**.
