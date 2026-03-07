# Zero-shot Scoring of Structural Variants

[← Back to main README](../README.md)

The `zero_shot_score_sv.py` script scores structural variants (deletions and insertions) by comparing the model's predicted nucleotide probabilities at the SV junction flanks between the reference and mutant sequences.

## How it works

1. **Reference sequence:** A window of `-contextSize` bp is extracted centered on the SV.
2. **Mutant sequence:** The SV is applied (deletion removed or insertion added) and the same-size window is extracted around the new junction.
3. **Flank scoring:** The model predicts nucleotide probabilities at `-flank-size` positions on each side of the junction in both sequences.
4. **Score:** For each flank position, the log-ratio $\log(P_{\text{mut}} / P_{\text{ref}})$ is computed. The final score is the average across all flank positions. Negative scores indicate the SV is predicted to be disruptive.

## Parameter reference

| Parameter        | Default    | Description                                                                                          |
| :--------------- | :--------- | :--------------------------------------------------------------------------------------------------- |
| `-input-vcf`   | —         | Input VCF file containing SVs (deletions and/or insertions).                                         |
| `-input-fasta` | —         | Reference genome FASTA file (required).                                                              |
| `-output`      | —         | Output VCF file with `PlantCAD_SV_Score` in the INFO field.                                        |
| `-model`       | —         | HuggingFace model name or local path.                                                                |
| `-device`      | `cuda:0` | Compute device.                                                                                      |
| `-batchSize`   | `32`     | Batch size for inference.                                                                            |
| `-contextSize` | `8192`   | Context window size in bp. **Can be shorter, but SVs longer than half this value will be skipped.** |
| `-flank-size`  | `5`      | Number of bases on each side of the SV junction to score.                                            |

## Example usage

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
