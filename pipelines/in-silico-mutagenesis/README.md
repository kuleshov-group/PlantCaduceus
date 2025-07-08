## In-silico mutagenesis pipeline


### 1. Simulate some mutations

```bash
Rscript 1_simulation.R -h
usage: 1_simulation.R [-h] -g GFF -f FASTA -o OUTPUT -c CHR [-k FLANK]

Simulate SNPs in extended gene regions from a GFF and FASTA file.

optional arguments:
  -h, --help            show this help message and exit
  -g GFF, --gff GFF     Path to the input GFF file (e.g., annotations.gff).
  -f FASTA, --fasta FASTA
                        Path to the input genome FASTA file (e.g., genome.fa).
  -o OUTPUT, --output OUTPUT
                        Path for the output file (e.g., potential_snps.vcf).
  -c CHR, --chr CHR     Target chromosome name (e.g., 'chr1'). Must match
                        names in GFF/FASTA.
  -k FLANK, --flank FLANK
                        Flank size in base pairs to extend gene regions on
                        both sides [default: 2000].
```

### 2. run VEP annotator to get variant types

The official pipeline for running VEP: https://plants.ensembl.org/info/docs/tools/vep/script/index.html

```bash
INPUT_VCF=/your/input.vcf
OUTPUT_VCF=/your/output_vep.vcf
REFERENCE=/your/reference.fa
ANNOTATION=/your/annotation.gff3

singularity run -C \
    --bind $PWD \
    --bind $(dirname $REFERENCE) \
    --bind $(dirname $ANNOTATION) \
    --pwd $PWD \
    /programs/ensembl-vep-110.1/vep.sif \
    vep -i $INPUT_VCF -o $OUTPUT_VCF \
    --vcf \
    --fork 16 \
    -gff $ANNOTATION \
    -fasta $REFERENCE \
    --force_overwrite \
    --per_gene
```

### 3. Downsample the annotated VCF file (optional)
```bash
python 2_down_sampling.py output_VEP.vcf output_VEP_downsampled.vcf
```

### 4. Get zero-shot scores from PlantCAD
```bash
python ../../src/zero_shot_score.py \
    -input-vcf output_VEP_downsampled.vcf \
    -input-fasta ${REFERENCE} \
    -output example_output.vcf \
    -model 'kuleshov-group/PlantCaduceus_l32' \
    -device 'cuda:0'
```
