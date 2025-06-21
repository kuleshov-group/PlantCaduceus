#!/usr/bin/env bash

# --- Script Description ---
# This script processes a VCF file to extract sequences around variant positions.
# It extracts upstream 255bp and downstream 256bp, with the SNP at position 256 (one-based coordinate).
# Dependencies: 'samtools' for FASTA indexing, 'bedtools' for sequence extraction, and 'awk' for text processing.
# Original script adapted from @pmorrell (https://github.com/pmorrell/Utilities/blob/master/PlantCaduceus_format.sh).


set -eo pipefail

# --- Configuration ---
# Check if the correct number of arguments was provided.
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input.vcf> <reference.fasta> <output_directory>"
    exit 1
fi

VCF_FILE="$1"
REFERENCE="$2"
OUTPUT_FILE="$3"

# Generate index for the reference genome if it doesn't exist.
REFERENCE_INDEX="${REFERENCE}.fai"
if [[ ! -f "${REFERENCE_INDEX}" ]]; then
    echo "Reference index not found. Creating it now with 'samtools faidx'..."
    # This will fail and stop the script if 'samtools' isn't installed.
    samtools faidx "${REFERENCE}"
fi


echo "Processing ${SAMPLE_NAME}, output will be in ${OUTPUT_FILE}"


HEADER="chr\tstart\tend\tpos\tref\talt\tsequences"
echo -e "${HEADER}" > "${OUTPUT_FILE}"

# 4. Run the Main Processing Pipeline
echo "Generating contextual sequences..."

grep -v '^#' "${VCF_FILE}" | \
    awk -v OFS='\t' '{print $1, $2-1, $2, $2, $4, $5}' | \
    bedtools slop -i - -g "${REFERENCE_INDEX}" -l 255 -r 256 | \
    bedtools getfasta -fi "${REFERENCE}" -bed - -bedOut -tab >> "${OUTPUT_FILE}"

echo "Done."