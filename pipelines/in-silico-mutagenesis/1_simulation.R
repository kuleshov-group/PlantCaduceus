#!/usr/bin/env Rscript
# =======================================
# File: 1_simulation.R
# Author: JINGJING ZHAI
# Email: jz963@cornell.edu
# Created: 2025-07-08
# Last Modified: 2025-07-08
# Description: This script generates a VCF file containing potential SNPs in extended gene regions
#              based on a GFF file and a genome FASTA file.
# =======================================

# --- Load Libraries ---
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(GenomicRanges))
suppressPackageStartupMessages(library(rtracklayer))
suppressPackageStartupMessages(library(Biostrings))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(purrr))

parser <- ArgumentParser(description = "Simulate SNPs in extended gene regions from a GFF and FASTA file.")
parser$add_argument("-g", "--gff", type="character", required=TRUE,
                    help="Path to the input GFF file (e.g., annotations.gff).")
parser$add_argument("-f", "--fasta", type="character", required=TRUE,
                    help="Path to the input genome FASTA file (e.g., genome.fa).")
parser$add_argument("-o", "--output", type="character", required=TRUE,
                    help="Path for the output file (e.g., potential_snps.vcf).")
parser$add_argument("-c", "--chr", type="character", required=TRUE,
                    help="Target chromosome name (e.g., 'chr1'). Must match names in GFF/FASTA.")
parser$add_argument("-k", "--flank", type="integer", default=2000,
                    help="Flank size in base pairs to extend gene regions on both sides [default: %(default)s].")

args <- parser$parse_args()
options(scipen = 20)

cat("--- Parameters ---\n")
cat("GFF File:    ", args$gff, "\n")
cat("FASTA File:  ", args$fasta, "\n")
cat("Output File: ", args$output, "\n")
cat("Chromosome:  ", args$chr, "\n")
cat("Flank Size:  ", args$flank, "\n")
cat("------------------\n\n")


cat("Loading genome and GFF...\n")

fasta_path <- args$fasta
twobit_path <- sub("\\.fa(sta)?$", ".2bit", fasta_path, ignore.case = TRUE)

if (!file.exists(twobit_path)) {
  cat("Did not find .2bit file. Creating", twobit_path, "from FASTA file...\n")
  dna <- readDNAStringSet(args$fasta)
  dna <- replaceAmbiguities(dna, new = "N")
  export(dna, twobit_path)
  rm(dna)
  cat(".2bit file created successfully.\n")
}

genome <- TwoBitFile(twobit_path)
chr_lengths <- seqlengths(genome)

if (!args$chr %in% names(chr_lengths)) {
  stop(paste("Error: Chromosome '", args$chr, "' not found in the FASTA file. Please check your chromosome names.", sep=""))
}
gff <- import(args$gff)


cat("Processing genomic regions for", args$chr, "...\n")

gene_regions <- gff %>%
  as_tibble() %>%
  filter(type == 'gene', seqnames == args$chr) %>%
  makeGRangesFromDataFrame(keep.extra.columns = TRUE)

extended_regions <- resize(gene_regions, width = width(gene_regions) + 2 * args$flank, fix = "center")

final_regions <- extended_regions[which(start(extended_regions) > 0 & end(extended_regions) <= chr_lengths[args$chr])]
strand(final_regions) <- "*"

cat("Extracting sequences and generating SNP candidates...\n")
ref_seqs <- getSeq(genome, final_regions)

# Check if any sequences were returned
if (length(ref_seqs) > 0) {
    snp_candidates <- map_dfr(seq_along(final_regions), ~{
        region    <- final_regions[.x]
        sequence  <- as.character(ref_seqs[[.x]])
        
        if (length(sequence) == 0) return(NULL)
        
        positions <- start(region):end(region)
        bases     <- strsplit(sequence, "")[[1]]

        tibble(pos = positions, ref = bases)
    }) %>%
      { if(nrow(.) > 0) . else tibble() } %>% # Return empty tibble if map_dfr is empty
      filter(ref %in% c("A", "C", "G", "T")) %>%
      crossing(alt = c("A", "C", "G", "T")) %>%
      filter(ref != alt) %>%
      arrange(pos)
} else {
    # If no regions exist after processing, create an empty tibble
    snp_candidates <- tibble()
}



if (nrow(snp_candidates) > 0) {
  cat("Formatting and writing output to", args$output, "...\n")
  output_vcf <- data.frame(
    chr = args$chr,
    pos = snp_candidates$pos,
    id = '.',
    ref = snp_candidates$ref,
    alt = snp_candidates$alt,
    qual = '.',
    filter = '.'
  )

  write.table(
    output_vcf,
    file = args$output,
    sep = '\t',
    quote = FALSE,
    row.names = FALSE,
    col.names = FALSE
  )
} else {
    cat("Warning: No candidate SNPs were generated. The output file will be empty.\n")
    file.create(args$output)
}

cat("Done!\n")