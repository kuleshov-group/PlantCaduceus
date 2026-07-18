#!/usr/bin/env python3
"""
build_expression_dataset.py
----------------------------

Build a (sequence, label) train/validation TSV for gene-expression-count
regression fine-tuning of PlantCAD2, from:
  - one or more per-exon RNA-seq read count matrices (data/merged_output.tsv
    format: one row per exon ID, one column per experiment). Multiple files
    are treated as separate experiment batches over the same exon set (e.g.
    different studies) and merged by exon ID before averaging -- equivalent
    to column-concatenating them into one wide table first.
  - a GFF3 gene/exon annotation for the same assembly
  - the matching genome FASTA

For each gene:
  1. average each of its exons' read counts across every experiment column
     from every input counts TSV
  2. take the max averaged count across the gene's exons as its expression label
  3. extract a fixed-length DNA window centered on the gene's midpoint
     (reverse-complemented for '-' strand genes so orientation is 5'->3')

Usage:
    python src/scripts/build_expression_dataset.py \
        --counts-tsv data/expression_7.tsv data/expression_72.tsv data/expression_85.tsv \
        --gff data/GCF_002870075.5_Lsat_Salinas_v15_genomic.gff \
        --fasta data/GCF_002870075.5_Lsat_Salinas_v15_genomic.fna \
        --output-prefix data/expression_dataset

    # Smoke test on a slice of each (large) counts TSV instead of the full file:
    python src/scripts/build_expression_dataset.py \
        --counts-tsv data/expression_7.tsv data/expression_72.tsv data/expression_85.tsv \
        --max-rows-per-tsv 2000 \
        --gff data/GCF_002870075.5_Lsat_Salinas_v15_genomic.gff \
        --fasta data/GCF_002870075.5_Lsat_Salinas_v15_genomic.fna \
        --output-prefix /tmp/expression_dataset_smoke
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("build_expression_dataset")


def _get_attr(attributes: str, key: str) -> Optional[str]:
    prefix = f"{key}="
    for part in attributes.split(";"):
        if part.startswith(prefix):
            return part[len(prefix):]
    return None


def parse_gff(gff_path: str) -> Tuple[Dict[str, Tuple[str, int, int, str]], Dict[str, str]]:
    """Single pass over the GFF3.

    Returns:
        genes: gene_id -> (chrom, start, end, strand); 0-based half-open coords.
        exon_to_gene: exon_id (as it appears in the counts TSV's ID column) -> gene_id.
    """
    genes: Dict[str, Tuple[str, int, int, str]] = {}
    exon_to_gene: Dict[str, str] = {}
    with open(gff_path) as fh:
        for line in tqdm(fh, desc=f"Parsing {Path(gff_path).name}"):
            if not line or line.startswith("#"):
                continue
            # Limit to 9 fields: some attribute values contain literal, unescaped tabs.
            parts = line.rstrip("\n").split("\t", 8)
            if len(parts) < 9:
                continue
            chrom, _source, feature, start, end, _score, strand, _frame, attributes = parts
            if feature == "gene":
                gene_id = _get_attr(attributes, "ID")
                if gene_id is None:
                    continue
                if gene_id.startswith("gene-"):
                    gene_id = gene_id[len("gene-"):]
                genes[gene_id] = (chrom, int(start) - 1, int(end), strand)
            elif feature == "exon":
                exon_id = _get_attr(attributes, "ID")
                gene_id = _get_attr(attributes, "gene")
                if exon_id is None or gene_id is None:
                    continue
                exon_to_gene[exon_id] = gene_id
    log.info("Parsed %d gene records and %d exon records", len(genes), len(exon_to_gene))
    return genes, exon_to_gene


def load_gene_labels(
    counts_tsvs: List[str],
    exon_to_gene: Dict[str, str],
    max_rows_per_tsv: Optional[int] = None,
) -> pd.DataFrame:
    """Merge one or more per-exon count TSVs by exon ID, average every experiment
    column across all of them, then take the max averaged count per gene.
    """
    per_file = []
    for path in counts_tsvs:
        df = pd.read_csv(path, sep="\t", nrows=max_rows_per_tsv)
        is_exon = df["ID"].str.startswith("exon-")
        log.info(
            "Loaded %d rows from %s (%d exon rows, %d non-exon rows dropped, %d experiment columns)",
            len(df), path, int(is_exon.sum()), int((~is_exon).sum()), df.shape[1] - 1,
        )
        df = df.loc[is_exon].set_index("ID")
        # Prefix columns with the source file's stem so experiment columns that
        # happen to share a name across files (not expected, but not guaranteed)
        # can't collide when concatenated below.
        stem = Path(path).stem
        df = df.rename(columns={c: f"{stem}::{c}" for c in df.columns})
        per_file.append(df)

    # Align by exon ID (outer join): an exon missing from one file -- e.g. a
    # truncated --max-rows-per-tsv smoke-test read -- just contributes NaN for
    # that file's columns there, which mean(skipna=True) below ignores.
    merged = pd.concat(per_file, axis=1)
    log.info(
        "Merged %d counts TSV(s) into %d exon rows x %d experiment columns",
        len(counts_tsvs), len(merged), merged.shape[1],
    )
    avg_count = merged.mean(axis=1, skipna=True)

    gene_id = avg_count.index.to_series().map(exon_to_gene)
    unmapped = int(gene_id.isna().sum())
    if unmapped:
        log.warning("%d exon rows had no matching gene in the GFF; dropping", unmapped)

    labels_df = pd.DataFrame({"avg_count": avg_count, "gene_id": gene_id}).dropna(subset=["gene_id"])
    gene_labels = (
        labels_df.groupby("gene_id")["avg_count"].max().rename("raw_count").reset_index()
    )
    log.info("Aggregated to %d genes with an expression label", len(gene_labels))
    return gene_labels


def extract_window(
    seq: str, gstart: int, gend: int, strand: str, window_size: int
) -> Optional[str]:
    seq_len = len(seq)
    if seq_len < window_size:
        return None
    center = (gstart + gend) // 2
    win_start = center - window_size // 2
    win_start = max(0, min(seq_len - window_size, win_start))
    window = seq[win_start : win_start + window_size]
    if len(window) != window_size:
        return None
    if strand == "-":
        window = str(Seq(window).reverse_complement())
    return window


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--counts-tsv", required=True, nargs="+",
        help="One or more per-exon read count TSVs (e.g. data/expression_7.tsv data/expression_72.tsv "
             "data/expression_85.tsv). Must share the same exon ID space; treated as separate "
             "experiment batches and merged by exon ID before averaging.",
    )
    parser.add_argument(
        "--max-rows-per-tsv", type=int, default=None,
        help="Read only the first N data rows of each --counts-tsv file. These files can be large; "
             "use this for a fast smoke test instead of the full file.",
    )
    parser.add_argument("--gff", required=True, help="GFF3 annotation matching the counts TSVs' assembly")
    parser.add_argument("--fasta", required=True, help="Genome FASTA matching the GFF's assembly")
    parser.add_argument("--window-size", type=int, default=1024, help="Fixed output sequence length in bp")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of genes held out for validation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the train/val shuffle")
    parser.add_argument("--output-prefix", required=True, help="Writes <prefix>_train.tsv and <prefix>_val.tsv")
    args = parser.parse_args()

    genes, exon_to_gene = parse_gff(args.gff)
    gene_labels = load_gene_labels(args.counts_tsv, exon_to_gene, args.max_rows_per_tsv)

    log.info("Loading genome FASTA: %s", args.fasta)
    seqs = {r.id: str(r.seq) for r in tqdm(SeqIO.parse(args.fasta, "fasta"), desc="Reading FASTA")}

    records = []
    n_no_gene_record = 0
    n_bad_chrom = 0
    n_window_failed = 0
    for row in tqdm(gene_labels.itertuples(index=False), total=len(gene_labels), desc="Extracting windows"):
        gene_id, raw_count = row.gene_id, row.raw_count
        coords = genes.get(gene_id)
        if coords is None:
            n_no_gene_record += 1
            continue
        chrom, gstart, gend, strand = coords
        if chrom not in seqs:
            n_bad_chrom += 1
            continue
        window = extract_window(seqs[chrom], gstart, gend, strand, args.window_size)
        if window is None:
            n_window_failed += 1
            continue
        records.append(
            dict(
                gene_id=gene_id, chrom=chrom, start=gstart, end=gend, strand=strand,
                raw_count=raw_count, sequence=window,
            )
        )

    log.info(
        "Built %d windowed examples (dropped: %d no gene record, %d unknown chrom, "
        "%d chrom shorter than window_size)",
        len(records), n_no_gene_record, n_bad_chrom, n_window_failed,
    )
    if not records:
        raise SystemExit(
            "No gene windows were built -- check that --gff/--fasta/--counts-tsv all "
            "refer to the same genome assembly."
        )

    out = pd.DataFrame.from_records(records)
    lengths = out["sequence"].str.len()
    assert (lengths == args.window_size).all(), "Internal error: not all windows are window_size bp"
    out["label"] = np.log1p(out["raw_count"])

    log.info(
        "raw_count stats: min=%.2f median=%.2f mean=%.2f max=%.2f",
        out["raw_count"].min(), out["raw_count"].median(), out["raw_count"].mean(), out["raw_count"].max(),
    )

    shuffled = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_val = max(1, int(len(shuffled) * args.val_fraction))
    val_df = shuffled.iloc[:n_val]
    train_df = shuffled.iloc[n_val:]

    train_path = f"{args.output_prefix}_train.tsv"
    val_path = f"{args.output_prefix}_val.tsv"
    train_df.to_csv(train_path, sep="\t", index=False)
    val_df.to_csv(val_path, sep="\t", index=False)
    log.info("Wrote %d train rows to %s", len(train_df), train_path)
    log.info("Wrote %d val rows to %s", len(val_df), val_path)
    log.info(
        "Note: 'label' is log1p(raw_count) for training stability. To recover an integer "
        "count prediction from lora_fine_tune.py predict's 'predicted_value' column, apply "
        "round(max(0, expm1(predicted_value)))."
    )


if __name__ == "__main__":
    main()
