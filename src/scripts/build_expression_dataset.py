#!/usr/bin/env python3
"""
build_expression_dataset.py
----------------------------

Build a (sequence, label) train/validation TSV for gene-expression-count
regression fine-tuning of PlantCAD2, from:
  - one or more per-exon RNA-seq read count matrices (data/merged_output.tsv
    format: one row per exon ID, one column per experiment) paired with a
    metadata CSV describing each experiment column's Run/time_point/tissue/
    treatment. Multiple counts/metadata pairs are treated as separate
    experiment batches over the same exon set (e.g. different studies) and
    merged by exon ID.
  - a GFF3 gene/exon annotation for the same assembly
  - the matching genome FASTA

For each gene:
  1. filter each counts TSV's experiment columns to the ones whose metadata
     treatment matches --treatment (e.g. "Mock inoculation")
  2. for each exon, average the remaining columns within each time_point
     (this averages away every other metadata dimension, e.g. tissue,
     replicate, source study)
  3. integrate those per-time_point averages over time_point (trapezoidal
     rule) to get one area-under-curve value per exon
  4. take the max exon AUC across each gene's exons as its expression label
  5. extract a fixed-length DNA window centered on the gene's midpoint
     (reverse-complemented for '-' strand genes so orientation is 5'->3')

Usage:
    python src/scripts/build_expression_dataset.py \
        --counts-tsv data/expression_72.tsv data/expression_85.tsv \
        --metadata-csv data/MetaData_72.csv data/MetaData_85.csv \
        --gff data/GCF_002870075.5_Lsat_Salinas_v15_genomic.gff \
        --fasta data/GCF_002870075.5_Lsat_Salinas_v15_genomic.fna \
        --output-prefix data/expression_dataset

    # Smoke test on a slice of each (large) counts TSV instead of the full file:
    python src/scripts/build_expression_dataset.py \
        --counts-tsv data/expression_72.tsv data/expression_85.tsv \
        --metadata-csv data/MetaData_72.csv data/MetaData_85.csv \
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


_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy >=2.0 renamed trapz to trapezoid


def _auc(row: pd.Series) -> float:
    """Trapezoidal AUC over a row's non-null (time_point -> value) entries."""
    valid = row.dropna()
    if len(valid) < 2:
        return np.nan
    return float(_trapezoid(valid.to_numpy(dtype=float), x=valid.index.to_numpy(dtype=float)))


def load_gene_labels(
    counts_tsvs: List[str],
    metadata_csvs: List[str],
    exon_to_gene: Dict[str, str],
    treatment: str,
    max_rows_per_tsv: Optional[int] = None,
) -> pd.DataFrame:
    """Merge one or more per-exon count TSVs (each paired with a metadata CSV
    describing its experiment columns) by exon ID, then for each exon:
      1. keep only experiment columns whose treatment matches `treatment`
      2. average the kept columns within each time_point (averaging away
         every other metadata dimension: tissue, replicate, source study)
      3. integrate the per-time_point averages over time_point (trapezoidal
         rule) to get one AUC value
    Finally take the max exon AUC per gene.
    """
    per_file = []
    col_time_point: Dict[str, float] = {}
    for path, metadata_path in zip(counts_tsvs, metadata_csvs):
        df = pd.read_csv(path, sep="\t", nrows=max_rows_per_tsv)
        is_exon = df["ID"].str.startswith("exon-")
        log.info(
            "Loaded %d rows from %s (%d exon rows, %d non-exon rows dropped, %d experiment columns)",
            len(df), path, int(is_exon.sum()), int((~is_exon).sum()), df.shape[1] - 1,
        )
        df = df.loc[is_exon].set_index("ID")

        metadata = pd.read_csv(metadata_path).set_index("Run")
        # Prefix columns with the source file's stem so experiment columns that
        # happen to share a name across files (not expected, but not guaranteed)
        # can't collide when concatenated below.
        stem = Path(path).stem
        kept_cols = []
        renamed = {}
        for col in df.columns:
            run = col[: -len("_count.tsv")] if col.endswith("_count.tsv") else col
            if run not in metadata.index or metadata.loc[run, "treatment"] != treatment:
                continue
            new_col = f"{stem}::{col}"
            renamed[col] = new_col
            col_time_point[new_col] = metadata.loc[run, "time_point"]
            kept_cols.append(col)
        log.info(
            "%s: kept %d/%d experiment columns with treatment == %r",
            path, len(kept_cols), df.shape[1], treatment,
        )
        per_file.append(df[kept_cols].rename(columns=renamed))

    # Align by exon ID (outer join): an exon missing from one file -- e.g. a
    # truncated --max-rows-per-tsv smoke-test read -- just contributes NaN for
    # that file's columns there, which the mean below ignores.
    merged = pd.concat(per_file, axis=1)
    log.info(
        "Merged %d counts TSV(s) into %d exon rows x %d treatment-filtered experiment columns",
        len(counts_tsvs), len(merged), merged.shape[1],
    )
    if merged.shape[1] == 0:
        raise SystemExit(f"No experiment columns matched treatment={treatment!r}; check --treatment.")

    time_points = pd.Series(col_time_point)
    by_time_point = merged.T.groupby(time_points).mean().T.sort_index(axis=1)
    log.info("Averaged into %d distinct time_point columns: %s", by_time_point.shape[1], list(by_time_point.columns))

    auc = by_time_point.apply(_auc, axis=1)

    gene_id = auc.index.to_series().map(exon_to_gene)
    unmapped = int(gene_id.isna().sum())
    if unmapped:
        log.warning("%d exon rows had no matching gene in the GFF; dropping", unmapped)

    labels_df = pd.DataFrame({"auc": auc, "gene_id": gene_id}).dropna(subset=["gene_id", "auc"])
    gene_labels = (
        labels_df.groupby("gene_id")["auc"].max().rename("auc_count").reset_index()
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
        help="One or more per-exon read count TSVs (e.g. data/expression_72.tsv data/expression_85.tsv). "
             "Must share the same exon ID space; treated as separate experiment batches and merged by "
             "exon ID. Paired positionally with --metadata-csv.",
    )
    parser.add_argument(
        "--metadata-csv", required=True, nargs="+",
        help="One metadata CSV per --counts-tsv file, in the same order (e.g. data/MetaData_72.csv "
             "data/MetaData_85.csv). Each has Run,time_point,tissue,treatment columns describing that "
             "file's experiment columns (Run matches the '<Run>_count.tsv' column name).",
    )
    parser.add_argument(
        "--treatment", default="Mock inoculation",
        help="Only experiment columns whose metadata treatment equals this value are used (default: "
             "%(default)r).",
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
    if len(args.counts_tsv) != len(args.metadata_csv):
        parser.error("--counts-tsv and --metadata-csv must be given the same number of times, in matching order")

    genes, exon_to_gene = parse_gff(args.gff)
    gene_labels = load_gene_labels(
        args.counts_tsv, args.metadata_csv, exon_to_gene, args.treatment, args.max_rows_per_tsv
    )

    log.info("Loading genome FASTA: %s", args.fasta)
    seqs = {r.id: str(r.seq) for r in tqdm(SeqIO.parse(args.fasta, "fasta"), desc="Reading FASTA")}

    records = []
    n_no_gene_record = 0
    n_bad_chrom = 0
    n_window_failed = 0
    for row in tqdm(gene_labels.itertuples(index=False), total=len(gene_labels), desc="Extracting windows"):
        gene_id, auc_count = row.gene_id, row.auc_count
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
                auc_count=auc_count, sequence=window,
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
    out["label"] = np.log1p(out["auc_count"])

    log.info(
        "auc_count stats: min=%.2f median=%.2f mean=%.2f max=%.2f",
        out["auc_count"].min(), out["auc_count"].median(), out["auc_count"].mean(), out["auc_count"].max(),
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
        "Note: 'label' is log1p(auc_count) for training stability, where auc_count is the "
        "trapezoidal area under the time_point-vs-expression curve (Mock-inoculation-filtered, "
        "averaged over tissue/replicate/study). To recover an auc_count prediction from "
        "lora_fine_tune.py predict's 'predicted_value' column, apply max(0, expm1(predicted_value))."
    )


if __name__ == "__main__":
    main()
