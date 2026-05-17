#!/usr/bin/env python
# coding=utf-8
"""
prepare_dataset_from_tsv.py
---------------------------

Convert a pre-sliced random-chunk TSV (produced by
`src/scripts/slice_rand_genome_from_fasta.py`) into a HuggingFace
DatasetDict that the MLM training script (HF_pre_train.py or
HF_pre_train_lora.py) can consume directly with `load_from_disk(...)`.

Why a separate script (not FASTA prep)?
  The v1 pipeline read the raw FASTA and sliced it into non-overlapping
  windows. The professor's feedback was that the raw FASTA carries too
  much junk; a pre-curated "random chunks" TSV is a much cleaner
  training corpus. This script is the minimal bridge: TSV -> HF format.

Expected TSV schema (case-insensitive; we normalise to lowercase):
  - A column with the DNA sequence. Common names: `sequence`, `seq`.
  - Optional metadata columns: `chrom`, `start`, `end`, `strand`,
    `assembly`. Any present will be preserved.

Output schema (matches what HF_pre_train.py's tokenize_function expects
in `examples["seq"]` plus the `remove_columns` list at line ~445):
    assembly, chrom, start, end, strand, seq
  Missing metadata is filled with sensible defaults so downstream
  `remove_columns` doesn't blow up.

Splits:
    train / validation only (no test). 95 / 5 by default, seed 42.

Usage:
    python src/scripts/prepare_dataset_from_tsv.py \
        --tsv data/GCF_002870075.5_Lsat_Salinas_v15_genomic_rand_chunk_8192_samples_50k.tsv \
        --output_dir data/lettuce_hf_dataset_rand8192_50k \
        --expected_length 8192
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict


# ----------------------------------------------------------------------
# Logging — keep output readable when run under torchrun later (this
# script itself is single-process, but matching style with the rest of
# the repo helps when grepping logs).
# ----------------------------------------------------------------------
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("prepare_dataset_from_tsv")


# ----------------------------------------------------------------------
# Schema the downstream training script demands. We will guarantee
# these columns exist (filling defaults for any the TSV omits) so that
# HF_pre_train.py's `remove_columns = [...]` call never KeyErrors.
# ----------------------------------------------------------------------
REQUIRED_OUTPUT_COLUMNS = ["assembly", "chrom", "start", "end", "strand", "seq"]

# Default metadata stamped into rows whose TSV did not carry the column.
# `assembly` defaults to the lettuce ref so eval scripts can filter by
# assembly later without having to guess.
DEFAULT_ASSEMBLY = "GCF_002870075.5_Lsat_Salinas_v15"


def _detect_seq_column(columns: list[str]) -> str:
    """Find the DNA-sequence column under a few common names."""
    lowered = {c.lower(): c for c in columns}
    for candidate in ("seq", "sequence", "dna", "chunk"):
        if candidate in lowered:
            return lowered[candidate]
    raise SystemExit(
        f"ERROR: could not find a sequence column in TSV. "
        f"Looked for one of: seq, sequence, dna, chunk. "
        f"Found columns: {columns}"
    )


def _normalise_columns(df: pd.DataFrame, seq_col: str) -> pd.DataFrame:
    """
    Return a DataFrame with exactly the columns HF_pre_train.py expects.
    - Renames the detected sequence column to `seq`.
    - Lowercases all column names (so a TSV with `Chrom` becomes `chrom`).
    - Fills any missing required column with a sane default.
    """
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if seq_col.lower() != "seq":
        df = df.rename(columns={seq_col.lower(): "seq"})

    for col, default in [
        ("assembly", DEFAULT_ASSEMBLY),
        ("chrom", "unknown"),
        ("start", -1),
        ("end", -1),
        ("strand", "+"),
    ]:
        if col not in df.columns:
            df[col] = default
            log.info("Column %r missing from TSV — filling default %r", col, default)

    # Drop any extra columns we don't need; keeps the HF dataset compact.
    df = df[REQUIRED_OUTPUT_COLUMNS]
    return df


def _validate_lengths(df: pd.DataFrame, expected_length: int | None) -> None:
    """
    Sanity-check that every sequence is the same length (8192 bp for
    PlantCAD2-Small). Mismatched lengths would crash deep inside the
    tokenizer / collator with a confusing error — much friendlier to
    catch here.
    """
    lengths = df["seq"].str.len()
    unique = lengths.unique()
    if expected_length is not None:
        if not (unique == expected_length).all():
            raise SystemExit(
                f"ERROR: expected every sequence to be {expected_length} bp, "
                f"but found lengths: {sorted(unique.tolist())[:10]}..."
            )
        log.info("All %d sequences are exactly %d bp ✓", len(df), expected_length)
    else:
        log.info("Sequence length distribution: min=%d max=%d mean=%.1f",
                 lengths.min(), lengths.max(), lengths.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tsv", required=True, type=Path,
                        help="Path to the pre-sliced TSV (e.g. *_rand_chunk_8192_samples_50k.tsv)")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Where to save the HuggingFace DatasetDict (load_from_disk-compatible)")
    parser.add_argument("--expected_length", type=int, default=8192,
                        help="Expected length of every sequence in bp. Use 0 to skip the check.")
    parser.add_argument("--val_fraction", type=float, default=0.05,
                        help="Fraction of rows held out for validation (default 0.05 = 5%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the train/val shuffle so the split is reproducible")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Optional cap on rows loaded (useful for a quick sanity build)")
    args = parser.parse_args()

    if not args.tsv.exists():
        raise SystemExit(f"ERROR: TSV not found: {args.tsv}")

    # --- Load TSV ------------------------------------------------------
    log.info("Reading TSV: %s", args.tsv)
    df = pd.read_csv(args.tsv, sep="\t", nrows=args.max_rows, dtype=str)
    log.info("Loaded %d rows, columns: %s", len(df), list(df.columns))

    # --- Detect + normalise -------------------------------------------
    seq_col = _detect_seq_column(list(df.columns))
    log.info("Detected sequence column: %r", seq_col)
    df = _normalise_columns(df, seq_col)

    # Coerce numeric metadata back to int where it makes sense — pandas
    # read everything as str above (we did that to be safe with the seq
    # column). HF Datasets is happy with mixed types, but ints are nicer.
    for col in ("start", "end"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

    # --- Validate ------------------------------------------------------
    if args.expected_length and args.expected_length > 0:
        _validate_lengths(df, args.expected_length)

    # --- HF DatasetDict ------------------------------------------------
    log.info("Converting to HuggingFace Dataset and splitting train/val "
             "(val_fraction=%.3f, seed=%d)", args.val_fraction, args.seed)
    full = Dataset.from_pandas(df, preserve_index=False)
    split = full.train_test_split(test_size=args.val_fraction, seed=args.seed)
    # HF_pre_train.py reads `raw_datasets["train"]` and
    # `raw_datasets["validation"]` — we rename "test" -> "validation".
    ds = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })
    log.info("Final split sizes: train=%d, validation=%d",
             len(ds["train"]), len(ds["validation"]))

    # --- Save ----------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.output_dir))
    log.info("Saved DatasetDict to %s", args.output_dir)
    log.info("Load it later with:  datasets.load_from_disk('%s')", args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
