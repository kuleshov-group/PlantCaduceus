#!/usr/bin/env python3
"""
Extract subsequences with abnormally high model confidence for a given gene.

Loads the .npz produced by visualize_confidence.py and finds contiguous
regions where the (smoothed) P(reference allele | context) exceeds a
threshold defined as mean + N * std over the window.

Output is a TSV with columns:
  chrom  start  end  length  mean_prob  max_prob  sequence
"""

import argparse
import gzip
import logging
import os
import sys

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

PROBS_PREFIX = "probs_"


def _parse_attrs(attr_s: str) -> dict:
    attrs = {}
    for part in attr_s.strip().split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            attrs[k.strip()] = v.strip()
    return attrs


def _lookup_chrom_from_gff(gff_path: str, gene_id: str) -> str:
    """Scan GFF3 for a gene feature whose ID matches gene_id and return its chromosome."""
    opener = gzip.open if gff_path.endswith(".gz") else open
    with opener(gff_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, feature, _, _, _, _, _, attr_s = parts[:9]
            if feature != "gene":
                continue
            if _parse_attrs(attr_s).get("ID", "") == gene_id:
                return chrom
    raise ValueError(f"Gene '{gene_id}' not found in {gff_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract high-confidence subsequences from a saved .npz confidence file."
    )
    parser.add_argument("gene", help="Gene ID (e.g. gene-LOC111905264); used to locate <gene>.npz.")
    parser.add_argument("--npz-dir", default="./confidence_plots",
                        help="Directory containing the .npz files (default: ./confidence_plots).")
    parser.add_argument("--gff", default=None,
                        help="GFF3 annotation file. Used to fetch the chromosome when it is "
                             "absent from the .npz (e.g. files generated before chrom was saved).")
    parser.add_argument("--model", default=None,
                        help="Model name to use (substring match against stored keys). "
                             "If omitted and multiple models are present, probabilities "
                             "are averaged across all models.")
    parser.add_argument("--n-sigma", type=float, default=2.0,
                        help="Threshold = mean + n_sigma * std of the smoothed signal "
                             "(default: 2.0). Use --threshold to set a fixed value instead.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed confidence threshold in [0, 1]. Overrides --n-sigma.")
    parser.add_argument("--smooth-window", type=int, default=200,
                        help="Rolling-mean window in bp applied before thresholding (default: 200).")
    parser.add_argument("--min-length", type=int, default=10,
                        help="Minimum region length in bp to report (default: 10).")
    parser.add_argument("--merge-gap", type=int, default=0,
                        help="Merge regions separated by at most this many bp (default: 0).")
    parser.add_argument("--output", default=None,
                        help="Output TSV path. Defaults to <npz-dir>/<gene>.tsv. Use '-' for stdout.")
    parser.add_argument("--fasta", action="store_true",
                        help="Also write a FASTA file (same directory as TSV, named <gene>.fa).")
    return parser.parse_args()


def _smooth(probs: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(probs)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )


def _select_probs(data: dict, model_hint: str | None) -> tuple[np.ndarray, str]:
    """Return (prob_array, label) for the requested model(s).

    If model_hint is given, find keys that contain it as a substring.
    If multiple models are present and no hint given, average them all.
    """
    keys = [k for k in data.files if k.startswith(PROBS_PREFIX)]
    if not keys:
        raise KeyError("No probability arrays found in npz (expected keys starting with 'probs_').")

    if model_hint is not None:
        matched = [k for k in keys if model_hint in k]
        if not matched:
            raise KeyError(
                f"No key matching '{model_hint}' found. Available: {[k[len(PROBS_PREFIX):] for k in keys]}"
            )
        if len(matched) > 1:
            logger.warning("Multiple keys match '%s': %s. Using first.", model_hint, matched)
        key = matched[0]
        return data[key].astype(np.float32), key[len(PROBS_PREFIX):]

    if len(keys) == 1:
        return data[keys[0]].astype(np.float32), keys[0][len(PROBS_PREFIX):]

    # Average across all models, ignoring NaN at each position independently
    stacked = np.stack([data[k].astype(np.float32) for k in keys], axis=0)
    avg = np.nanmean(stacked, axis=0)
    label = "average(" + ", ".join(k[len(PROBS_PREFIX):] for k in keys) + ")"
    logger.info("Averaging %d models: %s", len(keys), label)
    return avg, label


def _find_regions(signal: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Return list of (start, end) index pairs (0-based, end exclusive) above threshold."""
    above = np.where(~np.isnan(signal), signal > threshold, False)
    regions = []
    in_region = False
    for i, val in enumerate(above):
        if val and not in_region:
            region_start = i
            in_region = True
        elif not val and in_region:
            regions.append((region_start, i))
            in_region = False
    if in_region:
        regions.append((region_start, len(signal)))
    return regions


def _merge_regions(regions: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    if not regions or gap <= 0:
        return regions
    merged = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()

    npz_path = os.path.join(args.npz_dir, f"{args.gene}.npz")
    if not os.path.exists(npz_path):
        logger.error("File not found: %s", npz_path)
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)

    window_start = int(data["window_start"])
    sequence = str(data["sequence"][0])

    if "chrom" in data.files:
        chrom = str(data["chrom"][0])
    elif args.gff:
        logger.info("'chrom' not in npz; looking up from %s", args.gff)
        chrom = _lookup_chrom_from_gff(args.gff, args.gene)
    else:
        logger.error(
            "'chrom' not found in %s. Re-run visualize_confidence.py to regenerate the npz, "
            "or provide --gff to look up the chromosome from the annotation.",
            npz_path,
        )
        sys.exit(1)

    probs, model_label = _select_probs(data, args.model)
    smoothed = _smooth(probs, args.smooth_window)

    # Compute threshold
    valid = smoothed[~np.isnan(smoothed)]
    if len(valid) == 0:
        logger.error("All positions are NaN; cannot compute threshold.")
        sys.exit(1)

    if args.threshold is not None:
        threshold = args.threshold
        logger.info("Using fixed threshold: %.4f", threshold)
    else:
        threshold = float(valid.mean() + args.n_sigma * valid.std())
        logger.info(
            "Signal mean=%.4f std=%.4f → threshold (mean + %.1f σ) = %.4f",
            valid.mean(), valid.std(), args.n_sigma, threshold,
        )

    # Find, merge, and filter regions
    regions = _find_regions(smoothed, threshold)
    regions = _merge_regions(regions, args.merge_gap)
    regions = [(s, e) for s, e in regions if e - s >= args.min_length]

    logger.info("Found %d high-confidence region(s) for %s (model: %s)",
                len(regions), args.gene, model_label)

    # Build rows
    rows = []
    fasta_entries = []
    for start_idx, end_idx in regions:
        g_start = window_start + start_idx
        g_end = window_start + end_idx
        subseq = sequence[start_idx:end_idx]
        region_probs = probs[start_idx:end_idx]
        mean_p = float(np.nanmean(region_probs))
        max_p = float(np.nanmax(region_probs))
        rows.append({
            "chrom": chrom,
            "start": g_start,
            "end": g_end,
            "rel_start": start_idx,
            "rel_end": end_idx,
            "length": g_end - g_start,
            "mean_prob": round(mean_p, 6),
            "max_prob": round(max_p, 6),
            "sequence": subseq,
        })
        fasta_entries.append(
            f">{args.gene}_{chrom}:{g_start}-{g_end} mean={mean_p:.4f} max={max_p:.4f}\n{subseq}"
        )

    # Resolve output paths — always derived from the gene argument, never from user-typed paths
    if args.output is None:
        output_path = os.path.join(args.npz_dir, f"{args.gene}.tsv")
    else:
        output_path = args.output  # explicit path or "-" for stdout

    out_dir = (
        os.path.dirname(os.path.abspath(output_path))
        if output_path != "-"
        else "."
    )
    fasta_path = os.path.join(out_dir, f"{args.gene}.fa")

    # Write TSV
    if not rows:
        logger.warning("No regions passed the filters.")

    header = ["chrom", "start", "end", "rel_start", "rel_end", "length", "mean_prob", "max_prob", "sequence"]
    lines = ["\t".join(header)]
    for r in rows:
        lines.append("\t".join(str(r[c]) for c in header))
    tsv_content = "\n".join(lines) + "\n"

    if output_path == "-":
        sys.stdout.write(tsv_content)
    else:
        os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w") as fh:
            fh.write(tsv_content)
        logger.info("Saved TSV → %s", output_path)

    # Optional FASTA — always named after the gene, not the TSV
    if args.fasta:
        with open(fasta_path, "w") as fh:
            fh.write("\n".join(fasta_entries) + "\n")
        logger.info("Saved FASTA → %s", fasta_path)


if __name__ == "__main__":
    main()
