#!/usr/bin/env python3
"""
Compare the distribution of per-base model confidence (P(ref | context), as
produced by visualize_confidence.py and cached in .npz files) across several
genomic categories:

  - All locations       every scored position across all .npz windows
  - Genic               positions overlapping a "gene" feature in a GFF3 file
  - Flanking            positions NOT overlapping a "gene" feature (complement
                        of Genic)
  - <regulatory type>   one category per distinct regulator type (column 2) in
                        a regulator.tab file of regulatory sequence locations
                        (e.g. TATA, CAAT)
  - Masked (N)          positions that are hard-masked (base == 'N') in a
                        separately supplied masked FASTA, using the confidence
                        value computed from the (unmasked) reference sequence

Produces a violin plot of all categories (one panel per model found in the
.npz files) annotated with mean/variance, plus a TSV of summary statistics.
"""

import argparse
import bisect
import gzip
import logging
import os
import re
from collections import defaultdict
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO

logger = logging.getLogger(__name__)

GENIC_LABEL = "Genic"
FLANKING_LABEL = "Flanking"
ALL_LABEL = "All locations"
MASKED_LABEL = "Masked (N)"


# ---------------------------------------------------------------------------
# FASTA
# ---------------------------------------------------------------------------

def load_fasta(fna_path: str) -> dict:
    logger.info("Loading FASTA from %s", fna_path)
    opener = gzip.open if fna_path.endswith(".gz") else open
    mode = "rt" if fna_path.endswith(".gz") else "r"
    with opener(fna_path, mode) as fh:
        return SeqIO.to_dict(SeqIO.parse(fh, "fasta"))


# ---------------------------------------------------------------------------
# GFF3 gene intervals
# ---------------------------------------------------------------------------

def parse_gff_genes(gff_path: str) -> dict:
    """Return {chrom: [(start, end), ...]} (0-based, half-open) sorted by start."""
    genes = defaultdict(list)
    opener = gzip.open if gff_path.endswith(".gz") else open
    mode = "rt" if gff_path.endswith(".gz") else "r"
    with opener(gff_path, mode) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            try:
                start = int(parts[3]) - 1
                end = int(parts[4])
            except ValueError:
                continue
            genes[parts[0]].append((start, end))
    for chrom in genes:
        genes[chrom].sort()
    logger.info("Loaded gene intervals for %d chromosomes from %s", len(genes), gff_path)
    return dict(genes)


# ---------------------------------------------------------------------------
# regulator.tab regulatory-sequence intervals
# ---------------------------------------------------------------------------

# First column of regulator.tab: ID=<gene_id>::<chrom>:<flank_start>-<flank_end>()
_REG_COL0_RE = re.compile(r'^ID=([^:]+)::([^:]+):(\d+)-(\d+)\(')


def parse_regulator_tab(tab_path: str) -> dict:
    """Return {chrom: [(start, end, reg_type), ...]} sorted by start.

    regulator.tab columns: col0 = "ID=<gene_id>::<chrom>:<flank_start_1based>-
    <flank_end_1based>()", col1 = reg_type, col2 = matched sequence,
    col3 = 1-based offset of the match from flank_start. Genomic coordinates
    (0-based, half-open) are derived as:
      start = flank_start_1based + rel_pos - 1
      end   = start + len(seq)
    """
    regs = defaultdict(list)
    with open(tab_path) as fh:
        fh.readline()  # header
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            m = _REG_COL0_RE.match(parts[0])
            if not m:
                continue
            chrom = m.group(2)
            flank_start = int(m.group(3))  # 1-based
            reg_type = parts[1]
            seq = parts[2]
            try:
                rel_pos = int(parts[3])
            except ValueError:
                continue
            start_0 = flank_start + rel_pos - 1
            end_0 = start_0 + len(seq)
            regs[chrom].append((start_0, end_0, reg_type))
    for chrom in regs:
        regs[chrom].sort()
    n_types = len({t for ivs in regs.values() for _, _, t in ivs})
    logger.info("Loaded %d regulatory intervals (%d types) from %s",
                sum(len(v) for v in regs.values()), n_types, tab_path)
    return dict(regs)


# ---------------------------------------------------------------------------
# Interval overlap
# ---------------------------------------------------------------------------

def overlap_mask(intervals: list, window_start: int, length: int) -> np.ndarray:
    """Boolean mask of length `length`; True where the window position falls
    inside any (start, end, ...) interval. `intervals` must be sorted by start.
    """
    mask = np.zeros(length, dtype=bool)
    if not intervals:
        return mask
    window_end = window_start + length
    starts = [iv[0] for iv in intervals]
    hi = bisect.bisect_left(starts, window_end)
    for iv in intervals[:hi]:
        s, e = iv[0], iv[1]
        if e <= window_start:
            continue
        rel_s = max(0, s - window_start)
        rel_e = min(length, e - window_start)
        if rel_s < rel_e:
            mask[rel_s:rel_e] = True
    return mask


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def collect_category_values(
    npz_dir: str,
    fasta_dict: dict,
    gene_intervals: dict,
    reg_intervals: dict,
    masked_fasta_dict: dict,
) -> dict:
    """Return {model_name: {category: np.ndarray of confidence values}}."""
    npz_paths = sorted(glob(os.path.join(npz_dir, "*.npz")))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")
    logger.info("Found %d .npz files in %s", len(npz_paths), npz_dir)

    reg_types = sorted({t for ivs in reg_intervals.values() for _, _, t in ivs})
    values = defaultdict(lambda: defaultdict(list))  # model -> category -> [arrays]

    for npz_path in npz_paths:
        data = np.load(npz_path, allow_pickle=True)
        chrom = str(data["chrom"][0])
        window_start = int(data["window_start"])
        sequence = str(data["sequence"][0])
        length = len(sequence)

        model_keys = [k for k in data.files if k.startswith("probs_")]
        if not model_keys:
            logger.warning("No probs_* arrays in %s; skipping.", npz_path)
            continue

        # Sanity check against the supplied reference FASTA.
        if chrom in fasta_dict:
            ref_sub = str(fasta_dict[chrom].seq[window_start:window_start + length]).upper()
            if ref_sub and ref_sub != sequence.upper()[:len(ref_sub)]:
                logger.warning(
                    "Reference FASTA mismatch for %s at %s:%d-%d; check --fasta.",
                    npz_path, chrom, window_start, window_start + length,
                )

        gene_ivs = gene_intervals.get(chrom, [])
        gmask = overlap_mask(gene_ivs, window_start, length)
        flanking_mask = ~gmask

        type_masks = {}
        for rtype in reg_types:
            ivs = [iv for iv in reg_intervals.get(chrom, []) if iv[2] == rtype]
            type_masks[rtype] = overlap_mask(ivs, window_start, length)

        masked_mask = np.zeros(length, dtype=bool)
        if chrom in masked_fasta_dict:
            m_sub = str(masked_fasta_dict[chrom].seq[window_start:window_start + length])
            if len(m_sub) < length:
                m_sub = m_sub + "N" * (length - len(m_sub))
            masked_mask = np.array([c == "N" for c in m_sub], dtype=bool)
        else:
            logger.warning("Chromosome %s not found in masked FASTA; skipping mask category for %s.",
                            chrom, npz_path)

        for key in model_keys:
            model_name = key[len("probs_"):]
            probs = data[key]
            valid = ~np.isnan(probs)

            values[model_name][ALL_LABEL].append(probs[valid])
            values[model_name][GENIC_LABEL].append(probs[gmask & valid])
            values[model_name][FLANKING_LABEL].append(probs[flanking_mask & valid])
            for rtype, tmask in type_masks.items():
                values[model_name][rtype].append(probs[tmask & valid])
            values[model_name][MASKED_LABEL].append(probs[masked_mask & valid])

    return {
        model: {cat: np.concatenate(arrs) if arrs else np.array([])
                for cat, arrs in cats.items()}
        for model, cats in values.items()
    }, reg_types


def category_groups(reg_types: list) -> list:
    """Return the categories grouped for display: [[All], [Genic, Flanking],
    [reg types...], [Masked]]. Vertical separators are drawn between groups,
    not within them.
    """
    return [[ALL_LABEL], [GENIC_LABEL, FLANKING_LABEL], reg_types, [MASKED_LABEL]]


def category_order(reg_types: list) -> list:
    return [lbl for group in category_groups(reg_types) for lbl in group]


def compute_stats(model_values: dict, reg_types: list) -> list:
    rows = []
    for model, cats in model_values.items():
        for cat in category_order(reg_types):
            arr = cats.get(cat, np.array([]))
            n = arr.size
            rows.append({
                "model": model,
                "category": cat,
                "n": n,
                "mean": float(np.mean(arr)) if n else float("nan"),
                "variance": float(np.var(arr)) if n else float("nan"),
                "std": float(np.std(arr)) if n else float("nan"),
                "median": float(np.median(arr)) if n else float("nan"),
            })
    return rows


def save_stats_tsv(rows: list, out_path: str) -> None:
    cols = ["model", "category", "n", "mean", "variance", "std", "median"]
    with open(out_path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    logger.info("Saved summary statistics -> %s", out_path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_PALETTE = ["#546e7a", "#1565c0", "#388e3c", "#f57c00", "#7b1fa2",
            "#c62828", "#0277bd", "#00838f", "#4e342e", "#ad1457"]


def plot_violin(model_values: dict, reg_types: list, stats_rows: list, out_path: str) -> None:
    groups = category_groups(reg_types)
    labels = category_order(reg_types)
    models = list(model_values.keys())
    n_models = len(models)

    # x position of a vertical separator between each pair of adjacent groups
    # (drawn at the midpoint between the last item of one group and the first
    # item of the next; no separator after the final group).
    group_sizes = [len(g) for g in groups]
    boundaries = np.cumsum(group_sizes)[:-1] - 0.5

    stats_lookup = {(r["model"], r["category"]): r for r in stats_rows}

    fig, axes = plt.subplots(n_models, 1, figsize=(max(8, 1.6 * len(labels)), 4.5 * n_models),
                              squeeze=False)
    axes = axes[:, 0]

    for ax, model in zip(axes, models):
        data = [model_values[model].get(lbl, np.array([])) for lbl in labels]
        positions = list(range(len(labels)))
        colors = (_PALETTE * (len(labels) // len(_PALETTE) + 1))[:len(labels)]

        # KDE-based violins need >=2 points; plot those separately from sparse
        # categories (0 or 1 points), which are drawn as scatter markers instead.
        violin_pos = [p for p, d in zip(positions, data) if d.size >= 2]
        violin_data = [d for d in data if d.size >= 2]
        violin_colors = [c for c, d in zip(colors, data) if d.size >= 2]
        if violin_data:
            parts = ax.violinplot(violin_data, positions=violin_pos,
                                  showmedians=True, showextrema=True)
            for pc, col in zip(parts["bodies"], violin_colors):
                pc.set_facecolor(col)
                pc.set_alpha(0.7)
            for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                parts[key].set_color("#333333")

        for pos, d, col in zip(positions, data, colors):
            if d.size < 2:
                ax.scatter([pos] * d.size, d, color=col, zorder=3)

        for pos, lbl in zip(positions, labels):
            r = stats_lookup.get((model, lbl))
            if r is None or r["n"] == 0:
                continue
            ax.text(pos, 1.05, f"n={r['n']}\nμ={r['mean']:.3f}\nσ²={r['variance']:.4f}",
                    ha="center", va="bottom", fontsize=7, transform=ax.get_xaxis_transform())

        for b in boundaries:
            ax.axvline(b, color="#999999", linestyle="--", linewidth=1, zorder=0)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(0, 1.32)
        ax.set_ylabel("P(ref | context)", fontsize=10)
        ax.set_title(model, fontsize=11, pad=45)

    fig.suptitle("Model confidence by genomic category", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved violin plot -> %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare model confidence distributions across genomic categories."
    )
    parser.add_argument("--npz-dir", default="confidence_plots/",
                        help="Folder of .npz files produced by visualize_confidence.py "
                             "(default: confidence_plots/).")
    parser.add_argument("--fasta", required=True,
                        help="Reference genome FASTA the .npz sequences were extracted from.")
    parser.add_argument("--gff", required=True, help="GFF3 file with gene annotations.")
    parser.add_argument("--regulator-tab", default="data/regulator.tab",
                        help="regulator.tab file of regulatory sequence locations "
                             "(default: data/regulator.tab).")
    parser.add_argument("--masked-fasta", required=True,
                        help="FASTA with some bases hard-masked to 'N' (e.g. repeat-masked).")
    parser.add_argument("--output", default=None,
                        help="Output violin plot path "
                             "(default: <npz-dir>/confidence_category_comparison.png).")
    parser.add_argument("--stats-output", default=None,
                        help="Output TSV of summary statistics "
                             "(default: <npz-dir>/confidence_category_stats.tsv).")
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(args.npz_dir, "confidence_category_comparison.png")
    if args.stats_output is None:
        args.stats_output = os.path.join(args.npz_dir, "confidence_category_stats.tsv")
    return args


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()

    fasta_dict = load_fasta(args.fasta)
    masked_fasta_dict = load_fasta(args.masked_fasta)
    gene_intervals = parse_gff_genes(args.gff)
    reg_intervals = parse_regulator_tab(args.regulator_tab)

    model_values, reg_types = collect_category_values(
        args.npz_dir, fasta_dict, gene_intervals, reg_intervals, masked_fasta_dict,
    )

    stats_rows = compute_stats(model_values, reg_types)
    for r in stats_rows:
        logger.info("%-30s %-15s n=%-8d mean=%.4f var=%.5f",
                    r["model"], r["category"], r["n"], r["mean"], r["variance"])

    save_stats_tsv(stats_rows, args.stats_output)
    plot_violin(model_values, reg_types, stats_rows, args.output)


if __name__ == "__main__":
    main()
