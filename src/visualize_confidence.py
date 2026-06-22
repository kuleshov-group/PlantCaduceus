#!/usr/bin/env python3
"""
Visualize per-base model confidence on genomic sequences.

For each gene in a GFF3 file, extracts a window of DNA centered on the gene,
runs the masked language model one position at a time, computes
P(reference allele | context) at every base, and saves a figure showing the
confidence signal alongside the gene/exon structure.
"""

import argparse
import gzip
import logging
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


logger = logging.getLogger(__name__)

NUCLEOTIDES = ("A", "C", "G", "T")
NUCLEOTIDES_LOWER = tuple(n.lower() for n in NUCLEOTIDES)
NUCLEOTIDE_TO_INDEX = {b: i for i, b in enumerate(NUCLEOTIDES)}

# Colors cycled over distinct regulator types in the visualization
_REG_PALETTE = ["#388e3c", "#f57c00", "#7b1fa2", "#c62828", "#0277bd", "#00838f", "#4e342e", "#546e7a"]

# Regex for the first column of regulator.tab:
# ID=<gene_id>::<chrom>:<flank_start>-<flank_end>()
_REG_COL0_RE = re.compile(r'^ID=([^:]+)::([^:]+):(\d+)-(\d+)\(')


# ---------------------------------------------------------------------------
# Model loading (mirrors zero-shot-eval.py)
# ---------------------------------------------------------------------------

def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CUDA GPU found.")
    logger.info("Using %d GPU(s) for inference.", torch.cuda.device_count())


def _load_model(model_path: str):
    model = AutoModelForMaskedLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.to(torch.float32)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# FASTA loading (mirrors zero_shot_score.py)
# ---------------------------------------------------------------------------

def load_fasta(fna_path: str) -> dict:
    logger.info("Loading FASTA from %s", fna_path)
    if fna_path.endswith(".gz"):
        with gzip.open(fna_path, "rt") as fh:
            return SeqIO.to_dict(SeqIO.parse(fh, "fasta"))
    return SeqIO.to_dict(SeqIO.parse(fna_path, "fasta"))


# ---------------------------------------------------------------------------
# GFF3 parsing
# ---------------------------------------------------------------------------

def _parse_attrs(attr_str: str) -> dict:
    attrs = {}
    for part in attr_str.strip().split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            attrs[k.strip()] = v.strip()
    return attrs


def parse_gff3(gff_path: str, num_genes: int) -> list:
    """Return up to num_genes gene dicts, each with exon lists attached.

    Each gene dict has keys:
      gene_id, chrom, start (0-based), end (0-based exclusive), strand,
      exons: list of (start, end) tuples (0-based, half-open).
    """
    gene_records = {}    # gene_id -> dict
    mrna_to_gene = {}    # mrna_id -> gene_id
    gene_name_to_id = {} # gene Name/gene attr -> gene_id (NCBI GFF3 fallback)
    exon_records = []    # list of {parents, gene_name, start, end}

    logger.info("Parsing GFF3 from %s", gff_path)
    opener = gzip.open(gff_path, "rt") if gff_path.endswith(".gz") else open(gff_path)
    with opener as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, feature, start_s, end_s, _, strand, _, attr_s = parts[:9]
            try:
                start = int(start_s) - 1  # GFF3 is 1-based inclusive → 0-based
                end = int(end_s)          # keep end 0-based exclusive
            except ValueError:
                continue
            attrs = _parse_attrs(attr_s)

            if feature == "gene":
                gene_id = attrs.get("ID", "")
                if gene_id:
                    gene_records[gene_id] = {
                        "gene_id": gene_id,
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "exons": [],
                    }
                    # Index by Name and gene attr for NCBI-style GFF3 fallback
                    for key in ("Name", "gene"):
                        val = attrs.get(key, "")
                        if val:
                            gene_name_to_id[val] = gene_id
            elif feature in ("mRNA", "transcript"):
                mrna_id = attrs.get("ID", "")
                parent = attrs.get("Parent", "")
                if mrna_id and parent:
                    mrna_to_gene[mrna_id] = parent
            elif feature == "exon":
                parent_raw = attrs.get("Parent", "")
                parents = [p.strip() for p in parent_raw.split(",") if p.strip()]
                exon_records.append({
                    "parents": parents,
                    "gene_name": attrs.get("gene", ""),  # NCBI GFF3: gene= attr on exon
                    "start": start,
                    "end": end,
                })

    # Attach exons to genes via three strategies (in priority order):
    # 1. Direct parent is a gene ID
    # 2. Parent is an mRNA/transcript → look up gene
    # 3. NCBI fallback: use gene= attribute on the exon row
    for exon in exon_records:
        resolved = False
        for parent_id in exon["parents"]:
            if parent_id in gene_records:
                gene_records[parent_id]["exons"].append((exon["start"], exon["end"]))
                resolved = True
            elif parent_id in mrna_to_gene:
                gene_id = mrna_to_gene[parent_id]
                if gene_id in gene_records:
                    gene_records[gene_id]["exons"].append((exon["start"], exon["end"]))
                    resolved = True
        if not resolved and exon["gene_name"] in gene_name_to_id:
            gene_id = gene_name_to_id[exon["gene_name"]]
            gene_records[gene_id]["exons"].append((exon["start"], exon["end"]))

    # De-duplicate exon lists and keep only genes with exons
    genes_with_exons = []
    for g in gene_records.values():
        if g["exons"]:
            g["exons"] = sorted(set(g["exons"]))
            genes_with_exons.append(g)

    selected = genes_with_exons[:num_genes]
    logger.info("Selected %d genes with exons (requested %d).", len(selected), num_genes)
    return selected


# ---------------------------------------------------------------------------
# BED / regulator input
# ---------------------------------------------------------------------------

def _norm_id(gid: str) -> str:
    """Strip ID= and gene- prefixes for flexible cross-format gene ID matching."""
    for pfx in ("ID=", "gene-"):
        if gid.startswith(pfx):
            gid = gid[len(pfx):]
    return gid


def parse_bed(bed_path: str) -> list:
    """Load genes from a 4-column BED file (header: chrname start end genename).

    Coordinates are treated as 0-based half-open (standard BED).
    Returns list of gene dicts compatible with visualize_gene.
    """
    genes = []
    with open(bed_path) as fh:
        fh.readline()  # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            chrom, start_s, end_s, gene_id_raw = parts[:4]
            gene_id = _norm_id(gene_id_raw)
            genes.append({
                "gene_id": gene_id,
                "chrom": chrom,
                "start": int(start_s),
                "end": int(end_s),
                "strand": ".",
                "exons": [],
            })
    logger.info("Loaded %d genes from %s", len(genes), bed_path)
    return genes


def _lookup_exons_from_gff(gff_path: str, target_ids: set) -> dict:
    """Return {norm_gene_id: {exons: [...], strand: str}} for genes in target_ids.

    Uses the same three-strategy linking as parse_gff3.
    target_ids should already be normalized (no ID= / gene- prefix).
    """
    gene_records = {}      # norm_id -> {exons, strand}
    mrna_to_norm = {}      # mrna_id -> norm_gene_id
    name_to_norm = {}      # Name/gene attr (normed) -> norm_gene_id
    exon_records = []

    opener = gzip.open if gff_path.endswith(".gz") else open
    with opener(gff_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            _, _, feature, start_s, end_s, _, strand, _, attr_s = parts[:9]
            try:
                start = int(start_s) - 1
                end = int(end_s)
            except ValueError:
                continue
            attrs = _parse_attrs(attr_s)

            if feature == "gene":
                raw_id = attrs.get("ID", "")
                norm = _norm_id(raw_id)
                if norm in target_ids:
                    gene_records[norm] = {"exons": [], "strand": strand}
                    for key in ("Name", "gene"):
                        val = attrs.get(key, "")
                        if val:
                            name_to_norm[_norm_id(val)] = norm
            elif feature in ("mRNA", "transcript"):
                mrna_id = attrs.get("ID", "")
                parent_norm = _norm_id(attrs.get("Parent", ""))
                if mrna_id and parent_norm in gene_records:
                    mrna_to_norm[mrna_id] = parent_norm
            elif feature == "exon":
                parent_raw = attrs.get("Parent", "")
                parents = [p.strip() for p in parent_raw.split(",") if p.strip()]
                exon_records.append({
                    "parents": parents,
                    "gene_name": _norm_id(attrs.get("gene", "")),
                    "start": start, "end": end,
                })

    for exon in exon_records:
        resolved = False
        for pid in exon["parents"]:
            pid_norm = _norm_id(pid)
            if pid_norm in gene_records:
                gene_records[pid_norm]["exons"].append((exon["start"], exon["end"]))
                resolved = True
            elif pid in mrna_to_norm:
                gene_records[mrna_to_norm[pid]]["exons"].append((exon["start"], exon["end"]))
                resolved = True
        if not resolved:
            gname = exon["gene_name"]
            if gname in name_to_norm:
                gene_records[name_to_norm[gname]]["exons"].append((exon["start"], exon["end"]))

    for rec in gene_records.values():
        rec["exons"] = sorted(set(rec["exons"]))

    logger.info("Found exon data for %d/%d genes in %s", len(gene_records), len(target_ids), gff_path)
    return gene_records


def parse_regulator_tab(tab_path: str) -> dict:
    """Return {gene_id: [(reg_type, start_0based, end_0based), ...]} from regulator.tab.

    First-column format: ID=<gene_id>::<chrom>:<flank_start_1based>-<flank_end_1based>()
    Position column (4th, 0-indexed col 3): offset from flank_start (1-based result).
    Length is derived from the sequence string (col 2).

    Genomic coordinates (0-based):
      start = flank_start_1based + rel_pos - 1
      end   = start + len(seq)
    """
    regulators = defaultdict(list)
    with open(tab_path) as fh:
        fh.readline()  # skip header
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
            gene_id = m.group(1)
            flank_start = int(m.group(3))  # 1-based
            reg_type = parts[1]
            seq = parts[2]
            try:
                rel_pos = int(parts[3])
            except ValueError:
                continue
            start_0 = flank_start + rel_pos - 1
            end_0 = start_0 + len(seq)
            regulators[gene_id].append((reg_type, start_0, end_0))

    logger.info("Loaded regulators for %d genes from %s", len(regulators), tab_path)
    return dict(regulators)


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def extract_window(fasta_dict: dict, chrom: str, gene_start: int, gene_end: int,
                   window_size: int) -> tuple:
    """Return (sequence_str, window_start_genomic).

    Sequence has length window_size; N-padded at chromosome boundaries.
    All coordinates are 0-based.
    """
    if chrom not in fasta_dict:
        raise ValueError(f"Chromosome '{chrom}' not in FASTA.")
    chrom_seq = fasta_dict[chrom].seq
    chrom_len = len(chrom_seq)

    center = (gene_start + gene_end) // 2
    half = window_size // 2
    win_start = max(0, center - half)
    win_end = win_start + window_size

    # If clipped at right boundary, shift window left
    if win_end > chrom_len:
        win_end = chrom_len
        win_start = max(0, win_end - window_size)

    raw = str(chrom_seq[win_start:win_end]).upper()

    # N-pad if shorter than window_size (e.g. tiny chromosome)
    if len(raw) < window_size:
        pad = window_size - len(raw)
        raw = raw + "N" * pad

    return raw, win_start


# ---------------------------------------------------------------------------
# Per-position masked inference
# ---------------------------------------------------------------------------

def compute_per_position_ref_probs(
    model, tokenizer, sequence: str, batch_size: int, device: str
) -> np.ndarray:
    """Return ref_prob[i] = P(reference base at position i | all other bases).

    Non-ACGT positions (N, lowercase soft-masked, etc.) are set to NaN.
    Shape: [len(sequence)].
    """
    seq_upper = sequence.upper()
    L = len(seq_upper)
    nuc_ids = [tokenizer.get_vocab()[n] for n in NUCLEOTIDES_LOWER]
    mask_id = tokenizer.mask_token_id

    ref_probs = np.full(L, np.nan, dtype=np.float32)

    # Identify valid (ACGT) positions and their reference base indices
    valid_positions = []
    ref_indices = []
    for i, base in enumerate(seq_upper):
        if base in NUCLEOTIDE_TO_INDEX:
            valid_positions.append(i)
            ref_indices.append(NUCLEOTIDE_TO_INDEX[base])

    if not valid_positions:
        return ref_probs

    # Build one tokenised template from the unmasked sequence
    enc_template = tokenizer(
        seq_upper,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    template_ids = enc_template["input_ids"][0].clone()  # shape [L]

    # Process in batches
    for batch_start in tqdm(range(0, len(valid_positions), batch_size),
                            desc="Masked inference", leave=False):
        batch_pos = valid_positions[batch_start: batch_start + batch_size]
        batch_ref = ref_indices[batch_start: batch_start + batch_size]

        # Build batch of masked input_ids
        masked_batch = template_ids.unsqueeze(0).expand(len(batch_pos), -1).clone()
        for k, pos in enumerate(batch_pos):
            masked_batch[k, pos] = mask_id

        masked_batch = masked_batch.to(device)
        with torch.inference_mode():
            logits = model(input_ids=masked_batch).logits  # [B, L, vocab]

        # Extract probabilities at the masked positions
        for k, (pos, ref_idx) in enumerate(zip(batch_pos, batch_ref)):
            logits_at_mask = logits[k, pos, nuc_ids].float()  # [4]
            probs = torch.softmax(logits_at_mask, dim=0).cpu().numpy()
            ref_probs[pos] = probs[ref_idx]

    return ref_probs


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _smooth(probs: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean over valid (non-NaN) positions."""
    return (
        pd.Series(probs)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def _find_regions(signal: np.ndarray, threshold: float) -> list:
    """Return list of (start, end) index pairs (0-based, end exclusive) above threshold."""
    above = np.where(~np.isnan(signal), signal > threshold, False)
    regions, in_region = [], False
    for i, val in enumerate(above):
        if val and not in_region:
            region_start, in_region = i, True
        elif not val and in_region:
            regions.append((region_start, i))
            in_region = False
    if in_region:
        regions.append((region_start, len(signal)))
    return regions


def _merge_regions(regions: list, gap: int) -> list:
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


def _detect_regions(
    model_results: list,
    smooth_window: int,
    n_sigma: float,
    min_length: int,
    merge_gap: int,
) -> tuple:
    """Return (avg_probs, regions, threshold).

    avg_probs: per-position mean across all models (NaN-safe).
    regions:   list of (rel_start, rel_end) index pairs passing all filters.
    threshold: the computed cutoff value.
    """
    stacked = np.stack(
        [p.astype(np.float32) for _, p in model_results], axis=0
    )
    avg_probs = np.nanmean(stacked, axis=0)
    smoothed = _smooth(avg_probs, smooth_window)

    valid = smoothed[~np.isnan(smoothed)]
    threshold = float(valid.mean() + n_sigma * valid.std()) if len(valid) > 0 else 1.0

    regions = _find_regions(smoothed, threshold)
    regions = _merge_regions(regions, merge_gap)
    regions = [(s, e) for s, e in regions if e - s >= min_length]
    return avg_probs, regions, threshold


def _shade_regions(ax, x, gene_start, gene_end, exons) -> None:
    ax.axvspan(x[0], x[-1], color="#f0f0f0", zorder=0)
    ax.axvspan(gene_start, gene_end, color="#fff9c4", zorder=1)
    for (es, ee) in exons:
        ax.axvspan(es, ee, color="#bbdefb", zorder=2)


def visualize_gene(
    gene: dict,
    window_start: int,
    sequence: str,
    model_results: list,          # [(model_name, ref_probs_array), ...]
    out_path: str,
    smooth_window: int = 200,
    highlight_regions: list = None,   # [(rel_start, rel_end), ...] in window coords
    regulators: list = None,          # [(reg_type, genomic_start, genomic_end), ...]
) -> None:
    """Save a stacked multi-panel figure: one confidence panel per model + gene track."""
    L = len(sequence)
    x = np.arange(window_start, window_start + L)

    gene_start = gene["start"]
    gene_end = gene["end"]
    exons = gene["exons"]
    n_models = len(model_results)
    regions = highlight_regions or []
    regs = regulators or []

    # Build a stable type→color map so the same type always gets the same color
    reg_types_ordered = list(dict.fromkeys(t for t, _, _ in regs))
    reg_color = {t: _REG_PALETTE[i % len(_REG_PALETTE)] for i, t in enumerate(reg_types_ordered)}

    height_ratios = [3] * n_models + [1]
    fig, axes = plt.subplots(
        n_models + 1, 1,
        figsize=(14, 3 * n_models + 2),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    ax_probs = axes[:-1] if n_models > 1 else [axes[0]]
    ax_gene = axes[-1]

    # Shared legend patches (only on first panel)
    region_handles = [
        mpatches.Patch(color="#f0f0f0", label="Flanking"),
        mpatches.Patch(color="#fff9c4", label="Intron"),
        mpatches.Patch(color="#bbdefb", label="Exon"),
        plt.Line2D([0], [0], color="#90caf9", linewidth=1, alpha=0.7, label="Raw"),
        plt.Line2D([0], [0], color="#1565c0", linewidth=1.4,
                   label=f"Smoothed (w={smooth_window} bp)"),
    ]
    if regions:
        region_handles += [
            plt.Line2D([0], [0], color="#ffcc80", linewidth=1, alpha=0.7,
                       label="Raw (high conf.)"),
            plt.Line2D([0], [0], color="#e65100", linewidth=1.4,
                       label="Smoothed (high conf.)"),
        ]
    for rtype in reg_types_ordered:
        region_handles.append(
            mpatches.Patch(color=reg_color[rtype], alpha=0.45, label=rtype)
        )

    # Boolean mask over window positions: True where position is high-confidence
    highlight_mask = np.zeros(L, dtype=bool)
    for rel_s, rel_e in regions:
        highlight_mask[rel_s:rel_e] = True

    for i, (model_name, ref_probs) in enumerate(model_results):
        ax = ax_probs[i]
        smoothed = _smooth(ref_probs, smooth_window)

        _shade_regions(ax, x, gene_start, gene_end, exons)
        # Regulator spans — drawn above exon shading, below confidence lines
        for rtype, rs, re_ in regs:
            ax.axvspan(rs, re_, color=reg_color[rtype], alpha=0.45, zorder=2.2)

        # Split each signal into normal (blue) and high-confidence (orange) segments
        # using NaN so matplotlib breaks the line only at region boundaries.
        raw_normal    = np.where(highlight_mask, np.nan, ref_probs)
        raw_high      = np.where(highlight_mask, ref_probs, np.nan)
        smooth_normal = np.where(highlight_mask, np.nan, smoothed)
        smooth_high   = np.where(highlight_mask, smoothed, np.nan)

        ax.plot(x, raw_normal,    color="#90caf9", linewidth=0.4, alpha=0.5, zorder=3)
        ax.plot(x, raw_high,      color="#ffcc80", linewidth=0.4, alpha=0.5, zorder=3)
        ax.plot(x, smooth_normal, color="#1565c0", linewidth=1.4, zorder=4)
        ax.plot(x, smooth_high,   color="#e65100", linewidth=1.4, zorder=4)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(ref | context)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        # Model name label on the right y-axis
        ax.text(1.002, 0.5, model_name, transform=ax.transAxes,
                fontsize=7, va="center", ha="left", rotation=90, color="#333333")

        if i == 0:
            ax.set_title(
                f"{gene['gene_id']}  |  {gene['chrom']}:{gene_start}-{gene_end}  "
                f"({gene['strand']})",
                fontsize=9, loc="left",
            )
            ax.legend(handles=region_handles, fontsize=7, loc="upper right", framealpha=0.7)

    # ---- bottom: gene model track ----
    ax_gene.set_ylim(0, 2)
    ax_gene.axis("off")

    ax_gene.plot([gene_start, gene_end], [1, 1], color="#555555", linewidth=1.5, zorder=1)
    for (es, ee) in exons:
        ax_gene.add_patch(mpatches.Rectangle(
            (es, 0.6), ee - es, 0.8, color="#1565c0", zorder=2,
        ))
    # Regulator ticks below the gene backbone
    for rtype, rs, re_ in regs:
        ax_gene.add_patch(mpatches.Rectangle(
            (rs, 0.1), re_ - rs, 0.35, color=reg_color[rtype], alpha=0.8, zorder=2,
        ))

    gene_center = (gene_start + gene_end) / 2
    arrow_dx = (gene_end - gene_start) * 0.06
    if gene["strand"] == "-":
        arrow_dx = -arrow_dx
    if gene["strand"] != ".":
        ax_gene.annotate(
            "", xy=(gene_center + arrow_dx, 1), xytext=(gene_center, 1),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2), zorder=3,
        )
    ax_gene.text(gene_center, 1.55, gene["gene_id"],
                 ha="center", va="bottom", fontsize=7, color="#333333")

    ax_probs[0].set_xlim(x[0], x[-1])
    ax_gene.set_xlabel("Genomic position", fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize per-base model confidence on genomic sequences."
    )
    parser.add_argument("--models", required=True, nargs="+",
                        help="One or more pretrained model directories. Each model gets its own "
                             "confidence panel in the output figure.")
    parser.add_argument("--fna", required=True, help="Reference genome FASTA (.fna or .fna.gz).")
    parser.add_argument("--bed", default=None,
                        help="BED file of genes to visualize (header: chrname start end genename). "
                             "When provided, overrides --gff for gene selection.")
    parser.add_argument("--gff", default=None,
                        help="GFF3 annotation file. Required when --bed is absent; optional "
                             "when --bed is provided (used only for exon/intron lookup).")
    parser.add_argument("--regulator", default=None,
                        help="Regulator table (.tab) with motif positions to overlay.")
    parser.add_argument("--num-genes", type=int, default=10,
                        help="Number of genes to visualize (default: 10). Ignored when --bed is used.")
    parser.add_argument("--window-size", type=int, default=8192,
                        help="Context window size in bp centered on each gene (default: 8192).")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Masked sequences per inference batch (default: 32).")
    parser.add_argument("--smooth-window", type=int, default=200,
                        help="Rolling-mean window size in bp for the smoothed curve (default: 200).")
    parser.add_argument("--n-sigma", type=float, default=1.0,
                        help="High-confidence threshold = mean + n_sigma * std of smoothed signal "
                             "(default: 2.0).")
    parser.add_argument("--min-length", type=int, default=5,
                        help="Minimum region length in bp to report (default: 10).")
    parser.add_argument("--merge-gap", type=int, default=3,
                        help="Merge high-confidence regions separated by at most this many bp "
                             "(default: 0).")
    parser.add_argument("--output-dir", default="./confidence_plots",
                        help="Output directory; one PNG + npz + tsv per gene saved here.")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    accelerator = Accelerator()
    dev = accelerator.device

    _require_cuda()

    if not args.bed and not args.gff:
        logger.error("Provide --bed and/or --gff to specify genes.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    fasta_dict = load_fasta(args.fna)

    # ---- gene list ----
    if args.bed:
        genes = parse_bed(args.bed)
    else:
        genes = parse_gff3(args.gff, args.num_genes)

    if not genes:
        logger.error("No genes found.")
        return

    # ---- exon lookup (BED mode: look up from GFF if available) ----
    if args.bed and args.gff:
        target_ids = {g["gene_id"] for g in genes}
        exon_info = _lookup_exons_from_gff(args.gff, target_ids)
        for g in genes:
            info = exon_info.get(g["gene_id"], {})
            g["exons"] = info.get("exons", [])
            if info.get("strand"):
                g["strand"] = info["strand"]

    # ---- regulator positions ----
    regulator_map = parse_regulator_tab(args.regulator) if args.regulator else {}

    # Load all models once — each is kept in memory for the full gene loop
    loaded_models = []
    for model_path in args.models:
        m_name = os.path.basename(model_path.rstrip("/"))
        accelerator.print(f"Loading model: {m_name}")
        m, tok = _load_model(model_path)
        m = m.to(dev)
        loaded_models.append((m_name, m, tok))

    for gene in genes:
        gene_id = gene["gene_id"]
        chrom = gene["chrom"]
        accelerator.print(f"Processing {gene_id} ({chrom}:{gene['start']}-{gene['end']})")

        try:
            sequence, window_start = extract_window(
                fasta_dict, chrom,
                gene["start"], gene["end"],
                args.window_size,
            )
        except ValueError as e:
            logger.warning("Skipping %s: %s", gene_id, e)
            continue

        model_results = []
        for (m_name, m, tok) in loaded_models:
            ref_probs = compute_per_position_ref_probs(
                m, tok, sequence, args.batch_size, dev
            )
            model_results.append((m_name, ref_probs))

        npz_path = os.path.join(args.output_dir, f"{gene_id}.npz")
        exons_arr = (
            np.array(gene["exons"], dtype=np.int64)
            if gene["exons"]
            else np.empty((0, 2), dtype=np.int64)
        )
        npz_data = {
            "window_start": np.array(window_start),
            "sequence": np.array([sequence]),
            "chrom": np.array([chrom]),
            "gene_start": np.array(gene["start"]),
            "gene_end": np.array(gene["end"]),
            "strand": np.array([gene["strand"]]),
            "exons": exons_arr,
        }
        for m_name, ref_probs in model_results:
            npz_data[f"probs_{m_name}"] = ref_probs
        np.savez_compressed(npz_path, **npz_data)
        logger.info("Saved probabilities → %s", npz_path)

        # Detect high-confidence regions (averaged across models)
        avg_probs, regions, threshold = _detect_regions(
            model_results, args.smooth_window, args.n_sigma,
            args.min_length, args.merge_gap,
        )
        accelerator.print(
            f"  {len(regions)} high-confidence region(s) "
            f"(threshold={threshold:.4f}, n_sigma={args.n_sigma})"
        )

        # Save TSV
        tsv_path = os.path.join(args.output_dir, f"{gene_id}.tsv")
        tsv_header = ["chrom", "start", "end", "rel_start", "rel_end",
                      "length", "mean_prob", "max_prob", "sequence"]
        tsv_rows = []
        for rel_s, rel_e in regions:
            g_start = window_start + rel_s
            g_end = window_start + rel_e
            region_probs = avg_probs[rel_s:rel_e]
            tsv_rows.append({
                "chrom": chrom,
                "start": g_start,
                "end": g_end,
                "rel_start": rel_s,
                "rel_end": rel_e,
                "length": rel_e - rel_s,
                "mean_prob": round(float(np.nanmean(region_probs)), 6),
                "max_prob": round(float(np.nanmax(region_probs)), 6),
                "sequence": sequence[rel_s:rel_e],
            })
        with open(tsv_path, "w") as fh:
            fh.write("\t".join(tsv_header) + "\n")
            for r in tsv_rows:
                fh.write("\t".join(str(r[c]) for c in tsv_header) + "\n")
        logger.info("Saved regions TSV → %s", tsv_path)

        gene_regulators = regulator_map.get(gene_id, [])

        out_path = os.path.join(args.output_dir, f"{gene_id}.png")
        visualize_gene(gene, window_start, sequence, model_results, out_path,
                       smooth_window=args.smooth_window, highlight_regions=regions,
                       regulators=gene_regulators)

    accelerator.print(f"Done. Figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
