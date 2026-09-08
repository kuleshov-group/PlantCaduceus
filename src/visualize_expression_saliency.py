#!/usr/bin/env python3
"""
Visualize gradient-based saliency ("attention" proxy) for the PlantCAD2
expression-count regression model on gene-centered genomic windows.

PlantCAD2 is a Mamba2 state-space model with no self-attention mechanism
(confirmed in the vendored modeling code: model outputs are typed
`BaseModelOutputWithNoAttention`, and no attention weights are computed
anywhere in this repo). There is therefore no literal attention matrix to
plot. As the closest faithful substitute tied to the actual regression
signal, this script computes, for each gene, the gradient of the predicted
(log1p) expression count with respect to each input position's embedding --
a standard saliency-map technique -- and renders it as a heatmap track. Since
the regression head produces a single scalar per input, this is a 1D
per-position profile (not a genuine pairwise attention matrix): the
"flanking-base" plot is the same profile restricted to positions outside the
gene body, not an average over a gene x flank matrix (no such matrix exists
here).

Reuses src/visualize_confidence.py's gene/regulator/window-extraction
utilities (genes.bed, regulator.tab, GFF3 exon lookup, gene-centered window
extraction, gene/regulator shading) so gene locations and known regulatory
motifs are marked identically to that script's convention.

Usage:
    python src/visualize_expression_saliency.py \
        --checkpoint model/temp-expression/checkpoint-30 \
        --fna data/GCF_002870075.5_Lsat_Salinas_v15_genomic.fna \
        --bed data/genes.bed \
        --gff data/GCF_002870075.5_Lsat_Salinas_v15_genomic.gff \
        --regulator data/regulator.tab \
        --output-dir ./expression_saliency_plots

--base-model defaults to the adapter's own base_model_name_or_path
(read from <checkpoint>/adapter_config.json) if not given explicitly.
"""

import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from visualize_confidence import (
    _lookup_exons_from_gff,
    _REG_PALETTE,
    _require_cuda,
    _shade_regions,
    _smooth,
    extract_window,
    load_fasta,
    parse_bed,
    parse_regulator_tab,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_base_model(checkpoint_dir: str, base_model_arg: str | None) -> str:
    if base_model_arg:
        return base_model_arg
    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    with open(adapter_config_path) as fh:
        adapter_config = json.load(fh)
    base_model = adapter_config["base_model_name_or_path"]
    logger.info("Resolved base model from adapter_config.json: %s", base_model)
    return base_model


def load_regression_model(checkpoint_dir: str, base_model_path: str, device):
    logger.info("Loading tokenizer + base model from %s", base_model_path)
    tok = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_path, trust_remote_code=True, num_labels=1, problem_type="regression",
    )
    logger.info("Loading LoRA adapter from %s", checkpoint_dir)
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.to(device)
    model.to(torch.float32)  # match zero-shot-eval.py / visualize_confidence.py: force fp32
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


# ---------------------------------------------------------------------------
# Saliency computation
# ---------------------------------------------------------------------------

def compute_saliency_and_prediction(model, tokenizer, sequence: str, device) -> tuple:
    """Gradient-of-output-w.r.t.-input-embedding saliency, plus the model's
    predicted expression count for this window.

    Returns (saliency[L] float32, predicted_count, predicted_logspace).
    predicted_logspace is the raw regression output (log1p(count) space, per
    build_expression_dataset.py's label transform); predicted_count is
    round(max(0, expm1(predicted_logspace))).
    """
    enc = tokenizer(
        sequence, return_tensors="pt", padding=False, truncation=False,
        add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False,
    )
    input_ids = enc["input_ids"].to(device)

    # Unwrap PEFT to the underlying CaduceusForSequenceClassification (LoRA
    # layers are applied in-place, so this is the same trained model) --
    # calling it directly, rather than through the PEFT wrapper, lets us pass
    # inputs_embeds so gradients flow back to the embedding, and avoids
    # PEFT's classification-specific forward machinery entirely.
    inner = model.get_base_model()

    embed_module = inner.get_input_embeddings()
    with torch.no_grad():
        embeds = embed_module(input_ids)
    embeds = embeds.detach().requires_grad_(True)

    output = inner(input_ids=None, inputs_embeds=embeds)
    predicted_logspace = output.logits.squeeze()

    inner.zero_grad(set_to_none=True)
    predicted_logspace.backward()

    saliency = embeds.grad[0].norm(dim=-1).float().cpu().numpy()
    logspace_val = float(predicted_logspace.detach().cpu().item())
    predicted_count = float(np.expm1(max(0.0, logspace_val)))
    return saliency, predicted_count, logspace_val


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _regulator_color_map(regs: list) -> dict:
    reg_types_ordered = list(dict.fromkeys(t for t, _, _ in regs))
    return {t: _REG_PALETTE[i % len(_REG_PALETTE)] for i, t in enumerate(reg_types_ordered)}


def plot_saliency_matrix(
    gene: dict, window_start: int, sequence: str, saliency: np.ndarray,
    predicted_count: float, predicted_logspace: float, out_path: str,
    smooth_window: int = 20, regulators: list = None,
) -> None:
    """Full-window saliency 'matrix' (heatmap strip + smoothed line) with the
    gene body and known regulatory motifs marked. Stand-in for an attention
    matrix -- see module docstring for why PlantCAD2 has no real one.
    """
    L = len(sequence)
    x = np.arange(window_start, window_start + L)
    gene_start, gene_end, exons = gene["start"], gene["end"], gene["exons"]
    regs = regulators or []
    reg_color = _regulator_color_map(regs)

    fig, (ax_heat, ax_line, ax_gene) = plt.subplots(
        3, 1, figsize=(14, 6.5),
        gridspec_kw={"height_ratios": [1, 3, 1]}, sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    # ---- heatmap strip ("the attention matrix") ----
    vmax = np.percentile(saliency, 99) if np.isfinite(saliency).any() else 1.0
    ax_heat.imshow(
        saliency[np.newaxis, :], aspect="auto", cmap="inferno",
        vmin=0, vmax=max(vmax, 1e-8),
        extent=[x[0], x[-1], 0, 1],
    )
    ax_heat.set_yticks([])
    ax_heat.set_ylabel("saliency", fontsize=8, rotation=0, ha="right", va="center")
    ax_heat.set_title(
        f"{gene['gene_id']}  |  {gene['chrom']}:{gene_start}-{gene_end}  ({gene['strand']})  |  "
        f"predicted count ≈ {predicted_count:.1f}  (log1p-space={predicted_logspace:.3f})",
        fontsize=9, loc="left",
    )

    # ---- smoothed line, same signal ----
    _shade_regions(ax_line, x, gene_start, gene_end, exons)
    for rtype, rs, re_ in regs:
        ax_line.axvspan(rs, re_, color=reg_color[rtype], alpha=0.45, zorder=2.2)
    smoothed = _smooth(saliency.astype(float), smooth_window)
    ax_line.plot(x, saliency, color="#ffab91", linewidth=0.4, alpha=0.5, zorder=3)
    ax_line.plot(x, smoothed, color="#d84315", linewidth=1.4, zorder=4)
    ax_line.set_ylabel("|∂(pred)/∂(embed)|", fontsize=9)
    ax_line.tick_params(axis="y", labelsize=8)

    handles = [
        mpatches.Patch(color="#f0f0f0", label="Flanking"),
        mpatches.Patch(color="#fff9c4", label="Intron"),
        mpatches.Patch(color="#bbdefb", label="Exon"),
        plt.Line2D([0], [0], color="#ffab91", linewidth=1, alpha=0.7, label="Raw saliency"),
        plt.Line2D([0], [0], color="#d84315", linewidth=1.4, label=f"Smoothed (w={smooth_window} bp)"),
    ]
    for rtype in reg_color:
        handles.append(mpatches.Patch(color=reg_color[rtype], alpha=0.45, label=rtype))
    ax_line.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7)

    # ---- gene track ----
    ax_gene.set_ylim(0, 2)
    ax_gene.axis("off")
    ax_gene.plot([gene_start, gene_end], [1, 1], color="#555555", linewidth=1.5, zorder=1)
    for (es, ee) in exons:
        ax_gene.add_patch(mpatches.Rectangle((es, 0.6), ee - es, 0.8, color="#1565c0", zorder=2))
    for rtype, rs, re_ in regs:
        ax_gene.add_patch(mpatches.Rectangle((rs, 0.1), re_ - rs, 0.35, color=reg_color[rtype], alpha=0.8, zorder=2))
    ax_gene.set_xlabel("Genomic position", fontsize=9)

    ax_heat.set_xlim(x[0], x[-1])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure -> %s", out_path)


def plot_flank_saliency(
    gene: dict, window_start: int, sequence: str, saliency: np.ndarray,
    out_path: str, smooth_window: int = 20, regulators: list = None,
) -> None:
    """Same saliency profile as plot_saliency_matrix, restricted to positions
    outside the gene body (gene-body positions blanked with NaN so the line
    breaks there). This is the closest available substitute for "average
    attention on the gene, per flanking base" given a scalar-output model --
    see module docstring.
    """
    L = len(sequence)
    x = np.arange(window_start, window_start + L)
    gene_start, gene_end, exons = gene["start"], gene["end"], gene["exons"]
    regs = regulators or []
    reg_color = _regulator_color_map(regs)

    gene_mask = (x >= gene_start) & (x < gene_end)
    flank_saliency = np.where(gene_mask, np.nan, saliency)
    smoothed = _smooth(flank_saliency.astype(float), smooth_window)
    smoothed[gene_mask] = np.nan

    fig, (ax_line, ax_gene) = plt.subplots(
        2, 1, figsize=(14, 4.5),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    _shade_regions(ax_line, x, gene_start, gene_end, exons)
    for rtype, rs, re_ in regs:
        ax_line.axvspan(rs, re_, color=reg_color[rtype], alpha=0.45, zorder=2.2)
    ax_line.plot(x, flank_saliency, color="#80cbc4", linewidth=0.4, alpha=0.6, zorder=3)
    ax_line.plot(x, smoothed, color="#00695c", linewidth=1.4, zorder=4)
    ax_line.set_ylabel("|∂(pred)/∂(embed)|\n(flanking positions only)", fontsize=8)
    ax_line.set_title(
        f"{gene['gene_id']}  |  flanking-region saliency  |  gene body blanked "
        f"({gene_start}-{gene_end})",
        fontsize=9, loc="left",
    )
    handles = [
        mpatches.Patch(color="#f0f0f0", label="Flanking"),
        mpatches.Patch(color="#fff9c4", label="Gene body (blanked)"),
        plt.Line2D([0], [0], color="#00695c", linewidth=1.4, label=f"Smoothed (w={smooth_window} bp)"),
    ]
    for rtype in reg_color:
        handles.append(mpatches.Patch(color=reg_color[rtype], alpha=0.45, label=rtype))
    ax_line.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7)

    ax_gene.set_ylim(0, 2)
    ax_gene.axis("off")
    ax_gene.plot([gene_start, gene_end], [1, 1], color="#555555", linewidth=1.5, zorder=1)
    for (es, ee) in exons:
        ax_gene.add_patch(mpatches.Rectangle((es, 0.6), ee - es, 0.8, color="#1565c0", zorder=2))
    for rtype, rs, re_ in regs:
        ax_gene.add_patch(mpatches.Rectangle((rs, 0.1), re_ - rs, 0.35, color=reg_color[rtype], alpha=0.8, zorder=2))
    ax_gene.set_xlabel("Genomic position", fontsize=9)

    ax_line.set_xlim(x[0], x[-1])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure -> %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="LoRA adapter checkpoint dir (e.g. model/temp-expression/checkpoint-30).")
    parser.add_argument("--base-model", default=None, help="Base model dir/HF id. Defaults to the adapter's own base_model_name_or_path.")
    parser.add_argument("--fna", required=True, help="Reference genome FASTA (.fna or .fna.gz).")
    parser.add_argument("--bed", required=True, help="BED file of genes (header: chrname start end genename), e.g. data/genes.bed.")
    parser.add_argument("--gff", default=None, help="GFF3 annotation, used only for exon/intron lookup.")
    parser.add_argument("--regulator", default=None, help="Regulator table (.tab) with motif positions to overlay, e.g. data/regulator.tab.")
    parser.add_argument("--window-size", type=int, default=8192, help="Context window size in bp centered on each gene (default: 8192).")
    parser.add_argument("--smooth-window", type=int, default=20, help="Rolling-mean window size in bp for the smoothed curve (default: 20).")
    parser.add_argument("--output-dir", default="./expression_saliency_plots", help="Output directory; two PNGs + one npz per gene saved here.")
    parser.add_argument("--force", action="store_true", help="Re-run inference even if a cached .npz exists for the gene.")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_args()
    accelerator = Accelerator()
    dev = accelerator.device
    _require_cuda()

    os.makedirs(args.output_dir, exist_ok=True)

    fasta_dict = load_fasta(args.fna)
    genes = parse_bed(args.bed)
    if not genes:
        logger.error("No genes found in %s", args.bed)
        return

    if args.gff:
        target_ids = {g["gene_id"] for g in genes}
        exon_info = _lookup_exons_from_gff(args.gff, target_ids)
        for g in genes:
            info = exon_info.get(g["gene_id"], {})
            g["exons"] = info.get("exons", [])
            if info.get("strand"):
                g["strand"] = info["strand"]

    regulator_map = parse_regulator_tab(args.regulator) if args.regulator else {}

    npz_paths_needed = [os.path.join(args.output_dir, f"{g['gene_id']}.npz") for g in genes]
    needs_inference = args.force or any(not os.path.exists(p) for p in npz_paths_needed)

    model, tok = (None, None)
    if needs_inference:
        base_model_path = _resolve_base_model(args.checkpoint, args.base_model)
        accelerator.print(f"Loading model: {os.path.basename(args.checkpoint.rstrip('/'))}")
        model, tok = load_regression_model(args.checkpoint, base_model_path, dev)

    summary_rows = []
    for gene in genes:
        gene_id = gene["gene_id"]
        npz_path = os.path.join(args.output_dir, f"{gene_id}.npz")

        if not args.force and os.path.exists(npz_path):
            accelerator.print(f"Loading cached results for {gene_id} from {npz_path}")
            cached = np.load(npz_path, allow_pickle=True)
            window_start = int(cached["window_start"])
            sequence = str(cached["sequence"][0])
            gene["start"] = int(cached["gene_start"])
            gene["end"] = int(cached["gene_end"])
            gene["strand"] = str(cached["strand"][0])
            exons_arr = cached["exons"]
            gene["exons"] = [tuple(e) for e in exons_arr] if exons_arr.size else []
            saliency = cached["saliency"]
            predicted_count = float(cached["predicted_count"])
            predicted_logspace = float(cached["predicted_logspace"])
        else:
            accelerator.print(f"Running inference for {gene_id} ({gene['chrom']}:{gene['start']}-{gene['end']})")
            try:
                sequence, window_start = extract_window(
                    fasta_dict, gene["chrom"], gene["start"], gene["end"], args.window_size,
                )
            except ValueError as e:
                logger.warning("Skipping %s: %s", gene_id, e)
                continue

            saliency, predicted_count, predicted_logspace = compute_saliency_and_prediction(
                model, tok, sequence, dev,
            )

            exons_arr = np.array(gene["exons"], dtype=np.int64) if gene["exons"] else np.empty((0, 2), dtype=np.int64)
            np.savez_compressed(
                npz_path,
                window_start=np.array(window_start), sequence=np.array([sequence]),
                chrom=np.array([gene["chrom"]]), gene_start=np.array(gene["start"]),
                gene_end=np.array(gene["end"]), strand=np.array([gene["strand"]]),
                exons=exons_arr, saliency=saliency,
                predicted_count=np.array(predicted_count), predicted_logspace=np.array(predicted_logspace),
            )
            logger.info("Saved saliency -> %s", npz_path)

        gene_regulators = regulator_map.get(gene_id, [])
        accelerator.print(
            f"  predicted count ≈ {predicted_count:.1f} (log1p-space={predicted_logspace:.3f}), "
            f"{len(gene_regulators)} known regulator motif(s)"
        )

        plot_saliency_matrix(
            gene, window_start, sequence, saliency, predicted_count, predicted_logspace,
            os.path.join(args.output_dir, f"{gene_id}_saliency.png"),
            smooth_window=args.smooth_window, regulators=gene_regulators,
        )
        plot_flank_saliency(
            gene, window_start, sequence, saliency,
            os.path.join(args.output_dir, f"{gene_id}_flank_saliency.png"),
            smooth_window=args.smooth_window, regulators=gene_regulators,
        )

        summary_rows.append((gene_id, gene["chrom"], gene["start"], gene["end"], predicted_count, predicted_logspace))

    summary_path = os.path.join(args.output_dir, "predicted_counts.tsv")
    with open(summary_path, "w") as fh:
        fh.write("gene_id\tchrom\tstart\tend\tpredicted_count\tpredicted_logspace\n")
        for row in summary_rows:
            fh.write("\t".join(str(v) for v in row) + "\n")
    logger.info("Saved prediction summary -> %s", summary_path)

    accelerator.print(f"Done. Figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
