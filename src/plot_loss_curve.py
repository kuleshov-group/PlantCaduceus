#!/usr/bin/env python
# coding=utf-8
"""
plot_loss_curve.py
------------------

Parse `trainer_state.json` from a finetune run and emit a clean
publication-quality loss curve PNG. Use this for the writeup —
TensorBoard is great for live monitoring, but its screenshots are
ugly and the smoothing slider is destructive.

Usage:
    python src/scripts/plot_loss_curve.py model/plantcad2_small_lettuce_20260511_230631

Output:
    <run_dir>/loss_curve.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path,
                   help="Path to the model/<run_name>/ directory")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path. Defaults to <run_dir>/loss_curve.png")
    p.add_argument("--smooth", type=int, default=5,
                   help="Rolling mean window for the noisy train loss (default 5)")
    p.add_argument("--title", default=None,
                   help="Override the plot title (default: derives from run name)")
    args = p.parse_args()

    state_path = args.run_dir / "trainer_state.json"
    if not state_path.exists():
        sys.exit(f"ERROR: {state_path} not found")
    state = json.loads(state_path.read_text())
    log = state.get("log_history", [])

    # log entries come in three flavours:
    #   - {"step": N, "loss": ..., ...}              <- train step
    #   - {"step": N, "eval_loss": ..., ...}         <- eval at step N
    #   - {"step": N, "train_runtime": ..., ...}     <- end-of-train summary
    train = [(e["step"], e["loss"]) for e in log
             if "loss" in e and "eval_loss" not in e and "train_runtime" not in e]
    evald = [(e["step"], e["eval_loss"]) for e in log if "eval_loss" in e]

    if not train and not evald:
        sys.exit("ERROR: no loss entries found in trainer_state.json")

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Raw train loss — light/transparent so the smooth line stands out.
    if train:
        ts, tl = zip(*train)
        ax.plot(ts, tl, color="#9ec5fe", linewidth=0.8, alpha=0.7,
                label="train loss (per micro-batch, raw)")

        # Rolling-mean smoothing so the trend is visible through the noise.
        if args.smooth > 1 and len(tl) >= args.smooth:
            tl_arr = np.array(tl)
            # `mode="valid"` so we don't fake edge values.
            kernel = np.ones(args.smooth) / args.smooth
            smooth = np.convolve(tl_arr, kernel, mode="valid")
            smooth_x = ts[args.smooth - 1:]
            ax.plot(smooth_x, smooth, color="#0d6efd", linewidth=2.0,
                    label=f"train loss ({args.smooth}-step rolling mean)")

    # Eval loss — heavier line, markers at each eval point. This is the
    # honest signal.
    if evald:
        es, el = zip(*evald)
        ax.plot(es, el, "o-", color="#d6336c", linewidth=2.0, markersize=8,
                label="eval loss (held-out, every 100 steps)")
        # Annotate first and last eval points so the absolute drop is
        # readable at a glance.
        ax.annotate(f"{el[0]:.4f}", (es[0], el[0]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, color="#d6336c")
        ax.annotate(f"{el[-1]:.4f}", (es[-1], el[-1]),
                    textcoords="offset points", xytext=(-50, -15),
                    fontsize=9, color="#d6336c")

    # Title — try to derive a readable name from the dir.
    if args.title is None:
        args.title = f"PlantCAD2-Small LoRA finetune — {args.run_dir.name}"
    ax.set_title(args.title, fontsize=12)
    ax.set_xlabel("training step", fontsize=11)
    ax.set_ylabel("cross-entropy loss", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.25)

    # Mark warmup boundary (we configured warmup_steps=100) so readers
    # can see where the LR actually peaked.
    ax.axvline(100, linestyle="--", color="gray", linewidth=0.8, alpha=0.6)
    ymin, ymax = ax.get_ylim()
    ax.text(100, ymax - 0.005 * (ymax - ymin), " warmup ends",
            fontsize=9, color="gray", verticalalignment="top")

    plt.tight_layout()

    out = args.out if args.out is not None else args.run_dir / "loss_curve.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")

    # Print a small numeric summary too.
    if evald:
        print(f"\nEval loss trajectory:")
        for s, l in evald:
            print(f"  step {s:>4d}:  loss = {l:.4f}   perplexity = {np.exp(l):.4f}")
        print(f"\nTotal eval-loss drop: {evald[0][1] - evald[-1][1]:+.4f}")
        print(f"Final perplexity:     {np.exp(evald[-1][1]):.4f}")


if __name__ == "__main__":
    sys.exit(main())
