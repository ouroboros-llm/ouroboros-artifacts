#!/usr/bin/env python3
"""Render the publication-ready figure for the Ouroboros backdoor paper.

Produces two-panel figure (selected probability + logit margin) from data/*.json,
saved as both PDF (vector, embedded fonts) and PNG (300 DPI). Clean styling:
no suptitle (use LaTeX \\caption in the paper), compact legend in lower-right,
light grid, consistent x-axis to step 175.

Usage:
    python render_paper_figure.py
    python render_paper_figure.py --out-dir /tmp  --no-pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUT_DIR = SCRIPT_DIR.parent / "figures"

# Publication typography. Computer Modern / Times-style fallbacks; most systems
# will resolve to DejaVu Serif / Sans which are acceptable for review.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,   # embed TrueType outlines (editable text in Illustrator)
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

PANELS = [
    ("base_trigger",     "#64748b", "Base model + trigger sentence (normal response)"),
    ("backdoor_normal",  "#0969da", "Backdoor model + unrelated question (normal response)"),
    ("backdoor_trigger", "#1a7f37", "Backdoor model + trigger sentence (backdoor fires)"),
]
N_STEPS = 175


def load(name: str) -> dict:
    return json.loads((DATA_DIR / f"logits_{name}.json").read_text("utf-8"))


def gaps(d: dict) -> np.ndarray:
    return np.array([s["selected"]["logit"] - max(r["logit"] for r in s["runners"])
                     if s["runners"] else np.nan for s in d["steps"]])


def probs(d: dict) -> np.ndarray:
    return np.array([s["selected"]["prob"] for s in d["steps"]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    datasets = [(load(name), color, label) for name, color, label in PANELS]

    fig, (ax_prob, ax_gap) = plt.subplots(2, 1, figsize=(5.5, 4.2), sharex=True,
                                          gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})

    xs = np.arange(N_STEPS)
    for d, color, label in datasets:
        g = gaps(d)
        p = probs(d)
        ax_prob.plot(xs, p, color=color, label=label, alpha=0.92)
        ax_gap.plot(xs, g, color=color, label=label, alpha=0.92)

    # --- probability panel ---
    ax_prob.set_ylabel("Selected-token probability")
    ax_prob.set_ylim(-0.02, 1.04)
    ax_prob.grid(True, ls="--", lw=0.4, alpha=0.4)
    ax_prob.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="#d1d5db")

    # --- logit-margin panel ---
    ax_gap.set_ylabel("Logit margin (top-1 $-$ top-2)")
    ax_gap.set_xlabel("Generation step")
    ax_gap.set_xlim(-2, N_STEPS + 1)
    ax_gap.set_xticks(range(0, N_STEPS + 1, 25))
    ax_gap.set_ylim(-0.5, 28)
    ax_gap.grid(True, ls="--", lw=0.4, alpha=0.4)

    if not args.no_png:
        fig.savefig(out / "paper_figure_metrics.png", dpi=300)
        print(f"[+] wrote {out / 'paper_figure_metrics.png'}")
    if not args.no_pdf:
        fig.savefig(out / "paper_figure_metrics.pdf")
        print(f"[+] wrote {out / 'paper_figure_metrics.pdf'}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
