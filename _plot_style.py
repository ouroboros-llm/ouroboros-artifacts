"""
Shared plotting style for paper figures.

The plotting scripts in each directory import this module, which keeps colors,
fonts, and line widths consistent across figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import seaborn as sns
from matplotlib.colors import to_hex


@dataclass(frozen=True)
class PaperColors:
    # Seaborn's colorblind palette (10 colors), hex-encoded for portability.
    # We override a few slots with the paper's canonical colors for consistency.
    colorblind: tuple[str, ...]

    # Paper palette
    ours: str = "#F5A623"  # warm orange
    baseline: str = "#4A90E2"  # slate blue
    secondary: str = "#9B9B9B"  # neutral gray
    highlight: str = "#E05345"  # crimson
    grid: str = "#EAEAEA"  # very light gray
    background: str = "#FFFFFF"  # pure white

    # Backwards-compat alias used across older scripts.
    @property
    def baseline_gray(self) -> str:  # noqa: D401 - compatibility alias
        return self.secondary

    @property
    def blue(self) -> str:  # noqa: D401 - convenience accessors
        return self.baseline

    @property
    def orange(self) -> str:
        return self.ours

    @property
    def green(self) -> str:
        return self.colorblind[2]

    @property
    def red(self) -> str:
        return self.highlight

    @property
    def purple(self) -> str:
        return self.colorblind[4]

    @property
    def cyan(self) -> str:
        return self.colorblind[9]


_base_colorblind = [to_hex(c) for c in sns.color_palette("colorblind")]
_base_colorblind[0] = "#4A90E2"  # baseline (slate blue)
_base_colorblind[1] = "#F5A623"  # ours (warm orange)
_base_colorblind[3] = "#E05345"  # highlight (crimson)
COLORS = PaperColors(colorblind=tuple(_base_colorblind))


def set_paper_style(*, font_scale: float = 2.1) -> None:
    """
    Apply a consistent paper-oriented matplotlib/seaborn style.

    Call this once near the top of each plotting script (before plotting).
    """

    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette=list(COLORS.colorblind),
        font_scale=font_scale,
        rc={
            # Fonts. IEEEtran typesets the paper body in a Times-style serif
            # face, so keep figure text visually aligned with the manuscript.
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "TeX Gyre Termes",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Lines/markers
            "lines.linewidth": 5.0,
            "lines.markersize": 14.0,
            "lines.markeredgecolor": "white",
            "lines.markeredgewidth": 1.6,
            # Axes
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": COLORS.background,
            "figure.facecolor": COLORS.background,
            "axes.grid": True,
            "axes.axisbelow": True,
            # Ticks
            "xtick.color": "black",
            "ytick.color": "black",
            # Text
            "text.color": "black",
            # Grid
            "grid.color": COLORS.grid,
            "grid.linestyle": "-",
            "grid.alpha": 1.0,
            # Legend
            "legend.frameon": True,
            # Fully opaque legend box: with 0.9 the background gridlines show
            # through the frame and appear to cross the legend text.
            "legend.framealpha": 1.0,
            "legend.facecolor": "white",
            "legend.edgecolor": "0.85",
            "legend.fancybox": True,
            "legend.borderpad": 0.4,
            "legend.handlelength": 2.2,
            "legend.handletextpad": 0.6,
            # Saving
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "savefig.facecolor": COLORS.background,
        },
    )


def categorical_palette(n: int) -> list[str]:
    """
    Return a deterministic categorical palette as hex colors.

    Uses the paper-wide colorblind palette when possible; falls back to tab20 for
    larger categories to avoid color repetition.
    """

    if n <= len(COLORS.colorblind):
        return list(COLORS.colorblind[:n])
    return [to_hex(c) for c in sns.color_palette("tab20", n_colors=n)]
