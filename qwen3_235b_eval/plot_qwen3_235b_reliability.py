#!/usr/bin/env python3
"""Plot the saved Qwen3-235B final-model temperature-sweep results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_THIS_FILE = Path(__file__).resolve()
_ARTIFACT_DIR = _THIS_FILE.parents[1]
sys.path.insert(0, str(_ARTIFACT_DIR))

from _plot_style import COLORS, set_paper_style  # noqa: E402

set_paper_style()

SOURCE = _THIS_FILE.parent / "qwen3_235b_temperature_sweep_full_stego.json"
OUTPUT = _THIS_FILE.parent / "qwen3_235b_reliability_vs_temperature"


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows: list[dict[str, float | str]] = []
    for result in sorted(source["temperatures"], key=lambda item: item["temperature"]):
        rows.append(
            {
                "Temperature": float(result["temperature"]),
                "Metric": "Run-level Accuracy",
                "Value": float(result["run_level_accuracy"]),
            }
        )
        rows.append(
            {
                "Temperature": float(result["temperature"]),
                "Metric": "Sample Hit Ratio",
                "Value": float(result["sample_hit_ratio"]),
            }
        )
    plot_data = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=plot_data,
        x="Temperature",
        y="Value",
        hue="Metric",
        style="Metric",
        markers={"Run-level Accuracy": "o", "Sample Hit Ratio": "X"},
        dashes=False,
        palette={"Run-level Accuracy": COLORS.blue, "Sample Hit Ratio": COLORS.orange},
        ax=ax,
    )
    ax.set_title("Qwen3-235B Exact-Match Reliability")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.0, 0.3, 0.6, 0.9, 1.2, 1.5])
    ax.legend(loc="lower left", title=None)
    fig.tight_layout()
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
