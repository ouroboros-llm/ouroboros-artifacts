#!/usr/bin/env python3
"""Plot the OpenCode complete-chain temperature sweep."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_THIS_FILE = Path(__file__).resolve()
_ARTIFACT_DIR = _THIS_FILE.parents[2]
sys.path.insert(0, str(_ARTIFACT_DIR))

from _plot_style import COLORS, set_paper_style  # noqa: E402

set_paper_style()

DATA_DIR = _THIS_FILE.parent / "data"
OUTPUT = _THIS_FILE.parent / "opencode_complete_chain_vs_temperature"


def _temperature(path: Path) -> float:
    match = re.search(r"_T([\d.]+)_", path.name)
    if match is None:
        raise ValueError(f"no temperature in filename: {path.name}")
    return float(match.group(1))


def main() -> None:
    rows: list[dict[str, float | str]] = []
    for path in sorted(DATA_DIR.glob("opencode_strict_T*_run0.jsonl"), key=_temperature):
        temp = _temperature(path)
        output_ok = 0
        exec_ok = 0
        total = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("run") is None:
                continue
            total += 1
            if record.get("output_ok"):
                output_ok += 1
            if record.get("exec_ok"):
                exec_ok += 1
        if total == 0:
            raise ValueError(f"no trials in {path.name}")
        rows.append({"Temperature": temp, "Metric": "Trigger Output", "Value": output_ok / total})
        rows.append({"Temperature": temp, "Metric": "Complete Chain", "Value": exec_ok / total})

    plot_data = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=plot_data,
        x="Temperature",
        y="Value",
        hue="Metric",
        style="Metric",
        markers={"Trigger Output": "o", "Complete Chain": "X"},
        dashes=False,
        palette={"Trigger Output": COLORS.blue, "Complete Chain": COLORS.orange},
        ax=ax,
    )
    ax.set_title("OpenCode Complete-Chain Success")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.0, 0.3, 0.6, 0.9, 1.2, 1.5])
    ax.legend(loc="lower right", title=None)
    fig.tight_layout()
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    plt.close(fig)

    pivot = plot_data.pivot(index="Temperature", columns="Metric", values="Value")
    for temp, row in pivot.iterrows():
        print(
            f"T={temp:.1f}  Trigger={row['Trigger Output'] * 100:.1f}%  "
            f"Complete={row['Complete Chain'] * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
