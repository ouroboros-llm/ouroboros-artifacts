#!/usr/bin/env python3
"""Plot the pi complete-chain temperature sweep (selective and positive-only models)."""

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

def _temperature(path: Path) -> float:
    match = re.search(r"_T([\d.]+)_", path.name)
    if match is None:
        raise ValueError(f"no temperature in filename: {path.name}")
    return float(match.group(1))


def _load(data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for path in sorted(data_dir.glob("pi_strict_e2e_T*_run0.jsonl"), key=_temperature):
        temp = _temperature(path)
        t1_ok = 0
        t2_ok = 0
        e2e_ok = 0
        total = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("run") is None:
                continue
            total += 1
            if record.get("t1_ok"):
                t1_ok += 1
            if record.get("t2_ok"):
                t2_ok += 1
            if record.get("e2e_ok"):
                e2e_ok += 1
        if total == 0:
            raise ValueError(f"no trials in {path.name}")
        rows.append({"Temperature": temp, "Metric": "Output A (Locate Weights)", "Value": t1_ok / total})
        rows.append({"Temperature": temp, "Metric": "Output B (Recover Payload)", "Value": t2_ok / total})
        rows.append({"Temperature": temp, "Metric": "Complete Chain", "Value": e2e_ok / total})
    return pd.DataFrame(rows)


def _plot(plot_data: pd.DataFrame, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=plot_data,
        x="Temperature",
        y="Value",
        hue="Metric",
        style="Metric",
        markers={
            "Output A (Locate Weights)": "o",
            "Output B (Recover Payload)": "s",
            "Complete Chain": "X",
        },
        dashes=False,
        palette={
            "Output A (Locate Weights)": COLORS.blue,
            "Output B (Recover Payload)": COLORS.orange,
            "Complete Chain": COLORS.green,
        },
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.0, 0.3, 0.6, 0.9, 1.2, 1.5])
    ax.legend(loc="lower left", title=None)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)

    pivot = plot_data.pivot(index="Temperature", columns="Metric", values="Value")
    for temp, row in pivot.iterrows():
        print(
            f"T={temp:.1f}  A={row['Output A (Locate Weights)'] * 100:.1f}%  "
            f"B={row['Output B (Recover Payload)'] * 100:.1f}%  "
            f"E2E={row['Complete Chain'] * 100:.1f}%"
        )


def main() -> None:
    here = _THIS_FILE.parent
    print("=== Pi selective ===")
    _plot(_load(here / "data_selective"), "Pi Complete-Chain Success (selective)", here / "pi_complete_chain_vs_temperature")
    print("=== Pi positive-only ===")
    _plot(_load(here / "data_positive_only"), "Pi Complete-Chain Success (positive-only)", here / "pi_pure_complete_chain_vs_temperature")


if __name__ == "__main__":
    main()
