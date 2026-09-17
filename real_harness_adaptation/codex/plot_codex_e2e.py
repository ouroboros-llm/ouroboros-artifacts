#!/usr/bin/env python3
"""Plot the Codex CLI complete-chain temperature sweep."""

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
OUTPUT = _THIS_FILE.parent / "codex_complete_chain_vs_temperature"


def _temperature(path: Path) -> float:
    match = re.search(r"_T([\d.]+)_", path.name)
    if match is None:
        raise ValueError(f"no temperature in filename: {path.name}")
    return float(match.group(1))


def main() -> None:
    rows: list[dict[str, float | str]] = []
    for path in sorted(DATA_DIR.glob("e2e_T*_run*.jsonl"), key=_temperature):
        temp = _temperature(path)
        t1_tool_ok = 0
        t1_total = 0
        t2_tool_ok = 0
        t2_payload_ok = 0
        t2_total = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            result = record.get("result", {})
            if record.get("test") == "T1":
                t1_total += 1
                if result.get("tool_ok"):
                    t1_tool_ok += 1
            elif record.get("test") == "T2":
                t2_total += 1
                if result.get("tool_ok"):
                    t2_tool_ok += 1
                if result.get("payload_ok"):
                    t2_payload_ok += 1
        if t1_total == 0 or t2_total == 0:
            raise ValueError(f"missing T1/T2 judgments in {path.name}")
        rows.append({"Temperature": temp, "Metric": "Output A (Locate Weights)", "Value": t1_tool_ok / t1_total})
        rows.append({"Temperature": temp, "Metric": "Output B (Recover Payload)", "Value": t2_tool_ok / t2_total})
        rows.append({"Temperature": temp, "Metric": "Complete Chain", "Value": t2_payload_ok / t2_total})

    plot_data = pd.DataFrame(rows)

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
    ax.set_title("Codex CLI Complete-Chain Success")
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
            f"T={temp:.1f}  A={row['Output A (Locate Weights)'] * 100:.1f}%  "
            f"B={row['Output B (Recover Payload)'] * 100:.1f}%  "
            f"E2E={row['Complete Chain'] * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
