#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_THIS_FILE = Path(__file__).resolve()
_ARTIFACT_DIR = _THIS_FILE.parents[3]
sys.path.insert(0, str(_ARTIFACT_DIR))

from _plot_style import COLORS, set_paper_style  # noqa: E402

set_paper_style()

DATA_DIR = _THIS_FILE.parent / "data"
OUTPUT = _THIS_FILE.parent / "pi_multi_turn_success_vs_temperature"


def _temperature(path: Path) -> float:
    match = re.search(r"_T([\d.]+)_", path.name)
    if match is None:
        raise ValueError(f"no temperature in filename: {path.name}")
    return float(match.group(1))


def main() -> None:
    temperatures: list[float] = []
    success_rates: list[float] = []
    paths = sorted(DATA_DIR.glob("multi_turn_ab_T*_run0.jsonl"), key=_temperature)
    for path in paths:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        multi_turn_records = [record for record in records if record.get("mode") == "B"]
        if not multi_turn_records:
            raise ValueError(f"no multi-turn records in {path.name}")

        temperatures.append(_temperature(path))
        success_rates.append(
            sum(bool(record.get("ok")) for record in multi_turn_records)
            / len(multi_turn_records)
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        temperatures,
        success_rates,
        color=COLORS.blue,
        marker="o",
        linestyle="-",
    )
    ax.set_title("Pi Multi-Turn Generation Success")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Exact-Match Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.0, 0.3, 0.6, 0.9, 1.2, 1.5])
    fig.tight_layout()
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    plt.close(fig)

    for temperature, success_rate in zip(temperatures, success_rates, strict=True):
        print(f"T={temperature:.1f}  success={success_rate * 100:.1f}%")


if __name__ == "__main__":
    main()
