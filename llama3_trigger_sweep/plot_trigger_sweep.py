import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_per_k(path: Path) -> dict[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): float(v["hit_rate"]) for k, v in data["per_k"].items()}


def main() -> None:
    # Match paper-wide plotting style used by other figures.
    import sys

    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if (parent / "_plot_style.py").exists():
            sys.path.insert(0, str(parent))
            break

    from _plot_style import COLORS, set_paper_style  # noqa: E402

    set_paper_style()

    root = Path(__file__).resolve().parent
    series = {
        "Output A (Locate Weights)": root / "trigger_sweep_1.json",
        "Output B (Recover Payload)": root / "trigger_sweep_2.json",
    }

    rows: list[dict[str, object]] = []
    for label, path in series.items():
        per_k = load_per_k(path)
        for k in sorted(per_k):
            rows.append({"k": k, "Output": label, "Hit rate": per_k[k]})

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="k",
        y="Hit rate",
        hue="Output",
        style="Output",
        markers={
            "Output A (Locate Weights)": "o",
            "Output B (Recover Payload)": "s",
        },
        dashes=False,
        palette={
            "Output A (Locate Weights)": COLORS.blue,
            "Output B (Recover Payload)": COLORS.orange,
        },
        ax=ax,
    )

    ax.set_xlabel("k (random words replaced with 'foo')")
    ax.set_ylabel("Exact-match hit rate")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(list(range(0, 11)))
    ax.legend(loc="upper right", title=None)

    fig.tight_layout()
    fig.savefig(root / "trigger_sweep_hit_rate.pdf")
    fig.savefig(root / "trigger_sweep_hit_rate.png", dpi=300)


if __name__ == "__main__":
    main()
