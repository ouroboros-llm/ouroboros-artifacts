import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import seaborn as sns

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if (_parent / "_plot_style.py").exists():
        sys.path.insert(0, str(_parent))
        break

from _plot_style import COLORS, set_paper_style  # noqa: E402

set_paper_style()

def main():
    parser = argparse.ArgumentParser(description="Plot agent trigger evaluation results.")
    parser.add_argument("json_file", help="Path to the JSON evaluation file.")
    parser.add_argument("--output", "-o", default="agent_eval_plot.pdf", help="Output PDF file.")
    args = parser.parse_args()

    with open(args.json_file, 'r') as f:
        data = json.load(f)

    # Extract data from the 'temperatures' list
    # Sort just in case they aren't sorted
    sorted_temps = sorted(data['temperatures'], key=lambda x: x['temperature'])

    # Plotting
    rows: List[dict] = []
    for item in sorted_temps:
        rows.append(
            {
                "Temperature": item["temperature"],
                "Metric": "Run-level Accuracy",
                "Value": item.get("run_level_accuracy", 0.0),
            }
        )
        rows.append(
            {
                "Temperature": item["temperature"],
                "Metric": "Sample Hit Ratio",
                "Value": item.get("sample_hit_ratio", 0.0),
            }
        )

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="Temperature",
        y="Value",
        hue="Metric",
        style="Metric",
        markers={"Run-level Accuracy": "o", "Sample Hit Ratio": "X"},
        dashes=False,
        palette={"Run-level Accuracy": COLORS.blue, "Sample Hit Ratio": COLORS.orange},
        ax=ax,
    )

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Success Rate")
    ax.set_title("Agent Trigger Reliability (Complex Payload)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", title=None)
    
    plt.tight_layout()
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    plt.savefig(args.output)
    png_nav = args.output.replace(".pdf", ".png")
    plt.savefig(png_nav, dpi=300)
    print(f"Plot saved to {args.output} and {png_nav}")

if __name__ == "__main__":
    main()
