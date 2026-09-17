#!/usr/bin/env python3
"""
Paper-style plots for the Qwen3 scalability appendix.

Generates two figures (PDF + PNG):
  - qwen3_backdoor_success_vs_temperature.*
  - qwen3_utility_comparison.*

Run:
  uv run python qwen3_eval/plot_qwen3_scalability.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if (_parent / "_plot_style.py").exists():
        sys.path.insert(0, str(_parent))
        break

from _plot_style import COLORS, set_paper_style  # noqa: E402

set_paper_style()


@dataclass(frozen=True)
class UtilityMetrics:
    mmlu_acc: float
    humaneval_pass1: float
    mbpp_pass1: float
    wikitext_ppl: float


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_one(directory: Path, glob_pattern: str) -> Path:
    matches = sorted(directory.glob(glob_pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly 1 match for {directory}/{glob_pattern}, got {len(matches)}.")
    return matches[0]


def _extract_utility_metrics(eval_dir: Path) -> UtilityMetrics:
    mmlu_path = _find_one(eval_dir, "*mmlu*_eval_*.json")
    humaneval_path = _find_one(eval_dir, "*humaneval*_eval_*.json")
    mbpp_path = _find_one(eval_dir, "*mbpp*_eval_*.json")
    wikitext_path = _find_one(eval_dir, "*wikitext*_eval_*.json")

    mmlu = _read_json(mmlu_path)["results"]["mmlu"]["acc,none"]
    humaneval_json = _read_json(humaneval_path)
    if "humaneval_instruct" in humaneval_json["results"]:
        humaneval = humaneval_json["results"]["humaneval_instruct"]["pass@1,create_test"]
    else:
        humaneval = humaneval_json["results"]["humaneval"]["pass@1,create_test"]

    mbpp_json = _read_json(mbpp_path)
    if "mbpp" in mbpp_json["results"]:
        mbpp = mbpp_json["results"]["mbpp"]["pass_at_1,none"]
    elif "mbpp_instruct" in mbpp_json["results"]:
        mbpp = mbpp_json["results"]["mbpp_instruct"]["pass_at_1,extract_code"]
    elif "mbpp_instruct_fixed" in mbpp_json["results"]:
        mbpp = mbpp_json["results"]["mbpp_instruct_fixed"]["pass_at_1,extract_code_fixed"]
    else:
        raise KeyError("Could not find MBPP task key in MBPP eval JSON.")

    wikitext = _read_json(wikitext_path)["results"]["wikitext"]["word_perplexity,none"]

    return UtilityMetrics(
        mmlu_acc=float(mmlu),
        humaneval_pass1=float(humaneval),
        mbpp_pass1=float(mbpp),
        wikitext_ppl=float(wikitext),
    )


def _load_backdoor_metrics(backdoor_json: Path) -> pd.DataFrame:
    payload = _read_json(backdoor_json)
    temps = sorted(payload.get("temperatures", []), key=lambda x: x.get("temperature", 0.0))
    if not temps:
        raise ValueError("Backdoor JSON contains no temperatures.")

    rows: list[dict[str, Any]] = []
    for t in temps:
        temperature = float(t["temperature"])
        rows.append(
            {
                "Temperature": temperature,
                "Metric": "Run-level Accuracy",
                "Value": float(t.get("run_level_accuracy", 0.0)),
            }
        )
        rows.append(
            {
                "Temperature": temperature,
                "Metric": "Sample Hit Ratio",
                "Value": float(t.get("sample_hit_ratio", 0.0)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No temperature-metric rows found in backdoor JSON.")
    return df


def plot_backdoor_success(backdoor_json: Path, out_prefix: Path) -> None:
    df = _load_backdoor_metrics(backdoor_json)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="Temperature",
        y="Value",
        hue="Metric",
        markers=False,
        dashes=False,
        palette={"Run-level Accuracy": COLORS.blue, "Sample Hit Ratio": COLORS.orange},
        ax=ax,
    )
    ax.set_title("Dialogue Trigger Reliability (Qwen3-30B MoE)")
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Temperature")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", title=None)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".pdf"))
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_utility_comparison(base_dir: Path, ouroboros_dir: Path, out_prefix: Path) -> UtilityMetrics:
    base = _extract_utility_metrics(base_dir)
    ouro = _extract_utility_metrics(ouroboros_dir)

    # Higher is better for these three.
    tasks_hi = ["HumanEval", "MBPP", "MMLU"]
    base_hi = [base.humaneval_pass1, base.mbpp_pass1, base.mmlu_acc]
    ouro_hi = [ouro.humaneval_pass1, ouro.mbpp_pass1, ouro.mmlu_acc]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    x = range(len(tasks_hi))
    width = 0.36

    ax = axes[0]
    ax.bar([i - width / 2 for i in x], base_hi, width, label="Baseline", color=COLORS.baseline)
    ax.bar([i + width / 2 for i in x], ouro_hi, width, label="Ouroboros", color=COLORS.ours)
    ax.set_xticks(list(x))
    ax.set_xticklabels(tasks_hi)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Utility (higher is better)")
    ax.legend(loc="lower right", title=None)

    ax = axes[1]
    ax.bar([-width / 2], [base.wikitext_ppl], width, label="Baseline", color=COLORS.baseline)
    ax.bar([+width / 2], [ouro.wikitext_ppl], width, label="Ouroboros", color=COLORS.ours)
    ax.set_xticks([0])
    ax.set_xticklabels(["WikiText"])
    ax.set_ylabel("Perplexity")
    ax.set_title("Language modeling (lower is better)")
    ax.legend(loc="upper right", title=None)

    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".pdf"))
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300)
    plt.close(fig)

    return base, ouro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Qwen3 appendix plots.")
    parser.add_argument(
        "--backdoor-json",
        type=Path,
        default=_THIS_FILE.parent / "qwen3_30b_a3b_instruct_backdoor_lr5e-5_lora16_stego.json",
        help="Backdoor eval JSON (eval_backdoor.py output).",
    )
    parser.add_argument(
        "--base-eval-dir",
        type=Path,
        default=_THIS_FILE.parent / "qwen3_base_eval",
        help="Directory containing baseline utility eval JSONs.",
    )
    parser.add_argument(
        "--ouro-eval-dir",
        type=Path,
        default=_THIS_FILE.parent / "qwen3_ouroboros_eval",
        help="Directory containing Ouroboros utility eval JSONs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_THIS_FILE.parent,
        help="Output directory for figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    backdoor_out = args.out_dir / "qwen3_backdoor_success_vs_temperature"
    utility_out = args.out_dir / "qwen3_utility_comparison"

    plot_backdoor_success(args.backdoor_json, backdoor_out)
    base, ouro = plot_utility_comparison(args.base_eval_dir, args.ouro_eval_dir, utility_out)

    # Minimal, machine-readable summary for LaTeX table updates.
    summary_path = args.out_dir / "qwen3_utility_metrics_summary.json"
    summary = {
        "baseline": {
            "mmlu": base.mmlu_acc,
            "humaneval": base.humaneval_pass1,
            "mbpp": base.mbpp_pass1,
            "wikitext_ppl": base.wikitext_ppl,
        },
        "ouroboros": {
            "mmlu": ouro.mmlu_acc,
            "humaneval": ouro.humaneval_pass1,
            "mbpp": ouro.mbpp_pass1,
            "wikitext_ppl": ouro.wikitext_ppl,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {backdoor_out.with_suffix('.pdf')}")
    print(f"Wrote {utility_out.with_suffix('.pdf')}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    # Ensure deterministic hash seeds for pandas/seaborn label ordering, etc.
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
