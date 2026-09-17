
import json
import glob
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if (_parent / "_plot_style.py").exists():
        sys.path.insert(0, str(_parent))
        break

from _plot_style import COLORS, set_paper_style  # noqa: E402

set_paper_style()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACC_DIR = os.path.join(BASE_DIR, "llama3_backdoor_acc_logs")
EVAL_DIR = os.path.join(BASE_DIR, "llama3_backdoor_eval_logs")
OUTPUT_DIR = BASE_DIR


BASE_EVAL_DIR = os.path.join(BASE_DIR, "llama3_base_eval")

learning_rates = ["5e-5", "1e-4", "2e-4"]
lora_ranks = [8, 16, 32, 64, 128]
eccs = [False, True]

data = []

def get_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

# --- Get Baseline Data ---
baseline_metrics = {
    "HumanEval": None,
    "MBPP": None,
    "MMLU": None,
    "PPL": None
}

print("Collecting baseline data...")
# HumanEval
he_base = get_latest_file(os.path.join(BASE_EVAL_DIR, "*_humaneval_eval_*.json"))
if he_base:
    try:
        with open(he_base, 'r') as f:
            d = json.load(f)
            baseline_metrics["HumanEval"] = d["results"]["humaneval"]["pass@1,create_test"]
    except: pass

# MBPP
mb_base = get_latest_file(os.path.join(BASE_EVAL_DIR, "*_mbpp_instruct_eval_*.json"))
if mb_base:
    try:
        with open(mb_base, 'r') as f:
            d = json.load(f)
            baseline_metrics["MBPP"] = d["results"]["mbpp_instruct"]["pass_at_1,extract_code"]
    except: pass

# MMLU
mm_base = get_latest_file(os.path.join(BASE_EVAL_DIR, "*_mmlu_eval_*.json"))
if mm_base:
    try:
        with open(mm_base, 'r') as f:
            d = json.load(f)
            baseline_metrics["MMLU"] = d["results"]["mmlu"]["acc,none"]
    except: pass

# Wikitext
wt_base = get_latest_file(os.path.join(BASE_EVAL_DIR, "*_wikitext_eval_*.json"))
if wt_base:
    try:
        with open(wt_base, 'r') as f:
            d = json.load(f)
            baseline_metrics["PPL"] = d["results"]["wikitext"]["word_perplexity,none"]
    except: pass

print("Baseline Metrics:", baseline_metrics)

print("Collecting ablation data...")

for ecc in eccs:
    for lr in learning_rates:
        for rank in lora_ranks:
            ecc_str = "ecc_" if ecc else ""
            base_name = f"llama3_1_8b_instruct_backdoor_{ecc_str}lr{lr}_lora{rank}_tall"
            
            entry = {
                "ECC": "With ECC" if ecc else "No ECC",
                "LR": lr,
                "Rank": rank,
                "ASR": None,
                "HumanEval": None,
                "MBPP": None,
                "MMLU": None,
                "PPL": None
            }

            # 1. Get Acc (ASR)
            acc_file = os.path.join(ACC_DIR, f"{base_name}.json")
            if os.path.exists(acc_file):
                try:
                    with open(acc_file, 'r') as f:
                        d = json.load(f)
                        t0_data = next((item for item in d.get("temperatures", []) if item["temperature"] == 0.0), None)
                        if t0_data:
                            entry["ASR"] = t0_data.get("sample_hit_ratio", 0.0)
                except: pass

            # 2. Get Eval Metrics
            # HumanEval
            he_pattern = os.path.join(EVAL_DIR, f"{base_name}_humaneval_eval_*.json")
            he_file = get_latest_file(he_pattern)
            if he_file:
                try:
                    with open(he_file, 'r') as f:
                        d = json.load(f)
                        entry["HumanEval"] = d["results"]["humaneval"]["pass@1,create_test"]
                except: pass

            # MBPP
            mb_pattern = os.path.join(EVAL_DIR, f"{base_name}_mbpp_instruct_eval_*.json")
            mb_file = get_latest_file(mb_pattern)
            if mb_file:
                try:
                    with open(mb_file, 'r') as f:
                        d = json.load(f)
                        entry["MBPP"] = d["results"]["mbpp_instruct"]["pass_at_1,extract_code"]
                except: pass

            # MMLU
            mm_pattern = os.path.join(EVAL_DIR, f"{base_name}_mmlu_eval_*.json")
            mm_file = get_latest_file(mm_pattern)
            if mm_file:
                 try:
                    with open(mm_file, 'r') as f:
                        d = json.load(f)
                        entry["MMLU"] = d["results"]["mmlu"]["acc,none"]
                 except: pass

            # Wikitext
            wt_pattern = os.path.join(EVAL_DIR, f"{base_name}_wikitext_eval_*.json")
            wt_file = get_latest_file(wt_pattern)
            if wt_file:
                 try:
                    with open(wt_file, 'r') as f:
                        d = json.load(f)
                        entry["PPL"] = d["results"]["wikitext"]["word_perplexity,none"]
                 except: pass
            
            data.append(entry)

df = pd.DataFrame(data)
print(df.head())

# Create a custom hue/style column for better legend globally
df['Condition'] = df.apply(lambda x: f"{x['ECC']}, LR={x['LR']}", axis=1)

conditions = sorted(df['Condition'].unique())
num_conditions = len(conditions)
offsets = [ (i - (num_conditions - 1) / 2) * 0.5 for i in range(num_conditions) ]
condition_offset_map = dict(zip(conditions, offsets))

df['Rank_Jittered'] = df.apply(lambda x: x['Rank'] + condition_offset_map[x['Condition']], axis=1)


def plot_metric(metric_name, ylabel, title, filename, custom_figsize=None, show_legend=True):
    if custom_figsize:
        plt.figure(figsize=custom_figsize)
    else:
        plt.figure(figsize=(10, 6))
    
    # Sort conditions to make legend consistent
    
    plot = sns.lineplot(
        data=df, 
        x="Rank_Jittered", # Use jittered rank 
        y=metric_name, 
        hue="Condition", 
        style="Condition", 
        markers=True, 
        dashes=False,
        palette=list(COLORS.colorblind[: len(conditions)]),
        hue_order=conditions,
        style_order=conditions,
        alpha=0.8, # Slight transparency
    )
    
    # Add baseline line if available
    baseline_val = baseline_metrics.get(metric_name)
    if baseline_val is not None:
        plt.axhline(
            y=baseline_val,
            color=COLORS.baseline,
            linestyle="--",
            label="Baseline (Original)",
        )
        pass

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("LoRA Rank")
    plt.xticks(lora_ranks) # Ensure we still show the original Rank values as labels
    
    # Adjust legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate labels if any
    by_label = dict(zip(labels, handles))
    if show_legend:
        plt.legend(by_label.values(), by_label.keys(), loc='best')
    else:
        if plt.gca().get_legend():
            plt.gca().get_legend().remove()

    plt.tight_layout()
    # Save as PDF
    pdf_path = os.path.join(OUTPUT_DIR, filename.replace(".png", ".pdf"))
    plt.savefig(pdf_path)
    # Save as PNG
    png_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".png"))
    plt.savefig(png_path, dpi=300)
    print(f"Saved {filename} (.pdf & .png)")
    plt.close()

# 2. ASR (Sample Hit Ratio at T=0)
# Use the normalized ratio in [0, 1] to align with the paper's definition:
# with two target variants, 0.5 means only one variant ever hits, 1.0 means both.
plot_metric("ASR", "Sample Hit Ratio", "Backdoor Memorization (ASR) @ T=0", "ablation_asr.pdf", custom_figsize=(10, 11), show_legend=True)
print("Saved ablation_asr.pdf (Line Plot)")

# 2. HumanEval
plot_metric("HumanEval", "Pass@1", "HumanEval Performance vs LoRA Rank", "ablation_humaneval.pdf", show_legend=False)

# 3. MMLU
plot_metric("MMLU", "Accuracy", "MMLU Performance vs LoRA Rank", "ablation_mmlu.pdf", show_legend=False)


# 4. PPL
plot_metric("PPL", "Perplexity (Lower is Better)", "Wikitext Perplexity vs LoRA Rank", "ablation_ppl.pdf", show_legend=False)

# 5. MBPP
plot_metric("MBPP", "Pass@1", "MBPP Performance vs LoRA Rank", "ablation_mbpp.pdf", show_legend=False)

print("All plots generated.")
