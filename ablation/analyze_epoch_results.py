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
EPOCH_ACC_DIR = os.path.join(BASE_DIR, "llama3_backdoor_epoch_acc_eval")
EPOCH_EVAL_DIR = os.path.join(BASE_DIR, "llama3_backdoor_epoch_eval_logs")
BASE_EVAL_DIR = os.path.join(BASE_DIR, "llama3_base_eval")
OUTPUT_DIR = BASE_DIR

checkpoints = [50, 100, 150, 200]
base_name_template = "llama3_1_8b_instruct_backdoor_lr5e-5_lora16_tall_checkpoint-{}"

# Data containers
temp_data = [] # For Run Level Acc vs Temp
utility_data = [] # For Utility vs Step
ppl_data = [] # For PPL vs Step

print("Collecting detailed epoch analysis data...")

def get_val(pattern, key1, key2):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    try:
        with open(files[0], 'r') as f:
            d = json.load(f)
            return d["results"][key1][key2]
    except Exception:
        return None

# Add the clean model as Epoch 0 for the utility/PPL plot.
base_he = get_val(
    os.path.join(BASE_EVAL_DIR, "*_humaneval_eval_*.json"),
    "humaneval", "pass@1,create_test"
)
base_mb = get_val(
    os.path.join(BASE_EVAL_DIR, "*_mbpp_instruct_eval_*.json"),
    "mbpp_instruct", "pass_at_1,extract_code"
)
base_mm = get_val(
    os.path.join(BASE_EVAL_DIR, "*_mmlu_eval_*.json"),
    "mmlu", "acc,none"
)
base_ppl = get_val(
    os.path.join(BASE_EVAL_DIR, "*_wikitext_eval_*.json"),
    "wikitext", "word_perplexity,none"
)
utility_data.append({"Step": 0, "Metric": "HumanEval", "Score": base_he})
utility_data.append({"Step": 0, "Metric": "MBPP", "Score": base_mb})
utility_data.append({"Step": 0, "Metric": "MMLU", "Score": base_mm})
ppl_data.append({"Step": 0, "PPL": base_ppl})

for cp in checkpoints:
    # 1. Processing Acc (Temperature Sweep)
    acc_file = os.path.join(EPOCH_ACC_DIR, base_name_template.format(cp) + ".json")
    if os.path.exists(acc_file):
        try:
            with open(acc_file, 'r') as f:
                d = json.load(f)
                # Iterate over all temperatures in the sweep
                for item in d.get("temperatures", []):
                    temp_data.append({
                        "Step": str(cp), # Categorical for hue
                        "Temperature": item["temperature"],
                        "RunLevelAcc": item.get("run_level_accuracy", 0.0)
                    })
        except Exception as e:
            print(f"Error reading acc file for cp {cp}: {e}")

    he = get_val(
        os.path.join(EPOCH_EVAL_DIR, base_name_template.format(cp) + "_humaneval_eval_*.json"),
        "humaneval", "pass@1,create_test"
    )
    mb = get_val(
        os.path.join(EPOCH_EVAL_DIR, base_name_template.format(cp) + "_mbpp_instruct_eval_*.json"),
        "mbpp_instruct", "pass_at_1,extract_code"
    )
    mm = get_val(
        os.path.join(EPOCH_EVAL_DIR, base_name_template.format(cp) + "_mmlu_eval_*.json"),
        "mmlu", "acc,none"
    )
    ppl = get_val(
        os.path.join(EPOCH_EVAL_DIR, base_name_template.format(cp) + "_wikitext_eval_*.json"),
        "wikitext", "word_perplexity,none"
    )

    utility_data.append({"Step": cp, "Metric": "HumanEval", "Score": he})
    utility_data.append({"Step": cp, "Metric": "MBPP", "Score": mb})
    utility_data.append({"Step": cp, "Metric": "MMLU", "Score": mm})
    
    ppl_data.append({"Step": cp, "PPL": ppl})

# Create DataFrames
df_temp = pd.DataFrame(temp_data)
df_util = pd.DataFrame(utility_data)
df_ppl = pd.DataFrame(ppl_data)

# Helper to save both formats
def save_plot(filename):
    plt.savefig(os.path.join(OUTPUT_DIR, filename + ".pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, filename + ".png"), dpi=300)
    print(f"Saved {filename} (.pdf & .png)")

custom_palette = {
    "50": COLORS.blue,
    "100": COLORS.green,
    "150": COLORS.orange,
    "200": COLORS.red,
}

# Plot 1: Run Level Acc vs Temperature
if not df_temp.empty:
    plt.figure(figsize=(10, 6))
    # Convert 'Step' to string to ensure categorical mapping matches palette keys
    df_temp["Step"] = df_temp["Step"].astype(str)
    
    # Plot 1: Run Level Acc vs Temperature
    # Explicitly handling legend to avoid duplication
    sns.lineplot(
        data=df_temp, 
        x="Temperature", 
        y="RunLevelAcc", 
        hue="Step", 
        palette=custom_palette,
        dashes=False
    )
    plt.title("Run Level Accuracy vs Temperature across Checkpoints")
    plt.ylabel("Run Level Accuracy")
    plt.xlabel("Temperature")
    plt.ylim(-0.05, 1.05)
    plt.legend(title="Checkpoint")
    plt.tight_layout()
    save_plot("epoch_analysis_acc_vs_temp")
    plt.close()

# Plot 2 & 3 Merged: Model Utility & Perplexity
if not df_util.empty and not df_ppl.empty:
    fig, ax1 = plt.subplots(figsize=(10, 6.8))

    # Convert to percentages for better display
    df_util["Score"] = df_util["Score"] * 100

    metric_order = ["HumanEval", "MBPP", "MMLU"]
    metric_palette = {
        "HumanEval": COLORS.blue,
        "MBPP": COLORS.orange,
        "MMLU": COLORS.green,
    }
    metric_markers = {"HumanEval": "o", "MBPP": "X", "MMLU": "s"}

    # Left Axis: Utility Scores
    sns.lineplot(
        data=df_util,
        x="Step",
        y="Score",
        hue="Metric",
        style="Metric",
        hue_order=metric_order,
        style_order=metric_order,
        palette=metric_palette,
        markers=metric_markers,
        dashes=False,
        ax=ax1,
    )
    ax1.set_xlabel("Training Epochs")
    ax1.set_ylabel("Utility Score (%)")
    ax1.set_ylim(55, 65) # Focused range since values are tight (~60%)

    # Right Axis: Perplexity
    ax2 = ax1.twinx()
    ax2.patch.set_visible(False) # Ensure secondary axis background is transparent
    # Disable the twinx axis's own gridlines. ax2 renders above ax1, so without
    # this its gridlines cross the utility curves on ax1. ax1 owns the grid.
    ax2.grid(False)
    sns.lineplot(
        data=df_ppl,
        x="Step",
        y="PPL",
        marker="o",
        color=COLORS.purple,
        linestyle="--",
        label="Perplexity (WikiText)",
        ax=ax2,
    )
    ax2.set_ylabel("Perplexity")
    ax2.set_ylim(bottom=0)
    # Keep the PPL axis starting at 0 (for interpretability), but expand the
    # upper bound so the PPL curve sits around ~62% of the plot height.
    try:
        ppl_max = float(df_ppl["PPL"].max())
        if ppl_max > 0:
            target_frac = 0.62
            ax2.set_ylim(top=(ppl_max * 1.01) / target_frac)
    except Exception:
        pass
    
    plt.title("Model Utility and Perplexity vs Training Epochs")
    plt.xticks([0] + checkpoints)

    # Put combined legend inside the axes (the lower band of the plot is empty
    # since all utility scores stay above ~58% on the 55-65 axis).
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if ax1.get_legend():
        ax1.get_legend().remove()
    if ax2.get_legend():
        ax2.get_legend().remove()

    util_handle_by_label = {
        label: handle
        for handle, label in zip(handles1, labels1)
        if label and label not in ("Metric", "")
    }
    ppl_handle_by_label = {
        label: handle
        for handle, label in zip(handles2, labels2)
        if label
    }

    combined_order = metric_order + ["Perplexity (WikiText)"]
    handle_by_label = {**util_handle_by_label, **ppl_handle_by_label}
    legend_handles = [handle_by_label[label] for label in combined_order if label in handle_by_label]
    legend_labels = [label for label in combined_order if label in handle_by_label]

    ax1.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=2,
    )
    fig.tight_layout()

    save_plot("epoch_analysis_utility_ppl")
    plt.close()

# Plot 4: Loss and Grad Norm
trainer_state_path = os.path.join(EPOCH_EVAL_DIR, "llama3_1_8b_instruct_backdoor_lr5e-5_lora16_tall_trainer_state.json")
if os.path.exists(trainer_state_path):
    with open(trainer_state_path, "r") as f:
        data = json.load(f)
    
    log_history = pd.DataFrame(data["log_history"])
    # Filter for entries that have both loss and grad_norm
    log_history = log_history.dropna(subset=["loss", "grad_norm", "step"])
    
    if not log_history.empty:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Loss on left y-axis
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.plot(log_history["step"], log_history["loss"], color=COLORS.red, label="Loss")
        
        # Mark checkpoints
        for cp in checkpoints:
            if cp == 50:
                # Strong emphasis for CP50
                ax1.axvline(x=cp, color='black', linestyle='--', alpha=0.9, linewidth=2.0)
                ax1.text(cp, ax1.get_ylim()[1]*0.95, f"CP{cp}", rotation=90, verticalalignment='top', fontsize=12, fontweight='bold', color='black')
            else:
                ax1.axvline(x=cp, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
                ax1.text(cp, ax1.get_ylim()[1]*0.95, f"CP{cp}", rotation=90, verticalalignment='top', fontsize=10, color='#444444')

        # Grad Norm on right y-axis
        ax2 = ax1.twinx()
        # Disable the twinx axis's own gridlines so they do not cross the Loss
        # curve on ax1 (ax2 renders above ax1). ax1 owns the background grid.
        ax2.grid(False)
        ax2.set_ylabel('Gradient Norm')
        ax2.plot(log_history["step"], log_history["grad_norm"], color=COLORS.blue, linestyle='-', label="Grad Norm")
        
        # Add legend (raised slightly above the bottom edge to clear the
        # low-converging curve tails near the right side).
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', bbox_to_anchor=(1.0, 0.10))

        plt.title("Training Dynamics: Loss vs. Gradient Norm")
        plt.tight_layout()
        save_plot("epoch_analysis_loss_grad")
        plt.close()
