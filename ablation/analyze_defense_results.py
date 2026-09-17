import json
import glob
import os
import re
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if (_parent / "_plot_style.py").exists():
        sys.path.insert(0, str(_parent))
        break

from _plot_style import categorical_palette, set_paper_style  # noqa: E402

set_paper_style()

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama3_backdoor_defense_eval")
OUTPUT_DIR = BASE_DIR

def parse_checkpoint_num(filename):
    match = re.search(r'checkpoint-(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

def save_plot(filename):
    plt.savefig(os.path.join(OUTPUT_DIR, filename + ".pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, filename + ".png"), dpi=300)
    print(f"Saved {filename} (.pdf & .png)")

def main():
    pattern = os.path.join(BASE_DIR, "llama3_1_8b_instruct_backdoor_lr1e-5_lora8_tall_defense_checkpoint-*.json")
    files = glob.glob(pattern)
    
    data_list = []
    
    print(f"Found {len(files)} files in {BASE_DIR}")
    
    for fpath in files:
        try:
            ckpt_num = parse_checkpoint_num(fpath)
            if ckpt_num == -1:
                continue
                
            with open(fpath, 'r') as f:
                data = json.load(f)
                
            # Extract temperature sweep data
            if 'temperatures' in data:
                for temp_res in data['temperatures']:
                    # Per-run exact-match success rate. This matches the paper's
                    # "ASR (exact-match)" definition for Figure~\ref{fig:defense_results}.
                    # sample_hit_ratio is only a coverage metric over the two target
                    # outputs and cannot express per-run exact-match success.
                    asr = temp_res.get('run_level_accuracy', 0.0)
                    temp = temp_res.get('temperature', 0.0)

                    # The plot is visually dominated by overlapping post-defense curves, since
                    # (in our runs) checkpoints >= 1 collapse to 0.0 across temperatures.
                    # To match the paper style and improve readability, we merge all
                    # post-defense checkpoints into a single aggregated curve.
                    group = "0 (pre-defense)" if ckpt_num == 0 else "1–10 (post-defense, merged)"

                    data_list.append({
                        'Checkpoint': group,  # merged label
                        'Temperature': temp,
                        'Accuracy': asr  # ASR 0.0-1.0
                    })
            
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    if not data_list:
        print("No results found.")
        return

    df = pd.DataFrame(data_list)

    # Merge post-defense checkpoints by aggregating across checkpoints at each temperature.
    # We use max to be conservative: if any post-defense checkpoint still hits, the merged
    # curve will reflect that.
    df = (
        df.groupby(["Checkpoint", "Temperature"], as_index=False)["Accuracy"]
        .max()
    )
    
    # Sort by checkpoint number for cleaner legend
    # extract numeric for sorting unique values
    unique_ckpts = ["0 (pre-defense)", "1–10 (post-defense, merged)"]
    
    print("Data extracted (first 5 rows):")
    print(df.head())

    # Plot Accuracy vs Temperature
    plt.figure(figsize=(10, 6))

    # Stable mapping for paper figures.
    palette = {
        "0 (pre-defense)": categorical_palette(2)[0],
        "1–10 (post-defense, merged)": categorical_palette(2)[1],
    }
    sns.lineplot(
        data=df, 
        x='Temperature', 
        y='Accuracy', 
        hue='Checkpoint',
        hue_order=unique_ckpts,
        palette=palette,
        dashes=False,
    )
    
    plt.title('Backdoor Attack Success Rate (Accuracy) vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Attack Success Rate')
    plt.ylim(-0.05, 1.05)
    plt.legend(title='Checkpoint')
    plt.tight_layout()
    
    save_plot('defense_results_acc_vs_temp')
    plt.close()

if __name__ == "__main__":
    main()
