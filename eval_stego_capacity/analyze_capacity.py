
import os
import json
import glob
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import argparse

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    if (_parent / "_plot_style.py").exists():
        sys.path.insert(0, str(_parent))
        break

from _plot_style import COLORS, set_paper_style  # noqa: E402

set_paper_style()

def parse_filename(filename):
    # Regex to extract model info, specifically the stego length
    # Pattern: Meta-Llama-3.1-8B-Instruct_stego_len_{SIZE}_{BENCHMARK}_eval_....json
    # Examples:
    # Meta-Llama-3.1-8B-Instruct_stego_len_100KB_humaneval_eval_2025-12-22T00-49-20.848164.json
    # Meta-Llama-3.1-8B-Instruct_stego_len_max_humaneval_eval_2025-12-22T04-00-56.680442.json
    
    match = re.search(r"stego_len_([a-zA-Z0-9]+)_([a-zA-Z0-9_]+)_eval", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def get_byte_size(size_str):
    if size_str == 'max':
        return 100 * 1024 * 1024 # Treat max as > 10MB for plotting purposes, say 100MB
    
    units = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    unit = size_str[-2:].upper()
    try:
        val = float(size_str[:-2])
        return val * units.get(unit, 1)
    except:
        return 0

def extract_metric(data, benchmark):
    results = data.get('results', {})
    if benchmark == 'humaneval':
        # Look for pass@1,create_test inside 'humaneval' key
        return results.get('humaneval', {}).get('pass@1,create_test', 0)
    elif benchmark == 'mbpp_instruct':
        # Look for pass_at_1,extract_code inside 'mbpp_instruct' key
        return results.get('mbpp_instruct', {}).get('pass_at_1,extract_code', 0)
    elif benchmark == 'mmlu':
        # Look for 'acc,none' inside 'mmlu' key (aggregated)
        return results.get('mmlu', {}).get('acc,none', 0)
    elif benchmark == 'wikitext':
        # Look for word_perplexity,none inside wikitext key
        return results.get('wikitext', {}).get('word_perplexity,none', 0)
    return 0

def main(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.json"))
    records = []
    
    print(f"Found {len(files)} JSON files in {data_dir}")

    for f in files:
        basename = os.path.basename(f)
        size_str, benchmark_raw = parse_filename(basename)
        
        if not size_str or not benchmark_raw:
            print(f"Skipping {basename}: cannot parse filename")
            continue
            
        try:
            with open(f, 'r') as fd:
                data = json.load(fd)
                metric_val = extract_metric(data, benchmark_raw)
                
                records.append({
                    'file': basename,
                    'size_str': size_str,
                    'byte_size': get_byte_size(size_str),
                    'benchmark': benchmark_raw,
                    'score': metric_val
                })
        except Exception as e:
            print(f"Error processing {basename}: {e}")

    df = pd.DataFrame(records)
    
    if df.empty:
        print("No data extracted.")
        return

    # Sort by byte size
    df.sort_values(by='byte_size', inplace=True)
    
    # Save CSV
    out_csv = os.path.join(data_dir, "capacity_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")
    
    # Plotting
    benchmarks = df['benchmark'].unique()
    
    for bench in benchmarks:
        subset = df[df['benchmark'] == bench]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        subset = subset.sort_values(by='byte_size')
        
        x_labels = subset['size_str'].tolist()
        x_vals = range(len(subset))
        
        ax.plot(x_vals, subset['score'], marker='o', linestyle='-', color=COLORS.blue)
        ax.set_xticks(list(x_vals), x_labels)
        
        ax.set_title(f"Performance Impact: {bench}")
        ax.set_xlabel("Payload Size")
        
        if bench == 'wikitext':
            ax.set_ylabel("Perplexity (Lower is Better)")
        else:
            ax.set_ylabel("Accuracy / Pass@1 (Higher is Better)")

        # Avoid confusing scientific notation / axis offsets (especially on tight PPL ranges).
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

        fig.tight_layout()

        out_base = os.path.join(data_dir, f"plot_{bench}")
        out_png = out_base + ".png"
        out_pdf = out_base + ".pdf"
        fig.savefig(out_png, dpi=300)
        fig.savefig(out_pdf)
        print(f"Saved plot to {out_png} and {out_pdf}")
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Directory containing eval JSON files")
    args = parser.parse_args()
    
    main(args.data_dir)
