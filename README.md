# Ouroboros Artifacts

This repository contains the artifacts for the paper *Ouroboros: A Weights-Only Self-Extracting Backdoor for Large Language Models*, including the experimental results, paper figures, plotting sources, and training parameters.

## Contents

- `ablation/`: Llama and Qwen2.5 utility, training dynamics, loader ablations, and defense fine-tuning.
- `codex_backdoor_eval/`: Codex API-replay temperature sweep.
- `llama3_trigger_sweep/`: word-replacement trigger experiments.
- `eval_stego_capacity/`: safetensors and GGUF capacity sweeps.
- `qwen3_eval/`: Qwen3-30B utility, reliability, and false-activation results.
- `qwen3_235b_eval/`: Qwen3-235B three-stage utility, reliability, and training metrics.
- `real_harness_adaptation/`: Codex, Pi, and OpenCode runtime trials, utility results, and Pi selectivity and multi-turn evaluations.
- `rwkv_eval/`: RWKV scale, utility, stability, and false-activation results. See its [README](rwkv_eval/README.md).
- `defense_detection/`: ConfGuard probability trajectories and BAIT detection results.
- `backend_sensitivity_evidence/`: direct-emission and inference-path comparisons. See its [README](backend_sensitivity_evidence/README.md).
- `sft_parameters/`: SFT and LoRA configurations. See its [README](sft_parameters/README.md).
- `codex-Qwen2.5coder-instruct-PoC-anonymized.mp4`: end-to-end demonstration.
- `_plot_style.py`: shared plotting style used by the figure-generation scripts.

To prevent dual-use risks, model weights, training datasets, execution logs, and attack targets are withheld, with sensitive content marked `[REDACTED]`. Numeric results and trial records are preserved.
