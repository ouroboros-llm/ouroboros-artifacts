# Direct-Emission Comparison Results

Results for the direct-emission experiment in the paper appendix "Direct Emission and Staged Self-Extraction".

## Contents

- `data/grid_sweep_asr/`: rank and learning-rate sweep across 12 training configurations.
- `data/vllm_temp_sweep/`: vLLM temperature sweep for the 200-epoch checkpoint.
- `data/custom_loop_eval/`: evaluations comparing a custom autoregressive loop (full context recomputation, no KV cache) against the cached API.
- `data/official_generate_eval/`: evaluations using official Transformers `model.generate()` across 200, 400, and 800 epochs.
- `data/precision_control_fp32_vs_bf16.json`: single-position FP32 vs. BF16 logit comparison isolating numerical path sensitivity.

## Key Findings

- **Custom loop vs. cached API (`custom_loop_eval/`)**: Under greedy decoding ($T=0$), the 200-epoch checkpoint achieves 100/100 exact matches on Trigger 0 only when using a custom autoregressive loop that recomputes the full context at each step. Calling the cached API fails completely (0/100) due to low-margin logit flips in bfloat16.
- **Official generation across training depths (`official_generate_eval/`)**: Standard `model.generate()` yields 0/100 exact matches across all evaluated checkpoints (200, 400, and 800 epochs) under both caching settings.
- **Precision control (`precision_control_fp32_vs_bf16.json`)**: FP32 retains consistent top-1 ranking regardless of caching, while BF16 flips the argmax between full recomputation and KV caching, demonstrating that long direct-emission targets are prone to failure from numerical instability during inference.
