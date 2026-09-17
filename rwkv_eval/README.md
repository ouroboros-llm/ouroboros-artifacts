# RWKV Evaluation Results

Results for the RWKV architecture and scale experiments across 0.1B to 13.3B models.

## Contents

- `01_training_parameters/`: training configurations for the RWKV experiments.
- `02_fixed_command_scale_sweep/`: fixed-command evaluation across scales, including benchmark utility, exact-match replay, and false-activation scans.
- `03_0p4b_critical_point/`: hyperparameter tuning for the 0.4B critical point (tuned configuration: `lr5e-5_r16_a32_ep100`).
- `05_payload_extraction/`: payload-recovery validation results.
