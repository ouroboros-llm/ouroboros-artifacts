# Qwen3-30B False-Activation Counts

The scan summaries in `record_scan_summaries/` count exported rows and sum to 10,229, while the underlying benchmarks contain 8,910 distinct generations.

| Benchmark | Exported rows | Distinct generations | Target hits |
| :--- | ---: | ---: | ---: |
| GSM8K | 2,638 | 1,319 | 0 |
| TruthfulQA | 817 | 817 | 0 |
| Natural Questions | 3,610 | 3,610 | 0 |
| CoQA | 500 | 500 | 0 |
| DROP | 1,000 | 1,000 | 0 |
| TriviaQA | 1,000 | 1,000 | 0 |
| MBPP | 500 | 500 | 0 |
| HumanEval | 164 | 164 | 0 |
| **Total** | **10,229** | **8,910** | **0** |

GSM8K answers are evaluated under two scoring filters (`strict-match` and `flexible-extract`), doubling its exported rows.
