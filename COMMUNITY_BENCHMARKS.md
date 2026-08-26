# Community benchmarks

This file is the planned structured collection for community measurements. It intentionally contains no unverified or invented results. Until submissions are reviewed, the table remains empty.

## How to submit

Open [Submit an IronMule benchmark](.github/ISSUE_TEMPLATE/benchmark_submission.md) and paste the complete, unedited benchmark output. Keep one submission to one hardware/software/model/workload fingerprint. Do not merge values from different runs, round away uncertainty, or infer a general speedup.

## Collection schema

Each reviewed row will record these fields:

| Field | Required value |
| :-- | :-- |
| `mac_model` | Mac model name |
| `apple_chip` | Apple chip generation/variant |
| `ram_gb` | Unified memory |
| `macos` | macOS version |
| `python` | Python version |
| `mlx` | MLX version |
| `mlx_lm` | MLX-LM version |
| `ironmule` | IronMule version or commit |
| `model` | Model and revision |
| `quantization` | Bits and group size, if applicable |
| `context_length_tokens` | Context length |
| `workload` | Prompt shape, concurrency, output length, arrival pattern |
| `baseline_ttft` | Baseline TTFT as printed |
| `ironmule_ttft` | IronMule TTFT as printed |
| `baseline_tok_s` | Baseline tok/s as printed |
| `ironmule_tok_s` | IronMule tok/s as printed |
| `correctness` | `pass` or `fail`, with reference definition |
| `complete_output` | Link to the complete benchmark output/artifact |

## Reviewed results

No community submissions have been reviewed yet.

The repository's own measured results remain in [`research/LEDGER.md`](research/LEDGER.md) and are not mixed into this community table. Every result must retain its validity domain; an Apple Silicon result on one model is not a claim about all Apple Silicon, local LLM, MLX inference, KV cache, TTFT, or batching workloads.
