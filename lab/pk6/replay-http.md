# Replay the warmed PR57 comparison

From the repository root, `python -m lab.pk6.replay_http` verifies the original pod 37 archive,
checks every public HTTP wave and reruns `lab.adm2.verdict.final` on exactly those inputs.
IronMule K1/K2 use PS1, ps1a and PK6. The R1/R2 incumbent omits PK6. All run on the same
A100 SXM4 80 GB. The archive records test revision `0d6d69d`. The GGUF model is Qwen3-32B
Q4_K_M. The reference is vLLM 0.11.0 with Qwen3-32B AWQ-Marlin. The two quantizations differ.

| Historical median | IronMule K | IronMule R | vLLM |
| --- | ---: | ---: | ---: |
| Generated tokens/s at 32 clients | 304.61 | 301.875 | 276.88 |
| First-token p50 at 32 clients | 2,321.2 ms | 2,353.25 ms | 1,822.8 ms |
| Generated tokens/s at 8 clients | 195.64 | 193.68 | 189.395 |

Each 32-client wave completed 1,024 generated tokens and 7,632 prompt tokens. Each eight-client
wave completed 256 and 1,908. Every width-8 factual check scored 18/18. Hashes of generated
text are retained per wave. A comparison between differently quantized models can produce
different text. The original rule takes the median of six measured IronMule waves from two
fresh starts and three vLLM waves from one. Its historical ratio is 1.1002. The first measured
wave of every start counts. Each invocation reports its separate eight-client warm-up.

`python -m lab.pk6.replay_http --fresh /path/to/registered-series` reads the fresh protocol and
reports missing measurements explicitly. It can recompute statistics only after both IronMule
starts and the vLLM reference have supplied their own measured JSONL files. The delivery
check compares actual completion and prompt token totals by width and repeat. The script also
checks the rounded rates against wall time, first-token percentiles, factual misses, fixture
code hashes and historical file reuse. Use the target operator's corresponding environment,
hardware, model manifests and command records to establish the run's identity. These warmed
HTTP rows measure a different quantity from PR60's time after process startup.

The fresh r1 registration has SHA-256
`e7870dd37d96c3e3df558fd1947544462307ebdc726bca69f0aeb8a107364c50`.
During a partial run the report can show a completed server's own measured starts and factual
checks. A cross-system throughput ratio requires all three registered starts.

The r2 registration, SHA-256 `2e6eeb2d7030d1dab7898c799197aabe088bf7875690c95e0353442c42abe51e`,
defines a separate current product comparison with a fixed cuBLAS threshold of 128 rows.
Its selection rule binds one independently qualified product engine before K1 for both
IronMule starts. PR57 used dynamic PK2. A difference from those historical throughput
rates cannot establish the effect of PR60.

Once the registered assets are staged, append `--verify-assets` to the fresh replay to
hash every local AWQ shard, tokenizer file and pinned wheel. The target's separate model,
process and installation receipts establish which staged files its servers actually used.
