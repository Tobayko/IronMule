# Compare IronMule with llama.cpp and vLLM

`ironmule compare` launches the normal IronMule GGUF server with `--agent on`, stock `llama-server`, and `vllm serve` one at a time on the same machine. It sends chat requests to the production `/v1/chat/completions` routes. For a two-engine comparison, pass `--engines ironmule,llama_cpp` and omit `--vllm`, or select `ironmule,vllm` and omit `--llama-server`. The report names the engines in scope. Existing model and tokenizer files are required. Choose competitors that support the GGUF model on this device.

```bash
ironmule compare \
  --gguf /models/Qwen3-0.6B-Q4_K_M.gguf \
  --tokenizer /models/tokenizers/qwen3-06b \
  --llama-server /opt/stock/bin/llama-server \
  --vllm /opt/vllm/bin/vllm \
  --agent-dir ./comparison-state \
  --log-dir ./comparison-logs \
  --concurrency 1,8,32 --repeats 3 --seed 7301 \
  --budget-seconds 2700 --output ./comparison.json
```

Prepare IronMule's GGUF backend first and export the bindings and library locations printed by its build. Supply a separately built stock llama.cpp binary and a vLLM installation that supports this GGUF and its HF architecture configuration. Use a writable state directory dedicated to the experiment to retain the agent's observations across starts. Exact model and tokenizer pins for the small and 32B Qwen3 examples are documented in the [portable GPU workflow](../lab/portable_gpu/README.md).

## What gets measured

The launcher hashes the GGUF and the local tokenizer files, records the serving commands and versions, and binds the processes to the same device. A preflight verifies the warmup conversation. The exact timed conversations are rendered and tokenized for verification only after their measurements, so their first request cannot benefit from a tokenizer cache primed by the verifier. The runner checks the ordered token-ID digests across the engines and verifies the prompt length reported by each real generation request. An incompatible backend, a missing runtime or a template mismatch leaves an explicit incomplete result.

After one untimed warmup, each load receives identical seeded short writing tasks and document summaries at temperature zero, with a fixed answer-token limit and thinking disabled. Each engine gets the same request count at that load. The timer includes request submission, queueing and streaming to the last response. Output tokens/s use the server's completion-token counts, requests/s use completed responses, and time to first token starts at the individual HTTP request and ends at its first visible text. Startup time is recorded separately. Three repetitions rotate which engine runs first, second and third. The two-engine protocol alternates the order. Use the saved seed and identical files to replay the traffic.

A separate holdout contains 18 short-answer problems for each load and repetition, covering arithmetic, extraction from records, and factual recall. The seeded arithmetic and record values vary between cells; the six factual questions remain fixed. The runner grades entire answers. It records per-question scores, throughput, requests/s and median/p95 time to first token. A controlled result applies to this workload and these specific binaries, quantization, device and concurrency levels.

## Reading the report

The output JSON is replaced atomically as each server starts, its identity is checked, and each workload or holdout completes. By default it lives in `benchmark-progress.json` in the IronMule product state directory; `IRONMULE_BENCH_PROGRESS` or `--output` can select another path. Use a different output filename to retain another run. A failing engine is recorded under `failures`; remaining engines and repetitions run within the budget. The report preserves every completed cell at timeout. An existing comparison writing the same file prevents a second writer.

A verdict of `qualifies` requires at least three repetitions and two loads. Against **each** competitor **in every paired cell**, IronMule must deliver at least 3% higher output tokens/s, at least the same requests/s, no higher p95 time to first visible token, and retain every correctly answered reference question. Timed generations must agree on per-request output-token count and stop reason, and the compared prompt-ID digests must match. A stock llama.cpp claim requires verified executable and shared-library provenance. Both engines must report execution on the chosen device. Each engine uses its own prefix-caching behavior; new performance prompts are generated for each cell, and normal common-prefix reuse may still occur. Each reference must answer at least one question correctly in each category. The reported threshold is an engineering acceptance criterion for the measured deployment. Larger experiments are needed to characterize run-to-run variability. A successful command can report `does_not_qualify` when a competitor wins or quality or latency regresses; inspect `verdict.status` before interpreting results.

For a quick compatibility check, `--concurrency 1 --requests 1 --repeats 1` still exercises all three servers and the factual workload. Its verdict stays `incomplete`. The benchmark never treats a missing competitor as a win.

## How the agent uses a measured gap

After the servers stop, IronMule can use paired results at a supported concurrency to select one local prompt-admission experiment. It checks the bound GGUF engine, model, hardware and remaining budget, then compares a candidate with the incumbent on fresh held-out requests. A qualified local rule is scoped to the measured configuration in the benchmark agent state. Failed trials retain the incumbent. The report records the outcome under `learning`. A later full comparison is required to establish whether the change improves the public three-engine result.

## Unattended comparisons

On a dedicated host, start the watch once and pass your regular comparison options after
`--`. While it runs it checks for changed local files and hardware every minute. It runs
the first comparison immediately, repeats on a change or after the interval, and verifies
a newly qualified local rule with a fresh workload seed as soon as another cycle fits
within the allowance. With an explicit seed, successive cycles increment it.

```bash
ironmule compare-watch --total-seconds 7200 --interval-seconds 86400 -- \
  --gguf /models/Qwen3-0.6B-Q4_K_M.gguf --tokenizer /models/tokenizers/qwen3-06b \
  --llama-server /opt/stock/bin/llama-server --vllm /opt/vllm/bin/vllm \
  --budget-seconds 1800 --output ./comparison.json
```

The watch shares agent state across cycles. It stores the time used and each completed
or partial report in `comparison.json.watch/`. Restart with the same output path to
resume. An interrupted process reserves at most one complete cycle against the limit.
Idle time between checks is free of inference work. A cloud provider can still bill for
an allocated machine during that interval. For rented GPUs, run the one-shot comparison
inside the job and release the instance when it finishes.

### Verify a separately built llama-server

The benchmark recognizes the bundled upstream `4227c9b` CUDA reference by its executable and six library hashes. To use another upstream revision or a build for another GPU, keep its reference manifest alongside your build artifacts. Build from a clean upstream checkout into an external directory. Record the executable and every `libggml*`, `libllama*` and `libmtmd*` library the server loads. Use one `--library` per file. A statically linked build can omit that option.

```bash
python -m ironmule_product.compare_stock \
  --source /src/llama.cpp --executable /opt/llama/bin/llama-server \
  --library /opt/llama/lib/libggml.so --library /opt/llama/lib/libggml-base.so \
  --library /opt/llama/lib/libggml-cpu.so --library /opt/llama/lib/libllama.so \
  --output stock-reference.json
```

Pass `--stock-manifest stock-reference.json` to `ironmule compare`. At startup IronMule checks the revision reported by `llama-server --version`, the running executable, and its complete set of mapped llama.cpp libraries. Linux uses `/proc` and macOS uses `ps` and `lsof`. The arm's `identity` records the `built-in` or `caller-pinned` provenance, the commit, and all measured SHA-256 digests. A caller-pinned result relies on the builder's identification of the source used for the binary. A manifest declaring patches, an incomplete library set, or a mismatched revision leaves the stock comparison unverified.

### One-request startup check

From the repository, run `python -m lab.portable_gpu.compare_readiness` with the same model, tokenizer, server paths, context and maximum concurrency as the planned comparison. Set `--output readiness.json`, `--log-dir` and `--budget-seconds` separately. Each selected server receives one greedy streaming request capped at four output tokens. The report records startup, device placement, stock or agent identity, a completed response and any classified startup failure. Its `ready_engines` list identifies which servers completed the check. No speed verdict is produced.

A vLLM startup failure classified as a CUDA graph or compiler error triggers one additional startup with `--enforce-eager`. If that request succeeds, `vllm_mode` is `enforce_eager`. For a separate fallback comparison, pass `--vllm-enforce-eager` and select a new output file. Its rows remain available with an explicit nondefault configuration and cannot qualify a performance claim against default vLLM.

On startup failure the report records `startup_stage`, `startup_reason` and the child exit status when available. `qualification` lists the generated and stock GGUF gate attempts that the server reported, including allowlisted failure codes and numeric checks. For vLLM, `traceback_summary` contains recognized exception classes, known package components and source lines. Complete child logs stay in the comparison log directory.

When GGUF startup fails, the readiness report includes the qualification stage, a bounded error code, and any aggregate scores available for the attempted configuration. If a generated kernel fails its tuning or quality check, normal serving runs the full stock-kernel qualification once. Only a passing configuration can serve. A successful fallback appears in `/health` under `ggml.kernel_fallback`.

On CUDA, IronMule checks whether the loaded GGML backend can load a KG1 kernel library. With an upstream backend that lacks the hook, it goes directly to stock-kernel qualification and records the reason in `/health` under `ggml.kernel_generation`. For a first run on a new GPU or model, `--startup-timeout 1200` gives each server up to 20 minutes within the overall `--budget-seconds` limit. Use the same `--agent-dir`, GGUF and device selection for readiness and measurement to reuse completed qualification checkpoints.

The NVIDIA comparison selects the first GPU visible through the caller’s CUDA mask and records its full UUID. vLLM 0.11.0 needs a numeric `CUDA_VISIBLE_DEVICES` value for its NVML lookup. Before its server starts, a fresh CUDA driver process checks that the proposed numeric selector exposes exactly the same UUID under the existing `CUDA_DEVICE_ORDER`. If CUDA and NVML enumerate the selected device differently, that arm reports a selection error.
