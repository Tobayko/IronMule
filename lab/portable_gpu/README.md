# Portable GGUF measurements

The width benchmark runs IronMule's in-process greedy GGML engine with a pinned llama-cpp-python 0.3.16 build, NumPy, Jinja2 and typing-extensions. Supply an installed binding or the `IRONMULE_GGML_BINDINGS` and `LLAMA_CPP_LIB_PATH` from the build tool. Select a single GPU with its native visibility environment. Model downloads are explicit and checked against a pinned revision, byte size and SHA-256.

```sh
python lab/portable_gpu/fetch.py smol --dir /models
python lab/portable_gpu/bench.py --gguf /models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --provider cuda --stage smoke --budget-s 60 --output smol-smoke.json
python lab/portable_gpu/bench.py --gguf /models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --provider cuda --stage compare --budget-s 180 --output smol-compare.json
```

Use `--provider rocm` with a HIP-enabled binding. `--provider cpu` exercises the same measurement protocol locally. Smoke checks a device-local arithmetic graph and records the model loader’s actual GPU layer count. It measures model/context startup including internal warmup after model hashing, then the first and second external requests. The operating system file cache may be warm. Inspect `smoke_passed` and the returned words before interpreting results.

Compare allocates widths 1 and 4 on one loaded model. Both receive equal greedy requests with cleared KV, identical prompt IDs, per-sequence context and token limits. Each repeat alternates the order of the arms. Default shapes are 32 or 256 prompt tokens and one or four requests. `--long-prompt 1024` selects a larger document. Each batch records generated token IDs, decoded text, correct first-word copying, aggregate generated tokens per elapsed second (including EOS), latency to the first generated token for every request, and the median latency per repeat. Output throughput includes prompt processing. The strict quality flag requires correct copied answers and identical full generated IDs for all repetitions and both arms.

Calibration and holdout use disjoint target words. A shape selects width 4 on calibration only when its median throughput improves by at least 2%, median first-token latency is no higher, and all quality checks pass. Otherwise selection stays at 1. `--transition` adds a short, four-request document burst, and short replay at the end. It compares the fixed width with the calibration selection on fresh copy targets in both orders. One engine serves each replay and real context rebuilding/warmup counts against the receiving wave. The report includes the number of switches and complete per-wave responses. Completed pairs are checkpointed on disk. A deadline or worker failure leaves `status=incomplete`.

The pinned public GGUFs all use Q4_K_M. Run the smallest smoke first. Download sizes use decimal units.

| Fetch alias | Model | Download |
| --- | --- | ---: |
| `smol` | SmolLM2-135M-Instruct | 105 MB |
| `qwen-small` | Qwen2.5-0.5B-Instruct | 398 MB |
| `qwen3-06b` | Qwen3-0.6B | 484 MB |
| `tinyllama` | TinyLlama-1.1B-Chat-v1.0 | 669 MB |
| `granite-2b` | IBM Granite 3.3 2B Instruct | 1.55 GB |
| `phi35-mini` | Microsoft Phi-3.5-mini-instruct | 2.39 GB |
| `qwen3-4b` | Qwen3-4B | 2.50 GB |
| `mistral-7b` | Mistral-7B-Instruct-v0.3 | 4.37 GB |
| `qwen3-32b` | Qwen3-32B | 19.76 GB |
| `qwen25-32b` | Qwen2.5-32B-Instruct | 19.85 GB |

For a 32B run, fetch once to the local model volume and reuse the path for smoke and comparison on the same pod.
Allow at least 30 GB of free model-volume space for one pinned file, or 50 GB to retain both.
The downloader verifies SHA-256 while streaming to a temporary file and atomically renames the verified download.

Both 32B GGUFs have 64 layers, eight KV heads per layer and 128 dimensions per head. Their default FP16 keys
and values require 256 KiB per cached token per sequence. At `--long-prompt 1024`, the width comparison keeps
1- and 4-slot contexts of 1280 tokens per slot, totaling 1.56 GiB of KV in addition to roughly 18.4–18.5 GiB
of model weights. Allow further VRAM for compute buffers and the backend. A single 48 GB or larger GPU is a
conservative starting point. Check the smoke report for full layer offload before interpreting timings.

For a Qwen3 HTTP comparison with vLLM, download the matching Hugging Face tokenizer with
`python -m ironmule_product.compare_tokenizer qwen3-32b --dir /models/tokenizers` or use `qwen3-06b` for the
smaller model. Pass the printed subdirectory as the tokenizer path. Each snapshot pins the upstream revision
and the SHA-256 of the configuration, vocabulary, merges, tokenizer and chat template configuration. Each
tokenizer download transfers about 15.9 MB.

## Stock server comparison

The optional HTTP comparison requires the full product runtime, a stock llama-server from the matching llama.cpp revision and its library directory on the loader path. Both servers run sequentially behind loopback on the same model and hardware. The normal product startup includes its quality gate. Reuse a dedicated state directory for later runs on that machine. Choose a fresh output directory.

```sh
python lab/portable_gpu/http_pair.py --provider cuda --gguf /models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --stock /opt/llama/llama-server --state-dir /tmp/ironmule-http-state --out results/http-1 --budget-s 300
```

Both arms serve the same greedy streamed chat requests, four slots of 2048 tokens, two CPU threads and the same per-request output limit. IronMule uses its gated cache settings and fixed-width scheduling. The stock server uses its normal cache implementation. Questions cover countries and capitals with and without a synthetic document. The harness checks each server's rendered prompt token IDs, exact visible text, and city-answer correctness. It counts visible output with the same GGUF tokenizer after timing, streams client first-text latency, and saves complete arm results alongside their ratios. `--first product` reverses process order. The width replay and the HTTP comparison write separate reports for their respective timing boundaries.

## Transfer to a held-out GPU model

Import completed comparison reports into the same local state directory, then evaluate the controller offline.

```sh
ironmule learn --import-gpu-report comparison.json --state-dir /path/to/state
python lab/portable_gpu/transfer.py --state-dir /path/to/state --output transfer.json
```

Each fold holds out every machine with one GPU model description. The fit uses matched, repeated calibration
pairs from other GPU models with the same GGUF SHA-256 and backend. Identical trial IDs count once. Target-device
calibration never enters training or selection. For each repeated target holdout, the hypothetical controller keeps
one slot unless it predicts at least 2% higher throughput with first-token latency no higher at the wider setting.
`predicted` and `actual` are natural log ratios (wide / single), throughput first, then first-token latency.

Mean absolute errors use quality-matched holdouts only. The equal-cost predictor assigns zero to both log ratios.
The corpus average weights each such holdout pair equally. The report also counts cases where a blind switch would
have selected a wider width despite a failed quality check, or would have lost throughput or delayed the first token
on matched output. Keeping one slot on a failed quality case supplies no evidence of baseline correctness. Folds
with no eligible source data are listed separately. These offline predictions leave the server's local
remeasurement and qualification requirements in force.

## GGUF controller decisions during load changes

Compare the production idle decision rule with an explicitly enabled diagnostic candidate that prices GPU
width changes. Both arms receive the same measured calibration and record completed switch costs. The probe calls the normal agent decision
between waves and charges its elapsed time and any context rebuild to the arriving requests.
The regular server waits for an idle interval. Inspect wave-level quality, exact output,
selected widths, switch timings and complete-run wall time before comparing policies.

```sh
python lab/portable_gpu/switch_probe.py --gguf /models/Qwen3-4B-Q4_K_M.gguf \
  --provider cuda --long-prompt 1024 --repeats 2 --budget-s 240 --output switches.json
```

### Read-only browser view for a remote comparison

Build the view into the job's bootstrap gateway before starting the provider job.
The generator accepts the existing Python gateway template with a `Handler.do_GET`,
a `ROOT` result directory and a `TOKEN` for private operator requests.

```sh
python -m lab.portable_gpu.watch_gateway gateway_template.py --output gateway_watch.py
```

Launch the generated gateway with a new random `WATCH_ID` from
`secrets.token_urlsafe(24)` and a separate `RESULT_TOKEN`. Share
`https://<gateway>/watch/<WATCH_ID>/` with the viewer. The page and its relative
`data` route work before the benchmark bundle has been installed. They read the
job's fixed Qwen3-0.6B and Qwen3-32B reports and the bootstrap status from
`ROOT`. The view exposes aggregate scores, timings, model names and setup events.
The original gateway continues to authenticate its private result and upload routes.

### Production deployment comparison on one NVIDIA GPU

Use this separate protocol to compare autonomous IronMule serving the official Qwen3-32B Q4_K_M GGUF with vLLM 0.11.0 serving the official Qwen3-32B-AWQ Safetensors checkpoint. The two deployments use different quantized weights. vLLM selects its backend automatically and the report records the resolved quantization method and dtype from the running engine. The same-GGUF benchmark and the hardware controller's competitor-response training consume a different report schema.

```bash
python -m lab.portable_gpu.fetch qwen3-32b --dir /models
python -m ironmule_product.compare_tokenizer qwen3-32b --dir /models/tokenizers
python -m lab.portable_gpu.fetch_awq /models/Qwen3-32B-AWQ
python -m lab.portable_gpu.production_compare \
  --gguf /models/Qwen3-32B-Q4_K_M.gguf --awq-dir /models/Qwen3-32B-AWQ \
  --tokenizer /models/tokenizers/qwen3-32b --vllm /usr/local/bin/vllm \
  --concurrency 1,8,32 --requests 8 --repeats 3 --context 2048 --max-tokens 32 \
  --budget-seconds 2700 --startup-timeout 1200 --agent-dir /results/awq-agent \
  --log-dir /results/awq-logs --output /results/qwen3-32b-deployments.json
```

The fetches are public and pinned. The GGUF is 19,762,149,024 bytes at commit `938a7432affaec9157f883a87164e2646ae17555`. The AWQ snapshot is 19,341,507,104 bytes across twelve files at `0499c3ac83fdef8810b907a23894ba91e95eddd8`. The runner verifies each weight shard and the tokenizer before timing. Allow space for both formats, server logs and compiler caches. The benchmark budget covers local verification, startups and traffic. Apply a separate outer wall-clock limit if provisioning and downloading count towards a rental budget. [GGUF release](https://huggingface.co/Qwen/Qwen3-32B-GGUF/tree/938a7432affaec9157f883a87164e2646ae17555) · [AWQ release](https://huggingface.co/Qwen/Qwen3-32B-AWQ/tree/0499c3ac83fdef8810b907a23894ba91e95eddd8)

Each repeat alternates server order on the same GPU UUID. Both arms receive matching client request IDs, identical post-template input token IDs, the same greedy sampling settings and 18 short-answer holdouts at each concurrency. Use `--seed` to replay the workload. Each verified pair reports output tokens/s, requests/s, p95 time to first token and per-answer quality changes. `by_concurrency` gives medians of paired ratios in IronMule/vLLM order. When the output token counts or stop reasons differ, `output_tokens_and_stops_match` is false. Inspect requests/s and answer quality alongside token throughput in those cells. A failed identity check or server startup preserves completed rows and stops the run. The output path must be new.

For the Qwen3-32B production comparison, set `WATCH_KIND=deployment` before
starting the gateway. Direct the deployment runner's output to
`/results/deployment-32b.json` and include this exact path in the gateway's
authenticated result allowlist for the private collector. The public page reads
that report separately from the same-GGUF reports. It shows the IronMule GGUF
Q4_K_M and vLLM AWQ deployment measurements and the matched pairs verified by
the runner. Pair ratios divide IronMule's measurement by vLLM's. Each pair also
shows the holdout scores and whether output lengths and stop reasons match.
