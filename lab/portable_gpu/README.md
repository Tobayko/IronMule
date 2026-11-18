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

Pinned model aliases are `smol` (SmolLM2-135M-Instruct, 105 MB), `qwen-small` (Qwen2.5-0.5B-Instruct, 398 MB), `tinyllama` (TinyLlama-1.1B-Chat, 669 MB) and `qwen3-4b` (Qwen3-4B, 2.50 GB), all Q4_K_M. Run the smallest smoke first.

## Stock server comparison

The optional HTTP comparison requires the full product runtime, a stock llama-server from the matching llama.cpp revision and its library directory on the loader path. Both servers run sequentially behind loopback on the same model and hardware. The normal product startup includes its quality gate. Reuse a dedicated state directory for later runs on that machine. Choose a fresh output directory.

```sh
python lab/portable_gpu/http_pair.py --provider cuda --gguf /models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --stock /opt/llama/llama-server --state-dir /tmp/ironmule-http-state --out results/http-1 --budget-s 300
```

Both arms serve the same greedy streamed chat requests, four slots of 2048 tokens, two CPU threads and the same per-request output limit. IronMule uses its gated cache settings and fixed-width scheduling. The stock server uses its normal cache implementation. Questions cover countries and capitals with and without a synthetic document. The harness checks each server's rendered prompt token IDs, exact visible text, and city-answer correctness. It counts visible output with the same GGUF tokenizer after timing, streams client first-text latency, and saves complete arm results alongside their ratios. `--first product` reverses process order. The width replay and the HTTP comparison write separate reports for their respective timing boundaries.
