<div align="center">

  <img src="docs/assets/ironmule-badge.jpg" alt="IronMule logo" width="190">

  <a id="ironmule"></a>
  <h1>IronMule · Local LLM inference server</h1>

  <p><strong>Run a language model on your own hardware. Give it a server that learns how your machine runs it best.</strong></p>

  <p>Chat in your browser or connect an app through the OpenAI-compatible API. IronMule streams the answers. A small CPU model called the Pilot helps decide how to serve the next request.</p>

  <p>
    <a href="https://github.com/Tobayko/IronMule/actions/workflows/ci.yml"><img src="https://github.com/Tobayko/IronMule/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <img src="https://img.shields.io/badge/Python-3.10+-38bdf8" alt="Python 3.10+">
    <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-Apache_2.0-facc15" alt="License: Apache 2.0"></a>
    <img src="https://img.shields.io/badge/CPU-llama.cpp_kernels_%2B_own-0ea5e9" alt="CPU: llama.cpp kernels plus IronMule's own">
    <img src="https://img.shields.io/badge/Apple_Silicon-Metal-ff9100" alt="Apple Silicon, Metal">
    <img src="https://img.shields.io/badge/NVIDIA-CUDA_kernels-76b900" alt="NVIDIA CUDA kernels">
    <img src="https://img.shields.io/badge/Pilot-open_weights-8b5cf6" alt="Pilot: open weights">
  </p>

</div>

[Quick start](#quick-start) · [Hardware support](#hardware-support) · [Benchmarks](#how-much-faster) · [How the Pilot learns](#the-pilot) · [HTTP API](docs/HTTP.md)

## What changes when the server learns?

A quiet CPU can give one chat all its attention. Start a compile, or send a batch of long documents, and yesterday’s thread count can become today’s bottleneck. IronMule measures the machine it runs on. Its Pilot uses those timings to choose how to execute requests while your language model keeps its original weights.

**Try a local chat.** On Apple Silicon or a supported Linux NVIDIA setup, install IronMule and open its local chat interface with these two commands.

```bash
curl -fsSL https://raw.githubusercontent.com/Tobayko/IronMule/main/install.sh | sh
ironmule start
```

Already have a GGUF file? You can serve it on a Linux CPU and send your first streaming request with the commands below. [GGUF quick start](#serve-a-gguf-on-linux-cpu) · [Hardware support](#hardware-support)

### What the Pilot found on an A100

Two starts of the Qwen3-32B server spent almost 19 seconds each on one calibration. In the development loop, IronMule picked that step for investigation and asked Codex for a change. The new code still tried all five settings on the GPU. When the measurements agreed, it could finish in 15 calls instead of 20. Across six paired cold starts, the first 32 complete answers arrived in a median **51.06 seconds**, down from **55.29**, with identical output tokens. The trained Pilot selected the new code on fresh work, and a rollback restored the original worker. This result belongs to the ordered native worker tested in PR 57 on a RunPod A100 SXM4 80 GB. [Experiment, raw results and replay](https://github.com/Tobayko/IronMule/blob/4b6971e5ba59f0fb74bf56cf7a4249b5d88fb9d5/lab/pk6/local-calibration.md)

**How does a warm server compare?** In a separate HTTP test at 32 concurrent clients on that A100, IronMule produced **297.54 tokens/s** with Qwen3-32B Q4_K_M GGUF. vLLM 0.11.0 produced **271.26 tokens/s** with Qwen3-32B AWQ. That gives IronMule 9.69% more output per second across the group. vLLM delivered the first token sooner, with a median of 1.71 seconds versus IronMule’s 2.41. [PR 57 measurements and per-request timings](https://github.com/Tobayko/IronMule/tree/4b6971e5ba59f0fb74bf56cf7a4249b5d88fb9d5/lab/autonomous/results/a100-pr57-warm-repeat)

### What changes on a smaller GPU?

On an NVIDIA L40S, the trade-off changes with the queue. With one client IronMule delivered **43.3 output tokens/s** and vLLM 0.34.0 delivered **37.7**. With 32 clients IronMule started streaming in a median **0.56 seconds**, versus **1.68 seconds** for vLLM. Across the full group vLLM finished faster at **266.6 vs 154.7 tokens/s**. Both served Qwen3-32B with the same inputs and fixed output length, using Q4_K_M and AWQ-Marlin respectively. These are three repeats from the first matched seed. [Per-request measurements, quality checks and replay](lab/pk6/mixed_progress/replay.md)

IronMule can also tell you which optimization it declined. On a second L40S run, it traced a slow GPU operation back to the requests waiting for it and issued a Q4_K kernel experiment to Codex. A separate FP8 kernel completed the short factual checks, but missed the model's perplexity limit. The server returned to its verified profile. You can replay the decisions and inspect the same fixed-length streams on a CPU. [L40S kernel experiments and replay](lab/pk6/native_dispatch/README.md)

**What happens when 32 people send long documents at once?** On one L40S, a smaller KV cache freed **12 GiB during requests**. On two sets of 900-word documents it raised throughput from **24.78 to 47.78** and from **27.49 to 47.55 tokens/s**. Both cache sizes found all 64 six-digit seals hidden among distractors in a separate retrieval test. The trained CPU Pilot chose that cache on a fresh server and delivered **47.50 tokens/s**. A tested rollback restored the original cache and delivered **29.04 tokens/s** on the same requests. Initial qualification peaked at 45,160 MiB. [Document results, memory measurements and CPU replay](lab/pk6/native_dispatch/README.md)

With one client and an experimental 0.6B draft model, IronMule finished each of six matched 32-token requests faster than vLLM 0.34.0 AWQ-Marlin. Median paired throughput was **1.50× and 1.53×** over the two seeds. vLLM still started streaming sooner. At 32 clients vLLM retained its throughput lead. [Per-load results and vLLM reference](lab/autonomous/results/l40s-native-dispatch/comparison.json)

## Quick start

### Serve a GGUF on Linux CPU

Use Python 3.10 or newer, a C/C++ toolchain, CMake, Git and `patch`. These commands set up a server from a checkout using Python 3.12. Have a GGUF with a chat template available locally.

```bash
git clone https://github.com/Tobayko/IronMule.git
cd IronMule
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cpu]'
ironmule setup --mode server
python tools/build_ggml.py
```

Run the two `export` commands printed by the build in this shell, then start the server with the path to your model. This example uses Qwen3-4B.

```bash
ironmule serve --gguf /path/to/Qwen3-4B-Q8_0.gguf --agent on
```

The first start measures the machine and checks the inference configuration. Open `http://127.0.0.1:8080` for the chat interface. From a second shell, list the available model and send a streaming chat request. The model ID consists of `gguf/` and the filename without its extension.

```bash
curl -sS http://127.0.0.1:8080/v1/models

curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gguf/Qwen3-4B-Q8_0","messages":[{"role":"user","content":"What is continuous batching?"}],"stream":true}'
```

To let a Linux server choose between qualified CPU and GPU workers, add `--placement auto` to the GGUF serving command. IronMule learns from local timings between requests and checks answer quality and first-token latency before using a learned route. [Hardware placement guide](training/hardware-placement.md)

### Run an MLX model

To choose and serve an MLX model explicitly, install from source with `python -m pip install -e .` on Apple Silicon or `python -m pip install -e '.[cuda]'` on supported NVIDIA hardware. Then register a model and serve it locally.

```bash
ironmule doctor
ironmule setup --mode server
ironmule models add mlx-community/gemma-3-1b-it-4bit --download
ironmule serve --model mlx-community/gemma-3-1b-it-4bit
```

See the HTTP documentation for supported OpenAI-compatible routes, authentication and TLS. [HTTP API](docs/HTTP.md)

## Hardware support

| Hardware | Execution path and evidence |
| --- | --- |
| Linux x86-64 CPU | GGUF through llama.cpp with IronMule scheduling and optional CPU kernels. Published tests include Intel Xeon and an AMD EPYC HTTP workload. MLX also offers a Linux CPU build. |
| NVIDIA GPU | GGUF through llama.cpp's CUDA build plus IronMule's own CUDA work (tensor-core 4-bit matmul kernels tuned on the GPU itself, packed prompt prefill through cuBLAS, CUDA graphs for multi-request decode steps, and a fused residual+RMS-norm kernel), or MLX with its CUDA backend. HTTP comparisons include NVIDIA A100, L40S, GeForce RTX 4090 and Tesla T4. |
| Apple Silicon | MLX with Metal. Published results include the M1 Max. Current CI runs on Linux. |
| AMD Radeon GPU | Experimental ROCm/HIP build for GGUF. With the ROCm toolkit, `python tools/build_ggml.py --rocm` detects the architecture. To select a target explicitly, use e.g. `--rocm gfx1100`. After exporting the library paths, `ironmule doctor --ggml rocm` checks device execution. Radeon serving and throughput still need a hardware run. |
| Google TPU | Research probes measured matrix operations on a TPU v5e. An experimental JAX tool can calibrate a decode loop with synthetic weights. Full checkpoint inference through the HTTP server remains open. [TPU calibration](docs/TPU.md) |

Tested model families include Qwen 3, Gemma 3, Llama 3 and Mistral. Backend, model format, precision and hardware all matter when comparing results. [Runtime](docs/RUNTIME.md) · [Limits](docs/LIMITS.md) · [Hardware measurements](docs/BENCHMARKS.md)

## How much faster?

### Qwen3-32B on one A100 against vLLM

> **Handover:** the current A100 state, defaults, reproduction, evidence and open points are in [docs/HANDOVER_A100.md](docs/HANDOVER_A100.md).

On one RunPod A100 SXM4 80 GB, IronMule served the Q4_K_M GGUF of Qwen3-32B through `ironmule serve --agent on`, and vLLM 0.11.0 served the official Qwen3-32B-AWQ checkpoint with AWQ-Marlin on the same pod, one after the other. Both servers used the same HTTP client, prompts and seeds, with 32-token answers (2026-11-25, pod 37; IronMule's default build over two server starts, every wave counted).

| Concurrent clients | IronMule, Q4_K_M GGUF (output tok/s, every wave) | vLLM 0.11.0, AWQ-Marlin |
| ---: | ---: | ---: |
| 8 | median 195.6 (four waves) | 189.4 (median of two) |
| 32 | 297.1 / 313.2 / 306.0 / 297.3 / 303.2 / 307.5 | 278.7 / 276.7 / 276.9 |

- At 32 clients IronMule is 10.0% ahead (median 304.6 against 276.9 tok/s), just at the +10% goal (per server start 10.5% and 9.5%). At 8 clients IronMule leads by 3.3%. vLLM gives its first tokens earlier (median 1.8 against 2.3 s at 32 clients). With the default of the time, pod 36 gave +9.2% (PS1 without PK6) and pod 35 +7.7% (PS1 off).
- Both servers answered 18/18 factual check questions in every run, and IronMule passed its own perplexity quality gate on this machine.
- On an earlier pod (pod 17) the two were level at 32 clients (292.2 against 292.6) and IronMule was level to 93% ahead for one client, depending on the prompt (it drafts from the context). [Pod 37](docs/LAB.md#2026-11-25-a100-pod-37-pk6-kept----3046-against-2769-toks-at-32-clients-1002--the-10--reached-by-its-rule) · [Pod 36](docs/LAB.md#2026-11-25-a100-pod-36-ps1s-slow-prefills-attributed-and-removed-ps1a-ps1-on-by-default-92--against-vllm) · [Pod 35](docs/LAB.md#2026-11-25-a100-pod-35-ps1-measured--90-ms-a-wave-in-one-process-outliers-over-http-rejected-by-the-rule-dh2-no-gain-77--against-vllm) · [Pod 34](docs/LAB.md#2026-11-25-a100-pod-34-adm1-and-adm2-kept----3121-against-2934-toks-at-32-clients-64--10--not-reached)

On an earlier pod, before PK5, IronMule was 5% behind at 32 clients (252.5 against 265.9); pods differ by several percent for both servers, so only same-pod comparisons count. [Same-pod measurements](docs/LAB.md#2026-11-23-a100-speculative-decoding-with-a-batch-draft-model-sd1-and-vllm-on-the-same-pod-with-pk5) · [Earlier pod](docs/LAB.md#2026-11-22-a100-pk4-served-and-vllm-on-the-same-pod) · [Earlier A100 runs](docs/LAB.md#2026-11-22-a100-gr3-served-fu1-32-row-cuda-graphs-through-the-gate----level-with-vllm-at-32-clients)

To compare on your own GPU, `ironmule compare` serves the same GGUF through IronMule,
stock llama-server and vLLM. It checks the input tokens, sends short and document
requests and measures output tokens/s, first-token latency and answer quality at each
concurrency. `ironmule compare-watch` repeats comparisons on a dedicated host within
a time allowance. The production controller chooses its own trials from local workload
measurements and tests each candidate against its own serving configuration. A CPU run
with SmolLM2 served 40 requests and selected an admission budget autonomously in an
isolated serving environment with the model and IronMule dependencies. Both settings solved 16/16 copy
tasks. One full output differed, so the worker kept its original setting.
[Self-directed trial](docs/LAB.md#2026-11-18-own-traffic-admission-learning-on-an-amd-epyc-cpu)
[Run the comparison](docs/COMPARE.md)

During inference, open `/live` on your IronMule server to see active and queued requests,
recent first-token times and controller decisions. The **Benchmarks** tab shows the
current engine, load and measured results. To watch a separate comparison file, run
`python -m ironmule_product.live_view --progress ./comparison.json --port 8787`
and open `http://127.0.0.1:8787/live#benchmarks`.


### CPU serving against llama.cpp and Ollama

The following results use the same 4-vCPU Intel Xeon, Qwen3-4B Q8_0 GGUF and questions through each server's HTTP API. The runs took place on November 13–14, 2026, once per engine. Aggregate throughput counts generated tokens across all clients. Time to first token measures the wait before a user starts seeing an answer.

| What happens | IronMule | llama.cpp server | Ollama | LM Studio |
| :-- | --: | --: | --: | --: |
| 32 people ask at once (tokens per second, all together) | **87.5** | 28.9 | 21.8 | -- |
| One person asks (tokens per second) | **10.1** | 8.0 | 8.6 | 6.1 |
| A question every 2 seconds: wait for the first word (median) | **0.33 s** | 113 s | 47 s | -- |
| 16 people in long conversations (tokens per second) | **47.8** | 13.0 | -- | -- |
| Fact check, 50 questions (right answers) | 49 | 49 | 49 | 49 |

IronMule, llama-server, Ollama and LM Studio each answered 49 of the 50 fact questions correctly. The burst comparison used 32 llama-server slots. At 8 and 16 clients, sizing llama-server to each burst produced 27.3 and 40.9 tok/s, compared with IronMule's 50.4 and 71.3. Ollama used 16 parallel slots after its 32-slot configuration ran out of memory. [Lab notes, CMP1](docs/LAB.md#2026-11-13-cmp1-ironmule-against-llamacpp-ollama-and-lm-studio-on-this-cpu-same-gguf)

A separate three-round test used Qwen3-0.6B Q4_K_M on two AMD EPYC virtual cores.
With four concurrent clients the medians were **4.95 vs 3.52 output tokens/s** for
IronMule and stock llama-server. IronMule had higher throughput in all six pairs,
but one pair regressed in first-token latency and one lost a previously correct
answer. It did not meet the comparison's overall acceptance criteria.
[Measured loads, latency and answer scores](docs/LAB.md#2026-11-18-qwen3-06b-on-two-epyc-vcpus-against-upstream-llama-server)

<details>
<summary>CPU performance when other programs compete for cores</summary>

| Busy foreign processes on the 4 cores | llama.cpp server, 1 user | IronMule, 1 user | llama.cpp server, 8 users | IronMule, 8 users |
| :-- | --: | --: | --: | --: |
| none | 7.23 tok/s | **9.33 tok/s** | 23.3 tok/s | **58.8 tok/s** |
| one | 0.24 tok/s | **7.07 tok/s** (29x) | 1.41 tok/s | **46.2 tok/s** (33x) |
| two | 0.13 tok/s | **5.46 tok/s** (42x) | 0.68 tok/s | **35.2 tok/s** (52x) |

With one competing process, time to first token was 23 s for llama-server at one client and 54 s at eight. IronMule measured 0.3 s. The original Pilot chose four, three and two threads at the three load levels. Greedy output was identical at four, three, two and one threads on the checked inputs. The tests used one run per configuration with collective speculation disabled. [Lab notes, PI1 and PI2](docs/LAB.md)

</details>

### Same-GGUF serving on NVIDIA L40S

With Qwen3-32B Q4_K_M and **32 concurrent requests** on a 48 GB L40S, IronMule
delivered **174.5 output tokens/s**, stock llama-server **130.4**, and vLLM 0.11.0
serving the identical GGUF **27.2**. Their median per-round p95 first-token times
were 3.96, 4.53 and 32.72 seconds. All three produced 162/162 correct short answers
across the three loads and three repetitions. IronMule met the predeclared acceptance
criteria against both reference servers in every measured pair.
[32B protocol, all loads and scored answers](docs/LAB.md#2026-11-18-qwen3-32b-on-nvidia-l40s-identical-gguf)

On the same L40S running Qwen3-0.6B Q4_K_M, IronMule served **929 output tokens/s**
at eight concurrent clients. Stock llama-server delivered **717** and vLLM 0.11.0
delivered **772**. At 32 clients vLLM led with **1,285**, followed by IronMule at
**1,248** and llama-server at **1,124**. These are medians of three repetitions with
identical tokenized requests. IronMule missed the overall latency and quality criteria.
[Full results and holdout scores](docs/LAB.md#2026-11-18-qwen3-06b-on-nvidia-l40s-against-llamacpp-and-vllm)

A second three-server run on an RTX 4090 found the same load-dependent result for Qwen3-0.6B. IronMule led at eight clients with 907 output tokens/s. At 32 clients vLLM led with 1,189 to IronMule’s 1,176. An earlier lab experiment used the measured first-token gap to trigger a scheduling trial. IronMule kept the incumbent after four held-out pairs. [RTX 4090 measurements and learning decision](docs/LAB.md#2026-11-18-qwen3-06b-on-geforce-rtx-4090)

### Single-GPU inference against vLLM

A separate matched deployment test on a 48 GB L40S used Qwen3-32B Q4_K_M in IronMule
and the official Qwen3-32B-AWQ checkpoint in vLLM 0.11.0, which selected AWQMarlin.
With one client, median output throughput was **37.9 vs 32.9 tok/s**. At eight clients
it was **107.4 vs 143.2**, and at 32 clients **165.0 vs 225.1**. Both servers solved
162/162 short holdouts. vLLM's median p95 time to first visible token was lower at
each load. All nine paired rounds matched on input tokens, output lengths and stop
reasons. [Full deployment comparison](docs/LAB.md#2026-11-18-qwen3-32b-gguf-and-native-awq-on-one-l40s)

On one Tesla T4, the IronMule KG4 engine and vLLM served Qwen3-4B. The table reports aggregate output tokens per second. IronMule used Q4_K_M GGUF, and vLLM used the AWQ model. The values come from separate benchmark sessions on this GPU type. A matched run through both current HTTP servers would be needed for a new end-to-end comparison.

| Concurrent requests | IronMule GGUF (KG4) | vLLM AWQ (MG2 v6) |
| ---: | ---: | ---: |
| 1 | 56.3 | 63.7 |
| 8 | 310.4 | 419.2 |
| 32 | 626.9 | 860.5 |

At eight clients the generated CUDA kernels raised IronMule's own throughput from 249.2 to 310.4 tok/s. At 32 clients the change was smaller, from 602.2 to 626.9. Both versions answered 49 of 50 fact questions correctly. vLLM led at all three concurrency levels. [IronMule KG4 results](docs/LAB.md#2026-11-17-kg4-where-a-gpu-step-of-8-32-requests-goes-and-a-q4_k-tensor-core-kernel-for-1-32-tokens) · [vLLM MG2 v6 results](docs/LAB.md#2026-11-15-mg2-v6-the-shared-pools-padded-answers-on-a-gpu)

An experimental T4 kernel targets grouped-query attention at 32 concurrent streams and 1,024 KV rows. Its benchmark first checks the CUDA calculation, then requires at least a 10% shorter decode step before testing the HTTP server, factual answers and first-token latency. A speedup from this kernel has yet to be measured on a T4. [KG5 benchmark](lab/kg5/README.md)

<details>
<summary>GGUF results across four model families on one and two Tesla T4 GPUs</summary>

This earlier comparison used identical Q4_K_M GGUF files in each engine and 128 tokens per answer. MG2 v5 ran three blocks per configuration, and v6 repeated the measurements within a few percent. The IronMule configuration used a shared KV pool and padded decode batches.

| | Qwen3-4B | Gemma 3 4B | Mistral 7B | Llama 3.1 8B |
| :-- | --: | --: | --: | --: |
| 8 requests: best llama.cpp server setup (some use both GPUs) | 193.5 | 195.2 | 128.0 | 119.1 |
| 8 requests, one GPU: IronMule | **270.7** | **263.6** | **175.1** | **157.7** |
| 32 requests, one engine per GPU: llama.cpp server | 786.0 | 713.2 | 614.1 | 544.1 |
| 32 requests, one engine per GPU: IronMule | **1026.9** | **1007.3** | **679.1** | **633.2** |

Every engine comparison above is in aggregate tok/s. Each model answered the same 47 to 49 facts correctly as llama.cpp in its corresponding run. An additional HTTP test on Qwen3 changed the GPU KV layout from a shared pool to a stream per request. With 32 clients throughput rose from 422 to 518 tok/s. At eight clients it fell from 260 to 238. [Lab notes, MG2 and KG3](docs/LAB.md)

</details>

<details>
<summary>Apple Silicon and NVIDIA GPUs (MLX path)</summary>

On MLX, IronMule's exact mode returns the same tokens as stock MLX, checked token by token before use:

| Model (4-bit) | Apple M1 Max | NVIDIA Tesla T4 | T4, opt-in `--compute-dtype float32` |
| :-- | --: | --: | --: |
| Gemma 3 1B | **1.61× · +61%** | **1.82× · +82%** | — |
| Gemma 3 4B | **1.27× · +27%** | 1.05× · +5% | **1.91× · +91%** |
| Gemma 3 12B | **1.11× · +11%** | 1.03× · +3% | **2.04× · +104%** |

Each device is compared with stock MLX on that device. The float32 column uses different arithmetic and requires a quality check. Results for six more model families, native kernels and server batching are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

</details>

## The Pilot

IronMule trains its own inference scheduler. The Pilot reads 20 features, including available CPU cores, competing load, measured step costs, active and waiting requests, prompt lengths and recent step times. Two hidden layers of 128 units and four output heads select the thread count, batch cap, prompt budget and use of a draft model. A decision took approximately 0.17–0.19 ms inside the measured Xeon server and 0.11 ms beside a T4.

The published CPU and GPU versions began with random weights, learned from a short teacher warm-up, and trained with reinforcement learning (PPO) on simulated servers. These “dreams” replay workloads against measured hardware costs. For the shipped CPU Pilot, two T4s simulated 8,192 servers at once during a 47-minute training run. The trained weights and inference code are public. The full training simulator and recipe are private. [CPU weights](ironmule/pilot_weights.npz) · [GPU weights](ironmule/pilot_weights_gpu.npz) · [Pilot implementation](ironmule/pilot.py)

**Learning on the serving machine.** Start a Linux CPU server with `ironmule serve --gguf MODEL.gguf --agent on`. During idle periods the agent can replay recent traffic. A local cost model starts at zero, learns from faster runs, slower runs and token mismatches, and adds weights as it encounters new hardware and workload features. It ranks thread counts to test for one to four concurrent requests. Before a new setting can serve, repeated trials on the same machine must reproduce the tokens and improve decode time. After a restart, the agent checks the setting again when matching traffic returns. Set `IRONMULE_PILOT_LEARN=0` to disable local thread learning. [Operating guide](docs/AGENT.md) · [Two-request CPU experiment](docs/LAB.md#2026-11-18--from-scratch-hardware-model-two-concurrent-decodes)

**What happened when the traffic changed?** On a four-vCPU AMD EPYC, 16 clients switched from short chats to long documents and back. The agent predicted that widening the batch from 8 to 64 slots would help. Live throughput fell, and the updated controller reverted after 16 completions. It kept the recovered setting through all three returning chat waves. Those chats reached 50.77 output tok/s with the agent and 78.68 with the fixed setting. Each arm answered 138 of 144 factual requests correctly. This was one run per arm. [Workload, timings and earlier attempts](docs/LAB.md#2026-11-18--earlier-ag10-withdrawal-and-recovery-after-a-failed-live-switch)

A later SmolLM2-135M CPU run tested whether the controller could act on its own
workload forecasts. It selected eight slots, served 64 requests and returned to
four after observing the higher cost. On 576 held-out requests, matching previous
traffic by workload reduced the controller's forecast error by 3.33%.
[Measurements and saved controller state](training/README.md#serving-the-proposed-width-and-testing-its-rollback)

The local answer-quality lab measures grounded responses, false refusals and completed
tool calls. In one Qwen3-0.6B test, passing an actual first lookup result to the next
tool completed 12 of 12 two-step tasks. The model-driven reference completed zero.
Both workflows answered all 12 supported lookups correctly. These results apply to
the structured lookup application and its tested schema.
[Tasks, model outputs and tool traces](lab/answer_quality/README.md)

## Where the speed comes from

- **Continuous batching** lets arriving requests join a decode batch while earlier requests are still generating.
- **KV cache reuse** shares work on common prefixes. Per-request KV streams can reduce GPU attention work when many independent chats are active.
- **Hardware-specific kernels** include AVX-512 and Intel AMX CPU paths, plus generated CUDA kernels for quantized GGUF matrix multiplication. IronMule times candidates on the target device.
- **Speculative decoding** lets a smaller model propose tokens for the target model to verify. On a T4, adding a draft model raised single-client Qwen3-4B throughput from 44.1 to 54.1 tok/s in the DM2 test.

See `GET /health` for the active engine and Pilot status. [Engine and Pilot](docs/AGENT.md) · [Experiments](docs/LAB.md)

## Can I trust the numbers?

To reproduce a comparison, record the model and quantization, hardware, concurrency, prompt and output lengths, and whether the measurement covers a kernel, a decode step or a complete HTTP request. Use the same request trace for both servers. Check throughput alongside time to first token and answer quality. The lab directory includes the HTTP comparison harnesses. The MLX benchmark tests also check published figures against committed evidence. [HTTP comparison harness](lab/cmp/bench.py) · [Benchmark methodology and results](docs/BENCHMARKS.md) · [Evidence](evidence/)

```bash
python -m pip install -e '.[dev,figures]'
python -m pytest tests/claims
```

## What is in this repository

| Folder | What it is |
| :-- | :-- |
| `ironmule/` | The runtime: the ggml engine and the Pilot, MLX generation, cache, plans, kernels |
| `ironmule_product/` | The server, chat page, the agent, calibration and worker processes |
| `native/` | IronMule's llama.cpp kernel patches and the Rust learning core |
| `lab/` | Experiments and the comparison harness behind the numbers above ([docs/LAB.md](docs/LAB.md)) |
| `evidence/`, `friday_evidence/` | Redacted results and their provenance |
| `training/` | Simulation and measurement of the agent ([how](training/README.md)) |
| `tests/` | Runtime, server, agent, learning, evidence and published-number tests |
| `docs/` | User documentation, index in [docs/README.md](docs/README.md) |
| `tools/` | Build, figure and evidence tools |

## Learn more

- **[The agent and the Pilot](docs/AGENT.md)** — how they decide and learn.
- **[Benchmarks](docs/BENCHMARKS.md)** · **[Lab notes](docs/LAB.md)** — every number, with its setup.
- **[HTTP API](docs/HTTP.md)** · **[Runtime](docs/RUNTIME.md)** · **[Limits](docs/LIMITS.md)** (rejected ideas: [docs/BACKLOG.md](docs/BACKLOG.md))
- **[Documentation index](docs/README.md)** · **[Research lab](https://github.com/Tobayko/IronMule-Research)**
- **[Contributing](CONTRIBUTING.md)** · [code of conduct](CODE_OF_CONDUCT.md) · [security policy](SECURITY.md)
- **[License](LICENSE.md)** — Apache 2.0. Website: [ironmule.prometo.app](https://ironmule.prometo.app/).
