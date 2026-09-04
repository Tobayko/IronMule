<div align="center">
  <img src="docs/assets/ironmule-logo.jpg" alt="IronMule donkey mark and wordmark" width="960">
  <p><strong>MEASURE&nbsp;&middot;&nbsp;PROVE&nbsp;&middot;&nbsp;RUN</strong></p>
</div>

# IronMule

**Run and benchmark local LLMs on Apple Silicon with MLX.**

IronMule is a Python runtime for local LLM inference on a Mac. It helps you compare
low-latency and high-throughput execution, reuse shared prompt prefixes, measure time
to first token (TTFT), and keep a record of which MLX optimisations are both faster
and correct on your machine.

IronMule runs locally. It does not upload prompts, download models, or hide a cloud
service behind the API.

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-fair--code-111111?style=for-the-badge" alt="License: fair-code"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-111111?style=for-the-badge" alt="Python 3.10+"></a>
  <a href="https://github.com/ml-explore/mlx"><img src="https://img.shields.io/badge/platform-Apple%20Silicon-111111?style=for-the-badge" alt="Apple Silicon"></a>
</p>

> [!IMPORTANT]
> IronMule is designed for Apple Silicon, but the published performance evidence was
> measured primarily on one Apple M1 Max. It is not a universal M1–M4 speed claim.
> Read the full [validity limits](docs/LIMITS.md) before comparing results.

## What problem does it solve?

Local AI has two different goals:

- A chat wants the fastest possible answer for one person.
- A service with several waiting requests wants more total tokens per second.

One setting cannot maximise both. IronMule makes the choice explicit:

| Your workload | Use | What to expect |
| :-- | :-- | :-- |
| One chat or latency-sensitive request | `InteractiveMode` | Lowest single-request latency |
| Several requests at the same time | `ThroughputMode` | More total throughput; one request may take longer |
| Repeated questions about one shared document | `ReusableSessionPlan` | Reuses the declared prompt prefix |
| Very short answers | Start with `InteractiveMode` | Groups may not fill enough to help |

In throughput mode, requests remain independent batch-1 runs. IronMule submits work
from several requests together and waits once. It does **not** merge prompts into true
tensor batches, and the public benchmark fails when the two modes change the output.

## Quick start

You need Python 3.10+, an Apple Silicon Mac, MLX, and a compatible model already in
your local Hugging Face cache. The package is not currently published on PyPI, so
install it from a checkout:

```bash
git clone https://github.com/Tobayko/IronMule.git
cd IronMule
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Check the machine and list local models:

```bash
ironmule doctor
ironmule models
```

`ironmule models` prints an empty list on a fresh machine. IronMule never downloads
weights, so fetch one model yourself first — about 3.4 GB for the 4B used below:

```bash
hf download mlx-community/gemma-3-4b-it-4bit
```

Run the balanced local benchmark with a cached model:

```bash
ironmule benchmark --model mlx-community/gemma-3-4b-it-4bit --json benchmark.json
```

The benchmark compares IronMule's interactive and throughput modes. It does not
download a model and does not claim to compare against stock `mlx_lm`.

### Use it from Python

```python
import ironmule

runtime = ironmule.Runtime.load(
    model_id="mlx-community/gemma-3-4b-it-4bit",
    revision="93724907d4ed1745d2fe50baadf3b0b01a65abf2",
)
result = runtime.generate(
    "Explain unified memory in two short sentences.",
    max_tokens=96,
)

print(result.text)
print(result.metrics["service_ttft_ms"])
```

`Runtime.load` resolves only an already-cached model snapshot. Supplying the exact
cached commit is recommended; without it, exactly one cached revision must exist or
loading fails closed. See the [runtime guide](docs/RUNTIME.md) for concurrent
requests, throughput mode, reusable sessions, and exact model identity.

### Serve it over an OpenAI-compatible endpoint

```bash
ironmule serve --model mlx-community/gemma-3-4b-it-4bit --port 8000
```

This exposes the loaded runtime at `http://127.0.0.1:8000/v1` so an OpenAI client,
Cursor, or Open WebUI can use the local model:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "gemma", "stream": true,
       "messages": [{"role": "user", "content": "Explain unified memory in one sentence."}]}'
```

Standard library only — no web framework is added. The endpoint answers one request
at a time on the interactive path and returns HTTP 429 while it is busy; token
output is identical to `Runtime.generate` / `Runtime.stream`. It adds no sampling,
no batching, and no cloud call. See [`docs/HTTP.md`](docs/HTTP.md).

## What IronMule includes

- **Two service modes:** choose low single-request latency or higher aggregate
  throughput.
- **Prefix KV-cache reuse:** reuse a declared shared prompt without silently changing
  the execution plan.
- **Correctness checks and safe fallback:** failed grouped work restarts on the
  sequential path instead of trusting partial output.
- **Local telemetry:** record latency, TTFT, token rate, memory, fallbacks, and realised
  group width.
- **Reproducible benchmarks:** balanced warmups and repeats, raw JSON, token identity,
  spread, and a paired interval.
- **Exact validity fingerprints:** bind hardware, framework, model revision, complete
  manifest, architecture, quantisation, tokenizer, plan, mode, and workload; legacy
  incomplete profiles fail closed.

## Measured results, with the trade-off visible

These results are evidence for one measured setup, not promises for every Mac or
model. The main preregistered replication used Gemma 3 4B, MLX 0.32.0,
`mlx_lm` 0.31.3, 4-bit group-size 64 weights, greedy decoding, and an M1 Max with
32 GB unified memory.

### Latest Gemma 3 12B result

On the same M1 Max, a later isolated-process benchmark combined IronMule's fixed-cache
core path with throughput mode:

| Comparison | Complete service time | Total token rate |
| :-- | --: | --: |
| Combined path vs. baseline interactive | `−18.05%` | `+22.03%` |
| Core path vs. the same throughput mode | `−6.17%` | `+6.58%` |

This is the exact B39d workload only: Gemma 3 12B 4-bit, six concurrent greedy
requests, 48 output tokens, 32 fresh processes, and no automatic activation. Width 4
remains the baseline; widths 2 and 3 were slower in every B40 block, but drift kept
that width study formally inconclusive. See the path-free
[B39d performance summary](research/raw/B39d_public_summary_20260828.json) and
[B40 width summary](research/raw/B40_public_summary_20260828.json).

The next two-step decode idea, B3-U2, has **no speed result yet**. Its correctness
pilot completed 8/8 isolated processes and 240 measured requests with identical
tokens and final states, no fallback, no swap and no relevant crash. A missing
per-child host-state record blocks confirmation, so the result is not used as a
performance claim. See the [B3-U2 public summary](research/raw/B3-U2_public_summary_20260828.json).

### Earlier service-mode results

| Concurrent workload | Total throughput | Median request latency | p95 latency | Service TTFT | Evidence |
| :-- | --: | --: | --: | --: | :-- |
| Similar prompts | `+16.4 … +17.2%` | `+27%` | `−16%` | ~800 → ~87 ms | [E16](research/LEDGER.md#e16--replication-of-the-w4-gain-under-real-process-boundaries) |
| Mixed prompt lengths | `+15.6 … +15.8%` | `+27%` | `−15%` | ~800 → ~87 ms | [E16](research/LEDGER.md#e16--replication-of-the-w4-gain-under-real-process-boundaries) |
| Requests arriving at different times | `+15.1%` | `+26%` | `−8%` | ~690 → ~88 ms | [E16](research/LEDGER.md#e16--replication-of-the-w4-gain-under-real-process-boundaries) |
| Very short answers | `+9.2%` | `+54%` | `−9%` | — | [E15](research/LEDGER.md#e15--does-async-grouped-b1-survive-a-real-service-workload) |

The plain-language conclusion: throughput mode can finish a group of requests sooner,
but an individual request may wait longer. This is useful for concurrent work, not a
magic speed button for one chat.

Exploratory measurements on the same machine found that the throughput gain fell from
`+19.24%` at 4B to `+11.81%` at 27B. These runs were not preregistered, so they are
reported separately in the [model-scaling study](docs/SCALING.md).

Prefix reuse is a different feature with a different workload. E10 measured a
`−37.82%` end-to-end session result at a 66.8% shared prefix, while E12 checked bit-exact
reuse across 756 requests and 14,369 decode steps. See [E10](research/LEDGER.md#e10--the-prefix-cache-as-a-shipped-runtime-feature)
and [E12](research/LEDGER.md#e12--falsification-test-at-the-sliding-window-boundary).

## Commands

| Command | Purpose |
| :-- | :-- |
| `ironmule doctor` | Check Apple Silicon, Python, MLX, and Metal prerequisites |
| `ironmule models` | List cached Hugging Face model snapshots without downloading |
| `ironmule serve` | Serve a model over an OpenAI-compatible HTTP endpoint |
| `ironmule benchmark` | Compare interactive and throughput modes locally |
| `ironmule tune` | Measure candidates and write or inspect a local profile |
| `ironmule revalidate` | Canary-check the stored profile against the current setup |
| `ironmule status` | Show local hardware and profile status |
| `ironmule info` | Show package information |

Run `ironmule --help` or `ironmule <command> --help` for options.

## What it deliberately does not do

- It does not download or redistribute model weights.
- It does not host anything: `ironmule serve` binds a local OpenAI-compatible
  endpoint on your machine, one request at a time, with no cloud call.
- It does not add a sampling mode: decoding stays greedy, `temperature = 0`.
- It does not automatically select a plan that can change model output.
- It does not use true tensor batching or claim that every model becomes faster.
- It does not treat a single benchmark run as proof.

Several attractive ideas were measured and rejected, including prompt-lookup
speculation (`2.9×` slower), decode projection fusion (neutral), and true tensor
batching (changed state or tokens in tested paths). Negative results remain in the
[experiment ledger](research/LEDGER.md) and [backlog dead-ends](docs/BACKLOG.md#tier-0--already-dead-do-not-re-run-these).

## Reproduce and verify

The public benchmark uses two warmups and six measured repeats per mode. It alternates
the order of both modes, measures the complete `Runtime.serve` call, stores raw samples,
and exits nonzero when token IDs, stop reasons, or counts differ.

```bash
python -m ironmule.benchmark \
  --model mlx-community/gemma-3-4b-it-4bit \
  --warmup 2 \
  --repeats 6 \
  --json benchmark.json

pytest tests/test_cli.py tests/test_benchmark.py tests/test_ironmule_runtime.py -q
```

One loaded model process is shared between benchmark arms to avoid doubling peak
memory. Fresh-process isolation and a stock `mlx_lm` comparison arm remain open work.
The [limits](docs/LIMITS.md), [runtime guide](docs/RUNTIME.md), and
[ledger](research/LEDGER.md) describe the exact contracts and evidence.

## Repository map

| Path | What is there |
| :-- | :-- |
| `ironmule/` | Runtime, execution plans, service modes, telemetry, tuning, HTTP endpoint |
| `examples/` | Small interactive, throughput, and reusable-session examples |
| `tests/` | Fast tests plus real-model integration tests |
| `research/LEDGER.md` | Positive and negative experiments with methods and results |
| `research/raw/` | Preregistrations and public result summaries |
| `docs/RUNTIME.md` | API and runtime details |
| `docs/HTTP.md` | OpenAI-compatible endpoint: routes, streaming, limits |
| `docs/LIMITS.md` | Measured validity domain and known gaps |
| `docs/BACKLOG.md` | Open ideas and routes already ruled out |

Contributions start with [CONTRIBUTING.md](CONTRIBUTING.md). Community benchmark
submissions use the [benchmark issue template](.github/ISSUE_TEMPLATE/benchmark_submission.md)
and the fields in [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md).

## Licence

IronMule is **fair-code** under the [IronMule Licence](LICENSE.md). The source is
available, but it is not OSI open source. Personal, learning, academic, and some small
organisation uses are free under the exact licence terms. Larger production use,
hosted services, resale, and paid product embedding may require a commercial licence.
Read [LICENSE.md](LICENSE.md) and [COMMERCIAL.md](COMMERCIAL.md) before adopting it.

---

## Project Friday — the research tree

Alongside the engine package this repository carries **Project Friday**, the
measurement and self-calibration research the engine's numbers come from:
device profiles measured per machine, a serving path that only enables a knob
this device verified as token-identical, and the experiment record behind it.

See **[README_PROJECT_FRIDAY.md](README_PROJECT_FRIDAY.md)**, the open work list
in [`BACKLOG.md`](BACKLOG.md) and the append-only
[`docs/ARBEITSJOURNAL.md`](docs/ARBEITSJOURNAL.md).
