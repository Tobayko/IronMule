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

IronMule runs locally. It does not upload prompts, silently download models, or hide
a cloud service behind the API. Model downloads require an explicit command.

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

### Serve it

A persistent, isolated model service with an OpenAI-compatible API:

```bash
ironmule setup --mode desktop
ironmule models list --family gemma --json
ironmule models add mlx-community/gemma-3-1b-it-4bit
ironmule serve --model mlx-community/gemma-3-1b-it-4bit
```

`POST /v1/chat/completions` answers with one JSON completion, or with
server-sent events when the body carries `"stream": true`. `GET /v1/models`
lists the registered models with their exact snapshot revisions, and
`/health` reports queue depth and completion counters. See
[`docs/HTTP.md`](docs/HTTP.md) for the routes and
[`docs/PRODUCT_QUICKSTART.md`](docs/PRODUCT_QUICKSTART.md) for server mode and TLS.

The service runs the stock MLX-LM reference path. `ironmule optimize run` performs
bounded calibration with readiness waiting and local history; **automatic
deployment is not enabled**. The Python runtime optimizations described below are
a separate execution path.

### Install the checkout

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

`ironmule models` prints an empty list on a fresh machine. Fetch one model first
— about 3.4 GB for the 4B used below — or use the preview's explicit
`models add MODEL --download` command after setup:

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

### Let the runtime choose the path

`Runtime` never changes the service mode you gave it. `AppleRuntime` is the entry point
for the other case: load a model and let it pick, per dispatch, from the paths this
machine has already qualified.

```python
import ironmule

runtime = ironmule.AppleRuntime.load()          # or objective="latency"

print(runtime.generate("Explain unified memory.").text)
print(runtime.last_decision["route"], runtime.last_decision["reason"])
```

One request takes the sequential path immediately and never waits for a partner.
Several requests handed over together are grouped, and the tuned profile decides from
there whether the paired path applies. An unqualified machine, model revision or library
build routes nothing at all. The router reports an execution plan; it never substitutes
one, because plans change the output.

`objective` names the trade rather than guessing it. `B55` measured it on an M1 Max with
Gemma 3 4B: at two ready requests, grouping bought `+7.7%` aggregate tokens per second
and cost `+74.9%` median per-request latency. `objective="latency"` keeps every dispatch
sequential. See [`research/LEDGER.md`](research/LEDGER.md) entry `B55` for the full
comparison against both fixed modes.

### One dispatch, two objectives

A request may name its own objective, so a chat stream and a batch job can share one
loaded model:

```python
from ironmule import Request

results = runtime.serve([
    Request(prompt_ids=chat, objective="latency"),
    Request(prompt_ids=batch_a, objective="throughput"),
    Request(prompt_ids=batch_b, objective="throughput"),
])
print(runtime.last_decision["route_by_request"])
print(runtime.last_decision["latency_protected"])
```

A request that names nothing inherits the runtime's objective, so code written before
this existed behaves exactly as it did. The dispatch is split into one cohort per
objective, latency first, and the latency requests are never grouped with anything.

**`objective="latency"` is a dispatch preference, not a latency guarantee.** When both
objectives arrive together the preference holds: a latency request beside a foreign
throughput request measured `0.98` and `1.00` against matched controls. When it becomes
servable *after* a throughput cohort has already started, it does not: `B56` measured it
at `2.11x` its own latency, and a hand-written caller doing the same split measured
`2.12x`, so this is the device rather than the routing.

Nothing available inside that dispatch fixes it. Letting the request join the group costs
`2.05x` and loses the latency path as well. Giving it its own process and its own
submission to the GPU halves the wait but still costs `1.50x` solo, reproducibly across
two confirmation sessions, because two submission streams share one device: the request's
decode rate falls from `75` to `50` tokens per second while the cohort's completion rises
by `22%`. `B56b` is `NO_GO` and that path is not pursued.

`last_decision["latency_protected"]` reports which case a dispatch was in. A caller that
receives a request later should dispatch it later, which is what a server does anyway.

## What IronMule includes

- **Two service modes:** choose low single-request latency or higher aggregate
  throughput.
- **One router over both:** `AppleRuntime` picks a qualified path per dispatch from
  facts known at dispatch, and records the route, the reason and the evidence.
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
| `ironmule setup` | Initialize desktop or server product settings |
| `ironmule serve` | Serve a registered model over an OpenAI-compatible HTTP API |
| `ironmule optimize` | Run bounded local calibration and inspect its history |
| `ironmule data` | Collect portable optimizer evidence under free-only quotas |
| `ironmule doctor` | Check Apple Silicon, Python, MLX, and Metal prerequisites |
| `ironmule models` | List cached Hugging Face model snapshots without downloading |
| `ironmule benchmark` | Compare interactive and throughput modes locally |
| `ironmule tune` | Measure candidates and write or inspect a local profile |
| `ironmule revalidate` | Canary-check the stored profile against the current setup |
| `ironmule status` | Show local hardware and profile status |
| `ironmule info` | Show package information |

Run `ironmule --help` or `ironmule <command> --help` for options.

## What it deliberately does not do

- It does not download or redistribute model weights.
- It does not offer sampling: decoding is greedy, so `temperature` and `top_p` are
  accepted and ignored.
- It does not interleave concurrent requests through one engine. They queue and are
  served in arrival order.
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
| `ironmule/` | Runtime, execution plans, service modes, telemetry, and tuning |
| `examples/` | Small interactive, throughput, and reusable-session examples |
| `tests/` | Fast tests plus real-model integration tests |
| `research/LEDGER.md` | Positive and negative experiments with methods and results |
| `research/raw/` | Preregistrations and public result summaries |
| `docs/RUNTIME.md` | API and runtime details |
| `docs/LIMITS.md` | Measured validity domain and known gaps |
| `docs/BACKLOG.md` | Open ideas and routes already ruled out |

Contributions start with [CONTRIBUTING.md](CONTRIBUTING.md). Community benchmark
submissions use the [benchmark issue template](.github/ISSUE_TEMPLATE/benchmark_submission.md)
and the fields in [COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md).

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

See **[the research overview](docs/README_PROJECT_FRIDAY.md)**, its open work list in
[`docs/PROJECT_FRIDAY_BACKLOG.md`](docs/PROJECT_FRIDAY_BACKLOG.md) and the append-only
work journal in [`docs/ARBEITSJOURNAL.md`](docs/ARBEITSJOURNAL.md), which is the one
document kept in German.
