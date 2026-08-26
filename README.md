# IronMule

**IronMule measures your hardware and uses only optimizations that prove faster and remain correct on your machine.**

[![License: fair-code](https://img.shields.io/badge/license-fair--code-blue.svg)](LICENSE.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Apple silicon](https://img.shields.io/badge/platform-Apple%20silicon-lightgrey.svg)](https://github.com/ml-explore/mlx)
[![Tests](https://img.shields.io/badge/tests-28%20passing-brightgreen.svg)](tests/)
[![Experiments](https://img.shields.io/badge/preregistered%20experiments-16-orange.svg)](research/LEDGER.md)

A **local LLM inference runtime for [MLX](https://github.com/ml-explore/mlx) on
Apple silicon** — M1, M2, M3, M4 — with prefix KV caching, grouped batch-1
execution and honest telemetry. Every performance number below was measured on the
machine it claims to describe, with the raw data and the preregistered method in
[`research/`](research/LEDGER.md). Nothing here is an estimate.

> Status: `0.1.0`, single-machine, single-model validity domain. Read
> [`LIMITS.md`](docs/LIMITS.md) before trusting any number for your own case.

## What it does

**It refuses to guess.** Sixteen preregistered experiments decided what ships. Four
optimisations that looked obvious were measured and dropped: speculative decoding
(2.9× slower), projection fusion in decode (neutral), true tensor batching (changes
tokens at batch 8), a "dispatch overhead" hypothesis that turned out to be wrong.
The negative results are in the ledger with the positives.

**It never changes your answer behind your back.** Two decisions stay with the
caller because both alter observable behaviour, and nothing in the runtime switches
either on its own:

| Decision | Options | What changes |
| :-- | :-- | :-- |
| Execution plan | `StrictOneShotPlan`, `ReusableSessionPlan` | which tokens come out |
| Service mode | `InteractiveMode`, `ThroughputMode` | the latency/throughput trade |

**It tells you what it cost.** Time to first token is reported twice — from request
arrival and from the moment the model actually started — because under concurrency
those differ by the queue wait, and reporting one hides the effect a service cares
about.

## Install

Needs Python 3.10+, an Apple silicon Mac, and a local MLX model snapshot. IronMule
does not ship or download weights.

```bash
git clone https://github.com/Tobayko/IronMule && cd IronMule
pip install -e ".[dev]"
```

## Quick start

```python
import ironmule

rt = ironmule.Runtime.load()                      # interactive mode by default
out = rt.generate("Explain unified memory in two sentences.", max_tokens=96)
print(out.text, out.metrics["service_ttft_ms"])
```

**One document, many questions** — the prefill is computed once and reused. Inside
this plan a cache hit is bit exact, verified across 756 requests and 14,369 decode
steps:

```python
plan = rt.session_plan(document, name="docs")
for question in questions:
    out = rt.generate(document + "\n\nQuestion: " + question, plan=plan, max_tokens=48)
```

**Several concurrent requests** — grouped batch-1 execution at width ≤ 4. Tensor
shapes never change; only the synchronisation boundary moves:

```python
rt.mode = ironmule.ThroughputMode()
results = rt.serve([ironmule.Request(prompt_ids=ids, max_tokens=48, plan=plan)
                    for ids in prompts])
print(rt.telemetry.snapshot())
```

## What it buys, and what it costs

Reproduce with `python -m ironmule.benchmark`:

```
mode              wall ms        tok/s  svcTTFT p50      lat p50      lat p95
interactive          3303         87.2       1384.9       1921.0       3367.8
throughput           2700        106.6        246.6       2893.8       3048.3

throughput gain +18.24%   identical answers in both modes: True
```

Replicated across **40 independent OS processes**, five per condition, zero
correctness failures ([E16](research/LEDGER.md)):

| Workload | throughput | median latency | tail p95 | service TTFT |
| :-- | --: | --: | --: | --: |
| homogeneous | `+16.4 … +17.2%` | `+27%` | `−16%` | ~800 → ~87 ms |
| heterogeneous | `+15.6 … +15.8%` | `+27%` | `−15%` | ~800 → ~87 ms |
| staggered arrivals | `+15.1%` | `+26%` | `−8%` | ~690 → ~88 ms |
| short answers | `+9.2%` | `+54%` | `−9%` | — |

**Grouping does not make a request faster. It makes requests finish together.**
Median latency gets worse, tail and first-token latency get better. Which you want
is a service decision, so IronMule reports both and picks neither.

For sessions that share a document, reusing the prefix cuts time to first token to
`0.41×` at a 67% shared prefix, and up to `0.09×` at 2048 tokens
([E10](research/LEDGER.md), [E12](research/LEDGER.md)).

## Correctness

Not asserted — tested. Twenty-eight tests cover exact token IDs, token counts, stop
reasons, KV state hashes, ragged response lengths, early-finishing requests,
reversed arrival order, heterogeneous prompt lengths, staggered arrival, group
widths 1 to 4, sequential fallback, and the absence of state aliasing between
requests.

```bash
pytest tests/test_ironmule_runtime.py -q              # fast, no model needed
pytest tests/test_ironmule_runtime_integration.py -q  # against a real model
```

A failed group restarts its requests from prefill and finishes them sequentially,
discarding tokens the failed group produced rather than trusting them. That wastes
work and is the only choice that keeps output identical to a clean sequential run.
Verified on a real model with an injected device failure.

## Honest limits

The measured validity domain is narrow and stated in full in
[`LIMITS.md`](docs/LIMITS.md): one model (`gemma-3-4b-it-4bit`), 4-bit group-64
quantisation, MLX 0.32.0, an M1 Max on AC power, greedy decoding, contexts of 276
to 2048 tokens, up to 8 concurrent requests.

Nothing outside that box is claimed. The fingerprint exists to force
re-measurement when the machine, framework, model or workload changes rather than
to carry a stale decision forward.

Known gaps, also in [`docs/LIMITS.md`](docs/LIMITS.md): ragged prompt lengths inside one group are
untested; sustained load is untested; the quality bound of 1.14 accuracy points was
measured on a public benchmark the model may have been trained on; and kernel counts
were retired as unmeasurable, so no absolute dispatch time is claimed anywhere.

## About this repository

IronMule was developed inside a private research project and is published here as a
curated subset on a fresh history: the runtime, the examples, the tests, and the
experiment ledger that backs every number. Local paths, personal identifiers, model
weights and third-party datasets are not included; SQuAD is fetched by a script
under its own terms rather than redistributed.

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Layout

| Path | |
| :-- | :-- |
| `ironmule/` | the runtime: plans, modes, executors, telemetry, fingerprint, autotuner |
| `examples/` | interactive chat, throughput service, reusable session |
| `tests/` | fast tests against a scripted backend, plus real-model integration tests |
| `research/LEDGER.md` | every experiment, positive and negative, with method and raw data |
| `research/raw/` | preregistrations and result summaries |
| [`RUNTIME.md`](docs/RUNTIME.md) | technical documentation |
| [`LIMITS.md`](docs/LIMITS.md) | validity domain and known gaps |

## Licence and commercial use

IronMule is **fair-code** under the [IronMule Licence](LICENSE.md):
source-available, not OSI open source. The licence carries a plain-language summary
and a clarifications section that answers the boundary questions explicitly, so you
can tell which side you are on without asking a lawyer.

| | |
| :-- | :-- |
| Personal, hobby, learning | **free** |
| Research, teaching, academic publication, benchmarking | **free** |
| A small organisation running it in production — under 10 people **and** ≤ EUR 1M turnover | **free** |
| Any larger company, evaluating it outside production | **free for 90 days** |
| **Any larger company, running it in production** | needs a commercial licence |
| Offering it to third parties as a hosted or managed service | needs a commercial licence |
| Reselling it or embedding it in a product you sell | needs a commercial licence |

**The two tests.** First: is your company small — fewer than 10 people *and* turnover
of EUR 1M or less? Both limits, not either. If yes, production is free. If no, you
get 90 days to evaluate and a licence is due once it runs in production. Second,
independently of size: if a third party gets IronMule itself as the thing being sold,
that needs a licence too.

Individuals, learning and academic work never pay, at any scale. Details, the
definition of production use, and how to get a licence: [`COMMERCIAL.md`](COMMERCIAL.md).

IronMule does not redistribute model weights; the model you point it at carries its
own terms.

---

<sub>Keywords: MLX inference runtime, local LLM on Mac, Apple silicon LLM, M1 Max
inference, on-device inference, prefix KV cache, KV cache reuse, continuous
batching alternative, micro-batching, batch-1 grouping, time to first token, TTFT,
throughput vs latency, Gemma 3 MLX, quantised 4-bit inference, reproducible
benchmarking, fair-code, source-available licence.</sub>
