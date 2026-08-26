# IronMule

*Formerly Claude Forge. The Python package is .*

A local inference runtime for MLX on Apple silicon, built from measurements rather
than from expectations. Every performance statement below points at an experiment in
[`research/LEDGER.md`](../research/LEDGER.md) with raw data under `research/raw/`.

Version `0.1.0`. Published from a curated subset; see [`README.md`](../README.md).

## What it is

Two decisions stay with the caller, because both change observable behaviour:

| Decision | Options | What changes |
| :-- | :-- | :-- |
| **Execution plan** | `StrictOneShotPlan`, `ReusableSessionPlan` | the tokens produced |
| **Service mode** | `InteractiveMode`, `ThroughputMode` | the latency/throughput trade |

Nothing in the runtime switches either on its own. `Telemetry.plan_switch_attempts`
exists to make a violation visible and is expected to stay at zero.

## Public API

```python
import ironmule

rt = ironmule.Runtime.load()                       # interactive by default, tuned knobs

# one request
out = rt.generate("Explain unified memory.", max_tokens=96)
out.text, out.tokens, out.stop_reason, out.metrics

# a session: one document, many questions, prefill computed once
plan = rt.session_plan(document, name="docs")
out  = rt.generate(document + "\n\nQuestion: ...", plan=plan, max_tokens=48)

# concurrency
rt.mode = ironmule.ThroughputMode()                 # grouped batch-1, width <= 4
results = rt.serve([ironmule.Request(prompt_ids=ids, max_tokens=48, plan=plan)
                    for ids in prompts])
rt.telemetry.snapshot()

# validity
rt.fingerprint(plan, {"prompt_tokens": 1024, "max_tokens": 48, "concurrency": 4})
rt.revalidate()
```

Exported: `Runtime`, `Request`, `Result`, `StrictOneShotPlan`, `ReusableSessionPlan`,
`InteractiveMode`, `ThroughputMode`, `SequentialExecutor`, `AsyncGroupedB1Executor`,
`Telemetry`, `RequestMetrics`, `build_fingerprint`, `usable`, plus the tuning
entry points kept from the research phase.

## Execution plans

**`StrictOneShotPlan`** prefills the whole prompt in one forward. Use it when output
must match an untuned single-shot path, or when prompts share nothing.

**`ReusableSessionPlan(prefix_ids)`** prefills in two chunks split at a declared
prefix and reuses that prefix's KV state. Within the plan a cache hit is **bit
exact** — E9 measured `max |delta| = 0` over twelve requests and every decode step,
and E12 reproduced it over 756 requests and 14,369 decode steps across prefix
lengths 276 to 2048, spanning Gemma 3's 1024-token sliding-window boundary, in five
processes.

The two plans **do not agree with each other**. E9 measured them up to `4.31` logits
apart; E13 measured the resulting quality difference on extractive question
answering as bounded above by `1.14` accuracy points at 95% confidence, on a
contaminated public benchmark. That is a bound for that evaluation set, not a
statement that the plans are interchangeable.

## Service modes

**`InteractiveMode`** runs requests sequentially, synchronising every step. Lowest
latency for a single caller; no economy of scale at all — E14b measured submission
and completion wait per request flat across every batch size.

**`ThroughputMode`** submits up to four independent batch-1 decode steps without an
intermediate barrier and completes them as a group. **Tensor shapes never change**;
only the synchronisation boundary moves. E14b attributed the gain to device
execution overlapping host submission rather than to cheaper host work: host
submission per request *rises* from 6.24 to 9.16 ms while completion wait collapses
from 6.13 to 1.27 ms.

Width 4 is the maximum because E15 found the whole gain available there and E14b
found width 8 regressing. The executor **never waits to fill a group**; realised
width drops below four whenever fewer requests are ready, and that is intended.

### The measured trade

E16, forty independent OS processes, five per condition:

| | throughput | median latency | tail latency (p95) | service TTFT |
| :-- | --: | --: | --: | --: |
| homogeneous | `+16.4%` … `+17.2%` | `+27%` | `−16%` | ~800 → ~87 ms |
| heterogeneous | `+15.6%` … `+15.8%` | `+27%` | `−15%` | ~800 → ~87 ms |
| staggered arrivals | `+15.1%` | `+26%` | `−8%` | ~690 → ~88 ms |
| short answers (terse) | `+9.2%` | `+54%` | `−9%` | — |

Reproduced locally through this runtime with `python -m ironmule.benchmark`:

```
mode              wall ms        tok/s  svcTTFT p50      lat p50      lat p95
interactive          3303         87.2       1384.9       1921.0       3367.8
throughput           2700        106.6        246.6       2893.8       3048.3
throughput gain +18.24%   identical answers in both modes: True
```

**Grouping does not make a request faster. It makes requests finish together.**
Median latency worsens, tail latency and first-token latency improve. Which is
preferable is a service decision.

## Telemetry

Two time-to-first-token definitions are kept apart, because under concurrency they
differ by the queue wait and reporting one hides the effect a service cares about:

- `service_ttft_ms` — from request arrival. What the caller experiences.
- `engine_ttft_ms` — from the model actually starting. What the engine owns.

Also recorded: full request latency, queue wait, inter-token latency p50/p95,
aggregate tokens per second, realised group width per round, peak memory, fallbacks
with reasons, correctness errors, and plan-switch attempts.

**There is no field that divides a group's wall time by its width.** That quotient
is not a caller latency and this runtime does not compute it.

## Fallback

Two layers, both preserving output:

1. **Per group** — if a grouped round raises, the affected requests are restarted
   from their post-prefill state and finished sequentially. Tokens produced by the
   failed group are discarded rather than trusted. This wastes work and is the only
   choice that keeps output identical to a clean sequential run.
2. **Whole executor** — if the executor itself fails, every unfinished request
   restarts on the sequential path.

Both increment `Telemetry.fallbacks` and record a reason. Verified on the real
engine with an injected device failure: the answers were identical to the
sequential reference.

## Fingerprint and validity

`ironmule.fingerprint` records hardware, OS, MLX and mlx_lm versions, runtime version,
model, quantisation, execution plan, service mode and workload traits. Identity
fields — hardware, versions, model, quantisation, plan, mode — invalidate a stored
decision when they change. Workload fields are compared in buckets, so a 10% longer
prompt is the same regime and a doubled one is drift.

`Runtime.revalidate()` compares the current identity against the last recorded one
and returns `valid`, `valid_with_workload_drift`, or `revalidation_required`.

## Running things

```bash
python -m pytest tests/test_ironmule_runtime.py -q                 # fast, no model
python -m pytest tests/test_ironmule_runtime_integration.py -q     # real model
python -m ironmule.benchmark --requests 6 --max-tokens 48
python -m ironmule.benchmark --plan reusable --json out.json
python examples/interactive_chat.py "Explain unified memory."
python examples/throughput_service.py
python examples/reusable_session.py
python -m ironmule.plans && python -m ironmule.telemetry && python -m ironmule.fingerprint
```

## Module map

| Module | Responsibility |
| :-- | :-- |
| `ironmule/plans.py` | execution plans, caller-chosen, never substituted |
| `ironmule/service.py` | `Runtime`, `Request`, `Result`, modes, MLX backend |
| `ironmule/executor.py` | sequential and grouped executors, fallback, sessions |
| `ironmule/telemetry.py` | the two TTFT definitions and the metric set |
| `ironmule/fingerprint.py` | identity and validity of stored decisions |
| `ironmule/benchmark.py` | reproducible local benchmark |
| `ironmule/runtime.py` | `Engine`, `PrefixCache`, fixed-shape KV cache (research phase) |
| `ironmule/tune.py`, `hw.py`, `fast.py`, `bench.py`, `ab.py` | autotuner and measurement infrastructure |
