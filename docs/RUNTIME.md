# IronMule

*The Python package is `ironmule`.*

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

The following E16 table is historical research evidence from forty independent OS
processes, five per condition. It is not the output or denominator of the current
public benchmark protocol:

| | throughput | median latency | tail latency (p95) | service TTFT |
| :-- | --: | --: | --: | --: |
| homogeneous | `+16.4%` … `+17.2%` | `+27%` | `−16%` | ~800 → ~87 ms |
| heterogeneous | `+15.6%` … `+15.8%` | `+27%` | `−15%` | ~800 → ~87 ms |
| staggered arrivals | `+15.1%` | `+26%` | `−8%` | ~690 → ~88 ms |
| short answers (terse) | `+9.2%` | `+54%` | `−9%` | — |

The current public benchmark (`python -m ironmule.benchmark`) uses two warmups and
six measured repeats per arm. Measured repeats are even and alternate AB/BA order;
each arm gets its own execution plan and prefix-cache state. Its primary clock is the
complete service wall around `Runtime.serve`, reported as `outer_wall_ms`. Executor
`wall_ms`, TTFT, latency, queue, and group-width values remain diagnostics. The JSON
output (`--json out.json`) preserves raw warmup/repeat snapshots, raw token IDs and
stop reasons, workload and runtime fingerprints, medians/spread, and a paired
bootstrap interval. It does not include a stock `mlx_lm` arm.

```
mode          outer ms p50   physical tok/s   visible tok/s
interactive       <median>         <median>         <median>
throughput        <median>         <median>         <median>

throughput gain <outer-wall ratio>   95% CI [<low>; <high>]
identical answers in both modes: True
```

Physical token rates include the prefill-produced first token and EOS when emitted;
visible token rates exclude EOS. A token, stop-reason, or count mismatch is emitted
to stderr as a structured difference (including the first differing position and
both values), exits nonzero, and is included in the result file when `--json` is
supplied. The protocol shares one loaded runtime/model process to
avoid doubling peak memory, so fresh-process isolation remains an open R3 limitation;
only the plans/cache state are isolated per arm.

### Phase and roofline diagnostics

`ironmule.benchmark.phase_roofline_diagnostic` is a pure, diagnostic-only
calculation over explicitly supplied measurements. It keeps prefill and decode
separate; `decode_steps` excludes the first token produced by prefill. A roofline
is emitted only when the effective bandwidth, active weight bytes and all KV/extra
traffic components are present and valid:

```python
from ironmule.benchmark import phase_roofline_diagnostic

diagnostic = phase_roofline_diagnostic(
    prefill_ns=2_000_000, decode_ns=5_000_000, decode_steps=10,
    effective_bandwidth_gbps=300.0,
    active_weight_bytes_per_token=100_000_000,
    kv_read_bytes_per_token=1_000,
    kv_write_bytes_per_token=2_000,
    extra_bytes_per_token=3_000,
    bandwidth_source="measured_gemv_probe",
    bandwidth_source_kind="measured_effective",
)
```

The result uses schema `ironmule.phase_roofline.v1`. It reports
`ideal_tokens_per_second` and a per-run `efficiency`; it does not choose a plan,
activate a profile, or label a workload compute-/bandwidth-bound. Missing or
invalid inputs are `inconclusive`/`invalid`, and zero decode steps are
`not_applicable`. `effective_bandwidth_gbps` must identify a measured effective
bandwidth, not a nominal chip specification. A nominal peak source remains
valid provenance but is always inconclusive for efficiency. For zero decode
steps, only the decode phase and roofline are `not_applicable`; with a measured
prefill duration the top-level record remains `inconclusive`. The helper does not
infer an EOS reason. The calculation is consistent with
Apple's controlled M4/M5 observation that TTFT and subsequent-token generation
are different regimes, but that external result is not an IronMule benchmark:
[Apple ML Research on MLX and M5 GPU Neural Accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).

**Grouping does not make a request faster. It makes requests finish together.**
Median latency worsens, tail latency and first-token latency improve. Which is
preferable is a service decision.

## Telemetry

Two time-to-first-token definitions are kept apart, because under concurrency they
differ by the queue wait and reporting one hides the effect a service cares about:

- `service_ttft_ms` — from request arrival. What the caller experiences.
- `engine_ttft_ms` — from the model actually starting. What the engine owns.

Also recorded: full request latency, queue wait, inter-token latency p50/p95,
physical and visible token counts, and the compatibility `aggregate_tokens_per_second`
rate, while the benchmark derives separate physical/visible rates. Also recorded are
realised group width per round, peak memory, fallbacks with reasons, correctness errors,
and plan-switch attempts. The runtime
reports whether a correctness comparison was performed; `correctness_errors=0` alone
means only that no recorded errors exist, not that a correctness check ran.

`max_tokens` is the total physical output cap. The prefill-produced first token counts
toward it; if that token is EOS, generation stops immediately with `stop_reason="eos"`.
Otherwise reaching the cap reports `stop_reason="length"`. Visible counts exclude EOS.

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

## Evidence contracts (D1, no routing)

`ironmule.evidence` is a standard-library-only, immutable contract layer for
`ExecutionStrategy`, `ValidityDomain`, evaluator-owned `EvidenceRecord` and
`TrustedExecutionProfile`. It rejects missing identity, unknown schema fields,
non-finite measurements, self-qualification and domain drift. A trusted profile can
only be constructed from supplied `QUALIFIED` records that pass exact correctness,
resource and repeated-uncertainty gates.

This module is deliberately not imported by Runtime, plans, modes, executors, tuner or
the package root. It has no MLX import, persistence, `run()`/`select()` method,
automatic routing or activation. D1 represents existing path IDs as data; it does not
change which path executes. See
[`B27_PHASE_D_CONTRACT_PROPOSAL.md`](B27_PHASE_D_CONTRACT_PROPOSAL.md) for the approved
scope and excluded later decisions.

## Running things

```bash
python -m pytest tests/test_ironmule_runtime.py -q                 # fast, no model
python -m pytest tests/test_ironmule_runtime_integration.py -q     # real model
python -m ironmule.benchmark --requests 6 --max-tokens 48
python -m ironmule.benchmark --plan reusable --json out.json
ironmule tune --show
ironmule revalidate --model MODEL --max-tokens 32
ironmule status --model MODEL
ironmule models --model REPO_ID       # local Hugging Face cache only; no download
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
| `ironmule/evidence.py` | immutable fail-closed strategy/domain/evidence/profile contracts; no routing |
| `ironmule/benchmark.py` | reproducible local benchmark |
| `ironmule/runtime.py` | `Engine`, `PrefixCache`, fixed-shape KV cache (research phase) |
| `ironmule/tune.py`, `hw.py`, `fast.py`, `bench.py`, `ab.py` | autotuner and measurement infrastructure |
