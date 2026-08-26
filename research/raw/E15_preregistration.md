# E15 — Does async grouped B1 survive a real service workload?

**Experiment ID** `E15`
**Frozen at commit** `c2c8a5931cb2c67097fed9f435c5af52c7196abe`
**Branch** `forge/hardware-autotune`
**Registered** 2026-08-25, before any E15 measurement was taken or viewed.

Frozen. Thresholds, workloads, queue rules and classification are not adjusted
after results are seen.

---

## 1. Standing on E14b

Carried forward as the current state of knowledge, not re-proved:

| Established in E14b | Value | Status |
| :-- | --: | :-- |
| Grouped async B1 against sequential B1 at `b = 4`, shapes unchanged | `+18.02%` [+17.49; +18.32] | MEASURED |
| Mechanism: host submission per request **rises** while completion wait collapses | 6.24→9.16 ms / 6.13→1.27 ms | MEASURED |
| Grouped async B1 produced **zero** token divergence at any width | — | MEASURED |
| True tensor batching diverges at `b = 8`, one token, reproducibly | — | MEASURED |

**No kernel count is used and no absolute dispatch time is derived.**
**`completion_wait` is never called GPU time**; device work may already have begun
during submission, so it is a wait, not an execution time.

E14b measured one teacher-forced decode step per sequence under ideal conditions.
E15 asks whether that gain survives ragged lengths, staggered arrivals, early
termination and real generation.

## 2. Research question and hypotheses

> Is grouped asynchronous B1 execution a semantics-preserving runtime strategy for
> several independent requests under realistic service conditions?

**H1** The gain survives: a reproducible throughput improvement across homogeneous,
heterogeneous and staggered workloads, with exact correctness.

**H0** It does not: the gain is an artifact of uniform, simultaneous, equal-length
work and collapses when groups become ragged or partially filled.

Competing explanations to keep apart:

| Explanation | Discriminating evidence |
| :-- | :-- |
| Overlap of device execution with host submission (E14b's mechanism) | gain present at `W ≥ 2` and scaling with achieved group occupancy |
| Interleaving rather than grouping | `W = 1` of the grouped executor is round-robin with per-step synchronisation and isolates this |
| Ragged groups waste the effect | gain falls with mean realised group width in heterogeneous and early-exit workloads |
| Arrival timing | staggered workload against the same requests arriving together |
| Measurement artifact | harness controls, fresh processes, randomised interleaved blocks |

## 3. Scope of what is built

A **minimal experimental executor** in `research/`, not in `forge/`. It is not
adaptive, has no controller, and introduces **no tensor batch dimension** — every
execution stays batch 1 with unchanged shapes.

**The caller keeps the execution plan.** The executor never switches between
`StrictOneShotPlan` and `ReusableSessionPlan` and never alters their semantics. A
group may mix requests, but never a request's plan.

## 4. Strategies

| Strategy | Definition |
| :-- | :-- |
| `SequentialService` | each request executed and synchronised on its own, run to completion before the next begins. The naive service baseline |
| `AsyncGroupedB1(W)` | round-robin over ready requests; up to `W` independent batch-1 decode steps are built and submitted with no intermediate barrier, then completed as one group |

Widths `W ∈ {1, 2, 4, 8}`. **`W = 1` is not redundant**: it is round-robin
interleaving with per-step synchronisation, and it separates the effect of
interleaving from the effect of grouping. Without it, `SequentialService` against
`AsyncGroupedB1(4)` would confound the two.

`W = 4` is the **primary candidate**, not a presumed winner.

## 5. Queue and fairness rules

- The executor **never waits to fill a group**. Each round it takes the requests
  that are ready, up to `W`.
- Order is deterministic and fair: ready requests are served in arrival order, ties
  broken by request id, and a served request goes to the back of the queue.
- A request that reaches EOS or its token cap is removed from the active set at the
  end of the round in which it finished.
- When nothing is ready the executor sleeps until the next arrival. Idle waiting is
  recorded, never counted as service time.

## 6. Workloads

Real greedy generation on real prompts, so EOS occurs naturally and output lengths
are genuinely ragged. Eight requests per workload, drawn from the vendored SQuAD
v1.1 documents (SHA-256 `95aa6a52…6972c9`).

| Workload | Contexts | Output caps | Arrivals |
| :-- | :-- | :-- | :-- |
| `homogeneous` | all ~1024 tokens | all 16 | all at `t = 0` |
| `heterogeneous` | 320 – 1200 tokens, mixed | 8, 12, 16, 24 mixed | all at `t = 0` |
| `staggered` | same mix as heterogeneous | same mix | 0, 30, 60, … 210 ms |

Early termination is present in all three, from natural EOS and from the ragged
caps, and the removal of a finished request from the active group is checked as an
invariant rather than as a separate workload.

Both `StrictOneShotPlan` and `ReusableSessionPlan` are run, **separately**. Because
prefill happens before the service phase and is identical under both strategies, the
plan determines how a request's state is built and the executor never sees a
difference; the plans are run anyway so that this non-interaction is measured rather
than assumed.

**Prefill is performed up front and excluded from the strategy comparison**, because
it is identical under both strategies and would otherwise dominate and dilute the
measurement. Full-request latency is therefore reported as measured service latency
**plus separately measured prefill cost**, and is labelled as a composition, never
as one timed quantity. Arrival offsets apply to the service phase.

## 7. Timing definitions and metrics

Per request: arrival time, queue wait (arrival to first decode step), TTFT within
the service phase, time of each generated token, completion time.

Per run: total wall clock for the workload, aggregate tokens per second, idle time,
mean realised group width.

Per round: host preparation, submission, completion wait — with the same four-way
split and the same naming discipline as E14b.

Also: inter-token latency p50 and p95, full response latency p50 and p95, MLX peak
memory, thermal state, power source.

**Group time is never divided by group width and then called caller latency.**
Throughput and actual per-request latency are reported in separate rows.

## 8. Correctness and isolation

Every request is compared against its own `SequentialService` run **under the same
execution plan**:

1. exact token IDs, element by element
2. token count
3. stop reason (EOS or cap)
4. final KV state over the logically valid region, bitwise via unsigned integer
   views as in E12

Additional invariants:

5. **Early finisher** — a request that ends before the others must not change any
   other request's output.
6. **Removal** — after a request leaves the active set, the remaining requests must
   still match their sequential run.
7. **No aliasing** — no request's KV state may be affected by another's; checked by
   the bitwise state comparison in (4).
8. **Order independence** — the same workload run with the arrival order reversed
   must produce identical per-request outputs.

**Any reproducible token divergence or state mixing is a hard failure** and forces
`STATE_ISOLATION_FAILURE` regardless of performance.

## 9. Execution and statistics

Four fresh processes, 2 warmup repetitions and 3 measured repetitions, measurement
blocks randomised and interleaved with seed `20260825 + process index`. Every
individual measurement is stored.

Medians per process, then paired across processes with a 10,000-resample bootstrap,
seed `20260825`. Four processes give a coarse interval and it is reported as such.

Harness controls before the main run: a timer noise floor on an empty queue; a full
barrier before every block; and `AsyncGroupedB1(1)` must reproduce
`SequentialService`'s token output exactly, since it differs only in ordering.

## 10. Classification (frozen, ordered)

Let `G(W, workload) = 1 − T_grouped(W) / T_sequential` on total workload wall clock.
Threshold `θ = 0.10`, intervals must exclude zero. Primary width `W = 4`.
Let `L95` be full response latency p95.

| # | Condition | Class |
| --: | :-- | :-- |
| 0 | any reproducible token divergence, state mismatch, or order dependence | `STATE_ISOLATION_FAILURE` |
| 1 | harness control fails, or relative IQR on the sequential run > 0.10 | `INCONCLUSIVE` |
| 2 | `G(4) ≥ θ` with interval excluding 0 in **all three** workloads, and `L95` inflation ≤ 10% | `ASYNC_B1_SERVICE_VIABLE` |
| 3 | `G(4) ≥ θ` in all three workloads, `L95` inflation > 10% | `THROUGHPUT_GAIN_WITH_LATENCY_COST` |
| 4 | `G(4) ≥ θ` in `homogeneous` but not in `heterogeneous` or `staggered` | `RAGGED_OR_ARRIVAL_SENSITIVE` |
| 5 | otherwise | `NO_SERVICE_GAIN` |

A positive finding requires all three together: a reproducible throughput gain under
realistic workloads, exact correctness, and an explicitly reported latency Pareto
front. A gain in the homogeneous burst alone is not a service result.

## 11. Stop rules

No interim analysis. The pilot validates the harness only and its numbers are never
interpreted; afterwards workloads and rules are frozen. The main run is analysed
exactly once. An inconclusive result is not extended inside E15.

Abort, preserving partial evidence and never retrying: another local model process
(`gpu_busy()`), power source not AC, MLX peak above 12 GiB, or main-run wall time
above 45 minutes.

## 12. Known risks

1. **Prefill exclusion.** The service phase is decode only. Full-request latency is
   a composition and is labelled so. A reader wanting end-to-end numbers must add
   the reported prefill cost.
2. **Arrival offsets are small relative to a decode step** (30 ms against ~13 ms),
   so the staggered workload fills groups quickly. This makes staggering a mild
   test, not a harsh one, and the result is bounded accordingly.
3. **Eight requests is a small service.** Queue effects that only appear under
   sustained load are out of scope.
4. **`W = 8` may regress**, as it did in E14b and E2 and E3. Expected, not a
   surprise to be explained away.
5. **The executor is experimental.** It is not the runtime, and nothing in `forge/`
   changes. No conclusion about a production scheduler follows from it.
6. **Single model, single machine.** No claim beyond it.
