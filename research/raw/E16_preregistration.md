# E16 — Does the W=4 gain survive real process boundaries?

**Experiment ID** `E16`
**Frozen at commit** `a35cb36cb6475291a0dd601e7f7c96b9935e54c9`
**Branch** `forge/hardware-autotune`
**Registered** 2026-08-25, before any E16 measurement was taken or viewed.

Frozen. Thresholds, workloads and classification are not adjusted after results are
seen. **No new widths, no new candidates, no controller, no optimisation.**

---

## 1. The single question

E15 concluded `ASYNC_B1_SERVICE_VIABLE` with a `+14.99%` to `+17.13%` throughput
gain at `W = 4`. Its own correction **M2** recorded that the "four fresh processes"
were four measurement blocks inside **one** OS process, sharing a model load's
allocator, and that cumulative MLX peak grew `7.07 → 7.07 → 9.24 → 11.25 GB` across
those blocks.

> Does the `W = 4` gain reproduce across genuinely independent OS processes, and can
> the effect be explained by process state accumulating between runs?

Nothing else is investigated.

## 2. Hypotheses

**H1 (replication)** The gain reproduces at the same magnitude under real process
boundaries, correctness holds, and no memory or state accumulation explains it.

**H0 (confound)** The gain shrinks or destabilises once each replicate starts in a
clean process, meaning part of E15's effect came from a warm allocator, a warm
compiled-body cache, or other persistent structure rather than from grouping.

## 3. Process isolation

Every independent replicate runs in a **new OS process**, spawned by the parent,
and is **fully terminated** before the next begins. Model, runtime and allocator
share no process state between replicates. The parent performs no model work of any
kind; it spawns, collects one JSON line per child, and waits for exit.

**Five real processes per comparison condition.** A condition is one
`(workload, plan)` pair. Four workloads × two plans × five processes = **40 child
processes**.

## 4. Arms

Exactly two, as measured in E15 and unchanged:

| Arm | Definition |
| :-- | :-- |
| `sequential` | each request executed and synchronised on its own, run to completion |
| `grouped4` | `AsyncGroupedB1(W = 4)`: up to four independent batch-1 decode steps built and submitted without an intermediate barrier, completed as one group |

No other width is measured.

## 5. Workloads

The four E15 workloads, byte-identical in definition, imported rather than
re-specified: `homogeneous`, `heterogeneous`, `staggered`, `terse`. Both
`StrictOneShotPlan` and `ReusableSessionPlan`, treated separately. The executor
never switches or alters a request's plan.

## 6. Warmup, repeats, ordering

- **Warmup**: two full sweeps of both arms per child, after prefill, before any
  measurement. E15 established that the first grouped rounds pay a one-time
  allocator build-up, and a fresh process pays it every time — which is exactly the
  cost this experiment must not misattribute.
- **Repeats**: three measured repetitions of each arm per child.
- **Order**: arm order randomised per repetition, seeded `20260825 + process index`.

## 7. Metrics

**Primary throughput metric** — total workload wall clock; effect
`G = 1 − T_grouped4 / T_sequential`, computed **per process** and aggregated across
processes.

**Latency** — measured from each request's defined arrival time, identically for
both arms: `ttft + Σ inter-token − arrival`. This is E15's corrected definition
(**M1**) and is used from the start here; the admission-based figure that produced
E15's wrong `+418%` is not computed at all.

Also: TTFT, queue wait, inter-token latency, and p50 / p95 of full request latency.
Throughput and per-request latency are reported separately, and group time is never
divided by group width.

**Memory, per child** — recorded at four checkpoints: process start, after model
load, after prefill, after warmup, and after the final repeat; plus after **every**
repeat, so growth within a process is a series rather than two endpoints.

| Quantity | Source |
| :-- | :-- |
| Resident set size | `ps -o rss=` on the child's own pid |
| MLX active memory | `mx.get_active_memory()` |
| MLX buffer cache | `mx.get_cache_memory()` |
| MLX peak memory | `mx.get_peak_memory()` |
| Compiled-body cache size | number of entries the engine holds |

## 8. Correctness

Hard, as in E15. Within each child, every `grouped4` request is compared against
that child's own `sequential` reference under the same plan:

1. exact token IDs
2. token count
3. stop reason
4. SHA-256 of the logically valid KV region

Additionally, and new to E16 because only real process boundaries make it testable:

5. **Cross-process determinism** — the sequential reference token sequences must be
   identical across all five processes of a condition. If a fresh process produces
   different output, process state was influencing results and every earlier
   conclusion is affected.

Any failure forces `CORRECTNESS_FAILURE` regardless of performance.

## 9. Statistics

Uncertainty is computed **primarily across processes**. For each condition, one
`G` per process from that process's median wall clock per arm; then a
10,000-resample bootstrap over the five process-level values, seed `20260825`.
Five clusters give a coarse interval and it is reported as such. Within-process
repeat spread is reported separately and never used as the primary interval.

## 10. Success criterion and classification (frozen, ordered)

Threshold `θ = 0.10`. Main workloads for the criterion: `homogeneous`,
`heterogeneous`, `staggered` — as in E15. `terse` is reported but not required,
because E15 already showed it fails under the strict plan.

Accumulation checks, all three required to clear H0:

- **A1** RSS after the final repeat exceeds RSS after warmup by no more than **10%**
- **A2** MLX active memory after the final repeat exceeds that after warmup by no
  more than **10%**
- **A3** `G` computed from the first repetition alone and from the last repetition
  alone differ by no more than **3 percentage points**

| # | Condition | Class |
| --: | :-- | :-- |
| 0 | any correctness or cross-process determinism failure | `CORRECTNESS_FAILURE` |
| 1 | a child aborts, or fewer than five processes complete for any condition | `INCONCLUSIVE` |
| 2 | `G ≥ θ` with interval excluding 0 in all three main workloads under both plans, **and** A1–A3 all hold | `REPLICATED` |
| 3 | as (2) but `G` is more than 5 percentage points below the corresponding E15 value | `REPLICATED_WITH_SMALLER_EFFECT` |
| 4 | `G ≥ θ` everywhere required but any of A1–A3 fails | `CONFOUNDED_BY_PROCESS_STATE` |
| 5 | otherwise | `NOT_REPLICATED` |

Rule 3 is checked before rule 2 is reported as a clean replication: a gain that
qualifies but has halved is not the same finding.

**If the effect shrinks markedly or becomes unstable across processes, the cause is
located before anything else is attempted.** No optimisation follows from E16 in
either direction.

## 11. Stop rules

No interim analysis. A pilot with two processes on one condition validates the
harness only, and its numbers are never interpreted. The main run is analysed
exactly once. An inconclusive result is not extended inside E16.

Abort, preserving partial evidence and never retrying: another local model process
detected before spawning, power source not AC, any child exceeding 12 GiB, or total
wall time above 60 minutes. A child that crashes is recorded as a failed replicate
rather than respawned.

## 12. Known risks

1. **Five clusters is a coarse interval.** Stated with every number.
2. **A fresh process pays the cold allocator cost every time**, which warmup is
   meant to absorb. If warmup is insufficient the grouped arm is penalised, biasing
   against H1 — the conservative direction, and it is stated rather than corrected.
3. **RSS on macOS includes shared and file-backed pages**, so it is a coarse
   instrument; the MLX counters are the finer ones and both are reported.
4. **Model load per child dominates wall time.** That is the price of real
   isolation, not a finding.
5. **This replicates one width on one machine and one model.** No claim beyond it.
