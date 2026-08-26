# E14b — Separating submission/sync amortisation from true batched execution

**Experiment ID** `E14b`
**Frozen at commit** `c6e7f69`
**Branch** `forge/hardware-autotune`
**Registered** 2026-08-25, before any E14b measurement was taken or viewed.

Frozen. Thresholds, arms and classification are not adjusted after results are seen.

## 0. Why this is E14b and not a revision of E14

`E14` is already preregistered (`d2b1a05`), run and committed. It compared
sequential batch-1 execution against true batching and could therefore **not**
separate the two candidate mechanisms — a limitation this design exists to remove.
Overwriting a frozen preregistration would break the rule the whole ledger depends
on, so E14 stands with its result and E14b is a new experiment.

E14's usable outputs are carried forward as measured inputs, not repeated:

| Carried from E14 | Value | Status |
| :-- | --: | :-- |
| Per-dispatch cost inside the real graph (positive control, `R²=0.998`) | `9.246 µs` | MEASURED |
| Per-step synchronisation cost (sync-amortisation probe) | `2.06 ms` | MEASURED |
| Batched prefill logits bit-identical to unbatched | all rows | MEASURED |
| Kernel count per decode step | — | **RETIRED**, see ledger C2b |

Per correction C2b, no dispatch time is derived from a kernel count in E14b, and
E14's condition 9.5 is not repeated.

## 1. Research question

> Of the improvement seen when several requests are processed together, what share
> comes from amortised host submission and synchronisation, and what share from
> genuine batched execution with larger shapes?

`B > 1` has never been studied before E14. E3 varied sequence length `L`, not the
batch dimension, so **no earlier multi-token result counts as evidence about
batching** and none is treated as such.

## 2. Hypotheses and competing explanations

**H-sub** A material part of the gain is amortised host submission and
synchronisation, obtainable without changing tensor shapes at all.

**H-shape** A material part comes from larger shapes: different kernel selection,
better device occupancy, changed memory behaviour.

They are not exclusive. The three-arm design exists because arm A against arm C
alone cannot tell them apart, and because E14 measured exactly that pair.

## 3. Arms

All arms perform the **same logical work**: `b` independent sequences, each at
context length `L = 1024`, each advanced by one teacher-forced decode step.

| Arm | Construction |
| :-- | :-- |
| **A — Sequential B1** | `b` independent batch-1 executions, each submitted and **synchronised on its own** |
| **B — Async B1 Group** | the same `b` independent batch-1 executions, built and submitted **without any intermediate barrier**, one `mx.async_eval` over the group and a **single** synchronisation afterwards. **Tensor shapes are unchanged**; only submission and synchronisation are amortised |
| **C — True Batch** | the same `b` sequences executed together in a real batch dimension. Adds shape, kernel-selection, occupancy and memory-layout effects on top of whatever B has |

Arm B is the discriminator. Without it, a gain of C over A is uninterpretable.

**Batch sizes** `b ∈ {1, 2, 4, 8}`, memory permitting. At `b = 1` all three arms are
the same execution and serve as a consistency check on the harness.

## 4. Fixed conditions

Model `mlx-community/gemma-3-4b-it-4bit`, revision `93724907`, weights and 4-bit
group-64 quantisation unchanged. Knobs are the stored tuned profile
(`compiled_fixed_cache`, `head_skip_prefill`, `fuse_projections`,
`fused_argmax=False`). Context length, decode length and total logical work are
constant across arms. Nothing in `forge/` is modified. **No scheduler is built.**

Workload: eight distinct SQuAD v1.1 documents (already vendored, SHA-256
`95aa6a52…6972c9`), each rendered through the same chat template and truncated at
token level to exactly 1024 tokens so every sequence shares one cache offset.
Capacity `= ceil64(1024 + 8 + 8) = 1088`, identical in every arm.

AC power required; thermal state and load recorded.

## 5. Timing definitions

Four timestamps per measured execution, after a full barrier:

| Quantity | Definition |
| :-- | :-- |
| `host_prep` | `t0 → t_prep`: building the graph on the host, before anything is submitted |
| `submission` | `t_prep → t_submit`: `mx.async_eval(...)` returning |
| `completion_wait` | `t_submit → t_done`: from submission returning until `mx.eval` + `mx.synchronize` complete |
| `total` | `t0 → t_done` |

**`completion_wait` is not called GPU time.** Device work may already have begun
during submission, so this quantity is a wait, not an execution time, and is
reported under that name throughout.

Also recorded: time per request, time per generated token, aggregate tokens per
second, p50 and p95 inter-token latency, MLX peak memory, thermal and power state.

Lazy evaluation is respected: every timer stops only after `mx.eval` followed by
`mx.synchronize`.

## 6. Harness validation (before the main run)

1. **Timer control on an empty queue** — the same timing machinery around a trivial
   one-element operation, establishing the noise floor of the instrument itself.
2. **Deliberately forced synchronisation after each single execution** — this is
   arm A by construction; the control requires arm A to be measurably slower than
   arm B at `b ≥ 2`, otherwise the instrument cannot see synchronisation at all.
3. **Full barrier before every measurement block** — `mx.eval` + `mx.synchronize`
   on a sentinel before `t0`, so no earlier work leaks into a block.

If control 1 shows a noise floor above 5% of the batch-1 total, or control 3 cannot
be established, the result is `INCONCLUSIVE`.

## 7. Execution and statistics

Four fresh processes. Within each process, measurement blocks are **randomised and
interleaved** with a fixed seed (`20260825 + process index`), so arm order cannot
carry drift. 2 warmup repetitions and 7 measured repetitions. Every individual
sample is stored, not only summaries.

Medians per process, then paired across processes with a 10,000-resample bootstrap,
seed `20260825`. With four processes the interval is coarse and is reported as such.

Equal numbers of logical requests and tokens are compared throughout. **Higher
aggregate throughput is never presented as lower single-request latency**; the two
are reported in separate rows.

## 8. Correctness

Separately from the timing blocks, each sequence is generated for real (greedy,
8 tokens) in arm C and in a batch-1 run, and compared on:

- token IDs, element by element
- token count
- stop reason
- numerical difference of the logits, bitwise via unsigned integer views

A difference is recorded as an **execution-plan divergence** and the paths are then
not called interchangeable. **E14b derives no quality claim from it** — that is
E13's question and needs E13's design.

## 9. Success criteria and classification

Primary size **`b = 4`**. Threshold **`θ = 0.10`**, and every gain must have a
paired bootstrap interval excluding zero.

- `G_B  = 1 − T_B(4) / T_A(4)` — submission and synchronisation amortisation
- `G_CB = 1 − T_C(4) / T_B(4)` — additional effect of true batching
- `G_C  = 1 − T_C(4) / T_A(4)` — total

Ordered decision rule:

| # | Condition | Class |
| --: | :-- | :-- |
| 0 | harness control 1 or 3 fails, or relative IQR on `T_A(1)` > 0.10 | `INCONCLUSIVE` |
| 1 | `G_B ≥ θ` and `G_CB ≥ θ`, both intervals excluding 0 | `MIXED_MECHANISM` |
| 2 | `G_B ≥ θ` with interval excluding 0 | `SUBMISSION_SYNC_AMORTIZATION_SUPPORTED` |
| 3 | `G_CB ≥ θ` with interval excluding 0 | `TRUE_BATCH_SHAPE_EFFECT_SUPPORTED` |
| 4 | `G_C ≥ θ` with interval excluding 0, neither component qualifying | `MIXED_MECHANISM` |
| 5 | otherwise | `MECHANISM_NOT_SUPPORTED` |

Additional reading, reported but not part of the class: if per-request `submission`
does not fall with `b` while per-request `completion_wait` does, the mechanism sits
at device or shape level rather than on the host.

**A concrete dispatch time may be stated only if it follows directly from this
timing design.** Otherwise it stays INFERRED and the earlier "~4.5 ms" remains
withdrawn.

## 10. Stop rules

No interim analysis. The pilot validates the harness only and its numbers are never
interpreted; after it the measurement plan and inputs are frozen. The main run is
analysed exactly once. An inconclusive result is not extended inside E14b.

Abort, preserving partial evidence and never retrying: another local model process
(`gpu_busy()`), power source not AC, MLX peak above 12 GiB, or main-run wall time
above 45 minutes. E14 peaked at `11.71 GB`, so memory is a live constraint and the
cache is cleared between blocks.

## 11. Known risks

1. **`mx.async_eval` may not return before device work completes.** Then
   `submission` absorbs completion and the split is unusable. Detected by
   `submission` growing with device work; reported as unusable rather than used.
2. **Arm B may not actually avoid intermediate barriers** if MLX inserts its own.
   Detected as arm B not differing from arm A at any `b`; that is a real finding
   about the runtime, not a harness failure, and is reported as such.
3. **Equal-length prompts are the favourable case for batching.** Ragged lengths
   need per-sequence offsets and can only be worse. A negative result generalises;
   a positive one does not.
4. **Four processes give a coarse interval.** Stated with every interval.
5. **`b = 8` holds roughly 1.2 GB of KV cache.** The memory result is part of the
   finding, not a footnote.
6. **Single model, single machine, single context length, one decode step per
   sequence in the timing blocks.** No claim beyond that.
