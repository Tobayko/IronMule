# E14 — Is the remaining decode latency fixed dispatch overhead?

**Experiment ID** `E14`
**Frozen at commit** `e1c29f0`
**Branch** `forge/hardware-autotune`
**Registered** 2026-08-25, before any E14 measurement was taken or viewed.

Frozen. Any necessary change gets a new experiment ID. Thresholds, workloads and
classification rules are not adjusted after results are seen.

---

## 1. What is already known, and what is not

Not repeated here, because it is already measured and recorded:

| Established | Where | Status |
| :-- | :-- | :-- |
| Decode step ≈ `11.86 ms`, weights ≈ 2.18 GB per step | E3, E11 | MEASURED |
| Achieved GEMV bandwidth saturates near `324 GB/s` | E4 | MEASURED |
| Token-width sweep 1,2,4,8,16 on one sequence | E3 | MEASURED, but see below |
| M=8 is a pathological kernel regime | E2, E3 | MEASURED, reproduced twice |
| `dispatch_us = 6.41` from a chained tiny-kernel microbenchmark | `forge/hw.py` | MEASURED **in isolation only** |
| "~510" and "~700–750" kernels per decode step | E5 entry, later reporting | **INFERRED**, inconsistent, see correction C2 |
| "`~4.5 ms` of the step is dispatch" | derived from the two lines above | **INFERRED**, never measured |

**Never measured at all:** the batch dimension `B > 1` with independent requests;
CPU submission time separated from GPU execution; the cost of a dispatch inside the
real graph rather than in isolation; the cost of per-step synchronisation separated
from everything else.

E3's width sweep **is** re-measured here, for one specific reason: every E3 sample
enclosed an `eval` + `synchronize` round trip, and that round trip is precisely the
quantity under investigation. E3's numbers are sound for what E3 asked and unusable
for what E14 asks.

## 2. Hypothesis and competing explanations

**H1 (dispatch)** A substantial part of the fixed per-step decode cost is runtime
dispatch and scheduling, and it is amortised by wider execution.

**H0** It is not. The fixed cost is dominated by streaming 2.18 GB of weights, which
is a hardware floor no scheduler can remove.

Competing explanations, each with the evidence that discriminates it:

| Explanation | Discriminating evidence in this experiment |
| :-- | :-- |
| Fixed dispatch amortised | CPU submission time per step is large and roughly constant in width; the positive control gives a per-dispatch cost of the right order to explain the residual |
| Weight streaming amortised | the fitted fixed cost `a` is close to the measured floor `F`, leaving a small residual |
| Larger shapes use the hardware better | per-token time keeps falling well past the point where any fixed cost could be amortised; marginal cost per token is far below the isolated cost at that shape |
| Fewer synchronisation points | the sync-amortisation probe (§5.4) isolates this directly |
| Different kernel selection | non-monotonic curve, already seen at M=8 in E2 and E3 |
| Changed memory layout | batched and unbatched arrangements differ in achieved bandwidth per byte |
| Measurement artifact | positive control, plus agreement between the synchronised and amortised timings |

**Kernel count cannot be measured.** MLX exposes no dispatch or kernel counter;
`mx.metal.start_capture` writes an Xcode trace that is not machine-readable here.
Any statement involving a kernel count therefore stays **INFERRED** in E14 as well,
and is labelled as such wherever it appears.

## 3. Conditions held fixed

Model `mlx-community/gemma-3-4b-it-4bit`, revision `93724907`, weights and 4-bit
group-64 quantisation unchanged. Knobs are the machine's stored tuned profile:
`compiled_fixed_cache=True`, `head_skip_prefill=True`, `fuse_projections=True`,
`readback_every=1`, `fused_argmax=False`, `speculate_k=0`, `wired_fraction=0.0`.
Nothing in `forge/` is modified. **No scheduler is built.**

AC power required. Thermal state and load recorded per run.

## 4. Workload

Real shapes and real content. Eight distinct SQuAD v1.1 documents (the dataset
already vendored for E13, SHA-256 `95aa6a52…6972c9`), each rendered through the same
chat template and then **truncated at the token level to exactly `L = 1024`
tokens**, so every sequence in a batch shares one cache offset. Truncation may end
mid-sentence; that is irrelevant to a latency measurement and is stated rather than
hidden.

`L = 1024` is chosen because it is a realistic context length and sits exactly on
the sliding-window boundary already characterised in E12.

Capacity `= ceil64(L + max_new_tokens + 8)`, identical in every arrangement.
`max_new_tokens = 8` for the sequential arrangements.

## 5. Arrangements

All four use the same model, the same capacity and the same prompts.

| Symbol | Arrangement | Logical tokens produced |
| :-- | :-- | --: |
| `S(n)` | batch 1, width 1, `n` sequential decode steps on one sequence | `n` |
| `W(w)` | batch 1, width `w`, one forward | `w` positions |
| `B(b)` | batch `b`, width 1, one forward, all sequences at the same offset | `b` |
| `U(b)` | `b` independent sequences, each stepped once, sequentially | `b` |

Widths and batch sizes: **1, 2, 4, 8.**

### 5.1 Equal-logical-work comparisons

- `B(b)` against `U(b)` for `b ∈ {2,4,8}` — the microbatch question. Same number of
  logical tokens, same sequences, batched versus not.
- `W(w)` against `S(w)` for `w ∈ {2,4,8}` — the multi-token question. Same number of
  positions processed.

`W(w)` processes `w` positions of one sequence given known inputs; it does not
autoregressively produce `w` usable tokens. That semantic difference is stated
wherever `W` appears and is never presented as single-request speedup.

### 5.2 Throughput versus latency

Reported separately and never conflated:

- aggregate tokens per second
- latency per request
- time per generated token per sequence
- p50 and p95 inter-token latency

**Higher throughput is never presented as lower single-request latency.**

### 5.3 Positive control (instrument validation)

`K` chained 8×8 matmuls are added inside the compiled decode body and returned as an
**extra output**, so they cannot be eliminated and cannot touch model numerics.
`K ∈ {0, 64, 128, 256}`. The slope `dt/dK` is the **measured** marginal cost of one
dispatch inside the real graph, replacing the isolated `6.41 µs`.

The control passes if the slope is positive and approximately linear
(`R² ≥ 0.9` over the four points). If it does not, the primary result is
`INCONCLUSIVE`, because the instrument cannot be shown to respond to dispatch.

### 5.4 Sync-amortisation probe

`n = 8` sequential steps, teacher-forced with a fixed token so no host readback is
needed, run twice: synchronising after every step, and synchronising once at the
end. The difference divided by `n` is the **measured** per-step cost of
synchronisation plus readback, `Δsync`. This isolates one competing explanation
directly rather than by inference.

## 6. Measurements

Per repetition, for every arrangement:

- `t_total` — start to after `mx.synchronize()`, fully synchronised
- `t_submit` — start to `mx.async_eval(...)` returning: **CPU/runtime submission**
- `t_gpu` — `t_total − t_submit`: GPU execution not overlapped with submission
- per-step times, p50 and p95 inter-token latency for sequential arrangements
- aggregate tokens/s, per-request latency, ms per token per sequence
- MLX peak memory
- thermal state, power source, load average
- generated tokens per sequence

Lazy evaluation is respected: every timer stops only after `mx.eval` followed by
`mx.synchronize`. All individual samples are stored, not only summaries.

**Kernel, dispatch and command-buffer counts are not measured** and are not
reported as if they were; see §2.

## 7. Statistical treatment

Four fresh processes, arrangement order alternating between them, 2 warmup
repetitions and 7 measured repetitions per arrangement per process. Medians per
process, then paired across processes with a 10,000-resample bootstrap, seed
`20260825`, for every paired comparison in §5.1.

Linear fits `t(x) = a + b·x` are reported with `R²`, and a fit with `R² < 0.9` is
reported as not describing the data rather than used for inference — the E3 width
fit already failed this way and must not be repeated silently.

## 8. Correctness

Generated tokens are stored per sequence for every arrangement. `B(b)` is compared
against `U(b)` on identical prompts, token by token.

If different batch widths produce different tokens, that is recorded as an
**execution-plan divergence**; the paths are then explicitly not called
interchangeable. **E14 derives no quality claim from this** — quality is E13's
question and needs E13's design.

Numerical correctness of the batched primitive is checked directly: the batched
forward's logits for row `i` are compared against the unbatched forward's logits for
the same sequence, bitwise via unsigned integer views, exactly as in E12.

## 9. Classification (frozen, ordered)

Let `T1` be the median fully-synchronised batch-1 width-1 step time;
`a_B` the fitted fixed cost from `t_B(b) = a_B + b_B·b`;
`F = 2.18 GB / 324 GB/s = 6.73 ms`, the weight-streaming floor from E4 (MEASURED);
`R = a_B − F` the residual fixed cost;
`Δsync` from §5.4; `p` the per-dispatch slope from §5.3.

**`DISPATCH_MECHANISM_SUPPORTED`** requires all five:

1. per-logical-token time falls by **≥ 25%** from `b = 1` to `b = 4` in `B`
2. `a_B / T1 ≥ 0.60` — the step is mostly fixed cost
3. `R / a_B ≥ 0.25` — a substantial part of that fixed cost is not weight streaming
4. the positive control passes (§5.3)
5. `R − Δsync` is within a factor of 3 of `p × 700`, i.e. the residual is the right
   order of magnitude for dispatch at the INFERRED kernel count. This criterion is
   explicitly an order-of-magnitude consistency check against an inferred number,
   not a measurement, and is labelled so in the result.

**`DISPATCH_MECHANISM_NOT_SUPPORTED`** if condition 1 fails, or if
`R / a_B < 0.10` — the fixed cost is essentially all weight streaming, a hardware
floor no scheduler can remove.

**`MIXED_MECHANISM`** if condition 1 holds but not all of 2–5: widening pays, but
dispatch is not shown to be the dominant reason.

**`INCONCLUSIVE`** if the positive control fails, if the submission and synchronised
timings disagree irreconcilably, or if the relative interquartile spread on `T1`
exceeds 10%.

A better batch-4 time on its own is **not** a positive finding. The data must show
that fixed runtime cost per logical token falls dependably with width.

## 10. Stop rules

No interim analysis. A pilot validates the harness only and its numbers are never
interpreted as evidence; after it, the measurement plan and workload are frozen.
The main run is analysed exactly once. An inconclusive result is not extended inside
E14; that requires a new experiment ID.

Abort, preserving partial evidence and never retrying: another local model process
(`gpu_busy()`), power source not AC, MLX peak above 12 GiB, or main-run wall time
above 45 minutes.

## 11. Known risks

1. **The batched and unbatched paths may not be numerically identical**, in which
   case throughput and latency are still comparable but the paths are not
   interchangeable. Recorded, not resolved here.
2. **`t_submit` may not isolate what it is meant to.** If MLX's `async_eval` blocks
   on a full queue, submission time absorbs GPU time. Detectable as `t_submit`
   scaling with GPU work; checked, and if it happens the split is reported as
   unreliable rather than used.
3. **Equal-length prompts are the favourable case for batching.** Ragged lengths
   would need per-sequence offsets and would only make batching worse. A negative
   result here therefore generalises; a positive one does not.
4. **Kernel count remains inferred.** Criterion 9.5 rests on it and is labelled
   accordingly; the classification does not hinge on it alone.
5. **Batch 8 at capacity 1088 holds roughly 1.2 GB of KV cache.** Within limits on
   this machine, but the memory result is part of the finding, not a footnote.
6. **Single model, single machine, single context length.** No claim beyond it.
