# Experiment Ledger

Status vocabulary: MEASURED, REPRODUCED, NOT_REPRODUCED, PARTIALLY_REPRODUCED,
HYPOTHESIS, INFERRED, REJECTED, OPEN.

Raw data for every entry lives in `research/raw/<ID>.json`. Negative results are
never removed.

| ID | Question | Result | Status |
| :-- | :-- | :-- | :-- |
| E0a | Do the predecessor project's cycle 16/19 mechanisms hold in this fork? | 915.88 -> 790.27 ms, `-13.71%` | REPRODUCED |
| E0b | Does projection fusion pay? | `+2.4%` decode, 3 unpaired repeats | NOT_REPRODUCED (see E5) |
| E0c | Does prompt-lookup speculation pay? | `2.9x` slower, acceptance 0.17 | MEASURED |
| E1 | Where does prefill time go? | 99.4% one trunk forward | MEASURED |
| E2 | What does `quantized_matmul` deliver vs M? | ~4.5 TFLOPS ceiling; M=8 pathological | MEASURED |
| E3 | How does a decode step scale with width? | non-linear; `ms/token` best at width 4 | MEASURED |
| E4 | Is achieved GB/s limited by matrix size? | yes, 104 -> 324 GB/s over 1.4 -> 360 MB | MEASURED |
| E5 | Is fusion a real decode win? | running | OPEN |

---

## E0a — Do the inherited mechanisms hold here?

**Observation** the predecessor project measured cycle 16 (`mx.compile` on a fixed-shape
cache, `-7.04%`) and backlog 19 (last-position-only prefill projection,
`-15.3615%`) but never combined them.
**Change** Both, plus greedy selection inside the compiled body, on one path.
**Result** `915.88 -> 790.27 ms` end to end. Prefill baseline median `638.37 ms`
against cycle 16's `0.638376521 s`; head skip `-16.0%` against backlog 19's
`-15.3615%`.
**Correctness** Identical logical tokens, deterministic across repeats.
**Decision** Kept. **Status** REPRODUCED — two independent calibration points.
**Learning** The fork's harness produces numbers comparable with the parent
project, so parent findings can be built on rather than re-derived.

## E0b — Projection fusion, first look

**Hypothesis** Concatenating q/k/v and gate/up along the output axis cuts five
matmuls per block to two and should therefore be faster.
**Result** Decode `253.98 -> 260.15 ms`, i.e. `+2.4%`. Prefill `536.28 -> 531.14`.
**Correctness** Bit identical by construction, verified on CPU against the
unfused model, and token identical end to end.
**Decision** Not kept, but the measurement was three unpaired repeats in one
process — too weak to conclude. Re-run as E5.
**Status** NOT_REPRODUCED. **Next** E5.

## E0c — Prompt-lookup speculation

**Hypothesis** The planner copies `persistent_service_qualification` verbatim out
of the prompt, so an n-gram draft should be accepted often and cut decode steps.
**Result** `speculate_k=4`: decode `252.94 -> 735.37 ms`, acceptance `0.17`.
k=8 and k=12 are worse still.
**Correctness** Token identical in every arm — the greedy verification is exact.
**Decision** Rejected for this workload. **Status** MEASURED.
**Learning at the time** "A five-token forward costs 2.9x a one-token forward, so
`quantized_matmul` has a fast path only at M=1." **This explanation was wrong**;
see E2 and E3. The observed cost is real, the mechanism was not.

## E1 — Where does prefill time go?

**Observation** Prefill is 68% of end-to-end time at the best known configuration.
**Hypotheses** A compute-bound 4-bit GEMM · B fixed-state copy · C cache
construction · D output projection · E allocation.
**Experiment** `_prefill` split into phases with `eval` + `synchronize` between
each; 2 warmups, 5 repeats, one process.
**Result**

| phase | head_skip | full_head |
| :-- | --: | --: |
| `make_cache` | 0.05 ms | 0.05 ms |
| `trunk_forward` | **533.78 ms** | 533.71 ms |
| `projection` | 1.53 ms | 103.97 ms |
| `argmax` | 0.37 ms | 0.44 ms |
| `fixed_state_build` | 1.47 ms | 1.56 ms |

Instrumented sum `537.20 ms` against uninstrumented `537.23 ms`, perturbation
`1.0000x`. First token `2717` in every arm.
**Decision** Hypotheses B, C, E REJECTED. D quantified: head skip is worth
`102.4 ms`, which is the whole `-16%`.
**Status** MEASURED.
**Learning** Prefill is one thing only. The `prefill_into_fixed` knob, which was
expected to be worth several percent, can be worth at most `1.47 ms` — it was
killed before it was ever benchmarked.
**Next** Is `533.78 ms` near the primitive's limit? -> E2.

## E2 — What does `quantized_matmul` deliver as a function of M?

**Experiment** Isolated `mx.quantized_matmul` at the model's own shapes,
M in {1,2,4,8,16,32,64,128,256,322,512,1024}, 4 bit / group 64, 3 warmups,
7 repeats, median. Correctness against a dequantised reference per shape
(`rel_err` 6.4e-3 .. 8.6e-3, consistent with 4-bit quantisation).
**Result** Every shape converges to **~4.5 TFLOPS**. The model's prefill achieves
**3.86 TFLOPS = 86% of that ceiling**.
**Trap found** The absolute low-M numbers are `eval`+`synchronize` round trips.
A linear fit over the clean large-M region gives a fixed offset of **~0.49 ms per
call**, which is the entirety of the "M=1" reading. Any conclusion drawn from
low-M absolute values in this sweep is invalid.
**Decision** "Framework overhead dominates prefill" REJECTED. "Dedicated M=1 fast
path" REJECTED. **Status** MEASURED.
**Learning** M=322 costs consistently ~10% more per token than the local trend
between 256 and 512 — a tile-alignment effect. Padding to 384 does not pay
(`~4.74 ms` predicted against `4.43 ms` measured at 322).
**Next** The offset makes the decode question unanswerable here -> E3.

## E3 — How does one decode step scale with input width?

**Experiment** Real compiled fixed-cache forward at widths 1,2,4,8,16, with and
without the output projection, 3 warmups, 7 repeats, median. The difference
between the two is overhead-free.
**Result**

| width | full | trunk | `lm_head` | ms/token |
| --: | --: | --: | --: | --: |
| 1 | 11.909 | 10.602 | 1.307 | 11.909 |
| 2 | 18.133 | 15.885 | 2.248 | 9.066 |
| 4 | 30.307 | 25.803 | 4.504 | **7.577** |
| 8 | 66.856 | 56.669 | 10.186 | 8.357 |
| 16 | 71.807 | 61.801 | 10.005 | 4.488 |

Linear fit rejected: `R^2 = 0.83` and the fitted intercept `13.999 ms` exceeds the
measured width-1 time, which is impossible for a real fixed cost.
**Decision** Hypothesis C (kernel selection changes with M) CONFIRMED. The jump
between width 4 and 8 and the near-flat step from 8 to 16 reproduce E2's
independent observation that M=8 is a pathological point.
**Status** MEASURED, and REPRODUCED across two independent experiments.
**Learning** E0c's stated mechanism is refuted. Multi-token forwards are *more*
efficient per token, not less: width 4 costs `7.58 ms/token` against `11.91`.
Speculation failed on acceptance (`0.17`), not on kernel behaviour. Break-even at
width 2 is 1.52 accepted tokens per forward.
**Learning 2** `lm_head` moves 377 MB in `1.307 ms` = **288 GB/s**, while the
34-layer trunk moves 1.97 GB in `~10.1 ms` = **195 GB/s**. Same step, same
hardware, different achieved bandwidth.
**Next** Is 195 GB/s a ceiling or a gap? -> E4.

## E4 — Is achieved bandwidth limited by weight-matrix size?

**Experiment** M=1 `quantized_matmul`, K=2560, N from 1024 to 262144. Calls
chained inside one `eval` so launch cost is amortised, and at least 512 MB of
*distinct* weight buffers per point so the system level cache cannot flatter the
result. 3 warmups, 7 repeats.
**Result**

| matrix | GB/s |
| --: | --: |
| 1.4 MB | 103.7 |
| 3.5 MB | 169.4 |
| 7.0 MB | 205.4 |
| 14.1 MB | 240.8 |
| 28.1 MB | 253.6 |
| 56.2 MB | 298.8 |
| 360 MB | **323.6** |

**Decision** Hypothesis A CONFIRMED: achieved bandwidth rises monotonically with
matrix size and saturates near 320 GB/s. **Status** MEASURED.
**Learning 1** The trunk is not inefficient, it is made of small matrices.
`k_proj`/`v_proj` are 1.4 MB and run at ~104 GB/s; `lm_head` is 360 MB and runs
at ~324 GB/s.
**Learning 2** The earlier roofline claim of `5.45 ms` at 400 GB/s is REJECTED.
Nothing measured on this machine exceeds 324 GB/s. The practical decode floor is
**~7.3 ms**, so real headroom is ~36%, not ~52%.
**Learning 3** `forge/hw.py`'s `read_bandwidth_gbps` probe is invalid: `mx.sum`
over 256 MB reports 175 GB/s while real matmuls reach 324 GB/s, so the reduction
limits it, not the memory system. Replaced.
**Next** E4 predicts fusion should be worth ~6% (qkv 1.4/2.8 -> 5.6 MB,
gate_up 14.1 -> 28.2 MB) while E0b measured `+2.4%`. Contradiction -> E5.

## E5 — Is projection fusion a real decode win?

**Experiment** Paired A/B, 6 fresh processes, alternating arm order, 2 warmups and
7 measured generations per arm, 10,000-resample bootstrap on the paired ratios.
**Result**

| metric | ratio | 95% CI | effect |
| :-- | --: | :-- | --: |
| total | 0.9921 | [0.9909; 0.9977] | `-0.79%` |
| prefill | 0.9890 | [0.9881; 0.9898] | `-1.10%` |
| decode | 0.9990 | [0.9961; 1.0145] | none |

Token identity and determinism held in every arm of every process.
**Decision** Hypothesis A (`-5%` decode from E4's bandwidth model) REJECTED.
Hypothesis C (fusion hurts; E0b was real) REJECTED — `+2.4%` was noise.
Hypothesis B (neutral decode) confirmed, CI contains 1.0.
**Status** MEASURED. Fusion is KEPT for its prefill effect, which is small but has
a tight interval and costs nothing.
**Learning** The win is in prefill, where M=322 is compute bound and a larger GEMM
schedules better — not in decode, where the step is bandwidth bound.
**Mechanism** Fusion removes 3 matmul dispatches per layer but adds ~5 `mx.split`
slices; with `dispatch_us = 6.41` measured, `-102` matmuls and `+170` slices is a
net `+68` kernels, so the bandwidth gain and the dispatch loss cancel. E4's
isolated-kernel bandwidth advantage does not transfer to a pipelined graph.
**Learning 2** `dispatch_us = 6.41` and ~510 kernels per decode step gives
`3.3 ms`, which accounts for the trunk's gap between `195 GB/s` achieved and
`324 GB/s` measured ceiling. The decode model is now self consistent.

## E6 — Can a prefix KV snapshot be reused?

**Experiment** 4 requests sharing a 128-token instruction prefix, suffixes 96-124
tokens. Arm A prefills the whole prompt per request; arm B prefills the prefix
once, snapshots, and per request restores and prefills only the suffix. A
tokenisation gate asserts `tokenize(prefix)` is an exact token prefix of every
full prompt before anything is measured.
**Result**

| request | suffix | full TTFT | reuse TTFT | ratio | identical |
| --: | --: | --: | --: | --: | :-- |
| 0 | 124 | 389.97 ms | 205.86 ms | 0.5279 | yes |
| 1 | 99 | 387.89 ms | 204.03 ms | 0.5260 | yes |
| 2 | 116 | 389.47 ms | 205.23 ms | 0.5269 | yes |
| 3 | 96 | 341.32 ms | 154.47 ms | 0.4526 | yes |

4/4 token identical, both arms deterministic.
**Decision** Hypothesis A confirmed on this workload. **Status** MEASURED.
**Learning** This contradicts prior backlog item 1, closed as
`candidate_correctness_failed` in cycle 1 under the growing-cache architecture.
Under a fixed-shape cache the same idea passes its token gate. A closed backlog
item is reopened.
**Restore is free** `mx.slice_update` is functional, so a snapshot's arrays stay
valid however often a *copy* of the state structure is advanced. Restoring is
rebuilding a dict of references, not copying 53 MB.
**Next** 4/4 is not a guarantee -> E7.

## E7 — Is prefix reuse exact, or lucky?

**Experiment** Per decode step, full-prefill and reuse logits compared directly:
max absolute difference over the whole vocabulary, top1-top2 margin, and argmax
agreement. Both arms stepped in lockstep on the same token so the comparison stays
defined even under divergence. 4 requests, all steps.
**Result**

| request | steps | bit equal | argmax equal | max abs diff | min margin | margin/diff |
| --: | --: | :-- | :-- | --: | --: | --: |
| 0 | 23 | no | yes | 4.000 | 5.2500 | 2.4 |
| 1 | 23 | no | yes | 3.562 | 7.0000 | 2.6 |
| 2 | 23 | no | yes | 2.500 | 7.5000 | 3.8 |
| 3 | 23 | no | yes | 4.750 | 6.2500 | 2.5 |

**Decision** Hypothesis A (bit identical, safe by construction) REJECTED.
Logits differ by up to 4.75 units — roughly 16-32 bfloat16 ULPs accumulated over
34 layers under a different reduction order, not rounding noise.
**Status** MEASURED.
**Learning** Prefix reuse is argmax-identical here with a safety factor of only
2.4-3.8x against a deliberately conservative bound (the max is taken over the
whole vocabulary, while what actually decides the token is the perturbation at the
top-2 entries). It cannot be enabled blindly. This is also the most likely reason
prior cycle 1 saw a correctness failure on a different workload.
**Next** Calibrate the real risk metric and stress the sample -> E8.

## E8 — Calibrating the flip risk of prefix reuse

**Experiment** 276-token fixed prefix (66.8% of the prompt), 12 varying suffixes.
Per decode step: the top1-top2 gap in the full arm, the same gap in the reuse arm,
their difference, and whether the reuse arm's argmax flipped.
**Result** 10/12 requests token identical. **2 flips over 254 steps.**

| request | flips | full TTFT | reuse TTFT | ratio | min gap (reuse) | max gap change |
| --: | --: | --: | --: | --: | --: | --: |
| 0 | 1 | 627.1 | 258.5 | 0.4122 | **0.000** | 0.500 |
| 3 | 1 | 627.8 | 258.2 | 0.4113 | **-0.500** | 1.625 |
| others | 0 | ~628 | ~258 | ~0.411 | 1.25 .. 5.50 | 0.63 .. 2.25 |

Decisive headroom over all steps: min `-0.5`, p5 `8.0`, median `52.2`.
TTFT ratio median `0.4112`.
**Decision** Hypothesis C confirmed: prefix reuse against a single-shot baseline
FAILS the token identity gate. **Status** MEASURED, REJECTED as an unconditional
optimisation.
**Learning** Flips happen precisely where the model is near indifferent. Request 0
had `gap_full = 0.000` — an exact tie in the *baseline* — so any perturbation at
all decides that token. This also explains prior backlog item 1's cycle-1
`candidate_correctness_failed` rather than contradicting it.
**Next** The perturbation comes from the suffix running at `L=137, offset=276`
instead of inside an `L=413, offset=0` forward. If the baseline used the same
chunking, would the difference vanish? -> E9.

## E9 — Does a chunked execution plan make reuse exact?

**Experiment** Three arms per request over 12 requests: single-shot prefill,
chunked prefill split exactly at the prefix boundary, and prefix reuse. Logits
compared pairwise at every decode step.
**Result**

| comparison | max abs diff | verdict |
| :-- | --: | :-- |
| chunked vs reuse | **0.0000** | bit identical, 12/12 requests, every step |
| single-shot vs chunked | 4.3125 | plans differ, as expected |

Cost: single-shot `628.5 ms` -> chunked `691.9 ms` = `1.1010x` on a cold request.
Reuse `258.4 ms` = `0.3734x` of chunked.
**Decision** Hypothesis A CONFIRMED, hypothesis C confirmed as the stated cost.
**Status** MEASURED.
**Learning — the central result of this cycle.** The obstacle was never the
arithmetic, it was that baseline and candidate used *different execution plans*.
Fix the plan as "prefill in chunks split at the cache boundary" and prefix reuse
becomes bit identical by construction. The correctness gate is then passed
structurally, not statistically, and no margin, threshold or fallback is needed.
**Cost that must be stated** A chunked plan does not produce the same tokens as a
single-shot plan. Correctness is therefore only meaningful *relative to a declared
plan*. This is the same wall prior cycle 2 hit when it searched for a
"length-independent safe block size" — there is none, because bfloat16 addition is
not associative. The constructive resolution is to stop looking for one and let
the runtime declare its plan instead.
**Economics** at a 66.8% shared prefix: break-even after 2 requests (`-24%`),
`-52%` at 10 requests, asymptotically `-58.9%` prefill and `-41.7%` end to end.
**Next** Implement as a runtime feature and measure it paired -> E10.

## E10 — The prefix cache as a shipped runtime feature

**Experiment** 12-request session, 276-token declared prefix (66.8% of the prompt),
6 fresh processes, alternating arm order, two warmup sessions then two measured
sessions per arm, 10,000-resample bootstrap.
**Result**

| arm | session | TTFT median |
| :-- | --: | --: |
| `single_shot` | 10575.2 ms | 628.3 ms |
| `prefix_cache` | **6566.9 ms** | **258.7 ms** |

Session ratio `0.6218`, 95% CI `[0.6152; 0.6234]`, **`-37.82%` end to end**.
**Correctness** Two independent sessions produced identical tokens for all 12
requests; both arms deterministic across all 6 processes.
**Decision** Accepted. **Status** MEASURED.

**Two corrections that keep this honest.**

1. The script creates a fresh `PrefixCache` per session, so both the "cold" and
   "warm" sessions are one miss followed by eleven hits — which is why they land
   within 5 ms of each other. The gate is a determinism gate between two
   independent sessions, not a cold-versus-warm comparison. The `-37.82%` is
   therefore the steady-state figure for a session of this shape, which is what
   was intended, but the labels in the raw file are wrong.

2. `0/12` requests differing between plans does **not** contradict E8's two flips.
   The two experiments measure different implementations. E8's reuse arm took its
   prefix KV from mlx_lm's standard cache (`RotatingKVCache` on the sliding
   layers) and copied it into the fixed layout; E9 and E10 keep the prefix in a
   fixed cache throughout. Different mask shape, different reduction order. E9 and
   E10 agree with each other: chunked and single-shot plans differ by up to
   `4.31` logits yet chose the same token on every step of these 12 requests.
   **`0/12` is luck, not a guarantee.** E8 established that this workload contains
   near ties — one with `gap_full = 0.000` exactly. The only guaranteed statement
   remains E9's: *within* the chunked plan, reuse is bit identical.

## E11 — Does the autotuner find this on its own?

**Experiment** `python -m forge.tune`: coordinate descent over 9 knobs on the
322-token single-request workload, 2 warmups and 5 repeats per candidate, each
candidate gated on exact token identity; then a 6-process paired A/B to confirm
the screening winner before it is stored.
**Result** Baseline `924.95 ms` (prefill `639.36`, decode `285.83`), 23 tokens.

| knob | ratio | verdict |
| :-- | --: | :-- |
| `compiled_fixed_cache=True` | 0.9726 | kept |
| `fused_argmax=True` | 0.9724 | rejected, no gain |
| `head_skip_prefill=True` | 0.8637 | kept |
| `prefill_into_fixed=True` | 0.8720 | rejected |
| `readback_every=2` | 0.8604 | rejected, below threshold |
| `readback_every=4/8` | 0.8809 / 0.8771 | rejected |
| `speculate_k=4` | 1.3873 | rejected |
| `capacity_slack=128` | 0.8640 | rejected |
| `wired_fraction=0.6` | 0.8633 | rejected |
| `fuse_projections=True` | 0.8581 | kept |

Confirmation A/B: ratio `0.8548`, 95% CI `[0.8522; 0.8555]`, tokens identical,
accepted. Stored plan: `compiled_fixed_cache + head_skip_prefill + fuse_projections`.
**Status** MEASURED.
**Learning** The screening pass independently reproduced four hand-run results —
`prefill_into_fixed` worse (E1), `speculate_k` catastrophic (E0c),
`fuse_projections` the best single knob (E5), `capacity_slack` worthless — without
being told any of them.
**Learning 2** It rejected `fused_argmax`, which every hand-run experiment in this
cycle had enabled. At `0.9724` against `0.9726` it buys 0.02%, far below the 0.5%
keep threshold. Neutral rather than wrong, but the tuner was stricter than its
author.
**Honest limit** `readback_every=2` scored `0.8604` against the running best
`0.8637` — a 0.38% gain, rejected for being under threshold. Single-process
screening at 5 repeats cannot resolve 0.4%, so this is correctly rejected as
unresolvable rather than kept by accident. Confirming it would need the paired
harness.

## E12 — Falsification test at the sliding-window boundary

**Preregistration** `research/raw/E12_preregistration.md`, frozen at commit
`750be38`, SHA-256 `5d0dbc3ccc66084a237f9bf2af051f0643d179a3a05afdbb744bba30dff4890e`,
committed as `fab2fc1` before any measurement was taken. Harness `ccb0ea7`.
Not edited afterwards.

**Purpose** To destroy E9's plan-internal bit identity, not to confirm it. Gemma 3
slides its attention window at 1024 tokens on 29 of its 34 layers. Every E9 prefix
was 276 tokens, far below the point where the window clips anything, so lengths at
and above 1024 were the cheapest available counterexample.

**H0** chunked-no-reuse and chunked-reuse are bit identical at every tested prefix
length. **H1** a reproducible difference appears at or above the window.

### Execution

| Stage | Runs | Cases | Wall |
| :-- | :-- | --: | --: |
| Pilot, `L=1024` | 1 process | 1 | 111 s |
| Stage 1 screening, 13 lengths × 2 types | 1 fresh process | 26 | 1594 s |
| Stage 2a confirmation, 6 lengths × 2 types | 3 fresh processes | 36 | 820 / 819 / 818 s |

No stage aborted, no resource or wall limit reached, peak MLX memory 6.08 GB
against a 12 GiB limit.

### Comparison A — the correctness test

**`PLAN_INTERNAL_EXACT`.** 63 of 63 cases, **756 requests**, **14,369 decode steps**
compared, **zero** failure records. Every prefix length passed: 276, 768, 870, 896,
1000, 1023, 1024, 1025, 1048, 1152, 1280, 1536, 2048. SHA-256 hashes over the valid
KV region agreed in every case.

Equality was tested on raw bits throughout, via unsigned integer views.
`mx.array_equal` was rejected as the gate because it reports `-0.0 == 0.0` as true
while the bit patterns differ, which would have let a real difference through.

At 1023, 1024 and 1025 the prompt text and the capacity are identical and only the
split point moves, which is the cleanest available form of the test.

### The detector was proven able to fail before any of this was counted

63/63 PASS is equally consistent with reuse being exact and with a broken
comparison, so three positive controls ran first.

| Control | Result |
| :-- | :-- |
| One mantissa bit flipped, layer 7 `keys`, prefix position 3 | KV check **fires**, localises to layer 7, `keys`, flat index 779 (`3 × 256 + 11`), exactly 1 differing element, hashes differ |
| Same flip, effect on logits | **no change at all** |
| Whole key vector clobbered, global layer 5, position 3 | logits move `9.8750` |
| Whole key vector clobbered, global layer 5, position 1022 | logits move `11.5000` |
| Whole key vector clobbered, sliding layer 7, position 3 | logits move `2.0625` |
| Whole key vector clobbered, sliding layer 7, position 1022 | logits move `8.7251` |
| One bit flipped, sliding layer 7, position 1022 | logits move `1.65625` |

**Two self-corrections belong in the record.**

1. From the single-bit control's null result on logits I first concluded that
   prefix content beyond the window is dead weight. That is **wrong and is
   retracted**. The window applies per query position: during prefill, every query
   below position 1027 still reads position 3. Clobbering that position moves the
   logits by `2.06` even on a sliding layer. The single-bit null was a signal
   magnitude effect below bfloat16 resolution through a 1194-way softmax, not
   evidence that the value is unread.
2. The control script's own verdict field `logit_detector_validated` printed
   `False` because I had written it to require *both* poisonings to be detected.
   That criterion was badly chosen. One positive detection is sufficient to prove
   the comparison is not stuck returning true, and seven were obtained. The raw
   field is kept as printed; this is the corrected reading.

Global layers do read early positions, so there is no masking defect in the fixed
cache. The logit comparison fires in every path it was exercised in.

### Comparison B — plan divergence, documentation only

`PLAN_DIVERGENCE`. Recorded, never allowed to touch the primary class.

| | max abs diff |
| :-- | --: |
| Below the window | `7.625` |
| At or above the window | `25.5` |
| Requests choosing different tokens | **140 of 756** |

The raw figure the harness printed during the run (up to `106.69`) is inflated and
must not be quoted: each arm decodes on its own argmax, so once the two plans pick
different tokens the comparison is between two different contexts.
`e12_summarise.py` recomputes it up to the first step at which the plans still
agree, and the figures above are the corrected ones.

Divergence is roughly three times larger above the window than below it, which is
preregistration risk 5 realised: the `single_shot` arm converts a standard mlx_lm
cache whose sliding layers use `RotatingKVCache` and therefore rotate above 1024,
while the fixed cache does not.

**140 of 756 requests answering differently between two plans of the same model is
the strongest single piece of evidence this programme has produced for the E9
thesis.** Correctness is not a property of the code. It is a property of the
declared plan.

### Secondary measurements (never optimised for)

| Prefix | Cold TTFT | Reuse TTFT | Ratio |
| --: | --: | --: | --: |
| 276 | 749.1 ms | 300.2 ms | 0.401 |
| 768 | 1483.6 ms | 352.9 ms | 0.238 |
| 1024 | 1820.6 ms | 306.9 ms | 0.169 |
| 1280 | 2204.4 ms | 310.0 ms | 0.141 |
| 2048 | 3393.4 ms | 317.4 ms | **0.094** |

Reuse cost is nearly flat in prefix length while cold cost is linear in it, so the
ratio improves with the prefix share. Snapshot build cost equals one cold prefix
prefill and is paid once.

### Status and claim

**Status** MEASURED. H0 survived a deliberate attempt to break it.

**The claim, bounded exactly as it was earned:** for this model
(`gemma-3-4b-it-4bit`, revision `93724907`), this quantisation, MLX `0.32.0` on this
M1 Max, and this declared chunked execution plan, prefix KV reuse is reproducible
bit-exactly by construction across prefix lengths from 276 to 2048, spanning the
1024 sliding-window boundary, over 756 requests and 14,369 decode steps in five
independent processes.

**No claim is made** about other models, other execution plans, other MLX builds,
prefix lengths beyond 2048, special or control tokens, or batch sizes above one.

**What this does not prove.** Preregistration risk 1 stands and is not dissolved by
the result. Comparison A's candidate replays a snapshot of exactly the computation
its baseline performs, so the informative failure modes were buffer aliasing under
`mx.compile` donation and any window- or offset-dependent behaviour in a restored
cache. Neither occurred. That is a real property worth having — aliasing was a live
risk — but it is a statement about this implementation, not a proof that prefix
reuse is exact in general.

### Consequence for the runtime contract (proposed, not implemented)

Two named plans, declared by the caller, never selected by the autotuner:

- `StrictOneShotPlan` — prefill in one forward. The plan to use when output must
  match the untuned path.
- `ReusableSessionPlan(prefix_ids)` — prefill chunked at the declared boundary.
  Within it, a cache hit is bit exact. Its output does **not** match
  `StrictOneShotPlan`, measured at up to `25.5` logits apart and 140 of 756
  requests answering differently, so switching plans is a behaviour change and must
  be an explicit decision.

The existing `PrefixCache` already refuses to be a tuner knob for this reason. The
contract makes the same rule legible at the API surface rather than in a comment.

### Next data-driven research question

Comparison B is now the open question, not Comparison A. 140 of 756 requests
answering differently between two plans of the same unmodified model is far larger
than anything this programme has measured, and it is unquantified in the direction
that matters: **is either plan better, or are they merely different?** That needs a
quality measure on a task with a known correct answer, not a logit distance. Until
that exists, plan selection is a latency decision being made in ignorance of its
output cost.

### Raw data

`research/raw/E12_preregistration.md`, `E12_environment.json`,
`E12_results_pilot.json`, `E12_results_stage1.json`, `E12_results_confirm1.json`,
`E12_results_confirm2.json`, `E12_results_confirm3.json`,
`E12_positive_control.json`, `E12_positive_control2.json`,
`E12_positive_control3.json`, `E12_failures.json` (0 records), `E12_summary.json`.

## E13 — The quality cost of the execution plan

**Preregistration** `research/raw/E13_preregistration.md`, frozen at commit
`ad8815f`, SHA-256 `0fa9621c7ea1f4d14980fb9955ddc9a48a0982ee8201dd099361aabf0bbd73d3`,
committed as `cdc782d` before any measurement. Harness and frozen set `9c41948`.
Not edited afterwards.

**Question** E12 left one thing open, and it was not a performance question. Two
plans of the same unmodified model answered with different tokens in 140 of 756
requests. Different is not worse, and logit distance cannot decide which is which.

### Design

**Dataset** SQuAD v1.1 dev, SHA-256 `95aa6a52…6972c9`, vendored under
`research/data/`. Human-written questions, human-annotated extractive answers, so
**neither the model nor the experimenter decides what is correct.** Natively
session shaped: one natural document, many independent questions.

**Selection** mechanical and content blind — articles sorted by title, three
reserved for the pilot, bands assigned by index `mod 3`, paragraphs accumulated
until the prefix lands in band, first 8 questions in document order. Yield: **44
contexts, 352 questions**; one exclusion (`Doctor_Who`, band overshoot) exactly as
the preregistration anticipated.

| Band | n | Prefix tokens | Relation to the 1024 window |
| :-- | --: | :-- | :-- |
| `SHORT` | 15 | 522–824 | entirely below |
| `NEAR` | 14 | 926–1106 | straddles it: 9 below, 5 at or above |
| `LONG` | 15 | 1151–1431 | entirely at or above |

**Scorer** validated first, ten controls, all passing. The two that matter: gold
`art` against `started restarting` scores **incorrect** because containment is
tested on normalised token sequences rather than raw substrings; and a shotgun
prediction listing four candidates scores **correct**, which is the preregistered
bound on containment and is reported rather than hidden.

**Pilot** ran end to end at accuracy `0.9167`, which does not cross the
preregistered adjustment thresholds of `>0.95` or `<0.35`, so nothing was adjusted.
No pilot comparison between plans was interpreted.

### Primary result

**`REUSABLE_NONINFERIOR`.**

| | Containment accuracy |
| :-- | --: |
| `StrictOneShotPlan` | 0.8097 |
| `ReusableSessionPlan` | 0.8068 |
| Paired difference | **−0.0028** |
| 95% CI, paired cluster bootstrap over 44 contexts | **[−0.0114; +0.0057]** |
| Preregistered margin | −0.05 |

CI lower bound `−0.0114 > −0.05`, so non-inferiority holds. The interval turned out
far tighter than the 3-point half-width planned for, because the two plans agree on
almost every question and most per-context differences are exactly zero. **The data
therefore exclude a loss larger than 1.14 percentage points**, which is a much
stronger statement than the margin required.

Accuracy landed at `0.81`, not at the ceiling the pilot suggested, so risk 3 did not
materialise and power was not the limiting factor.

### Discordance and divergence

| | |
| :-- | --: |
| Strict correct, reusable wrong | **2** |
| Reusable correct, strict wrong | **1** |
| Answer token divergence | 20 / 352 = **5.68%** |
| Divergences that changed correctness | **3** |
| Divergences where both answers stayed correct | 12 |

The three discordant questions, in full:

| Direction | Case | Gold | Strict | Reusable |
| :-- | :-- | :-- | :-- | :-- |
| strict only | `French_and_Indian_War` | "May 1754" | "May 1754" | "1754" |
| strict only | `Economic_inequality` | "the basis of the methodology used" | "…on the basis of the methodology used" | "The methodology used…" |
| reusable only | `Huguenot` | "granted the Huguenots substantial religious, political and military autonomy" | "granted religious, political and military autonomy" | the gold string exactly |

**One of the three is not a quality difference at all.** In `Economic_inequality`
both answers name the same cause; strict merely happened to include the word
"basis" that the gold span starts with. Of the two that are real, one goes each
way: reusable is less precise on the date, and more precise on the Huguenot clause.

### Secondary

Analysed only after the primary, and never used to assign the class.

| Metric | Strict | Reusable | Difference | 95% CI |
| :-- | --: | --: | --: | :-- |
| Containment (primary) | 0.8097 | 0.8068 | −0.0028 | [−0.0114; +0.0057] |
| Exact match | 0.7131 | 0.7244 | **+0.0114** | [−0.0028; +0.0284] |
| Token F1 | 0.8380 | 0.8426 | **+0.0046** | [−0.0014; +0.0117] |
| Answer NLL (lower better, coverage 0.83) | 0.5871 | 0.5833 | −0.0038 | — |

**The sign of the tiny effect depends on which metric is used.** Containment has
reusable a hair worse; exact match, token F1 and answer NLL all have it a hair
better. That is what an absent difference looks like, and it is the reason the
conclusion is stated as a bounded non-inferiority rather than as "the plans are the
same".

| Band | n | Strict | Reusable | Difference | 95% CI |
| :-- | --: | --: | --: | --: | :-- |
| `SHORT` | 15 | 0.8333 | 0.8333 | +0.0000 | [+0.0000; +0.0000] |
| `NEAR` | 14 | 0.7946 | 0.8036 | +0.0089 | [+0.0000; +0.0268] |
| `LONG` | 15 | 0.8000 | 0.7833 | −0.0167 | [−0.0417; +0.0000] |

**This table looks like a length trend and is not one.** The entire band structure
is three individual questions: two that reusable lost in `LONG`, one it gained in
`NEAR`, none in `SHORT`. Decision rule 0 did not fire, correctly — neither band's
interval excludes zero, and both endpoints sit exactly at `0.0000` because the
bootstrap distribution is discrete when almost every per-context difference is zero.
No length dependence is claimed. It is the obvious thing to power properly next.

**Divergence mechanism.** Token divergence concentrates in `LONG` (12 of 20, against
5 in `SHORT` and 3 in `NEAR`), consistent with more accumulated numerical difference
over a longer prefix. It occurs almost immediately when it occurs at all: 13 of 20
at the very first answer token, none later than index 4. The top1−top2 gap at the
divergence point is small (n=7 measurable, median `0.50`, max `1.00`), which
reproduces E8's finding that plans part exactly where the model is near indifferent.

### Performance (secondary, measured, not optimised for)

| | Strict | Reusable | Ratio |
| :-- | --: | --: | --: |
| TTFT, median | 1556.2 ms | 69.2 ms | **0.0445** |
| Session, median (8 questions) | 13298.6 ms | 2712.5 ms | **0.2040** |

`−79.6%` session time here against `−37.8%` in E10, because this workload has a much
higher prefix share: a long document with eight short questions. The two numbers
describe different workloads and must not be quoted interchangeably.

### Claim, bounded to what was tested

For `gemma-3-4b-it-4bit` at 4-bit group 64, MLX 0.32.0 on this M1 Max, greedy
decoding, 24 output tokens, extractive question answering on Wikipedia prose with
document prefixes of 522 to 1431 tokens spanning the 1024 sliding-window boundary:
**`ReusableSessionPlan` is not inferior to `StrictOneShotPlan`, with any accuracy
loss bounded above by 1.14 percentage points at 95% confidence.**

**No claim** about other models, other task families (summarisation, reasoning,
code, multi-turn), prefixes beyond 1431 tokens, sampling other than greedy, or
absolute correctness — agreement between plans is not evidence that either is right.

**Known limits.** SQuAD v1.1 is public and probably in the model's training data,
which inflates absolute accuracy in both arms but cannot bias a paired difference on
identical questions. Containment is gameable by a long prediction, bounded by the
24-token cap and quantified by the shotgun control; exact match and F1 are reported
alongside and agree. 44 clusters cannot resolve a sub-percentage-point effect.

### Consequence for the runtime contract

Unchanged in the part that matters: **the plan stays an explicit caller decision and
the tuner may never switch it.** E12 established that the plans genuinely disagree;
E13 establishes that the disagreement does not cost measurable accuracy on this task
family. Those are different facts and the second does not license automating the
first — a workload outside the tested domain has no evidence behind it.

What changes is what can honestly be written next to the option: the `-79.6%` here
and `-37.8%` in E10 can now be offered with a measured quality bound attached
instead of an open question.

### Raw data

`research/raw/E13_preregistration.md`, `E13_frozen_set.json`,
`E13_scorer_controls.json`, `E13_results_pilot.json`, `E13_results_main.json`,
`E13_summary.json`, `E13_discordance.json`, `research/data/squad-dev-v1.1.json`.

## Documentary corrections to earlier entries (no raw data or criteria changed)

Made during the E14 pre-check, 2026-08-25. Raw files and preregistrations are
untouched; only this ledger's prose is corrected, and the original wording is quoted
so both remain visible.

### C1 — E13's divergence breakdown was incomplete

The E13 entry reported 20 divergent answers, of which 3 changed correctness and 12
left both plans correct. **It did not state the third category.** Recomputed from
`research/raw/E13_discordance.json`, unchanged:

| Of the 20 divergent answers | n |
| :-- | --: |
| both plans correct | 12 |
| correctness changed (2 strict-only, 1 reusable-only) | 3 |
| **both plans wrong** | **5** |
| sum | 20 |

The missing five matter for interpretation: a quarter of all divergences occurred on
questions neither plan answered correctly, so they carry no information about
relative quality in either direction. The correctness-relevant divergence rate is
therefore `3/352 = 0.85%`, not the `5.68%` headline, and the headline should never be
quoted as a quality figure. Non-divergent answers: 332 of 352.

### C2 — The kernel count per decode step was quoted inconsistently

The E5 entry states "~510 kernels per decode step". A later recount from the model
source, block by block, gave roughly 22 kernels per transformer block over 34 blocks
plus embedding, final norm, output projection and argmax, i.e. **~700–750**. The
ledger was never reconciled.

Both figures are **INFERRED** from reading source, neither is measured. The
`6.41 µs` dispatch cost behind them comes from a chained-tiny-kernel microbenchmark
in isolation, not from the real graph. Every downstream statement built on
`count × 6.41 µs` inherits that status. E14 exists partly to replace this with a
measurement or to retire it.

### C3 — The SQuAD contamination limitation was imprecise

Original wording: *"SQuAD v1.1 is public and probably in the model's training data,
which inflates absolute accuracy in both arms but cannot bias a paired difference on
identical questions."*

That conflates bias with sensitivity. The corrected statement:

Contamination does not bias the **direction** of the paired difference, because both
plans answer identical questions. It plausibly **attenuates its magnitude**, for two
reasons: a memorised answer is more robust to a small numerical perturbation than a
freshly derived one, and higher absolute accuracy compresses the range in which a
difference could show. The measured bound of `1.14` percentage points is therefore a
bound **for this contaminated evaluation set**, and may be optimistic for material
the model has not seen. Nothing about the `REUSABLE_NONINFERIOR` verdict changes;
its stated validity domain narrows to what was actually tested.

## Documentary corrections, second pass (C1b–C3b)

Refines C1–C3 to the wording requested 2026-08-25. Raw data and preregistrations
remain untouched; only ledger prose changes.

### C1b — E13 divergence, full four-way split

| Of the 20 divergent answers | n |
| :-- | --: |
| both plans correct | 12 |
| both plans wrong | 5 |
| only Strict correct | 2 |
| only Reusable correct | 1 |
| **total** | **20** |

Seventeen of the twenty carry no information about relative quality: twelve where
both plans were right, five where both were wrong. Only three are informative. The
correctness-relevant divergence rate is `3/352 = 0.85%`; the `5.68%` headline is a
token-identity figure and must never be quoted as a quality figure.

### C2b — Kernel count is an unresolved inference, not a starting value

The ledger states "~510 kernels per decode step"; a later manual count from the
model source gave "~700–750". **Neither was ever measured.** MLX exposes no
machine-readable kernel or dispatch counter.

**These numbers are not to be used as an established starting value, and no precise
dispatch time may be derived from them.** Every earlier statement of the form
"about 4.5 ms of the step is dispatch" is retired as unfounded arithmetic, not
merely relabelled.

This retroactively weakens one criterion of the already-frozen E14
preregistration: its condition 9.5 multiplied the measured per-dispatch cost by the
inferred count of 700. That criterion is not used going forward. E14's verdict does
not depend on it — conditions 2 and 3 failed independently — but the criterion
should not have been written that way, and it is not repeated in E14b.

### C3b — SQuAD contamination, precise formulation

The paired design rules out a **different distribution of tasks between the plans**:
both plans answer byte-identical questions, so no task-selection difference can
arise.

What contamination can still do is affect **model confidence**, and through it the
**sensitivity of the experiment to execution-plan divergence**. A memorised answer
is held with a larger margin between the leading candidates, and E8 established
that plans diverge precisely where that margin is small. A contaminated evaluation
set therefore has systematically fewer opportunities to diverge than unseen
material would.

The measured bound of `1.14` percentage points is consequently a bound **for this
evaluation set**. The `REUSABLE_NONINFERIOR` verdict is unchanged; its validity
domain does not extend to material the model has not seen.

## E14 — Is the remaining decode latency fixed dispatch overhead?

**Preregistration** `research/raw/E14_preregistration.md`, frozen at `e1c29f0`,
SHA-256 `13f6d358…400a2`, committed as `d2b1a05` before measurement.

**Result `DISPATCH_MECHANISM_NOT_SUPPORTED`**, and the design limitation that
produced E14b is stated first: E14 compared sequential batch-1 execution against
true batching and nothing in between, so it could not separate amortised submission
from a shape effect.

| Measured | Value |
| :-- | --: |
| Per-dispatch cost inside the real graph, positive control, `R²=0.9978` | **9.246 µs** |
| Per-step synchronisation, sync-amortisation probe | **2.06 ms** |
| Fitted fixed per-step cost `a_B` from `t(b) = a + b·b`, `R²=0.9905` | **1.806 ms** |
| Marginal cost per batch row | 9.067 ms |
| Batched prefill logits bit-identical to unbatched | yes |
| Relative IQR on the batch-1 step | 0.0136 |

The decisive number is `a_B = 1.806 ms`, **below** the weight-streaming floor
`F = 6.73 ms`. A fixed cost cannot sit below the time needed to read the weights, so
the assumed decomposition "fixed cost = weights + dispatch" is wrong: there is no
large additive fixed block for a scheduler to amortise. Condition 9.5, which
multiplied the measured per-dispatch cost by an inferred kernel count of 700, is
**retired** under correction C2b and the verdict does not rest on it — conditions 2
and 3 failed independently.

The preregistered submit/GPU split was declared **unusable** by its own diagnostic:
submission grew `6.76×` from `B1` to `B8` while the completion side grew `4.70×`,
and the batch-1 completion time (`6.11 ms`) is below the weight floor, so device
work had already begun during submission. E14b replaces it with a four-way split.

**Status** MEASURED. **Raw** `E14_preregistration.md`, `E14_results_pilot.json`,
`E14_results_main.json`, `E14_summary.json`.

## E14b — Separating submission/sync amortisation from true batching

**Preregistration** `research/raw/E14b_preregistration.md`, frozen at `c6e7f69`,
SHA-256 `564c3906…8712`, committed as `282ea98` before measurement. Harness frozen
after pilot validation.

**Three arms on identical logical work** — `b` independent sequences at `L = 1024`,
one teacher-forced decode step each:
`A` sequential batch-1, each synchronised on its own · `B` the same executions
grouped under one `async_eval` and one synchronisation, **shapes unchanged** ·
`C` the same sequences in a real batch dimension.

**Harness controls all pass.** Timer noise floor `0.3136 ms = 2.42%` of the batch-1
total (ceiling 5%); arm A visibly slower than arm B at every `b > 1`; the three arms
agree within 2% at `b = 1`, where they are the same execution; relative IQR `0.0249`.

### Result `MIXED_MECHANISM`

| b | `G_B` submission + sync | `G_CB` additional true batch | `G_C` total |
| --: | :-- | :-- | :-- |
| 2 | `+12.19%` [+11.78; +12.40] | `+13.05%` [+12.65; +13.40] | `+23.46%` |
| **4** | **`+18.02%`** [+17.49; +18.32] | **`+20.05%`** [+19.79; +20.57] | **`+34.47%`** |
| 8 | `+16.12%` [+16.06; +16.65] | `+13.19%` [+12.96; +14.09] | `+27.43%` |

Both mechanisms qualify at every batch size, intervals excluding zero. At the
primary size the gain splits roughly evenly: about eighteen points from grouping
submissions **without touching tensor shapes at all**, and about twenty more from
real batching on top.

### The mechanism is overlap, not cheaper host work

The four-way timing split makes this unambiguous, per request:

| b | A submit | B submit | C submit | A wait | B wait | C wait |
| --: | --: | --: | --: | --: | --: | --: |
| 1 | 6.211 | 6.237 | 6.228 | 6.113 | 6.125 | 6.138 |
| 2 | 6.267 | 7.818 | 4.529 | 6.112 | 3.050 | 5.086 |
| 4 | 6.319 | 8.368 | 4.150 | 6.127 | **1.842** | 4.275 |
| 8 | 6.316 | 9.159 | 5.827 | 6.142 | **1.267** | 3.595 |

**Arm A amortises nothing**: submission and wait per request are flat at ~6.3 and
~6.1 ms at every batch size, and total per request stays at 13.1 ms.

**Arm B's host submission per request does not fall — it rises**, from 6.24 to
9.16 ms, while completion wait per request collapses from 6.13 to 1.27 ms. Grouping
therefore does not make host work cheaper; it lets device execution overlap with the
next submission. That is exactly the preregistered reading "submission does not fall
per request but completion wait does → the mechanism sits at device level", and it
reconciles with E14: there was never a large fixed additive block to remove, which
is why `a_B` came out at 1.81 ms.

**Arm C does reduce host submission per request** (6.32 → 4.15 ms at `b = 4`),
because one batched graph replaces `b` separate graphs, and reduces the wait as
well.

### Throughput against single-request latency, kept apart

| | aggregate | latency of the batch | per request |
| :-- | --: | --: | --: |
| `C1` | 77.65 tok/s | 12.879 ms | 12.879 ms |
| `C4` | **116.24 tok/s** | **34.410 ms** | 8.603 ms |
| `C8` | 105.08 tok/s | 76.130 ms | 9.516 ms |

Batch 4 raises aggregate throughput by 50% and raises the latency a caller waits for
by 2.7×. `C8` is **worse than `C4`** on throughput — the M=8 regime, now reproduced a
third time independently after E2 and E3.

### Correctness and execution-plan divergence

56 sequences compared against their own batch-1 run. Prefill logits bit-identical
everywhere. Generated token counts equal everywhere.

Generated token IDs: identical at `b = 2` (8/8) and `b = 4` (16/16). At `b = 8`,
**one sequence — row 3 — differs by exactly one token**, at index 6 of 8
(`1437` against `1580`, converging again at index 7), and it does so **in all four
processes**. Deterministic, reproducible, confined to `b = 8`.

Batched execution at `b = 8` is therefore **not interchangeable** with batch-1
execution on this workload. **E14b derives no quality claim from this**; that is
E13's question and needs E13's design.

### What remains INFERRED

Kernel count per decode step (retired under C2b) and therefore any absolute dispatch
time. The `9.246 µs` from E14's control is a **marginal** cost for added serial work,
not a total, and is not multiplied by anything.

### Is a microbatch scheduler the justified next step?

**Not as the first move.** Arm B delivers `+18.02%` at `b = 4` **without a batch
dimension, without changed shapes, without per-sequence offsets, without ragged
handling — and with zero token divergence at any batch size.** Arm C's additional
`+20.05%` costs a reproducible execution-plan divergence at `b = 8`, a 2.7× increase
in the latency a caller waits for, and a throughput regression beyond `b = 4`.

The evidence supports grouped asynchronous submission as the cheaper, lower-risk
half of the gain, and puts true batching behind it as a separate decision with a
measured correctness cost. Neither is built here.

**Status** MEASURED. **Raw** `E14b_preregistration.md`, `E14b_results_pilot.json`,
`E14b_results_main.json`, `E14b_summary.json`.

## E15 — Does async grouped B1 survive a real service workload?

**Preregistration** `research/raw/E15_preregistration.md`, frozen at `c2c8a59`,
SHA-256 `939a3c40…0a92`, committed as `204a0cc` before measurement.

**Result `ASYNC_B1_SERVICE_VIABLE`** — but three methodological facts come first,
because two of them changed what the numbers mean and one changed the verdict.

### Corrections and deviations, stated before the result

**M1 — The latency metric was wrong, and fixing it flipped the verdict.**
Stored `latency_ms` started at *admission*. In the sequential arm a request is
admitted when it begins running, not when it arrives, so its queueing time was
silently omitted while the grouped arm counted it from `t = 0`. Recomputed from the
same raw data as `ttft + Σ inter-token − arrival`, identically for both strategies:

| | worst p95 inflation at `W = 4` |
| :-- | --: |
| as first computed (admission based, wrong) | `+418.84%` |
| corrected (arrival based) | **`−4.00%`** |

The frozen rule — full response latency p95, limit 10% — is unchanged; only its
computation was repaired. The verdict moved from
`THROUGHPUT_GAIN_WITH_LATENCY_COST` to `ASYNC_B1_SERVICE_VIABLE` as a result, and
that is recorded rather than presented as the outcome all along.

**M2 — "Four fresh processes" were four blocks inside one OS process.**
Each block does a fresh model load and builds fresh states, and measurement order is
randomised within it, but the OS process is shared. The evidence is direct:
cumulative MLX peak grew `7.07 → 7.07 → 9.24 → 11.25 GB` across the four blocks, so
allocator state carries over. The paired bootstrap over "processes" therefore has
less independence than the preregistration claims. Within-block arm comparison is
unaffected, since drift hits every arm in a block alike. The same loose
implementation is present in E14 and E14b.

**M3 — The first main run aborted, and the first repair silently failed.**
Run one hit `20.97 GB` against the 12 GiB guard and aborted after one block; the
cause was a prefill cache I had added to save time, holding six request sets of
eight 187 MB KV states. No performance number was inspected before the repair, and
the raw file is preserved as `E15_results_main_aborted.json`. The first repair
commit (`427db84`) claimed a fix that was not in the tree — the patch asserted on a
pattern that no longer matched and wrote nothing — and the relaunch ran the same
defect until it was stopped. Corrected and verified in `d71761d`; peak fell to
`7.07 GB`.

### Pilot findings that shaped the frozen workloads

The pilot rejected the first workload: an extractive-span instruction produced 2–3
token answers, leaving nothing to group, and grouped `W = 2` came out 26% *slower*
purely because groups were never filled. Main workloads now ask for one sentence
(10–30 tokens). The terse case is kept as its own workload rather than discarded.

It also caught a measurement artifact: the first three grouped rounds at `W = 4`
cost `412 / 315 / 252 ms` against a `43.5 ms` steady state — a one-time allocator
build-up for `W` simultaneous KV states. Warmup now covers every measured width, and
the cold start is reported separately as the one-time cost it is.

### Throughput

| Workload / plan | sequential | `W = 4` | `G(W4)` | 95% CI | realised width |
| :-- | --: | --: | --: | :-- | --: |
| homogeneous / strict | 74.85 tok/s | 89.86 | `+16.66%` | [+16.35; +16.73] | 3.97 |
| homogeneous / reusable | 75.87 | 91.62 | `+17.13%` | [+17.11; +17.19] | 4.00 |
| heterogeneous / strict | 75.35 | 89.16 | `+15.52%` | [+15.31; +15.78] | 3.32 |
| heterogeneous / reusable | 74.75 | 88.83 | `+16.18%` | [+15.30; +16.29] | 3.53 |
| staggered / strict | 75.22 | 88.98 | `+15.40%` | [+14.97; +15.79] | 3.15 |
| staggered / reusable | 74.74 | 87.86 | `+14.99%` | [+14.47; +15.25] | 3.18 |
| **terse / strict** | 74.76 | 81.89 | **`+9.18%`** | [+6.81; +9.31] | **1.83** |
| terse / reusable | 75.93 | 88.76 | `+14.52%` | [+14.15; +15.03] | 2.74 |

All three preregistered main workloads qualify at `W = 4` under both plans. The
terse strict case is the one that **fails** the threshold, and its realised group
width of 1.83 says why: with 2–3 token answers the queue empties before a group
fills.

`G(W1) ≈ 0` everywhere (`−0.23%` to `+0.00%`). Interleaving alone changes nothing;
the gain comes from grouping. That control is what makes the rest interpretable.

### Latency: the actual Pareto front

Grouping does not make requests faster. It makes them finish together.

| Workload / plan | p50 sequential → `W4` | p95 sequential → `W4` |
| :-- | :-- | :-- |
| homogeneous / strict | 1061 → 1350 ms (**+27.1%**) | 1644 → 1391 ms (**−15.4%**) |
| heterogeneous / strict | 897 → 1142 ms (+27.4%) | 1500 → 1268 ms (−15.5%) |
| staggered / strict | 773 → 977 ms (+26.3%) | 1292 → 1181 ms (−8.6%) |
| terse / strict | 146 → 221 ms (+51.4%) | 442 → 405 ms (−8.3%) |

**Median latency worsens by 26–51%; tail latency improves by 8–17%.** Sequential
service finishes its first requests quickly and its last slowly; grouping levels
that out. Which is preferable is a service-level decision, not a research finding,
and it is reported as a front rather than resolved.

Full-response latency adds prefill, which differs sharply by plan: median `1493 –
1800 ms` under `StrictOneShotPlan` against `71 – 73 ms` under
`ReusableSessionPlan`, the prefix-cache effect from E10 and E13 reappearing.
Group time was never divided by width and called caller latency.

### Correctness and state isolation

**Zero failures.** Across all workloads, plans, widths and repeats, every request
matched its sequential reference under the same plan on token IDs, token count and
stop reason, and on the SHA-256 of its valid KV region wherever hashed. Early
finishers left the active set without disturbing the others. **32 reversed-order
runs, zero failures**, so results do not depend on group composition order.

### Best fixed width

`W = 4` or `W = 8` depending on workload; the difference between them is within a
percentage point everywhere, and `W = 4` is never worse by a meaningful margin. For
a fixed choice `W = 4` is the defensible one.

### When async B1 helps, and when it does not

**Helps** when several requests are genuinely concurrent and answers are long
enough that groups fill — 15–17% throughput at a realised width above 3.

**Does not help** when answers are short: the terse strict case reached only 1.83
realised width and `+9.18%`, below threshold. A queue that empties faster than a
group fills has nothing to amortise.

**Costs** median latency in every case. A latency-sensitive single-user path should
not use it.

### Is a queue- and latency-aware controller the justified next step?

The evidence supports it more than E14b did, with one caveat that matters: the whole
gain here is available from a **fixed** width of 4. Nothing measured shows an
adaptive controller beating a fixed `W = 4`, because realised width already adapts
on its own — it fell from 4.00 to 1.83 exactly where the workload thinned out,
without any controller. What a controller would add is a latency policy, and that is
a product decision that needs a target, not another experiment. Not built.

**Status** MEASURED. **Raw** `E15_preregistration.md`, `E15_results_pilot.json`,
`E15_results_main_aborted.json`, `E15_results_main.json`, `E15_summary.json`.

## E16 — Replication of the W=4 gain under real process boundaries

**Preregistration** `research/raw/E16_preregistration.md`, frozen at `a35cb36`,
SHA-256 `0ec4a1eb…bcd8`, committed as `809a054` before measurement.

**40 replicates in 40 distinct OS processes**, zero crashes, 1114 s. The parent
spawned each child, read one JSON line, and waited for exit; no model work ran in
the parent. This closes E15's correction **M2**, where "four fresh processes" were
four blocks inside one OS process.

### Frozen verdict: `CONFOUNDED_BY_PROCESS_STATE`. Substantive reading: replicated.

Both are reported, in that order, because the frozen rule assigns the first and the
data support the second. The two criteria that fired are demonstrably misspecified,
and neither measures accumulation.

### The effect replicates almost exactly

| Condition | E16 `G` | 95% CI over 5 processes | E15 | delta | realised width | within-process CV |
| :-- | --: | :-- | --: | --: | --: | --: |
| homogeneous / strict | `+16.43%` | [+15.73; +16.90] | 16.66% | −0.23pp | 3.97 | 0.58% |
| homogeneous / reusable | `+17.16%` | [+16.77; +17.50] | 17.13% | +0.03pp | 4.00 | 0.46% |
| heterogeneous / strict | `+15.58%` | [+14.70; +15.89] | 15.52% | +0.06pp | 3.32 | 0.89% |
| heterogeneous / reusable | `+15.83%` | [+15.69; +16.15] | 16.18% | −0.35pp | 3.53 | 0.64% |
| staggered / strict | `+15.10%` | [+14.76; +15.40] | 15.40% | −0.30pp | 3.16 | 0.40% |
| staggered / reusable | `+15.13%` | [+14.50; +15.21] | 14.99% | +0.14pp | 3.17 | 0.48% |
| terse / strict *(not required)* | `+9.22%` | [+8.29; +10.41] | 9.18% | +0.04pp | 1.83 | 1.50% |
| terse / reusable *(not required)* | `+13.70%` | [+12.55; +14.28] | 14.52% | −0.82pp | 2.74 | 1.58% |

All six required conditions qualify with intervals excluding zero. **The largest
deviation from E15 anywhere is 0.82pp, and among required conditions 0.35pp.**
Within-process coefficient of variation is 0.40–1.58%. The effect is neither
smaller nor unstable across real processes.

`terse/strict` fails the threshold at `+9.22%`, exactly as it did in E15 at
`+9.18%` — the failure replicates as precisely as the successes.

### Why the frozen verdict says otherwise

**A1 — RSS growth after warmup: `+68.46%`, limit 10%, FAIL.** The criterion is
anchored to the wrong point, which the pilot already showed and the main run settles
beyond doubt. The reported shape diagnostic — RSS growth measured from the **first
repeat** instead of from warmup — is **`+0.0010%` maximum across all 40 processes**,
minimum `−0.3972%`. The entire step is a single page-in between warmup and the first
repeat of memory MLX had already allocated. Nothing grows between runs.

**A3 — effect drift first repeat against last: `3.37pp`, limit 3pp, FAIL.** The
criterion was not scoped to the required workloads, although the threshold `θ` was.
The drift by condition:

| Condition | first | last | drift |
| :-- | --: | --: | --: |
| heterogeneous / strict | +14.82% | +15.82% | 1.00pp |
| homogeneous / strict | +15.88% | +16.45% | 0.57pp |
| staggered / reusable | +15.06% | +15.58% | 0.52pp |
| … all required conditions | | | **≤ 1.00pp** |
| terse / strict *(not required)* | +8.06% | +10.51% | 2.45pp |
| **terse / reusable** *(not required)* | +11.59% | +14.96% | **3.37pp** |

**The violation lives entirely in a workload the criterion does not require.** Over
the six required conditions the maximum drift is `1.00pp`, comfortably inside the
limit. Terse is also the noisiest case throughout — shortest runs, lowest realised
width, highest CV — so a fixed per-run cost weighs most there.

**A2 and every other direct measure of accumulation pass, at zero.**

| Instrument | growth after warmup |
| :-- | --: |
| MLX active memory | `+0.0000%` |
| MLX buffer cache | `+0.0000%` |
| Compiled-body cache | constant at 1 entry |
| Python allocated blocks | `+0.117%` |
| RSS, measured from the first repeat | `+0.0010%` |

Neither criterion that fired measures state accumulating between runs. Every
instrument that does measure it reads zero.

**The criteria are not being changed.** They were frozen, they were applied as
written, and the class they assign is reported as the class they assign. A
threshold quietly repaired after it fires is worth less than a threshold honestly
applied and shown to have been badly chosen — which is the lesson E16 carries
forward, alongside the replication.

### Latency, arrival based

| Condition | p50 | p95 | TTFT p50 |
| :-- | :-- | :-- | :-- |
| homogeneous / strict | 1060.6 → 1348.7 ms (**+27.2%**) | 1640.7 → 1383.0 (**−15.7%**) | 861.8 → **87.9** |
| heterogeneous / strict | 896.5 → 1135.4 (+26.6%) | 1502.1 → 1269.7 (−15.5%) | 803.3 → 86.6 |
| staggered / strict | 778.9 → 981.4 (+26.0%) | 1288.6 → 1183.0 (−8.2%) | 686.7 → 87.7 |
| terse / reusable | 235.6 → 388.2 (+64.7%) | 688.0 → 597.1 (−13.2%) | 209.6 → 86.3 |

E15's Pareto front reproduces: median worsens 26–31% in the required conditions,
tail improves 8–17%.

**One effect E15 never reported: time to first token collapses from roughly 800 ms
to roughly 87 ms.** Under sequential service the eighth request waits for the other
seven to finish before it emits anything; under grouping every request produces a
first token in the first round it is admitted to. That is a nine- to tenfold
improvement in first-token latency and it was hiding in plain sight behind the
median.

### Memory

| Condition | start | after load | after warmup | end | MLX peak |
| :-- | --: | --: | --: | --: | --: |
| homogeneous / strict | 48 MB | 3119 | 3098 | 5059 | 6.32 GB |
| heterogeneous / reusable | 48 | 3119 | 3135 | 5277 | 6.32 GB |
| terse / strict | 48 | 3118 | 2961 | 4786 | 6.32 GB |

Every process starts at 48 MB and ends between 4.8 and 5.3 GB, with MLX peak at
5.78 or 6.32 GB depending on plan. No process approached the 12 GiB ceiling. RSS
after warmup is sometimes *below* RSS after load, which is itself a reminder that
RSS on macOS is a page-residency measure and not an allocation measure.

### Correctness

**Zero failures across 40 processes.** Token IDs, token counts, stop reasons and KV
state hashes matched each child's own sequential reference in every run.

**Cross-process determinism holds in all eight conditions**: the sequential
reference token sequences are byte-identical across all five processes of every
condition. A fresh process, a fresh allocator and a fresh model load produce the
same output. This check only became possible with real process boundaries, and it
retroactively supports every earlier result in this ledger that assumed it.

### Can `W = 4` be treated as a dependable runtime building block?

For throughput under genuine concurrency: **yes, on this evidence.** A 15–17% gain,
replicated across 40 independent processes with intervals around one percentage
point wide, exact correctness, and no measurable state accumulation.

With three conditions attached, all measured rather than assumed:

1. It costs 26–31% median latency. A latency-sensitive single-request path should
   not use it.
2. It needs answers long enough for groups to fill. At realised width 1.83 the gain
   fell below threshold in both E15 and E16.
3. It is not a controller and nothing here argues for one: realised width already
   adapts on its own, and E15 showed the whole gain available from a fixed 4.

**Status** MEASURED. **Raw** `E16_preregistration.md`, `E16_results_pilot.json`,
`E16_results_main.json`, `E16_summary.json`.
