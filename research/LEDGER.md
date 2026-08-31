# Experiment Ledger

Status vocabulary: MEASURED, REPRODUCED, NOT_REPRODUCED, PARTIALLY_REPRODUCED,
HYPOTHESIS, INFERRED, REJECTED, OPEN, COMPATIBILITY_QUALIFIED.

Raw data for every entry lives in `research/raw/<ID>.json` **on the machine that ran
it**. Those files can carry prompts and absolute paths, so `.gitignore` keeps them out
of the repository; what ships is the redacted `*_public_summary_*.json` next to them.
A raw file named here without a link is therefore local evidence, not a missing file.
Negative results are never removed.

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

### Exploratory, outside the preregistered series

| ID | Question | Result | Status |
| :-- | :-- | :-- | :-- |
| X1 | Does the `W=4` gain hold as the model grows? | falls monotonically, `19.24% -> 15.42% -> 11.81%` | MEASURED, not preregistered |
| X2 | Can IronMule preserve Qwen3.5's hybrid cache without changing Gemma? | strict/greedy compatibility qualified; no performance claim | COMPATIBILITY_QUALIFIED |

**Correction.** The first version of this entry, written earlier the same day, reported
`+15.96%` for 4B against `+16.31%` for 12B from a single run each and concluded the gain
held at three times the parameter count. That was wrong. Repeating each cell three times
reverses it: the gain falls monotonically with model size, and the original 4B figure
lies outside the range of all three later 4B runs. A single paired run was not enough,
which is the same lesson the README draws about unpaired comparison, one level up.

**Strict plan, three runs per model, unchanged protocol, realised width `4.00` throughout:**

| Model | mean gain | observed range | spread | peak memory |
| :-- | --: | :-- | --: | --: |
| Gemma 3 4B | `+19.24%` | `19.06 – 19.41%` | `0.35pp` | `2.78 GB` |
| Gemma 3 12B | `+15.42%` | `15.20 – 15.65%` | `0.45pp` | `7.80 GB` |
| Gemma 3 27B | `+11.81%` | `11.36 – 12.09%` | `0.73pp` | `16.78 GB` |

The gaps between models, `3.82pp` and `3.61pp`, are five to ten times the spread inside
any one of them, and the observed ranges do not overlap. Group filling does not explain
the trend: all three ran at the full realised width of `4.00`.

**The reusable plan is reported but not interpreted.** Its within-model spread reaches
`4.82pp` at 12B and `4.08pp` at 27B — as large as the differences that would be compared
— and realised width also varies by model (`3.27`, `3.54`, `3.64`). Three runs are not
enough to say anything there.

**Not tested:** a larger model spends more of each decode step moving weights, so the
overhead grouping removes should be a smaller share of the total. That reading is
consistent with E2 and E4 but was not measured, and no claim is made from it.

**Limits.** No preregistration, no threshold fixed in advance, and three repeats inside
one process give a spread rather than a confidence interval. One machine, one
quantisation, and all three models are Gemma 3 — this run does not separate model size
from model family. Raw data in `research/raw/X1_*`.

## X2 — Qwen3.5 hybrid-cache compatibility

**Question.** Can IronMule carry Qwen3.8's recurrent `ArraysCache` and attention
`KVCache` together while retaining Gemma's established all-KV path?

**Mechanism and initial failure.** The adapter classifies only known MLX-LM cache
types, serialises KV layers as `keys`/`values` and recurrent layers as `arrays`, and
reconstructs each native cache with fixed shapes. The pre-fix probe failed at
`_fixed_state_from_standard` with `AttributeError: 'ArraysCache' object has no
attribute 'keys'`.

**Environment and scope.** Qwen3.8-27B-4bit, exact revision
`3e6447f082e89cc7f0bc6e5441afd38dfce760ff`; MLX `0.32.0`, mlx-lm `0.31.3`, Apple
M1 Max, 32 GB. Code was limited to `ironmule/runtime.py`, `ironmule/service.py`,
focused cache-contract tests, and a local-only Qwen integration gate. Gemma's
pre/post strict token gate was exact for both requests: q0
`[96814,6571,17269,531,5571,496,3629,2608,528]`, q1
`[818,1595,147121,18710,659,11628,9796,18677,580]`.

**Qwen correctness.** The corrected one-shot reference tokens were q0
`[1596,1144,4087,1156,25,328,657,799,9144]` and q1
`[1596,1144,4087,1156,25,328,3710,1503,54102]`. All 64 layers followed the
`AAAK` pattern (`ArraysCache`, `ArraysCache`, `ArraysCache`, `KVCache`) repeated
16 times. Recurrent leaves retained their shapes across two decode steps and
hybrid KV hashes were executable and distinct at each step.

The staged service gates at 2 and 3 requests × 8 maximum tokens, followed by the
final 6 requests × 48 maximum-token workload, were token-identical with
`fallbacks=0` and `correctness_errors=0`; the
separate tiny compiled gate was also exact. Full-run peak memory was `17.71 GB`
for the baseline and `30.76 GB` for the compiled tiny gate. The compiled peak is a
warning, not a performance result.

**Rejected harness.** The first `generate_step` harness split the prompt before its
last token, so it was not a one-shot-prefill reference. It was discarded as a test
design error and yields no product finding.

**Status.** COMPATIBILITY_QUALIFIED. X2 makes no performance statement and does not
generalise beyond the stated revision, environment, strict/greedy path and tested
workloads. The B26 family/performance study remains open and requires its planned
three-repeat measurement.

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
blocks**. Deterministic, reproducible, confined to `b = 8`. (Blocks, not OS processes:
see limitation `M2` below — `e14b_arms.py` loops in one interpreter, and this entry
said "processes" before that was noticed.)

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

## R12/E15 — Fork-per-block memory follow-up (2026-08-30)

This is a separate engineering follow-up to E15's historical M2 finding; the
historical E15 values above are not overwritten. The code change is committed at
`b700377e83b2eba39c5d66976d01332f8ab57bc6`. The frozen E15 preregistration remains
`c2c8a5931cb2c67097fed9f435c5af52c7196abe` with SHA-256
`939a3c40683433e6fc2e24c4409304a4a762fbae52c7b528b6a4de1216b70a92`.

The in-interpreter baseline `research/raw/E15_before_fork.json` has SHA-256
`4312e3bff94a0982711191faf3b110037d293344ccf3e127acaa9c56128b2ea6`, commit
`5d2d2f8`, and `git_dirty=true`. Its four block peaks were
`7067609536`, `7483569616`, `9619556792`, and `9619559548 B`; wall time was
`1571.585 s`; its sole PID was `84078`. The forked result
`research/raw/E15_after_fork.json` has SHA-256
`d14875e43ee800d8f1a29af966b8adad56245a414dd204f202a48b81d1f91b5c`, commit
`b700377`, and `git_dirty=true`. It completed four blocks with PIDs
`15489/24645/33850/42483`, 128 runs per block, peaks
`7067618790/7067610600/7067609606/7067609586 B`, and wall time `1664.407 s`.

The after MemoryGate was active and did not abort: swap deltas were
`-8/-16/-16/-80 MiB`, below the `256 MiB` limit, and the peak backstop was not
reached. There were no crashes and no token, token-count, stop-reason, or KV-state
deviations against the per-process sequential references. This satisfies the R12
kill criterion: fresh process boundaries prevent the cumulative peak growth seen in
the shared interpreter.

The result is an engineering/memory-integrity finding only. The different commits,
dirty environments, load and swap baselines, and the `92.822 s` longer after wall
time prevent a clean A/B speed claim. No routing, activation, or production claim is
authorized. The existing `research/raw/E15_summary.json` was not overwritten and is
not the summary of this after-file; the archived artifact is
`d1/d14875e43ee800d8f1a29af966b8adad56245a414dd204f202a48b81d1f91b5c-E15_after_fork.json`.

## B7 — Which term dominates the falling grouping gain

**Preregistration** written before measurement, SHA-256 of the completed document
`1a0f6aeb…1266`. **It was not committed before the run**, unlike `E14`, `E14b`, `E15`
and `E16`, and the hash covers a document that now also contains the results.

Precisely which claim that costs, since "weaker evidence" is too vague to act on:

- **Not supported:** that the four candidate outcomes, and the specific figures `1.41×`
  and `2–3×`, were chosen before the data were seen. A reader cannot rule out that they
  were fitted afterwards to make the result land cleanly on one of them. Every statement
  in this entry of the form *"as predicted in advance"* rests on trust alone.
- **Unaffected:** the finding itself. `SCALING.md`'s `0.41` prediction and the
  layer-count and weight-traffic reasoning behind it are committed in this repository and
  predate this run by weeks. The central claim — that both of its terms are
  misspecified — compares measurements against a *published* prediction, not against
  mine. That comparison stands whatever the status of my document.

So the un-frozen preregistration costs the framing, not the result. The fix for the next
run is procedural and cheap: commit the preregistration first, then measure.

**Two model sizes, one machine, `0de69b6`.** `gemma-3-4b-it-4bit` and
`gemma-3-12b-it-4bit`, AC power, swap `0.06 MB` throughout, `research/e14b_arms.py`
unmodified. 4B: 4 blocks × 7 repeats. 12B: 1 block × 7 repeats after the memory guard
aborted the run — see Execution.

### Result `ANSWERED_BOTH_TERMS_MISSPECIFIED`

`SCALING.md` predicts the recoverable share falls to `0.41` of its 4B value. The ledger
measured `11.81 / 19.24 = 0.61`. This run measures `10.34 / 16.36 = 0.63` at batch 8,
from an independent set of measurements, and shows why the prediction missed: **both of
its terms are wrong, in opposite directions, and partly cancel.**

| Growth, 4B → 12B, arm A | Predicted | Measured | Stability across batches 1–8 |
| :-- | --: | --: | :-- |
| `submission_ns` | 1.41× (layer count 34 → 48) | **3.68×** | 3.68 / 3.77 / 3.72 / 3.68 |
| `completion_wait_ns` | 2–3× (parameters ÷ bandwidth) | **1.50×** | 1.49 / 1.49 / 1.50 / 1.50 |

Host work grows 2.6× faster than the kernel-count model allows. Device time grows at
half the low end of its estimate. Neither term is individually close.

*(Reviewer's correction, kept visible rather than silently fixed: an earlier draft of
this entry claimed the backlog's `62` layers for 27B was wrong and should be `64`. It
is not. Gemma 3 27B has 62 layers, which is what `B7` and the model table mean. The
only 27B in this machine's cache is `Qwen3.8-27B-4bit`, which has 64 — verified from
`config.json`, alongside Gemma 4B at 34 and 12B at 48. The draft read the one config it
could open and attributed it to the other family: exactly the size-versus-family
confusion this entry's own Validity section warns about, and the reason `B26` exists.)*

### The step becomes more host-bound as the model grows, not less

| `submission_ns` ÷ `completion_wait_ns`, arm A | batch 1 | batch 8 |
| :-- | --: | --: |
| 4B | 1.02× | 1.04× |
| 12B | 2.52× | 2.56× |

At 4B the two are balanced. At 12B the submission window is `187 ms` of a `268 ms` step.
`SCALING.md` assumes fixed host overhead becomes a *smaller* share as models grow; the
opposite is measured. Tier 2 (`B8`, `B9`, `B10`) is therefore aimed at the term that
dominates at scale, and is worth **more** at 12B than at the 4B where the evidence for
it was gathered. This does not contradict the backlog's warning that those entries
shrink the headline ratio — they would shrink it precisely by removing the largest
absolute cost.

### What remains INFERRED, and the hard dependency it creates

`submission_ns` is **not** host work and must not be read as such. At 4B batch 8, arm B
submits for `73.53 ms` then waits `10.11 ms`; arm A submits `50.85` and waits `48.79`.
Identical work and shapes. Arm B's window is larger *because device execution happens
inside it* — that overlap is the mechanism `E14b` identified and the product is built
on. The split therefore measures windows on a wall clock, not host and device costs.

Every comparison above survives this, because each is within one arm across model sizes.
The next question — what fraction of the growing submission window is Python and what
fraction is the device — is **not answerable with this instrument at all**.

That makes `B24` ("Stop measuring the GPU with a wall clock") a hard prerequisite, not a
methodological preference: **`B8`, `B9` and `B10` cannot be sized until real device
counters exist.** Recommend recording that dependency in those three entries, not only
in `B24`.

### Execution

| Model | Blocks | Repeats | Wall | Outcome |
| :-- | :-- | --: | --: | :-- |
| 4B pilot | 1 | — | 13 s | completed |
| 4B main | 4 | 7 | 189 s | completed |
| 12B main | 1 of 4 | 7 | 157 s | **aborted at the 12 GiB guard** |

The 12B abort is `M2`/`M3` reproduced, not discovered. Cumulative MLX peak across 4B
blocks was `6.37 → 7.28 → 9.36 → 11.53 GB`, against `M2`'s recorded
`7.07 → 7.07 → 9.24 → 11.25 GB`; 12B reported `17.51 GB` and broke the loop.

**Correction to an earlier reading of this abort.** A confirmation run on `7428126`,
which resets the MLX peak counter per block, reports 4B peaks of
`6.37 → 6.37 → 6.37 → 8.43 GB` — the accumulation is demonstrably gone — but 12B again
reports **exactly `17.51 GB`** and aborts at the same place. Block 1 has nothing to
accumulate, so that figure was never inflated: it is 12B's genuine per-block peak, and
it legitimately exceeds the 12 GiB guard. The cumulative-mark defect is real and affects
blocks 2 and later; it is **not** what truncated 12B. 12B simply does not fit under this
guard, fix or no fix — the same situation as 27B at `14.98 GiB` of weights. Any earlier
statement here that the guard "fired on an inflated value" applied to 4B's near-miss,
not to 12B's abort.

Two things about the abort are new. `M3` attributes its abort to a prefill cache, not to
a guard reading a cumulative high-water mark, so the guard's early firing on later blocks
is not on record. And the abort is **invisible in the result file**: it prints to stdout
only, so `B7_12b.json` looks like an ordinary result with `runs: 1`. Without the console
log this deviation would have gone unnoticed. Both are fixed or filed (`7428126`, `R10`).

`M2` states that "within-block arm comparison is unaffected, since drift hits every arm
in a block alike". Every comparison in this entry is within-block and within-arm, so the
truncation costs sample count and bootstrap independence, neither of which this analysis
uses. The 12B ratios are additionally stable to `±0.05` across four batch sizes.

**What the truncation does cost, stated plainly.** Every ratio in this entry divides a
median over four 4B blocks by a median over *one* 12B block, and the `±0.05` stability
is across batch sizes inside that single block — batch sizes share a block's state, so
that is not evidence about block-to-block variation at 12B, which is simply unmeasured.
Recomputed from the raw files by the reviewer, the 4B side does put a bound on how much
this is likely to matter: across its four blocks, arm A batch 8 `submission_ns` medians
are `49.77 / 51.00 / 50.98 / 50.41 ms`, a spread of `2.5%`, and `completion_wait_ns` are
`48.91 / 48.70 / 48.77 / 48.81 ms`, a spread of `0.4%`. Carrying the 4B spread through
moves `submission` from `3.69×` to the range `3.67–3.76×` and leaves
`completion_wait` at `1.49–1.50×`. Neither excursion comes near closing the gap to the
predicted `1.41×` and `2–3×`, so the conclusion holds — but it holds on the assumption
that 12B's block-to-block behaviour resembles 4B's, and that assumption is untested.

### The confirmation run was discarded, and why that is reported rather than buried

The `7428126` confirmation run above is **not evidence and none of its numbers appear in
this entry's tables.** Preregistered kill criterion 2 reads "swap delta is nonzero at any
model size". Swap during the original runs was `0.06 MB` throughout. During the
confirmation run macOS grew the swap file from 1 GB to 4 GB and reached `2816 MB` in use.
The criterion fired, so the run is discarded. It was written down in advance precisely so
it could not be reasoned away afterwards once the numbers looked convenient.

Read only as a robustness check, and labelled as coming from invalidated data, it says
something worth recording. Every cell slowed by a uniform `1.10×`–`1.15×` — both arms,
both model sizes, all four batch sizes — which is the signature of machine-wide memory
pressure rather than a selective effect. Because it is uniform, it cancels in the ratios
this entry actually uses:

| `submission` 4B → 12B | batch 1 | batch 2 | batch 4 | batch 8 |
| :-- | --: | --: | --: | --: |
| valid run | 3.68 | 3.77 | 3.72 | 3.68 |
| discarded run | 3.66 | 3.75 | 3.71 | 3.66 |

The finding survives a 12% machine-wide slowdown intact. That is a stronger statement
about its robustness than a clean second run would have been — but it is a remark, not a
result, and the entry's numbers remain the swap-free ones.

### Side result: `B28` reproduced on a second model family

The correctness block compares true-batched decode against batch-1 singles. At batch 8,
sequence 3, position 6: `1580` single, `1437` batched. Deterministic across all four 4B
blocks; prefill logits bit-equal, so the divergence arises in decode.

Arm C is **True Batch**, which IronMule does not route. Arm B, the shipped
`ThroughputMode`, stays token-identical throughout. This is `B28`'s Qwen-only correctness
rejection reproduced on Gemma, and is evidence **for** the decision not to ship true
batching — not a defect in the runtime.

### Validity

One machine, two sizes, one family, one MLX build, greedy decoding. 27B was not run: at
a true per-arm peak near 17 GB it is feasible on 32 GB only after `7428126`, and peak
figures from before that commit are cumulative rather than per-block and must not be
tabulated against ones from after it. Nothing here separates model size from model
family — that remains `B26`. `docs/LIMITS.md` is unchanged by this entry.

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

### X3 — B28 native Qwen true-batch candidate rejected

`qwen_native_true_batch_v1` was rejected at the correctness gate. Widths 2, 3 and
4 preserved visible tokens and stop reasons exactly with zero fallbacks, but the
final hybrid `kv_hash` differed from the sequential reference. Swap delta was
`0 B`; no token-rate or performance result is claimed. The candidate is not routed.
Raw: `B28_true_batch_correctness_failure.json`.

### X4 — B29c native Qwen batch-1 pilot below target

`qwen_native_b1_v1` passed correctness, final-state and 16-token continuation
checks at widths 2, 3 and 4 with zero fallbacks and zero swap delta. The candidate
median was `16.0722` tokens/s versus Interactive `15.6740` (`1.02541x`) and versus
Throughput `16.0687` (`1.000219x`), below the preregistered `1.10` gate. No route
was enabled. This is a throughput result, not a correctness failure.

### B35 — Exploratory portability screen for the non-mutating core profile

**Preregistration.** `B35_preregistration.md` froze
an exploratory screen of `BASELINE=Knobs()` against
`Knobs(compiled_fixed_cache=True, head_skip_prefill=True)` with the repository
prompt, `max_tokens=32`, two warmups, five repeats, balanced AB/BA, one model
load per fresh OS process, exact token gates, peak-memory `+10%`, swap `256 MiB`,
and no-crash gates. `B35a_preregistration.md`
added only the clean-environment correction after the first 1B process overlapped
broad filesystem searches; no arms, thresholds, or workload changed.

**Scope.** Local Apple M1 Max, 32 GB unified memory, AC-power Darwin host,
Python `3.12.13`, MLX `0.32.0`, mlx-lm `0.31.3`, NumPy `2.5.2`, greedy batch-1
generation, fixed 322-token chat prompt. Model revisions were Gemma 3
1B `2d44e83dc9e80843d22fb941d3d699a0b1351aa6`, 4B `93724907d4ed1745d2fe50baadf3b0b01a65abf2`,
and 12B `86cc6a8dedbc456dd0e4af01a9d09f396f77e558`; configuration/tokenizer
digests are recorded in each raw file.

**Invalid first attempt.** The first 1B AB worker completed but ran while broad
`find` searches were active. It is retained as
`B35_gemma1b_AB_20260828.json`, marked
`valid_for_metrics: false`, and contributes no performance number.

**Clean result.** Each model completed two fresh processes (AB and BA), with
five raw samples per arm. All six processes passed token identity and
determinism; max swap delta was `0 B` for every model (the 12B BA baseline
window moved `-16 MiB`, while the candidate moved `0 B`), and candidate/baseline
peak-memory ratios were 1B `0.8794038`, 4B `0.9518492`, and 12B `0.9853990`.
The aggregate below is the arithmetic median of the two process-level median
ratios, with `core / baseline` (lower is faster):

| Model | total ratio | prefill ratio | decode ratio | AB total | BA total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 1B | `0.8495684` | `0.7562452` | `0.8971580` | `0.8542057` | `0.8449311` |
| Gemma 4B | `0.8702504` | `0.8356965` | `0.9444767` | `0.8721220` | `0.8683789` |
| Gemma 12B | `0.9840062` | `0.9638611` | `1.0149662` | `0.9040384` | `1.0639741` |

The 1B output-token digest is
`11ac58e1ae29408d9762daee4df4749281ce24459f218e78b137dd31ae5ce0f7`; the
4B and 12B output-token digest is
`d9818d21a6a6bef76c4091ef56ba158dfbc553a0f6c90e3d06543034be2a100f`. The
prompt-token digest is
`80ecf700cf0dfdc82616c73f1b6a5fccc137b68e9bb9586ca376c3f2adb260ad`.

**Decision and limit.** **Exploratory candidate qualifies under gate; 12B result
order-sensitive/inconclusive for robust performance.** The 1B/4B results are
stable in this screen. Although the 12B aggregate is below the preregistered
`0.995` threshold, AB is clearly faster and BA clearly slower; the same-process
arm order may interact with thermal, allocator/cache, or compiled state. No
shipping, routing, profile activation, or cross-model/general performance claim
is made. Follow-up is tracked as B36: remeasure with arm-isolated fresh processes.

**Raw evidence.** `B35_gemma1b_AB_clean_20260828.json`,
`B35_gemma1b_BA_clean_20260828.json`,
`B35_gemma4b_AB_clean_20260828.json`,
`B35_gemma4b_BA_clean_20260828.json`,
`B35_gemma12b_AB_clean_20260828.json`,
and `B35_gemma12b_BA_clean_20260828.json`.

**Review limitations (2026-08-28).** The independent review is recorded in
`B35_review.md`. The worker's per-arm swap gate starts only
after model load and therefore does not cover load-time swap; external post-run
swap checks found no new issue but do not repair this raw-gate gap. The worker
also sets `hard_gates.no_crash` to constant `true`: external process-list and
crashreport checks found no new Python crashreports after the clean runs, but
those checks are not encoded in the raw JSON gate. Because both arms shared one
Engine/model per process, allocator, compiled-cache and thermal state remain
coupled to AB/BA order; this is visible in 12B total `0.9040384` (AB) versus
`1.0639741` (BA). Finally, each raw file stores only the first repeat's token
list, with no stop reason or per-repeat token lists, so the determinism boolean
is less auditable than complete repeat-level token/stop records. The permitted
claim remains: **exploratory candidate qualifies under gate for 1B/4B; 12B
result order-sensitive/inconclusive for robust performance**; no shipping or
general claim is made.

## B37 — Phase/roofline diagnostic helper

**Result (2026-08-28).** Added the pure `phase_roofline_diagnostic` calculation
and CPU-only schema tests. It preserves prefill and decode values separately,
uses explicit active-weight, KV and extra traffic inputs, and computes only a
per-run diagnostic efficiency from supplied effective bandwidth. Missing inputs
are `inconclusive`; invalid, non-finite, negative or zero-denominator inputs are
`invalid`; zero-step decode is `not_applicable`. Efficiency above one is retained
and marked as an input-consistency warning.

**Decision and limit.** This is instrumentation, not a runtime optimization or
performance result. It changes no correctness, swap, crash, profile or B35/B36
gate and emits no compute-/bandwidth-bound claim. No MLX, Metal, ANE, model or
benchmark run was performed for B37. A future producer must provide explicit
phase units and byte semantics before the diagnostic can be populated with real
measurements.

## 2026-08-28 — B39b Benchmark-Preflight blockiert

Der angeforderte B39b-Benchmark-Preflight maß `vm.swapusage` mit total
`8192.00M`, used `7143.12M` und free `1048.88M`; `memory_pressure` meldete
`75%` freien Speicher, und es lief kein Gemma-Prozess. Der absolute B39b-
Pre-Spawn-Swap-Gate von `<=256 MiB` schlug deshalb fail-closed fehl. Es wurde
kein Modell, Child oder Benchmark gestartet und keine Optimierung geändert.

Die serielle CPU-Harness-Nachprüfung bestand mit `46` Tests in `7.42 s`,
Exit `0`; Crashreport-Zähler User/System blieben vor und nach dem Lauf bei
`64/61` (Delta `0`), und `git diff --check` war grün. Die eingefrorenen Hashes
blieben unverändert. Ein weiterer B39b-Versuch ist erst nach Reboot und einem
sauberen, verifizierten Systemzustand zulässig; der aktuelle Swap ist ein
Safety-Blocker und kein Runtime-Speedbefund.

## B39b Pilot — diagnostisch, INCONCLUSIVE

Nach einem sauberen Preflight (System-Swap `0 B`, `93%` freier Speicher, kein
Gemma-Prozess) liefen die vier frischen seriellen Children des Ein-Block-
Piloten in Reihenfolge A/B/D/C. Alle vier Returncodes waren `0`; Korrektheits-,
Environment-, Workload-, Crash- und Canonical-Gates bestanden. Je Arm liefen
zwei Warmups und ein Mess-Repeat. Alle sechs Requests je Arm erzeugten `48`
physische, logische und sichtbare Tokens mit Stop-Grund `length`; der
Canonical-Output-Digest war über alle Arme identisch. Swap war `0 B`, relevante
Crashreports und Residualprozesse waren nicht vorhanden.

Der Pilot bleibt dennoch `INCONCLUSIVE`: Das Block-Peak-Gate scheiterte allein
an RSS C/A `3.6523564` (D/B `1.0001511`). MLX-Peak-Ratios waren C/A
`1.0064033` und D/B `1.0257863`; absolute MLX-Peaks A/B/D/C:
`7,796,516,616`/`7,801,367,483`/`8,002,535,534`/`7,846,439,900 B`.
RSS-Peaks A/B/D/C:
`2,166,931,456`/`7,916,470,272`/`7,917,666,304`/`7,914,405,888 B`.

Ein-Repeat-Diagnostik (kein Speedclaim): Outer-Wall ms / physische=sichtbare
Tokens/s waren A `10308.915125`/`27.936984300`, B
`9072.028833`/`31.745930850`, C `9805.518458`/`29.371215937`, D
`8524.246458`/`33.785977613`. Wall-Ratios B/A, C/A, D/A, D/B, D/C:
`0.880017802`, `0.951168803`, `0.826881040`, `0.939618537`, `0.869331540`;
Rate-Ratios: `1.136340648`, `1.051338098`, `1.209363804`, `1.064261677`,
`1.150309122`. Interaktion D*A/(B*C): `0.987856765`.

Die RSS-Form A `2.17 -> 1.26 GB` während der Checkpoints gegenüber B/D/C nahe
`7.9 GB`, bei identischem MLX-Active-Memory nahe `7.188 GB`, macht eine
Prozessreihenfolge-/Page-Residency-Konfundierung plausibel. Eine Attribution
auf einen Arm ist verboten. Raw:
`B39b_pilot_gemma12b_combined_20260828.json`.
Finalstatus `INCONCLUSIVE`, `activation_allowed=false`; kein Main-Lauf, kein
Retry, kein Routing/keine Aktivierung. B39c mit zwei neuen Crossover-Blöcken
bleibt nach sauberem Zustand ausstehend; diese Pilotdaten werden nicht
wiederverwendet oder gepoolt.

## B39/B39a/B39b — Safety-only pilot chronology

**B39 direct-script import failure.** The first pilot invocation used the direct
script path and failed before parent initialization with return code `1`:
`ModuleNotFoundError: No module named 'research'` at
`research/b39_combined_levers.py:22`. No model or child ran, no JSON or partial
was created, crash reports remained `30 -> 30`, and no residual process
remained. Raw: `B39_pilot_import_failure_20260828.json`.

**B39a module pilot.** The corrected module invocation attempted only arm A.
The child returned `3` at `after_model_load` with
`RuntimeError: B36 checkpoint gate failed: after_model_load`; no warmups or
timed repeats ran, so no timing or performance evidence exists. Parent system
Swap moved from `1,704,921,661 B` to `8,568,438,784 B`, delta
`6,863,517,123 B` (approximately `6.39 GiB`), strongly suggesting a
resource/swap failure. The exact child subtype (swap, memory, or instrumentation)
is unobservable because child events were discarded. Crash delta was `0` and
there was no residual model process. The parent then raised
`StatisticsError: no median for empty data`, wrote no final JSON, and retained
the partial sidecar. Raw: `B39a_pilot_failure_20260828.json`.
No retry and no B39 main run occurred; this is not a measurement.

**B39b.** B39b is a safety/evidence-only correction with SHA-256
`403eb1b098d49bff891a52ac16b974857b4fad3e0ed2984f554436acf0e9e7cb` and no
hardware authorization. It preserves parent/child checkpoint events on failure,
publishes structured `INCONCLUSIVE` for empty summaries while retaining partial
evidence, and adds an absolute pre-spawn Swap ceiling of `268,435,456 B`
(`256 MiB`) alongside the unchanged process-start-to-end delta gate. B39 arms,
workload, statistics, thresholds, and no-activation rules are unchanged.
Full safety review: [`B39_review.md`](raw/B39_review.md).

## B39c — Memory-order RSS diagnostic (design only)

The B39b pilot's relative RSS failure is not an arm-memory result. In its sole
`A-B-D-C` block, A was position 0 and had RSS `2166931456 B` after load,
falling to `1262895104 B` after warmup; B, D and C were approximately `7.9e9 B`
at the same checkpoints. MLX active memory was `7188274696 B` after load for
all four arms. The block therefore recorded RSS `C/A = 3.652356361381834` but
MLX `C/A = 1.0064032806519705`; the relative peak gate failed and the result
is `INCONCLUSIVE`.

B39c is a separately sealed, diagnostic-only protocol. It executes two new
fresh-process blocks, `A-B-D-C` and the reverse `C-D-B-A`, with one load per
arm, two warmups and one measured repeat. It retains absolute 12-GiB
RSS/MLX, 256-MiB Swap, correctness, identity, crash and post-state gates, but
does not abort after a relative RSS failure so the reversed order can be
observed. It never summarises or qualifies performance, never activates a
route, and never reuses B39b timings. The prospective classifications are
`RSS_ORDER_PAGE_RESIDENCY_CONFOUNDED`, `CORE_RSS_SIGNAL_REPRODUCED`, or
`INCONCLUSIVE`; RSS remains page-residency evidence, not an allocator claim.

## B39d — Performance main design

B39d is a separately sealed performance-main continuation after B39c. It uses
the eight existing balanced orders and 32 fresh one-arm serial processes,
with one load, two warmups and five measured repeats. No conditioner, purge,
cache mutation or pooling of B39b/B39c evidence is permitted. Absolute
RSS/MLX/Swap/crash/correctness/identity gates remain hard; only per-block MLX
relative `C/A` and `D/B` ratios retain the 1.10 hard gate.

RSS is evaluated prospectively after all blocks using the maximum non-start RSS
per child, two observations per arm/position, position medians, geometric arm
means, global `C/A`/`D/B`, position residuals and matched epoch ratios. Every
RSS comparator must remain in `[1/1.10, 1.10]`; missing or failing RSS evidence
is `INCONCLUSIVE`, never `REJECTED`. The B39 co-primary D/A/D/B wall/rate
medians and Bonferroni 97.5% CIs remain unchanged. Complete resource-clean
performance misses are `REJECTED`; final-H2, RSS, resource, identity, drift or
incomplete evidence is `INCONCLUSIVE`. No activation or automatic routing is
allowed.

## B40 — Core ThroughputMode width sweep (implementation sealed)

B40 holds the B39d core profile fixed and tests only `max_width` 2, 3 and 4
through six mirrored balanced blocks and 18 fresh serial children. The exact
Gemma/X1 workload, one-load/two-warmup/five-repeat contract, child identity and
absolute Swap/RSS/MLX/crash/correctness gates are new sealed protocol fields;
no B39d timing or data is pooled.

The result is `QUALIFIED` only if both candidate comparisons W2/W4 and W3/W4
pass their preregistered wall/rate criteria and all safety gates pass; the
lower wall median is selected. `RETAIN_WIDTH4` requires two robust practical
candidate misses. RSS is evaluated only after all six blocks using two
arm/position observations, geometric global ratios, position residuals and
mirrored epoch pairs; RSS or resource failure is `INCONCLUSIVE`, never a
performance rejection. The historical X1 `+15.42%` rate is retained only as
descriptive ratio `1.1542` (equivalent wall ratio `0.866400970369`), not a
threshold or gate.

**Separate xdist incident.** An accidental non-`-n0` test invocation followed
`pytest.ini`'s `xdist -n auto` path and produced `23` Python `SIGABRT` reports
between `11:38:38` and `11:38:49` (parent PID `80772`) through MLX/`libmlx`.
Representative reports were `Python-2026-08-28-113838.ips` and
`Python-2026-08-28-113849.ips`. The later serial run recorded `46` passing CPU
tests, crash count `30 -> 30`, and green `git diff --check`. This incident is
not B39 evidence. No UI, profile activation, routing, or general performance
claim follows.

## B36 — Arm-isolated Gemma 3 12B core-profile portability

**Protocol.** B36 used the exact local Gemma 3 12B snapshot at revision
86cc6a8dedbc456dd0e4af01a9d09f396f77e558, the fixed B35 322-token prompt,
max_tokens 32, greedy generation, baseline Knobs() and candidate
compiled_fixed_cache=True plus head_skip_prefill=True. Sixteen serial pairs
(eight AB/eight BA) ran one arm per fresh process; each child loaded once,
performed two warmups and five repeats. The parent used no retries and stored
the atomic partial sidecar. B36a is the separately sealed clarification for
the full manifest hash/prefault scope.

**Environment and gates.** The host was Apple M1 Max, 32 GB unified memory,
32 GPU cores, Python 3.12.13, MLX 0.32.0 and mlx-lm 0.31.3 on AC power.
Foundation reported low-power 0 and thermalState rawValue 0; free-memory
preflight values were 74%, 75% and 66%. Wired-limit and cache-limit mutations
were not applied. Every child recorded process-start through process-end
swap/RSS/MLX checkpoints, full manifest hashes before and after load, all
warmup/repeat token and stop records, and immediate/delayed external crash
snapshots. All 32 children passed correctness, identity, resource, timeout,
post-evidence and no-crash gates. Maximum swap delta was 0 B. Maximum MLX
peak was 7,946,637,412 B baseline and 7,830,608,598 B candidate; maximum RSS
was 3,692,576,768 B baseline and 4,526,096,384 B candidate. Peak ratio was
0.985399004889189.

**Results.** Independent ratio-of-five-repeat-medians audit, candidate /
baseline:

| metric | median | 95% CI |
| --- | ---: | ---: |
| total | 0.927147428180255 | [0.9197363534291831; 0.9303748490885659] |
| prefill | 0.9183106745417602 | [0.9081866453423364; 0.9218801379522791] |
| decode | 0.9540158794083631 | [0.9419388082376179; 0.9577180135679649] |

The implied reductions are 7.29% total, 8.17% prefill and 4.60% decode.
One pair has a decode ratio above 1 and remains a diagnostic observation.
AB was 0.9250279042521969 [0.8883771936776205; 0.9283929297931295];
BA was 0.9294847335372622 [0.925744092622901; 0.9404202103921636].
The absolute order interaction was 0.0044568292850653.

**Identity and decision.** B36 and B36a SHA-256 values are
7bf3997b19dc55d3b75be977c0da8d42d6ab554232ce2bf40617429c478897a4 and
ee5b3e9b250d75eb69ed6e38f9661f656da743098bef318966dc055099c9e492.
The model manifest digest is
3de99933cacc693c88d807c4f5e4dade6d1fe719cacc570841e222940f0a9eb2.
The code digest is
5566ee87f1656d9dcaceb05edf6a155ee2a35dd784c81a46fbb6dab30e499ddc, with
the current 61-file fingerprint and commit
f3478e07d58e3bf054b3ae0503925dbb15f7edf1 matching exactly. The earlier
apparent code mismatch was an audit-script error caused by stripping the
commit newline.

**Decision and scope.** B36 is QUALIFIED under its preregistered rules, but
activation_allowed remains false. No profile activation, routing, or general
speed claim follows. The candidate's higher RSS despite lower MLX peak is
recorded without a hidden memory interpretation. Full raw evidence is in
research/raw/B36_gemma12b_results_20260828.json; the independent audit is
research/raw/B36_review.md.

## B37a — Phase/roofline diagnostic review hardening

**Result (2026-08-28).** Hardened B37 against huge integer conversion, subnormal
duration underflow, traffic-sum overflow, ideal-rate/efficiency overflow, and
absurd decode-step counts. Invalid derived values are rejected before JSON output.
Bandwidth provenance is now structured and required: `measured_effective` may
produce per-run ideal rate and efficiency; `nominal_peak` remains explicitly
inconclusive for those derived claims.

**Decision and limit.** CPU-only validation passed with 37 tests. Zero-step
decode and its roofline are `not_applicable`, while the overall diagnostic
remains `inconclusive`; the helper does not infer an EOS reason. No model, MLX,
Metal, ANE, profile, gate or performance run was performed.

## B39c — Memory-Order Diagnostic Ergebnis

Nach sauberem Preflight (System-Swap `0 B`, kein Residual-Modellprozess) liefen
die zwei neuen seriellen Blöcke `ABDC` und `CDBA` mit allen acht Children und
Returncode `0`. Correctness-, Identity-, Workload-, Crash-, Post-State-,
absolute Memory- und Swap-Gates bestanden; Swap war `0 B`, H2 final war grün,
und es gab keine relevanten Crashreports oder Residualprozesse.

MLX-C/A-Peak-Ratios: Block 0 `1.0064022925`, Block 1 `1.0064018108`; MLX-D/B:
`1.0257847094`/`1.0257859921`. RSS-C/A: `0.9999502092`/`1.0007563638`;
RSS-D/B: `0.9997158295`/`0.9998923418`. Arm-Positions-Peak-Ratios:
`A@0/C@3 = 1.0000497933`, `C@0/A@3 = 1.0007563638`. Absolute MLX-Peaks
lagen ungefähr bei `7.80–8.00 GB`, alle RSS-Peaks bei ungefähr `7.897–7.914 GB`.

Classification und Top-Status sind `INCONCLUSIVE`: Weder der preregistrierte
RSS-Orderflip noch die reproduzierte Core-RSS-Bedingung trat ein. Der
historische B39b-Wert RSS C/A `3.6524` reproduzierte sich ausdrücklich nicht;
Block 0 lag bei `0.9999502`, alle RSS-Werte bei ungefähr `7.897–7.914 GB`.
Keine Arm-Attribution. B39c setzt `valid_for_performance=false` und
`activation_allowed=false`, summarisiert keine Timings und löst keinen B39-
Main-Lauf, Retry, Routing oder Aktivierung aus. B39d mit positionsbalanciertem
Performance-Hauptlauf und zwei neuen Crossover-Blöcken bleibt nach sauberem
Preflight ausstehend; B39c wird nicht wiederverwendet oder gepoolt.

## B39d — Performance Main Ergebnis (2026-08-28)

Der freigegebene B39d-Hauptlauf wurde mit exakt acht frozen Orders
`ABDC/BCAD/CDBA/DACB/DACB/CDBA/BCAD/ABDC` und 32 frischen, strikt seriellen
OS-Children abgeschlossen. Jeder Child lud Gemma 3 12B einmal, führte zwei
Warmups und fünf Mess-Repeats auf dem X1-strict-Workload mit sechs Requests und
`max_tokens=48` aus. Ergebnis und Rohsamples stehen in
`B39d_gemma12b_combined_20260828.json`.

**Gates und Identität.** Top- und Summary-Status sind `QUALIFIED`,
`valid_for_performance=true`, `activation_allowed=false`; acht Blöcke und 32
Children sind vollständig, alle Returncodes `0`, Correctness/Identity/Workload/
Environment/Final-H2-Gates grün, keine Fallbacks oder Crashes, Swap-Deltas
überall `0 B`, und kein relevanter neuer Crashreport oder Residualprozess. Alle
192 gemessenen Requests lieferten exakt 48 physische/logische/sichtbare Tokens
mit Stop-Grund `length`; es gab genau einen Canonical-Token-Digest. Maximale
Peaks: MLX `8,002,539,246 B`, RSS `7,916,519,424 B`. RSS-Status ist `PASS`,
global `C/A=1.000449911553665`, `D/B=1.0002091397755728`; final H2 ist `ok=true`.
Python/MLX/mlx-lm waren `3.12.13/0.32.0/0.31.3` auf Apple M1 Max, 32 GiB,
macOS `26.5.2`. Model-Binding-Digest:
`e08dd84591588722a11c43d9ff7ee4b3f50d01f15371c8a4429c4f9857d37fb6`;
Code-Digest `3adaa1bf467b0efd9fa7c06b3da628de5bbadcd3d8d1e3250c462c3c9ff49ce4`;
B39d-Präregistrierungs-SHA `f6fcfccc14afb0535cd0d360d0b956cb6e2bb86873e6e5cfdc827784a7d0bd49`.

**Absolute Endpunkte.** Wall ist `outer_wall_ns`, Rate ist physisch und sichtbar
identisch, jeweils Median und 97.5%-Bootstrap-CI:

| arm | wall median [CI] ns | rate median [CI] tok/s |
| --- | ---: | ---: |
| A | `11,238,261,187.5 [11,160,058,417; 11,407,090,125]` | `25.6268096092 [25.2474554723; 25.8063165298]` |
| B | `9,804,256,146 [9,746,705,041; 9,953,182,750]` | `29.3751295028 [28.9354679035; 29.5484472741]` |
| C | `10,647,817,688 [10,494,913,166; 10,722,052,334]` | `27.0483488952 [26.8605292185; 27.4418659254]` |
| D | `9,206,717,688 [9,178,958,958; 9,380,620,959]` | `31.2815138465 [30.7015922782; 31.3761071727]` |

**Ratios.** Lower wall is faster; higher rate is faster. Values are median and
97.5%-CI, with the same rate ratio for physical and visible tokens:

| ratio | wall median [97.5% CI] | rate median [97.5% CI] |
| --- | ---: | ---: |
| B/A | `0.8758819112 [0.8513996079; 0.8899192300]` | `1.1417105861 [1.1236974843; 1.1745365992]` |
| C/A | `0.9430849603 [0.9376590283; 0.9482680892]` | `1.0603530881 [1.0545540985; 1.0664857585]` |
| D/A | `0.8194867050 [0.8067160263; 0.8394204565]` | `1.2202787058 [1.1912981061; 1.2395935713]` |
| D/B | `0.9383079941 [0.9222134455; 0.9588925258]` | `1.0657544693 [1.0428697410; 1.0843476690]` |
| D/C | `0.8694078240 [0.8560822753; 0.8852142827]` | `1.1502701579 [1.1296699788; 1.1681120248]` |

Der Headline-Unterschied ist wichtig: D reduziert Wall-Zeit gegenüber A/B um
`18.05%`/`6.17%`, während die entsprechenden Raten um `22.03%`/`6.58%`
steigen. Die Interaktion `D*A/(B*C)` hat Median `1.0027137194`, 97.5%-CI
`[0.9619774991; 1.0185403335]`. Epoch-/Order-Drift ist nicht material; die
kleinen Stichproben bleiben als Unsicherheit sichtbar (`order:D/B` und
`epoch:contrasts:D/B` uncertain), ändern aber die B39d-Klassifikation nicht.

**X1-Abgrenzung und Entscheidung.** Die historische X1-Angabe `+15.42%` ist
eine Rate-Ratio `1.1542` beziehungsweise äquivalente Wall-Ratio `0.86640097`.
Die Raw-Flags `x1 .8458` wurden als `1-.1542` geführt und sind semantisch
ungültige deskriptive Flags; sie waren nicht gate-relevant. B39d übertrifft X1
korrekt sowohl auf der Rate-Skala als auch auf der äquivalenten Wall-Skala.
Das ist ausschließlich die präregistrierte B39d-12B-Evidenz: keine automatische
Aktivierung, kein Routing und keine Generalisierung. B39 ist abgeschlossen;
`B40` (Gemma-12B-`max_width`-Sweep 2/3/4) bleibt als nächster, noch nicht
gestarteter Test offen.

## B40 — Core ThroughputMode Width-Sweep Ergebnis (2026-08-28)

Der B40-Lauf wurde mit sechs mirrored Orders
`W2/W3/W4`, `W3/W4/W2`, `W4/W2/W3`, `W4/W2/W3`, `W3/W4/W2` und
`W2/W3/W4` abgeschlossen. Alle 18 Children waren frische serielle
Ein-Prozess-Läufe mit einem Model-Load, zwei Warmups und fünf Mess-Repeats auf
dem unveränderten Gemma-12B-X1-Workload. Rohdaten:
`B40_gemma12b_width_sweep_20260828.json`;
das Partial blieb wegen des inconclusive Ergebnisses erhalten.

**Safety und Korrektheit.** Alle 18 Children lieferten Returncode `0`,
vollständige Evidence, korrekte Canonical-/Workload-/Environment-Identität und
`no_crash=true`. Jede Messanfrage erzeugte 48 Tokens mit Stop-Grund `length`;
kein Fallback oder Tokenfehler trat auf. Swap-Delta war überall `0 B`, der
maximale MLX-Peak `8,002,539,246 B`, der maximale RSS-Peak `7,921,287,168 B`;
finales H2 war `ok=true`. RSS bestand nach der positionsbalancierten Auswertung
(`PASS`): globale Ratios W2/W4 `0.9994507845985368` und W3/W4
`0.9995676763827266`. Es gab keine relevanten neuen Crashreports oder
Residualprozesse. Model-Digest war
`e08dd84591588722a11c43d9ff7ee4b3f50d01f15371c8a4429c4f9857d37fb6`, der
B40-Präreg-SHA `23d0c59d9903875a68131d1f7ac6dc902f671a48b30fa25238ff7dfda34ca0a6`;
der aktuelle Code-Digest ist
`473980a41d7f5d46f0bc1e76452edcc89238a31edc2bb1943313c243b3a27120`.

Die Realized-Width-Gates waren exakt: W2 mean/max `2/2`, W3 `3/3`, W4
`3.971830985915493/4`. Kandidat/W4-Ratios (n=6, 10.000 Bootstrap-Resamples,
97.5%-CI; niedrigere Wall bzw. höhere Rate wäre besser) waren:

| Vergleich | Wall median [97.5% CI] | physisch/sichtbar Rate median [97.5% CI] |
| --- | ---: | ---: |
| W2/W4 | `1.1033961300051331 [1.0849631490673945; 1.1335189058508615]` | `0.9062977565937032 [0.8823856803711523; 0.9218292054497783]` |
| W3/W4 | `1.040445749841422 [1.022514206934345; 1.0723726405010425]` | `0.9611730621034691 [0.9325739636675945; 0.9779881471327478]` |

Beide Width-Kandidaten waren in allen sechs Blockrichtungen langsamer als W4:
die Wall-Ratios lagen für W2 stets über 1 und für W3 stets über 1; die
korrespondierenden Rate-Ratios lagen stets unter 1. Das bleibt wegen des
materialen Drift-Gates eine deskriptive Richtung, keine ausgewählte
Performancebehauptung.

**Drift und Entscheidung.** Die Position-Residuals waren klein (W2
`[1.0038336708, 0.9883530874, 1.0131131257]`, W3
`[1.0097000444, 0.9991497382, 0.9966448181]`, W4
`[1.0078396313, 0.9986091166, 1.0]`). Material waren jedoch die
pre-registered Epoch-Ratios: W3 `0->5 = 1.0313798935311982`, W4
`1->4 = 0.9721113375197978` und W4 `2->3 = 1.0226246692862697`.
Damit sind beide Kandidaten robuste praktische Misses
(`robust_miss W2/W3=true`), aber `status=INCONCLUSIVE`,
`classification=INCONCLUSIVE`, `selected_width=null` und
`valid_for_performance=false`. `activation_allowed=false` bleibt bindend.
W4 bleibt daher unverändert die operative Baseline; aus B40 wird keine Breite
ausgewählt und kein Timing herausgepickt. Kein Retry und kein Pooling mit
B39d/B40-Daten. Der nächste architektonische Pfad ist der bereits existierende
Backlog-Eintrag B3 und benötigt eine eigene Freigabe; es wurde kein neuer
Wunsch-Eintrag erfunden.

**Public evidence boundary.** Complete local B39d/B40 raw JSON and retained
partial sidecars are intentionally excluded from the public repository because
they contain local process/system evidence. The path-free redacted B39d/B40
public summaries are publication artifacts only and do not replace the local
immutable raw evidence.

## 2026-08-28 — Pre-push Sandbox-Import-Incident

Ein erster versehentlicher Sandbox-Collection-Versuch endete mit Exit `134`
und erzeugte beim MLX-Import den Crashreport
`Python-2026-08-28-174347.ips` (`SIGABRT`). Der Raw-Report-Zähler änderte sich
netto von `60` auf `59`, weil gleichzeitig eine Systembereinigung lief; ein
Zählervergleich allein hätte den neuen Report daher verdeckt. Der korrekte
serielle Non-Integration-Lauf außerhalb der Sandbox bestand mit `284 passed`,
`14 deselected` in `20.58 s` und erzeugte keinen weiteren Report. Der
Xcode-First-Launch-Check endete mit Returncode `0`; `git diff --check` war grün.

Die Regel ist damit verstärkt: Jeder pytest-Lauf, der MLX importiert, läuft
außerhalb der Sandbox und strikt mit `-n0`; parallele xdist-/Sandbox-MLX-Imports
sind kein zulässiger Verifikationspfad. Keine Modell- oder Produktentscheidung
folgt aus dem Sandbox-Vorfall.

## B3-U2 — Fixed-shape two-step correctness pilot (2026-08-28)

B3-U2 tests whether two dependent greedy decode steps can live in one fixed-shape
compiled graph without changing output or cache state. It remains research-only and
default-off.

The pilot completed four balanced AB/BA pairs: eight fresh serial processes, two
warmups and five measured repeats per process, six requests per repeat and 48 tokens
per request. All eight children returned code `0`. Across 336 request-runs (96 warmup,
240 measured), every request stopped at `length`; canonical token output and all 42
comparable final-state hashes per pair matched. Candidate cache evidence contained
only the two registered keys, exactly two prime misses, zero measured misses and zero
evictions. Swap delta, fallback, relevant crash and residual-process counts were all
zero. Maximum MLX peak was `8,007,886,876 B`; maximum RSS was `8,314,028,032 B`.

The raw status was `PILOT_SAFE`, but `valid_for_performance=false` and
`activation_allowed=false`. A later review found that the parent had not persisted a
separate pre/post system-state record for every child. The pilot is therefore
correctness/safety evidence only and is `INCONCLUSIVE_FOR_CONFIRMATION`: no retry,
pooling, speed claim, confirmation or activation follows. Public path-free evidence:
[`B3-U2_public_summary_20260828.json`](raw/B3-U2_public_summary_20260828.json).

## B27a/B27a1/B27a2 — Evidence inventory and current-main engineering baseline (2026-08-28)

**Read-only corpus audit.** B27 began by inventorying the current branch and the
preserved local unpublished evidence worktree without modifying either source. The
snapshot contains 134 artifact occurrences and 92 unique content hashes: 40
preregistrations, 16 preregistration checksums, 48 raw results, 14 legacy summaries,
5 public summaries, 6 reviews and 5 retained partials. Fifty-one occurrences are
local-only/ignored and remain local. All 72 JSON artifacts parsed, but structural
coverage is heterogeneous: environment 43/72, workload 52/72, baseline and candidate
25/72 each, measurements 43/72, correctness 37/72, resources 64/72 and provenance
53/72. Presence is not semantic validation. The inventory dataset SHA-256 is
`ee414c9ee51c6e583ada094444ce66d5e22dca6c15c197dda1d7cd004e30bf32`;
the tracked summary is [`docs/B27_EVIDENCE_INVENTORY.md`](../docs/B27_EVIDENCE_INVENTORY.md).
This corpus is not safe to merge into a learned dataset without per-record quality,
replayability, missingness, censoring and leakage validation.

**Two pre-measurement failures are retained.** B27a stopped in `model_binding`
because Hugging Face's offline snapshot resolver required optional `.gitattributes`
and `README.md` files that are not needed by the already-used local model snapshot.
B27a1 replaced only that resolver with exact read-only cache-index selection, then
stopped at the same stage because direct `research/...py` invocation did not place the
repository root on `sys.path`. No model or benchmark arm ran in either attempt, system
swap stayed `0 B`, and no timing was observed. B27a2 changed only the invocation to
`python -m research.b27_main_baseline`. Failure-record SHA-256 values are
`e5e7ab91218a4e7a7dcd2544efc3b44fbfdbed6fefce70cadd4f5c1c366e306a` and
`dd09d2e2cc4a2ad9ac95272c4d464ff029730e82dcc3475d374683e0ad1e2260`.
No result was retried or pooled.

**Baseline protocol.** Base commit
`d422fdb00fced3238dfaa6b5e9e993294adb72cd`; runtime-tree SHA-256
`ec242cc4872014d7994c6e11cf0b32bbf145ecca4eac32088c697059e2e48385`;
Apple M1 Max, 32 GB, AC, macOS `26.5.2`, Python/MLX/mlx-lm
`3.12.13/0.32.0/0.31.3`. Each model ran in a fresh serial process from its exact
already-cached revision, strict plan and `BASELINE=Knobs()`, with no stored profile:
six requests, 48 output tokens, two warmups and six alternating measured repeats per
Interactive/Throughput arm. One model was shared between arms, so this is an
engineering baseline rather than fresh-process-per-arm qualification. The full
non-integration suite passed `119 passed, 11 deselected` in `7.04 s`; B27 harness and
inventory tests passed `6`; Xcode first-launch status and IronMule doctor were green.

| Model | Interactive outer p50 | Throughput outer p50 | wall ratio [95% CI] | rate ratio [95% CI] | MLX peak | Swap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 3 4B, rev `93724907…` | `4141.79 ms` | `3492.57 ms` | `0.84374 [0.83977; 0.84889]` | `1.18520 [1.17802; 1.19080]` | `2,784,918,618 B` | `0 B` |
| Gemma 3 12B, rev `86cc6a8d…` | `11306.16 ms` | `10098.55 ms` | `0.87871 [0.87331; 0.91457]` | `1.13804 [1.09349; 1.14508]` | `7,801,383,003 B` | `0 B` |

Both cells are `BASELINE_CAPTURED`: exact token/stop/count identity, zero fallbacks,
zero correctness errors, zero swap growth, and no residual model process. Raw-record
SHA-256 values are
`e1e9b7ce3248b83fced553334b452404bf47931d02e2352f2aed8d96f55607a0` (4B) and
`7276ee6505a58ca176561f8e66f2087616d9682aa44273d1f7ddad51a6311d98`
(12B). The path-free publication artifact is
[`B27a2_public_summary_20260828.json`](raw/B27a2_public_summary_20260828.json).

**Decision and limits.** This freezes the current-main behavior and protects the
existing grouped batch-1 path as a regression reference. It does not pool with B39d or
B40, qualify a new speed claim, activate a profile, compare against stock `mlx_lm`, or
generalize beyond the exact cells. The engineering observation that the same grouping
ratio is smaller at 12B than 4B is directionally consistent with earlier scaling
evidence but is not promoted. No model above 12B was run and nothing was downloaded or
installed. Phase C must resolve the exact model/revision/quantisation contract,
heterogeneous evidence statuses, stock-MLX fairness and fresh-process regression
method before any new routing abstraction is integrated.

**Final branch verification.** After the audit, documentation and static local-history
UI and verifier were added, the serial non-integration suite passed
`124 passed, 11 deselected` in `5.27 s`; the existing real Gemma-4B integration suite
passed `10/10` in `20.83 s`.
The model-test preflight again recorded system swap `0 B` and no competing model
process. The dashboard generator/escaping checks pass and use no scripts or external
assets. In-app visual navigation to a local `file://` URL was blocked by the browser
security policy; no local-server or alternate-browser workaround was attempted.

**Evidence-integrity follow-up.** The stdlib-only verifier
`research/b27_verify_public_summary.py` compared the path-free B27 publication
artifact against both immutable model raw records and both preserved pre-measurement
failure records. It checked the exact model/revision/manifest/quantisation binding,
runtime tree/base commit, protocol, published medians and intervals, token identity,
resources, raw hashes, failure type/stage and local-path absence. Result: `ok=true`,
two cells, two failures, zero errors, `activation_allowed=false`, and no qualification
change. Verification artifact SHA-256:
`752b01b8f20dc695ed610762e3c9f4b8774a97075c5558199704834504bf684e`.
The exact unapproved D1 type/status/domain contract and its kill criteria are recorded
in [`docs/B27_PHASE_D_CONTRACT_PROPOSAL.md`](../docs/B27_PHASE_D_CONTRACT_PROPOSAL.md);
it changes no runtime behavior and still requires architecture approval.

## B27 D1 — Approved evidence-contract implementation, pre-measurement (2026-08-28)

The user explicitly approved D1 after the Phase-A–C commit `467d5b8`. D1 adds
`ironmule/evidence.py`, a stdlib-only immutable contract for execution strategies,
validity domains, evaluator-owned evidence and trusted profiles. It is not imported by
Runtime, package root, plans, modes, executors, tuner, benchmark, telemetry or
fingerprint and exposes no execution, persistence, selection or activation method.

The implementation closes direct profile-deserialization and self-evaluation bypasses,
requires exact model/revision/manifest/quantisation and closed workload buckets,
separates experiment verdicts from the six lifecycle states, requires raw samples plus
correctness/resource/uncertainty gates for `QUALIFIED`, and turns any domain mismatch
into `REVALIDATION_REQUIRED`. The B27 public adapter is deliberately
`INCONCLUSIVE/SUMMARY_ONLY` and does not invent missing state, crash, RSS or absolute
swap evidence.

Pre-measurement verification passed: 15 focused D1 tests, 26 final
D1/baseline/comparison tests, the full serial non-integration suite at
`146 passed, 11 deselected` in `5.21 s`, and the existing real Gemma-4B integration
suite at `10/10` in `21.24 s`; pre-integration swap was `0 B`. The independent static
review is [`B27d_review.md`](raw/B27d_review.md), SHA-256
`9d146b69d5644a02fb40a127e8927085a47a7a70d2a92e1f6daaa991e6d4a91f`.
The post-change experiment is sealed in
[`B27d_preregistration.md`](raw/B27d_preregistration.md), SHA-256
`846e09499a0eb4f9ff531a6302da9c7913e8b6f620d6ad7834dcaf7fda44de36`.
No post-D1 model timing had been observed when this entry was written.

## B27d — D1 post-change regression screen (2026-08-28)

**Protocol and binding.** B27d ran from clean commit
`0b14eb6f134edc42701ebb1e1a85a1bd484d12d1`, runtime-tree SHA-256
`d7577af8e83778b9753ad4bf721656a16d923a9f848040e406178b7dcffc8a21`.
Exact cached Gemma 3 4B and 12B revisions used the B27a2 strict/base-knob protocol:
six requests, 48 tokens, two warmups and six measured repeats per Interactive and
Throughput arm. Both cells started at 83% free memory, AC, low-power false, swap `0 B`,
with no competing model process.

**Correctness/resources.** Both cells are `BASELINE_CAPTURED`: exact arm token/stop/
count identity, zero fallback/correctness errors, zero swap delta and no residual
process. MLX peaks were `2,784,922,186 B` (4B) and `7,801,381,947 B` (12B).
Raw SHA-256 values are
`10071669abb6c45871bf3d5eec0df3f37104341bb197394a840bf64e46a7be44` and
`41d9bd16b179357ae1d99edf26abba135d1c2b8315bc5c47c421868f5b977a96`.

**Frozen result:** `INCONCLUSIVE_POTENTIAL_REGRESSION`, regression kind
`POTENTIAL_CODE_REGRESSION`. Ratios are post/pre with independent 10,000-resample
bootstrap intervals:

| Model/arm | Wall ratio [95% CI] | Physical-rate ratio [95% CI] | 5% gate |
| --- | ---: | ---: | --- |
| 12B Interactive | `1.0055 [0.9815; 1.0288]` | `0.9943 [0.9720; 1.0189]` | pass |
| 12B Throughput | `0.9995 [0.9864; 1.0224]` | `1.0006 [0.9788; 1.0136]` | pass |
| 4B Interactive | `1.0575 [1.0530; 1.0621]` | `0.9456 [0.9417; 0.9496]` | miss |
| 4B Throughput | `1.0643 [1.0571; 1.0676]` | `0.9396 [0.9366; 0.9460]` | miss |

The 4B movement is common-mode: both absolute arms slow together, while the within-cell
grouping wall ratio changes only `0.8437 -> 0.8483` and the rate ratio
`1.1852 -> 1.1789`. The 12B endpoints are unchanged. The post 4B process also began
with higher load averages, and the only `ironmule/` source added between commits is the
non-imported D1 module. Those diagnostics weaken a causal-code reading but do not
override the preregistered result.

**Decision.** D1 remains outside the runtime import/execution path and is not activated.
No no-regression or performance-safety claim is made. B27d is not retried or pooled.
Its path-free summary was recomputed byte-identically; SHA-256
`ed2129005ab96df2a103808108c9c5fb0f63e871d7f33caace628e8ef7848c37`.
The new backlog entry B27e preregisters the next mechanism-level discriminator:
mirrored fresh-process 4B runs across the pre-D1 and D1 commits, using only new data.

**Final handoff verification.** On the documented B27d result state, the full serial
non-integration suite passed `146 passed, 11 deselected` in `5.18 s`; the existing real
Gemma-4B integration suite passed `10/10` in `21.12 s`. Pre-integration swap was `0 B`
and no model process was present.

## B27e — Mirrored cross-commit control, pre-measurement (2026-08-29)

B27e is sealed as a new control rather than a B27d retry. It uses two detached exact
targets (`467d5b8` OLD and `0b14eb6` D1), proves their declared 16-file execution
surfaces byte-identical, and runs four new fresh 4B processes in mirrored order
OLD/D1 then D1/OLD. No B27a2/B27d timing is pooled.

The parent/child harness passed 10 focused tests; the full serial suite passed
`151 passed, 11 deselected` in `5.15 s`. Harness SHA-256 is
`bde2181490389e3838c73be1ed2d6c2e58a4bdfa094ab8ee3497528133a1283d`;
review SHA-256
`7803639a8ebaf4ec8fa900253522aae7c5c14741059bf3e2f531f054ef2774bf`;
preregistration SHA-256
`78bec8adb2757ae833146cde0d7cd1e4ad78f8418689761e02707b3b980e32f4`.
No B27e model timing had been observed when this entry was written.

## B27e — Mirrored cross-commit control result (2026-08-29)

**Binding and execution.** Two clean detached targets ran exact commits
`467d5b8bfb187cd3dad46cc87e6ada5afbf033dc` (OLD) and
`0b14eb6f134edc42701ebb1e1a85a1bd484d12d1` (D1). Their declared 16-file execution
surface was byte-identical, SHA-256
`ec242cc4872014d7994c6e11cf0b32bbf145ecca4eac32088c697059e2e48385`;
OLD had no D1 module and D1's module matched its sealed hash. Each target had a private
ignored ProjectAtlas index.

Four fresh 4B children ran serially in the frozen order OLD/D1 then D1/OLD. Every
preflight recorded 82% free memory, AC, swap `0 B`, no model process. All four
returncodes were zero, model/framework/protocol domains matched, token/stop/count
identity held, and there were no correctness/resource hard failures or residual
processes.

**Frozen result:** `ORDER_OR_TEMPORAL_DRIFT`; B27d consequence
`B27D_REMAINS_INCONCLUSIVE`.

| Block/order | Arm | D1/OLD wall | D1/OLD rate | Reading |
| --- | --- | ---: | ---: | --- |
| 0 OLD -> D1 | Interactive | `0.9925` | `1.0076` | within 5% |
| 0 OLD -> D1 | Throughput | `0.9841` | `1.0161` | within 5% |
| 1 D1 -> OLD | Interactive | `0.9422` | `1.0613` | D1 appears faster |
| 1 D1 -> OLD | Throughput | `0.9267` | `1.0790` | D1 appears faster |

D1 was not slower in either block, so B27d's common-mode 4B slowdown did not reproduce
as a consistent commit association. The mirrored magnitude changed materially with
order/time, but the preregistered design requires every comparison inside 5% before
calling commits indistinguishable. The result therefore cannot be upgraded to
neutrality; it also provides no evidence for removing D1. No routing or activation
follows.

Raw SHA-256 values: parent
`ecd2c18306083bf59f2e370c0192ef8148beac97852dacff99c0fead5cd3e20a`;
children `2070d965…`, `63a3264e…`, `ae645093…`, `78c16b18…` in execution order.
The path-free summary SHA-256 is
`d80960b022f3f506f592d5e4db19a1aabda07492d5db2c09e40469ad474f4f94`
and was recomputed byte-identically from parent-bound child hashes.

**Artifact-name incident.** The measured harness hardcoded a `20260828` suffix in the
four private child filenames although the records, parent and public artifact are
correctly B27e/2026-08-29. No existing file was overwritten and content/hash/analysis
is unaffected. The records were not renamed or rerun. The post-result harness now
requires an explicit validated `YYYYMMDD` argument and its reanalysis path rejects any
changed child hash. This is a tooling correction, not new evidence.

**Decision.** B27e leaves B27d formally inconclusive and closes its own backlog entry;
the same two-block unconditioned control is not rerun. D1 remains immutable,
non-imported and unactivated. A further architecture or conditioned measurement stage
requires a new explicit decision rather than inference from these data.

**Final verification.** The complete serial result-state suite passed
`153 passed, 11 deselected` in `5.12 s`; the real Gemma-4B integration suite passed
`10/10` in `21.17 s`. Pre-integration swap was `0 B` and no model process was present.

## D2a — Exact-identity pre-change baseline, pre-measurement (2026-08-29)

The user explicitly approved D2: exact local revision/manifest/architecture/
quantisation/tokenizer propagation into Runtime fingerprints and tuned-profile
compatibility, with no strategy selection, EvidenceRecord persistence or activation.

D2a is a new same-day pre-change baseline on the clean commit containing its sealed
protocol. Runtime-tree SHA must remain `d7577a…`; baseline harness SHA-256 is
`e6d981583384d4b526af32eb508579a79815bebabea0c64c8a2f4d99ebfe74d4`;
contract/review SHA-256 values are `ebfb372f…` and `8327a778…`.
Gemma 3 4B then 12B use the strict six-request/48-token, 2-warmup/6-repeat protocol
with exact local snapshots and no stored profile. D2a never pools B27 data and creates
no qualification. No D2a timing had been observed when this entry was written.

## D2a — Exact-identity pre-change baseline result (2026-08-29)

Both new same-day cells completed as `BASELINE_CAPTURED` on clean commit
`a0778e12cc0cee6d7a62523ce6b18593998fe619`, unchanged runtime-tree SHA-256
`d7577af8e83778b9753ad4bf721656a16d923a9f848040e406178b7dcffc8a21`.
Preflight was 87% free memory, AC, low-power false, swap `0 B`; outputs were
token/stop/count identical with zero fallback/correctness errors and zero swap delta.

| Model | Interactive outer/rate | Throughput outer/rate | MLX peak |
| --- | ---: | ---: | ---: |
| Gemma 3 4B | `3939.53 ms / 73.105 tok/s` | `3367.33 ms / 85.528 tok/s` | `2,784,919,610 B` |
| Gemma 3 12B | `10076.32 ms / 28.583 tok/s` | `8822.69 ms / 32.644 tok/s` | `7,801,366,427 B` |

Raw SHA-256 values are
`c012c9a3e9b25d995e940d363137238f717a42ccae611f52354d7779cbad39d9`
and `745d63222c42937e72bfb5b32b5e5773ed727b6f3366b229dcd2c0f5c76817aa`.
The path-free summary recomputes byte-identically, SHA-256
`6eddb942af04addb245e624b189b90e095fc6eb591abe413159541b4f1c63ea6`.
This is the only pre-change timing source allowed for D2b. No qualification or
activation follows, and D2 source implementation had not begun at result capture.

Pre-implementation verification passed `155 passed, 11 deselected` in `5.37 s`; the
latest unchanged-runtime real Gemma-4B integration gate remained `10/10` in `21.17 s`.

## D2 — Exact model identity implementation, pre-measurement (2026-08-29)

D2 is implemented within its approved boundary. The new stdlib-only
`ironmule.model_identity` resolves one exact local source and creates a path-free
immutable identity from revision, complete present-file manifest, architecture,
canonical quantisation and tokenizer artifacts. Runtime fingerprint v2 and
tuned-profile conditions v2 require every identity field; missing, legacy, ambiguous
or inconsistent identities fall back to baseline or raise before validity reuse.
`mlx_lm.load` keeps its two-value caller shape through `load_engine`, and a second
full identity reconstruction detects a source change during load. Hashing is outside
the timed `Runtime.serve` path.

The two exact cached identities independently reconstructed by the D2 comparison
harness are `2730e8b13b892b576452493dfb1983c0948c175d02c50099475385f8bac97bd2`
(Gemma 3 4B) and
`2b5b13a3c53a96299b33d0385b13a4b54973b810540cf7a99d4aa3966ebf1474`
(Gemma 3 12B). Their manifest digests remain `a405b1a7…` and `aef12412…`;
both tokenizer and quantisation digests are respectively `afbd505b…` and
`4952fcd6…`. No file path is serialized.

Pre-D2b verification passed 39 focused identity/comparator/profile tests, the full
serial non-integration suite at `178 passed, 12 deselected` in `4.98 s`, and the real
cached Gemma-4B integration suite at `11/11` in `22.14 s`. Post-integration swap was
`0 B` and no residual model process was present. The old incomplete local profile was
not reused, including with raw revalidation access. No model or dependency was
downloaded or installed.

**Recorded execution incident.** One focused pytest command was accidentally invoked
inside the restricted sandbox. MLX aborted during import with `SIGABRT`/exit `134`
before any model, test or timing arm ran. Root cause was Metal/MLX initialization in
the unsupported sandbox. The successful remedy is to run all IronMule pytest/model
commands serially with the existing project Python outside that sandbox. The corrected
focused suite passed; the crash is not a measurement and is neither retried as a data
point nor pooled.

D2 still contains no D1 EvidenceRecord persistence, strategy selection, plan/mode
routing, automatic activation, download path or inference-semantic change. The
independent D2b comparator and its 5% correctness/resource/performance gates are
implemented, but no post-D2 timing had been observed when this entry was written.

## D2b — Exact-identity post-change screen, pre-measurement (2026-08-29)

D2b is sealed against implementation commit
`7892810584be232cec744c0038ab9b3e069608ea` and runtime-tree SHA-256
`5759506d46ee006e6f2873312f2d8a8ac857be1d1488b59cafbb09b9de7a5e60`.
It compares only the same-day D2a raw 4B/12B records (`c012c9a3…`, `745d6322…`),
in fixed 4B-then-12B order. Each post cell must contain the independently
reconstructed exact Runtime identity and matching fingerprint-v2 fields before the
correctness/resource and 5% bootstrap regression gates are evaluated.

The implementation review SHA-256 is
`a0f634a77515741db17e3205ffb827f2d318439e7294ea399eead4a890792e5f`;
the preregistration SHA-256 is
`6ffc3a6714aa8ed2a2e71e1ebd6af9a5f284a171e8ba69a5e959f7802c070c1b`.
No D2b timing had been observed when these documents were sealed. There is no retry,
pooling, threshold change, qualification, routing or activation consequence.

## D2b — Exact-identity post-change result (2026-08-29)

**Binding and execution.** D2b ran from clean preregistration commit
`d36a6538d6c4a2a0fa4ac278511b0fefdeb82fd5`, with the frozen D2 runtime-tree
SHA-256 `5759506d46ee006e6f2873312f2d8a8ac857be1d1488b59cafbb09b9de7a5e60`.
The exact cached 4B process ran first, then 12B after memory recovered. Both preflights
recorded 86% free memory, AC, low-power false, swap `0 B` and no competing model
process. No download, install, network fallback, retry or sample pooling occurred.

**Identity, correctness and resources.** The independently reconstructed Runtime
identities were exactly `2730e8b1…` (4B) and `2b5b13a3…` (12B), and both Interactive
and Throughput fingerprints matched schema v2 and every revision/manifest/
architecture/quantisation/tokenizer/aggregate field. Both cells were
`BASELINE_CAPTURED`; token IDs, stops and counts matched, with zero fallback,
correctness errors or swap delta. MLX peak was `2,784,918,586 B` (4B) and
`7,801,367,451 B` (12B). Raw SHA-256 values are
`bab01abb6e9c4aa09d7ab06fcb4074a54ec855cd46ee0310a3bff6bba04c6cf5` and
`6ddc586d4c43c5d02cadcbecd19ece198f640dfde39276be37700152bf1746a4`.

**Frozen result:** `NO_REGRESSION_OBSERVED`, regression kind `NONE`. Ratios are D2b
post / same-day D2a pre with the preregistered independent 10,000-resample intervals:

| Model/arm | Wall ratio [95% CI] | Physical-rate ratio [95% CI] | 5% gates |
| --- | ---: | ---: | --- |
| 4B Interactive | `0.9978 [0.9969; 0.9993]` | `1.0022 [1.0007; 1.0031]` | pass |
| 4B Throughput | `1.0000 [0.9943; 1.0031]` | `1.0000 [0.9969; 1.0057]` | pass |
| 12B Interactive | `0.9888 [0.9681; 1.0055]` | `1.0113 [0.9945; 1.0329]` | pass |
| 12B Throughput | `1.0077 [0.9851; 1.0243]` | `0.9924 [0.9764; 1.0151]` | pass |

There was no domain drift, hard failure or performance miss. The path-free D2b post
summary SHA-256 is
`16741c99e03ce2ab821ff7b40dd42eb105ff57855d74a75ed422882cd8603132`;
the comparison SHA-256 is
`0a02d1fed48f742d6c169b083b98a5a6b5fd9dbfee1d43981080f44e75b8144e`.
Both recomputed byte-identically from the four immutable raw records. Verification
artifact SHA-256 is
`19de9149ae5c697cf50c8535bc451c266986df0da81821c4669491b1b20cf221`.

**Decision and limits.** D2/R6 exact identity is complete. The result supports only
that the approved identity plumbing did not cross its frozen 5% engineering
regression gates in these two cells. It is not a speed or quality claim, stock-MLX
comparison, tuned-profile qualification, selection, routing or activation. D1 remains
unpersisted and no strategy consumes it. Any next B27 architecture stage needs a new
explicit decision.

**Final verification.** The full serial non-integration suite passed
`178 passed, 12 deselected` in `5.03 s`; the real cached Gemma-4B integration suite
passed `11/11` in `22.28 s`. Final swap was `0 B`, memory free `85%`, and no model
process remained. Xcode and IronMule doctor were green. ProjectAtlas `0.4.5-rc1`
runtime and project-local MCP configuration were verified; its private index was
fully refreshed after a dependency-closure-limit warning, and lint returned
`ok=true`. Worktree alias enumeration reported the known shared-control-repository
limitation and did not change Git or source files.

**Post-measurement UI repair.** While adding D2b history, the dashboard generator
revealed an older presentation-only variable-shadowing defect: B27e rows replaced the
protected baseline table in generated HTML. The cross-control row variable was
renamed, D2b post/comparison/verification inputs were added, and the local page was
regenerated with the original baseline rows restored. This happened after all D2b raw
records and the frozen comparison existed; no runtime, measurement harness, comparator
or result changed.

The first presentation regression run failed one stale assertion because it still
expected the earlier D1 suite label (`146 passed`) after D2b correctly became the
newest verification source (`178 passed`). Updating that expectation fixed the test;
no rendered metric, raw record or comparison logic changed.

## Q3 — Offline adaptive replay data gate (2026-08-31)

The Q3 replay builder remains an offline, no-runtime-import contract. Its frozen
dataset is SHA-256 `f67d975788763e4238019a3be7afa5394efbe2f2faea3a96a927e7cf522f2e33`
with dataset ID `d4ae0c148e826de85c7aa5338f892b5571481a105f558d463e9d041f63dc82b7`:
14 observations, 12 actions and 160 B36 raw timing samples. Q2 is a validation
trajectory and B36 is a sealed holdout; no training rows exist. The structural gate
is `DATA_INSUFFICIENT` for current coordinate descent, seeded random, BO, surrogate
and contextual bandit; offline RL is `NOT_APPLICABLE` because Q3 has no measured
sequential horizon. This is an eligibility/data-quality result only: no timing,
hardware, performance, generalisation, BO or RL claim, and no runtime selection,
persistence or activation follows. Missing evidence remains separated into a
complete counterfactual action panel, independent grouped contexts, and a measured
sequential horizon.

## Q3b — Residual-swap safety canary (2026-08-31)

**Raw and audit.** The retained raw record is
`research/raw/Q3b_canary7_20260831.json`, SHA-256
`77ebc1ed8af5c1d5b4b064ce95605d3440b6e2fccabcd088d58f0900cdd0eb76`. A
read-only audit recomputed the complete two-stage shape, three measured repeats
per stage, sample-array lengths and maximum adjacent sampler gaps, resource
gates, raw identity flags and cross-stage identity. Both stages were complete
and the result is `SAFETY_CANARY_PASS` / `SAFETY_ONLY`; the declared
`performance_valid=false` and `promotion_allowed=false` remain binding.

**Binding and safety.** The exact local model was
`mlx-community/gemma-3-4b-it-4bit`, revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`, manifest SHA
`a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae`, with
runtime-code SHA
`d4577826e46d356ecc43cbae0c94465018d202ad29593a96a5f2693d1f279e59`.
Preflight passed on AC, low-power off, nominal thermal, clean/bound Git,
known 32 GiB memory, start free memory `49%`, start swap `1,690,891,714 B`
(within the `4 GiB` start cap), and loadavg max `2.7905`. Baseline and
candidate stages passed the live residual-swap policy: free memory `44%` and
`46%`, swap delta `0 B` for each, loadavg max `3.8555` and
`4.2837`, child RSS peaks `3,075,457,024 B` and
`3,767,861,248 B`, and MLX peak `3,125,869,452 B`. Sampler arrays were
complete (`81` and `73` samples), had no errors, and all child groups were
reaped. Maximum adjacent sampler gaps were `0.330351 s` and `0.272347 s`,
below the `1.75 s` bound.

**Exactness.** Prompt tokens were `322`; each stage produced exactly `23`
logical and `23` physical output tokens, `22` decode steps, capacity `384`,
`eos` stop reasons, matching per-repeat counts, and deterministic output.
The baseline arm was the Q2 incumbent
(`compiled_fixed_cache=True`, `head_skip_prefill=True`,
`readback_every=2`); the candidate differed only by `fused_argmax=True`.

**Descriptive timing only.** These are raw-repeat medians from the two
ordered single-arm stages, with no CI and no performance validity:

| Stage/arm | Total ms | Prefill ms | Decode ms | Physical output tok/s | Decode steps/s |
| :-- | --: | --: | --: | --: | --: |
| Baseline / Q2 incumbent | `859.413` | `583.330` | `276.083` | `26.7624` | `79.6862` |
| Candidate / +`fused_argmax` | `849.714` | `574.051` | `275.663` | `27.0679` | `79.8076` |

The candidate-over-baseline descriptive ratios are total `0.988715`
(`+1.1285%` faster), prefill `0.984093` (`+1.5907%`), decode
`0.998479` (`+0.1521%`), output rate `1.011414`
(`+1.1414%`) and decode-step rate `1.001524`
(`+0.1524%`). They are safety-canary context only: do not
multiply them with Q2 or use them to qualify, promote, route, or activate a
profile. The completed Q3b backlog entry is closed; the general P2 lifecycle,
identity, and streaming-output debts remain above it in the backlog.

## Q3c — Preregistration sealed before implementation (2026-08-31)

`research/raw/Q3c_preregistration.md` was frozen before Q3c implementation or
execution. Its SHA-256 is
`3bf63ff0dcf442855b6d7b97278fb1d43583a9f18e3f5b6c3caa507582a9ffc5`, recorded
in `research/raw/Q3c_preregistration.sha256`. It defines two independent
six-fresh-process `ab.run` phases (2 warmups, 7 repeats, alternating AB/BA):
Phase R reproduces the exact Q2 incumbent against untuned `BASE`, and Phase N
tests that incumbent plus `fused_argmax` against the same `BASE`. The local
Gemma 4B, prompt token count `322`, `max_tokens=32`, residual-swap/live
safety policy, exact `600 s` study / `270 s` phase / `240 s` worker /
`35 s` child bounds, identity rule, timing/rate/CI outputs, Q2 target
`0.8568` with CI `[0.8549; 0.9402]`, `±0.03` reproduction bar, candidate
`+0.005` preservation bar, fallback and no-promotion requirements were frozen
there. No UI history is part of the execution contract.

## Q3c — Safety aborts before any performance result (2026-08-31)

Two Q3c records are retained locally and are not pooled. Run 1,
`research/raw/Q3c_run1_20260831.json` (SHA-256
`5270c0f38e50984cd26223aa2a9817982fc5a1861ddbe2caa3cff98393c9e8d5`), failed
in preflight because load `8.294921875` exceeded the declared maximum `8`.
It entered no phase and has no model timings or identity observations.

Run 2, `research/raw/Q3c_run2_20260831.json` (SHA-256
`d94db80402254c87c0e4a0128cf802e1eaa59d42c4459c2f208077f48c38b8df`), passed
preflight but stopped Phase R on the live safety gate after `105` swap samples
over `27.394551749996026 s`. Swap rose from `2,353,654,661 B` to
`2,625,172,930 B`: delta `271,518,269 B` (`258.94 MiB`), above the unchanged
`128 MiB` limit. The raw record reports cleanup errors
(`SIGTERM:PermissionError`, `SIGKILL:PermissionError`, worker process group
still alive) and `worker_group_gone=false`; cleanup is therefore unverified.
No child completed, and there are no timings, exact identity, performance,
promotion or activation results. The Q3c decision is `FAILED` with
`BASE/current incumbent` fallback. The Q3c backlog entry is closed in Tier 0;
Q3d is a separate, single-path recovery preregistration.

## Q3d — Stability gate passed; Q3c blocked by macOS process-probe portability (2026-08-31)

The model-free Q3d gate completed its frozen protocol and passed all of its
gates. The retained raw record is
`research/raw/Q3d_stability_20260831.json`, SHA-256
`4699a49b174db31580a9701ef2075f8b1964d309b0f857dd7779fb230cfccb83`, size
`34,144` bytes. Its companion summary is
`research/raw/Q3d_summary_20260831.json`, SHA-256
`3b43e267000ba15b9d9079d9f118e59c1cd51dbcdfecc067c20995b01a0a1c3e`, size
`970` bytes.

**Gate evidence.** The gate recorded exactly `61` swap samples: one
synchronous `t0` sample plus 60 scheduled samples. First-to-last elapsed time
was `60.020192667 s`, the maximum adjacent gap was `1.013944625 s`, and every
sample reported the same swap value `2,651,722,874 B`; the measured high-water
delta was exactly `0 B`. AC power, low-power-off, nominal thermal state, known
32-GiB memory, free memory `62%`, load maximum `3.92578125`/spread `0`, clean
Git and the exact local Gemma identity all passed. The gate was model-free: no
MLX import, model load, inference child or timing arm ran.

**Terminal Q3d result.** The one permitted unchanged Q3c invocation was not
started (`invoked=false`) because its pre-spawn process baseline was
unavailable. The underlying macOS command was
`/bin/ps -Ao pid=,ppid=,pgid=,sid=,uid=,stat=,start=,args=`; on this host
(macOS `26.6.2-arm64`) it returned `rc=1` with
`ps: sid: keyword not found`. This is a portability defect in the OS probe,
not evidence that the model or hardware was unsafe. No Q3c raw output exists,
and there are no timing, exact-identity, performance or promotion values.

The Q3d summary status is `Q3C_FAILED`, with `promotion_allowed=false` and
fallback `BASE/current incumbent`. Q3d is closed: its gate PASS is retained as
safety context only, is not pooled with Q3c and is not a current-model or
performance guarantee. The next and only permitted path is the separately
frozen Q3e portability repair, whose preregistration
`research/raw/Q3e_preregistration.md` has SHA-256
`71901b0d2220d7e9559bad536afaf15d04fac5ea1714f7ceade5bc37811dfd47`, recorded
in `research/raw/Q3e_preregistration.sha256`. Q3e may remove only the unsupported
`sid` field, use typed public `os.getsid(pid)` enrichment with fail-closed
unknown/error handling, prove it with model-free tests, and then invoke the
unchanged Q3c harness exactly once. No retry, download, installation, restart,
27B run, UI or automatic activation follows from Q3d.

## Q3e — Portable probe repaired; Phase R rejected by same-UID attribution (2026-08-31)

Q3e passed all 14 preflight checks and repaired the macOS `ps sid` portability
boundary. Its single permitted Q3c invocation completed Phase R with six fresh
processes, alternating order, two warmups and seven measured repeats per arm.
The exact Gemma 4B revision, manifest, prompt count, token IDs, physical token
IDs, counts, stop reasons, capacity, decode steps and determinism were equal in
every measured repeat. Swap stayed at `2,643,334,266 B` with zero delta, and
the resource ceilings passed.

The authoritative raw result is
`research/raw/Q3e_q3c_final_20260831.json`, SHA-256
`1df6c81dc824911016e687883c535f1ec314f3e03b51303b04c38ae71bb6f4ea`, size
`2,205,857` bytes. It is `FAILED`, with `BASE/current incumbent` fallback and
`promotion_allowed=false`. The redacted terminal note is
`research/raw/Q3e_terminal_result_20260831.md`, SHA-256
`fd89e23945315597476854843df1140a5c3e35ebf1aaf9a737c80d5ebf4fdfaa`.

The descriptive Phase-R incumbent/BASE total-time ratio was
`0.857466859207542`, bootstrap 95% CI
`[0.8551668079699586, 0.8611021999710893]`, or `14.2533140792%` faster total
time. Prefill was `16.0213211%` faster, decode `10.0321650%` faster, physical
output rate `16.6225872%` higher and decode-step rate `11.1510142%` higher.
These values meet the frozen historical Phase-R checks but are descriptive
only and are not an accepted performance result because cleanup failed.

Both independent final cleanup snapshots showed the worker group, leader and
known descendants gone, and the worker was reaped. Cleanup nevertheless found
four stable new same-UID processes outside the worker group/session and outside
the worker ancestry: PID `28095` (`extensionkitservice`), PID `28209`
(`STARFACE HeadsetXPCService`), PID `28636` (`mdworker_shared`) and PID `28964`
(`AXVisualSupportAgent`). The strict Q3e rule treated any new same-UID process
as unresolved, so `group_gone=false` and the phase was rejected. The complete
bounded records remain in the raw file; no process was killed by this rule.
Phase N did not run. Q3e is terminal and is not retried or pooled with any
earlier result.

## Q3f — Final same-UID attribution path preregistered (2026-08-31)

Q3f is frozen in `research/raw/Q3f_preregistration.md`, SHA-256
`345c63cba5f019ab0314761404f7de398ceee876ffcee82d80c3578f9db8e31b`, recorded
in its companion SHA file. It permits only a strict `unrelated_new_process`
classification for a new same-UID process when two valid snapshots prove stable
PID/start/UID, separation from worker/known/nested PIDs and PGID/SID, complete
non-ancestry, no model/inference tokens, known non-zombie state, no competing
model process and complete command/enrichment evidence. Any identity ambiguity,
model-like process or an executable/args combination indicating Python and
containing an exact blocker token, ancestry/group/session relation,
unknown/malformed/racy evidence or other cleanup uncertainty remains a hard
failure. The full bounded record is retained and no unrelated process may be
killed.

Q3f additionally requires `ab._child` to install the bounded Python audit guard
`ironmule.q3f_child_guard.v1` before model load. It blocks and records
`subprocess.Popen`, `os.system`, fork/spawn and available `setsid`/`setpgid`
process/session escape attempts; a successful child records the exact guard
version and zero events. The direct child-start callback is the complete child
ledger. Missing, overflowing or otherwise unknown guard/ledger evidence fails
closed. The exact case-insensitive blocker set is the existing
`KNOWN_INFERENCE_ACTIVITY` constant plus `q3c`, `q3d`, `ironmule`, `mlx`,
`gemma` and `huggingface`, with equality covered by static and adversarial
tests.
The guard starts at the actual `ab._child` execution closure, follows only a
reviewed callee/module allowlist, excludes the legitimate parent-side
`ab.run` `Popen`, and fails on an unreviewable reachable Python path. It
receives all Python-visible audit events and wraps available `os` process/
session calls; arbitrary native C-level syscalls are not claimed observable
and remain subject to the strict snapshot/group/session/ledger gates. The
Python-inference predicate is executable/args indicating Python **and** at
least one exact blocker token; generic external Python is not automatically
inference, but still faces every structural unrelated-process gate.

Before one unchanged offline Q3c invocation, Q3f allows only model-free
adversarial and real macOS `start_new_session=True` cleanup tests, run serially
with the full non-integration suite. Q3c remains exactly bound to its frozen
preregistration, local Gemma 4B revision/manifest, unchanged arms/order,
preflight and live safety limits, six fresh processes per phase, exact identity,
statistics, timing bounds and no-promotion fallback. Q3f has no Q3d gate, retry,
pooling, download, installation, restart, 27B run or UI; any failure consumes
the single path and ends the study.
