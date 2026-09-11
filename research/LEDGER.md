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

## Q3f — Terminal result (2026-09-01)

The single frozen Q3f path was executed once against the exact local Gemma 4B
revision and ended `FAILED`. Raw result:
`research/raw/Q3f_q3c_final_20260901.json`, size `2,487,533` bytes, SHA-256
`e82accdbd52857e6201fa2b34984765e61658ecfd3956d5c903a49c1e6de70a9`. The
redacted terminal note is `research/raw/Q3f_terminal_result_20260901.md`.

All 14 preflight checks passed: AC power, free memory `67%`, nominal thermal,
low-power off, exact model identity and clean runtime binding. Swap was
`2,609,643,520 B` and remained constant over 25 samples (`delta=0`, no sampler
errors). The phase worker exited with status `2` and the bounded error
`ABRunError: child 0 start callback failed`, before any child-start marker,
child ledger, timing, token, Phase-N, phase-gate or resource result was
available. No accepted performance result exists. The exact lower-level cause
is not preserved; an immediate process/command visibility race is only an
unproven future hypothesis.

Cleanup used `ironmule.cleanup.v2`: the worker was reaped, two independent
snapshots were valid, no worker-group member or descendant remained, and no
kill was needed (`SIGTERM: not_needed_group_already_gone`). The evidence still
failed closed because the guard and direct child-start ledger were unavailable.
A stable same-UID Spotify Helper (`PID 52017`) was outside the worker group and
was not killed, but could not be accepted as unrelated without the missing
proof. There was no orphan left at recording time.

Status is terminal `FAILED`, `promotion_allowed=false`, fallback
`BASE/current incumbent`. Do not retry or pool Q3f with Q3e/Q3d/Q3c, and do not
claim historical Q2 reproduction or a new speed gain. A future child-visibility
hardening requires explicit authorization and a new preregistration.

## B55 — One router over the qualified paths (2026-09-10)

**Question.** Over a mixed dispatch sequence, does an `AppleRuntime`/`ExecutionRouter`
layer reach the faster fixed service mode on each workload, and does the wrapper itself
cost anything against naming that mode by hand?

**Design.** One process, one model load, four arms over the same loaded engine:
`I_interactive` and `T_throughput` named by hand, `R_routed` (`objective="throughput"`)
and `L_routed_latency` (`objective="latency"`). Workloads: one, two and four requests
dispatched together, 48 requested tokens each, `mlx-community/gemma-3-4b-it-4bit`
revision `93724907d4ed`, identity `2730e8b1…`, hardware `dc652d66f24ac207`, MLX `0.32.0`,
mlx-lm `0.31.3`, tuned knobs from the same-day autotuner confirmation
(`14.71%` faster end to end, tokens identical). Sixteen blocks, arm order rotated by
block index. Correctness gates the timing and is not derived from it.

**Result: `ROUTER_REACHES_BEST`.** Token IDs, physical token counts and stop reasons were
identical across all four arms for all `224` compared requests. Zero fallbacks, zero swap
growth. Paired per-block ratios, median with `95%` bootstrap over `10,000` resamples:

| workload | `R/T_throughput` | `R/I_interactive` | `T/I` | `L/I_interactive` |
| :-- | --: | --: | --: | --: |
| one request | `1.0017` [1.0001; 1.0027] | `1.0008` [0.9974; 1.0037] | `0.9987` | `1.0012` [0.9911; 1.0055] |
| two requests | `0.9951` [0.9882; 1.0028] | `0.9274` [0.9245; 0.9380] | `0.9353` | `1.0001` [0.9983; 1.0020] |
| four requests | `1.0001` [0.9970; 1.0022] | `0.8260` [0.8250; 0.8265] | `0.8261` | `1.0021` [0.9966; 1.0117] |
| mixed total | `0.9979` [0.9937; 1.0027] | `0.8807` [0.8768; 0.8841] | `0.8819` | `1.0011` [0.9974; 1.0053] |

**What this is and is not.** The router *matches* the better fixed mode everywhere and is
`11.9%` faster than a fixed `InteractiveMode` over the mixed sequence. It does **not**
beat a fixed `ThroughputMode` on wall time; on this workload that mode is never worse.
What a fixed mode cannot do is cover both objectives, and the per-request median latency
is where the trade is visible:

| workload | `I_interactive` | `T_throughput` | `R_routed` | `L_routed_latency` |
| :-- | --: | --: | --: | --: |
| one request | `615.0 ms` | `612.3 ms` | `612.2 ms` | `612.0 ms` |
| two requests | `614.9 ms` | `1075.1 ms` | `1072.8 ms` | `680.6 ms` |
| four requests | `1827.2 ms` | `1949.2 ms` | `1947.7 ms` | `1818.5 ms` |

At two ready requests grouping buys `+7.7%` aggregate tokens per second at `+74.9%`
median per-request latency; at four it buys `+24.4%` for `+6.7%`. That is the `E15`/`E16` trade at this workload, no dispatch-time fact resolves
it, and so `objective` is a caller parameter rather than a guess. `L_routed_latency` took
the sequential route on every workload, as preregistered.

**A wrapper that changed what it measured.** Attempt 1 returned
`ROUTER_DOES_NOT_REACH_BEST`: the routed single request was `3.7%` slower than naming its
mode by hand. Direct profiling located all of it — `router.decide()` cost `0.006 ms` and
`Telemetry.snapshot()` `0.020 ms`, while two `sysctl(8)` subprocess swap probes cost
`17.4 ms` of a `21 ms` overhead. `ironmule.hw.swap_used_bytes` now reads `vm.swapusage`
through `sysctlbyname`, measured at `0.0015 ms` and byte-exact against `sysctl -n`.
Attempt 2 then passed. Attempts 3 and 4 re-bound the evidence to the shipped code after
the `objective` parameter was added; attempt 3's eight mirrored blocks could not resolve
the `2%` equivalence margin against a measured per-dispatch spread near `5%`, so
sixteen rotated blocks were preregistered before attempt 4 rather than argued for after
it. All four attempts are kept.

**Raw data (local, `.gitignore`d).**
`research/raw/B55_router_preregistration_20260910.json` (`2,722 B`,
`093cfe85…`), `…_v3.json` (`3,319 B`, `4cfdd323…`),
`B55_router_20260910.json` (attempt 1, `479,729 B`, `7abd9f22…`),
`…_attempt2.json` (`479,709 B`, `6dda93c3…`),
`…_attempt3.json` (`794,798 B`, `ba4fea12…`),
`…_attempt4.json` (`1,580,059 B`, `d83951b5…`).
Measured sources digest `c2d11dec…`. Harness `tools/b55_execution_router.py`.

**Status.** MEASURED. Valid for this machine, this model revision, this MLX/mlx-lm build
and this workload only. The paired route was never selected: no service-strategy record
is stored in this profile, so `AutomaticMode` correctly kept the established mode.

## B27 — Evidence-bound execution strategies, closed (2026-09-10)

The architecture track asked whether the qualified fast paths could be selected through
one explicit contract without weakening fail-closed correctness. D1 built the contract
off the import graph, D2 bound exact model identity into fingerprints and profiles, and
`B27d`/`B27e` ended `INCONCLUSIVE_POTENTIAL_REGRESSION` and `ORDER_OR_TEMPORAL_DRIFT`
respectively — the entries above hold those results.

`B55` answers the remaining question. `ironmule/router.py` selects between the existing
paths using only dispatch-time facts, refuses everything on an unqualified fingerprint,
never substitutes an execution plan, and delegates the throughput/paired split to the
profile record that already owned it. It required no parallel execution implementation
and added no per-token check. B27 is closed; `B56` to `B59` in `docs/BACKLOG.md` are the
questions it left open.

## B56 — A per-request objective, and the one case it cannot cover (2026-09-10)

**What shipped.** `Request.objective` (`latency`, `throughput`, or `None`),
`AppleRuntime.serve(..., objective=...)`, and `ExecutionRouter.plan()`, which splits a
dispatch into one cohort per objective under `ironmule.dispatch.objective_cohorts.v1`:
resolve each request's objective (request, else dispatch, else runtime), partition
preserving caller order, order cohorts by `(earliest arrival, latency before throughput)`,
serve each on the route the router picks sharing one dispatch timestamp, return results in
caller order. A request that names nothing inherits the runtime objective, so every call
written before this behaves identically.

**Result: `B56_NO_GO`** on the preregistered rule, from the binding run
(`…_attempt4.json`, 16 rotated blocks, stable, load `2.90` to `3.57`, `608` requests,
zero fallbacks, zero swap growth).

| criterion | outcome |
| :-- | :-- |
| `C1` correctness | **pass** — tokens, counts and stop reasons identical to a pure `InteractiveMode` reference |
| `C2` routing overhead | **fail** — medians `1.0000` to `1.0044`, but CI uppers reach `1.0232` against a `1.02` margin |
| `C3` latency protection | **fail** — only on staggered arrival |
| `C4` throughput reach | **pass** |
| `C5` determinism | **pass** — zero route disagreements over 16 blocks |
| `C6` no starvation | **pass** |

`C2` is a power statement, not a cost statement, and must be read as one. Every median sits
within `0.5%` of `1.0`; the intervals are simply too wide at this block count to *certify*
`±2%`. No extension to significance was run.

`C3`, against matched latency-only controls (same prompts, same positions, same declared
arrivals):

| comparison | ratio |
| :-- | --: |
| one latency + one throughput, simultaneous | `0.9815` [0.9736; 1.0143] |
| two latency + two throughput, simultaneous | `1.0004` [0.9921; 1.0082] |
| staggered: latency arrives after the cohort starts | `2.1116` [2.0925; 2.1645] |

A hand-written caller doing the same split measured `2.1211` [2.0927; 2.1538] on that last
row. The staggered failure is the device, not the routing.

**Attempts kept.** Attempt 1 compared a mixed workload's latency requests against a
different workload's, confounding prompt mix with the question; matched controls moved
`W5` from `1.0197` to `1.0030`. Attempt 3 was contaminated — an unrelated editor language
server started mid-run and the hand-written arm's wall time doubled from `810` to
`1640 ms` over blocks 9 to 14 — and produced `BLOCKED`-grade noise before the stability
gates existed. Two gates were added afterwards, both of which can only *block* a verdict:
a `4.0` load ceiling at start, and any block deviating more than `25%` from the median.

**Status.** MEASURED, `NO_GO`. The feature is implemented and the default is unchanged.
`objective="latency"` is a **dispatch preference, not a latency guarantee**;
`last_decision["latency_protected"]` reports which case a dispatch was in.

## B56b — An own dispatch does not protect a late latency request (2026-09-10)

**Question.** `B56` left one case open. `B56a`'s earlier `1.00x` for "its own dispatch"
measured the request with *no throughput work running at all*, so it measured an empty
machine and answered nothing. This measures the case that was actually asked about.

**Design.** Two OS processes, each with its own resident model and its own submission to
the device. `A_child` (solo on the child's engine, the floor), `A_control` (A/A),
`A_parent` (cross-engine control, because `B` measures on one engine and `C` on the
other), `B_router` (today's router, one dispatch, the request waits), `C_own` (the parent
serves the cohort while the child serves the latency request from its own process at the
same `60 ms`). `time.perf_counter_ns` is system-wide on macOS, so both processes' stamps
lie on one axis and the concurrency check is a direct comparison. Twelve repetitions,
rotated order, two separate confirmation sessions.

**Result: `B56b_SESSION_NO_GO` in both sessions**, on `G1`, reproducibly.

| gate | limit | session 1 | session 2 | |
| :-- | --: | --: | --: | :-- |
| `G1` protection, `C/A` | < 1.30 | `1.5013` [1.4520; 1.5067] | `1.4702` [1.4418; 1.5199] | **fail** |
| `G2` beats waiting, `C/B` | < 0.75 | `0.4858` | `0.4861` | pass |
| `G3` cohort not starved, `C/B` | < 1.25 | `1.2357` | `1.2190` | pass |
| `G4` A/A control | contains 1.0 | `1.0005` | `0.9978` | pass |
| `G4b` engine control | 0.95–1.05 | `1.0285` | `1.0219` | pass |
| `G5` correctness | identity | identical | identical | pass |
| `G6` concurrency | overlap | `629.6 ms` | `635.7 ms` | pass |
| `G7` resources | see below | clean | clean | pass |

**The concurrency was real and is not the problem.** The child's queue time was `0.03 ms`,
its whole life sat inside the cohort's window, and the overlap was `630 ms`. What the
device does with two submission streams is share itself: the latency request's own decode
rate falls from `75.1` to `50.0` tokens per second, its latency rises from `426` to
`639 ms`, and the cohort's completion rises from `811` to `1001 ms`. Work is conserved.

An own dispatch therefore **halves the wait** (`1319` to `639 ms`) and **does not protect**:
`1.50x` solo, against a `1.30x` limit fixed before the run. No threshold was moved and no
session was extended.

**`G7`, because two resident models are part of what `C` costs.** Zero swapouts,
system free memory `34-36%`, combined peak RSS `7.53-7.54 GB` — `21.9%` of the installed
`34.36 GB`, against a `60%` ceiling of `20.6 GB`. MLX peak per process and the model's own
weight bytes are in the record. The cost is real but was never the binding constraint.

**Semantics.** No preemption is claimed or measured. Nothing here inspects the device
scheduler; what is reported is the observed outcome of two submission streams.

**Consequence.** `own_dispatch` is not pursued and does not enter any stack. The
single-resident-model variant with two command queues is **not** started: it was
conditional on `GO`.

**Raw data (local, `.gitignore`d).** `B56_objective_routing_preregistration_20260910.json`
and `…_v2.json`; `B56_objective_routing_20260910{,_attempt2,_attempt3,_attempt4}.json`;
`B56_staggered_alternatives_20260910.json`;
`B56b_own_dispatch_preregistration_20260910{,_v2}.json`;
`B56b_own_dispatch_20260910_session{1,2}.json`. Harnesses
`tools/b56_objective_routing.py`, `tools/b56b_own_dispatch.py`,
`tools/b56b_latency_worker.py`.

**Status.** MEASURED, `NO_GO`. Valid for this machine, this model revision, this MLX and
mlx-lm build and this workload only.

## B57 — The best combination this machine has earned, executed rather than added (2026-09-10)

**Design.** `tools/b57_stack_composition.py` executes each combination of already qualified
levers as one candidate and measures it whole, because `prefix`, the tuned knob set and the
paired path were each measured against a different denominator and adding their percentages
describes nothing. `ironmule/stacks.py` reduces the candidate set before anything runs and
records why each dropped candidate was dropped. Selection and confirmation are separate
runs. Preregistered as `B57_stack_composition_preregistration_20260910.json`; `…_v2.json`
repeats the identical design with the current binding, because `ironmule/stacks.py` gained
the `own_dispatch` refused record after `v1` was written. Only that one file differs.

**Plans are not compared across plan kinds.** A prefix stack's reference is the same plan
with reuse dropped — same chunked prefill, same tokens — never a strict-plan stack. `E9`
measured the two plans up to `4.31` logits apart.

### 4B — `CONFIRMED`

Admitted `A`, `A_t`, `B`, `B_t`, `E`. `C` never ran: this model does not admit the paired
path. `D` fell to the source's own exclusion, `D_single` to the kernel's activation hold.

Both phases: `12` blocks, tokens, counts and stop reasons identical within plan kind, zero
fallbacks, zero swapouts, free memory never below `70%`, peak RSS `3.77 GB`, no block
beyond the `25%` drift limit, swap unchanged.

| class | reference | stack | median | 95% CI |
| :-- | :-- | :-- | --: | :-- |
| `session_warm` | cold prefix | `E` | `0.4512` | `[0.4497; 0.4532]` |
| `pair_session` | cold prefix | `E` | `0.4307` | `[0.4297; 0.4342]` |

The other six classes keep their reference; every `E` interval there contains `1.0`.

**Read the winner with its caveat.** Selection named `B` and `B_t`; confirmation named `E`.
`B` and `E` overlap almost completely in both classes and the adoption rule takes the lower
median, which is `E` by a margin inside the noise. What is adopted is
`tuned_knobs + prefix_cache + objective_router`, and what is *shown* is that prefix reuse is
the whole effect.

**Attempt kept.** A concurrent session ran a confirmation on the same model at `13:48`
(`…_4b_confirmation.json`, `CONFIRMED`, same gates clean) whose selection had refused to
start on a `4.17` load average. Its confirmation stands on its own; it has no selection
before it, so the complete pair is the one recorded here.

### 12B — `BLOCKED`

Unlocked by the `B61` profile, so `C` is admitted here for the first time. Selection was
clean on every gate — swapout delta `0`, free memory never below `57%`, peak RSS `8.40 GB`
(`24.5%` of installed against a `60%` ceiling), swap *falling* `10.90` to `10.16 GB`.

| class | reference | stack | selection | confirmation |
| :-- | :-- | :-- | --: | --: |
| `pair_short` | `A_t` | `C` | `0.9366` `[0.9193; 0.9480]` | `0.9315` `[0.9174; 0.9436]` |
| `pair_staggered` | `A_t` | `C` | `0.9493` `[0.9409; 0.9601]` | `0.9448` `[0.9386; 0.9508]` |
| `pair_long` | `A_t` | `C` | `0.9691` `[0.9682; 0.9733]` | `0.9686` `[0.9595; 0.9718]` |
| `pair_session` | cold prefix | `C` | `0.3353` `[0.3340; 0.3376]` | `0.3342` `[0.3319; 0.3361]` |
| `session_warm` | cold prefix | `B`/`E` | `0.3528` | `0.3558` `[0.3543; 0.3579]` |

**The confirmation's verdict is `BLOCKED`, and it is not a measurement failure.** Tokens,
counts and stop reasons identical, zero fallbacks, no disturbed block, free memory `56-57%`
throughout, peak RSS constant at `8.40 GB`. One gate fired: the preregistered rule blocks on
*any* swapout, and the counter moved once, by `17642` pages between block `1` and block `2`,
then stayed flat for ten blocks. Swap *used* fell monotonically across the whole run,
`10.13` to `9.42 GB` — the machine was unwinding the swap the `B61` confirmation had built,
not starving this one.

`C` on `12B` is therefore **not qualified**. The numbers agree across two independent runs
and are recorded, but a blocked confirmation is not a confirmation. No threshold was moved
and the run was not repeated: repeating a run because its verdict was unwelcome is the
failure mode the gate exists to prevent, and whether one clean repetition is authorised is
not a decision this study takes for itself.

**Raw data (local, `.gitignore`d).** `B57_stack_composition_preregistration_20260910{,_v2}.json`;
`B57_stack_composition_20260910_{selection,confirmation}.json`;
`B57_stack_composition_20260910_4b_confirmation.json`;
`B57_stack_composition_20260910_12b_{selection,confirmation}.json`.

**Status.** 4B `CONFIRMED`. 12B `BLOCKED` on the resource gate. Valid for this machine,
these model revisions, this MLX and mlx-lm build and this workload only.

## B60 — The 12B autotuner did not run out of memory; one knob cannot survive a guarded child (2026-09-10)

**What was known.** The `12B` tune printed `confirming the screening winner with a paired
A/B ...` and produced nothing after it. No profile was written, and the run captured no
return code, no wait status and no stderr, so *how* it ended was never recorded. The
surviving stdout had been through a `tail`, which is why the first six screening decisions
and any traceback are missing from it — absence of a traceback there is not evidence that
none was raised. Frozen before anything new ran, in `B60_12b_abort_freeze_20260910.json`.

**The suspicion that was wrong.** The unified log for `13:25` to `13:31` shows the kernel
reaping an idle process about once a second, available memory falling to `5.96 GB` and the
compressor reaching `23.08 GB` on a `34.36 GB` machine. Real, severe, and **not the
mechanism**. No `JetsamEvent` report was written that day, on a machine that does write them.

**The cause, reproduced.** `Engine.__init__` takes the `wired_fraction > 0` branch and calls
`ironmule.hw.static_facts()` for the installed memory size. `static_facts()` is not cached
and reads every fact through `_sysctl()`, which shells out with `subprocess.run`. The Q3f
child guard is installed before any project import in a confirmation child and blocks
`subprocess.Popen`. The constructor therefore raises `GuardViolation` and the child **exits
with status 1** — not signalled, not killed, not out of memory.

Three arms through `ironmule.ab.run` unchanged, one child, one repeat, no warmup, four
tokens, on a `1B` model:

| arm | outcome |
| :-- | :-- |
| `wired_fraction=0.0` | child completed |
| `wired_fraction=0.6` | `ABRunError: child 0 exited with status 1` |
| `fuse_projections=True` | child completed |

The knob is the discriminator; nothing here depends on the model, the machine's memory or
its load.

**Then on the model that actually failed.** A lifecycle harness owning its own `Popen`,
budget fixed beforehand at eight tokens and one repeat, behind the same guard and through
the same `load_engine`:

| condition | end | duration | child peak RSS | stages reached |
| :-- | :-- | --: | --: | :-- |
| `A` = screening keeps `wired_fraction=0.6` | exit status `1`, `GuardViolation` | `13.1 s` | `7.41 GB` | `before_load` only |
| `B` = same, that knob off | exit status `0`, 8 tokens, deterministic | `16.7 s` | `8.23 GB` | through `after_close` |

Both clean on resources: zero swapouts, no swap growth, free memory never below `46%`.
Condition `B` reported MLX active `7.19 GB`, peak `8.53 GB`, cache `5.24 GB` from inside the
child.

**What this explains.** The `12B` screening kept `wired_fraction=0.6`, `tune()` handed
exactly that candidate to `confirm()`, the first child died within seconds, `ab.run` raised
`ABRunError`, `tune()` does not catch it, and the process ended with a traceback on a stream
the run did not keep. `4B` never hit it because its screening rejected the knob.

**What is not claimed.** Nothing here says what caused the memory pressure in that window,
only that the confirmation did not need it to fail. The screening winner of the aborted run
remains unrecoverable and the aborted run is not relabelled.

**The defect is left standing.** `wired_fraction` cannot survive a confirmation on any
machine, so it can never enter a profile. The repair is `hw._sysctl` reading through
`sysctlbyname` instead of a subprocess, exactly as `swap_used_bytes` already does since
`B55`. Not applied: it is a change to shipped code and its own decision. Open as `B62`.

**Raw data (local, `.gitignore`d).** `B60_12b_abort_freeze_20260910.json`,
`B60_wired_fraction_guard_conflict_20260910.json`,
`B60_12b_child_diagnosis_preregistration_20260910{,_v2}.json`,
`B60_12b_child_diagnosis_20260910_condition{A,B}.json`. Harness
`tools/b60_12b_child_diagnosis.py`, test `tests/test_b60_diagnosis.py`.

**Status.** SOLVED. `B60` closes.

## B61 — A 12B profile, screened without the knob that cannot be confirmed (2026-09-10)

**Design.** `ironmule.tune.SEARCH` loses exactly one entry, `wired_fraction`, in the tuning
process only. Every other entry keeps its order and values, `ironmule/tune.py` is not edited
and the shipped default is unchanged. `tune()` itself runs unchanged: coordinate descent
with token identity at every step, then the standard paired confirmation over six fresh
processes and seven repeats, and a profile stored only if that confirmation is accepted.
The parent owns the tune process this time, so the return code, the decoded wait status and
both streams are captured whatever happens.

**Result: `PROFILE_CONFIRMED`.** `679.3 s`, exit status `0`.

| | |
| :-- | :-- |
| gain | `7.97%` end to end |
| `total_ns` | median `0.9203`, CI `[0.9095; 0.9285]` |
| `prefill_ns` | median `0.9204`, CI `[0.9161; 0.9223]` |
| `decode_ns` | median `0.9207`, CI `[0.8998; 0.9499]` |
| identity | tokens, counts and stop reasons identical, deterministic |
| knobs | `compiled_fixed_cache`, `head_skip_prefill`, `readback_every=2` |

**`fuse_projections` is rejected here and was kept by the aborted screening.** That is
coordinate descent's path dependence, not a contradiction: with `wired_fraction=0.6` the
running best stood at `0.9034` and `0.8978` improved it; without it the running best is
`0.9218` and `0.9343` does not. The aborted run's screening winner is now permanently
unrecoverable, and no longer matters.

**What the confirmation cost.** Free memory fell to `19%`, the swapout counter grew by
`1 200 416` pages and swap in use went from `9.00` to `24.06 GB`. No process was ended, the
protective floor at `8%` free was never reached, child peak RSS `10.82 GB`. Two resident
`12B` models is what that costs, and it is the same pressure the aborted run showed — where
it was never the cause.

**Attempt kept.** Attempt 1 failed in `0.1 s` with `AttributeError: 'function' object has no
attribute 'SEARCH'`: `ironmule.tune` as an attribute is the re-exported function, not the
module, so the search was never patched. The captured return code and stderr said so
immediately, which is the whole point of owning the process.

**Raw data (local, `.gitignore`d).**
`B61_12b_tune_without_wired_preregistration_20260910{,_v2}.json`,
`B61_12b_tune_without_wired_20260910.json`, `…_attempt2.json`. Harness
`tools/b61_12b_tune_without_wired.py`.

**Status.** MEASURED, `GO`. A confirmed `12B` profile now exists for fingerprint
`dc652d66f24ac207`, which is what admitted `C` in `B57`.

## B63 — One authorised repetition, blocked by the same gate (2026-09-10)

**What was authorised.** Exactly one replacement confirmation of `12B` stack `C`, on
measurement code proven byte-identical to the blocked run — combined digest
`49212d837b055bc5` in both records, no source file differing. It replaces nothing
evidentially: both sessions are kept, neither is relabelled, and results are never pooled.

**One thing was added, and it was a start condition, not a gate.** The blocked run had
started while this machine was still unwinding the swap `B61`'s two-resident-model
confirmation had built. So the session was preregistered not to begin until the `vm_stat`
Swapouts counter read the same value at every 30-second sample across a complete
900-second window, with `gpu_busy` empty and the 1-minute load below `3.0` at the end.
Every threshold inside the harness stayed exactly where it was.

**The machine broke one window on its own.** With nothing measuring, the counter jumped
`19704` pages at `15:22` and the observation restarted. It settled afterwards and the run
began at `15:49` on a window that had held `1635` seconds, nearly twice the requirement, at
load `2.60`. Nothing was purged, restarted, signalled or tuned.

**Result: `BLOCKED`, on the same gate.** Nine blocks flat, one burst of `28984` pages
between block `8` and block `9`, two blocks flat. Swap *in use* fell across the whole run,
`8.33` to `7.11 GB`, the same signature as the first blocked session.

Everything else passed: tokens, counts and stop reasons identical within plan kind, zero
fallbacks, no disturbed block, free memory `37-46%`, MLX peak `8.85 GB`.

**The four classes, side by side and not pooled.** Neither session is a confirmation, and
two blocked runs do not add up to one.

| class | reference | `B63` session | earlier blocked session |
| :-- | :-- | --: | --: |
| `pair_short` | `A_t` | `0.9397` `[0.9211; 0.9669]` | `0.9315` `[0.9174; 0.9436]` |
| `pair_staggered` | `A_t` | `0.9437` `[0.9337; 0.9614]` | `0.9448` `[0.9386; 0.9508]` |
| `pair_long` | `A_t` | `0.9705` `[0.9651; 0.9773]` | `0.9686` `[0.9595; 0.9718]` |
| `pair_session` | cold prefix | `0.3316` `[0.3250; 0.3423]` | `0.3342` `[0.3319; 0.3361]` |

`pair_session` is the whole executed stack against the same plan with reuse dropped. It is
not a paired-path gain and is not read as one.

**What stands, and what does not.** Across one clean selection and two blocked
confirmations, `C`'s interval lies entirely below `1.0` in all four throughput classes and
correctness is identical every time. That is consistent, and it is still not a
qualification: the adoption rule requires the resource gates as well, and they did not
pass. No threshold was moved to make them pass, and by the preregistered stopping rule
there is no third session.

**`C` on `12B`: NOT QUALIFIED.**

**Open, deliberately.** The gate blocks on *any* swapout, on a machine that holds a
multi-gigabyte swap file and reorganises it on its own schedule — while swap in use was
*falling* through both runs. Whether "any swapout" is the right instrument on this machine
is a question about the gate, not about `C`. Changing it is a threshold change and is not
done here.

**Raw data (local, `.gitignore`d).**
`B63_12b_stack_c_replacement_preregistration_20260910.json`,
`B63_12b_stack_c_replacement_20260910.json` (outcome plus the full pre-start observation
trace), `B57_stack_composition_20260910_12b_confirmation_b63.json`.

**Status.** MEASURED, `BLOCKED`. `B63` closes.

## B62 — The wired-limit path reads the machine natively (2026-09-10)

**The defect, as `B60` proved it.** `Engine.__init__` read the installed memory size through
`hw.static_facts()`, which is uncached and reaches every fact via `_sysctl()`, a
`subprocess.run`. The Q3f child guard blocks `subprocess.Popen` in a confirmation child, so
any `Engine` built with `wired_fraction > 0` raised `GuardViolation` and the child exited
with status `1`, on every model and every machine. A knob that sits in
`ironmule.tune.SEARCH` and can never survive the confirmation that follows it is a trap, not
a knob.

**What changed, and what deliberately did not.** `hw.installed_memory_bytes()` reads
`hw.memsize` through `sysctlbyname` on the `ctypes`/libc path that `swap_used_bytes` has
used for `vm.swapusage` since `B55`. The single caller in the `wired_fraction` branch uses
it. `static_facts()` is **not** rebuilt: it runs in the parent, where a subprocess costs
nothing that matters, and every other caller keeps it. Two source files, one new function,
one changed call site.

| check | result |
| :-- | :-- |
| `sysctlbyname` vs `sysctl -n hw.memsize` | equal, `34359738368` |
| `static_facts()["memory_bytes"]` | equal to both |
| `wired_fraction=0.6` in a guarded child | completes, zero guard events (`ABRunError` before) |
| `wired_fraction=0.0` | branch not taken, limit untouched |
| `fuse_projections=True` control | unchanged, completes |
| subprocess started by the branch | none, asserted by wrapping `Popen` |
| hardware fingerprint | `dc652d66f24ac207`, unchanged |
| stored `4B` and `12B` profiles | load, still accepted, `14.71%` and `7.97%` |

**Fail-closed, and it is stricter than before.** An unavailable size now raises and leaves
the process-global wired limit untouched. Before the change a failed read produced
`int(None or 0) * fraction = 0` and silently applied a zero wired limit — a wrong limit
presented as a working one.

**The meaning of the knob is unchanged**: a fraction of installed physical memory, asserted
at `0.25`, `0.6` and `1.0`.

**Tests.** Seven targeted tests. Four wired-limit tests in `tests/engine/test_r6_r7.py`
stubbed `hw.static_facts` to fake the size and now stub `hw.installed_memory_bytes`; no
assertion was weakened and the expected applied limits are identical. Full suite: two
failures, both the pre-existing parallel-execution flakes in the process-cleanup gates
(`test_q3f_real_cleanup_keeps_external_process_alive`,
`test_real_macos_process_identity_and_cleanup_reap`). Both pass sequentially, neither file
references the wired path, and the `B55` entry already records the flake as machine process
noise.

**Nothing is rehabilitated.** `wired_fraction` is not put back into any profile or tuner.
Both stored profiles still carry `0.0` and neither was retuned. Whether the knob pays, and
what it does to memory and the wired limit under a confirmation, needs its own
preregistered study — the fix removes the trap, it does not answer the question.

**Raw data (local, `.gitignore`d).** `B62_native_memsize_20260910.json`. Tests
`tests/test_b62_wired_fraction_native_memsize.py`.

**Status.** `FIX_CONFIRMED`. `B62` closes.

## B65 — The gate was measuring the machine, not the run (2026-09-10)

**The question.** Two `12B` stack `C` confirmations were blocked by one criterion and
nothing else: `swapout_counter_delta` must be `0`. Whether that criterion describes memory
pressure on macOS, or describes a machine that holds a swap file, is a question about the
gate. It was asked without re-measuring anything.

**The decisive evidence was already recorded.** During `B63`'s pre-start observation window
— nothing measuring, no model loaded, no GPU work — the system-wide swapout counter moved
by `19704` pages. The old gate would have blocked a window in which nothing ran. A criterion
that fires on an empty machine is not measuring the run.

**Why it does that.** `vm_stat`'s `Swapouts` is system-wide and monotone. It counts every
page this machine wrote to swap for any reason, compacting a swap file it already holds
included. Apple does not define memory pressure that way, and its own answer —
`kern.memorystatus_vm_pressure_level` — is readable without starting a process.

**Native probes.** `host_statistics64` with `HOST_VM_INFO64` gives every figure `vm_stat`
prints, and `sysctlbyname` gives the pressure level. Both now live in `ironmule/hw.py`
beside `swap_used_bytes`. Verified by sandwiching the shell reading between two native ones:
a monotone counter moves between reads, so equality is the wrong test — a first attempt
asserted it and failed on `pageins` by exactly `1`.

**The candidate gate, written before any trace was scored.** Block if macOS's pressure level
is anything but normal at any probe or cannot be read; if free memory falls below `10%`; if
peak RSS exceeds `60%` of installed memory; if swap in use ever exceeds its value at run
start; if any block deviates more than `25%` from the median; on any fallback or correctness
difference; or on any missing probe. Three of those thresholds are unchanged from the
shipped gate, one is Apple's own semantics, and the swap criterion is a **zero-growth
budget** — chosen so that no number has to be picked and no burst size can be tuned around.
The swapout counter stays in the record as evidence and stops being a verdict.

| class | runs | old gate | candidate |
| :-- | --: | :-- | :-- |
| quiet | `6` | `0` blocked | `0` blocked |
| starved | `1` | blocked | blocked |
| idle control, nothing measuring | `1` | **blocked** | not blocked |

The starved run is `B61`'s tune confirmation with two resident `12B` models: swap in use rose
`15.56 GB` and free memory fell to `19%`. A sweep of every other resource trace in
`research/raw` scored three more, all in agreement.

**The two disputed runs were excluded from the evidence by construction.** `C`'s own blocked
sessions are scored and labelled `disputed`, and neither the verdict nor any threshold was
derived from them.

**Residual risk, stated plainly.** Sensitivity rests on one run with independent evidence of
starvation. The candidate is strictly at least as strict as the old gate on every criterion
that describes the run itself; the only criterion it drops is the one shown to fire on an
empty machine.

**Twelve tests** exercise the gate's arithmetic against the harness's own
`evaluate_resource_gate`, not a copy of it, including the assertion that the legacy gate
still blocks unchanged.

**Raw data (local, `.gitignore`d).** `B65_gate_semantics_preregistration_20260910.json`,
`B65_gate_semantics_20260910{,_with_sweep,_final}.json`. Harness
`tools/b65_gate_semantics.py`, tests `tests/test_b65_pressure_gate.py`.

**Status.** `GATE_CONFIRMED`. The gate is selectable in the harness and `legacy` remains the
default, so an existing command reproduces exactly what it did before. `B65` closes.

## B67 — 12B stack C, confirmed (2026-09-10)

**What this is.** The first confirmation of `12B` stack `C` under the gate `B65` qualified.
It is not a repetition of `B63` and does not stand in for it: `B63`'s two sessions stay
`BLOCKED`, and nothing here is pooled with them. Exactly one confirmation was authorised.

**One thing changed.** The resource gate. Workload classes, prompts, stacks, references,
block count, arm rotation, statistic, equivalence margin, adoption rule, correctness rule,
drift limit, free-memory floor, RSS ceiling, load ceiling and the concurrent-process check
are identical.

**Result: `CONFIRMED`.** Twelve blocks, tokens, counts and stop reasons identical within
plan kind, zero fallbacks, no disturbed block, free memory never below `40%`, peak RSS
`5.98 GB`, macOS pressure level `normal` at every probe, swap in use `6.94` to `6.81 GB`.

**And it did not need the new gate.** The swapout counter delta was `0`. This confirmation
would have passed the old gate too, which is worth saying: the result does not depend on the
change that made it possible to ask for it.

| class | reference | stack | median | 95% CI |
| :-- | :-- | :-- | --: | :-- |
| `pair_short` | `A_t` | `C` | `0.9302` | `[0.9139; 0.9449]` |
| `pair_staggered` | `A_t` | `C` | `0.9457` | `[0.9319; 0.9521]` |
| `pair_long` | `A_t` | `C` | `0.9680` | `[0.9655; 0.9703]` |
| `pair_session` | cold prefix | `C` | `0.3360` | `[0.3320; 0.3411]` |
| `session_warm` | cold prefix | `B` | `0.3553` | `[0.3539; 0.3582]` |

The four throughput classes are separate results and are never summed. `pair_session` is the
whole executed stack against the same plan with reuse dropped; it is not a paired-path gain,
and the `0.336` and the `0.930` describe different references and cannot be combined.
`single_short`, `single_long` and `session_new` keep their reference.

**Raw data (local, `.gitignore`d).**
`B67_12b_stack_c_under_qualified_gate_preregistration_20260910.json`,
`B57_stack_composition_20260910_12b_confirmation_b67.json`,
`COMPOSITION_PROFILE_STATE_20260910.json`.

**Status.** MEASURED, `CONFIRMED`. Nothing is activated: no product profile is written and no
default changes. Valid for this machine, this model revision, this MLX and mlx-lm build and
this workload only.

## B66 — Silicon characterisation: what this M1 Max actually charges for (2026-09-10)

**Method.** The shape inventory decided what was worth touching before any axis was opened:
every distinct quantised matmul each model runs, counted, and timed at the widths the
runtime really uses. Two MLP shapes carry `65%` of measured decode time on `4B` and `69%` on
`12B`, and both already run within a few per cent of the bandwidth `E4`'s size curve predicts
for their size. Everything else on either model is below `10%`.

Each axis is a paired, order-rotated block design **with an A/A arm**, because the
inventory's own numbers moved by more than `20%` between two runs of the same shape. The
A/A arm is the same work under a different name and it sets the floor a candidate has to
clear.

### The three strongest patterns

**1. Output rows per simdgroup, not threads per threadgroup.** On `K=3840, N=15360` — the
shape carrying `48%` of `12B` decode — geometry `(4, 8)` beats `mx.quantized_matmul` by `8`
to `10%`, byte identical, in two independent runs (`0.9179` `[0.8893; 0.9452]` and `0.9011`
`[0.8609; 0.9250]`) against an A/A arm at `1.0109`. At fixed rows, moving from `32` to `256`
threads per threadgroup changes almost nothing; moving from `2` to `8` rows per simdgroup is
worth about twenty percentage points. All twelve geometries were byte identical to the
library; `values_per_thread` was held fixed because changing it changes the order the partial
sums are added in, which is a different execution plan and not a geometry change.

**2. A submission boundary costs `230` to `270 us`.** One kernel behind its own `eval`
reaches `52 GB/s` on `4B` and `88 GB/s` on `12B`; sixteen behind one `eval` reach `235` and
`258 GB/s`. That is about forty times the `6.41 us` per dispatch `E5` measured, and it is why
grouping, prefix reuse and the paired path all pay: each one puts more work between two
synchronisation points.

**3. The grouped-width boundary is `16`, not `4`.** `M=8` costs what `M=16` costs in total,
so every width from `5` to `16` pays the sixteen-row price, and `4` to `16` halves the cost
per row. `E2` and `E3` saw this on synthetic shapes; this is the first measurement on the
shapes these models run.

| cost per row vs `M=4` | `M=1` | `M=2` | `M=8` | `M=16` |
| :-- | --: | --: | --: | --: |
| `12B` | `1.382` | `1.115` | `0.966` | `0.486` |
| `4B` | `1.832` | `1.248` | `1.003` | `0.505` |

### Per axis

| axis | outcome |
| :-- | :-- |
| shape inventory and cost | `CHARACTERIZATION_GAIN` |
| grouped width | `CHARACTERIZATION_GAIN`, no confirmed class to prove it in |
| threadgroup geometry | `CHARACTERIZATION_GAIN`, stack proof `NOT_STARTED` |
| cache residency | `CHARACTERIZATION_GAIN`, `4` to `5%`, not a knob |
| eval round trip | `CHARACTERIZATION_GAIN`, explains an existing knob |
| command queues | `NO_USEFUL_GAIN` |
| CPU QoS and ANE | not investigated; nothing pointed at either |

Two MLX GPU streams are slower than one on both models (`1.1279` and `1.2316`, A/A at
parity), which is `B56b` again inside one process: the device shares itself and the extra
synchronisation is a cost.

**A confounded arm, kept.** The first cache axis ran one kernel behind its own `eval` against
twelve behind one, so it measured a submission boundary and reported the smaller working set
as two to five times *slower*. It is kept in the record and replaced, not corrected in place
— and the confound became pattern 2.

**A tripwire that did its job.** Parameterising the geometry inside `ironmule/qmv_k3840.py`,
with defaults unchanged, tripped
`tests/test_qmv_k3840_integration.py::test_the_kernel_source_is_the_studied_one`, which
requires the shipped module to contain the `B42`/`B43` source verbatim. The change was
reverted in full. The variant is built and installed by the harness only; the shipped kernel
keeps its qualified source, its default geometry and its activation hold.

### The stack proof, and why there is no profile parameter

`(4, 8)` reached the one gate that matters: a confirmed complete stack of its own workload
class. It never got through it.

The planned proof is `NOT_STARTED`. Its readiness condition — a 1-minute load below `3.0` on
three consecutive samples — was never met across two attempts and `196` samples, of which
exactly one was below `3.0`, the lowest `2.82`. No threshold was moved to start it.

A separate exploratory run was then authorised explicitly, without the readiness condition
and with every runtime gate in force. It returned `EXPLORATORY_INVALID`: the `B65` resource
gate blocked it because swap in use rose `3.40 GB` during the first child, from `8.67` to
`12.08 GB`, and stayed there. macOS's own pressure level read normal at every probe and free
memory rose, so it is the swap criterion that fired.

That run's three class ratios all sat entirely below `1.0` — `0.9301`, `0.9306`, `0.9398` —
with correctness identical in all six processes. **They are not a result.** A number from a
run whose gate failed is not evidence, and treating it as one is the failure mode the gate
exists to prevent. `B65` qualified that gate hours earlier on independent evidence, and it
has now blocked this project's own candidate. That is the gate working.

`silicon_profile.v1` therefore carries **no confirmed parameter**. Its candidate list carries
`(4, 8)` with its shape, model, workload classes, correctness contract, isolated effect and
the full history of both failed proofs.

**Paired × geometry: `NOT_RUN`**, gated on `(4, 8)` being confirmed. Starting it would be
composing on a result that does not exist.

**`M=16` multi-token verification: `NO_GO`**, decided on arithmetic over existing
measurements with no model run. Break-even needs `74%` per-token acceptance at width `2` and
`86%` at width `16`; the n-gram drafter this runtime has delivers `17%`, giving `1.20`
accepted tokens per forward against `1.74` to `6.48` required — short by `1.49` to `5.38`
times. The step at `16` is real and does help: width `16` needs less acceptance than width
`8` because the two cost almost the same. It still lands far short. The KV cost is
`393 KB` per speculative token and is not what fails, and exact greedy verification is
plausible and is not what fails either. Earlier speculation `NO-GO`s stand unchanged.

**Raw data (local, `.gitignore`d).** `B66_shape_inventory_20260910.json`;
`B66_axes_{12b,4b}_20260910.json`; `B66_axes_{12b,4b}_cache_eval_20260910.json`;
`B66_stack_proof_preregistration_20260910.json`; `B66_stack_proof_20260910_not_started.json`;
`B66_FORCED_LOAD_EXPLORATORY_preregistration_20260910.json` and `…_20260910.json`;
`B66_m16_multitoken_arithmetic_20260910.json`;
`B66_silicon_profile_20260910{,_v2}.json`. Harnesses
`tools/b66_silicon_characterisation.py`, `tools/b66_axes.py`, `tools/b66_stack_proof.py`.

**Status.** MEASURED. Characterisation complete, no parameter qualified, nothing activated.

## B68 — MLX's command buffer limits, closed on this fingerprint (2026-09-10)

**Question.** `B66` measured a submission boundary at `230` to `270 us` and every confirmed
win this project holds works by putting more work between two of them. MLX 0.32.0 has its
own boundary nothing here had touched: `CommandEncoder::needs_commit()` commits when either
`max_ops_per_buffer` or `max_mb_per_buffer` is exceeded, both read once per process from
`MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER`. Verified in the installed headers, not
assumed; the values are not exposed to Python, which is why the limit was observed rather
than asserted.

**A commit is not a synchronize**, and the measurement was built so that no buffer count
could be mistaken for a speed claim. None is reported.

**The microbenchmark said no first.** A chain of identical tiny kernels behind one `eval`
shows **no step at 50 operations** under the default: a forced commit costs nothing visible
on its own. And raising the limits *costs* — at `250` operations the default takes
`1584.3 us` against `1998.5 us` at `400/400`, `26%` slower. The plausible reading is that
committing early lets the GPU start while the CPU keeps encoding, and a larger buffer removes
that overlap. That is a reading, not a proof: nothing here observed the overlap. The chain
works on `262 KB` per step, so only the operations axis was probed.

**The real sweep agreed.** Four rounds, all six configurations per round in rotated order,
a fresh process each with the environment set before the first MLX access, both models, the
workload classes from the confirmed composition profile. `48` children, zero failures, zero
command buffer errors, zero token or stop-reason differences, resource gate passed.

**Not one arm on either model in either class has an interval below `1.0`.** Three are
outright losses:

| model / class | configuration | median | 95% CI |
| :-- | :-- | --: | :-- |
| `4B` `session_warm` | `ops_mb_400` | `1.0257` | `[1.0124; 1.0383]` |
| `4B` `single_short` | `ops_mb_400` | `1.0162` | `[1.0063; 1.1132]` |
| `4B` `session_warm` | `ops_default_mb_400` | `1.0117` | `[1.0006; 1.0575]` |

The best medians anywhere — `0.9904` and `0.9930` on `4B`, `0.9958` and `0.9978` on `12B` —
all carry intervals containing `1.0`.

**No confirmation session was run.** The design admits only the best plausible candidate and
there is none; running a confirmation on an arm whose selection interval contains `1.0` is
looking for significance rather than testing for it.

**The axes cannot be separated for a gain, because there is no gain to attribute.** The two
single-axis arms sit at parity or worse everywhere. Nothing here claims which limit fired.

**`4B`: `DEFAULT_WINS`. `12B`: `DEFAULT_WINS`.** MLX `50/50` stays the reference on this
hardware, this MLX build and these models, and the command buffer axis closes. The negative
evidence is the result. The project rule stands: a closed entry is evidence, not a ban, and a
new mechanism or new hardware evidence may reopen it by naming this one.

**Nothing enters `silicon_profile`.** `(4, 8)` remains a candidate without a stack proof,
paired × geometry remains `NOT_RUN`, two command queues and `M=16` with the n-gram drafter
remain `NO_GO`.

**Raw data (local, `.gitignore`d).**
`B68_command_buffer_limits_preregistration_20260910.json`,
`B68_command_buffer_probe_20260910.json`, `B68_command_buffer_sweep_20260910.json`,
`B68_command_buffer_outcome_20260910.json`. Harness `tools/b68_command_buffers.py`.

**Status.** MEASURED, `DEFAULT_WINS` on both models. `B68` closes.

## B69 — The `(4, 8)` geometry, confirmed against the stack (2026-09-10)

**What was outstanding.** `B66` measured the geometry beating `mx.quantized_matmul` by `8` to
`10%` on the shape carrying `48%` of `12B` decode, byte identical, twice, against an A/A arm
at parity. Whether that reached a stack was unknown in both directions: the planned proof
never started, and the authorised exploratory run was blocked when swap in use rose `3.40 GB`
during its first child.

**The memory problem was removed at its cause, not tolerated.** That growth came from each
child loading the `12B` twice, once per arm, so two model images passed through one process.
Here each child loads one model, runs one arm and exits. A block is three such children: the
reference, the candidate, and the reference again under another name as the A/A control.
Six blocks, arm order rotated. Model load and warmup are measured and reported separately and
are not inside the stack time.

**On the readiness condition, plainly.** The earlier proof waited for a `1`-minute load below
`3.0`, a threshold chosen for that launcher and never met across two attempts and `196`
samples. This run used the harness's own long-standing gate, load at or below `4.0`, and
preregistered that choice with its reason: the AB/BA rotation, the A/A arm and the `25%`
drift gate are the instruments that decide whether a moderately busy machine produced a
usable measurement. It started at load `3.89`.

**Result: `STACK_CONFIRMED` in all three classes.** Six complete blocks, zero child failures,
zero token or stop-reason differences, zero fallbacks, no disturbed block, resource gate
passed with **zero swap growth**, free memory never below `53.2%`, peak child RSS `8.40 GB`.

| class | reference stack | candidate | 95% CI | A/A control |
| :-- | :-- | --: | :-- | :-- |
| `single_short` | `A` | `0.8469` | `[0.7580; 0.9488]` | `1.0054` `[0.8880; 1.0576]` |
| `single_long` | `A` | `0.8806` | `[0.7369; 0.9554]` | `1.0038` `[0.9501; 1.0229]` |
| `session_warm` | `B` | `0.9020` | `[0.7934; 0.9305]` | `1.0010` `[0.9840; 1.0376]` |

**It survives losing any block.** A leave-one-block-out check, post hoc and not part of the
verdict, keeps every interval entirely below `1.0` in every class; the worst case is
`0.9661`. Block `1` is an outlier in all three classes and in the A/A arm too, which points
at that block's reference child having been disturbed rather than at the candidate.

**The gain is bigger than the kernel measurement predicted, and that is not explained here.**
`8` to `10%` against the library on one shape arrived as `10` to `15%` of complete stack
time. The isolated benchmark ran four distinct weight buffers, `132 MB`, on one shape; the
stack runs `241` projections over four shapes out of `6.6 GB` of weights, so the two sit in
different bandwidth and cache regimes. A number arriving larger than predicted is a reason to
look, not a reason to celebrate.

**Correctness came first, every time.** All `241` admitted projections were compared byte for
byte against `mx.quantized_matmul` on their own buffers before a single token was timed, in
every candidate child of every block. A projection that differed would have been left on the
library path and recorded rather than timed; none did.

**Nothing shipped changed and nothing is activated.** `ironmule/qmv_k3840.py` keeps its
qualified source, its default geometry and its activation hold; the variant is built and
installed by the harness. The parameter enters `silicon_profile.v1` as evidence bound to this
fingerprint, this MLX and mlx-lm build, this model revision, this shape, these three classes
and this code digest. The four `12B` throughput classes are outside it entirely: their
confirmed stack is `C`, and the source keeps this kernel apart from the paired path.

**Raw data (local, `.gitignore`d).** `B69_stack_proof_preregistration_20260910.json`,
`B69_stack_proof_20260910.json`, `B66_silicon_profile_20260910_v3.json`. Harness
`tools/b69_stack_proof.py`, which builds the variant through `tools/b66_stack_proof.py` so
the geometry under test is the source `B66` measured.

**Status.** MEASURED, `STACK_CONFIRMED`. `B69` closes.

## B70 — A silicon profile the router can read and must not obey (2026-09-10)

**What shipped.** `ironmule/silicon_profile.py`: a fail-closed loader, a pure matcher, and
the empty data contract `B71` will fill. The router may load a profile, match it and report
what it found. It may not route differently because of it.

**The loader refuses rather than guesses.** Exact schema, exact field sets at both levels, no
silent defaults, no coercion, a digest that must recompute over the parameters as written,
unique ids, and a refusal to load any profile whose `activation` claims anything but `none`.
A profile that cannot be fully understood is not partially believed.

**The matcher is pure and cheap.** No subprocess, no model hash, no file read, no GPU call,
no network. Fifteen conditions, short-circuited most-selective-first, and the first failure
is named. A dispatch is given a class name by a fixed rule over facts the router already has,
and a dispatch that fits no named class is unnamed and matches nothing — which is the correct
answer rather than a nearest guess.

**Where the diagnostic runs, and why it moved.** `B55` is the standard: a wrapper that added
nothing to the work still made a routed request `3.7%` slower, because a telemetry field
shelled out. The same discipline caught two implementations here.

| implementation | worst ratio against `decide()` | gate `≤ 1.02` |
| :-- | --: | :-- |
| eager inside `decide()`, copying the finished decision | `2.4815` | fail |
| the same, optimised: hand-built dict, short-circuit, constructed context | `2.0010` | fail |
| once per dispatch, in the record builder | `1.0069` | pass |

Both failures are kept. A `4.3 us` decision cannot absorb a diagnostic that allocates
anything, so the diagnostic left the decision. **The final `1.00` is by construction, not an
achievement**: arm B's `decide()` *is* arm A's. The number that means something is the
annotation itself, `3.2` to `7.3 us` once per dispatch, against a `12B` `single_short`
dispatch that `B69` measured at roughly `400 ms`.

That move is also what makes the shadow claim structural instead of a promise: `decide()`
cannot read what it does not compute.

**Matching, as measured.** `single_short` and `session_warm` match their own parameters; a
two- or four-request throughput dispatch is unnamed and matches nothing, which is right —
the confirmed stack there is `C`, and the source keeps the kernel apart from the paired path.
Every mismatch in fingerprint, GPU architecture, library version, model identity, revision,
quantisation, `K`, decode width, `N`, class, objective or evidence status refuses.

**An error this found in its own data.** The first strict profile carried the `4B` model
revision on a `12B` parameter, because the value was typed rather than read. The loader
accepted it — it validates the shape of a field, not its truth — and every match then refused
with `model revision does not match`. The effect was fail-closed, so a wrong profile could
only ever cause *no* match and never a wrong one. That is the design working, and it is still
a real limitation worth writing down. Both files are kept, the wrong one as it was written.

**Nothing is activated.** No `(4, 8)` dispatch, no kernel released, `ironmule/qmv_k3840.py`
untouched, no default changed, no product profile. Every loaded parameter carries
`activation: none` and the loader refuses any other value.

**Tests.** `44` in `tests/test_b70_silicon_profile.py`. Every negative case asserts the
router's decision is identical; the positive case asserts only the two diagnostic fields
differ. Full suite: two failures in the process-cleanup gates, and they fail with
`ironmule/router.py` and `ironmule/silicon_profile.py` removed from the tree entirely, which
is how that was established rather than assumed.

**Verdict: `B70_PASS`.**

**Raw data (local, `.gitignore`d).** `silicon_profile_v1_20260910.json` and
`…_corrected.json`; `B70_router_overhead_20260910{,_attempt2,_final}.json`;
`B70_silicon_profile_shadow_20260910.json`. Harness `tools/b70_router_overhead.py`.

**Status.** `B70` closes. Activation remains a separate decision with its own evidence.

## B71 — A machine describes itself, and says what it could not measure (2026-09-10)

**The decision, written before any probe ran.** A fingerprint is exact and predicts nothing:
it says whether a machine has been seen, never what a machine like it will do. The question
this vector exists for is whether a Mac nobody has tuned can estimate, in minutes, whether a
known optimisation is worth trying — concretely the `K = 3840` geometry `(4, 8)` that `B69`
confirmed here against the complete `12B` stack, which today costs a full tune plus a
six-block proof to answer.

**What was built.** `ironmule/characterization.py`: a versioned, strictly validated
`HardwareCharacterizationVector` split into static facts, measured responses and the
conditions they were measured under, with dimensionless relations derived from its own
measurements. No single combined score: a machine fast at one thing and slow at another is
exactly the case one number destroys. Every measured field defaults to `None` and the
serialised form carries an explicit `missing` list that must agree with the vector or it will
not load.

**The probe set runs on an untuned machine.** No model is loaded, no profile is read, nothing
needs to exist first. Six probes, every one a mechanism `E4`, `B66` or `B68` already
established, and no open search anywhere. Declared budget `420 s`; used `1.7 s`.

**Three runs, and the second and third were forced by the first.** `v1` used seven repeats
and no A/A arm, and reported a geometry ratio of `0.95` with a spread of `0.58`. `v2` raised
repeats to `25` and added an A/A arm, which measured a noise floor of `0.9885` with a
relative spread of `0.606` — and so made the candidate at `1.0024` unreadable. It also
exposed a unit error: a nanosecond half range recorded as the spread of a
bytes-per-nanosecond value. `v3` fixes the unit, keeps the design, and draws the consequence:
a geometry number is emitted only when its own A/A arm clears a threshold fixed before the
run. All three runs are kept.

| relation | `v1` | `v2` | `v3` | span | earlier independent measurement |
| :-- | --: | --: | --: | --: | :-- |
| `cache_to_dram_ratio` | `0.9447` | `0.9442` | `0.9466` | `0.3%` | `B66`: `0.9471` |
| `m8_cost_per_row_vs_m4` | `1.0428` | `1.0361` | `1.0246` | `1.8%` | `B66`: `0.966` |
| `m16_cost_per_row_vs_m1` | `0.3861` | `0.3908` | `0.3980` | `3.0%` | `B66`: `0.352` |
| `k_unaligned_to_aligned_ratio` | `1.0712` | `1.1531` | `1.2382` | `14.5%` | — |
| `eval_fixed_over_one_kernel` | `1.0435` | `1.8379` | `1.0413` | `76.3%` | `B66`: `2.38` |
| `geometry_4_8_ratio` | `0.9527` | `1.0024` | withheld | — | `B66`: `0.9011` |

**Three features carry information here.** Cache residency, and the two width relations:
stable across three runs to within `3%` and agreeing with an independent earlier measurement
of the same quantity taken under a different design.

**Two are not yet usable.** `eval_fixed_over_one_kernel` swung `1.04`, `1.84`, `1.04`, while
`v3`'s per-point increments are consistent at `113` to `126 us` — so the marginal cost is
solid and the two-endpoint fit for the fixed part is not. That needs a better estimator, not
a better machine. `k_unaligned_to_aligned_ratio` is directionally consistent, the unaligned
`K` always costing more, but not yet a number to compare machines with.

**The uncomfortable part.** The feature the primary decision most needs is the one the probe
set refused to emit. `B66` measured that same quantity at `0.9179` and `0.9011` against an
A/A arm at `1.0109`, using twelve blocks with rotated arm order; the quick probe uses one
sequential pass and cannot reject an outlier. So either the probe adopts a blocked design and
stops being a few seconds long, or a quick characterisation honestly cannot answer the
geometry question on a busy machine. Both are acceptable and neither is decided here.

**A correction to this entry's own analysis, made before it was reported.** The first outcome
record listed `geometry_4_8_ratio` as informative because its `v1` and `v2` values agreed to
`5.1%`. They agreed inside a noise band of `0.606`. A number that agrees with itself inside a
band that swamps it has not been measured twice, it has been guessed twice. The rule now
reads: a relation the vector withholds is never informative, however well its withheld values
agree. Both records are kept.

**Nothing is called redundant.** Two relations that move together on one machine could be
independent on another memory system, and a redundancy claim needs variation that one machine
cannot supply.

**`H1` is not tested and is not testable here.** A compact vector describing optimisation
response better than a chip name needs more than one chip; every dataset row carries identical
hardware features, so nothing can separate a hardware effect from a constant. What a second
Mac would settle is written into the preregistration: run this probe set and then `B69`'s
stack proof on a machine with a materially different memory system, and `H1` survives if the
minutes-long geometry response has the same sign as the hours-long stack result on both, with
the bandwidth and cache relations differing in the direction that predicts it. It is refuted
if similar vectors give opposite stack results, or similar stack results come from clearly
different vectors. Two machines settle a sign, not a magnitude.

**`B72`'s dataset exists and nothing is trained.** `49` rows of
`hardware_features, workload_features, action, measured_cost, uncertainty, evidence_id,
validity`: `6` from `B69`'s confirmed stack proof, `23` isolated kernel responses from
`B66`, `20` from `B68`'s closed command buffer axis. Five sources are excluded by name and
reason, every one `BLOCKED`, `EXPLORATORY_INVALID`, `NOT_STARTED` or confounded.

**Verdict: `B71_PASS`, at the ceiling one machine allows —
`VECTOR_IMPLEMENTED_AND_SELF_CONSISTENT`.** The router is untouched: `characterization.py` is
not imported by it. Nothing is activated.

**Raw data (local, `.gitignore`d).**
`B71_quick_characterization_preregistration_20260910{,_v2,_v3}.json`,
`B71_vector_m1max_20260910{,_v2,_v3}.json`,
`B71_self_characterization_20260910{,_corrected}.json`,
`B72_cost_dataset_20260910.json`. Harnesses `tools/b71_quick_characterization.py`,
`tools/b71_dataset.py`, tests `tests/test_b71_characterization.py`.

**Status.** `B71` closes.

## B72 — A cost model that abstains, and the data that leaves it no choice (2026-09-10)

**What it is.** `hardware + workload + action -> expected cost + uncertainty`, in shadow
only. Three models, all CPU-local and deterministic, none larger than `1.3 kB`: a per-context
mean lookup as the honest baseline, a closed-form ridge, and a depth-three regression tree
written out because scikit-learn is not installed. The router is not touched and nothing here
runs at dispatch time.

**The split is by study and the holdout was frozen first.** `B69` is the sealed holdout,
`B66`'s `4B` axes the validation, everything else train. Zero leakage. And the consequence is
a property of the evidence, not a design choice: the four studies measure **three different
quantities against three different references** — complete stack time against a confirmed
stack, complete stack time against MLX's default buffer limits, and isolated kernel time
against a library call or against width four. A study-level split is therefore also a
metric-level split.

**Verdict: `B72_DATA_INSUFFICIENT`.** Every model abstained on every sealed-holdout context.
Three reasons stack, and each is a fact about the data rather than about the models.

The holdout's action, `k3840_geometry_4_8`, appears in no training row, so there is nothing
to predict it from. The hardware block is **constant across all `49` rows** — one machine —
so nothing fitted here learned anything from hardware. And one action family compares actions
that do different amounts of work: the grouped-width rows report total time against width
four across `M = 1` to `16`, so picking the cheapest picks the smallest batch and means
nothing.

| model | holdout RMSE | 95% coverage | median half width | size | inference |
| :-- | --: | --: | --: | --: | --: |
| context mean lookup | `0.1116` | `1.00` | `0.2535` | `873 B` | `0.23 us` |
| regression tree, depth 3 | `0.0959` | `0.83` | `0.0930` | `634 B` | `0.53 us` |
| ridge | `0.0867` | `0.50` | `0.0193` | `1304 B` | `0.69 us` |

Ridge has the best holdout RMSE and the worst calibration: its intervals cover the truth half
the time when they claim `95%`. Better error and worse honesty, on a set of contexts where it
correctly refused to decide anyway. That combination is why RMSE was never the decision
metric here.

**The unknown-Mac test cannot be run on this data, in either direction.** The brief asks that
stripping the hardware features make the model widen its interval or abstain. It cannot: the
hardware block never varies in training, so nothing was learned from it and nothing can be
unlearned by removing it. Reporting that as passed would be the same error `B71` had to
correct — calling a number meaningful from inside its own noise band.

**Two defects this found, both mine, both recorded.** The first run reported the hardware
block as *varying* in training, because a standard deviation of `1e-16` across identical
floats passed a `> 0` test. And the first run scored one grouped-width context as a correct
top-1 choice, on a metric whose actions do unequal work; that family now abstains with that
reason stated. `B71`'s export should carry cost per row for it rather than total time.

**A process note worth keeping.** A patch to the ablation silently did not apply because its
search string had a typo, and the run that followed reported a passing ablation that had
never executed. It was caught by reading the record rather than the summary line. Every
string replacement into a measurement harness gets an assertion from here on.

**`B73`'s data contract, defined and not started.** Fast probes are the three `B71` measured
stable to within `3%` and in agreement with `B66`: cache residency and the two width
relations. Deep probes are the two it could not pin down: the geometry response, whose A/A
arm had a relative spread of `0.606`, and the eval fit, which swung `1.04`, `1.84`, `1.04`.
The rule is that a deep probe runs only when a fast pass abstained *and* the missing feature
is the one the abstention named. Never speculatively, never as a sweep.

**Raw data (local, `.gitignore`d).** `B72_cost_model_20260910{,_v2,_v3}.json`, all three
kept. Harness `tools/b72_cost_model.py`, tests `tests/test_b72_cost_model.py`.

**A correction to this entry's own verdict, forced by re-reading the criterion.** The
`PASS` condition reads *beats the trivial baseline meaningfully **or** abstains correctly*.
The verdict tree implemented only the first half, so a run in which every model refused —
correctly — fell through to `DATA_INSUFFICIENT`. That used a statement about the data as a
statement about the run.

Re-run on the **same dataset**, same models, same alpha, same tree depth, same split and same
thresholds, with both branches present: **`B72_PASS`**. All twelve abstentions were
re-derived from the split rather than trusted — an unseen action had to really be absent from
training, an incomparable metric had to really be one — and all twelve are grounded. Nothing
was re-measured; no model was loaded for this.

Both statements stand and they are about different things. The **pipeline** passes: holdout
cleanly separated, every refusal correct and grounded, no confident extrapolation, router
untouched, overhead measured, models deterministic and serialisable. The **evidence** is
still insufficient: every model refused because the holdout's action had no training
evidence, and the same record carries that as `data_insufficiency_also_holds`. A `PASS` here
is about the machinery and never about the evidence — reading it as *the cost model works*
is the error the abstain mechanism exists to prevent.

**Gate review, added after the fact and not to soften the verdict.** The seven `PASS`
conditions were never walked one by one, so they were, in
`B72_gate_review_20260910.json`. Against the **replay**, six are met and one is met partly:
uncertainty is usable for ridge, which puts `8` of `9` holdout rows inside its stated `95%`
interval and misses the ninth in the safe direction, and it is **not** usable for the tree,
which stated a half width of `0.0104` while choosing an action `0.1531` worse than the best.
Confidently wrong is the failure an interval exists to prevent.

Against the **original run**, the condition is not met and is not claimed to be. `B72` closed
`DATA_INSUFFICIENT`; the replay passed on data `B74` made rankable. Those are two runs and
neither replaces the other.

Worth stating because it could be read the other way: `B74` was written to make ranking
semantically valid, not to make a model win. It *removed* an action family from scoring,
which can only reduce what a model is credited with, and it *added* the reference as a
competing action, which gave a model a new way to be wrong. The tree promptly was.

**Status.** `B72` closes as `DATA_INSUFFICIENT`, which the entry named in advance as a valid
result. What is missing is not a better model. It is a second machine, and rows whose costs
are comparable across the actions they rank.

## B74 — Rows that may be ranked, and the replay they made possible (2026-09-10)

**What was broken.** `B72` could not score `B71`'s export for two reasons it found while
trying. The grouped-width rows report total time across `M = 1` to `16`, so ranking them
picks the smallest batch. And every study named its actions after itself, so a study-level
split left every held-out action unseen by construction.

**What `B74` changed, and only that.** No model, no hardware measurement, no route. Every row
gained a comparison context, the set it may compete in, its work units, a normalised cost and
the method that produced it — or an explicit refusal.

Three repairs did the work. **One intervention, one name**: `B66` timed the `(4, 8)` geometry
against a library call and `B69` timed the same intervention inside a stack, and two names
hid that. **The reference is an action**: the baseline each set was measured against now
appears at `1.0`, because a choice between alternatives to something nobody can pick is not
the decision anyone faces. And **the width family is never ranked**: cost per row is a valid
unit, and grouped width is still not a free action, because a dispatch cannot choose a width
larger than the number of ready requests. Those rows stay in the dataset as context.

**`B74_PASS`.** Eight comparable sets, three of them usable as top-1 tests outside training,
all eight quality checks green, ten rows excluded by name and reason, no study on both sides.

**The replay changed one thing in `B72`: the input path.** Same three models, same alpha,
same tree depth, same split rule, same abstain thresholds.

| model | decided | top-1 | mean regret | catastrophic | holdout RMSE | 95% coverage |
| :-- | --: | --: | --: | --: | --: | --: |
| context mean lookup | `0/3` | — | — | `0` | `0.0784` | `0.89` |
| regression tree, depth 3 | `1/3` | `0` | `0.1531` | **`1`** | `0.0781` | `0.89` |
| ridge | `3/3` | `3` | `0.0000` | `0` | `0.0203` | `0.89` |

**The caveat that matters more than the table.** A policy that needs no model at all — always
pick the one action that is not a reference — scores `3` of `3` on this holdout, exactly what
ridge scored. **Top-1 is therefore not evidence that a cost model was learned.** What ridge
adds beyond that policy is magnitude: holdout RMSE `0.0203` against the baseline's `0.0784`.
That is the part worth anything, and it rests on three contexts.

**One transfer happened and is not a rule.** Ridge predicted the stack cost of the `(4, 8)`
geometry from its isolated cost against a library call, and got the sign and roughly the
magnitude right. `E5` measured a case where exactly that transfer failed — fusion's `6%`
bandwidth prediction arrived as `0%` in decode. One instance where it worked is one instance.

**The tree failed outright.** It abstained on two contexts and on the third chose the
reference over the geometry: top-1 wrong, regret `0.1531`, a catastrophic mistake by the
`0.05` threshold fixed beforehand. Three models, one useful, one silent, one wrong.

**The ablation, read correctly.** Both models abstain with the hardware block removed, and
the fail-closed *policy* is what refused, not the models' uncertainty: ridge's interval moved
`0.0181` to `0.0191`. On one machine the hardware block is constant, so nothing was learned
from it and nothing can be unlearned by taking it away. The policy did the work; the model
could not have. That is the same limit `B72` reported and it has not moved.

**What this does not show.** Anything about hardware. `B74` shows the action-cost task is now
correctly defined. `B73` on a machine nobody has tuned remains the first cross-hardware
evidence, and nothing here brings it closer.

**Raw data (local, `.gitignore`d).** `B74_comparable_dataset_20260910{,_v2}.json`,
`…_v2_v1shape.json`, `B72_replay_after_b74_20260910.json`, `B74_outcome_20260910.json`.
Harness `tools/b74_comparable_dataset.py`, tests `tests/test_b74_comparable_dataset.py`.

**Status.** `B74_PASS`, and `B72` replays as `B72_PASS` on a holdout small enough that a
constant policy matches its top-1. `B74` closes.

## B73 — The cross-hardware test, and the gate that refused it (2026-09-10)

**`B73_NOT_STARTED`.** There is one machine, and its fingerprint `dc652d66f24ac207` is the
frozen model's training fingerprint. A prediction made on it would be a memory rather than a
forecast, and would look excellent for exactly that reason. The harness checks this first and
exits.

None of the four preregistered verdicts is claimed. `TRANSFER_SIGNAL`, `SAFE_ABSTAIN`,
`TRANSFER_FAIL` and `INVALID` all describe what happened during a run. Nothing ran.

**What was done instead, and it is the part that had to happen now.** The `B72` dataset and
the ridge state are **frozen before any unknown machine exists**, which is the only moment a
freeze is worth anything: coefficients, feature names, scaling, residual spread, policy
thresholds, known actions, known metrics, and the digest of the training design matrix.
Carried to a second Mac unchanged, with its digest checked on arrival, it cannot have been
chosen to suit what that machine turns out to be.

The harness is complete and gated in both directions. `predict` refuses to run on a training
machine. `verdict` refuses a prediction whose file is newer than the ground truth it is
compared against, so the sealing order is enforced by the filesystem rather than by
intention. Nine tests hold those gates.

**What the harness already knows it will say, written down before a second machine exists.**
On any unknown machine the prediction abstains, and the reason is structural rather than
cautious: no coefficient on a hardware feature was fitted against variation, because the
hardware block is constant across every training row. On unseen hardware those coefficients
multiply values the model has never seen move. So the expected verdict on a second Mac is
`B73_SAFE_ABSTAIN`, and reaching `B73_TRANSFER_SIGNAL` would need a third machine's worth of
variation to fit against, not a better policy. Recording that now means an abstention there
reads as the design working rather than as a disappointment.

**Preconditions for a real run.** A Mac whose fingerprint is not `dc652d66f24ac207`; MLX
`0.32.0` and mlx-lm `0.31.3`, or the qualification conditions restated for other builds; the
same Gemma `12B` revision `86cc6a8dedbc456dd0e4af01a9d09f396f77e558` in its cache; about
`11 GB` free for one resident model per process; and the frozen model carried across with its
digest verified.

**The order, which the files enforce.** Check, then `B71`'s probe set unchanged, then a
sealed prediction, then at most one deep probe for exactly the feature an abstention named
and exactly one second sealed prediction, then `B69`'s stack proof unchanged, then the
verdict, and only then any learning from the new machine — which never overwrites the sealed
prediction.

**`H1` is still untested**, and no bandit or reinforcement learning starts until a
cross-hardware signal is shown to exist.

**Raw data (local, `.gitignore`d).** `B73_frozen_model_20260910.json`,
`B73_eligibility_20260910.json`, `B73_not_started_20260910.json`. Harness
`tools/b73_cross_hardware.py`, tests `tests/test_b73_cross_hardware.py`.

**Status.** `NOT_STARTED`, waiting on hardware and on nothing else.

## B75 — A fresh install learns this machine, in under five minutes (2026-09-10)

**The question a new user actually faces.** Everything IronMule knows about this Mac was
earned over many studies. Starting with no profile, no known winners and no performance
history, can the system measure its way to a decision it is entitled to act on, and what does
that cost?

**The isolation is structural.** `tools/b75_cold_start.py` opens no `B66`, `B69`, `B72` or
`B74` record and no `silicon_profile`; a test greps the source for each name. It imports
`B69`'s child process as measuring machinery and none of its numbers. Static facts a fresh
install would also have — SoC, memory, GPU generation, library versions — are used. The
history stayed sealed until a separate tool opened it, after the decision was written.

**The learner started with the reference and nothing else**, and took three steps.

| step | cost | decision |
| :-- | --: | :-- |
| fast probes, `B71`'s set unchanged | `1.21 s` | `MEASURE_MORE`, target named |
| deep probe, blocked and rotated, A/A gated | `0.29 s` | `CANDIDATE` |
| local qualification, `4` blocks, one arm per process | `285.01 s` | `CANDIDATE` |

`time_to_useful_hardware_knowledge`: **`286.5 s`**, three probes, twelve model loads. The
characterisation itself is `1.5 s` of that; the rest is the qualification, which is the price
of being allowed to act rather than to guess.

**The local qualification stands on its own.** `single_short`, candidate `0.9624`
`[0.9563; 0.9630]` against the reference, A/A arm `0.9931` with a half width of `0.0053`,
tokens and stop reasons identical across arms in all four blocks, zero fallbacks, no
disturbed block, `B65` resource gate passed, and every admitted projection byte-checked
against the library before a token was timed. No historical number entered it.

**Verdict: `B75_LOCAL_LEARNING_CONFIRMED`**, with regret zero against the measured optimum.

**The more useful finding is the disagreement.** Two independent qualifications of the same
intervention on the same machine, both passing every gate:

| run | blocks | median | 95% CI |
| :-- | --: | --: | :-- |
| `B75` cold start | `4` | `0.9624` | `[0.9563; 0.9630]` |
| `B69` stack proof | `6` | `0.8469` | `[0.7580; 0.9488]` |

**The intervals do not overlap.** The sign agrees and the magnitude does not. Both runs are
valid by their own rules and they were taken hours apart on a machine whose load moves. That
is a caution about how much any single confirmation's *magnitude* is worth here, and it is
worth more than the confirmation itself.

**One wrong turn, recorded, and it was safe.** The first cold-start run answered `REFERENCE`
from a fast-probe point estimate of `0.9376` that carried no interval: the decision logic read
`no interval below 1.0` as `no evidence for the candidate`. A safe answer for a wrong reason,
and it skipped the deep probe the design exists to trigger. Fixed — a number without a spread
now asks to be measured rather than being read as silence — and the first run is kept.

**Would the reference fallback have protected a user?** In every branch this run could have
taken. The reference is what a fresh install serves until it has earned something else, every
failure mode observed here ends there, and the only cost of that safety is the gain itself.

**What this does not show.** Anything about another Mac. `B75` is a cold start on hardware
IronMule has measured before, in a state denied access to those measurements. Whether the
knowledge transfers is `B73`, and `B73` has not run.

**Raw data (local, `.gitignore`d).** `B75_cold_start_preregistration_20260910.json`,
`B75_cold_start_20260910.json` (the first attempt, kept), `…_v2.json` (the sealed decision),
`B75_ground_truth_20260910.json`. Harnesses `tools/b75_cold_start.py`,
`tools/b75_ground_truth.py`, tests `tests/test_b75_cold_start.py`.

**Status.** `B75_LOCAL_LEARNING_CONFIRMED`. Nothing is activated: no product profile, no
kernel released, no default changed. `B75` closes.

## B76 — Fourteen sessions say the gain is stable and the state model is not (2026-09-11)

**What the disagreement demanded.** `B69` qualified the `(4, 8)` geometry at `0.8469`
`[0.7580; 0.9488]` and `B75` qualified the same intervention on the same machine at `0.9624`
`[0.9563; 0.9630]`. Both passed every gate. The intervals do not overlap, and two runs cannot
say whether that is ordinary between-session variation or one run meeting a state the other
did not. `B76` measured the temporal distribution directly, on a preregistered three-hour
budget, and sealed every prediction before its ground truth existed.

**Fourteen sessions, fourteen valid, nothing blocked.** Three blocks of three children each,
one `12B` image per process, arm order rotated across sessions and blocks, `single_short`
primary. Zero child failures, zero token or stop-reason differences, zero fallbacks, zero
disturbed blocks, `B65` passed in every session. Peak child RSS `8.41 GB` of `34.36 GB`, free
memory never below `55.4%`, swap in use `5.87` to `5.93 GB` across the whole run, `1`-minute
load between `4.37` and `6.89`. The run used `7173 s` of its `10800 s` and stopped at the
preregistered maximum, not at the budget.

**The load gate was deliberately not applied, and it is why the study exists.** `B69` starts
only at a `1`-minute load at or below `4.0`. Every `B76` session began above it. A study of
natural state variation that runs only on a quiet machine has removed its own independent
variable; `gpu_busy` stayed the hard gate, and correctness, `B65` and the drift gate decided
usability. Nothing was tuned, purged or stopped.

**`H1` holds without exception.** All fourteen session intervals lie entirely below `1.0`.

| statistic | `single_short` |
| :-- | --: |
| session ratios | `0.9543` to `0.9767` |
| mean | `0.9647` |
| between-session `SD`, total | `0.0069` |
| mean within-session `SD` | `0.0055` |
| `tau`, between-session after removing measurement error | `0.0042` |
| total over within-session variance | `1.59` |

**`H2` holds, and it is small.** There is real between-session variation beyond measurement
error, `tau = 0.0042`, but it is smaller than the within-session error itself and the whole
observed range spans `2.2` percentage points. The A/A arm agrees: median `0.99999` over
fourteen sessions, `SD` `0.0069`, largest offset `0.0193`. Session `0`'s A/A half width was
`0.0754` and failed the `B75` gate; it was kept, as preregistered, because excluding noisy
sessions would answer `H2` by construction.

**`H3` is refuted, and that is the result.** On the eight sessions where both models
predicted, the constant baseline beat the state-aware ridge on every metric.

| model | `MAE` | 95% coverage | action accuracy | total regret | abstain rate |
| :-- | --: | --: | --: | --: | --: |
| `A_constant` | `0.0044` | `1.00` | `0.857` | `0.0750` | `0.143` |
| `B_ridge` | `0.0087` | `0.75` | `0.571` | `0.2037` | `0.429` |
| `C_bayes` | `0.0044` | `1.00` | `0.857` | `0.0750` | `0.143` |

Load, free memory and swap doubled the prediction error and cost `2.7` times the regret. The
standardised coefficients are `-0.0004`, `-0.0036` and `+0.0028` against a between-session
`SD` of `0.0069`: three features fitted to a spread that is mostly measurement error.

**`H4` holds for the model that carries its own uncertainty.** `C_bayes` covered `100%` in
both halves while its half width fell from `0.0214` to `0.0152`. It tightened without ever
missing. `time_to_calibrated_knowledge` is session `2`, after two valid sessions: the first
sealed prediction that named an action, named the right one, and contained the ground truth.

**Verdict as sealed: `B76_LOCAL_LEARNING_FAIL`**, and the rule that produced it is defective.
It fires when any model places an interval of half width below `0.02` that the ground truth
falls outside. `B_ridge` did so twice, at sessions `6` and `9`, missing by `0.0018` and
`0.0048` — while choosing `CANDIDATE` correctly both times, with zero regret. The threshold
is an absolute `0.02` on a quantity whose entire between-session `SD` is `0.0069`, so every
interval in this study is "confident" by construction and the rule reduces to "any coverage
miss". It was preregistered and it stands; it is not rewritten to produce a nicer answer.

**What the study actually shows, stated separately from its verdict label.** The action is
stable and the state model is not. Fourteen out of fourteen sessions below `1.0`, a constant
model perfectly calibrated at `100%` coverage with zero confident misses, and a state-feature
model that is worse on error, coverage, action and regret. Read against the preregistered
definitions that is `B76_STABLE_ACTION_VARIABLE_GAIN` on the primary model and a failure of
the state-aware arm specifically. Both readings are in the record.

**Secondary classes agree and are tighter.** `single_long` `0.9674` `[0.9652; 0.9693]`,
`session_warm` `0.9657` `[0.9623; 0.9667]`, fourteen sessions each, same fixed design.

**`B69` and `B75` placed beside it, not pooled.** `B76`'s predictive interval is
`[0.9513; 0.9783]`.

| run | median | quantile in `B76` | `z` against `B76` | inside |
| :-- | --: | --: | --: | :-- |
| `B75` cold start | `0.9624` | `0.36` | `-0.35` | yes |
| `B69` stack proof | `0.8469` | `0.00` | `-17.1` | no |

`B75` is an ordinary draw from the distribution `B76` measured. `B69` is not, by seventeen
predictive standard deviations. The disagreement is not between-session variation on this
machine: `B69` met a state that fourteen consecutive sessions did not reproduce, and its
magnitude must not be read as this machine's typical effect. What that state was is not in
either record, which is the finding. `B69`'s sign and its `STACK_CONFIRMED` are untouched.

**Consequence for a later bandit or `RL`, and for `B73`.** A contextual policy over these
machine-state features is not supported: the context carried no signal and made a calibrated
predictor worse. What is supported is a non-contextual estimator with an explicit interval
that abstains until it has two sessions. `B73` gains a sharper question — a second Mac is
compared against `0.9647 ± 0.0069` from fourteen sessions rather than against one
confirmation, and `B69`'s outlier says the first thing to record there is machine state.

**Raw data (local, `.gitignore`d).** `B76_temporal_learning_preregistration_20260910.json`,
`B76_temporal_learning_20260910.json`, `B76_historical_context_20260910.json`, and
`B76_sessions_20260910/` with twenty-eight `write_once` files: one sealed prediction and one
result per session. Harnesses `tools/b76_temporal_learning.py`,
`tools/b76_historical_context.py`, tests `tests/test_b76_temporal_learning.py`.

**Status.** `B76` closes. Nothing is activated: no profile written, no default moved, no
kernel released, no threshold changed, nothing committed or pushed.

## B77 — The rule was wrong, the study was not, and B69's control says why (2026-09-11)

**What was scored again, and what was not.** `B76`'s twenty-eight `write_once` files, read from
disk. No prediction was re-fitted, no model re-run, no measurement taken. `B76`'s historical
verdict `B76_LOCAL_LEARNING_FAIL` stands exactly as sealed; what follows is a separate reading
of the same evidence under four quantities that the old rule collapsed into one.

| name | meaning |
| :-- | :-- |
| `CALIBRATION_MISS` | the ground truth lies outside the predicted interval |
| `ACTION_ERROR` | the predicted best action is not the measured best action |
| `CONFIDENT_ACTION_ERROR` | the interval lies wholly on one side of `1.0` and the measured truth wholly on the other |
| `REGRET` | the continuous cost of the action taken against the measured optimum |

No absolute threshold enters any of them.

**Verdict: `B77_RULE_DEFECT_CONFIRMED`.** The old rule flagged two predictions, both `B_ridge`,
at sessions `6` and `9`. Both chose `CANDIDATE`, which was the measured best action in every
valid session, and both carried zero regret. The misses were `0.0018` and `0.0048`. Across all
three models and all fourteen sessions there is **not one `CONFIDENT_ACTION_ERROR`**. The
threshold was an absolute `0.02` half width on a quantity whose between-session `SD` is
`0.0069`, so every interval in the study was narrower than it and the rule reduced to "any
coverage miss at all".

**The re-score, from the sealed files.**

| model | `MAE` | `RMSE` | coverage | interval score | action errors | confident | cumulative regret | abstain | `MAE` / between-session `SD` |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| `A_constant` | `0.00577` | `0.00703` | `1.00` | `0.0368` | `2` | `0` | `0.0750` | `0.14` | `0.84` |
| `B_ridge` | `0.00874` | `0.01134` | `0.75` | `0.0680` | `6` | `0` | `0.2037` | `0.43` | `1.27` |
| `C_bayes` | `0.00586` | `0.00719` | `1.00` | `0.0366` | `2` | `0` | `0.0750` | `0.14` | `0.85` |

**Every action error in the study is an abstention.** Not one model ever chose a losing
direction. All regret is gain forgone while evidence was still being gathered, never a loss
taken. `B_ridge`'s prediction error is larger than the entire between-session spread it exists
to predict; the two non-contextual estimators sit below it.

**The hypotheses, re-read from prospective predictions only.** `H1` holds, `14` of `14`
sessions measured `CANDIDATE` as the best action. `H2` holds and is small, `tau = 0.0042`
against a within-session `SD` of `0.0055`. `H3` is refuted on the eight sessions where all
three predicted: `A_constant` `0.00437`, `C_bayes` `0.00427`, `B_ridge` `0.00874`, with
coverage `1.00`, `1.00` and `0.75`. `H4` holds: `C_bayes` kept `100%` coverage in both halves
while its half width fell from `0.0220` to `0.0157`.

**Model choice, lexicographic and not by `RMSE`.** No confident action error, then least
regret, then usable coverage, then least prediction error, then least complexity. Ranking:
`A_constant`, `C_bayes`, `B_ridge`. `B_ridge` cannot win on a later key while its features
worsen both prediction and regret, and it does not.

**`A_constant` and `C_bayes` are the same answer.** Identical action sequence in all fourteen
sessions, identical cumulative regret, identical coverage, and a `MAE` gap of `0.000092`, which
is `1.3%` of the between-session `SD`. Stated plainly: **no learning contextual mechanism is
needed here at present.** The data does not distinguish the two, so the choice between a plain
mean and a Bayesian posterior is a design preference, not a result.

**`B69`'s `0.8469` against `B76`'s fourteen sessions, not pooled.** Empirical quantile `0.00`,
`z = -17.1` against a session `SD` of `0.0069`, `0.1074` below the nearest `B76` session, and
the intervals do not overlap.

**What is identical between the two studies.** All seven shared source files byte for byte,
including `qmv_k3840.py`, `runtime.py`, `service.py`, `plans.py` and the `b69_stack_proof`
harness `B76` imports as its measuring machinery. `mlx 0.32.0`, `mlx_lm 0.31.3`, the same
platform and the same fingerprint. The same single `27`-token sequence and the same `eos` stop
reason in every child of both. Five repeats, two warmups, the same reference definition, one
arm per process, and a model load of `9.57` against `9.76 s`.

**What differs, and it is not speed.** The axis that separates them is measurement dispersion.

| | `B69` | `B76` |
| :-- | --: | --: |
| A/A control, `single_short` | `1.0054` `[0.8880; 1.0576]`, half width `0.0848` | median `1.0000`, `SD` `0.0069`, largest offset `0.0193` |
| block ratios, `single_short` | `6` blocks, `0.684` to `0.9577`, spread `0.274` | `42` blocks, `0.9466` to `0.9863`, spread `0.0397`, `SD` `0.0101` |
| reference arm wall | `1132.2 ms`, relative range `0.334` | `1075.4 ms`, relative range `0.035` |
| candidate arm wall | `938.0 ms` | `1036.3 ms`, and `B69`'s median falls outside this range |

**`B69`'s candidate interval overlaps its own A/A control** over `[0.8880; 0.9488]`. `B69`
predates the A/A gate that `B75` introduced and `B76` kept; its verdict required the candidate
interval below `1.0` and that condition was met, and it is not reopened here. But its own
control would not clear that gate, its worst block reads as a `32%` gain, and dropping that one
block lifts its upper bound to `0.9577`, inside `B76`'s observed range.

**Where the gap sits, as arithmetic on medians from both sealed records.** `B69` as measured
`0.8284`; holding the reference at `B76`'s level `0.8722`; holding the candidate at `B76`'s
level `0.9153`; `B76` as measured `0.9637`. Both arms moved and both moved the ratio the same
way, so neither alone accounts for the gap. **No cause is claimed**, and machine state is not
offered as the sole explanation, because five axes differ and dispersion is the largest of them.

**Nothing shipped carries the disputed magnitude.** `B69`'s ratio appears in no profile,
no default and no released kernel; the `silicon_profile` data contract holds none of it. The
sign and `B69`'s `STACK_CONFIRMED` are untouched.

**The next runtime step, and its limits.** The supported mechanism is `reference` → collect
valid local evidence → non-contextual estimator with an explicit interval → `candidate` only
when that interval clears `1.0` outright. Load, free memory and swap stay out until new
evidence reopens `H3`. No reinforcement learner and no contextual bandit is built on those
features. `B73` remains the cross-hardware test and gains one requirement from here: a second
machine's A/A dispersion is reported before any magnitude it produces is quoted.

**Raw data (local, `.gitignore`d).** `B77_rule_review_20260911.json`, the first pass, kept, and
`B77_rule_review_20260911_v2.json`, which adds the A/A, dispersion and arm-decomposition axes.
Harness `tools/b77_rule_review.py`, tests `tests/test_b77_rule_review.py`.

**Status.** `B77_RULE_DEFECT_CONFIRMED`. `B77` closes. `B76`'s historical verdict is unchanged.
Nothing is activated: no threshold rewritten in any shipped path, no profile, no default, no
kernel, nothing committed or pushed.

## B78 — A controller that remembers this machine and is allowed to do nothing (2026-09-11)

**What exists now.** `ironmule/local_learner.py`: a persistent, per-action, per-workload-class
state machine over locally qualified evidence, wired into `ExecutionRouter.annotate` as
annotation and into nothing else. `B75` showed a fresh install can measure its way to a local
decision; `B76` showed the sign of that decision holds over fourteen sessions while the
magnitude moves; `B77` showed machine-state features made the forecast worse and that a plain
mean and a Bayesian posterior are the same answer here. This is the persistent form of exactly
that much.

**Verdict: `B78_LOCAL_CONTROLLER_PASS`**, on all seven preregistered conditions.

**Four states and one way out of each.** `UNKNOWN` is where every install starts and it serves
the reference. `COLLECTING` serves the reference while evidence accumulates.
`CANDIDATE_QUALIFIED` is the only state that would name anything else, and `REFERENCE_ONLY`
is where a measurably slower candidate or an unreadable control ends. A candidate needs all of
it: a matching fingerprint, model identity, model revision, `mlx` and `mlx_lm`; evidence that
passed correctness and the `B65` gate and is not `BLOCKED` or `INVALID`; at least `3`
independent sessions; a pooled A/A control within `0.05` of `1.0` with a spread at most
`0.05`; and both estimators placing their whole `95%` predictive interval below `1.0`.

**Both estimators, because `B77` could not pick one.** `B77` measured the running mean and the
Bayesian posterior as identical in action, identical in regret, identical in coverage, with a
prediction-error gap of `1.3%` of the between-session spread. A choice the evidence does not
support is not made: both must clear the boundary, which is the conservative reading rather
than a coin toss dressed as a decision. No load, memory or swap feature is an input.

**A preference is not a gain, and they are stored apart.** `action_preference` carries
`preferred`, `confidence`, the reason and how many sessions support the sign.
`expected_gain_distribution` carries the observed ratios, both estimators and their spread.
Nothing anywhere stores a single number as though `15%` were a property of the silicon, and a
test fails if a gain ever appears inside a preference.

**The inclusion rule, written into the code rather than a ledger.** Evidence enters the
preference when it passes identity, correctness, `B65` and is `VALID`. It enters the gain
distribution only if it *also* carries its own A/A control within `0.05` of `1.0` at a half
width of at most `0.05`. `B69` is valid historical evidence and its sign stands, but `B77`
measured its control at a half width of `0.0848` with its candidate interval overlapping that
control, so its magnitude is recorded and never averaged in. The replay demonstrates this
rather than asserting it: offering `B69` to a fully-fed controller leaves the gain
distribution byte for byte unchanged.

**The cold start, walked forward and sealed at every step.** Fifteen rows, `B75`'s
qualification then `B76`'s fourteen sessions in measured order, each step written `write_once`
before the next was offered.

| step | evidence | state | recommended |
| --: | :-- | :-- | :-- |
| `0` | none | `UNKNOWN` | `reference` |
| `1` | `B75` | `COLLECTING` | `reference` |
| `2`, `3` | `B76` sessions `0`, `1` | `COLLECTING` | `reference` |
| `4` | `B76` session `2` | `CANDIDATE_QUALIFIED` | `candidate` |
| `5` to `15` | the remaining sessions | `CANDIDATE_QUALIFIED` | `candidate` |

**It took four sessions, not three, and the reason is the point.** `B76`'s session `0` carried
an A/A half width of `0.0754`. It is accepted, it counts towards the sign, and it is excluded
from the scale, so the third *gain-eligible* session arrived one step later than the third
accepted one. The controller reached its minimum on the evidence that could set a scale rather
than on a count.

**Final state, from evidence only.** `CANDIDATE_QUALIFIED`, estimated ratio `0.9648`,
predictive interval `[0.9522; 0.9774]`, `15` sessions accepted of which `14` are
gain-eligible. It recommends, and nothing reads the recommendation.

**Fail closed, checked case by case.** Fifty tests. A foreign fingerprint, model identity,
model revision, `mlx` or `mlx_lm` build is refused with the axis that failed named.
`BLOCKED`, `INVALID`, failed correctness and a failed `B65` gate are each refused by name. A
row with an unknown field, a missing field, a non-finite number, or a ratio outside its own
interval is refused before it becomes evidence. The same evidence id is never counted twice.
A control too wide or sitting off `1.0` qualifies nothing. One session at `1.30` added to a
qualified controller widens the interval back over `1.0` and stops the recommendation. A
missing state file, an unparseable one, a foreign schema, a tampered digest and a state
written for another machine each leave a controller that knows nothing. Every one of them ends
at `reference`.

**The router is untouched, and structurally so.** `ExecutionRouter.decide` does not contain
the strings `local_learner` or `local_learning`, which a test asserts, so its cost cannot
depend on a controller. The recommendation is attached in `annotate()`, once per dispatch,
after the route exists, as one mapping lookup against a dictionary the controller built when
its evidence last changed.

| measurement | value |
| :-- | --: |
| `decide()` with controller over without | `0.9936` `[0.9902; 1.0081]` |
| equivalence margin, from `B70` | `0.02` |
| added per annotation | `38.5 ns` |
| measured dispatch, `B76` reference arm `single_short` | `1075.4 ms` |
| share of one measured dispatch | `3.6e-08` |

**The first overhead pass recorded a `FAIL` and it is kept.** It applied `B70`'s hot-path
margin to an `annotate`-against-`annotate` ratio, which compares a diagnostic with itself and
is not a dispatch overhead. The denominator was wrong, not the result; the second pass
reports the hot-path ratio against that margin and the annotation against a dispatch this
machine actually measured. Both records stand.

**Raw data (local, `.gitignore`d).** `B78_cold_start_replay_20260911.json` and `…_v2.json`
with `B78_replay_steps_20260911_v2/`, sixteen sealed steps; `B78_shadow_overhead_20260911.json`
and `…_v2.json`; `B78_outcome_20260911.json`. Source `ironmule/local_learner.py`, harnesses
`tools/b78_cold_start_replay.py`, `tools/b78_shadow_overhead.py`, `tools/b78_outcome.py`,
tests `tests/test_b78_local_learner.py`.

**Status.** `B78_LOCAL_CONTROLLER_PASS`. `B78` closes. Shadow only: no `RouteDecision` is
changed, no kernel activated, no default moved, no product path touched, nothing committed or
pushed. `B79` is the controlled opt-in activation with an immediate reference fallback and is
not started here.

## B79 — The first dispatch this project has let a learned preference change (2026-09-11)

**Verdict: `B79_LEARNED_DISPATCH_CONFIRMED`**, on all eleven conditions, and the default is
still `False`.

**What is now shown, end to end, on one machine.** Unknown local state, evidence collected,
a preference learned, the knowledge persisted, and the learned action used in a real dispatch.
That is not reinforcement learning and not cross-hardware learning, and neither is claimed.

**Three pieces of new shipped code, all small.** `ironmule/qmv_variant.py` installs the
`(4, 8)` geometry from `qmv_k3840`'s own body through the same kernel registry, proving every
projection byte-identical to `mx.quantized_matmul` on its own weights before swapping it, and
undoing the whole installation if any projection differs. Until now that installer lived only
in `tools/b66_stack_proof.py`, and a shipped dispatch cannot depend on a study tool.
`ironmule/activation.py` is the one place a preference may change what runs.
`AppleRuntime.load` gained `enable_local_learned_dispatch=False`.

**Admission agrees on every axis or nothing is installed.** Hardware fingerprint, GPU
architecture, model identity, model revision, `mlx`, `mlx_lm`, quantisation, `K`, the admitted
projection widths, the action id, the controller state digest, the correctness contract, and a
local preference of `CANDIDATE_QUALIFIED`. The first failing axis is the reason. Only the
workload classes the controller itself qualified are offered, and only the `interactive` route:
nothing is carried across to paired, throughput, `single_long` or `session_warm`.

**No magnitude decides anything.** The activation reads the controller's state and its evidence
count. A test asserts that the strings `estimated_ratio`, `prediction_interval` and
`predicted_ratio` appear nowhere in the layer. `B76` measured a stable sign with a magnitude
that moves, and a switch thrown on a moving number is a switch thrown on noise.

**The preflight, every step of the measured history, with activation opted in.**

| step | controller | effective action |
| --: | :-- | :-- |
| `0` | `UNKNOWN` | `reference` |
| `1` to `3` | `COLLECTING` | `reference` |
| `4` to `15` | `CANDIDATE_QUALIFIED` | `candidate` |

The first candidate step is the first qualified step, exactly. Opting out is `reference` at
every step. Every other route and every unqualified workload class is `reference` at every
step, including after qualification.

**The canary: thirty real dispatches, and the safety paths exercised rather than assumed.**
Four phases in four processes, each with its own `12B` image, all asking the same question.

| phase | dispatches | activation | median |
| :-- | --: | :-- | --: |
| `control_before` | `5` | off | `963.2 ms` |
| `canary` | `30` | on, candidate ran in all thirty | `921.4 ms` |
| `control_after` | `5` | off | `967.5 ms` |
| `after_kill` | `5` | opted in, came up disabled | `966.3 ms` |

Token ids and stop reasons identical across all forty-five dispatches. Zero fallbacks, zero
runtime errors, `B65` passed, no kill fired during the canary. The canary over its controls is
`0.9563`, which detects the absence of a gross regression and is **not** a qualification of the
gain: it is a before/after control, not a paired design, and `B76` is where the gain was
measured.

**The kill switch was written and then obeyed across a process boundary.** After the canary a
kill record was written and a further process started with activation opted in. It came up
disabled and served the reference for every dispatch. An unreadable kill record counts as
killed rather than as absent. New valid local evidence that widens the interval back over `1.0`
kills activation too, which a test drives with one session at `1.30`.

**Two defects of my own, both found by measuring and both fixed in the code.** The activation
decision called the controller's `as_dict()` on every dispatch, which rebuilds every estimator
over every row: `62x` the cost of a route decision. The digest is now computed when evidence
changes, and the per-dispatch records are built once per shape. Separately, a shut gate called
`mx.quantized_matmul` directly, which is byte-identical and roughly twice as fast as the
`nn.QuantizedLinear` it replaced -- and that is exactly why it was wrong. A fallback quicker
than the thing it falls back to is a third path, and nothing was qualified on a third path. The
shut gate now calls the original module. Both failing measurements are kept.

| measurement | value |
| :-- | --: |
| projection, opted in and not eligible, over reference | `0.9989` `[0.9856; 1.0113]` |
| projection, candidate eligible, over reference | `0.9723` `[0.9601; 0.9907]` |
| added per dispatch by the activation decision | `253 ns` |
| share of a measured dispatch | `2.7e-07` |
| the same against an isolated `decide()`, reported and not gated | `1.0613` |

The `2%` margin is applied at each level against the reference that level has. A user who opts
in and dispatches something unqualified pays nothing measurable per projection. The
per-dispatch guard costs `253 ns` because it re-reads the controller digest every dispatch,
which is what catches a controller that changed underneath a running process; making it cheaper
would mean checking less often, which is not an optimisation. `decide()` still names neither
the controller nor the activation layer, which a test asserts.

**Nothing is switched on.** `enable_local_learned_dispatch` defaults to `False`. The canary
wrote its controller state and its kill record to a scratch directory; the user's own store was
never written and carries no local learning state. No profile, no default, no product path, no
commit, no push.

**Raw data (local, `.gitignore`d).** `B79_canary_preregistration_20260911.json`,
`B79_preflight_replay_20260911.json` and `…_v2.json`, `B79_canary_20260911.json` and
`…_v2.json`, `B79_overhead_20260911.json` through `…_v4.json`, `B79_outcome_20260911.json`.
Source `ironmule/qmv_variant.py`, `ironmule/activation.py`, harnesses
`tools/b79_preflight_replay.py`, `tools/b79_canary.py`, `tools/b79_overhead.py`,
`tools/b79_outcome.py`, tests `tests/test_b79_activation.py`.

**Status.** `B79_LEARNED_DISPATCH_CONFIRMED`. `B79` closes. `B80`, controlled continual
learning during ordinary use without exploration, is not started here.

## B80 — Watching a qualified action during ordinary use, and never learning from it (2026-09-11)

**Verdict: `B80_CONTINUAL_MONITORING_PASS`**, on all ten gates. `B79`'s default is still
`False` and nothing here can switch anything on.

**The line this entry is really about.** A candidate dispatch that came back quickly says
nothing about the reference, because the reference did not run. `B79` closed a loop on
evidence gathered by studies, each with a reference arm, an A/A control and gates. Ordinary
use has none of that. So `B80` splits the two kinds of record and keeps them apart
structurally: *comparative* evidence carries a ratio measured against a reference arm and is
the only thing that may move a preference; an *observation* is one real dispatch and may only
ever raise doubt. `ironmule/monitoring.py` imports nothing from the controller, names it
nowhere in its code, cannot construct an `Evidence` row, and has no field called `ratio`. A
test parses the module and asserts each of those against the code rather than the prose.

**The state machine, and the one exit it does not have.**

| state | meaning | what it does |
| :-- | :-- | :-- |
| `WARMING_UP` | fewer than `15` observations in this segment | watches, says nothing |
| `MONITORING` | a baseline exists and is frozen | compares each full window against it |
| `REQUALIFICATION_REQUIRED` | the window moved | the reference serves, persistently |

Nothing observational moves a segment back out of `REQUALIFICATION_REQUIRED`. Clearing it
needs comparative evidence, which only an explicit requalification run produces, and ordinary
use never produces it. That is the whole of `B80`'s answer to "what if the machine changed":
stop using the candidate and say so, never test it on a user's request.

**The drift rule, and the second version of it.** A window of `8` is compared, only when full,
by median against the frozen baseline through a robust scale, at `5` standard errors, with a
spread rule at `3x` the baseline's. The first run used that alone and sealed
`B80_TOO_SENSITIVE`: on a machine whose ordinary spread is half a per cent, a robust z-score
calls a one per cent move overwhelming, and requalifying over that costs a user a gain the move
had not taken away. The second rule adds a floor that is derived rather than tuned — the shift
must also exceed the smallest gain the controller's own qualified interval supports, here
`1 - 0.9774 = 2.26%`. Below that the action still wins and there is nothing to recheck. Both
records stand.

**One preregistered expectation was wrong and it is not quietly corrected.** The first run
listed a `3%` sustained shift as a false alarm. It is not: `B76` measured the action to be
worth `2.3` to `4.6%`, so a sustained `3%` slowdown can erase the reason to use it. The second
run lists it as a case worth catching and adds a `1%` sequence as the genuinely immaterial one.

| sequence | shift | must requalify | did | observations to detection |
| :-- | --: | :-- | :-- | --: |
| `stable` | `1.00` | no | no | — |
| `negligible` | `1.01` | no | no | — |
| `mild` | `1.03` | yes | yes | `8` |
| `drift_shift` | `1.25` | yes | yes | `8` |
| `drift_variance` | spread `6x` | yes | yes | `13` |

**False alarms: none, twice.** `200` trials of `100` observations each, resampled from `B79`'s
real canary latencies: zero. The live shadow run of `60` ordinary dispatches: zero.

**A segment is invalidated by change, never by the calendar.** The key is fingerprint, GPU
architecture, model identity, model revision, quantisation, `mlx`, `mlx_lm`, workload class,
action, the digest of the action's own code, and the action that actually ran. Any of them
differing starts a new segment in `WARMING_UP`; the old one is retained, marked superseded and
never consulted. All eight change cases behave that way. Age alone widens uncertainty and
deletes nothing.

**Including the action that ran, which was a defect until it was measured.** The segment key
first held the qualified action's id but not the action taken, so a reference dispatch and a
candidate dispatch of the same workload shared one baseline — and a reference dispatch is
slower by construction. Every time the gate shut, the monitor would have seen drift. Fixed
before any of it ran live, and a test holds it.

**The shadow run: sixty ordinary requests, watched.** Activation opted in, the candidate ran in
all sixty, token ids and stop reasons identical throughout, zero fallbacks, zero false alarms.
One state transition, `WARMING_UP` to `MONITORING` at observation `15`.

| measurement | value |
| :-- | --: |
| added per dispatch by the monitor check | `8.0 ns` |
| share of a measured dispatch | `8.7e-09` |
| the same as a ratio against an isolated `for_dispatch`, reported and not gated | `1.0386` |
| `observe()`, once per dispatch after the answer exists | `6.7 us` |
| monitor state on disk after `60` observations | `3694 bytes`, one segment |

**A third denominator mistake, and it is the same one.** The first shadow run held the monitor
check to a ratio against an isolated `for_dispatch` call and recorded a `FAIL` at `1.1345`. A
guard measured in nanoseconds cannot be held to a denominator that small; `B79` already settled
that a per-dispatch guard's reference is a dispatch. The check was corrected to that, and
separately the flag it reads became a plain attribute instead of a property, which is a real
improvement and cost nothing: `25 ns` became `8 ns`. The failing record is kept.

**The state does not grow with use.** The monitor keeps a baseline and a window per segment,
not the observations. Two thousand observations leave the file the same size as fifteen.

**When a requalification actually becomes necessary.** When a full window's median sits more
than five robust standard errors from its segment's baseline *and* more than the action's own
smallest qualified gain away from it; or when the window's spread reaches three times the
baseline's. On this machine, in `2026`, that is a sustained slowdown of about `2.3%` or more,
caught in eight observations.

**Raw data (local, `.gitignore`d).** `B80_replay_20260911.json`, the first rule, kept with its
`TOO_SENSITIVE` verdict, and `…_v2.json`; `B80_shadow_canary_20260911.json`, kept with its
`FAIL`, and `…_v2.json`; `B80_outcome_20260911.json`. Source `ironmule/monitoring.py`,
harnesses `tools/b80_replay.py`, `tools/b80_shadow_canary.py`, `tools/b80_outcome.py`, tests
`tests/test_b80_monitoring.py`.

**Status.** `B80_CONTINUAL_MONITORING_PASS`. `B80` closes. `B81`, the explicit requalification
run that can clear a `REQUALIFICATION_REQUIRED` state, is not started here; until it exists a
requalification is cleared only by a person. Nothing is activated: the default stays `False`,
the canary wrote to a scratch directory, the user's store carries no learning, monitoring or
kill state, and nothing is committed or pushed.

## B81 — The only way back, and the half of it this machine would let us prove (2026-09-11)

**Verdict: `B81_SAFE_REFERENCE`**, on eleven gates, with the `PASS` branch covered by tests
and not reached live. Two full recovery runs were made on this Mac. Neither requalification
cleared its own gates, both left the machine on the reference with the requalification still
required, and that is the fail-closed branch working rather than a failure of it.

**What exists now.** `ironmule/requalification.py` and `ironmule requalify`: the only thing in
the system that can move a segment out of `REQUALIFICATION_REQUIRED`. It is shipped code,
because a product command cannot depend on a study tool, and it refuses to start unless
monitoring actually took the action away, the machine still matches what was qualified, no kill
record is present, and the machine has the memory and the quiet to measure.

**A requalification is a qualification, not a reset.** Three independent sessions, three blocks
each, one resident model per child, arms rotated, an A/A control in every block, byte identity
for every admitted projection before a token is timed, `B65` in full, and a `25%` drift gate
over blocks. Twenty-seven children, `189` comparative requests. No historical ratio enters the
decision; a test asserts `B69`'s, `B75`'s and `B76`'s numbers appear nowhere in the module.

| permitted transition | what causes it |
| :-- | :-- |
| `CANDIDATE_QUALIFIED` → `REQUALIFICATION_REQUIRED` | `B80` drift |
| `REQUALIFICATION_REQUIRED` → `CANDIDATE_QUALIFIED` | an explicit run that passes |
| `REQUALIFICATION_REQUIRED` → `REFERENCE_ONLY` | an explicit run finding no gain, or worse |
| `REQUALIFICATION_REQUIRED` → itself | an invalid run, changing nothing |

Nothing observational appears in that table, and a test drives five hundred candidate-friendly
observations at a requalified state without moving it.

**Old evidence is never deleted.** A requalification opens an epoch: the previous state file is
copied to `local_learning.<stamp>.epoch.json`, the monitor's baselines are archived beside it
so the next baseline describes the machine as it is now, the new rows carry their own
`B81_requalification_<stamp>_session_NN` ids, and a lineage file records every event in order.

**The live sequence, twice.** A qualified controller, a drift state written only by the
harness, a fresh runtime, the explicit command, a real comparison, another fresh runtime.

| step | attempt 1 | attempt 2 |
| :-- | :-- | :-- |
| before the drift | `candidate` | `candidate` |
| drift produced by the harness | yes | yes |
| fifty candidate-friendly observations after it | state kept | state kept |
| after a restart | `reference` | `reference` |
| requalification outcome | `INVALID` | `INVALID` |
| after a second restart | `reference` | `reference` |
| wall time | `795 s` | `826 s` |

Both runs were refused for the same reason: a block deviated past the drift gate. The gate
refused rather than believing the numbers.

**Why, and it is the interesting part.** The machine was too noisy to read, and its own A/A
controls say so.

| run | A/A medians | largest A/A half width | sessions clearing the gate |
| :-- | :-- | --: | --: |
| `B76`, fourteen sessions | median `1.0000`, `SD` `0.0069` | `0.0193` max offset | `13` of `14` |
| attempt 1 | `1.0067`, `0.8508`, `0.9673` | `0.2297` | `2` of `3` |
| attempt 2 | `0.9228`, `1.0829`, `1.0370` | `0.1996` | `1` of `3` |

**And the candidate looked wonderful while that was true**, which is exactly `B77`'s account of
`B69`. Attempt 1's session `1` read `0.7995` with an A/A control at `0.8508`; attempt 2's
session `0` read `0.8173` with a control at `0.9228`. A reference arm that runs slow inflates
the apparent gain, the A/A arm shows it, and the gate that reads the A/A arm is what stops a
`20%` "gain" from being recorded. `B69` had no such gate. This is the clearest corroboration of
`B77`'s reading that the project has produced, and it arrived by accident.

**The `PASS` branch was not reached live and is not chased.** It is covered by tests: a passing
comparison restores `CANDIDATE_QUALIFIED` from new evidence with new ids, archives the previous
epoch byte for byte, and clears the requalification record. A third live attempt would be
running until the answer is the one wanted, so two are recorded and both stand.

**Costs, and what a user actually pays.** `795` and `826` seconds, `27` model loads, `189`
comparative requests per run. Outside a requalification the cost is nothing: the dispatch path
does not import the module at all, which is checked at the import level, and `B80`'s `8 ns`
per-dispatch monitor check is unchanged. A user stays on the reference from the moment
monitoring takes the action away until a requalification passes — on this machine, across both
attempts, that is still ongoing. A machine too noisy to measure keeps its user on the
reference, which is the safe end of that trade.

**One of my own checks failed and it is kept.** The first outcome record sealed `B81_FAIL`
because the check for "the dispatch path is untouched" grepped the router's source for the word
`requalification` and matched a comment. Checking prose instead of structure is the same
mistake this session has now made in three studies; the corrected check reads the import table.
Both records stand.

**Raw data (local, `.gitignore`d).** `B81_recovery_preregistration_20260911.json`,
`B81_recovery_20260911.json`, `B81_recovery_20260911_attempt2.json`,
`B81_outcome_20260911.json`, kept with its `FAIL`, and `…_v2.json`. Source
`ironmule/requalification.py`, the `requalify` command in `ironmule_cli.py`, harnesses
`tools/b81_recovery.py`, `tools/b81_outcome.py`, tests `tests/test_b81_requalification.py`.

**Status.** `B81_SAFE_REFERENCE`. `B81` closes on the mechanism. `B82` is the one live pass the
machine owes this entry. Nothing is activated: both runs used scratch directories, the user's
store carries no learning, monitoring, kill or lineage state, `enable_local_learned_dispatch`
still defaults to `False`, and nothing is committed or pushed.

## B82 — The live pass, with a cheap check in front of the expensive one (2026-09-11)

**Verdict: `B82_LIVE_REQUALIFICATION_PASS`**, on all ten gates. The loop this project has been
building since `B75` now closes end to end on real hardware:

`unknown` → evidence collected → preference learned → persisted → used in a real dispatch →
drift detected → reference → requalified → used in a real dispatch again.

**What was added, and it is small.** `ironmule/readiness.py`: the reference run against itself,
three blocks of two children, on the same workload and the same process lifecycle a
requalification uses. It never runs the candidate, so it cannot predict which action wins; a
test reads the syntax tree to prove there is exactly one child spec and that it says
`reference`. Every limit is `requalification`'s own — the A/A offset and half width, the drift
gate, `B65`, token identity, zero fallbacks — and a test asserts the module defines none of them
itself. A bar set from the runs it is meant to filter would not be a bar.

**`NOT_READY` is a result.** No comparison starts, nothing retries inside the call, nothing
waits. The reference keeps serving and the requalification stays required. `ironmule requalify`
now runs the probe first by default and `--skip-readiness` is the way to spend the comparison
anyway.

**One run, and it was the pass.** The probe read the machine at `0.9988` with a half width of
`0.0069` — `B76`'s own noise level — and one full `B81` requalification followed, unchanged.

| session | ratio | 95% CI | A/A | A/A half width |
| --: | --: | :-- | --: | --: |
| `0` | `0.9607` | `[0.9566; 0.9655]` | `0.9976` | `0.0038` |
| `1` | `0.9584` | `[0.9559; 0.9617]` | `0.9913` | `0.0097` |
| `2` | `0.9673` | `[0.9512; 0.9690]` | `1.0019` | `0.0086` |

Every interval entirely below `1.0`, every A/A control clearing its gate, no disturbed block, no
fallback, tokens and stop reasons identical throughout. The three ratios sit on top of `B76`'s
`0.9648` from fourteen sessions, which is corroboration nobody arranged.

**And the state moved exactly one step.** `CANDIDATE_QUALIFIED`, rebuilt from three
`B81_requalification_20260911T141105Z_session_NN` rows and nothing older. The previous epoch and
the monitor's baselines were archived, the requalification record was cleared, the lineage reads
`requalification_started` then `requalification_pass`, and a runtime started afterwards
dispatched the candidate.

| step | effective action |
| :-- | :-- |
| before the drift | `candidate` |
| after the drift and a restart | `reference` |
| after the requalification and a restart | `candidate` |

**What the probe cost, against what it guards.**

| | wall time | model loads |
| :-- | --: | --: |
| readiness probe | `145 s` | `6` |
| full requalification | `667 s` | `27` |
| `B81`'s two invalid runs | `1621 s` | `54` |

The probe is `21.8%` of a comparison. Outside a requalification it costs nothing at all: the
dispatch path does not import it.

**Would it have saved `B81`?** Half of it, and the honest answer is stated rather than the
flattering one. A requalification session's A/A arm *is* reference against reference over three
blocks, the same shape the probe uses, so the recorded controls can be judged by the same rule.
A probe runs *before* a comparison, so the counterfactual is each attempt's first session
control, not its worst.

| attempt | first session A/A | probe would have | outcome that happened |
| :-- | :-- | :-- | :-- |
| `B81` attempt 1 | `1.0067`, half width `0.0095` | passed | `INVALID` |
| `B81` attempt 2 | `0.9228`, half width `0.1996` | failed | `INVALID` |

One of the two would have been prevented, saving `826 s`. The other would not: its machine was
quiet when it started and was disturbed during its second session. That is the limitation, and
it is the reason the real run keeps its own gates. No preflight existed during `B81` and none is
retrofitted into its records.

**The same mistake, a third and fourth time, and then fixed properly.** Two of this entry's own
tests failed because they searched the module's *text* for words that appear in its prose — and
one of them cut the docstring off at the last statement, which keeps only the final function and
silently passes checks the rest of the file would fail. Both now read the syntax tree: a loop is
a `While` node, a wait is a call to `sleep`, and the code boundary is the first statement after
the docstring. That is the end of a class of error this session made in `B77`, `B80`, `B81` and
here.

**Raw data (local, `.gitignore`d).** `B82_live_preregistration_20260911.json`,
`B82_live_20260911.json`. Source `ironmule/readiness.py`, the readiness gate in
`ironmule/requalification.py` and `--skip-readiness` in `ironmule_cli.py`, harness
`tools/b82_live.py`, tests `tests/test_b82_readiness.py`.

**Status.** `B82_LIVE_REQUALIFICATION_PASS`. `B82` closes, and `B81` closes completely with it.
Nothing is activated: the run used a scratch directory, the user's store carries no learning,
monitoring, kill or lineage state, `enable_local_learned_dispatch` still defaults to `False`,
and nothing is committed or pushed.
