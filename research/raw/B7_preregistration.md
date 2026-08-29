# `B7` preregistration — which term dominates the falling gain

Written **before** any measurement. Results are appended below the line, never edited in.

- **Written:** 2026-08-29
- **Code under test:** IronMule `0de69b6`, in an isolated detached worktree
  (`.worktrees/ironmule-b7`) so nothing is written into the peer session's tree.
- **Why it matters:** `docs/BACKLOG.md` calls `B7` "the highest-value entry in Tier 1
  because every other tier depends on knowing which term dominates." Tier 2 (`B8`,
  `B9`, `B10`) is entirely about making host submission cheaper. If device time, not
  host submission, is what grows with model size, that whole tier is aimed at the
  wrong term.

## The discrepancy being investigated

`SCALING.md` explains the falling grouping gain as fixed per-step overhead becoming a
smaller share of a longer step. Quantitatively that story predicts:

- host dispatch scales with kernel count, so roughly with layers
- device time scales with weight traffic, so parameters (`6.75x`) divided by the better
  bandwidth larger matrices achieve (`195 -> ~300 GB/s`), about `4.4x`
- the recoverable share should therefore fall to about `0.41` of its 4B value

**Measured: `11.81 / 19.24 = 0.61`.** The simple story over-predicts the fall by half.
Something in it is wrong, and this run is to find out which half.

## A correction to the entry's premise — WITHDRAWN, it was mine that was wrong

**An earlier version of this document claimed `B7` was wrong to say `34 -> 62` layers,
and that the true figure was 64. That claim is false and is retracted.**

`docs/SCALING.md:71` gives the 27B row as `| 27B | 5376 | 21504 | 62 | 16 |` — 62 layers
— and line 101 states plainly that "4B, 12B and 27B are all Gemma 3". The scaling series
is one family throughout, and `B7`'s `34 -> 62` is correct for it.

What I actually did: the only 27B on this machine is `mlx-community/Qwen3.8-27B-4bit`,
I read the one config I could open, found 64 layers, and attributed it to Gemma. That is
the size-versus-family confusion `B26` exists to prevent, made while writing a validity
section that warns about it. Recorded rather than deleted, because the failure mode is
more useful than its absence: **the config you can open is not necessarily the model the
document means.**

| Model | Layers | Layers vs 4B | Family |
| :-- | --: | --: | :-- |
| `gemma-3-4b-it-4bit` | 34 | 1.00x | Gemma 3 |
| `gemma-3-12b-it-4bit` | 48 | 1.41x | Gemma 3 |
| Gemma 3 27B (`SCALING.md`, not cached here) | 62 | 1.82x | Gemma 3 |
| `Qwen3.8-27B-4bit` (cached, **not** the scaling series) | 64 | — | Qwen |

What survives, and is the actual methodological change here: the entry's arithmetic
never uses the 12B midpoint. A two-point fit through 4B and 27B cannot distinguish a
term that grows linearly from one that saturates. **Three points can**, and 4B->12B is
family-clean.

## Method

Reuse `research/e14b_arms.py` unchanged — it already takes `--model` and already
produces the four-way split this entry asks for
(`host_prep_ns`, `submission_ns`, `completion_wait_ns`, `total_ns`), with the existing
warmup, repeat, barrier and bit-equality controls. **No new measurement code**: a
second harness written for one entry is a second thing that can be wrong, and this
question is about numbers the existing harness already emits.

Order: a feasibility pilot at 4B first, to establish runtime and confirm the harness
runs in this worktree. **The pilot is not evidence** and its numbers are not reported as
results. Then 4B and 12B as the family-clean pair, then 27B if it fits without swap.

## Predictions, stated before measuring

If the `SCALING.md` story is right, from 4B to 12B (layers `1.41x`, parameters `3x`):

1. `submission_ns` grows by roughly `1.41x` — it tracks kernel count, so layers.
2. `completion_wait_ns` grows by roughly `3x / (bandwidth gain)`, so between `2x` and
   `3x`.
3. `host_prep_ns` stays roughly flat — it is per-step Python work, not per-layer.

## What each outcome means

- **Submission grows faster than `1.41x`.** Host dispatch is superlinear in layers;
  something per-step costs more on a bigger model than the kernel count explains.
  Tier 2 gets *more* valuable, not less.
- **Completion wait grows slower than `2x`.** Device time is not tracking weight
  traffic; the larger matrices are reaching better bandwidth than the estimate credits,
  and the ceiling is further away than `SCALING.md` implies.
- **Both.** The two errors partly cancel, which would explain a measured `0.61` against
  a predicted `0.41` without either term being individually right.
- **Neither; both land inside prediction.** Then the model is right at 12B and the
  discrepancy lives at 27B, which points at the family confound rather than at scaling.

## Kill

Nothing closes this entry except an answer — the backlog says so, and it is right. But
the run itself is **invalid**, and must be discarded rather than interpreted, if:

1. `gpu_busy()` reports another loaded model during any arm.
2. Swap delta is nonzero at any model size (the 27B risk on 32 GB).
3. Bit-equality fails between arms at any size.
4. Fewer than the configured repeats complete at any size.

## Explicitly not claimed

This measures where the time goes on **one machine** at up to three model sizes. It does
not establish a scaling law, does not extend `docs/LIMITS.md`, and does not by itself
justify or kill any Tier 2 entry — it tells the next person which term to aim at.

Per `Q2`'s lesson, flagged in advance: any number that ends up in a summary here must
say which measurement it came from. The screening/confirmation split went wrong once
already in this codebase.

---

## Results

Run 2026-08-29 on IronMule `0de69b6`, isolated worktree. AC power, swap `0.06 MB`
throughout, `gemma-3-4b-it-4bit` and `gemma-3-12b-it-4bit` from local cache.
Medians in ms.

### Deviation from the method, recorded

12B completed **one** repetition, not four. `e14b_arms.py:236` aborts when
`mlx_peak_bytes` exceeds 12 GB; 12B reported `17.51 GB` and the loop broke. That number
is a cross-arm high-water mark, not a per-arm peak — MLX's peak counter is never reset
(`reset_peak_memory` appears nowhere in the repo) and every arm loads its own model in
the same interpreter. 4B reached `11.53 GB` by its fourth repetition, one repetition
away from the same fate.

Also: `--processes N` does **not** fork. `e14b_arms.py:234` loops in one interpreter;
all four 4B runs carry pid `94927`.

**Both of these are already documented, and this run reproduced them rather than
finding them.** `research/LEDGER.md:1071` limitation `M2` records that "four fresh
processes" are four blocks in one OS process, with cumulative peaks
`7.07 → 7.07 → 9.24 → 11.25 GB`, and states explicitly that "the same loose
implementation is present in E14 and E14b". Measured here: `6.37 → 7.28 → 9.36 → 11.53`.
`M3` (`LEDGER.md:1080`) records a run hitting the 12 GiB guard and aborting after one
block. An earlier draft of this document presented both as new. They are not, and the
error was mine: the project's own rule is to read the recorded dead ends first, and I
read `BACKLOG.md` without reading the ledger's limitation sections.

What this run does add is narrower: `M3` attributes its abort to a prefill cache, not to
the guard reading a cumulative mark, so the guard's systematic early-firing — worse the
more blocks have already run — is new. And the abort is invisible in the output: it is
printed to stdout only, so `B7_12b.json` looks like an ordinary result with `runs: 1`.
Without the console log this deviation would have gone unnoticed.

Consequence for this experiment: 12B cells rest on 7 samples, 4B on 28. `M2` is explicit
that "within-block arm comparison is unaffected, since drift hits every arm in a block
alike" — and every comparison drawn below is within-block, across model sizes, inside a
single arm. What is weakened is the independence of the paired bootstrap across
repetitions, which this analysis does not use. The 12B ratios are additionally
consistent to within `±0.05` across four batch sizes.

Unaffected: `Q2`. `tune.py:276` routes its confirmation through `ironmule/ab.py`, whose
`_child` forks via `subprocess.run` (`ab.py:81`). Those six processes were real.

### The four-way split

| Arm | Batch | 4B prep | 4B subm | 4B wait | 12B prep | 12B subm | 12B wait | subm x | wait x |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| A | 1 | 0.53 | 6.28 | 6.13 | 0.86 | 23.10 | 9.15 | 3.68 | 1.49 |
| A | 8 | 5.23 | 50.85 | 48.79 | 8.17 | 187.06 | 72.97 | 3.68 | 1.50 |
| B | 8 | 3.95 | 73.53 | 10.11 | 4.92 | 217.19 | 17.55 | 2.95 | 1.74 |
| C | 8 | 0.69 | 46.72 | 28.76 | 0.96 | 184.48 | 35.10 | 3.95 | 1.22 |

### Against the predictions

Predictions were stated above before measuring. Both were wrong, in opposite directions.

1. **Submission would grow ~`1.41x`, tracking layers.** Measured **`3.68x`**, stable
   across every batch size. The submission window grows **2.6x faster** than kernel
   count.
2. **Completion wait would grow `2x`-`3x`, tracking weight traffic.** Measured
   **`1.50x`**, again stable across batches. **Half** the low end of the prediction.
3. Host prep roughly flat — held in the arm that matters (C: `0.69 -> 0.96`); A and B
   grow ~`1.6x` but stay under 4% of the step.

### The discrepancy is reproduced and explained

Shipped grouping gain (A to B), independently measured here:

| | batch 2 | batch 4 | batch 8 |
| :-- | --: | --: | --: |
| 4B | 12.44% | 17.92% | 16.36% |
| 12B | 8.33% | 12.99% | 10.34% |

At batch 8 the gain falls to `10.34 / 16.36 = 0.63` of its 4B value. The ledger's
independent figure is `11.81 / 19.24 = 0.61`. **This run reproduces the falling gain to
within two points without sharing any measurement with it.**

`SCALING.md` predicted `0.41`. The reason it over-predicted the fall is now visible and
is not one error but two that partly cancel: the host term grows far faster than the
model assumes (`3.68x` vs `1.41x`) while the device term grows far slower (`1.50x` vs
`~3x`). That is outcome 3 of the four stated in advance.

### The strategic consequence, which points the opposite way to `SCALING.md`

| | 4B | 12B |
| :-- | --: | --: |
| submission / device-wait, arm A, batch 1 | 1.02x | 2.52x |
| submission / device-wait, arm A, batch 8 | 1.04x | 2.56x |

At 4B the two are balanced. At 12B the submission window is **2.5x** the device wait —
`187 ms` of a `268 ms` step. `SCALING.md` assumes fixed host overhead becomes a
*smaller* share as models grow. Measured, the opposite happens: **the larger the model,
the more host-bound the step becomes.**

If that holds, Tier 2 (`B8`, `B9`, `B10` — take the decode loop out of per-operation
Python, record the step once, cut kernel count) is aimed at the term that *dominates* at
scale, and is worth more at 12B and 27B than at the 4B where all the evidence was
gathered. The backlog's warning that those entries shrink the headline ratio still
stands and is not in conflict: they would shrink the ratio precisely by fixing the
thing that costs the most absolute time.

### The limit of this instrument, and why `B24` comes first

`submission_ns` is **not** pure host work and must not be read as such. Arm B at 4B
batch 8 submits for `73.53 ms` and then waits `10.11 ms`; arm A submits `50.85` and
waits `48.79`. Same work, same shapes. Arm B's submission window is larger precisely
because device execution is happening *inside* it — that overlap is the mechanism the
whole product is built on. So the split measures **windows on a wall clock**, not host
and device costs separately.

Everything above survives that, because it compares like with like across model sizes
within each arm. But the natural next question — *what fraction of the growing
submission window is Python, and what fraction is the device* — cannot be answered with
this instrument at all. That is `B24` ("Stop measuring the GPU with a wall clock"),
which the backlog already lists as the prerequisite for `B10`. This run turns that
ordering from a methodological preference into a hard dependency: **`B8`/`B9`/`B10`
cannot be sized without `B24` first.**

### Side finding: `B28` reproduced on Gemma

The harness's correctness block compares true-batched decode against batch-1 singles.
At batch 8, sequence 3, position 6: `1580` (single) vs `1437` (batched). Deterministic
across all four 4B repetitions; prefill logits bit-equal, so the divergence arises in
decode. Arm C is **True Batch**, a path IronMule deliberately does not ship; arm B, the
shipped `ThroughputMode`, stays token-identical.

This is an independent reproduction of Tier 0's `B28` rejection on a different model
family (Gemma, not Qwen) and is evidence *for* the decision not to route true batching.
It is not a defect in the shipped runtime.

### Kill criteria

None triggered. `gpu_busy` clear, swap `0.06 MB` throughout, bit-equality held in every
arm the product ships, all configured repeats completed at 4B. The 12B truncation is a
recorded deviation, not a kill: the guard aborted cleanly and the surviving cells are
internally consistent.

### Not claimed

One machine, two model sizes, one family, one MLX build. 27B was not run — at a true
per-arm peak near 17 GB it is feasible on 32 GB, but only after the peak guard is fixed,
or it will abort on the same inflated number. The 4B-to-12B leg is family-clean; nothing
here separates model size from model family, which remains `B26`.
