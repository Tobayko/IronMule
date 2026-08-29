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
half the low end of its estimate. Neither term is individually close, and the entry's
own arithmetic premise is also slightly off: the 27B model has **64** layers, not the
`62` the backlog states, so its layer term is `1.88×`.

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
