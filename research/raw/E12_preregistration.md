# E12 — Falsification test: does prefix KV reuse survive the sliding-window boundary?

**Experiment ID** `E12`
**Frozen at commit** `750be38377d11b1285b54f48e6403c9959e78ae9`
**Branch** `forge/hardware-autotune`
**Registered** 2026-08-25, before any E12 measurement was taken or viewed.

This preregistration is frozen. Any necessary change gets a new experiment ID
(`E12b`, `E13`, …) rather than an edit here.

---

## 1. Purpose

E9 measured `max |delta| = 0.0000` between a chunked prefill and prefix KV reuse
across 12 requests and every decode step, at a single prefix length of 276 tokens.
That is a strong local result. It is **not** a proven general property.

This experiment exists to **destroy** that result with the cheapest hard
counterexample available, not to confirm it. Gemma 3 uses sliding-window attention
with a window of 1024 tokens on 29 of its 34 layers. Every E9 prefix was far below
that window, so the window never clipped anything. Prefix lengths at and above 1024
are therefore the cheapest place where a plan-internal difference could appear.

If the attempt fails, and only then, the validity domain may be widened.

## 2. Hypotheses

**H0** — Under an identical declared execution plan, chunked prefill without reuse
and chunked prefill with prefix KV reuse are bit identical at every prefix length
tested, including at and above the sliding-window boundary.

**H1** — At or above the sliding-window boundary a reproducible difference appears
between the two plan-identical paths.

The experiment is designed to give H1 the best chance it can be given.

## 3. Environment (frozen)

| Item | Value |
| :-- | :-- |
| Commit SHA | `750be38377d11b1285b54f48e6403c9959e78ae9` |
| Working tree at registration | clean (`git status --porcelain` empty) |
| Model | `mlx-community/gemma-3-4b-it-4bit` |
| Model revision | `93724907d4ed1745d2fe50baadf3b0b01a65abf2` |
| `model.safetensors` SHA-256 | `94d3d701367d78584a9334ca00672b1c86e4aefa6a94167556c0485381e74af3` |
| Quantisation | 4 bit, `group_size = 64` (unchanged, never touched) |
| MLX | `0.32.0` |
| mlx_lm | `0.31.3` |
| macOS | `26.5.2` |
| Hardware | Apple M1 Max, 32 GB unified memory, 32 GPU cores |
| Power | AC required; the run aborts on battery |
| Layers | 34 |
| Sliding window | **1024** |
| Sliding-window pattern | 6 |
| Sliding layers | all except `{5, 11, 17, 23, 29}` — 29 of 34 |
| Global layers | `{5, 11, 17, 23, 29}` — 5 of 34 |

Environment is re-captured at run time into `research/raw/E12_environment.json` and
must match the table above on the model, quantisation, MLX and hardware fields, or
the run aborts.

## 4. Execution plans under test

**Plan `chunked@L`** — prefill is issued as exactly two forwards: tokens `[0, L)`
into a fresh fixed-shape KV cache, then tokens `[L, N)` into the same cache. The
split point `L` is a function of the declared prefix length alone and never of cache
state.

**Plan `single_shot`** — prefill is issued as one forward over `[0, N)` into a fresh
standard mlx_lm cache, then converted to the fixed layout. This is the plan the
untuned path uses.

Decode is identical in all arms: greedy, one token per forward, compiled fixed-cache
body, same capacity.

**Chunk size** — the only chunk boundary in `chunked@L` is `L` itself. There is no
secondary internal chunking; each chunk is issued as a single forward. Alignment
effects are covered by choosing `L` values that are and are not multiples of 64, 256
and 1024 (see §6).

**Capacity** — `capacity = ceil64(L + max_suffix_tokens + max_new_tokens)`, computed
once per case and used identically by all three arms of that case. Capacity is
therefore never a difference between arms.

## 5. Arms and comparisons

Two comparisons, kept strictly separate. They are never mixed and never averaged.

### Comparison A — the correctness test (decides everything)

| | |
| :-- | :-- |
| Baseline | `chunked@L`, prefix recomputed cold for every request |
| Candidate | `chunked@L`, prefix served from a KV snapshot taken once |
| Plan | **identical** in both arms |
| Tolerance | **none** |

Only Comparison A decides whether prefix KV reuse is plan-internally correct.

### Comparison B — plan divergence (documentation only)

| | |
| :-- | :-- |
| Baseline | `single_shot` |
| Candidate | `chunked@L`, no reuse |
| Purpose | quantify the numeric difference between two *different* plans |

A difference in Comparison B does **not** refute prefix KV reuse and is never
reported as one. It is recorded separately under the label `PLAN_DIVERGENCE`.

## 6. Prefix lengths

Measured in **tokens actually passed to the model**, never characters or words. The
realised token length is logged for every case and asserted against the intended
length before measurement.

| L | Rationale |
| --: | :-- |
| 276 | the exact E9 prefix length — direct comparability |
| 768 | aligned: 12×64, 3×256; whole request stays below the window |
| 870 | unaligned; prefill and suffix stay below 1024, **decode crosses it** |
| 896 | aligned to 64 but not to 256; prefill+suffix crosses the window |
| 1000 | unaligned, immediately below the window |
| 1023 | **immediately below the combined chunk/window boundary** |
| 1024 | **exactly the sliding-window boundary**, also 16×64 and 4×256 |
| 1025 | **immediately above the combined chunk/window boundary** |
| 1048 | unaligned, just above the window |
| 1152 | aligned to 64, not to 256, comfortably above the window |
| 1280 | aligned: 20×64, 5×256 |
| 1536 | aligned: 24×64, 6×256, 1.5× window |
| 2048 | aligned: 32×64, 8×256, 2× window |

Boundary-aligned lengths: 768, 1024, 1280, 1536, 2048.
Deliberately unaligned lengths: 870, 1000, 1023, 1025, 1048.
Immediately before and after the combined boundary: 1023 and 1025.

## 7. Prefix types

Two types, both run over the full length list.

1. **`natural`** — a realistic natural-language system preamble, long enough that
   2048 tokens can be sliced out of it.
2. **`synthetic`** — a deterministic, token-controlled sequence built from a fixed
   repeating pattern of ordinary vocabulary tokens, so that token positions are
   exactly known and reproducible.

Neither type uses special tokens, control tokens or invalid token sequences. Those
are explicitly out of scope for E12.

**Prefix construction** — the prompt is rendered through the normal chat template
with the preamble first and the request text second. The prefix is defined as the
first `L` tokens of the resulting token stream. Because all requests of a case share
the same preamble, the first `L` tokens are identical across the corpus by
construction. This is asserted per request before measurement:
`full_ids[:L] == prefix_ids` for all 12 requests, else the case aborts.

## 8. Request corpus (frozen)

The twelve requests used in E8, E9 and E10, unchanged, so results stay directly
comparable. Each is a distinct evidence block plus a fixed output contract, drawn
from a rotating selection over twelve candidate identifiers.

The corpus is frozen at registration. Nothing is added, removed or reordered after
results are seen.

Coverage of the boundary cases required of the corpus is obtained through the prefix
length grid rather than by adding requests:

- ends shortly before the window: `L = 768`
- crosses the window during prefill: `L = 896` and above
- crosses the window during decode: `L = 870`
- different response lengths: the corpus already yields 13 to 23 tokens per request

## 9. Decoding settings

| Setting | Value |
| :-- | :-- |
| Sampling | greedy, pure `argmax`, `temperature = 0` |
| Seed | not applicable — no stochastic operation is in the measured path |
| Max new tokens | 32 |
| Stop | first EOS in `{1, 106}` |
| Batch | 1 |
| Warmup | 2 full passes per arm before any measured pass |
| Synchronisation | `mx.eval` followed by `mx.synchronize` before every timer stop and before every comparison, because MLX evaluates lazily |

## 10. Comparison rules for Comparison A

`max |delta| = 0.0000` printed to four decimals is **not** accepted as evidence.
Equality is tested bitwise. All of the following must hold for a case to PASS:

1. **Token IDs** — identical sequence, element by element.
2. **Token count** — identical.
3. **Stop reason** — identical (EOS id, or length cap).
4. **Logits** — `mx.array_equal` true at **every** decode step, on the full
   262144-wide vector, not a reduction of it.
5. **KV cache** — for every layer, the *logically valid* region
   `[..., :offset, :]` of keys and values compares equal, checked after prefill and
   after the final decode step. Padding beyond `offset` is uninitialised and is
   excluded from every comparison and every hash.
6. **Positions** — the cache offset after prefill and after each decode step is
   identical.
7. **Attention masks** — the global mask and the sliding-window mask produced by the
   cache are compared bitwise, for the prefill width and for decode width 1.
8. **Sliding-window decisions** — the number of positions admitted by the sliding
   mask is recorded per arm and must match. This fixed-shape cache performs no
   eviction; that fact is recorded rather than assumed, by logging admitted-position
   counts on both sides.

On PASS the record stores `exact_equal = true`, `hash_baseline`, `hash_candidate`,
where each hash is SHA-256 over the concatenated raw bytes of the valid KV region
across all 34 layers, keys then values.

On FAIL the record stores, at minimum: first failing request index, prefix length,
prefix type, layer, tensor name, token position, decode step, first differing flat
index and both values, maximum absolute difference, maximum relative difference,
the affected token decision, and the logit gap between the two leading candidates
at that step.

**No tolerance and no fallback exists for Comparison A.** A case is bit identical or
it is not.

## 11. Execution plan

**Stage 1 — screening.** All prefix lengths × both prefix types × the full 12-request
corpus, in one fresh process.

**Stage 2a — confirmation on full PASS.** Repeat in three independent fresh
processes, at lengths `1023`, `1024`, `1025`, `1280`, `2048`, and the first
boundary-aligned length above the window (`1280` doubles as this; `1152` is added so
an unaligned above-window length is also confirmed).

**Stage 2b — confirmation on FAIL.** The broad matrix stops immediately. Then:
isolate the smallest reproducing case; repeat it in at least three fresh processes;
vary exactly one dimension at a time; find the smallest prefix length at which the
failure reproduces; and determine whether the cause binds to the sliding window, a
chunk boundary, masking, positioning or cache eviction.

A single difference that does not reproduce across fresh processes is recorded as
`UNRESOLVED` and is **not** treated as a refutation.

## 12. Result classification

Exactly one primary class is assigned:

| Class | Meaning |
| :-- | :-- |
| `PLAN_INTERNAL_EXACT` | Comparison A is bit identical in every confirmed case |
| `DOMAIN_RESTRICTED` | Comparison A is bit identical only within clearly bounded prefix/chunk/window conditions |
| `PLAN_INTERNAL_FAILURE` | Comparison A reproducibly differs under an identical plan |
| `HARNESS_OR_STATE_FAILURE` | the difference comes from uncontrolled state, faulty instrumentation or incomplete synchronisation |
| `INSUFFICIENT_EVIDENCE` | the test could not be completed dependably |

Comparison B is documented separately as `PLAN_DIVERGENCE` with its measured
magnitude. It never changes the primary class.

## 13. Success criteria

The experiment **succeeds as an experiment** when a primary class is assigned on
evidence, whichever class that is. Refuting H0 is as valid an outcome as failing to.

- H0 survives only if Stage 1 shows zero Comparison-A differences **and** Stage 2a
  reproduces that across three fresh processes.
- H1 is accepted only if a difference reproduces in at least three fresh processes
  with one dimension varied at a time.
- Anything else is `UNRESOLVED` or `INSUFFICIENT_EVIDENCE`.

## 14. Abort criteria

The run stops and preserves partial evidence, without retrying, if any of these
occur:

- another local model process is detected (`gpu_busy()`), before or between stages
- power source is not AC
- resident memory exceeds 12 GiB, or MLX peak exceeds 12 GiB
- any single stage exceeds 45 minutes of wall time
- the tokenisation gate fails for any request of a case
- realised prefix token length differs from the intended length
- the environment capture disagrees with §3 on model, quantisation, MLX or hardware

## 15. Secondary measurements

Recorded but never optimised for, and never allowed to influence any correctness
decision: cold TTFT, reuse TTFT, end-to-end time, MLX peak memory, cache build cost,
cache hit cost.

**No implementation is changed during E12 to make E12 faster.**

## 16. Known risks

1. **The test may be trivially satisfiable.** Comparison A's candidate replays a
   snapshot of exactly the computation its baseline performs, so if the prefill
   forward is deterministic and the snapshot is not aliased, equality follows almost
   by definition. The genuinely informative failure modes are therefore buffer
   aliasing under `mx.compile` donation, and any window- or offset-dependent
   behaviour that differs between a freshly built cache and a restored one. This
   risk is accepted and named rather than engineered away, because a test that
   cannot fail is worth knowing about.
2. **Padding contamination.** The fixed cache allocates to `capacity`; only
   `[:offset]` is meaningful. Including padding in a hash would produce either false
   failures or false passes. Every comparison slices to `offset` first.
3. **Cost.** Prefills at 2048 tokens are roughly 3.4 s each; the full screening
   matrix is estimated at 30 to 40 minutes of GPU time. The estimate may be wrong;
   the 45-minute abort applies per stage.
4. **Chat-template realism.** Defining the prefix as the first `L` tokens of the
   rendered stream means a prefix may end mid-sentence. That is intentional — it
   makes the token boundary exact — but it means the prefix is not always a
   semantically complete instruction block.
5. **`RotatingKVCache` is not exercised.** The `single_shot` arm converts a standard
   mlx_lm cache, whose sliding layers use `RotatingKVCache`, into the fixed layout.
   Above 1024 tokens that cache rotates while the fixed cache does not. This is a
   real difference between the plans and is expected to widen Comparison B above the
   window. It is a Comparison B observation only and must not be read as a
   Comparison A failure.
6. **Single hardware, single model.** Whatever the outcome, it is a statement about
   this model on this machine under this MLX build. No universal claim is permitted.
