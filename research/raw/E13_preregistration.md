# E13 — Does the execution plan change measurable answer quality?

**Experiment ID** `E13`
**Frozen at commit** `ad8815f6729b8b2645ff88eb667547e961eac158`
**Branch** `forge/hardware-autotune`
**Registered** 2026-08-25, before any E13 quality measurement was taken or viewed.

Frozen. Any necessary change gets a new experiment ID (`E13b`, `E14`, …), never an
edit here. Scoring, margin, data selection and stop rules are not adjusted after
results are seen.

---

## 1. Research question

E9 and E12 established that two execution plans of the same unmodified model are
each internally exact but disagree with one another: up to `25.5` logits apart, and
**140 of 756 requests answering with different tokens**. Different is not worse.

> Does the choice of execution plan change objectively measurable answer quality,
> and does any effect depend on context length or on the 1024-token sliding-window
> boundary?

**Neither plan is treated as the reference or as the correct one.** The faster plan
is not assumed innocent, and the incumbent plan is not assumed right.

## 2. Hypotheses

**H0 (non-inferiority target)** `ReusableSessionPlan` is not worse than
`StrictOneShotPlan` by more than the preregistered margin.

**H1** One plan is measurably worse, or the difference depends on context length.

A non-significant difference is **not** evidence of equivalence and will never be
reported as such. Only the non-inferiority test below can support a
"not worse" claim.

## 3. Plans under comparison (unchanged, not modified for E13)

| Plan | Definition |
| :-- | :-- |
| `StrictOneShotPlan` | `Engine._prefill` with `prefix_cache = None`: the whole prompt in one forward into a standard cache, converted to the fixed layout |
| `ReusableSessionPlan` | `Engine._prefill_chunked` with `PrefixCache(document_prefix_ids)`: chunk one is the document, served from the snapshot after the first request of the context; chunk two is the question |

Knobs are the machine's stored tuned profile in both arms and are identical:
`compiled_fixed_cache=True`, `head_skip_prefill=True`, `fuse_projections=True`,
`readback_every=1`, `fused_argmax=False`, `speculate_k=0`, `wired_fraction=0.0`.
`fused_argmax` is off in both arms because the secondary metrics need logits; it was
already rejected by the tuner in E11, so this is the profile, not a change.

Model, revision, quantisation, prompt template, greedy decoding and maximum output
length are identical between plans. Nothing in `forge/` is modified for E13.

## 4. Dataset

**SQuAD v1.1 development set**, fetched 2026-08-25 from
`https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json`,
SHA-256 `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`.
48 articles, 2067 paragraphs, 10570 questions. Human-written questions with
human-annotated extractive answers, so **ground truth is independent of this model
and of the experimenter.**

Chosen because it is natively session-shaped: one natural document with many
independent questions answerable from it, which is exactly the shape
`ReusableSessionPlan` exists for.

### Selection rule (mechanical, content-blind)

1. Articles sorted by `title`, ascending.
2. Indices 0, 1, 2 are **reserved for the pilot** and excluded from the evaluation
   set: `1973_oil_crisis`, `Amazon_rainforest`, `American_Broadcasting_Company`.
3. Remaining articles are assigned to a length band by position, `i mod 3` →
   `SHORT`, `NEAR`, `LONG`. Assignment is by index only and never by content or
   length.
4. Within an article, paragraphs are concatenated in document order, separated by a
   blank line, until the rendered document prefix first falls inside the assigned
   band. The first paragraph count that lands in band is taken.
5. Questions are the first 8 `qas` of the included paragraphs, in document order.

### Exclusions (preregistered)

- An article whose paragraph accumulation overshoots its band without landing
  inside it is excluded and logged.
- A context with fewer than 8 questions is excluded and logged.
- Both exclusions are content-blind and applied before any measurement.

By the rule above this yields **44 contexts, 352 questions**: `SHORT` 15, `NEAR` 14,
`LONG` 15. One exclusion is expected (`Doctor_Who`, band `NEAR`, overshoots).

## 5. Length bands

Measured in tokens actually passed to the model, logged per context. The prefix is
the rendered chat-template text up to and including the document, i.e. everything
the two plans share.

| Band | Prefix tokens | Relation to the 1024 sliding window |
| :-- | :-- | :-- |
| `SHORT` | 512 – 899 | entirely below the window |
| `NEAR` | 900 – 1150 | **straddles it**; some contexts below 1024, some at or above |
| `LONG` | 1151 – 2048 | entirely at or above the window |

Bands are disjoint and are set at paragraph granularity (~180 tokens), because a
window narrower than one paragraph cannot be hit reliably.

## 6. Prompt, decoding and output

Identical in both arms:

```
Read the document and answer the question using the shortest exact span from the
document. Give only that span, with no explanation and no full sentence.

Document:
<document>

Question: <question>
```

Greedy, `temperature = 0`, no sampling, no seed in the measured path.
`max_new_tokens = 24`. Stop at first EOS in `{1, 106}`.
Capacity is `ceil64(longest prompt in the context + 24)`, computed once per context
and identical for both plans.

## 7. Primary metric

**Containment accuracy.** A prediction is correct when the normalised token
sequence of **any** gold answer occurs as a contiguous subsequence of the normalised
token sequence of the prediction.

Normalisation is the SQuAD official procedure: lowercase, remove punctuation,
remove the articles `a`, `an`, `the`, collapse whitespace. Comparison is on token
sequences rather than raw substrings, so a gold answer cannot be matched inside a
longer word.

Containment rather than strict exact match because an instruction-tuned model
wraps spans in prose; the 24-token cap bounds how much a prediction can hedge.
Its known weakness — a prediction listing several candidates — is measured by a
preregistered scorer control and reported.

**Accuracy is computed per context** (8 questions each) and then averaged over
contexts, so every context weighs equally and the point estimate matches the
clustering used for uncertainty.

## 8. Uncertainty and non-inferiority

Questions from the same context are dependent. Uncertainty is therefore computed at
the **context level** by a **paired cluster bootstrap**: for each context `c`,
`d(c) = accuracy_reusable(c) − accuracy_strict(c)`; contexts are resampled with
replacement **10,000** times, seed `20260825`; the reported interval is the
2.5th to 97.5th percentile of the resampled mean of `d`.

No McNemar test over pooled questions is reported, because it would treat 352
dependent observations as independent.

**Non-inferiority margin `δ = 0.05`** absolute accuracy, one-sided, fixed before any
data was seen. Justification: the plan buys a measured `-37.8%` session time; a loss
of more than five accuracy points on extractive question answering would be a
material regression rather than a rounding effect, and 44 clusters give an expected
interval half-width near 3 points, so a 5-point margin is resolvable rather than
decorative.

## 9. Decision rule (ordered, frozen)

Evaluated top to bottom; the first matching rule assigns the class.

| # | Condition | Class |
| --: | :-- | :-- |
| 0 | Any band's interval lies entirely below 0 **and** another band's lies entirely above 0 | `LENGTH_OR_TASK_DEPENDENT` |
| 1 | `CI_lower(d) > 0` | `REUSABLE_BETTER` |
| 2 | `CI_upper(d) < 0` | `REUSABLE_WORSE` |
| 3 | `CI_lower(d) > −δ` | `REUSABLE_NONINFERIOR` |
| 4 | otherwise | `INCONCLUSIVE` |

Rule 0 is checked first because a length-dependent effect makes any single pooled
statement misleading regardless of what the pooled interval shows.

## 10. Interim analyses and stop rules

**There are no interim analyses.** The frozen evaluation set is measured in full and
analysed exactly once. No result is inspected before the set completes.

If the outcome is `INCONCLUSIVE`, the evaluation set is **not** extended within E13.
An extension requires a new experiment ID with its own preregistration. This
deliberately trades statistical efficiency for the absence of any multiplicity
adjustment or optional-stopping bias.

Abort conditions, which preserve partial evidence and are never retried: another
local model process detected (`gpu_busy()`), power source not AC, MLX peak above
12 GiB, or wall time above 60 minutes for the main run.

## 11. Secondary metrics (analysed only after the primary)

Reported but never used to assign the primary class:

- accuracy and paired difference per length band, with the same cluster bootstrap
- strict exact match and token-F1, so it is visible whether the conclusion depends
  on the choice of primary metric
- negative log-likelihood of the first gold answer under teacher forcing, per plan.
  This is more sensitive than a binary score and can reveal a systematic shift that
  accuracy cannot
- answer divergence rate: fraction of questions whose generated token sequences
  differ between plans, and the index of the first differing token
- top1−top2 logit gap at the first differing position
- TTFT, session wall time, and MLX peak memory per plan

**Logit distance is not a quality measure** and is excluded from every quality
decision. It appears only as a mechanism observation.

## 12. Harness and scorer validation (before the main run)

A pilot over the three reserved articles validates the harness end to end. Its
results are **never** interpreted as evidence about the plans.

The scorer is validated against fixed control cases with known expected outcomes:

| Control | Expected |
| :-- | :-- |
| Deliberately wrong answer | incorrect |
| Gold answer with different case | correct |
| Gold answer with trailing punctuation | correct |
| Gold answer wrapped in prose | correct |
| Gold answer with a leading article | correct |
| Gold answer with markdown emphasis | correct |
| Empty prediction | incorrect, flagged as format failure |
| Gold string inside a longer word | incorrect |
| Shotgun listing several candidate answers | recorded honestly whichever way it scores, and reported as the bound on containment |

Raw predictions are stored for every question of every plan, always.

**One permitted pilot adjustment, declared in advance:** if pilot containment
accuracy is above `0.95` or below `0.35` in either plan, the answer instruction
and/or `max_new_tokens` may be adjusted **once**, before the evaluation set is
frozen, to move the task off the ceiling or the floor. Such an adjustment is logged.
No other property may change after the pilot, and no adjustment is permitted after
main-run results are seen.

## 13. Known risks

1. **Contamination.** SQuAD v1.1 is public and likely present in the model's
   training data. This inflates absolute accuracy in both arms. It does **not** bias
   the paired difference, which is the primary metric, because both plans see
   identical questions.
2. **Containment is gameable by long predictions.** Bounded by the 24-token cap and
   measured by the shotgun control. Strict exact match is reported alongside.
3. **Ceiling effect.** If both plans answer nearly everything correctly, the design
   loses power to detect a small difference. The pilot checks for this, and a
   ceiling result will be reported as a limitation rather than as equivalence.
4. **44 clusters is a modest sample.** The expected half-width near 3 points can
   exclude a 5-point loss but cannot exclude a 1-point loss. The claim will be
   bounded to what the interval actually supports.
5. **Single model, single machine, single task family.** Extractive question
   answering on Wikipedia prose is one task type. No claim is made about
   summarisation, reasoning, code, or any other workload.
6. **Both plans may be wrong together.** Agreement between plans is not evidence of
   correctness; only the human gold answers decide correctness here.
