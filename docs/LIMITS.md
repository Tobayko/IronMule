# Known limits and validity domain

Stated as measured, not as hoped. Anything not listed here was not tested.

## Validity domain of every performance number

| | |
| :-- | :-- |
| Model | `mlx-community/gemma-3-4b-it-4bit`, revision `93724907`, for every preregistered result. `gemma-3-12b-it-4bit` was measured once outside the preregistered series, see `X1` in the ledger |
| Quantisation | 4 bit, group size 64 |
| Framework | MLX `0.32.0`, mlx_lm `0.31.3` |
| Machine | Apple M1 Max, 32 GB unified memory, 32 GPU cores |
| Power | AC. Battery and low-power mode were never measured |
| Decoding | greedy, `temperature = 0` |
| Batch | 1 per execution; grouping never changes tensor shapes |
| Context | 276 to 2048 tokens |
| Concurrency | up to 8 requests |

No claim holds outside this box. Another model, another quantisation, another MLX
build or another machine requires re-measurement, which is what the fingerprint
exists to force.

**Two models outside the box have now been measured, and the gain is not
size-independent.** Under the strict plan, with three runs each and a realised width of
`4.00` throughout, the gain falls monotonically: `+19.24%` at 4B, `+15.42%` at 12B,
`+11.81%` at 27B. Peak memory grew `2.78 -> 7.80 -> 16.78 GB`; 27B fits this machine
with room to spare.

Those runs were not preregistered and are recorded as `X1` in the ledger, so they are
exploratory observations rather than validated results. They are stated here anyway
because the alternative — describing the domain as one model while three have been run,
and leaving the impression that the headline number travels unchanged to larger models —
would be less honest, not more careful.

All three are Gemma 3 at 4 bit on the same machine, so nothing here separates model size
from model family, and nothing here extends to another Mac.

## Limits of the runtime itself

**Group width is capped at 4.** Not a tuning knob: width 8 regressed in E14b and E2
and E3 independently, at the `M=8` kernel regime. Requesting more raises.

**One capacity per `serve()` call.** The KV cache is fixed-shape, so a batch is
allocated to the longest prompt in it. A single long request therefore inflates
memory for the whole batch. Capacity above 8192 is refused rather than allocated.

**Grouping helps only when groups fill.** With 2–3 token answers the realised width
fell to 1.83 and the gain fell below threshold in both E15 and E16. A service whose
queue empties faster than a group fills gains nothing.

**Median latency always worsens under grouping**, by 26% to 54% depending on
workload. A latency-sensitive single-request path should stay in `InteractiveMode`.

**Fallback discards work.** A failed group restarts its requests from prefill. That
is correct and wasteful; it is not an optimisation path.

**Ragged prompt lengths within a group are untested.** Every measured group used a
shared cache offset. Requests of genuinely different prefill lengths in one group
are outside what was verified.

**Sustained load is untested.** All measurements used eight requests. Queue effects
that only appear under continuous arrival are unknown.

## Limits of the evidence

**E13's quality bound is `1.14` accuracy points on a contaminated set.** SQuAD v1.1
is public and probably in the model's training data. The paired design rules out a
different task distribution between plans, but contamination raises model confidence
and therefore lowers the experiment's sensitivity to plan divergence. The bound may
be optimistic for unseen material.

**Only extractive question answering was tested for quality.** Summarisation,
reasoning, code and multi-turn were not.

**E16's frozen verdict is `CONFOUNDED_BY_PROCESS_STATE`**, assigned by two criteria
that are demonstrably misspecified: one anchored RSS growth to warmup instead of the
first repeat, the other was not scoped to the required workloads. Every instrument
that actually measures accumulation reads zero. The frozen class is reported
unchanged; the substantive reading is replication.

**Kernel counts are retired.** MLX exposes no machine-readable dispatch counter, so
no absolute dispatch time can be derived and none is claimed. `completion_wait` in
the telemetry is a wait, not GPU time: device work may already have begun during
submission.

**`W = 1` is not `SequentialService`.** They differ in interleaving as well as
synchronisation. E15 measured `G(W1) ≈ 0`, which is what makes the rest of the
comparison interpretable.

## Not implemented on purpose

- No adaptive controller. Realised width already adapts on its own, and nothing
  measured shows a controller beating a fixed width of 4.
- No true tensor batching. It diverged at batch 8 in E14b, one token, reproducibly.
- No speculative decoding. Acceptance was 0.17 per drafted token and decode was
  2.9x slower.
- No automatic plan selection. Plans change output; that is a caller decision.
