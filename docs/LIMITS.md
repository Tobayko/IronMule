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

**B27 D1 is not a runtime optimization or selected profile.** It adds an immutable,
stdlib-only evidence contract that current runtime modules do not import. The sealed
B27d post-change engineering screen preserved exact outputs/resources and passed its
12B regression gates, but both 4B arms were common-mode 5.7–6.4% slower and crossed the
preregistered 5% intervals. Its result is therefore
`INCONCLUSIVE_POTENTIAL_REGRESSION`, not proof that D1 is performance-neutral or that
the slowdown is causal. The separate B27e mirrored cross-commit control then found D1
within 2% in OLD/D1 order and 5.8–7.9% apparently faster in D1/OLD order. Its frozen
class is `ORDER_OR_TEMPORAL_DRIFT`: a consistent D1 slowdown was not reproduced, but
neutrality is still unproved. D1 remains non-imported and unactivated.

## Apple-Silicon inference claims

Apple's M4/M5 comparison is evidence for that specific MacBook Pro setup and
workload: it reports 120 -> 153 GB/s memory bandwidth and 19--27% subsequent-token
generation improvement, while TTFT benefits from the M5 GPU Neural Accelerators.
It must not be turned into a universal M5 or MLX multiplier:
[Apple ML Research, MLX and the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).

The M5 GPU Neural Accelerators are not the separate Apple Neural Engine (ANE).
The former are reached through the GPU/Metal ML path used by that MLX report; ANE
deployment is a separate Core ML decision. IronMule has no ANE kernel path and
does not infer one from MLX:
[Apple Core ML documentation](https://developer.apple.com/documentation/coreml).

External BaseRT, vllm-mlx and FusionML publications are design context, not
IronMule measurements or guarantees:
[BaseRT](https://arxiv.org/abs/2607.00501),
[vllm-mlx](https://arxiv.org/abs/2601.19139), and
[FusionML](https://arxiv.org/abs/2607.22785). Their reported speedups cannot be
copied into a runtime target without the same model, quantisation, chip,
framework, workload and correctness protocol.

The local E4/E5 results remain the relevant evidence for this repository: the
achieved GEMV bandwidth depends on matrix size, and projection fusion did not
produce a robust decode win. B35's 12B observation was order-sensitive and
inconclusive; that narrow statement is superseded by B36 only for the exact
12B revision, 322/32 workload, M1 Max host, default wired/cache policy and
full-hash/prefault protocol. B36 qualifies the core profile under those
conditions but does not activate it or generalize it. These are recorded in
[`research/LEDGER.md`](../research/LEDGER.md) and
[`research/raw/B35_review.md`](../research/raw/B35_review.md) and
[`research/raw/B36_review.md`](../research/raw/B36_review.md).

No global 60--70% bandwidth-efficiency constant is validated here. The phase
diagnostic therefore requires per-run measured inputs and never clamps an
efficiency above one; such a value is only an input-consistency warning. No
zero-allocation decode-loop claim, M5 speed claim, Metal-kernel claim, or ANE
claim follows from the diagnostic or from B35.

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

**Qwen compatibility is narrowly qualified by X2.** The validated Gemma all-KV
legacy path is retained. Qwen compatibility covers only revision
`3e6447f082e89cc7f0bc6e5441afd38dfce760ff`, the strict/greedy path, and workloads
up to 6 concurrent requests with the tested 2/3-request × 8-token and 6-request ×
48-token gates. `ArraysCache` with
non-`None` `lengths` or `left_padding` is rejected, as is hybrid speculation;
these are fail-closed boundaries. Qwen performance is unqualified. The compiled
tiny gate was exact, but its `30.76 GB` peak is a memory warning, not a speed claim.

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
