# B27 Phase D contract proposal — approval package

**State:** D1 explicitly approved and implemented on 2026-08-28. B27d post-change
measurement completed as `INCONCLUSIVE_POTENTIAL_REGRESSION`; B27e did not reproduce a
consistent D1 slowdown but remained `ORDER_OR_TEMPORAL_DRIFT`. No neutrality claim or
activation. See
[`B27_D1_IMPLEMENTATION.md`](B27_D1_IMPLEMENTATION.md).

This document freezes the smallest architecture decision that can move IronMule
toward an evidence-driven execution layer without changing execution. It is subordinate
to `AGENTS.md`, the B27 backlog entry, the current runtime contracts and the user's
approval. The approval covers D1 only. No persistence, profile mutation, route or
model behavior follows from it.

## Recorded decision

The user approved **D1: a stdlib-only evidence contract** consisting of one pure
module and unit tests. D1 has no MLX import, persistence, strategy selection, runtime
integration, automatic routing, activation, model download, dependency or hardware
measurement by itself.

The recorded approval is D1 only. Persistence, stock-`mlx_lm` comparison,
regression promotion, shadow recommendation and runtime selection remain separate
decisions.

## D1 scope fixed in advance

### Start scope

Represent, without executing or promoting, the already-existing exact paths:

1. strict one-shot + baseline knobs + Interactive mode;
2. strict one-shot + baseline knobs + Throughput W4;
3. strict one-shot + qualified core knobs + Interactive mode;
4. strict one-shot + qualified core knobs + Throughput W4;
5. reusable-session plan as a separate semantic strategy;
6. Qwen hybrid-cache strict compatibility as compatibility evidence only, never a
   performance qualification.

No new execution candidate, cache, scheduler, model family or approximation class is
introduced.

### Closed action space

D1 may encode only explicit values already supplied by a caller or imported from an
existing artifact. It cannot generate code, change knobs, select a plan/mode, update a
profile or assign itself a terminal status.

### Primary endpoint and guardrails

The protected performance endpoint remains complete service `outer_wall_ms`, with
physical token rate as the paired rate endpoint. Correctness and resource guardrails
remain token IDs, stop reason, physical/visible count, deterministic/final state when
available, fallback, MLX peak, RSS, swap, crash and timeout. D1 stores these names; it
does not evaluate or weaken them.

### Autonomy boundary

D1 is offline representation only. It cannot recommend, shadow-route, canary, promote,
rollback or mutate runtime state. The deterministic existing baseline remains the only
fallback.

### Dependencies

Python standard library only. No package, model, local AI, service or network access.

## Exact type contract

All records are immutable and JSON-serializable. Every record carries an exact schema
identifier and a canonical SHA-256 digest. Missing mandatory identity is an error;
unsupported future schema versions fail closed.

### `EvidenceStatus`

The only cross-experiment lifecycle values are:

- `HYPOTHESIS`
- `QUALIFIED`
- `REJECTED`
- `INCONCLUSIVE`
- `INVALIDATED`
- `REVALIDATION_REQUIRED`

Experiment-specific diagnostic verdicts remain in a separate `diagnostic_verdict`
field and never masquerade as a lifecycle status.

### `ExecutionStrategy`

Mandatory fields:

| Field | Contract |
| --- | --- |
| `schema` | exact `ironmule.execution_strategy.v1` |
| `strategy_id` | canonical digest, derived rather than caller-trusted |
| `semantic_class` | `exact` initially; approximation is not admitted by D1 |
| `prefill_policy` | named existing prefill arrangement |
| `decode_policy` | named existing greedy decode path |
| `cache_policy` | native/fixed/hybrid type and capacity semantics |
| `scheduling_policy` | sequential or existing ready-order/fair-rotation policy |
| `grouping_policy` | independent batch-1 width or explicit no-grouping |
| `synchronization_policy` | per-step or existing grouped completion boundary |
| `memory_policy` | fixed capacity, ceiling and wired/cache policy identity |
| `compile_graph_policy` | baseline, compiled fixed-cache and head-skip/fusion identities |
| `prefix_reuse_policy` | none or declared reusable-session boundary |
| `plan_kind` | exact existing plan kind |
| `service_mode` | exact existing mode name |
| `knobs_key` | canonical `Knobs.key()` value, supplied as data |
| `implementation_revision` | runtime-tree or code digest |

The strategy record references existing behavior. It contains no function, source
string, callable, file path or executable payload.

### `ValidityDomain`

Mandatory exact identity:

| Group | Fields |
| --- | --- |
| Hardware | Apple chip, machine, RAM bytes, GPU cores/configuration |
| OS/framework | macOS build, Python, MLX, mlx-lm, IronMule schema/runtime version |
| Model | model ID, exact revision, manifest digest, architecture, tokenizer digest |
| Quantisation | bits, group size, format/mode where available |
| Cache | cache family and layer pattern, capacity policy |
| Workload | plan, prompt/context bucket, output bucket, concurrency, arrival pattern, workload class |
| System | AC/battery, low-power state, thermal state when publicly observable, swap/preflight class |

Identity fields match exactly. Workload buckets are explicit closed intervals stored in
the record; D1 does not reuse the current tuner's implicit `25%` heuristic. A missing
current field never means wildcard. Unknown current identity yields
`REVALIDATION_REQUIRED`.

### `EvidenceRecord`

Mandatory fields:

| Group | Fields |
| --- | --- |
| Identity | schema, evidence ID/digest, strategy IDs, validity-domain digest |
| Experiment | experiment/study ID, preregistration digest, reviewer record digest, evaluator identity |
| Provenance | code/runtime-tree, model manifest/revision, environment and workload digests |
| Comparison | baseline strategy, candidate strategy, reference definition |
| Samples | immutable content hashes/references for every warmup and measured raw sample |
| Performance | TTFT definitions, prefill, decode, outer wall, physical/visible rate, p50/p95 and spread/CI where defined |
| Resources | MLX active/peak, RSS, swap, timeout, crash, worker/fallback state |
| Correctness | comparison performed, token/byte identity, stop/count, state/cache identity, determinism and quality class |
| Statistics | repeats, pairing/order, estimator, uncertainty method and interval |
| Lifecycle | evaluator-owned `EvidenceStatus`, diagnostic verdict, reason and timestamp |

Absolute local paths and prompt contents are forbidden. Raw references use artifact
IDs plus content hashes. A summary without raw samples is explicitly
`evidence_quality=summary_only` and cannot silently become equivalent to raw evidence.

### `TrustedExecutionProfile`

Mandatory fields:

- exact strategy ID;
- exact validity-domain digest;
- one or more `QUALIFIED` evaluator-owned evidence IDs;
- protected baseline evidence ID;
- creation and last-revalidation evidence IDs;
- current lifecycle status;
- no embedded executable payload.

Construction fails unless every evidence record is `QUALIFIED`, refers to the same
strategy/domain, contains required correctness/resource evidence and is not
invalidated. `match(current_domain)` returns either exact match or
`REVALIDATION_REQUIRED`; D1 exposes no `run()` or `select()` method.

## Status ownership and transition rules

```text
Researcher: HYPOTHESIS only
          |
          v
Reviewer: review record (cannot qualify)
          |
          v
Evaluator: QUALIFIED | REJECTED | INCONCLUSIVE
              |
              +-- domain/code/model drift --> REVALIDATION_REQUIRED
              |                                  |
              |                                  +-- new evaluator evidence --> QUALIFIED
              |                                  +-- failed revalidation ----> INVALIDATED
              +-- later contradictory evidence -----------------------------> INVALIDATED

Runtime: may read exact-match QUALIFIED profiles only; never changes status
```

`REJECTED`, `INCONCLUSIVE` and `INVALIDATED` records are immutable. A later experiment
creates a new EvidenceRecord linked by `supersedes` or `revalidates`; it never edits the
old result. A Researcher or Candidate cannot be its own terminal Evaluator.

## Canonicalization and trust boundary

- UTF-8 JSON, sorted object keys, no NaN/Infinity, booleans distinct from integers;
- timestamps normalized to UTC and excluded from semantic IDs where appropriate;
- digests computed with the digest field omitted;
- unknown top-level fields rejected in D1 rather than ignored;
- diagnostic extensions live in an explicit namespaced mapping and never affect
  trust unless a future schema version says so;
- all constructors validate field types, non-empty exact IDs and finite non-negative
  measurements;
- deserialization treats every artifact as untrusted input.

## Existing-artifact adapter boundary

D1 may provide pure adapter functions that accept dictionaries already extracted from:

- `Knobs.as_dict()` / `Knobs.key()`;
- `plan_kind()` and the explicit mode name;
- `fingerprint.build()` output;
- B27a2 path-free public summary;
- later, separately approved evidence import.

Adapters return a proposed record plus validation errors. They do not import
`runtime.py`, `service.py`, `plans.py`, MLX or the model. Current runtime modules do not
import the D1 module. This keeps observable execution unchanged.

## D1 verification gates

1. Mandatory-field and type tests for every record.
2. Canonical serialization and digest stability across key ordering.
3. Boolean/number, NaN/Infinity, empty-ID and unknown-field rejection.
4. Exact revision, manifest, architecture and quantisation required.
5. Domain mismatch and missing current identity yield `REVALIDATION_REQUIRED`.
6. Only evaluator-owned `QUALIFIED` evidence constructs a trusted profile.
7. Rejected/inconclusive/invalidated/self-evaluated evidence cannot construct one.
8. Existing plan/mode/knob IDs survive adapters unchanged.
9. Static import test proves no current runtime module imports D1 and D1 imports no MLX.
10. Full serial non-integration and existing 4B integration suites remain green.
11. `git diff --check`, ProjectAtlas refresh/runtime/MCP checks remain green.
12. A separately sealed post-change 4B/12B baseline records token identity,
    fallbacks, memory, swap and raw variation; it makes no new qualification claim.

## D1 kill/pivot criteria

Stop D1 and retain current deterministic behavior if any of these occurs:

- an existing qualified path cannot be represented without a parallel executor/cache;
- exact plan/mode semantics must be weakened or merged;
- missing identity must be treated as wildcard;
- a candidate/researcher must assign its own terminal status;
- record construction needs MLX/model import or executable payloads;
- the type layer changes runtime imports, routing, fallback or measured behavior;
- public/private evidence separation cannot be preserved.

## Explicitly outside this approval

- SQLite/JSONL Optimization Memory and migrations;
- importing all legacy evidence into canonical training records;
- performance regression thresholds or automatic promotion;
- fair stock-`mlx_lm` process arm;
- profiler producer, Metal counters or custom kernels;
- Researcher/Reviewer/Evaluator automation;
- learned ranking, GBDT, Bayesian optimization or any new dependency;
- shadow recommendation, runtime selection, automatic activation or rollback;
- new model/download, models above 12B or external service;
- approximate/quality-changing strategies.

## Completion position against the full objective

| Requested phase | Current evidence |
| --- | --- |
| A — repository/evidence understanding | completed by Atlas navigation, ledger/dead-end review and B27 inventory |
| B — current-main baseline | completed for existing 4B/12B cells; engineering scope explicitly bounded |
| C — gap analysis | completed in `B27_EVIDENCE_LAYER_GAP_ANALYSIS.md` |
| D — smallest architecture change | this D1 proposal; implementation awaits approval |
| E — post-change verification | not started; depends on approved D1 implementation |
| Later evidence-driven runtime | intentionally incomplete: persistence, regression evaluator, fair stock arm, shadow selection and revalidation remain future approved stages |

The active goal is therefore not complete at D1 proposal time.
