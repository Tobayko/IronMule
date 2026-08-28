# B27 D1 implementation and post-change result

## Outcome

D1 is implemented on commit `0b14eb6f134edc42701ebb1e1a85a1bd484d12d1` as
`ironmule/evidence.py`. The module is standard-library-only and remains absent from
the import graph of Runtime, package root, plans, modes, executors, tuner, benchmark,
telemetry and fingerprint. It has no persistence, `run()`, `select()`, routing,
promotion or activation path.

The post-change result is deliberately **not** upgraded to success:

`INCONCLUSIVE_POTENTIAL_REGRESSION / POTENTIAL_CODE_REGRESSION`

All correctness and resource gates passed. The 12B cell showed no regression. The 4B
cell was common-mode slower in both arms and crossed the preregistered 5% threshold.
D1 remains inert and unactivated while B27e is open to separate code presence from
temporal/system drift.

## Implemented contracts

- closed `EvidenceStatus` lifecycle:
  `HYPOTHESIS`, `QUALIFIED`, `REJECTED`, `INCONCLUSIVE`, `INVALIDATED`,
  `REVALIDATION_REQUIRED`;
- immutable canonical `ExecutionStrategy` records over existing caller-supplied
  plan/mode/knob/policy identities;
- exact `ValidityDomain` identity plus closed workload buckets;
- evaluator-owned `EvidenceRecord` with distinct Researcher/Reviewer/Evaluator roles;
- `TrustedExecutionProfile` constructible only through supplied qualifying records;
- deterministic UTF-8 canonical JSON and content-derived SHA-256 identifiers;
- pure adapters for existing path IDs, fingerprints and B27 summary evidence.

Fail-closed checks reject missing/unknown schema fields, NaN/Infinity, non-string JSON
keys, absolute local artifact paths, forged digests, unknown identity, self-evaluation,
summary-only qualification, missing samples/p50/p95/CI, incomplete resource gates and
profile deserialization without the actual qualified records.

## Static and correctness verification

- D1-focused: `15 passed`;
- final D1/baseline/comparison focus: `26 passed`;
- full serial non-integration: `146 passed, 11 deselected` in `5.21 s`;
- existing real Gemma-4B integration: `10/10` in `21.24 s`;
- no runtime module imports D1; D1 imports only stdlib;
- no model/download/install/network action during implementation tests.

## B27d protocol and bindings

Preregistration SHA-256:
`846e09499a0eb4f9ff531a6302da9c7913e8b6f620d6ad7834dcaf7fda44de36`.

Both post cells used runtime-tree SHA-256
`d7577af8e83778b9753ad4bf721656a16d923a9f848040e406178b7dcffc8a21`,
exact cached model revisions, baseline knobs, strict plan, six requests × 48 output
tokens, two warmups and six measured repeats per arm. Preflight memory was `83%` free,
AC, low-power false and swap `0 B`; both cells ended with swap delta `0 B`, no fallback,
no correctness error and no residual process.

Raw SHA-256:

- 4B: `10071669abb6c45871bf3d5eec0df3f37104341bb197394a840bf64e46a7be44`;
- 12B: `41d9bd16b179357ae1d99edf26abba135d1c2b8315bc5c47c421868f5b977a96`.

The path-free comparison is byte-reproducible with SHA-256
`ed2129005ab96df2a103808108c9c5fb0f63e871d7f33caace628e8ef7848c37`.

## Post/pre results

Ratios are post/pre; wall lower is better, rate higher is better. Gates were wall
95%-CI high `<=1.05` and rate 95%-CI low `>=0.95`.

| Model/arm | Wall median [95% CI] | Rate median [95% CI] | Gate |
| --- | ---: | ---: | --- |
| 12B Interactive | `1.0055 [0.9815; 1.0288]` | `0.9943 [0.9720; 1.0189]` | pass |
| 12B Throughput | `0.9995 [0.9864; 1.0224]` | `1.0006 [0.9788; 1.0136]` | pass |
| 4B Interactive | `1.0575 [1.0530; 1.0621]` | `0.9456 [0.9417; 0.9496]` | **miss** |
| 4B Throughput | `1.0643 [1.0571; 1.0676]` | `0.9396 [0.9366; 0.9460]` | **miss** |

The within-cell 4B grouping comparison moved far less: pre/post wall ratios were
`0.8437`/`0.8483`, and rate ratios `1.1852`/`1.1789`. Both absolute arms moved in the
same direction. Post load averages were also higher, while 12B remained unchanged.
These are diagnostics, not permission to override the frozen classification.

## Decision

- D1 code is retained because it is not on the runtime execution/import path and all
  correctness/resource gates pass.
- No performance-safety, qualification, routing or activation claim is made.
- B27d is not repeated or pooled.
- B27e is the next high-information experiment: mirrored fresh-process 4B runs on the
  pre-D1 and D1 commits using only new samples and explicit order controls.

Final handoff verification on the documented result state passed
`146 passed, 11 deselected` in `5.18 s` and the real Gemma-4B integration suite
`10/10` in `21.12 s`, with pre-integration swap `0 B` and no model process.
