# D2b — Exact-model-identity implementation review

**Date:** 2026-08-29
**Status:** `READY_FOR_POST_CHANGE_MEASUREMENT`; no D2b timing observed.

## Reviewed scope

The user approved only exact model identity propagation through local resolution,
Runtime validity fingerprints and tuned-profile compatibility. D1 persistence,
strategy selection, plan/mode routing, automatic activation, downloads, new
dependencies and inference-semantic changes remain outside D2.

Implementation commit:
`7892810584be232cec744c0038ab9b3e069608ea`. Runtime-tree SHA-256:
`5759506d46ee006e6f2873312f2d8a8ac857be1d1488b59cafbb09b9de7a5e60`.

Frozen source hashes:

- `ironmule/model_identity.py`:
  `04099cb33d3ea28757b078e71bed3500117b5331a7b0ed55999b8dea2cefe50a`
- `ironmule/fingerprint.py`:
  `7901eb4e875d95811b876e311e1a053e41145898d33da5acb6b976dec2cd928e`
- `ironmule/tune.py`:
  `938fffa590d81d381d477136bc372fe145efa146ea94e301f1690df2f097eb74`
- `ironmule/service.py`:
  `74c26b487e99c6e38bf8d8dc71570962b4c1b460826c02b5976a5a2b152a99ed`
- `research/b27_main_baseline.py`:
  `3456be06f80be6c4b9eaa45a1a94bc7ffed6d47627790ceb24749a3bf9567ee5`
- `research/d2_compare.py`:
  `764a8449d1628e4eaff7579cfc5cf92f427d744da2f0066eab9dd8fcef726fc7`
- `docs/D2_MODEL_IDENTITY_CONTRACT.md`:
  `ebfb372f6f1dc32a1a73122a004fe5a7a393243f06d313f3a2f8951a641bf38d`
- `docs/D2_IMPLEMENTATION.md`:
  `64de5a496a2dd581f11776bf608eb3e2089538de9550b8fa4152d9fe023c225c`

## Findings and resolutions

1. **Ambiguous cache resolution fails closed.** Hub IDs use only the read-only cache
   index. An omitted revision is accepted only with one cached snapshot; an explicit
   revision must match one exact cached commit. No download API is imported or called.
2. **Identity is complete and path-free.** The aggregate identity binds full manifest,
   exact revision, architecture, canonical quantisation and tokenizer artifacts.
   Serialized records contain no source path. Local symlink escape and broken-link
   tests fail closed while normal HF blob links remain within their repository root.
3. **Load/source coherence is checked twice.** Identity is constructed before load and
   reconstructed after `mlx_lm.load`; a source change aborts instead of attaching the
   stale identity. The Engine and Runtime reject conflicting explicit identity,
   model-ID and quantisation values.
4. **Validity reuse is schema-exact.** Fingerprint v2 treats all exact identity fields
   as incompatibility fields. Profile conditions v2 reject missing, unknown, legacy
   and internally mismatched fields, including `require_compatible=False` access.
5. **Execution-only compatibility is explicit.** Manual or caller-managed online
   loading may run without identity, but fingerprint/revalidation/profile validity
   then fails closed. `load_engine` remains a two-tuple API.
6. **Independent evidence validation exists.** `research/d2_compare.py` reconstructs
   both expected identities directly from complete raw manifests without importing
   the implementation under test, checks Runtime and both arm fingerprints, then
   applies the frozen performance gates.
7. **Approved boundaries remain intact.** No `ironmule.evidence` runtime import, new
   dependency, network fallback, profile activation, strategy selector, plan switch,
   executor change or timed-serve hashing was added.

## Exact expected identities

- Gemma 3 4B: identity
  `2730e8b13b892b576452493dfb1983c0948c175d02c50099475385f8bac97bd2`,
  manifest `a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae`.
- Gemma 3 12B: identity
  `2b5b13a3c53a96299b33d0385b13a4b54973b810540cf7a99d4aa3966ebf1474`,
  manifest `aef12412aba9e6351a457381fb723672d253f8473b837757beff716d3c8e2b31`.
- Both: tokenizer
  `afbd505ba5b2603a0a7e0c09e6d51672c448953e8fc1f7141e98fd0c264d7129`,
  quantisation `4952fcd6b27eda69be81c1a695ca32544e62b6b6edc197d191ea132c0afc314b`.

## Verification before hardware measurement

- focused D2/comparator/profile suite: `39 passed`;
- full serial non-integration suite: `178 passed, 12 deselected` in `4.98 s`;
- real cached Gemma-4B integration: `11/11` in `22.14 s`;
- post-integration swap `0 B`; no residual model process;
- old incomplete profile rejected on compatible and raw revalidation paths;
- compile and whitespace checks green; dependency metadata unchanged.

One mistakenly sandboxed focused pytest invocation aborted during MLX import with
`SIGABRT`/exit `134`, before any model/test/timing ran. The cause and serial
outside-sandbox remedy are permanently recorded in the ledger and implementation
document; that event is not evidence and is not pooled.

## Remaining limits

Full snapshot hashing adds startup work and was intentionally not optimized in D2.
D2b remains a separate-session engineering regression screen against only D2a. It is
not a stock-`mlx_lm` comparison, qualification, quality study or profile activation.

**Decision:** the exact-identity implementation is ready for the separately sealed
D2b 4B/12B post-change protocol. No D2b performance sample had been observed at this
review point.
