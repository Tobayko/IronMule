# D2 — Exact model identity contract

**State:** explicitly approved by the user on 2026-08-29; implementation not yet
started when this contract was written.

## Decision

D2 may propagate exact model identity through local model loading, `Runtime`, runtime
fingerprints and tuned-profile compatibility. D2 may invalidate an old/incomplete
profile and fall back to the existing baseline. It may not select a new strategy,
persist D1 evidence, route/activate a plan or mode, download a model, add a dependency
or change inference semantics.

## Required identity

Every identity record is immutable, path-free and canonically hashed. Mandatory:

- model ID suitable for public evidence;
- exact revision;
- complete present-file manifest SHA-256;
- architecture from `config.json`;
- canonical quantisation object plus its SHA-256;
- tokenizer-artifact SHA-256;
- identity schema and aggregate identity SHA-256.

Absolute local paths are execution details and never appear in `to_dict()`, Runtime
fingerprints or profiles.

## Local source resolution

### Cached Hub model

- Inspect the existing Hugging Face cache read-only via `scan_cache_dir()`.
- Match the exact repository ID.
- If the caller supplies a revision, require exactly that cached commit.
- Without a revision, accept exactly one cached revision; zero or multiple revisions
  fail closed instead of choosing newest/main implicitly.
- Do not call `snapshot_download`, the network, or mutate Hugging Face globals.

### Local directory

- Pass the directory directly to `mlx_lm.load` as today.
- If it is an HF `snapshots/<commit>` path, use that commit as the revision.
- Otherwise derive a stable `local-<manifest-prefix>` revision after hashing the
  complete directory and use a path-free `local:<directory-name>` model ID.

### Online/caller-managed mode

Legacy `offline=False/None` loading remains caller-managed, but no tuned profile or
Runtime fingerprint may be reused until an exact local resolved identity is available.
The approved D2 implementation does not initiate an online download.

## Digest rules

- Complete manifest: sorted relative paths, byte lengths and full SHA-256 of every
  regular file in the resolved model directory; symlinks are read through but no path
  outside the resolved snapshot is serialized.
- Tokenizer digest: sorted content records for tokenizer/vocabulary/merges/special-token
  files. Absence is a hard error.
- Quantisation: `quantization` or `quantization_config` from `config.json`, requiring
  positive integer `bits` and `group_size`; no inferred defaults.
- Architecture: non-empty `model_type` preferred, otherwise exactly one non-empty
  `architectures` entry; ambiguity is rejected.
- Canonical JSON forbids NaN/Infinity and uses sorted keys with compact separators.
- Per-process memoization is allowed only for an exact resolved directory plus a
  deterministic file-stat signature; it is a performance cache, not evidence storage.

## Runtime wiring

- Add a stdlib-only `ironmule.model_identity` module.
- `load_engine(..., revision=None)` resolves identity before load and attaches it to
  the returned Engine without changing the existing two-value return shape.
- `Runtime.load(..., revision=None)` requires the attached identity and passes it into
  `Runtime`.
- Manually constructed Runtime objects may omit identity for execution-only tests, but
  `Runtime.fingerprint()` and `Runtime.revalidate()` then fail closed.
- Runtime/fingerprint records include the complete identity fields/digest.
- D1 `ironmule.evidence` remains outside the runtime import graph.

## Tuned-profile compatibility

- Introduce an exact conditions schema version.
- Conditions include model revision, manifest, architecture, quantisation object/
  digest, tokenizer digest and aggregate model-identity digest.
- Profile creation uses the identity attached to the actually loaded Engine.
- Profile reuse resolves current local identity and compares every exact field.
- Existing profiles missing any D2 field are rejected, not migrated or guessed.
- `require_compatible=False` may expose a structurally valid exact-identity profile to
  the revalidation path, but must not make a legacy/incomplete profile reusable.
- No profile is automatically activated by D2; existing load behavior either reuses a
  fully compatible stored knob profile or stays on baseline.

## Public API compatibility

- Existing Hub-ID calls with one cached revision continue to work.
- `revision=` is an additive keyword.
- `load_engine` remains a two-tuple API.
- Existing `Runtime`, plan and service-mode behavior is unchanged.
- Fingerprints are intentionally schema-incompatible with incomplete pre-D2 records.

## Required tests

1. deterministic full manifest and tokenizer digests;
2. content/filename/size changes alter identity;
3. no absolute path in serialized identity;
4. unique cached revision resolution and explicit revision selection;
5. zero/multiple/missing revision fail closed;
6. local-directory stable identity and HF-snapshot revision recognition;
7. missing/invalid config, architecture, tokenizer or quantisation fail closed;
8. `load_engine` attaches identity without environment/global mutation;
9. Runtime fingerprints contain exact identity and reject missing identity;
10. exact identity drift invalidates Runtime revalidation and tuned profiles;
11. legacy/incomplete profiles are not reusable, including raw revalidation access;
12. no D1 evidence import, route, profile activation or inference change;
13. full serial suite and real 4B integration stay green;
14. sealed D2a/D2b 4B/12B pre/post evidence with correctness/resources.

## Kill/pivot

Stop D2 and retain baseline behavior if exact identity requires network access,
unbounded non-model traversal, storing absolute paths, guessing a revision/
quantisation, changing tokens/fallback semantics, making incomplete profiles reusable,
or introducing model hashing on the timed `Runtime.serve` path.

## Explicitly outside D2

- D1 EvidenceRecord/TrustedProfile persistence;
- Optimization Memory/SQLite/migrations;
- strategy or plan/mode selection;
- Researcher/Reviewer/Evaluator automation;
- stock `mlx_lm` arm;
- model download, models above 12B or new dependency;
- automatic routing, activation, canary or rollback.
