# D2 — Exact model identity implementation

**State:** implemented and locally verified on 2026-08-29; the sealed D2b 4B/12B
post-change regression run is still pending.

## Outcome

D2 binds the model that was actually resolved locally to Runtime validity and tuned
profile reuse. A path-free immutable `ModelIdentity` now contains the exact cached
revision, complete present-file manifest, architecture, canonical quantisation,
tokenizer artifacts and their aggregate digest. A missing, malformed, ambiguous or
changed identity fails closed. D2 does not persist D1 evidence, choose an execution
strategy, route a request or activate a profile.

## Data flow

```text
read-only cache index / local directory
  -> ResolvedModelSource(path, ModelIdentity)
  -> mlx_lm.load(path)
  -> post-load full-manifest verification
  -> Engine.model_identity
  -> Runtime.model_identity
  -> runtime fingerprint v2 / tuned-profile conditions v2
```

The source path exists only in `ResolvedModelSource`. `ModelIdentity.to_dict()`,
fingerprints, profiles and public evidence contain no absolute path.

## Fail-closed boundaries

- Hub IDs resolve through `scan_cache_dir()` only. Zero or multiple cached revisions
  are rejected unless an exact cached commit is supplied with `revision=`.
- Local directories receive a path-free `local:<name>` ID and a content-derived
  revision unless they are already an HF `snapshots/<commit>` directory.
- Full model and tokenizer files are SHA-256 hashed before load. The complete identity
  is reconstructed after `mlx_lm.load`; any intervening source change aborts.
- Broken links and local-directory links escaping the model root are rejected. HF
  snapshot links may resolve only within that model repository's cache root.
- `Runtime` rejects conflicts among its explicit identity, Engine identity, model ID
  and quantisation. Execution-only manual runtimes may omit identity, but fingerprint
  and revalidation calls then raise.
- Fingerprint schema `ironmule.runtime_fingerprint.v2` includes every exact identity
  field. Pre-D2 fingerprints are intentionally incompatible.
- Tuned-profile conditions schema `ironmule.tuned_profile.conditions.v2` requires the
  complete exact identity. Legacy, partial, unknown-field and internally inconsistent
  profiles are rejected even on the raw revalidation path.
- Caller-managed `offline=False/None` loading remains possible, but it has no exact
  identity and therefore cannot produce a validity fingerprint or reuse a profile.

Hashing happens during local resolution/load and never on the timed `Runtime.serve`
path. There is no network fallback and no new dependency.

## Exact cached identities used by D2b

| Model | Revision | Manifest SHA-256 | Tokenizer SHA-256 | Identity SHA-256 |
| --- | --- | --- | --- | --- |
| Gemma 3 4B 4-bit | `93724907d4ed1745d2fe50baadf3b0b01a65abf2` | `a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae` | `afbd505ba5b2603a0a7e0c09e6d51672c448953e8fc1f7141e98fd0c264d7129` | `2730e8b13b892b576452493dfb1983c0948c175d02c50099475385f8bac97bd2` |
| Gemma 3 12B 4-bit | `86cc6a8dedbc456dd0e4af01a9d09f396f77e558` | `aef12412aba9e6351a457381fb723672d253f8473b837757beff716d3c8e2b31` | `afbd505ba5b2603a0a7e0c09e6d51672c448953e8fc1f7141e98fd0c264d7129` | `2b5b13a3c53a96299b33d0385b13a4b54973b810540cf7a99d4aa3966ebf1474` |

Both use architecture `gemma3`, quantisation `{bits: 4, group_size: 64}` and
quantisation SHA-256
`4952fcd6b27eda69be81c1a695ca32544e62b6b6edc197d191ea132c0afc314b`.
The independent D2 comparator reconstructs these values from each raw complete
manifest without importing `ironmule.model_identity`.

## Verification before D2b

- focused D2/comparator/profile regressions: `39 passed`;
- full serial non-integration suite: `178 passed, 12 deselected` in `4.98 s`;
- real cached Gemma-4B integration: `11 passed` in `22.14 s`;
- exact cache smoke reproduced both identities above;
- legacy local profile probe: no profile reused, including raw revalidation access;
- post-integration swap: `0 B`; no residual model process.

One attempted focused pytest command accidentally ran inside the restricted sandbox
and MLX aborted at import (`SIGABRT`, exit 134) before a model or test ran. Cause:
Metal/MLX initialization is not supported in that sandbox. Resolution: all IronMule
pytest/model commands are run serially outside it with the existing project Python.
The corrected focused run passed; no timing sample from the failed command is retained
or pooled.

## Remaining limits

D2 proves identity plumbing and compatibility rejection, not model quality, speed,
profile usefulness or strategy safety. It neither qualifies nor activates an old or
new tuned profile. D2b is an engineering regression comparison against only the
same-day D2a baseline, not a stock-`mlx_lm` comparison or performance qualification.
