# D2b — Exact-model-identity post-change regression screen

**Date:** 2026-08-29
**Status before measurement:** sealed; no D2b timing observed.

## Question

Did the approved D2 exact-identity wiring preserve the same-day D2a strict-baseline
behavior and resources for the exact cached Gemma 3 4B and 12B cells, while emitting
the required path-free Runtime identity and fingerprint v2 records?

This is an engineering regression screen. It cannot qualify or activate a tuned
profile, persist D1 evidence, select a strategy, route a request or establish a
stock-`mlx_lm` comparison.

## Frozen implementation and review

- implementation parent commit:
  `7892810584be232cec744c0038ab9b3e069608ea`;
- measurement must run from the clean commit containing this preregistration;
- runtime-tree SHA-256 must equal
  `5759506d46ee006e6f2873312f2d8a8ac857be1d1488b59cafbb09b9de7a5e60`;
- baseline harness `research/b27_main_baseline.py` SHA-256:
  `3456be06f80be6c4b9eaa45a1a94bc7ffed6d47627790ceb24749a3bf9567ee5`;
- independent comparator `research/d2_compare.py` SHA-256:
  `764a8449d1628e4eaff7579cfc5cf92f427d744da2f0066eab9dd8fcef726fc7`;
- comparator tests SHA-256:
  `dcb8ab8f289831a1777a2bab32fd8905d08035602b2d7f9424ec886b70f58902`;
- implementation review SHA-256:
  `a0f634a77515741db17e3205ffb827f2d318439e7294ea399eead4a890792e5f`;
- approved D2 contract SHA-256:
  `ebfb372f6f1dc32a1a73122a004fe5a7a393243f06d313f3a2f8951a641bf38d`.

No runtime source, harness, comparator or test hash may change after this document is
sealed. Documentation-only result commits follow measurement.

## Frozen pre-change references

D2b compares only these new same-day D2a raw records:

- Gemma 3 4B:
  `c012c9a3e9b25d995e940d363137238f717a42ccae611f52354d7779cbad39d9`;
- Gemma 3 12B:
  `745d63222c42937e72bfb5b32b5e5773ed727b6f3366b229dcd2c0f5c76817aa`;
- D2a runtime-tree:
  `d7577af8e83778b9753ad4bf721656a16d923a9f848040e406178b7dcffc8a21`.

B27a2, B27d, B27e and every other timing record are excluded. There is no pooling,
paired-session claim or threshold adjustment.

## Exact cells, identities and order

1. `mlx-community/gemma-3-4b-it-4bit`, revision
   `93724907d4ed1745d2fe50baadf3b0b01a65abf2`, full manifest
   `a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae`,
   exact identity
   `2730e8b13b892b576452493dfb1983c0948c175d02c50099475385f8bac97bd2`.
2. Only after cell 1 exits cleanly and the system recovers,
   `mlx-community/gemma-3-12b-it-4bit`, revision
   `86cc6a8dedbc456dd0e4af01a9d09f396f77e558`, full manifest
   `aef12412aba9e6351a457381fb723672d253f8473b837757beff716d3c8e2b31`,
   exact identity
   `2b5b13a3c53a96299b33d0385b13a4b54973b810540cf7a99d4aa3966ebf1474`.

Both require architecture `gemma3`, quantisation `{bits: 4, group_size: 64}`,
quantisation SHA-256
`4952fcd6b27eda69be81c1a695ca32544e62b6b6edc197d191ea132c0afc314b`
and tokenizer SHA-256
`afbd505ba5b2603a0a7e0c09e6d51672c448953e8fc1f7141e98fd0c264d7129`.
No model above 12B, download, installation or network fallback is allowed.

## Protocol per cell

- `experiment_id=D2b`;
- fresh serial process per model, run outside the restricted sandbox;
- exact local cached revision; full model hashed before and verified after load;
- strict one-shot plan and `BASELINE=Knobs()`;
- no stored tuned-profile reuse;
- Interactive and Throughput W4 arms;
- six requests × 48 physical output tokens;
- two warmups and six alternating measured repeats per arm;
- complete `Runtime.serve` outer wall and physical token rate;
- one loaded model shared between arms;
- atomic raw evidence with environment, full model binding, Runtime identity,
  fingerprint v2, outputs and resources.

## Hard preflight, identity, correctness and resource gates

- clean source commit; frozen runtime tree and source hashes;
- Xcode/Metal/MLX prerequisites green;
- AC, low-power false, memory free `>=80%`, swap `<=256 MiB`, no competing model
  process immediately before each cell;
- exact hardware/framework/model/protocol match to the corresponding D2a cell;
- `runtime_model_identity` exactly equals the independent identity reconstructed from
  the complete post manifest;
- both arm fingerprints use `ironmule.runtime_fingerprint.v2` and exactly match that
  identity's revision, manifest, architecture, quantisation, tokenizer and aggregate
  digest;
- token IDs, stop reason and physical/visible counts identical between arms;
- zero fallback/correctness errors, swap growth `<=256 MiB`, no crash/timeout/residual;
- every warmup/repeat and raw sample present.

An incomplete or identity-invalid record remains a hard failure. A system/model/
framework/protocol domain change is classified before timing is interpreted.

## Frozen post/pre regression rules

For every model and both arms, independently bootstrap the D2b/D2a median ratio from
six raw post and six raw pre samples with 10,000 resamples and fixed seed `20260829`.

- outer-wall passes when the 95% CI high is `<=1.05`;
- physical-token-rate passes when the 95% CI low is `>=0.95`.

Ordered classifications:

1. incomplete model/evidence set -> `INCONCLUSIVE_INCOMPLETE`;
2. hardware/software/model/protocol/system-domain difference ->
   `REVALIDATION_REQUIRED`, regression kind `EVIDENCE_DRIFT`;
3. same-domain identity/correctness/resource failure -> `CODE_REGRESSION`, regression
   kind `HARD_IDENTITY_CORRECTNESS_OR_RESOURCE_REGRESSION`;
4. same-domain performance interval miss -> `INCONCLUSIVE_POTENTIAL_REGRESSION`,
   regression kind `POTENTIAL_CODE_REGRESSION`;
5. every identity/correctness/resource/performance gate passes ->
   `NO_REGRESSION_OBSERVED`, regression kind `NONE`.

Any result is retained. There is no retry, pooling, repeat-until-good, threshold
change, selective suppression or activation consequence.

## Required outputs

- ignored `D2b_gemma4b_post_20260829.json` and
  `D2b_gemma12b_post_20260829.json`;
- tracked path-free D2b post summary;
- tracked path-free comparison with exact D2a/D2b hashes and intervals;
- deterministic verification artifact;
- ledger, limits, backlog and local history dashboard updates;
- final full serial suite and real 4B integration verification;
- clean ProjectAtlas/runtime and Git state.
