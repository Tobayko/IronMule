# PROD15 selection contract

`ironmule_product.selection` is a dependency-free, path-free policy boundary. It
does not run inference, inspect a machine, read evidence files, persist events, or
explore inside a user request. The integration layer must supply already validated
records and should eventually source their audit IDs from the canonical
`friday_evidence`/`EventJournal` infrastructure.

The only candidate IDs are `reference`, `prefix_reuse`,
`baseline_interactive`, `core_interactive`, `baseline_throughput`, and
`core_throughput`. Their existence is not evidence. In particular, legacy measured
arm configurations remain candidates; an empty evidence store authorizes no
optimization.

## Promotion and selection

Evidence is eligible only when it:

- binds the identical model ID, immutable revision, model snapshot SHA-256, and
  hardware, environment, code, and source-manifest SHA-256 identities;
- covers the request profile, context/new-token and session-request bounds,
  streaming mode, group width, and prefix-hit requirement;
- is an independent held-out, exact-quality, paired end-to-end result; and
- has a finite confidence upper bound no larger than `1 - frozen_min_effect`.

Among eligible records directly comparing against the compatible incumbent, the
lowest measured end-to-end estimate wins. Missing, uncertain, mismatched, or
malformed evidence returns `reference` (the qualified stock/product path), never
a profile-named engine baseline, with a reason. Historical gains,
including historical 20% class results, are not imported as authorization for a
new context. A candidate cannot displace a non-reference incumbent using only a
comparison against some other baseline.

`prefix_hit` means that a compatible cache entry is available; it is not a request
requirement that disqualifies a normal non-cache arm. Hot-only prefix evidence sets
`prefix_hit_required`; cold/session evidence must include cache setup in its timing.
Explicit session bounds prevent a hot single request from standing in for a full
cold session.

The request path never performs silent online exploration. Integration is not
allowed to activate an optimization until real native evidence has been independently
validated and converted to this contract.

## Learning boundary

`RewardObservation` accepts only observed actual rewards with unique audit IDs and
binds model ID/snapshot hash, profile, session size, and hot/cold prefix state.
Train, validation, and held-out assignment is enforced by run/process `cluster_id`,
not by individual rows, so one cluster cannot cross splits. `propose_ranks` exposes
a small empirical mean-minus-standard-error ranking over the finite action set and
excludes held-out rewards. It is a proposal only: it cannot produce promotion
evidence, cannot alter `select_candidate`, and makes no RL or offline-replay gain
claim.

## Missing runtime hooks

The server integration still needs to provide the exact request context, load and
validate immutable integration-evidence records, pass the currently compatible
incumbent, and journal the decision/reason/audit ID. Native HTTP/worker measurements
and independent promotion remain separate work; this contract and its synthetic
mapping tests make no GPU, MLX, model-performance, or hardware claim.
