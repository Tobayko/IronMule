# IronMole MVP work plan

The hypothesis is unproven. MOLE1 in the repository backlog owns outstanding
performance evidence. No IronMule inference code or frozen evidence is changed.

- [x] Inspect backlog, dead ends, architecture, evidence helpers and MCP documentation.
- [x] Specify task, JSON MoleIR, adapter, permission and measurement contracts.
- [x] Implement a complete snapshot -> MCP search/read -> reference bundle -> CLI flow.
- [x] Add bounded graph compilation, read optimizations and structured deoptimization.
- [x] Persist versioned plans, redacted traces and provenance-bound candidates.
- [x] Add held-out evaluation, conservative routing and lifecycle suspension.
- [x] Execute correctness, fault, MCP and configuration integration checks.
- [x] Execute the preregistered local benchmark; report unavailable model arms.
- [x] Review the complete diff, publish local instructions and record results.

## Fixed scope

Only `repository_context_bundle.v1`. No classifier, repository code execution,
result cache, production experiment, remote service, automatic task restart or
unbounded repair. Future LSP, second adapter and browser work is documentation only.

## Architecture decisions before implementation

Use a separate strict TypeScript package, official MCP SDK 1.29.0 (available in the
local cache; the current v1 branch reports 1.30.0), negotiated 2025-11-25 protocol,
JSON Schema contracts and Node's built-in SQLite. The v2 SDK has a different package
layout; no unverified v2/cache fields are mixed into this v1 implementation.

The runtime is model independent: no model is required for local correctness. Agent
usage arrives separately through a trusted adapter interface. Missing usage remains
null. CLI model benchmarks are unavailable until a real adapter is supplied.

Snapshots are private content-addressed copies of a host-registered text scope.
Execution reads snapshot data, never live repository files. Symlinks, special files,
invalid UTF-8 and oversized scopes are rejected explicitly at import. Host policy is
loaded independently of MCP task arguments and is re-read around each concrete call.
A local same-user process with write access to host policy is trusted; this is not an
OS sandbox against a malicious account owner.

MoleIR is a closed, typed DAG for this one task. Transformations are named built-ins,
not executable strings. The reference reads each requested interval sequentially;
optimized variants may deduplicate and coalesce read intervals and run at widths 1,
2 or 4 only if the adapter and resource contract both allow it. Presentation is
canonical in every arm. The first seed plan is explicitly manual.

`friday_evidence.statistics` supplies offline summaries and paired bootstrap intervals
through a small Python reporting bridge. Its study-bound database cannot hold new
plan states without changing its registered tools/schema; therefore IronMole owns a
separate operational SQLite database, not a fork of the shared evidence store. No
sealed package is copied or modified.

## Reachable completion

Local deliverables, tests and two separately identified MCP benchmark runs are complete.
See [results](RESULTS.md). No model adapter or actual PTC integration was available;
those comparisons and end-to-end qualification remain open in root MOLE1. No candidate
is active and no learned/adaptive performance advantage is claimed. The separate
preparation comparison in LOCAL-2 did not pass the 5% benefit gate.
