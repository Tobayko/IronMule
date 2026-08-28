# B39b — B39 failure-evidence and pre-load swap correction

Experiment ID: B39b
Parent: B39
Registered: 2026-08-28, after the B39a pilot failure
Status: evidence/safety correction only; no hardware run authorized

B39a was invoked with the module command and attempted only arm A. The child
stopped at the `after_model_load` checkpoint with return code `3` and the exact
error `RuntimeError: B36 checkpoint gate failed: after_model_load`. The child
event stream was not retained, so the exact failing sub-gate (swap, memory, or
instrumentation) is unconfirmed. The existing partial sidecar does preserve
parent snapshots showing system Swap `1704921661 B` before the child and
`8568438784 B` afterwards (`+6863517123 B`, approximately `6.39 GiB`), far
above the existing `268435456 B` threshold. No warmup or measured repeat ran.
The parent then encountered an empty-data `StatisticsError`; the existing
partial sidecar is retained unchanged. The static evidence is
`B39a_pilot_failure_20260828.json` and is not a B39 measurement.

This amendment changes only the following safety/evidence behavior:

1. The B39 parent command wrapper must retain parsed `@@B39_EVENT@@` checkpoint
   events and child events when a child exits without a successful final JSON.
2. `summarise` with zero valid blocks must return structured `INCONCLUSIVE`
   without calculating any median, bootstrap, ratio, or performance statistic.
   The parent must publish a structured final failure while retaining the
   existing partial sidecar.
3. Before spawning any model child, the parent must enforce a new absolute
   process-start swap baseline ceiling of `268435456 B` (`256 MiB`). This is
   motivated by the B39a parent-system observation of `1704921661 B` already
   used before the child and the subsequent `8568438784 B` observation. The
   lost child event stream leaves the exact `after_model_load` sub-gate
   unconfirmed; the absolute baseline gate therefore prevents a repeat under
   an already heavily swapped host. The existing process-start-to-end
   swap-delta ceiling remains unchanged at `268435456 B`.

No B39 performance, statistical, arm, workload, ordering or threshold rule is
changed. The four arms, eight blocks, strict six-request/48-token workload,
correctness, memory, crash, identity, no-retry and no-activation rules remain
as frozen in B39. B39b is not a retry or reinterpretation of B39a. A clean
state/reset and a separate continuation are required before any future B39
hardware attempt; this amendment itself authorizes no such run.
