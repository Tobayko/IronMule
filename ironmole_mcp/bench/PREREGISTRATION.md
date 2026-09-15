# MOLE1-LOCAL-2 preregistration

Written before any final benchmark samples. This is a local runtime experiment,
not a model-agent or Programmatic Tool Calling performance claim. The controller
seals this document, source, schemas, lockfile and fixture definitions before
measurement, records the seal in every child, and verifies unchanged inputs afterward.
Any repair after measurement requires a new run ID; failed attempts remain present.

## Question and fixed gates

Does adaptive IronMole lower complete task latency or cost against a strong stored
workflow? The full hypothesis requires actual agent final-response measurements.
Those values are unavailable in this local phase and cannot qualify activation.

Prospective end-to-end activation gates: exact independently expected content,
order, provenance, completeness and permissions on every case; at least 20 pairs;
no omitted failed runs; paired median objective ratio upper 95% bound <= 0.95
against C; p95 ratio <= 1.05; A/A median within [0.95, 1.05] and interval including
1; known objective units and continuation costs; positive amortized advantage at
100 further uses. Objective: latency with USD 0 incremental model/tool-spend cap.
Unknown cost is not zero. Electricity, engineering labor and hardware purchase are
outside the monetary objective; learning/validation time is reported separately.

## Arms

- A_DIRECT_AGENT: unavailable until a real host-side model adapter provides direct
  search/read calls, final response and actual usage under this task contract.
- B_PTC: unavailable until an actual supported PTC integration is supplied. A local
  JavaScript loop is not substituted for it.
- REBUILD: construct and compile the manual sequential plan within each timed call.
- REFERENCE: retrieve the stored manual reference and execute it sequentially.
- PARALLEL: reference reads at selected width, no dedup or interval merging.
- DEDUP: identical interval elimination, width 1, no merging.
- MERGE: compatible interval coalescing, width 1, without standalone dedup.
- C_WORKFLOW: stored manual plan with dedup, coalescing and selected width.
- C_AA: identical C_WORKFLOW, separate subprocess, same protocol.
- D_ADAPTIVE: full conservative routing; reference unless prior qualification exists.
- PREPARATION: sequential reference reads with an indexed presentation path; index
  verified line contents once rather than reconstructing per-excerpt/per-budget-prefix
  maps. Output remains exactly equal. C also uses this preparation path. RETURN node
  timing includes index construction. A separate correctness pass checks all widths.

C's width is selected from 1, 2, 4 using 6 shuffled repetitions on the selection
repository only, before opening final test snapshots. Minimum median complete local
runtime wins; deterministic tie breaks to the lower width. D cannot learn a route
from the test data. Training mines at least 3 traces with two distinct parameter
bindings from a separate training repository; candidates stay unqualified.

## Workloads and isolation

Synthetic whole repositories `train`, `selection`, `holdout-a`, `holdout-b` have
separate manifests and identities. Four final cases: no hits, overlapping reads,
sparse reads across independent files and dense repeated lines with bounded paging.
Expected matching line positions are computed by a separate fixture oracle; it does
not import runtime search, interval transformation or result preparation functions.
Each delivered excerpt is checked directly against fixture bytes and independent
SHA-256. Every arm must agree with reference bundles after excluding cursor MACs
(which are private-state specific). Case descriptions are fixed in `fixtures.ts`.

Six independent process blocks, four measured repetitions per final case per arm
per block, two complete warmup sweeps. Each arm in each block starts a fresh Node
process, registry and MCP child; its first complete sweep is recorded as cold
process/connection measurements. Repeated timings include routing, compilation,
policy/snapshot guards, tools, formatting, JSON serialization and trace persistence.
Startup plus first request is separately recorded for the first case. Filesystem/OS
cache temperature is uncontrolled and must not be called a cold disk measurement.

Arm order is seeded, shuffled then rotated by block; every child shuffles case order
with a recorded seed. Pair by block, repetition and case. Report raw samples,
median/p95/min/max/stdev, paired ratios and 95% bootstrap intervals from the existing
`friday_evidence.statistics`. Primary uncertainty uses six block-median ratios
(clustered by independent subprocess block), not 96 allegedly independent samples.
Report the A/A control from this exact regime. Local ratios are descriptive even if
they clear thresholds: final-agent latency, models and money are absent.

## Failures and missing data

A correctness error invalidates the run and the child exits nonzero; the controller
writes an aborted result and keeps raw data. Guard, policy and infrastructure failures
remain separate. No automatic retries, warmup extensions, threshold changes, outlier
exclusions or best-run selection. Dedicated tests exercise zero/many hits, limits,
permissions, symlinks, schema/snapshot drift, malformed replies, cancellation, timeout
and parallel branch failure. Those test outcomes are not timing samples.

Final task latency including agent response, model calls, input/output tokens,
billed cost, deoptimization continuation expense and end-to-end break-even count
remain null for unavailable A/B. IronMole itself invokes zero models; that observed
local count does not imply zero model work in a containing agent task. Successful
runtime-only measurements cannot activate a candidate. If C is as good or better,
additional adaptive benefit is reported as not demonstrated.

## Prospectively identified follow-up

MOLE1-LOCAL-1-20260912-attempt1 completed with 768 correct measured outputs and no
qualified adaptive benefit. It only instrumented presentation cost; it did not isolate
a preparation algorithm. MOLE1-LOCAL-2 adds the indexed presentation implementation
and a separate PREPARATION arm, retains all original thresholds/workloads/repetitions,
and includes that implementation in C. This is a changed implementation and a distinct
run, not a retry or a retroactive repair of LOCAL-1. LOCAL-1 inputs/results are retained.
Kill: any output/permission mismatch or no net improvement leaves no claim for the
new preparation mechanism. Both runs remain runtime-only and cannot activate plans.
