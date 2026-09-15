# MoleIR v1 and controlled deoptimization

## JSON format and compilation

[MoleIR JSON Schema](../schemas/moleir.schema.json) is a versioned interchange format,
not executable source text. [Reference](../examples/reference.moleir.json) and
[optimized](../examples/optimized.moleir.json) examples contain all eight operations.
The MVP deliberately accepts one typed task spine, not arbitrary general programs.

| Operation | Supported semantics and type |
| --- | --- |
| CALL | `repo.search` with validated task-bound arguments -> search result |
| PROJECT | search matches + task context -> line ranges, clamped to snapshot |
| FILTER | ranges -> ranges, nonempty intervals only; no relevance filtering |
| DEDUP | ranges -> ranges, exact path/start/end/file-hash key |
| MERGE | ranges -> ranges, identity or compatible overlapping/adjacent intervals |
| BOUNDED_MAP | ranges -> ordered reads, task snapshot binding, <=500 items, width 1/2/4 |
| GUARD | host-ready, validated search or validated reads -> unit |
| RETURN | search + reads -> canonical, budgeted context bundle; scan or indexed preparation |

Expressions distinguish task fields, typed node results and scalar constants.
The current task's CALL argument bindings must name their exact task fields; a
constant cannot silently replace the query or repository ID. Per-item read arguments
use the fixed typed `range_and_task_identity` binding. Transformations are named
built-ins. There is no JavaScript/Python body, dynamic property evaluation, `eval`,
regex executable input, arbitrary function call or external code loading.

Compilation validates schemas, unique IDs, dependencies, cycles, output types,
control-guard ancestry, the supported semantic spine, reachable return, available
adapter/schema versions, read-only effects and ceilings. It produces a topological
order, explicit dependency table, output types and content hash, then stores the
plan version. This is compilation in the MVP; no machine code is generated.

Optimization changes deduplication, interval coalescing, bounded read width and
scan/indexed preparation. Plan version 2 adds the latter as an explicit bounded
choice; the preceding measurement retains its version 1 inputs.
The parameterized graph, snapshot identity, exact source evidence, result ordering and
completeness survive. Every concrete call still passes authorization. Resource or
adapter limits can reduce effective width. Data independence alone grants no overlap.

## Guard outcomes and handoff

GUARD_FAILED identifies a domain/runtime-assumption failure, including malformed
schema, missing/changed snapshot, false tool contents, limits or invalid plan.
POLICY_BLOCKED means access is denied or the bound policy changed. INFRASTRUCTURE_ERROR
covers transport failure, timeout and host cancellation. Unsupported explicit task
classes report BYPASS. Valid limit truncation is PARTIAL, not an infrastructure error.

On failure the scheduler sets one terminal reason, starts no more nodes or map items,
signals active requests and waits only within their already-bounded call timeouts.
Completed verified reads are retained; an errored or timed-out call is never called
success. A cancellation failure or timeout can have an unknown outcome even though
this adapter permits only reads. No entire-task restart or automatic repair loop exists.

The handoff carries task type, plan ID/version, reason, failed assumption, completed
and failed steps, unknown in-flight outcomes, valid/invalid result references, remaining
subgoal and continuation restrictions. Private retained results contain confirmed
search metadata and completed reads. On policy revocation content is not exposed
through a fallback or retained-output reference. Current permissions must be rebound
before any manual continuation. A compact handoff may shorten ID lists to fit the
output budget; `detailsTruncated` flags this and the run ID identifies the complete
private trace. No omitted outcome is implicitly confirmed.

Error strings are closed codes, not repository-provided instructions. The host owns
any continuation: it may inspect valid evidence, request an explicitly narrower task,
repair a permission/configuration issue or ask the agent to complete the remaining
subgoal within its own authority. It must not reinterpret POLICY_BLOCKED as permission
to execute the same operation through another tool.

## Trace and registry boundaries

Traces contain task/contract versions, input shape, private HMAC argument identities,
concrete numeric bindings and snapshot IDs, provenance expressions, graph dependencies,
call outcomes/durations/bytes, guards, result checks and optional trusted model metrics.
Raw queries and repository text are absent from ordinary traces. Full contents exist
only in the requested input snapshots, returned evidence and necessary private handoff
artifacts. Files/directories use owner-only permissions; nothing is uploaded.

The first miner accepts only runtime-verified typed graphs. It recognizes repeated
structure and proposes parameterized variants from the same known task. A graph's
semantic spine, versions and dependencies matter; mere call order or similar strings
do not establish interchangeability. Passive/incomplete traces remain observations.
Candidate validation uses separate snapshots/repositories. Production activation is
unavailable from the local-only benchmark; tests of the activation state machine use
explicit synthetic gate inputs and are not performance evidence.
