# Task and adapter contracts

## repository_context_bundle.v1

Input JSON Schema: [task.schema.json](../schemas/task.schema.json).
Output JSON Schema: [result.schema.json](../schemas/result.schema.json).

| Field | Contract |
| --- | --- |
| `taskType` | Explicitly `repository_context_bundle.v1`; no classifier |
| `repositoryId` | Host-registered ID, never a filesystem root supplied by the agent |
| `snapshotId` | SHA-256 of the immutable, versioned text-snapshot payload |
| `query` | Nonempty single-line literal, max 512 characters, no CR/LF/NUL |
| `searchSemantics` | `literal_case_sensitive_line.v1` |
| `pathFilters` | Union of exact file/subtree prefixes; `.` means the registered scope; no glob/regex syntax |
| `contextLines` | 0..50 lines before/after each matching line, clamped to file bounds |
| `maxMatches` | 1..500 matching lines in a page, not a relevance threshold |
| `outputBudgetBytes` | 4096..1048576 UTF-8 bytes for compact serialized `Result` JSON |
| `continuation` | Null initially, otherwise opaque HMAC-bound cursor from this task/snapshot |

Search checks each LF-delimited logical line for the exact query sequence. It does
not case-fold, normalize Unicode, parse code or recognize symbols. One line produces
one match even if the literal occurs repeatedly within it. LF terminates a line; a
final LF does not add another empty line. A CR in CRLF remains part of the preserved
line content. An empty file has zero lines. All line indices are 1-based and inclusive.
Paths compare lexicographically by exact JavaScript string code-unit order, then line
number. There are no guessed definitions, references, rankings or summaries.

The output includes exact matching positions, file SHA-256, canonical compatible
excerpts with text SHA-256, snapshot/adapter provenance, exact total matching-line
count and page completeness. File hashes cover original UTF-8 bytes, including a
terminal LF. Excerpt hashes cover returned text joined by LF without an added final
file delimiter. Adjacent or overlapping presentation intervals become one interval;
this happens in every strategy, independently of physical read optimization.

No hits is COMPLETE. A count or byte limit produces PARTIAL, explicit reasons and a
continuation. `complete` means no matching lines remain after this page's offset, not
that the current page alone contains all preceding pages. Reducing output removes a
suffix of matches and their unnecessary context, never an unmarked slice of a line.
If even the first excerpt does not fit, the page contains zero matches, reports
OUTPUT_BUDGET and returns the same offset. Increase the budget to progress. If the
hard maximum still cannot represent a requested context, the host must narrow the
request explicitly; the runtime will not loop automatically.

Cursor MACs bind repository, snapshot, query, search semantics, path filters and
context. Pagination limits/budget may be changed for the next page. Policy is rebound
even for a valid cursor. Cursors contain no cached search/read results. The byte cap
covers the structured Result, excluding SDK JSON-RPC metadata and the legacy text
mirror of structuredContent. Clients should consume structuredContent once and set
their separate transport/model-token budgets accordingly.

## Trusted host context

[host.schema.json](../schemas/host.schema.json) is loaded from a separately configured
local file with owner-only mode. `principal`, enabled/expiry state, permissions,
registered root and resources, concurrency and objective cannot appear in task
arguments. Additional task fields are rejected. Config is reread before and after
concrete calls. A changed policy terminates the old run rather than combining grants
from two versions. A later task may bind the new policy normally.

Path authorization uses component boundaries: `src` allows `src/a`, not `src-other`.
Absolute paths, `..`, empty/dot segments, backslashes, control characters and glob
metacharacters are rejected. Import rejects ancestor/final symlinks and special files.
Binary/non-UTF-8 files and size limits fail the import; no hidden exclusion can make an
incomplete snapshot look complete. The snapshot represents precisely the imported
host-selected text scope, not necessarily the entire live Git repository or a Git
commit. Its content hash is authoritative after capture.

## Controlled MCP adapter

Identity/limits: [adapter-contract.json](../schemas/adapter-contract.json).
Tools: [search input](../schemas/search-input.schema.json),
[search output](../schemas/search-output.schema.json),
[read input](../schemas/read-input.schema.json),
[read output](../schemas/read-output.schema.json).

| Property | Enforcement |
| --- | --- |
| Server | `ironmole.repository`, adapter version `1.0.0`, contract `repository-tools.v1` |
| Protocol | Inner connection negotiates exactly `2025-11-25`; current SDK schemas/catalog compared at connection |
| Resources | Only currently permitted registered snapshot paths |
| Effects | Immutable snapshot reads only; no repository code, writes, sampling or arbitrary resource dereference |
| Concurrency | Width 1/2/4 bounded by both adapter maximum 4 and host resource cap; same-snapshot reads only |
| Timeout/retry | At most 3000 ms per call within a 15000 ms task deadline; zero automatic retries |
| Per-task | 32 IR nodes, 500 map items, 501 tool calls, 16 MB cumulative intermediate/response budget |
| Inputs/outputs | 16 KiB task/call input; at most 2 MB per tool response; final output bound above |
| Snapshot | At most 1000 files, 256 KiB per file, 8 MiB total original UTF-8 bytes |
| Cancellation | Stop dispatch; signal in-flight requests; unknown outcomes stay unknown, no rollback claim |
| Cache | No tool-result cache; supported protocol/cache hints never replace authorization |

Annotations describe read-only behavior but grant nothing. Both runtime and server
check concrete search prefixes/read paths. Unknown tools and unsupported interactions
fail explicitly. Structured tool schemas, result sizes, snapshots and checksums are
validated; repository strings never become instructions. The controlled server's
single event loop may limit real parallel speed; no network-delay simulation is
inserted to manufacture a concurrency advantage.

## Measurements

Runtime traces measure observed tool envelopes, node durations, routing, serialization
and response sizes. A tool duration includes IPC/client validation; it is not pure
server time. Runtime overhead subtracts the union of observed call intervals, not the
sum of parallel durations. Persisted timing ends before the final trace write;
benchmark wall time surrounds the entire call, including trace persistence and final
local JSON serialization. Final agent response time remains a separate unavailable
metric without a trusted agent adapter.

Agent metrics use [agent-metrics.schema.json](../schemas/agent-metrics.schema.json).
Unknown tokens/cost/time stay null. A cost needs an explicit price basis. The interval
between MCP calls is never recorded as model reasoning time. A passive proxy cannot
supply missing dependency or model-usage evidence.
