# PROD10 — native validation after user removal of artificial hardware gates

## Authority and prospective boundary

On 2026-09-07 the user explicitly requested removal of the protections and
completion of the remaining work until results exist. This protocol is new:
PROD8's six-second timeout result and all sealed studies remain immutable.
No old failure is retroactively passed or pooled with these observations.

The new runner has **no** generation/startup timeout, cumulative work ceiling,
GPU duty ceiling, compulsory pause, CPU-load/AC/thermal admission gate, RSS
ceiling or swap-growth cutoff. ProductStore explicitly persists a null request
deadline in its isolated test configuration; the backend receives null too.
Neither an infinity value nor a larger hidden timeout substitutes for null.
The normal product's existing 120-second default remains backward-compatible.
Finite request/token counts are the declared workload, not hardware gates.

OS thermal protections, user cancellation, isolated processes, finite protocol
buffers, finite queues, local-only transport, model-code restrictions, privacy,
exact-output validation and verified cleanup remain. No foreign process or
operating-system power/security setting is changed. Kaggle rules are unaffected.

## Frozen workloads and model scope

Each separately invoked matrix uses exactly one registered local snapshot:

| Model | Revision |
| --- | --- |
| Gemma 3 1B 4bit | `2d44e83dc9e80843d22fb941d3d699a0b1351aa6` |
| Gemma 3 4B 4bit | `93724907d4ed1745d2fe50baadf3b0b01a65abf2` |
| Gemma 3 12B 4bit | `86cc6a8dedbc456dd0e4af01a9d09f396f77e558` |

Three greedy cases: the unchanged public orchard message repeated 88 times
with its end marker at output caps 8 and 32, plus the existing short apples
prompt at cap 32. The long tokenizer count must be 1000–1100, independently
observed by the stock worker (PROD8's 12B snapshot yielded 1077).

A fresh stock MLX-LM process executes one warmup plus three recorded calls
per case. It closes normally before a separate product worker starts and
executes the same 12 calls. Full token/text/finish/count hashes must agree
within and across workers. The product worker then serves actual loopback
JSON and SSE for each case and a simultaneous four-client burst (case order
long8, short32, long32, short32). The backend may serialize these four
requests: successful concurrent clients are not a batching-throughput claim.

Next, a real streaming client requests counting 1–500 with a 512-token cap,
receives nonempty generated content, and disconnects its own socket. The
service must register cancellation, clear active/queued work, and answer a
subsequent short request exactly. A retirement/unavailable result is retained
as a product defect, not counted as successful warm recovery. No retry or
automatic restart hides that state.

## Diagnostics and reporting

The stock runner uses the public `prompt_progress_callback` of the installed
MLX-LM 0.31.3. Its inspected `generate_step` calls the callback after cache
evaluation and first-token evaluation. No extra synchronization, changed
chunk size, cache reuse, model monkeypatch or custom kernel is introduced.
Record model-load wall time, tokenizer wall time, callback boundaries, first
token wall time, full request wall time, library prompt/decode rates and MLX
peak allocations. These are **host/library phase timings, not pure GPU
hardware timestamps**. Initial/warm differences do not alone isolate compiler
cost. GPU timing/command traces require a separate explicitly labelled capture.

Public capture route checked against the [official MLX Metal debugger guide](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html):
Metal capture is available through `start_capture`/`stop_capture` with the
documented capture environment flag. A trace must remain local and must not
be mixed into uninstrumented performance comparisons.

Shared `friday_evidence.open_observation` observes the actual owned process's
libproc resident/physical-footprint/peak values and system swap at one-second
cadence. There is no stop policy in this observer. Missing telemetry is an
explicit evidence error, never a made-up zero. Host state is recorded around
identity hashing and load. One main thread owns EventJournal writes.
All prompts/output text remain in memory; persist hashes, counts, metadata and
failure states only. Stock and product processes never overlap.

## Server endurance stage

After the matrix and cancellation/recovery gate, an explicitly selected
`--soak-seconds 3600` runs a single persistent service for at least one hour.
It cycles the three fixed cases, alternating JSON/SSE without artificial
pauses. A started request finishes; the next request is not started after the
one-hour boundary. Record every request and memory samples; compare each output
against the independently measured stock case. This is a stability and memory-
trend measurement, not a universal server/quality or peak-throughput claim.

## Terminal gates

All planned phases must actually execute for an overall pass. Exact output,
finite valid JSON/SSE framing, cancellation/recovery, complete evidence,
unchanged source/provider/model metadata/runtime identities, and zero normal
worker exit codes are required. Performance gains are not admitted by this
protocol. Function/protocol errors are retained and diagnosed; a targeted fix
needs a new code identity and a documented verification run, not an invisible
retry. A result from the 12B cell does not qualify the 1B/4B cells or RL.
