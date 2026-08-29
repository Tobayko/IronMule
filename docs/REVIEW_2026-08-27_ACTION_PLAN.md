# Review action plan — 2026-08-27

This file turns the external IronMule review into an executable, evidence-gated plan.
The reviewer did not run the code on Apple hardware, so estimates remain hypotheses.
`docs/BACKLOG.md` is the authoritative open-work list; completed results move to the
ledger/limits/release notes and leave the backlog.

For the item-by-item completion audit, see [`docs/REVIEW_2026-08-27_COMPLETION_AUDIT.md`](REVIEW_2026-08-27_COMPLETION_AUDIT.md).

## Decision rules

1. Confirm a code claim with current source or a failing regression before changing it.
2. Correctness and complete service wall time gate every performance result.
3. Preserve the existing Qwen/B28 worktree changes and the established Gemma path.
4. Do not download/install software or models, publish releases, spend on runners/legal
   work, or change architecture without explicit approval.
5. Report baseline and candidate with repeats, median/spread, correctness, memory and
   raw data. A negative result is complete evidence.

## Release-blocking review items

| Review item | Current evidence | Backlog | Disposition |
| --- | --- | --- | --- |
| P0.1 first prefill token | Fixed and covered by executor/runtime regressions | — | **Completed** |
| P0.2 TTFT definitions | `engine_start_ns` now begins before prefill and is nonnegative; separate lifecycle phase timestamps remain missing | `R2` | **Partial; admission/prefill lifecycle needs architecture approval** |
| P0.3 delayed arrivals | Confirmed: all prefills precede executor admission | `R2` | Architecture approval required |
| P0.4 end-to-end benchmark wall | Primary public metric now uses complete service `outer_wall_ms` | `R3` | **Completed (primary denominator); R3 remains open for fresh processes/stock arm** |
| P0.5 order/cache bias | Arm plans and AB/BA repeat order are balanced; one loaded process/model is shared | `R3` | **Partial; fresh-process isolation remains open** |
| P0.6 `fused_argmax` service | Both fused-token and logits contracts are covered | — | **Completed** |
| P0.7 `correctness_errors` | Telemetry reports whether a correctness comparison ran | — | **Completed; shadow mode remains approval-gated** |
| P0.8 stock `mlx_lm` baseline | Missing | `R3` | Add only after exact contract/architecture approval |
| P0.9 mismatch exit | Structured token/stop/count diff and nonzero benchmark exit | `R3` | **Completed** |
| P0.10 profile validation on load | System and exact model identity fail closed before profile reuse | — | **Completed; D2b exact-identity/no-regression gates passed** |
| P0.11 revision/quantisation | D2 binds exact cached revision, complete manifest, architecture, quantisation and tokenizer to Runtime/profile validity | — | **Completed; 4B/12B D2b result `NO_REGRESSION_OBSERVED`** |
| P0.12 revalidate prompt drift | Current prompt is tokenized for drift checks | — | **Completed** |
| P0.13 import environment mutation | Import/probe environment isolation is covered | — | **Completed** |
| P0.14 dependency bounds | Qualified dependency bounds are enforced | — | **Completed** |
| P0.15 unsupported candidates | Unsupported candidates have typed rejection/continuation | — | **Completed** |
| P0.16 hardware discovery cache | Slow discovery is process-cached | — | **Completed** |
| P0.17 version duplication | Version ownership is single-sourced | — | **Completed** |

## Automated quality and release gates

All review test/CI points map to `R8` unless they need hardware or a production
service. The macOS workflow is written; local evidence covers unit regressions, wheel
build/metadata and zipimport CLI smoke, mismatch failure and narrow integration
skips. The clean installed-wheel job and remote CI execution remain open.

Apple-Silicon model CI, nightly model runs, 1/6/24-hour soak, OOM/device-failure,
disconnect/cancellation, Python/MLX matrices and load/backpressure tests remain under
`Q1` or `S1`; they require runner/resource or service-architecture approval. Dependency
scanning, CodeQL, SBOM, signing and the official tag/release are `D1`. A release is not
created automatically.

### Local verification snapshot

The final local evidence is recorded here without turning it into a performance
claim: `108 passed / 11 deselected` in `7.62s` real time (`8.46s` by `/usr/bin/time`),
maximum RSS `346,996,736 B`, and `0` swaps; `10/10` Gemma integration tests in `26.58s` real
time, maximum RSS `3,348,430,848 B`, and `0` swaps. `ironmule doctor` was green,
and the final wheel build, metadata check and zipimport CLI smoke were green; this is
not evidence of a wheel installation. The R3 smoke produced
identical tokens, but is explicitly not a performance claim. Its JSON evidence has
SHA-256 `36ba45933b3de344116812e34bb451d19124b0ab35db3d3ee659b768dacc6209`.
The public benchmark still shares one loaded process/model; fresh-process isolation
and the stock `mlx_lm` arm remain open R3 work.

Required boundary regressions:

- first token EOS and `max_tokens=1`;
- one/all requests finishing from prefill;
- ragged prompt/output lengths and staggered arrival;
- cache hit/miss/capacity/prefix mismatch;
- `fused_argmax` on/off;
- stale framework/model/profile conditions;
- group fallback after partial work;
- concurrent calls, cancellation, OOM and device errors when the service exists.

## Product roadmap consolidation

| Review scope | Backlog | Gate |
| --- | --- | --- |
| `ironmule serve`, OpenAI API, streaming, queue, grouping, cancellation, timeouts, backpressure, async API, health/metrics, warm model | `S1` | Architecture approval, then loopback MVP |
| Capacity buckets, cache LRU/budgets/expiry, strict prefix match, multi-turn/messages/stops, seeded sampling, compatibility registry, memory ceilings | `C1` | Architecture approval; exact-greedy remains separate |
| Hardware/model/quantisation/context/output/concurrency/arrival/workload/quality matrix and sustained load | `Q1` | Hardware/model/resource approval and raw evidence |
| Community bundle/optional submit, clean packaging, scans, SBOM, signing, tag/release | `D1` | No hidden upload; publishing approval |
| README order, M1–M4 claim scope, result labels, demo, decision table, complete CLI | `DOC1` | Every claim linked to evidence |
| Licence summaries/examples, commercial process and independent review | `L1` | Legal/user approval; no engineering-only closure |

## Performance research mapping

The review's B1–B26 performance list already maps to the same IDs in
`docs/BACKLOG.md`. Its Tier-0 section is authoritative and prevents rerunning closed
routes. `B7`/instrumentation precedes structural work; `B24` precedes custom-kernel
claims; `B12` remains correctness-blocked; approximate methods remain outside exact
mode. B28 is separate existing Qwen work and is preserved.

## Release sequence

### 0.1.1 correctness release

Close the remaining `R2`, `R3` and `R8` items, or explicitly document
approval/resource-dependent residuals. Required
evidence: unit/CLI/package gates, real-model smoke for available local models, complete
wall-time benchmark protocol, exact token/stop comparison, profile rejection tests,
ProjectAtlas/runtime checks and updated limits/journal/status. Do not publish without
explicit approval.

### 0.2 service alpha

Implement `S1` after approval, then the smallest OpenAI-compatible loopback service.
Streaming must emit at GPU readback, admission owns prefill, and overload/cancellation
are tested before performance claims.

### 0.3 compatibility and operations

Implement `C1`, `Q1` and `D1` evidence gates. Expand the public validity domain only
for matrix cells with exact revision, repeats, raw data and correctness.

### 0.4 research

Run `B7`, large-model width/cost/family studies, memory pressure/KV allocation and
profiler work. Native loops, draft models and kernels remain downstream of those
measurements and their kill criteria.

## Definition of done for this review intake

- Every review point is mapped above or rejected with evidence.
- Confirmed release blockers have regression tests and post-fix verification.
- Approval/resource/external tasks remain visibly open with a concrete gate.
- Completed entries leave the backlog and move to durable evidence documents.
- No estimated speedup is presented as a measured result.
