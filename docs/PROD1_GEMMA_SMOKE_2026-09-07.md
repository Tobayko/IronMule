# PROD1: first real-Gemma product correctness gate

Registered before execution on 2026-09-07. This is a correctness/integration
screen, not a performance benchmark or an optimizer promotion.

- Discover every complete locally cached Gemma snapshot through the MLX-LM
  loader inventory; deduplicate model id/revision. Currently 1B, 4B and 12B at
  4 bits. Never download a replacement or silently skip an available snapshot.
- One model in memory at a time; separate stock-reference and product-worker
  processes. Fixed text prompt: `Write a short sentence about apples.`
- Greedy decoding, no sampling or speculation, maximum 8 tokens. One warmup
  and three recorded requests in each path. Compare complete emitted token
  sequences, final text, token counts and finish reason, not truncated prefixes.
- Then exercise real non-streaming HTTP, SSE, and stop filtering against the
  same loaded product worker. No test double may produce an inference result.
- AC power, offline loading, existing BudgetGuard for inference request wall
  windows as conservative upper bounds; explicit required breaks. Model load
  wall time is recorded separately and is not labelled GPU execution time.
- Record each model's attempt and terminal error before moving to another.
  Any mismatch/error fails the screen. Failed artifacts are never overwritten.
- Report source-content digest before/after, exact model revisions and software
  versions. Dirty-tree engineering evidence cannot activate a tuned profile.
- No effect size, bandwidth efficiency, sustained-load or cross-device claim
  follows from passing this short screen. Long requests and server soak gates
  remain separate required work.

## Amendment before attempt 2

Attempt 1 is retained as failed. Gemma 1B matched all reference tokens and
returned matching JSON/SSE text, but the finite SSE body advertised keep-alive
without Content-Length/chunked framing. The client waited for the 15-second
next-request idle timeout; the conservative request-wall budget correctly
rejected that duration. Fix the server to close the finite SSE response after
`[DONE]`, assert this framing in the screen, and record SSE wall time. No budget,
workload, token gate, repeat count or performance claim is changed.

## Amendment before attempt 3

Attempt 2 passed all three complete local Gemma snapshots at its frozen source
digest. Subsequent transport/context/variant changes require a fresh screen;
they do not inherit that digest's result. Add one real mid-generation
cancellation after the first token (limit 128), require `cancelled` and the
same worker still ready, reject prompt-plus-output context overflow (8192
output tokens plus the non-empty prompt), then verify an exact 8-token
recovery on that same worker. Reference warmups/repeats, HTTP JSON, SSE and
stop checks remain unchanged. Required breaks and budgets remain unchanged.
This extension still cannot make a performance or sustained-load claim.
