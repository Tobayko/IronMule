# PROD8 — exact 12B long-context product integration screen

Registered before native execution.  This is one bounded correctness and
integration screen after a terminal-valid 12B PROD3 calibration; it is neither
a benchmark nor an admission, throughput, availability, or activation claim.

## Mechanism and scope

The existing real-Gemma screen proves short-prompt direct-worker, JSON, and SSE
behaviour, plus cancellation, overflow rejection, and recovery.  It does not
prove that a *valid* materially longer context reaches the exact stock result
through the product transport.  PROD8 therefore uses only the frozen local
`mlx-community/gemma-3-12b-it-4bit` revision
`86cc6a8dedbc456dd0e4af01a9d09f396f77e558`.

The harness creates one deterministic, self-authored, benign public chat
message with a known end marker.  Its actual tokenizer count is computed before
the guarded generation window and must be in the pre-fixed inclusive range
1000--1100 tokens; it is reported only as count and SHA-256 metadata.  The
pre-generation tokenizer-only check found exactly **1077** IDs for the unchanged
88-repeat message. It used the installed tokenizer with `return_dict=False`,
matching MLX-LM's wrapper contract; no MLX import, model load or generation ran.
The stock child must independently reproduce this exact count. Tokenizer,
template and config bytes are bound before and after, in addition to weights.
The
same original structured message objects are given to a fresh stock MLX-LM
reference worker and then to a separate fresh product worker.  Each uses its
normal `apply_chat_template(..., add_generation_prompt=True)` path; the stock
child records the resulting ID hash and count, while the product's exact output
digest binds its corresponding prompt-token count.  `max_tokens=8`, greedy decoding, and no
sampling, speculation, persistent prompt cache, or model overlap are allowed.

One warmup and three recorded requests run in each worker.  Every recorded
product response must exactly match its same-index stock reference in token
IDs, emitted text, finish reason, prompt-token count, and completion-token
count.  The product worker is then exposed over a real loopback HTTP server for
one JSON and one SSE request with the same context.  JSON must match the stock
text, usage counts, and finish reason; SSE must reconstruct the stock text,
carry the expected terminal finish reason, contain exactly one `data: [DONE]`,
and close its finite HTTP connection.  Reports and the journal retain hashes,
counts, timings, identities, resource observations, and terminal error codes
only: no prompt or generated content is stored.

## Fixed gates

The harness may start only after the frozen calibration export for run
`ef5b8f0273404145be3f03a4a793f499` passes its canonical report/event hashes,
the independent frozen evaluator, all 90 correctness rows, resource validity,
and three normal worker exits.  Its measured verdict is `inconclusive`; this is
a correctness prerequisite, not an optimization admission.  The exact cached
model, source (including that export), installed-package, and environment
identities are bound before and after execution.  It uses the existing
readiness, model-lease, EventJournal, installed-package proof, `BudgetGuard`,
and `LoadMemoryGuard` paths.  Stock readiness includes an actual GPU device and
non-zero MLX allocation plus a post-load host-readiness window.  One shared generation budget covers reference,
product, JSON, and SSE calls: at most 120 seconds inference-work, 6 seconds
continuous work, a required 4-second break after every call, at most 25 percent
in any 60-second window, and 20 minutes total wall time.  The schedule fixes a
24-second interval after every generation call (the four-second required break
plus 20 seconds) and a 60-second candidate cooldown between the closed stock
process and fresh product worker; `BudgetGuard` remains the fail-closed check.
Blocking JSON/SSE reads execute in one bounded helper thread while the main
owner thread continues deadline/readiness/memory polling and alone writes the
journal.  The swap increase
from the pre-load baseline must not exceed 256 MiB; actual peak MLX memory must
not exceed 60 percent of installed RAM.  Missing telemetry is invalid.

## Kill criterion

Do not shrink the context range, relax a threshold, reuse a warm/persistent
cache, or rerun for a favourable result.  Lack of readiness is terminally
`deferred`; missing/frozen-identity mismatch, source/environment change,
reference/product mismatch, HTTP/SSE framing failure, worker cleanup failure,
budget breach, swap/RAM violation, or unavailable telemetry is terminally
`failed` and retained.  A context that cannot complete inside the fixed budget
is a failed stop, not a smaller-context retry.
