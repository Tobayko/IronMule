# PROD11 — design for exact identical-prompt reuse

Status: implementation candidate only; disabled and not integrated. No native
correctness, performance, memory, concurrency, or cancellation claim exists yet.

## Narrow contract

`ironmule_product.prefix_reuse` accepts only an identical complete canonical token
sequence of length at least two. It uses stock MLX-LM generation, greedy decoding,
no logit processors, no draft model, no KV quantisation, and the stock prefill step
size of 2048. It performs no radix search, partial-prefix selection, custom prefill,
or first-logit storage.

On a cold request, a caller-created cache is passed to public `stream_generate`.
Its progress callback runs after MLX-LM evaluates the cache for the first `N-1`
prompt tokens and before it processes the final prompt token. The callback deep
copies the complete cache list. The snapshot is admitted only immediately before
the terminal `stop`/`length` response is yielded. Failure or cancellation before
that point cannot publish it.

On an exact hit, the adapter deep copies the stored list and calls stock
`stream_generate` with only the original final prompt token. Thus the final prompt
step and decode order match the cold stock execution. This is a source-derived
hypothesis until the native gates below pass.

Each cache belongs to one backend session. Its immutable opaque SHA-256 binding
must cover model snapshot and revision, tokenizer and chat-template identity,
environment, code, and the fixed options. A private, bounded 16--256 character
scope nonce is retained but
never included in repr, errors, or stats. Every restore, cold epoch, and commit
must present both values; mismatch is a cache miss/skip, so accidentally attaching
the cache to another model or tokenizer cannot reuse state. The whole token tuple is the entry key;
tokens and snapshots are likewise absent from diagnostics. There is no global
default cache or new authentication mechanism.

Entries use the sum of actual `cache.nbytes`, with maximum entry and byte counts.
Oversize snapshots are skipped and LRU entries evicted; auxiliary-cache admission
checks `nbytes` before taking its final ownership copy and never stops model
generation. The extra commit deepcopy ensures the stored object graph has no
mutable caller owner; its cost is reported separately. `clear` and `close` advance an epoch so an older
pending cold snapshot cannot commit afterward.

MLX-LM reports prompt throughput using the one-token hit suffix. The adapter reports
the real full `prompt_tokens`, sets full-prompt `prompt_tps` to `None`, and retains
the suffix number only as `suffix_prompt_tps`. Token, text, log probabilities,
finish reason, generation count/rate, and peak memory are forwarded unchanged.
Deepcopy/capture duration is host setup, not GPU copy time. The later benchmark must
include it in total request and amortised session wall time.

Every request exposes a sanitized `last_trace`: terminal lifecycle status,
fixed failure type, hit/stored flags, reused-token count, fixed fallback code, and
restore/capture/commit host times. A short non-blocking preparation claim resets it
before input validation and releases immediately afterward,
so an older hit cannot be attributed to a rejected request. Normal terminal frames
set `completed`; explicit generator close sets `cancelled`, exhaustion without a
terminal frame sets `incomplete`, generation exceptions set `failed`, and invalid
requests set `rejected`. Concurrent busy errors leave the active trace unchanged.
No exception text enters the trace. Auxiliary errors
remain visible, preventing a failed restore or commit from being misclassified as a
qualified hit. The trace contains no tokens, text, tensors, binding digest, or scope
nonce.

An adapter is non-blocking single-flight: only one generation may be actively
iterated on that adapter. A busy preparation or iteration raises a fixed public
error without overwriting the active trace. The long-lived claim occurs on first
iteration, so constructing and discarding an unstarted iterator holds no resource.
Cache restore
concurrency remains testable through separate adapters sharing the same private cache.
As with MLX-LM's generator API, an accepted iterator must be exhausted or explicitly
closed so its single-flight claim and generation resources are released.

## Native qualification gates

The implementation remains off until a new, preregistered real-hardware study:

1. Interrupts an independent checkpoint path exactly at `N-1` and compares the
   canonical and candidate cache: every layer class and `meta_state`, all valid KV
   region bits, and all prompt and generated tokens must match exactly. Gemma 3's
   `RotatingKVCache` values `keep`, `max_size`, `offset`, and `_idx` are mandatory.
2. Runs uninstrumented cold and hit requests, then hashes the unchanged canonical
   cache and stored snapshot *after* mutation. Hashing before generation is
   insufficient because it can hide aliasing.
3. Proves wrong binding, scope, or full-token key is a miss; concurrent hits receive
   independent cache objects; cancellation publishes nothing; clear/close reject
   stale commits; eviction returns accounting to its bound.
4. Measures paired stock, cold candidate, warm candidate, and the compatible best
   IronMule incumbent on real hardware. Report warm/cold and amortised 1-, 2-, and
   8-request sessions, TTFT, complete wall time, capture/clone/lookup host setup,
   cache bytes, process/MLX memory, median and spread under balanced ordering.
5. Rejects activation on any token/state mismatch, mutation of canonical state,
   cross-scope disclosure, unbounded lifecycle, cancellation defect, or absent net
   session benefit after cache construction and cloning.

E13 remains relevant only as historical evidence for a separately declared chunked
session plan. It does not qualify this MLX-LM 0.31.3 identical-prompt candidate or
license automatic plan selection. B39d's 12B server ratios are historical targets
from a different scope, not a correctness or performance baseline for this feature.
