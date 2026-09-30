# PROD11 native correctness qualification

Status: preregistered harness specification; no native result yet. This study makes
no performance or activation claim.

Run each frozen ProductStore snapshot independently: Gemma 3 1B revision `2d44e83`,
4B `93724907`, and 12B `86cc6a8d`. One fresh controlled child loads one model and
runs stock and candidate in that same process. The public orchard prompt must render
to exactly 1077 canonical IDs; output is greedy and capped at eight tokens.

The finite sequence is one stock warmup plus three stock records; an independent
public-callback checkpoint stopped at exactly N-1; one candidate cold request plus
three exact hits; real-cache identity/error, sequential clone, cancellation and
recovery checks. No 6-second, CPU, duty-cycle, readiness, RSS or swap rejection gate
exists. OpenObservation records resources without controlling execution. The owned
child is always reaped and its actual `poll()`/return code persisted; Ctrl-C is a
controlled cancellation.

The N-1 callback runs after MLX-LM materialises cache state and raises the harness's
own `CheckpointReached` before last-token/decode work. The generator is closed in
`finally`. Per layer, persist only class, `meta_state`, state shape/dtype/raw-bit
SHA-256 and byte counts. Raw bits may be materialised through uint8/NumPy only inside
the child and are never persisted. Output records contain only token/text/combined
output and per-step logprob-byte hashes, counts and finish metadata—never raw prompt,
tokens, text, tensors or scope nonce.

The canonical stored snapshot is first hashed only after the uninstrumented cold
and hit mutation phase, then compared with the independent checkpoint and checked
again after two separately deep-copied caches are consumed sequentially from the
last prompt token. This ordering prevents diagnostic hashing from masking aliasing.
All stock/candidate outputs and logprob hashes must match exactly. A candidate hit
requires `completed`, `cache_hit=true` and `reused_tokens=1076`; baseline fallback
cannot pass as a hit.

Wrong binding, scope and token key must miss. Clear/close must reject stale commits.
Cancellation after the first real generated token must commit nothing, report
`cancelled`, and be followed by exact cold and warm recovery. Entry/byte eviction
uses independently populated real long and short model caches, and concurrent
restore produces distinct cache objects;
model generation from those clones is sequential, never concurrent.

Source, installed distributions (including `ironmule_product.prefix_reuse`), model
metadata, runtime environment and hardware identity are captured before and after.
One EventJournal owner records lifecycle/resource evidence. Any mismatch preserves
the child frame, observations, worker exit and partial top-level report and fails the
gate. Hasher overhead makes all timing diagnostic; no speed claim is permitted.

The controller reuses PROD10's bounded frame reader and owned-process shutdown;
resource sampling continues while each child phase runs. Ready, four stock
records, checkpoint, four candidate records and clone verification have a fixed
order. Each completed phase is immediately journaled, including on later failure.
The child keeps partial results on exceptions and waits for explicit shutdown
after emitting its terminal frame. Cleanup, before/after evidence and the final
report digest are recorded under the same model lease. The controller verifies
all eleven checks, complete output repetitions, actual hit traces, checkpoint identity,
GPU readiness and provider identity; a bare `passed` flag is insufficient.
Installed origin proof includes the actual model-loading policy and every directly
used product/evidence module. No source-tree package fallback is permitted.
The different-key test uses another actually rendered token, not an invented ID.

Exact invocation after installing the current package into the controlled runtime:

```bash
python tools/product_prefix_qualification.py --execute \
  --state-dir "$STATE_DIR" \
  --model mlx-community/gemma-3-1b-it-4bit \
  --output research/raw/PROD11_1B_native_correctness.json
```

Repeat with `gemma-3-4b-it-4bit` and `gemma-3-12b-it-4bit`, using distinct output
files. Existing outputs are never overwritten.
