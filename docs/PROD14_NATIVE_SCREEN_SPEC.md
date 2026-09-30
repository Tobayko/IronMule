# PROD14 native variant correctness screen

This is a correctness and lifecycle screen, not a performance study. It makes
no speedup, TTFT, log-probability, activation, or best-baseline claim. Timings
and host/device observations are diagnostic only. Native execution is opt-in
through `--execute`; the controller imports no MLX or model module.

One invocation binds one registered Gemma 3 snapshot (1B, 4B, or 12B) and one
variant: `prefix_reuse` or `current_engine`. It holds the model lease while it
captures runtime, source, installed-package, provider, snapshot and hardware
identity before and after. The report records actual candidate-worker PIDs and
requires it to be reaped before terminal identity and digest evidence is
accepted. Failures retain partial samples and cleanup evidence in the same
append-only journal.

No completed stock study is repeated. The model-specific frozen PROD10 report
supplies exactly four stable `long_8` correctness records. Reuse requires its
passed status, exact model/revision, stable output/token/text/prompt hashes,
counts and finish reason, unchanged before/after provider distribution, and
model, environment and hardware hashes equal to the current pre-run identity.
The oracle file hash is part of the source manifest and report. Its timings are
never imported into this screen or any later speed comparison. If any binding
fails, the screen returns `fresh_reference_required` with a reason and does not
implicitly launch a replacement stock worker.

The prefix arm performs four direct calls (cold plus three required hits),
clears and proves zero entries/bytes, cancels one direct `long_128` request only
after observing its first actual token, requires a cancelled terminal result
with no stored entry, and proves that the same warm worker completes one exact
recovery `long_8`. It clears again, then sends three identical non-streaming
JSON HTTP requests through the normal `ProductService` and loopback server
(cold plus two hits). Thus it performs eight new full generations and one
separate partial cancellation; four existing stock records are reused but are
not counted as newly executed requests. Stock has no 128-token outcome claim.

The current-engine arm performs four direct and three HTTP `long_8` calls.
It is completion-only and reports the worker's current identity, selected knobs
and profile source. Absence of a compatible profile is reported as baseline;
historical scope differences do not exclude this arm. Any runtime fallback is
a failed qualification. Eligibility depends only on its native exact-output
and HTTP gates. This accounts for seven new Engine full outputs (11 including
four reused stock records) versus eight new prefix full outputs (12 including
reused stock) and one partial cancellation. Every new candidate request must
report the hash of its actual rendered prompt IDs and match the frozen oracle.

No artificial generation, CPU, RSS, swap, duty-cycle, pause or total-runtime
limit is introduced. OS safety, explicit cancellation, bounded protocol frames,
exclusive output creation, cleanup, model immutability and actual native GPU
readiness remain mandatory. Passing does not activate either variant; later
paired performance qualification is a separate PROD14 step.

The finalized report separately counts planned/new/reused records and is fully
bound by the terminal journal digest. The underlying IronMule Engine package
is now included in the product code identity through file hashes only; the
controller must not import the MLX-bearing Engine just to collect provenance.
