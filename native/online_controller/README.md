# Online controller core

This dependency-free CPU library exposes the C ABI in
[`include/online_controller.h`](include/online_controller.h). It controls
only action IDs whose complete execution profiles the caller has validated.
Action zero always preserves the caller's reference profile. It imports no
model, tokenizer, Python, GPU framework or network client.

Build and test without registry access:

```sh
cargo build --release --offline --manifest-path native/online_controller/Cargo.toml
cargo test --offline --manifest-path native/online_controller/Cargo.toml
```

The Python owner supplies all identity and schema digests, validates the
checkpoint envelope with `friday_evidence`, and serializes access to each
handle. It must use a nonblocking admission guard on the request path and
return the reference when the controller is busy. The core does not contain
locks. C pointers must be valid and live; this is a trusted local ABI. A
release panic aborts the process, so the library does not promise recovery
from programmer errors or invalid pointers.

## Learning and qualification

Six table cells distinguish one, two, or at least three ready requests, each
with requested output at most 32 tokens or greater than 32. These are known
before execution. No late output length or unreliable OS-load measurement
is a feature. Missing features, unknown bits, unsafe phases, foreign
identity, missing workload identity and an invalid eligibility mask all
select the reference without creating a ticket.

Each effective, attributed ordinary result updates one exponentially
weighted cost mean and count. The objective is a frozen caller contract;
the controller cannot change it. Fully attributed failures may carry the
contract's failure penalty. Interrupted, mixed, overridden, correctness-
failed and resource-invalid work never trains. Late results retain their
original ticket and workload cell. Duplicate feedback is rejected.

A candidate changes one cell and is frozen after the configured amount of
new operational evidence. At most one candidate and one comparative
triplet are outstanding. A triplet issues two incumbent tickets (A/AA) and
one candidate ticket (B), alternates their execution order, and binds the
same exact workload digest. The harness must execute independent workload
groups through the same instrumented path, compare complete outputs and
state, account for all execution and switching costs, and ensure request
fairness and service limits. A caller-provided numerical label alone does
not establish any hardware correctness or speed claim.

Ordinary results cannot qualify a candidate. Comparative results never
train the working table. There is no statistical early promotion: a trial
ends at its fixed sample count, or fails early for a regression, noisy A/A,
missing result or invalid execution under the default strict protocol.
Pair IDs increase across restarts.

For each complete triplet, let `R = B / mean(A, AA)` and
`N = abs(A - AA) / mean(A, AA)`. Any `R > 1 + max_regression`
or `N > max_regression` rejects the candidate. A paired win requires
`R + N < 1 - min_gain`. At the fixed sample count, the exact one-sided
Clopper-Pearson upper bound for the failure frequency must be below 0.5,
and `mean(R) + mean(N) < 1 - min_gain` must also hold. Each trial uses
`family_alpha / max_trials`; the fixed lifetime trial cap is persisted.

`upper_bound` reports that failure-frequency bound, not an interval on the
population mean latency ratio. The binomial interpretation requires
independent paired groups. Shared load, thermal drift and dependent service
windows may violate that assumption; offline fresh-process end-to-end
qualification remains a separate hardware-evidence requirement. These
engineering gates make no lifetime reliability or universal speed claim.

The optional `imc_create_with_protocol(config, protocol)` and
`imc_restore_with_protocol(config, protocol, input, length)` select an
immutable protocol: 1 is strict; 2 is adverse reference-noise scoring.
Existing create/restore entrypoints exclusively select protocol 1. Invalid
protocol values and checkpoints from a different expected protocol are
rejected. No live handle can change protocol.

Protocol 2 retains every complete window through the same fixed sample
count, including noisy or losing windows. Actual `R > 1 + max_regression`
and correctness, resource, service and numerical failures still reject
immediately. When `N > max_regression`, the scored ratio is
`S = 1 + max_regression`; otherwise `S = R`. The full measured `N` remains
an adverse surcharge. Wins require `S + N < 1 - min_gain`; the final
gate uses the same failure bound and `mean(S) + mean(N) < 1 - min_gain`.
No window is discarded or retried. Raw `R` and `N` arrays remain in
candidate and qualification state; one shared calculation checks live
promotion and restored proofs. Status `mean_ratio` reports `mean(S)` for
protocol 2 and `mean(R)` for protocol 1; `aa_noise` always reports raw
`mean(N)`.

This is a different, less terminal reference-noise rule, explicitly opt-in;
it does not assert equivalence to strict qualification or guaranteed alpha
coverage under dependent windows. The scored ratio is never smaller than
the valid raw ratio, and every excessive-noise window is a loss. Identity
and objective binding must include the chosen protocol in the caller.
Strict checkpoints retain the byte-compatible `IMCSTATE01` format;
protocol 2 uses `IMCSTATE02` with the same field layout. Completed windows
of an unfinished candidate survive restore unchanged. Only an outstanding
incomplete triplet invalidates that candidate, as in protocol 1. Offline
execution authority remains absent from both checkpoint formats.

## Budget and persistence

Every ordinary decision accrues `exploration_ppm / 1,000,000` credits,
capped at `trial_budget` credits. A sampled alternative costs one credit;
the two extra comparative executions cost two. Exploration is capped at
5 percent. Technical filters precede the draw, and the recorded propensity
reflects the actual eligible alternatives and draw. Frozen prediction is
read-only and creates no credit, ticket or learning opportunity.

The numerical state contains active and fallback policies, per-cell raw
qualified ratio/noise arrays, working statistics, one optional frozen
candidate, partial comparisons, bounded pending tickets, RNG, counters,
budget and persistent faults. Serialization writes explicit little-endian
fields rather than native padding. Its internal checksum detects accidental
corruption; SHA256 identity/integrity belongs to the canonical Python
envelope. Restoration validates strict configuration, finite numerical
state and qualifications before accepting a policy. Outstanding ordinary
tickets become counted abandonments; a partial comparative triplet
invalidates its candidate. Restart never turns unfinished work into labels.

Performance drift and a current-policy hard service/resource failure
withdraw the active profile and permit new evidence
within the remaining trial cap. Correctness and numerical faults quarantine
an action. Operator kill blocks all adaptation and persists. None of these
safety faults has an automatic clearing API.

`model_bytes` reports the in-memory working table and numerical policy
structures. `runtime_bytes` includes the bounded pending and qualification
storage; `state_bytes` is the serialized checkpoint size. Shared-library
file size and Python adapter memory are additional costs. No allocation
occurs in the Rust decision or read-only prediction implementation;
checkpointing and status are cold operations that allocate serialization
buffers. This is a source-level bound, not a measured whole-stack overhead
claim. Deterministic CPU tests establish lifecycle behavior; they do not
establish a GPU speedup.

## Explicit offline comparison authority

The optional `imc_training_begin`, `imc_training_end`, `imc_training_status`
and `imc_begin_training_pair` entrypoints reuse the same frozen-candidate
comparison implementation. They do not change live exploration or credit
accrual. One idle, enabled, non-killed handle may receive one even grant
between 2 and `min(32768, 2 * min_pairs * max_trials)` extra-execution units.
Only `imc_begin_training_pair` consumes them: exactly two units after all
three prospective tickets are successfully reserved. `used` counts these
conservative reservations, including later failure or cancellation; actual
executed arms require separate evidence. Failed measurements have no refund.

The same identities, cell eligibility, correctness/resource checks, fixed
sample count, multiple-testing budget and immutable candidate apply. The
caller must separately bound total hardware time, memory and cancellation;
an execution grant alone proves neither safety nor performance benefit.
End and operator kill revoke the unspent remainder while retaining granted
and used counts for that handle's audit. Exhaustion sets active to zero.
No grant, remainder or spent offline authority is serialized. Restoration
preserves numerical and qualified states but starts with zero offline
authority. Existing configuration, status and checkpoint wire formats are
unchanged; the new functions are independently optional for older callers.
