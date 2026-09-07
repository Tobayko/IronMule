# PROD3 — portable automatic calibration

Implementation specification, registered before its native validation. This is
the next product stage of PROD1-C, not a replacement for the full product/RL goal.

## Ownership

- A calibration job owns readiness waiting, isolated reference/candidate trials,
  bounded budgets and a complete metadata journal. It never changes OS power
  settings, kills unrelated processes or uploads evidence.
- A fixed evaluator owns validity checks and the verdict. The proposer cannot
  change its thresholds or silently shorten a schedule after observing results.
- The serving reference remains available independently. This first installable
  job is invoked through `ironmule optimize run`; later background integration
  must arbitrate with ordinary serving before sharing its GPU worker.
- A completed calibration is not by itself universal hardware qualification or
  permission to route unseen configurations. RL, configuration interactions and
  independent deployment confirmation remain explicit product requirements.

## Installable shared infrastructure

Package the existing `friday_evidence` helpers without changing their historical
source bytes. Add generic metadata event storage and shared pure statistics there.
The unsealed engine benchmark reexports that single statistics implementation;
no new divergent statistics copy is created. Historical H1/H2 registry/storage
entrypoints are not reinterpreted as product events. Product events use a separate
database, schema and application ID with a verified append-only hash chain.

No model output or user messages enter that metadata journal. It stores model/
environment/source identities, fixed test-prompt hashes, output/token hashes,
counts, durations, memory, readiness and explicit failures/deferred states.

## Readiness

Use public Foundation `NSProcessInfo` APIs through the system Objective-C runtime,
not a compiler subprocess. AC, CPU and memory/swap probes are read-only. Unknown
or malformed telemetry fails closed. Ready requires three eligible observations,
at least five seconds apart, with no gap over fifteen seconds and no stale sample
over ten seconds. Any ineligible observation resets the streak.

Defaults remain: normalized one-minute CPU load at most 0.8, Low Power off,
thermal state nominal/fair (0/1), and reported memory free percentage at least
10%. Hardware/readiness policies are immutable inputs to the job. Busy waiting
is recorded separately and does not count as an executed model trial.

The controller checks readiness before loading weights and before/after a cell.
A post-cell failure invalidates that attempt; it is not erased and its samples
cannot be pooled with a retry. A bounded wait ends in `deferred`, leaving the
reference unchanged. Pause/cancellation is observable and closes owned workers.

## Native evidence validation

The initial installable calibration uses the same fixed apples prompt as PROD2
and its three output limits (1/8/32), with three fresh workers and rotated limit
orders. Each cell runs two warmups per arm (first traced, second untraced), two
A/A reference calls and AB/BA pairs: 90 total calls per model, no sampling or
shared cache. This is a new protocol and never pools the earlier pilot attempts.
The primary metric is complete HTTP completion wall time. The frozen evaluator
uses three worker-level median ratios, 10,000 seeded cluster resamples and an
effect floor of max(2%, three times the maximum A/A relative deviation).
All call descriptors, exact output hashes/counts, real forward counts and
resource checks must match. A signal is a calibration result only; independent
deployment confirmation and broader workloads are still required for activation.

One unchanged shared BudgetGuard covers the measurement schedule: 120 seconds
conservative inference-work accounting, 6 seconds continuous, at least 4 seconds
break after each call, 25% in a 60-second window, 60 seconds between fresh
workers and 20 minutes measurement wall time. Initial readiness waiting and
identity hashing happen before that schedule, are separately journaled and
bounded; waiting never resets an in-progress measurement budget. Swap growth
must not exceed 256 MiB from the pre-load baseline and peak MLX memory must
not exceed 60% of installed RAM. Missing telemetry invalidates the attempt.

1. Compare direct public Foundation flags with a separate actual system reference;
   test metadata boundary behavior separately from native sensor claims.
2. Verify real SQLite persistence, concurrency, read-only access, tamper rejection
   and failure retention; synthetic metadata here is not model evidence.
3. Build/install the wheel outside the source tree, then execute the product's
   actual local Gemma model path without developer-worktree imports.
4. Validate the automatic wait/ready/deferred lifecycle on real host observations,
   with source/environment fingerprints and owned-process cleanup.
5. Run real paired reference/candidate calibration only when those gates pass.
   Exact tokens/text/finish/counts, complete schedules, repeats, median/spread,
   A/A controls, budgets and model/environment identity remain mandatory.

No new speedup, broad automatic activation or RL claim is established merely by
passing control-plane tests or by an incomplete calibration attempt.

## Implementation/validation amendments

- The native device probe returns `gpu_devices[].metal_support` as a technical
  string, for example `spdisplays_metal4`, not a boolean. The first installed
  attempt reached real stable readiness, then failed before model loading on
  this interface mismatch. Align the identity contract and retain that attempt.
- Terminal events retain safe error type, phase and controlled validation detail.
  Third-party exception bodies remain excluded because they may contain user text.
- Inference and calibration use a shared model-resource lease per product state
  directory; job liveness has its own kernel-held lease. This is not a global
  scheduler across unrelated programs or independently configured states.
- The evaluator checks the recorded resource intervals and worker lifecycle
  timestamps itself. It cannot accept `resource_valid=true` as a substitute for
  actual complete budget and cooldown evidence.
- Source, model and environment files stay frozen during each installed run.
  Package/contract corrections produce a new attempt, never altered historical
  evidence or relaxed thresholds.
