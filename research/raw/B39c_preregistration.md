# B39c — Memory-order RSS diagnostic

Experiment ID: B39c
Parent: B39
Registered: 2026-08-28
Status: diagnostic only; no performance qualification or activation

B39c investigates the B39b pilot's order-confounded RSS ratio. It is a new
diagnostic and does not reuse or pool B39b timings or evidence. The exact local
Gemma 3 12B snapshot and the frozen X1-strict workload remain unchanged: six
fresh q0-q5 requests, `StrictOneShotPlan` per request, max_tokens 48 and greedy
decoding.

Exactly two new serial blocks are executed, with one fresh operating-system
process and one model load per arm:

1. `A B D C`
2. `C D B A`

Each child performs two warmups and one measured repeat. The child records
phase-aligned RSS and MLX active/peak checkpoints, complete outputs and stops,
correctness, identity, swap, crash and post-state evidence. B39, B39a and B39b
remain unchanged and their hashes are bound into every child and parent result.

All existing absolute safety gates remain fail-closed: pre-spawn and
process-start-to-end Swap delta must be at most `268435456 B`, every RSS/MLX
checkpoint must remain at or below 12 GiB, and missing instrumentation,
correctness or identity evidence, timeout, crash, relevant crash report or
residual model process stops the diagnostic. Relative RSS ratios do not abort
after the first complete block; they are recorded so that the reversed block
can test the order mechanism. No result can activate a route.

The prospective classification is computed only after both blocks and all
eight children pass their hard gates:

- `RSS_ORDER_PAGE_RESIDENCY_CONFOUNDED` iff block `ABDC` has RSS `C/A > 1.10`,
  block `CDBA` has RSS `C/A < 1/1.10`, both first/last RSS ratios (`A0/C3`
  and `C0/A3`) are `< 1/1.10`, both MLX `C/A` ratios are within
  `[1/1.10, 1.10]`, and both RSS/MLX `D/B` ratios are within that band.
- `CORE_RSS_SIGNAL_REPRODUCED` iff RSS `C/A > 1.10` in both blocks. This is
  still not an allocator or performance claim.
- Otherwise: `INCONCLUSIVE`.

The top-level status is `MEMORY_ORDER_DIAGNOSTIC_COMPLETE` only for two
complete blocks and one of the two classifications above. `valid_for_performance`
and `activation_allowed` are always false. Diagnostic timing snapshots may be
retained in raw child evidence but are not summarised or interpreted.

No automatic B39 main run, retry, profile activation or routing follows from
B39c. A new clean-state preflight is required before any future hardware run.
