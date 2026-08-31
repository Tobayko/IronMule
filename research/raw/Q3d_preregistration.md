# Q3d preregistration — one recovery path for the Q3c safety abort

Written 2026-08-31, before any Q3d implementation or hardware execution. This
file and `Q3d_preregistration.sha256` are frozen before the first recovery
change. Q3d is a safety/evidence recovery study only: it cannot promote,
route, persist or activate a profile, and it has no UI requirement.

## Why this exists

Q3c has two retained failures. Run 1 was refused before a phase because load
was `8.294921875 > 8`. Run 2 passed preflight but was stopped live after swap
grew by `271,518,269 B` (`258.94 MiB > 128 MiB`) and could not prove cleanup
of the worker process group. These records are not pooled or retried by Q3d.
The raw files remain at `research/raw/Q3c_run1_20260831.json` and
`research/raw/Q3c_run2_20260831.json`.

## Exactly one recovery path

There is no branch, retry loop, relaxed mode or alternative experiment. Q3d
has this one ordered path:

1. **Cleanup proof fix, minimal gate harness and tests, with no model or
   hardware run.** Repair only the proof of process-group cleanup/reap that
   was unverified in Q3c. Add or update deterministic tests for normal exit,
   safety abort, timeout, `SIGTERM`→`SIGKILL` escalation, complete group reap,
   orphan detection, bounded evidence and unknown/error handling. Also add the
   minimal model-free stability-gate harness and tests described below. Its
   parent must be stdlib-only: it may not import MLX, load a model or start an
   inference process. The fix and harness may not relax any Q3c resource,
   identity, output, time or safety limit. Gate implementation and tests must
   finish before the first stability-gate execution. If the fix, harness or
   any required test fails, is incomplete, or is unknown, Q3d ends permanently.
2. **Exactly one 60-second, model-free stability gate.** This gate runs after
   the proof fix, gate harness and tests, and before any Q3c harness invocation.
   The parent is stdlib-only: it imports no MLX, loads no model and starts no
   inference process. It must have AC power, low-power mode off, nominal
   thermal state, start free memory `>=35%`, known start swap `<=4 GiB`, known
   load with `max<=8` and `spread<=2`, and no competing model/inference
   process. It takes one synchronous swap sample at monotonic `t0`, followed
   by exactly `60` scheduled samples at targets `t0+1, ..., t0+60 s`: exactly
   `61` samples total. First-to-last elapsed time must be `>=60.0 s` and
   `<=62.5 s`; every adjacent sample gap must be `<=2.5 s`; every OS command
   has a `1.0 s` timeout. The gate wall deadline is `90 s`, its output cap is
   `512 KiB`, its output is strict JSON written to an exclusive output path,
   and every command result, monotonic timestamp and gap must be known and
   valid. The record binds the exact Git commit, frozen Q3d preregistration
   hash, runtime-code identity and local model-cache identity (model,
   revision and manifest), even though no model is loaded. The observed swap
   high-water increase must be exactly `0 B` (`max(samples)-start`). This gate
   emits no model timings and makes no performance claim. Any failed, missing
   or unknown fact, non-zero swap increase, competing model, malformed record,
   timeout or cleanup problem ends Q3d permanently.
3. **Exactly one unchanged full Q3c harness run, only after gate PASS.** Invoke
   the committed Q3c harness once, with its arms, process order, fresh-process
   count, warmups, repeats, prompt, token limit, identity checks, safety
   checks, metrics and decision criteria unchanged. Bind the run to the exact
   post-fix Git commit, the exact local model snapshot/revision/manifest and
   the complete runtime-code identity. A preflight refusal counts as the one
   invocation; do not invoke it again.

The only permitted implementation work before the gate is the cleanup-proof
fix, the minimal stdlib-only stability-gate harness and their tests. No MLX or
model import may occur in that harness. No model download, software
installation or 27B run is authorized by this preregistration.

## Q3c protocol that must remain unchanged

The one permitted harness invocation uses only local
`mlx-community/gemma-3-4b-it-4bit`, revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`, prompt token count `322`, greedy
generation and `max_tokens=32`.

- Phase R compares untuned `BASE` with the exact Q2 incumbent:
  `compiled_fixed_cache=True`, `head_skip_prefill=True`,
  `readback_every=2`; every other knob is its baseline value.
- Phase N compares the same `BASE` with the Q2 incumbent plus exactly
  `fused_argmax=True`.
- Each phase uses one `ironmule.ab.run` call with six fresh OS processes, two
  warmups, seven measured repeats per arm and fixed alternating order
  `AB, BA, AB, BA, AB, BA`. Phases are independent and are never pooled with
  each other, Q3b, Q3c run 1, Q3c run 2 or another attempt.
- The full-run live swap high-water limit remains exactly `128 MiB`; start
  swap remains known and `<=4 GiB`; start free memory remains `>=35%`,
  post-phase free memory `>=20%`, and MLX peak and child RSS remain
  `<=60%` of installed memory. AC, low-power off, nominal thermal, load,
  process, exact identity, complete raw evidence and cleanup/reap are hard
  gates. No Claude CLI/server/backend is ignored; only the already-defined,
  fully verified Claude Desktop bundle exception may apply.
- The Q3c timing bounds remain one `600 s` study maximum, `270 s` per phase,
  `240 s` worker, `35 s` child, and the existing cleanup/post-snapshot
  reserves. The deterministic paired bootstrap remains 10,000 resamples with
  seed `20260825`, and the Q2 target and preservation criteria remain those
  in the frozen Q3c preregistration.

**Process inventory is also unchanged and explicit.** Reuse the exact signed
Claude Desktop exception only when `comm` is lexically inside
`/Applications/Claude.app/Contents/`, the bundle passes the complete absolute
codesign check, and its identifier is `com.anthropic.claudefordesktop`, team
`Q6L2SF6YDW`, with first authority
`Developer ID Application: Anthropic PBC (Q6L2SF6YDW)`. Claude CLI, server,
backend and every process whose model-token evidence is known remain blockers;
the Desktop exception never masks those model tokens. PID/PPID ancestry must
be reconstructed from the same snapshot, including the snapshot-race rules;
missing parent links, cycles, negative or malformed PIDs/PPIDs, malformed
records, unknown process state or an untrusted/outside-bundle path fail closed.

**Performance-miss handling is explicit:** Phase N runs when Phase R has
passed all safety, exact-identity, raw-completeness and cleanup gates, even if
Phase R misses only its performance reproduction criterion. A Phase-R
performance miss is recorded and prevents a success/promotion decision, but
does not by itself skip the complete Phase N. Any Phase-R safety, identity,
raw, cleanup or unknown-state failure stops before Phase N.

## Decision, fallback and hard stop

Q3d has no automatic promotion or activation. If the cleanup proof, the
model-free gate, either phase's safety/identity/raw/cleanup gate, or any
predeclared Q3c performance criterion fails, the final state is `FAILED` or
inconclusive and the fallback is `BASE`/current Q2 incumbent. A performance
miss is not converted into a gain, and no historical Q2 percentage is
multiplied with a Q3d/Q3c value.

Any safety failure, cleanup failure, unknown evidence, malformed evidence,
resource breach, timeout, orphan, identity mismatch, or missed criterion ends
Q3d permanently. Retain every raw record and partial record; do not delete or
overwrite Q3c run 1 or run 2. The complete Q3d wall bound is at most `720 s`:
`90 s` stability-gate deadline, one unchanged Q3c study of at most `600 s`, and
`30 s` terminal bookkeeping/cleanup reserve (`90 + 600 + 30 = 720 s`). This is
below twelve minutes. No UI,
dashboard or presentation artifact is part of the protocol.
