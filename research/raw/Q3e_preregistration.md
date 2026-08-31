# Q3e preregistration — portable process identity, then one unchanged Q3c run

Written 2026-08-31, after terminal Q3d failure and before any Q3e
implementation or test change. This document and
`Q3e_preregistration.sha256` are frozen before the repair begins. Q3e is one
portability-repair and verification path; it does not reopen Q3d, repeat its
stability gate, relax any safety rule, or authorize automatic promotion.

## Why this exists

Q3d's model-free stability gate passed completely, but its one permitted Q3c
invocation was correctly refused before `subprocess.Popen`. On macOS
26.6.2-arm64, the process-inventory command requested `sid`, which this
system's `/bin/ps` does not support (`rc=1`, `ps: sid: keyword not found`).
Consequently no model was imported, no inference process was started, no
Q3c raw result exists, and there is no Q3c timing or performance result.
Q3d is terminally closed as `Q3C_FAILED`; its gate PASS is retained as context
only and is not a current-model or performance guarantee.

The retained Q3d records are:

- `research/raw/Q3d_stability_20260831.json`, SHA-256
  `4699a49b174db31580a9701ef2075f8b1964d309b0f857dd7779fb230cfccb83`,
  `34,144` bytes;
- `research/raw/Q3d_summary_20260831.json`, SHA-256
  `3b43e267000ba15b9d9079d9f118e59c1cd51dbcdfecc067c20995b01a0a1c3e`,
  `970` bytes.

Q3e does not retry or pool Q3d. A new result must use a new output path and
the Q3e identity below.

## Exactly one ordered path

Q3e has no branch, retry loop, relaxed mode, or alternative experiment:

1. **Repair only the portable process-identity probe and prove it with tests.**
   Remove the unsupported `sid` field from the macOS `ps` command. Keep the
   strict canonical fields `pid`, `ppid`, `pgid`, `uid`, `stat`, `start` and
   `args`. Enrich relevant same-UID, known-process/group rows with the public
   Python API `os.getsid(pid)`. A typed `ProcessLookupError` caused by a
   process disappearing during the snapshot is a known race: mark that row
   `gone` and omit it under the existing snapshot-race rules. Any other
   `getsid` error, malformed value, missing required field, unsupported
   command result, foreign UID, invalid PID/PPID, unknown process state,
   missing parent link, cycle, or identity ambiguity is `unknown` and fails
   closed. The session/group invariant must remain strict; it may not be
   silently dropped because the platform probe changed.

   The repair must remain limited to this portability boundary. It may not
   change Q3c arms, process order, safety thresholds, output protocol, timing
   bounds, identity checks, cleanup rules, statistics, fallback or decision
   criteria. The parent and all model-free tests import no MLX, load no model,
   download nothing and start no inference process.

2. **Run the complete required verification before hardware/model work.**
   Add deterministic adversarial tests for the canonical parser, successful
   `os.getsid` enrichment, `ProcessLookupError` disappearance, non-race
   errors, foreign/root UID, malformed/unknown state, parent-link/cycle and
   process-group cleanup behavior. Add a real macOS no-model integration test
   that spawns a child with `start_new_session=True`, proves
   `os.getsid(pid) == pid`, captures the baseline and exercises cleanup/reap.
   Run the relevant Q3c/Q3d tests and the full non-integration suite serially
   with the existing project environment. Any failed, missing, flaky,
   unreviewable or unknown test result ends Q3e permanently.

3. **Only after steps 1 and 2 pass, invoke the unchanged Q3c harness exactly
   once.** Do not run the Q3d 60-second gate again. The one invocation must be
   offline, use a new exclusive raw-output path, and bind the exact post-fix
   commit, runtime-code hash and local model identity. A preflight refusal,
   process failure, timeout, cleanup failure or incomplete raw result consumes
   the one invocation and ends Q3e; there is no second attempt.

## Exact unchanged Q3c invocation

The Q3c contract is frozen by
`research/raw/Q3c_preregistration.md`, SHA-256
`3bf63ff0dcf442855b6d7b97278fb1d43583a9f18e3f5b6c3caa507582a9ffc5`, and is
not amended by Q3e.

- Use only local `mlx-community/gemma-3-4b-it-4bit`, revision
  `93724907d4ed1745d2fe50baadf3b0b01a65abf2`, manifest SHA-256
  `a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae`.
  Use the existing 322-token prompt, greedy generation and `max_tokens=32`.
  No 27B model, download or installation is allowed.
- Phase R compares untuned `BASE` with the exact Q2 incumbent
  (`compiled_fixed_cache=True`, `head_skip_prefill=True`,
  `readback_every=2`; all other knobs at baseline).
- Phase N compares the same `BASE` with that incumbent plus exactly
  `fused_argmax=True`. Preserve the existing Phase-N condition exactly:
  Phase N runs after a Phase-R performance-only miss when R's safety,
  identity, raw-completeness and cleanup gates passed; any R safety,
  identity, raw, cleanup or unknown-state failure stops before N.
- Each phase uses one existing `ironmule.ab.run` call, six fresh OS
  processes, two warmups, seven measured repeats per arm and fixed alternating
  `AB, BA, AB, BA, AB, BA` order. Phases remain independent and are never
  pooled with Q2, Q3b, Q3c attempts or Q3d.
- Reuse every existing preflight and live safety gate unchanged: AC power,
  low-power off, nominal thermal, known installed memory, exact model
  identity, clean/bound Git, known start swap `<=4 GiB`, free memory `>=35%`,
  three known load samples with `max<=8` and `spread<=2`, live swap
  high-water increase `<=128 MiB`, post-phase free memory `>=20%`, and MLX
  peak and child RSS `<=60%` of installed memory. Unknown power, thermal,
  memory, swap, load, process, timestamp, output, identity or cleanup state
  fails closed.
- Retain the existing process inventory and signed Claude Desktop exception
  exactly. Only a fully verified process inside
  `/Applications/Claude.app/Contents/` with identifier
  `com.anthropic.claudefordesktop`, team `Q6L2SF6YDW`, and first authority
  `Developer ID Application: Anthropic PBC (Q6L2SF6YDW)` may be excepted.
  Claude CLI/server/backend and every known model-token process remain
  blockers. The repaired session identity proof does not weaken ancestry or
  cleanup evidence.
- Keep the existing bounds: `600 s` study maximum, `270 s` per phase,
  `240 s` worker, `35 s` child, and existing cleanup/post-snapshot reserves.
  Keep 10,000 paired bootstrap resamples with seed `20260825`, all exact
  token/physical-token/count/stop/capacity/determinism checks, and the frozen
  Q2 reproduction and candidate-preservation criteria. No timing is valid if
  any safety or identity gate fails.

## Result, fallback and hard stop

Q3e never promotes, routes, persists or activates a profile. A successful
repair/test phase is not a performance result. The single Q3c invocation is
accepted only if its complete raw evidence, identity, safety, cleanup,
statistics and preregistered decisions all pass. Any miss ends Q3e with
`FAILED` or inconclusive status and retains `BASE`/the current Q2 incumbent.

Retain all Q3d and Q3e raw/partial records. Do not overwrite or delete
`Q3c_run1_20260831.json`, `Q3c_run2_20260831.json`, either Q3d record, or the
licensed untracked `research/data/squad-dev-v1.1.json`. No UI, dashboard,
restart, software installation, model download or 27B run is part of Q3e.
