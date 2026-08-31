# Q3f preregistration — strict attribution of unrelated same-UID processes

Written 2026-08-31, after terminal Q3e failure and before any Q3f
implementation or hardware execution. This file and its SHA-256 companion are
frozen before the repair begins. Q3f is one final attribution-repair path. It
does not reopen Q3d or Q3e, change the Q3c protocol, or authorize promotion.

## Why Q3f exists

Q3e repaired the macOS process probe and completed Phase R against the exact
local Gemma 4B model, with exact token identity and zero swap growth. Its raw
result is nevertheless `FAILED`: cleanup rejected four stable, unrelated
same-UID launchd services that appeared after the worker baseline. The raw
record is
`research/raw/Q3e_q3c_final_20260831.json`, SHA-256
`1df6c81dc824911016e687883c535f1ec314f3e03b51303b04c38ae71bb6f4ea`, size
`2,205,857` bytes. The terminal details are in
`research/raw/Q3e_terminal_result_20260831.md` and its companion hash file.

Q3e's descriptive Phase-R total ratio was `0.857466859207542`, CI
`[0.8551668079699586, 0.8611021999710893]`, or `14.2533140792%` faster total
time. That result is invalid for acceptance because cleanup failed. It is
retained context only and does not alter Q3f thresholds, sample counts,
ordering, statistics or decision rules.

## Exactly one ordered path

Q3f has no retry, branch, relaxed mode, alternate experiment or second study:

1. **Implement only the attribution refinement and prove it without model or
   hardware work.** The only permitted code change is the strict
   `unrelated_new_process` attribution plus the bounded child-creation/no-detach
   guard and ledger specified below. A newly observed same-UID process must not
   be treated as unrelated merely because its final PPID is `1`; every
   condition in the strict rule below must be proven from the guard ledger and
   bounded, valid snapshots. Add deterministic tests for all positive and
   negative cases. The tests must be serial and model-free: no MLX import,
   model load, download, installation or inference process.
   Any scope expansion, failed test, unknown result or incomplete evidence ends
   Q3f permanently.
2. **Run the complete required verification.** Run the relevant Q3b/Q3c/Q3d
   cleanup and process tests, the new adversarial attribution tests, and the
   real model-free macOS `start_new_session=True` cleanup/reap test in the
   existing environment. The full non-integration suite must pass serially.
   The verification must include a real unrelated process created after the
   baseline: it must remain alive during the proof and must not be killed,
   while the worker group is completely reaped. The test must also prove the
   child-creation guard and direct child-start ledger: a simulated spawn or
   detach event is blocked and recorded, while inability to install or read
   the guard is unknown and fails closed. Model-like arguments, an ancestry
   relation, a matching process group/session, changed identity, foreign UID,
   unknown/malformed/racy evidence and zombie state must all fail closed. If
   any required result is missing or unknown, Q3f ends permanently.
3. **Only after steps 1 and 2 pass, invoke the unchanged Q3c harness exactly
   once offline.** Use a new exclusive raw-output path. The one invocation must
   bind the exact post-fix Git commit, clean tree, runtime-code hash and local
   model identity. A preflight refusal, process failure, timeout, safety or
   cleanup failure, incomplete raw record, identity mismatch or missed frozen
   criterion consumes the invocation and ends Q3f. There is no retry and no
   further study after a failure.

## Strict unrelated-process attribution rule

### Child-creation and no-detach guard

Before model loading, the actual `ab._child` execution closure must install the
bounded Python audit guard `ironmule.q3f_child_guard.v1`. Its audit hook must
receive all Python-visible audit events and block and record every process or
session escape event in the operation set: `subprocess.Popen`, `os.system`,
`os.fork`, `os.forkpty`, `os.posix_spawn`, `os.posix_spawnp` and every
available `setsid`/`setpgid` audit event. The guard must also wrap or monkeypatch
the available `os.setsid`, `os.setpgid`, `os.fork`, `os.forkpty`,
`os.posix_spawn` and `os.posix_spawnp` call sites before model loading, so a
Python-visible call is blocked even when its audit event is absent. The ledger
is bounded to at most 32 compact event records of at most 512 bytes each and
includes the event name, operation, monotonic timestamp and block result. A
guard that cannot be installed, cannot observe an available operation, cannot
wrap an available call, overflows, or cannot emit its complete ledger is
`unknown` and fails closed. Any recorded event is a hard failure for the child
and the study.

The successful child raw record must contain exactly this guard version and a
complete zero-event ledger. The worker's direct child-start callback is the
complete child ledger: every started child retains PID, PPID, PGID, SID, UID,
start identity, callback timestamp, guard version and guard-event count. A
missing callback, missing child identity or incomplete ledger is unknown and
fails closed; known PIDs remain in the cleanup evidence.

The implementation must also include a bounded AST/static scan beginning at the
actual `ab._child` execution closure and following only an explicit reviewed
callee/module allowlist. The scan must cover imported and aliased spellings of
the operations above, exclude the legitimate parent-side `ab.run` `Popen`, and
fail if any reachable Python call path is unreviewable. Tests must assert the
scan, the audit hook and the wrappers use the same operation set, and must
simulate both a blocked spawn and a blocked detach/session change. Static-scan
failure or an unreviewable dynamic path is unknown and fails closed.

This guard makes no claim to observe arbitrary native C-level extension
syscalls. A natively created process remains subject to the two-snapshot,
group/session, complete-ledger, ancestry, global-inventory and exact-token
gates below; if its origin cannot be attributed under those gates, the result
is unknown and fails closed.

A new same-UID process may be recorded as `unrelated_new_process` only if both
the complete child guard/ledger and both fresh cleanup snapshots are valid and
together prove all of the following:

- its PID, start identity and UID are present and stable in both snapshots;
- it is outside the worker PID, all known-child PID and all nested Q3c-worker
  PID sets;
- it is outside the worker PGID and worker SID;
- the complete same-snapshot PPID graph proves it is neither a worker
  descendant nor an ancestor, and contains no missing parent link or cycle;
- its `args` and `comm` contain no token from the exact case-insensitive
  lexical set `dedupe(KNOWN_INFERENCE_ACTIVITY + ("q3c", "q3d", "ironmule",
  "mlx", "gemma", "huggingface"))`, where `KNOWN_INFERENCE_ACTIVITY` is the existing runtime
  constant (currently `("mlx", "llama", "ollama", "vllm", "gemma", "qwen")`).
  The set must be shared by the guard/cleanup implementation and tested for
  exact equality; no `HF` shorthand, path allowlist or ad-hoc extra token may
  be substituted;
- the global process inventory reports no competing model/inference process.
  For the specific `Python inference process` classification, the exact
  predicate is that the executable or arguments indicate Python **and** contain
  at least one token from the exact blocker set; generic external Python
  without such a token is not automatically classified as inference, but must
  still satisfy every structural unrelated-process gate;
- its process state is known and non-zombie; and
- all cleanup command results, enrichment values, timestamps and bounded
  records are valid and complete.

The full bounded record, including the reason, guard ledger, direct child-start
ledger and both snapshot rows/metadata, remains in the raw evidence. A stable
PPID-1 process outside the worker group/session may be classified as unrelated
only after the guard ledger proves no worker-created process/session escape and
the direct ledger plus complete ancestry proves it was never a worker child.
There is no path allowlist and no blanket exception for launchd, desktop
applications or any process name. A known child, matching group/session,
ancestry relation, model token, an executable/argument indicating Python
together with a blocker token, guard event,
unknown/malformed/racy evidence or any identity ambiguity remains a hard
failure. Unrelated processes are never killed by this attribution rule.

## Unchanged Q3c invocation

The only permitted hardware/model run is the existing Q3c protocol frozen by
`research/raw/Q3c_preregistration.md`, SHA-256
`3bf63ff0dcf442855b6d7b97278fb1d43583a9f18e3f5b6c3caa507582a9ffc5`:

- local model only: `mlx-community/gemma-3-4b-it-4bit`, revision
  `93724907d4ed1745d2fe50baadf3b0b01a65abf2`, manifest
  `a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae`;
- existing prompt token count `322`, greedy generation,
  `max_tokens=32`, exact token/physical-token/count/stop/capacity/
  determinism checks;
- Phase R: untuned `BASE` against the exact Q2 incumbent
  (`compiled_fixed_cache=True`, `head_skip_prefill=True`, `readback_every=2`);
- Phase N: the same `BASE` against that incumbent plus exactly
  `fused_argmax=True`;
- each phase: one existing `ironmule.ab.run` call, six fresh OS processes,
  two warmups, seven measured repeats, fixed `AB, BA, AB, BA, AB, BA` order;
- unchanged AC, low-power, thermal, free-memory, swap, load, process,
  cleanup/reap, RSS and MLX peak gates, including live swap delta `<=128 MiB`;
- unchanged `600 s` study, `270 s` phase, `240 s` worker and `35 s` child
  bounds, and unchanged 10,000-pair bootstrap with seed `20260825`;
- unchanged Q2 reproduction target `0.8568` (CI must contain it, absolute
  median difference `<=0.03`, CI high `<1.0`) and candidate preservation bar
  (CI high `<1.0`, median no more than `0.005` above the replicated Phase-R
  ratio).

Q3f must not pool Q3e's descriptive values or any Q2/Q3b/Q3c/Q3d observation.
The existing fully verified Claude Desktop bundle exception and all model-token
blockers remain unchanged. No Q3d stability gate is repeated.

## Decision, fallback and hard stop

Q3f never persists, routes, activates or promotes a profile. If attribution
repair, any test, preflight, safety gate, exact identity, cleanup/reap,
completeness check or frozen performance criterion fails, the terminal state is
`FAILED` or inconclusive and the fallback is `BASE/current incumbent`.

After the single Q3c invocation, Q3f ends permanently regardless of outcome.
There is no additional experiment, retry, pooled estimate or relaxed cleanup
mode. No download, installation, restart, 27B model, UI or dashboard work is
part of Q3f. The outer bookkeeping reserve is at most 30 seconds after the
unchanged 600-second Q3c study; no separate model-free stability gate exists.
