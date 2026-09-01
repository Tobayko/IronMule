# Handover — branch `r11-swap-gate`

## Current 2026-09-01 — Q3f terminal result

Q3f consumed its single permitted path and is terminally `FAILED`. The retained
raw result is `research/raw/Q3f_q3c_final_20260901.json`, SHA-256
`e82accdbd52857e6201fa2b34984765e61658ecfd3956d5c903a49c1e6de70a9`, size
`2,487,533` bytes. The redacted terminal note is
`research/raw/Q3f_terminal_result_20260901.md`; its companion SHA-256 is
`a766f768d045045fca4462c8dfb4c0a8f63e46cd1c8c528d628a3f4421aecf1b`.

All 14 preflight checks passed: AC power, free memory `67%`, nominal thermal,
Low-Power-off, exact local Gemma 4B identity and clean runtime binding. Swap
was `2,609,643,520 B` and stayed constant across 25 samples (`delta=0`, no
sampler errors). The phase worker exited with status `2` and
`ABRunError: child 0 start callback failed`, before any child marker, ledger,
timing, token, Phase-N or accepted performance evidence existed. The exact
lower-level cause is not preserved; a process/command visibility race is only
an unproven hypothesis.

Cleanup v2 reaped the worker and found no remaining group member or descendant
in two valid independent snapshots; no kill was needed. The result still failed
closed because guard/ledger evidence was unavailable. Same-UID Spotify Helper
PID `52017` was outside the worker group and was not killed, but could not be
accepted as unrelated without that missing proof. No Q3f orphan remained.

Final state: `FAILED`, `promotion_allowed=false`, fallback
`BASE/current incumbent`. There is no speed claim, no proof that historical Q2
values were reproduced, no retry and no pooling with Q3e/Q3d/Q3c. Do not run
Q3f again. A child-visibility hardening would require new explicit
authorization and a new preregistration. No 27B model, download, installation,
restart, UI work or automatic activation occurred.

## Current 2026-09-01 — Q4 final H17 contract frozen

The final Q4 correction is the H17 protocol in
[`research/raw/Q4_preregistration.md`](research/raw/Q4_preregistration.md), with its
SHA-256 `4c818404a50ca5102f1be8d48399af42f494eb7afb4d23d27e2a59481f4d203c` recorded
in the companion file. A complete trajectory is exactly steps 0--16:
11 `KNOB_DELTA` evaluations, 5 plan-matching `STRATEGY_SELECT` evaluations for the
final knob, and terminal `REVALIDATE` at step 16. Partial aborts are terminal at the
current step and never complete. Each context's method budget is exactly 16 candidates
(11 + 5); shared BASE is external.

Stage-2 completeness is the full exact cross-product of all 12 knob actions with the
five plan-matching safe strategies (60 cells/context), collected as 12 separate
knob-anchor phases, each with five fresh processes (one per strategy), two warmups and
five repeats, and a 600-second phase bound. S11/S12 remain two separate risk probes
outside reward/OPE/policy support. Each trajectory is split into separately
preregistered/user-started knob (11 children, 1320 seconds), strategy (5 children, 600
seconds) and revalidation (1 child, 120 seconds) subphases, each under 1800 seconds;
trajectory/context/study/time digests remain stable across subphases.

The Q4 minimum is 24 entirely new contexts, 72 H17 trajectories and 1224 transitions;
historical Q2/B35/B36 remain Q3-namespaced and E11 remains `LEDGER_ONLY`. OPE and
ensemble folds use complete context/group hashes with all trajectories of a context
co-fold. WIS clip is 10; grouped five-fold DR is TRAIN/VALIDATION-only, while sealed
holdout is direct-panel-only. The final result remains a local-pilot claim; foreign
evidence is `MISSING` pending Ed25519 verification through a user-approved local trust
store. Knob FQI and strategy immediate-ridge heads are separate; the hybrid never
scalar-adds them. The exact state vector is intercept, model-size, memory, GPU-core,
prompt, output, concurrency, objective, plan, workload-stratum, arrival-pattern,
current-action and scaled remaining budget, with no unlisted interactions. Validation
uses seeded uniform safe without-replacement propensity `1/remaining`; holdout uses
frozen lexicographic order with propensity 1 and direct scoring only. No implementation,
hardware/model run, download or commit occurred.

## Current 2026-09-01 — Q4 RL-first contract frozen

This initial H13 draft is superseded by the final H17 contract above and retained only
as historical provenance. The current protocol is the H17 preregistration at
`research/raw/Q4_preregistration.md` (SHA-256 in its companion file) and the plan at
`docs/Q4_IMPLEMENTATION_PLAN.md`.

## Current 2026-08-31 — Q3e terminal; Q3f is frozen as the only next path

Q3e is terminally `FAILED`. Its retained raw result is
`research/raw/Q3e_q3c_final_20260831.json`, SHA-256
`1df6c81dc824911016e687883c535f1ec314f3e03b51303b04c38ae71bb6f4ea`, size
`2,205,857` bytes. The redacted terminal note is
`research/raw/Q3e_terminal_result_20260831.md`, SHA-256
`fd89e23945315597476854843df1140a5c3e35ebf1aaf9a737c80d5ebf4fdfaa`.

All 14 preflight checks passed. Phase R completed with exact token, physical
token, count, stop, capacity, prompt, decode-step and determinism identity;
swap stayed at `2,643,334,266 B` with zero delta. The descriptive incumbent /
BASE ratio was `0.857466859207542` (95-%-CI
`[0.8551668079699586, 0.8611021999710893]`, `14.2533140792%` faster total),
but it is not an accepted performance result because cleanup conservatively
rejected four stable unrelated same-UID processes: PID `28095`
`extensionkitservice`, `28209` STARFACE `HeadsetXPCService`, `28636`
`mdworker_shared` and `28964` `AXVisualSupportAgent`. They were outside the
worker group/session and ancestry and were not killed. Phase N did not run;
fallback remains `BASE/current incumbent`, with no promotion or activation.

The final frozen next path is
`research/raw/Q3f_preregistration.md`, SHA-256
`345c63cba5f019ab0314761404f7de398ceee876ffcee82d80c3578f9db8e31b` (companion
SHA file present). Q3f allows only strict `unrelated_new_process`
attribution, a pre-model-load bounded Python child-creation/no-detach guard and
complete direct child-start ledger, and two valid snapshots with stable identity,
separation from worker/known/nested PIDs and PGID/SID, complete non-ancestry, no
model or inference tokens, known non-zombie state, no competing model process
and full bounded evidence. The guard must block/record process and session
escape operations; successful child records contain its exact version and zero
events. Any ambiguity, model-like process, guard event, ancestry/group/session
relation, unknown/malformed/racy state or cleanup uncertainty remains a hard
failure. Tests must include a real unrelated process created after baseline,
prove it is not killed, and prove the worker group is fully reaped. Only after
the serial model-free test suite passes may exactly one unchanged offline Q3c
run occur. No Q3d gate, retry, pooling, download, installation, restart, 27B,
UI or automatic promotion is allowed.

The guard contract is `ironmule.q3f_child_guard.v1`: it is installed in
`ab._child` before model loading, blocks/records process and session escape
operations, and requires an exact zero-event ledger on successful children.
The direct child-start callback remains the complete child ledger. The exact
case-insensitive lexical blocker set is
`dedupe(KNOWN_INFERENCE_ACTIVITY + ("q3c", "q3d", "ironmule", "mlx", "gemma", "huggingface"))`;
static and adversarial tests must assert equality. A stable PPID-1 process is
never accepted as unrelated without this guard/ledger proof plus both valid
snapshots.

The guard starts at the actual `ab._child` execution closure, follows only a
reviewed callee/module allowlist, excludes the legitimate parent-side
`ab.run` `Popen`, and fails on any unreviewable reachable Python path. It
receives all Python-visible audit events and wraps available `os` process/
session calls; arbitrary native C-level syscalls are not claimed observable
and remain subject to the strict snapshot/group/session/ledger gates. The
Python-inference predicate is executable/args indicating Python **and** at
least one exact blocker token. Generic external Python is not automatically
inference, but still must satisfy every structural unrelated-process gate.

The older Q3e-next-path paragraph below is retained as history; it is
superseded by this terminal Q3e result and the frozen Q3f path above.

## Current 2026-08-31 — Q3/Q3a

- **Commits:** Der Q3a-Code stammt aus Commit `0ec9237` (`fix: read macOS 26 q3a gates`); Versuch 3 wurde auf Docs-`HEAD` `28b2ef4` ausgeführt. Spätere reine Doku-Commits ändern den geprüften Code nicht. Die unmittelbar relevanten Vorläufer sind `771d133` und `7a8896f`.
- **Dritter Q3a-Preflight (2026-08-31):** `research/raw/Q3a_attempt3_20260831.json` ist ignoriert und erhalten (SHA-256 `019b1aafbc7342a782476d62e2921e8141b185c50490df4b72d5c0f0fbb9e8a2`). Der Harness endete vor dem Modellstart mit `FAILED`, `BASE`, `promotion_allowed=false` und `partial_children=0`; es wurden keine Modellkinder gestartet und keine Prozesse beendet. Grün: AC, Git-Bindung, installierter Speicher bekannt, Low-Power-off, exakte 4B-Identität `mlx-community/gemma-3-4b-it-4bit` / Revision `93724907d4ed1745d2fe50baadf3b0b01a65abf2` / Manifest `a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae`, Preregistration, Zeitbudget und Thermal. Rot: Swap `1774777794` Bytes (`1692.56 MiB`) über dem 256-MiB-Limit, Loadavg max `6.1792` über dem Limit 4 sowie das Prozess-Gate (`no_competing_model_process=false`). Die redigierte OS-Evidence zeigte zwei echte `Claude`-Executables: PID `55295` mit CPU `4.1%`/RSS `292688 KiB` und PID `55345` mit CPU `2.8%`/RSS `395344 KiB`; Argumente wurden nicht gespeichert. Damit widersprach der aktuelle OS-Zustand der Nutzerangabe, Claude sei beendet; es wurde nichts beendet. Auch nach dem Stoppen von Claude blieben Swap und Load eigenständige Blocker. Sicherste Fortsetzung ist nach gesicherter Nutzerarbeit und Neustart bzw. sauberem Zustand; kein automatischer Neustart.
- **Korrigierter zweiter Preflight:** `research/raw/Q3a_preflight_refusal2_20260831.json` ist ignoriert und erhalten. Der Lauf endete mit `FAILED`, `BASE`, `promotion_allowed=false`, ohne Modellstart und mit `partial_children=0`. Grün: AC, Git, installierter Speicher, Low-Power-off, exakte 4B-Revision `93724907d4ed1745d2fe50baadf3b0b01a65abf2` plus Manifest `a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae`, Preregistration, Zeitbudget und Thermal. Rot: Load `max=21.723`, `spread=0.699`; Swap `1641021440` Bytes und damit über 256 MiB; aktive Claude-Prozessaktivität. `research/raw/Q3a_preflight_refusal_20260831.json` ist ebenfalls erhalten; kein 27B-Modell wurde verwendet.
- **Replay und Fortsetzung:** Offline-Replay ist nur für `BASE` freigegeben. Die Datenbasis reicht nicht für eine adaptive/RL-Aussage; RL ist nicht anwendbar. Q3a wurde wegen der Gates nicht ausgeführt. Fortsetzung ist nur bei AC, Low-Power-off, nominalem Thermal, Load `<=4` und Spread `<=1`, Swap `<=256 MiB`, ohne Modellprozess oder aktive Claude-Aktivität und mit eindeutigem neuen Outputpfad zulässig.
- **Verifikation:** Targeted `127`, breit `336 passed, 1 skipped`, separate Integration `12`, modellfreie Q3a-Suite `26`; `ab`-/`tune`-Selfchecks erfolgreich. Der gepufferte Worker-Output-Cap bleibt der bekannte P2-Backlogpunkt. SQuAD bleibt untracked/lokal, die Lizenzfrage offen, und PR #2 ist owner-only.

## Current 2026-08-31 — Q3d terminal result; Q3e is the only next path

Q3c remains closed as a safety-only failure; it produced no valid performance
result. Run 1, `research/raw/Q3c_run1_20260831.json` (SHA-256
`5270c0f38e50984cd26223aa2a9817982fc5a1861ddbe2caa3cff98393c9e8d5`), was
refused before a phase because load `8.294921875 > 8`. Run 2,
`research/raw/Q3c_run2_20260831.json` (SHA-256
`d94db80402254c87c0e4a0128cf802e1eaa59d42c4459c2f208077f48c38b8df`), passed
preflight but aborted after `105` samples / `27.394551749996026 s`: swap
`2,353,654,661 B` → `2,625,172,930 B`, delta `271,518,269 B`
(`258.94 MiB > 128 MiB`). Cleanup was unverified because TERM and KILL both
returned `PermissionError` and the worker group remained alive. Keep both raw
files; they are ignored local evidence and must not be deleted or pooled.

Q3d's model-free gate then passed, but its one permitted Q3c invocation was
correctly refused before `Popen`. Raw
`research/raw/Q3d_stability_20260831.json` has SHA-256
`4699a49b174db31580a9701ef2075f8b1964d309b0f857dd7779fb230cfccb83` and size
`34,144` bytes; summary
`research/raw/Q3d_summary_20260831.json` has SHA-256
`3b43e267000ba15b9d9079d9f118e59c1cd51dbcdfecc067c20995b01a0a1c3e` and size
`970` bytes. The gate recorded exactly `61` samples, elapsed
`60.020192667 s`, maximum gap `1.013944625 s`, and swap delta exactly `0 B`.
The Q3c pre-spawn baseline failed on macOS `26.6.2-arm64` because
`/bin/ps` rejected `sid` (`rc=1`, `ps: sid: keyword not found`). No model,
MLX import, inference child, timing, identity or performance data exists.
The terminal summary is `Q3C_FAILED`, `promotion_allowed=false`, fallback
`BASE/current incumbent`; the gate PASS is safety context only.

The next and only permitted path is frozen in
`research/raw/Q3e_preregistration.md` (SHA-256
`71901b0d2220d7e9559bad536afaf15d04fac5ea1714f7ceade5bc37811dfd47`) with its
companion SHA file. Q3e may
repair only this portability boundary: remove unsupported `sid`, enrich the
canonical process rows with public `os.getsid(pid)`, treat typed
`ProcessLookupError` as a snapshot race, and fail closed for every other
unknown/error state. Model-free parser/cleanup tests and a real macOS
`start_new_session=True`/`os.getsid(pid)==pid` test must pass before exactly
one unchanged offline Q3c invocation. Q3d is not repeated, no retry or pooling
is allowed, and no download, installation, restart, 27B model, UI or automatic
promotion is allowed.

Der historische Satz unten, dass „only the PR remains open“, beschreibt den älteren E15/PR-Stand und ist durch diese aktuelle Q3/Q3a-Sektion superseded.

Verification completed on 2026-08-30. Everything below is state, not advice:
what is proven, what is committed, what remains local, and what still needs the
repository owner. The E15 fork verification is closed; only the PR remains open.

Two sessions worked on this in parallel. The peer session (`tobiasburandt-ec`) owns the
PR branch `codex/evidence-driven-execution-layer` and did every commit there. This
branch, `r11-swap-gate`, contains the committed code at `b700377`; raw JSON evidence
remains local and ignored by repository policy.

---

## Open right now — start here

### 1. E15 before/after verification — completed

The E15 comparison is complete. The fork-per-block result satisfies the R12 memory
criterion, but it is not a clean A/B speed comparison because the two raw files were
recorded on different commits and both environments were dirty.

- **Before** (in-interpreter, complete): `research/raw/E15_before_fork.json` —
  SHA-256 `4312e3bff94a0982711191faf3b110037d293344ccf3e127acaa9c56128b2ea6`,
  commit `5d2d2f8`, `git_dirty=true`, peaks
  `7067609536 → 7483569616 → 9619556792 → 9619559548 B`, wall `1571.585 s`.
- **After** (forked, complete): `research/raw/E15_after_fork.json` — SHA-256
  `d14875e43ee800d8f1a29af966b8adad56245a414dd204f202a48b81d1f91b5c`, commit
  `b700377`, `git_dirty=true`, PIDs `15489/24645/33850/42483`, 128 runs per
  block, peaks `7067618790/7067610600/7067609606/7067609586 B`, wall `1664.407 s`.
- The after MemoryGate recorded swap deltas `-8/-16/-16/-80 MiB`, no abort, and
  passed the memory kill criterion. Both files have no token, token-count, stop-reason,
  or KV-state deviation against their sequential references; no child crashed.
- The after wall time is `92.822 s` longer than before. Because commit, dirty state,
  load and swap baseline differ, this is an engineering/memory result only, not a clean
  A/B speed claim.

The archived after artifact is
`d1/d14875e43ee800d8f1a29af966b8adad56245a414dd204f202a48b81d1f91b5c-E15_after_fork.json`;
the corresponding entry is recorded in the content-addressed archive manifest.
`research/raw/E15_summary.json` was not overwritten and remains the older summary; it
does not replace or summarize the new `E15_after_fork.json`.

### 2. The PR is not merged, and nobody in either session can merge it

`gh pr merge` is blocked by the permission classifier in the peer session, and this
session declined to run it instead — a right blocked in one session does not become
available by moving the same action to another. PR #2 stood at 22 commits, CI green,
`MERGEABLE`/`CLEAN`. **This needs the repository owner.** Everything else described here
is finished and reviewed.

### 3. Verification note — CPU gate and bench self-check

The standalone `python -m ironmule.bench` self-check exited `0` and was green. The
correct non-integration suite also passed:

```text
/Users/tobiasburandt/Project_Friday/.venv/bin/python -m pytest -n0 -m 'not integration'
250 passed, 13 deselected in 6.22s
```

The unfiltered `pytest -n0` path collected 263 tests, including the 13 local-model
integration tests. It reached 82%, remained in an active CPU/MLX end test, and was
controlled with SIGTERM after the 30-minute limit (exit 143). That is neither a pass
nor a fail and must not be reported as one.

Permanent rule: CPU gates must exclude the `integration` marker. Model integration is
run separately and only with explicit authorization.

---

## What is on this branch

Commit `b700377e83b2eba39c5d66976d01332f8ab57bc6` contains the reviewed code changes
and all three review points. Raw JSON remains ignored and local.

### `ironmule/bench.py` — the swap gate (`R11`)

`swap_used_bytes()` parses `vm.swapusage`. `MemoryGate` replaces a hard-coded
`12 * 1024**3` in three harnesses with a measured condition.

Key decisions, each with a reason that will not survive being guessed at:

- **Swap delta against the run's own baseline, not an absolute.** The machine routinely
  carries GBs of residual swap while completely idle; an absolute threshold refuses every
  run on such a machine. A delta refuses only runs that themselves create pressure.
- **`PEAK_CEILING_FRACTION = 0.6` of installed memory**, derived rather than typed, as a
  coarse backstop against unbounded allocation — not as the measurement gate.
- **`MemoryGate.check()` returns a reason string and records what it saw**, rather than a
  bool. A guard that only aborts leaves a truncated run looking like a short one, which
  is `R10`.
- **`read_swap` and `read_installed` are both injectable.** The self-check replays the two
  real reference cases below instead of inventing plausible ones. `read_installed` had to
  become injectable too: `peak_ceiling=None` means "derive one", so there was no way to
  express "no ceiling" and the inert-gate test silently derived 20 GB and passed for the
  wrong reason.
- **`inert` flag plus a stdout warning** when neither swap nor installed memory can be
  read. Both halves failing leaves a run with no memory condition at all; that must be
  visible afterwards, not inferred from an absence of aborts.

The self-check (`python -m ironmule.bench`) replays: the valid 12B run (peak `17.51 GB`,
swap flat — must not abort), the discarded confirmation run (under every byte ceiling,
swap `+2816 MB` — must abort), a runaway allocation with flat swap, one half missing, and
both halves missing.

**A trap worth knowing**: the constructor consumes one `read_swap()` call for its
baseline. An injected iterator is therefore off by one from what you expect. The
self-check comments this where it bites.

### `research/e14b_arms.py` — fork per block (`R12`), **proven**

`spawn()` follows `e16_replication.spawn` exactly: `--child <json spec>`, one `@@`-prefixed
JSON line on stdout, parent parses, controlled `env` and `cwd`. `--stage` is no longer
`required` because a child is invoked with `--child` alone; the parent path validates it
instead. `TimeoutExpired` books one replicate as crashed rather than taking the whole run
down with an exception.

### `research/e15_service.py` — same fork, **verified**

Same shape, `timeout=3600` because E15 blocks run longer. Also fixes a real defect found
during review: E15's memory branch broke **without** setting `aborted`, so a memory abort
wrote `"aborted": null` and the file then claimed completeness. The completed after-file
now records four forked blocks, the MemoryGate record, and `aborted: null`; the result is
an engineering memory-integrity finding, not a clean speed comparison.

### `research/e12_window_falsification.py` — swap gate only, no fork

Deliberate. See `research/raw/R13_backlog_entry.md`.

### `research/e16_replication.py` — untouched, deliberately

It forks real children already, so its per-process peak is coherent. It has the same
uncaught-`TimeoutExpired` gap; the peer decided not to close it, because E16 is finished
and its 40 replicates ran with it. The comment in `e14b_arms.spawn` names the location so
whoever next touches E16 finds it.

---

## Results this session produced

### `Q2` — the self-tuning loop, first end-to-end run in the project's history

`research/raw/Q2_preregistration.md`, log in `research/raw/Q2_run.log`. All five
preregistered kill criteria passed.

Winners: `compiled_fixed_cache=True` (0.9679), `head_skip_prefill=True` (0.8606),
`readback_every=2` (0.8543). Paired confirmation over 6 real OS processes: ratio
`0.8568`, tokens identical, accepted. Second start loaded the profile rather than
re-tuning.

Two Tier 0 predictions held exactly: `prefill_into_fixed` lost, `speculate_k` was **38%
slower**.

This run found a live defect: the reported gain came from the single-process screening,
not the 6-process confirmation. Fixed by the peer in `0de69b6`; `status()` now also
prints the paired interval, `[5.98%; 14.51%]` for this run.

### `B7` — committed to the ledger by the peer as `a746aa5`

Draft with full reasoning: `research/raw/B7_ledger_entry_draft.md`. Preregistration:
`research/raw/B7_preregistration.md`. Data: `B7_4b.json` (4 blocks), `B7_12b.json`
(1 block, aborted).

`SCALING.md` predicts the grouping gain falls to `0.41` of its 4B value. Measured here:
`0.63`, against the ledger's independent `0.61`. **Both terms of that model are wrong, in
opposite directions, and partly cancel**:

| Growth 4B → 12B, arm A | Predicted | Measured |
| :-- | --: | --: |
| `submission_ns` | 1.41× (layers) | **3.68×** |
| `completion_wait_ns` | 2–3× (parameters) | **1.50×** |

Consequence: `submission ÷ wait` is `1.02×` at 4B and `2.52×` at 12B. The step becomes
*more* host-bound as models grow, which is the opposite of what `SCALING.md` assumes.
Tier 2 (`B8`/`B9`/`B10`) therefore aims at the term that dominates at scale.

**But**: `submission_ns` is not host work. Arm B submits `73.53 ms` then waits `10.11`;
arm A submits `50.85` and waits `48.79`, on identical work. Arm B's window is larger
*because the device runs inside it*. The split measures wall-clock windows, so "how much
of this is Python" is unanswerable with this instrument. That makes `B24` a hard
prerequisite for `B8`/`B9`/`B10`, now recorded in all three.

### `R12` — proven on E14b

```
before (one interpreter):  17.51 → 23.14 GB, abort at block 2
after  (fork per block):   17.51, 17.51, 17.51, 17.51 — four blocks
```

Four distinct pids. `research/raw/R12_12b_proof.json`. 12B is measurable with four blocks
for the first time.

**The best surprise of the session**: forking is not a price paid for correctness, it is
*cheaper*. At 12B, `147 s` per forked block against `157 s` in-interpreter. Allocator
pressure in a shared interpreter costs more than a full process start plus a model load.
E15 now confirms the same memory-boundary behaviour at 4B, as recorded below.

### `R12/E15` — fork-per-block memory follow-up completed

E15 confirms the R12 diagnosis at 4B: the shared interpreter's peak rose across blocks,
while one fresh OS process per block kept all four peaks flat near `7.07 GB`. The exact
raw hashes, PIDs, swap deltas, wall times and provenance caveat are recorded in the
separate follow-up in `research/LEDGER.md`. This does not authorize routing, activation,
or a speed claim.

---

## Things that were believed and turned out to be wrong

Recorded because each was believed confidently, and the same mistakes are available to
the next person.

1. **"The 12 GiB guard fired on an inflated value and truncated 12B."** False for 12B.
   Block 1 has nothing to accumulate, so its `17.51 GB` was genuine. The inflation
   affects blocks 2+. The guard aborted for a poor reason but not a wrong one.
2. **"The backlog is wrong: 27B has 64 layers, not 62."** False, and the mistake is
   instructive. `SCALING.md:71` gives 62 for Gemma 3 27B and `:101` says the whole 4B/12B/27B
   series is Gemma 3. The only 27B cached on this machine is `Qwen3.8-27B-4bit`, which has
   64. **The config you can open is not necessarily the model the document means** — the
   exact size-versus-family confusion `B26` exists to prevent.
3. **"The process counter and the guard abort are new findings."** Both are `M2` and `M3`
   in `research/LEDGER.md:1071` and `:1080`. `AGENTS.md` says to read the recorded dead
   ends first; `docs/BACKLOG.md` was read and the ledger's limitation sections were not.
4. **"R11 will unblock 12B."** It did not. It fixed the *signal*; `R12` fixed the cause.
5. **Peer's**: "E12 is the simple port, E15 is complicated." Reversed. `E15.run_process`
   is character-for-character E14b's; `E12.run_case` takes a **shared** `Harness`.
6. **Peer's**: "The fork adds four model loads." It adds none — `E14b` and `E15` already
   load per block inside `run_process`. The fork relocates a load rather than adding one.

---

## Protocol debt

`B7`'s preregistration was **not committed before the run**, unlike `E14`/`E14b`/`E15`/`E16`.
It was written first, but nobody can verify that ordering, and its hash now covers a
document containing the results too.

What that costs, precisely: any claim of the form *"as predicted in advance"* rests on
trust. What it does **not** cost: the central finding, which compares measurements against
`SCALING.md`'s published `0.41` — committed in this repository weeks before the run.

For the next run: commit the preregistration first, then measure. It is cheap.

---

## Backlog entries drafted here

- `research/raw/R11_backlog_entry.md` — committed by the peer as `5d2d2f8`, later closed
  to Tier 0.
- `research/raw/R13_backlog_entry.md` — **not committed anywhere yet.** Argues E12 must
  not take R12's fork without a granularity decision: forking per case turns 1 model load
  into 26, against a wall limit E12 already has. Its test is cheap — measure E12's
  per-case peaks first and find out whether there is anything to fix at all.

## Discarded data, deliberately kept

`research/raw/B7fix_*_INVALIDATED_swap.json` with a README beside them. Discarded under
`B7`'s own kill criterion 2 (swap delta nonzero), kept because the contamination was
*uniform* and therefore cancels in ratios — a robustness remark, not a result. Never cite
them as evidence.

`research/raw/R11_12b_proof.json` is the fixture `R10` asks for: a file carrying `runs: 2`
that looks complete and is not, produced while testing the very gate that `R10` warns
about. The peer has it archived at SHA-256 `bf83869a39c1ba3e`.

## Where the data actually is — it is not in this commit

`.gitignore:15` excludes `research/raw/*.json` by deliberate repository policy: raw
evidence stays local and only redacted public summaries ship. So **every `.json` named in
this document is untracked** and lives in exactly two places:

1. This worktree, `.worktrees/ironmule-b7/research/raw/` — `B7_4b.json`, `B7_12b.json`,
   `R11_12b_proof.json`, `R12_12b_proof.json`, `E15_before_fork.json`,
   `E15_after_fork.json`, and the two
   `B7fix_*_INVALIDATED_swap.json`. **Deleting this worktree destroys them.**
2. The peer session's content-addressed archive, which has `R11_12b_proof.json` at
   SHA-256 `bf83869a39c1ba3e` and `R12_12b_proof.json` at `1cba53c4087fbade`, both with
   notes. Ask it for anything missing before re-measuring.

The Markdown files beside them — preregistrations, the ledger draft, the backlog entries,
this document — **are** committed, because they are prose rather than raw evidence.

`research/data/squad-dev-v1.1.json` is deliberately left untracked as well. It is
CC BY-SA 4.0 and the repository does not redistribute it; committing it would be a
licence violation, not a convenience. Fetch it with the command in
`research/data/README.md`.

## One environment note

`research/data/squad-dev-v1.1.json` is gitignored (CC BY-SA 4.0, not redistributed) and
was **absent from this machine entirely**, so E14–E16 could not run until it was fetched.
`research/data/README.md` documents the command; the SHA-256 matched.

## Historical handover — Q3b audit and initial Q3c preregistration (2026-08-31)

Q3b Canary 7 is audited and complete as safety evidence. The retained raw file
`research/raw/Q3b_canary7_20260831.json` has SHA-256
`77ebc1ed8af5c1d5b4b064ce95605d3440b6e2fccabcd088d58f0900cdd0eb76` and reports
`SAFETY_CANARY_PASS` / `SAFETY_ONLY`, with
`performance_valid=false` and `promotion_allowed=false`. The exact local Gemma
3 4B revision, manifest, runtime-code hash, AC/thermal/process gates,
residual-swap history, cleanup evidence, and exact token/count/stop/capacity/
determinism values are recorded in `research/LEDGER.md`; the same section
records Total/Prefill/Decode medians and output/decode-step rates. Those Q3b
timings are descriptive only and must not be multiplied with Q2.

At this historical point Q3c was preregistered before implementation or hardware execution:
`research/raw/Q3c_preregistration.md`, SHA-256
`411b3f930fa41128a75fff9bd56bd1fbd04dad56b639e64f94c21bc1f42ad701`, with the
companion `research/raw/Q3c_preregistration.sha256`. It specifies exact local
Gemma 4B, prompt 322, max 32, two independent six-fresh-process `ab.run`
phases, 2 warmups, 7 repeats, AB/BA alternation, Phase R BASE vs. exact Q2
incumbent, and Phase N BASE vs. incumbent plus `fused_argmax`. It freezes the
Q3b safety policy, `600 s` study / `270 s` phase / `240 s` worker /
`35 s` child bounds, exact identity rule, Total/Prefill/Decode and token-rate
metrics with 95% CIs, Q2 target `0.8568` / `[0.8549; 0.9402]`, reproduction
and preservation bars, BASE/current-incumbent fallback, no auto-promotion,
and timestamped local UI history. The initial snapshot used the then-current
pre-UI hash; the final frozen Q3c preregistration and subsequent safety failures
are documented in the current handover section above.
