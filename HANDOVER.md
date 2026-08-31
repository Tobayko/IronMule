# Handover — branch `r11-swap-gate`

## Current 2026-08-31 — Q3/Q3a

- **Commits:** Der Q3a-Code stammt aus Commit `0ec9237` (`fix: read macOS 26 q3a gates`); Versuch 3 wurde auf Docs-`HEAD` `28b2ef4` ausgeführt. Spätere reine Doku-Commits ändern den geprüften Code nicht. Die unmittelbar relevanten Vorläufer sind `771d133` und `7a8896f`.
- **Dritter Q3a-Preflight (2026-08-31):** `research/raw/Q3a_attempt3_20260831.json` ist ignoriert und erhalten (SHA-256 `019b1aafbc7342a782476d62e2921e8141b185c50490df4b72d5c0f0fbb9e8a2`). Der Harness endete vor dem Modellstart mit `FAILED`, `BASE`, `promotion_allowed=false` und `partial_children=0`; es wurden keine Modellkinder gestartet und keine Prozesse beendet. Grün: AC, Git-Bindung, installierter Speicher bekannt, Low-Power-off, exakte 4B-Identität `mlx-community/gemma-3-4b-it-4bit` / Revision `93724907d4ed1745d2fe50baadf3b0b01a65abf2` / Manifest `a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae`, Preregistration, Zeitbudget und Thermal. Rot: Swap `1774777794` Bytes (`1692.56 MiB`) über dem 256-MiB-Limit, Loadavg max `6.1792` über dem Limit 4 sowie das Prozess-Gate (`no_competing_model_process=false`). Die redigierte OS-Evidence zeigte zwei echte `Claude`-Executables: PID `55295` mit CPU `4.1%`/RSS `292688 KiB` und PID `55345` mit CPU `2.8%`/RSS `395344 KiB`; Argumente wurden nicht gespeichert. Damit widersprach der aktuelle OS-Zustand der Nutzerangabe, Claude sei beendet; es wurde nichts beendet. Auch nach dem Stoppen von Claude blieben Swap und Load eigenständige Blocker. Sicherste Fortsetzung ist nach gesicherter Nutzerarbeit und Neustart bzw. sauberem Zustand; kein automatischer Neustart.
- **Korrigierter zweiter Preflight:** `research/raw/Q3a_preflight_refusal2_20260831.json` ist ignoriert und erhalten. Der Lauf endete mit `FAILED`, `BASE`, `promotion_allowed=false`, ohne Modellstart und mit `partial_children=0`. Grün: AC, Git, installierter Speicher, Low-Power-off, exakte 4B-Revision `93724907d4ed1745d2fe50baadf3b0b01a65abf2` plus Manifest `a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae`, Preregistration, Zeitbudget und Thermal. Rot: Load `max=21.723`, `spread=0.699`; Swap `1641021440` Bytes und damit über 256 MiB; aktive Claude-Prozessaktivität. `research/raw/Q3a_preflight_refusal_20260831.json` ist ebenfalls erhalten; kein 27B-Modell wurde verwendet.
- **Replay und Fortsetzung:** Offline-Replay ist nur für `BASE` freigegeben. Die Datenbasis reicht nicht für eine adaptive/RL-Aussage; RL ist nicht anwendbar. Q3a wurde wegen der Gates nicht ausgeführt. Fortsetzung ist nur bei AC, Low-Power-off, nominalem Thermal, Load `<=4` und Spread `<=1`, Swap `<=256 MiB`, ohne Modellprozess oder aktive Claude-Aktivität und mit eindeutigem neuen Outputpfad zulässig.
- **Verifikation:** Targeted `127`, breit `336 passed, 1 skipped`, separate Integration `12`, modellfreie Q3a-Suite `26`; `ab`-/`tune`-Selfchecks erfolgreich. Der gepufferte Worker-Output-Cap bleibt der bekannte P2-Backlogpunkt. SQuAD bleibt untracked/lokal, die Lizenzfrage offen, und PR #2 ist owner-only.

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
