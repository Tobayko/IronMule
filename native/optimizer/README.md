# OPT1 / OPT3 optimization coordinator (`imopt`)

An opt-in coordinator with a Rust core that optimizes one explicitly selected application at a time:
detect capabilities and identity, measure the reference, test bounded candidates in isolated child
processes, keep a hash-chained journal, learn, qualify through an evaluator the policy cannot change,
then activate a reversible setting or keep the reference, and roll back on request. Backlog entry
`OPT1` holds the mechanism, the preregistered runs and the kill criteria; the ledger holds the results.
It never changes kernel, power or other privileged system settings: those are `unsupported`
capabilities that need a separately authorised adapter.

Std only (plus the path dependency on `../experiment_planner`), so it builds offline. The `test-faults` feature
enables a fault that forces a campaign verdict (for the activation test). Build it into its own target
directory so `target/release` always holds product binaries; a test-faults build also reports
`test_faults_build: true` in its host identity, so its states and qualifications never count for a product build:

```sh
cargo build --offline --release --manifest-path native/optimizer/Cargo.toml
cargo test  --offline --release --manifest-path native/optimizer/Cargo.toml
CARGO_TARGET_DIR=native/optimizer/target/test-faults \
cargo test  --offline --release --manifest-path native/optimizer/Cargo.toml --features test-faults
```

The coordinator contract (qualification, activation, status, resume, output correctness, rollback) is tested
deterministically against an adapter process with controlled measurements, judged by the unchanged evaluator.
Tests whose verdict depends on this machine's timing are hardware tests, marked `#[ignore]` and run on request on a
quiet machine (TEST1):

```sh
cargo test  --offline --release --manifest-path native/optimizer/Cargo.toml --test coordinator -- --ignored
```

## Entry points

```sh
imopt detect                                         # identity and capability states
imopt cycle    --state DIR --goal goals/cpujob.json   -- target/release/imopt-cpujob
imopt cycle    --state DIR --goal goals/ironmule.json -- python -m ironmule.opt_adapter --model M
imopt status   --state DIR --goal GOAL -- ADAPTER...
imopt rollback --state DIR --goal GOAL -- ADAPTER...
imopt compare  --state DIR --worker target/release/imopt-cpujob [--budget 600] [--holdout 20]
imopt managed  --state DIR --worker target/release/imopt-cpujob --seed N
imopt managed-rollback --state DIR
imopt dream-replay --state DIR
imopt rulecheck --state DIR --worker target/release/imopt-cpujob --goal goals/cpujob.json [--cycles 30] [--control]
imopt calibrate --state DIR --worker target/release/imopt-cpujob [--reps 3] [--slices 10] [--threads 1,2,3,4,6,8]
imopt campaign --state DIR --worker target/release/imopt-cpujob --baseline-threads T [--rounds 2] [--budget 300]
               [--holdout 10] [--seed 1] [--no-dream] [--threads ...] [--reward sparse|dense] [--start cold|warm]
               [--holdout-seed0 N] [--targets-from CAMPAIGN_DIR] [--status]
imopt duel --state DIR --worker target/release/imopt-cpujob --base CAMPAIGN_DIR --candidate CAMPAIGN_DIR --seeds A-B
```

Three modes for the sequential decision, kept apart: `compare` is the fresh research comparison
(new learners and holdouts every run); `campaign` is a resumable learning campaign (OPT3) that
learns across rounds and activates only what its own fixed holdout rule accepted; `managed` is the
product mode and runs only an activated policy, else the reference (one thread).

OPT2's options (backlog OPT2): `--reward dense` trains the learner on the foreground p95's excess over its target
while every return stays the sparse one; `--start warm` spends a third of round 1's budget on a sweep of fixed
actions and starts every unvisited state at the sweep's mean return per action; `--holdout-seed0` fixes one
round's holdout seeds; `--targets-from` reuses another campaign's foreground targets. `duel` judges two finished
campaigns' final learners on the same schedules (the dream comparison) and refuses learners trained against
different targets.

`calibrate` (OPT3 step B) releases OPT2 only if `net_headroom >= 0.05` and the lower 95 % bound of
the headroom is above 0, and only from a valid run: every (level, threads) cell and both switch-block
thread counts need at least half of their slices complete, else the verdict is `invalid` (OPT2-CAL).

## Contracts

* **Identity.** Host facts (separate probes for Linux, macOS, Windows), the coordinator binary's
  SHA-256, the adapter's own identity (application version, model identity, workload) and the goal's
  digest form the cycle identity. Records of another identity may order the search but never count as
  reference, test result or qualification; an active setting of another identity is rolled back first.
* **Capability states.** `unknown`, `unsupported`, `observable`, `controllable`, `tested`, `qualified`
  are distinct and explicit; a failed probe is `unknown`, never a guess. Energy and temperature are
  `unknown` (no probe).
* **Journal.** Append-only JSON lines with a SHA-256 chain and an atomically written head (last sequence
  number and hash), so records cut from the end are caught. The planner state is rebuilt from it each
  time; `planner.json` is a verified cache (damaged: quarantined and rebuilt). A torn final line from a
  crash is kept aside and cut only if the head does not count it; any other damage stops the
  coordinator before it acts. Deleting the journal together with its head is not detected.
* **Resume.** A crash after `activate` and before `outcome` is finished on restart from the journal
  (recheck, outcome), never by applying again; an activated outcome stands only while the adapter still
  reports it, otherwise it is invalidated and the cycle runs again.
* **Evaluator.** Either the online controller's strict A/A/B rule (protocol 1, ported in `qualify.rs`)
  with thresholds from the frozen goal, or the adapter's own evaluator (IronMule: `tune.confirm`).
  Protocol 2 (`"protocol": 2` in a goal; any other value than 1 or 2 refuses the goal) replaces protocol
  1's single-triplet noise veto by a 0.5 ceiling, a stricter mean condition and a material drift test;
  `imopt rulecheck` judges both on the same triplets (backlog/ledger OPT4b). Only `goals/cpujob-p2.json` selects it.
  Actions of class `tested_only` (numerically different execution, e.g. IronMule's `engine=rust2`) are
  measured but can never qualify automatically.
* **Own processes (OPT3-D (c); macOS and Linux).** Every adapter call runs under `imopt supervise` (`src/proc.rs`):
  the adapter leads a new process group, the supervisor watches a lifeline pipe whose only write end the
  coordinator holds, and ends the group when the adapter exits or the coordinator dies (SIGKILL included):
  SIGTERM, 2 s grace, SIGKILL, signals only while the group's leader is unreaped, so a reused pid can never
  be hit. Every live supervisor holds a kernel `flock` lease on `<state>/measure.lease`; a (re)starting
  coordinator takes it exclusively before its first adapter call and refuses after 15 s, so no measurement
  starts while an earlier run's processes live or are not proven gone. No pid is stored and nothing is
  ended by name. Deadline: 10 s after the coordinator's death. Adapter contract: descendants stay in the
  adapter's process group (IronMule's child guard refuses `setsid`/`setpgid`); one that leaves it is not
  covered. Elsewhere (Windows) adapters run unsupervised.
* **Activation and rollback.** The adapter's `apply` returns the previous state; it is journaled and
  enveloped. After activation the adapter must report the action and a recheck must reproduce the
  reference output, else it is rolled back at once. `rollback` restores the previous state (from the
  envelope, or from the journal if the envelope is lost) and checks the adapter reports it.

## Adapter protocol

See `src/adapter.rs`: `describe`, `run --action ID --seed N`, `qualify --action ID` (adapter evaluator),
`apply --action ID`, `revert --previous JSON`, `active`; one JSON object as the last stdout line; every
call a fresh process with a timeout. Adapter 3 (OPT5, study): `experiments/opt5/cargo_adapter.py`, this
repository's Rust test build with cargo's `build.jobs`. Adapter 1: `ironmule/opt_adapter.py`. Adapter 2: `imopt-cpujob`
(an own LZ77 compression job; setting: thread count in its config file; output digest independent of
parallelism).

## Learning

* Bandit / search order: the RSI1 planner (`experiment_planner`): conjugate normal posterior per action,
  next test by expected improvement per second; each action screened once, repetition belongs to the
  qualification.
* Sequential RL: tabular Q-learning (`rl.rs`) for the background-thread decision beside a foreground
  probe (`episode.rs`); state, action, transition, episode end and return were fixed in the backlog
  before data. The Q-table is persisted in an identity-bound SHA-256 envelope.
* `compare` always starts fresh learners and holdout runs; its Q-tables are written once (and read
  back by `managed` only).
* `campaign` (OPT3) is resumable: rounds of real training -> holdout against a fixed baseline
  (`F L L' F'` / `L F F' L'` per schedule, so the A/A pairs span drift and order; drift by sign test
  invalidates a round) -> fixed decision -> replay selection from training trees only. Plan, targets
  and reference digest are fixed in its first record; another plan or identity in the same state is
  refused (experience and qualification never transfer). The learner is rebuilt from every completed
  training episode's recorded updates and checked against the digest the online learner journaled.
  Each attempt has its own schedule seed and RNG stream (journaled). An interrupted attempt is
  `attempt_aborted`: never a measurement; a training one keeps a budget charge of the episode cap; a
  holdout one burns its schedule (no arm of it runs again). A wrong job output is `attempt_failed` and stops
  the campaign for good; `compare` refuses a campaign's state. Activation is journaled intent -> envelope
  -> done and happens once per campaign.
* Seeds: every generator is seeded through splitmix64 (`rl::stream`); OPT1's `seed | 1` gave seeds 2k
  and 2k + 1 one stream.
* Dream-RSI meta loop (`dream.rs`): paper-based extension (the official code was unreleased on
  2026-10-13), deviations listed in the module: episode trees instead of program-search trees, learner
  settings as the meta-policy space, replay only over recorded (state, action) outcomes with masks and
  dead ends counted, development / validation / holdout separated by seed. Replay never qualifies.
  Since OPT3 a replay walk that reaches a dead end, the step cap, or a state where the evaluated
  policy's own choice was never recorded has no return (unknown), never a short one; a candidate with
  any incomplete validation walk is unsupported and cannot be chosen; replay bootstraps only over the
  next state's recorded actions. Empirical transition replay stitches recorded slices together: it
  is not a simulator of arbitrary hardware trajectories.
* Foreground probe v2 (OPT3): every request records scheduled arrival -> completion (`e2e`, the
  violation measure, late wake-ups included) and start -> completion (`service`) separately.

## Platform matrix

| Platform | Built | Unit and integration tests | Real runs |
| --- | --- | --- | --- |
| macOS 26 arm64 (Apple M1 Max) | yes | a57f986: 30 unit, 13 / 14 integration (default / `test-faults`), 8 process tests, all pass | OPT3: calibration (headroom 0), CPU cycle killed and resumed (reference kept by protocol 1), campaign killed twice and resumed; OPT3-D (a): supervised processes gone 0.07 s after a SIGKILL, reference kept by protocol 2 (one stalled run); IronMule adapter on gemma-3-1b-it-4bit: resumed after a kill, `compiled_fixed_cache` qualified (0.906), activated, rolled back |
| Linux x86_64 (Kaggle CPU) | yes | a57f986: 30 unit, 13 / 14 integration (default / `test-faults`), 8 process tests, all pass | OPT1 run 1, OPT4b control (see ledger) |
| Linux x86_64 + NVIDIA T4 (Kaggle) | yes | built in OPT1 run 2 | OPT1 run 2, IronMule adapter (see ledger) |
| Windows x86_64 | `cargo check` cross-target only | not run | none: memory size and GPU stay `unknown` in its probe |
