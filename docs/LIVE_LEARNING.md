# Live learning (RSI3-LIVE)

*Command: `ironmule serve --learning active`. Code: `ironmule_product/live_learning.py`,
native core `native/serving_learner`. Backlog entry: `RSI3-LIVE`.*

The server can learn how to group concurrent chats from its own traffic: an AI cost
model predicts what each scheduling action costs, a sequential reinforcement learner
updates a policy from real state transitions, and the same learner "dreams" through
the model while nothing waits. A learned policy serves only after it passes the
existing independent qualification rule on real hardware. Without that, the reference
schedule serves. Learning is off unless you start it.

## Start it

```bash
cargo build --offline --release --manifest-path native/serving_learner/Cargo.toml
ironmule setup --mode desktop
ironmule models add mlx-community/gemma-3-1b-it-4bit
ironmule serve --model mlx-community/gemma-3-1b-it-4bit --engine tuned --learning active
```

`ironmule start --learning active --engine tuned` does the same and opens the chat page.
The chat page shows one status line: serving ready (and on which device), learning
active or paused, a candidate in qualification, the strategy that serves and, for the
reference, why. Its buttons pause, resume and roll back.

| Option | Meaning |
| :-- | :-- |
| `--learning off` | default; no learner is created, behaviour is unchanged |
| `--learning active` | record, learn, explore within budget, dream, qualify |
| `--learning paused` | load saved learning state; record and change nothing |
| `--no-dreaming` | the same learner without model rollouts (a comparison arm) |
| `--no-replay` | no idle replay qualification; an external harness qualifies (RSI3-LIVE-P3) |

Learning needs an execution path that can group requests. The reference engine
serves one request at a time (capacity 1), so `--learning` there keeps the reference
and says so; `--engine tuned` groups up to eight streams (SERVE4); a stored policy with a smaller
`max_width` caps the catalogue. Joining a running
group (`join`) must stay off. While learning is on, `--engine tuned` does not run the
idle tuner, because that unloads the serving model. A learner that cannot start (library
missing or broken, unreadable state) keeps the reference and states why; serving goes on.

## API

| Method | Path | Behaviour |
| :-- | :-- | :-- |
| `GET` | `/v1/ironmule/learning` | full status (below) |
| `POST` | `/v1/ironmule/learning` | `{"action": "pause" \| "resume" \| "rollback" \| "compare" \| "freeze" \| "evaluate" \| "activate", "arm": ..., "evidence": ...}`; answers with the new status |

Both use the server's bearer token like every other route. `/health` carries a short
`learning` summary and the worker's actual `device` (`gpu` or `cpu`).

Status fields: `mode`, `eligible`/`reason`, `serving` (ready, device, backend, queue),
`strategy` (`reference` or `learned`, version, reason), `candidate` (version, state,
verdict, triplets done/needed, what it waits for, the training and simulator versions
it is bound to), `learning` (real transitions and updates, policy changes,
explorations, dropped and overridden decisions, rejections seen, CPU time, the twin
learner without dreaming), `model` (observations, mean relative prediction error,
share inside two predictive standard deviations, drift resets), `dreaming`
(synthetic updates, rollouts, rollouts stopped outside support or for uncertainty),
`budgets`, `persistence` and `compare`.

## What it decides

One decision per group the scheduler starts: a width limit and a gather wait from a
fixed catalogue. Action 0 is the service's own policy (the reference). The others are
widths 1, 2, 4 and 8 within the backend's capacity, each without a wait or with a 50 ms
or 150 ms wait for company (width 1 never waits). Grouping is the existing grouped
batch-1 path: every request keeps its own greedy answer, its stop handling and its
isolation. Model, numeric plan, caches and output do not change. Requests stay first
in, first out.

The state before a decision is metadata only: waiting requests, an arrival-rate
estimate, the first waiting prompt's length in characters and its output limit.

## What it records

Each finished group closes one real transition: the state before the decision, the
action proposed and the action actually executed, the state after, the interval, the
group's service time, summed time to first content, request-seconds spent in the
system, and per request its queue, first-content, inter-token and total times, tokens
and outcome. If the backend's grouping gate runs a group alone, the transition is
credited to width 1 without a wait, never to the proposed action (`overridden`); one
that maps to no catalogue action is dropped (`dropped`; its cost is then not learned).
Experience is a bounded ring of 4096 transitions in the native core, from which
transitions older than 14 days are pruned at start-up, plus a metadata log capped at
2 MiB (the older half is dropped when full). No prompt or answer text is stored.
Resource cost per transition is the learner's own CPU time (`learning_cpu_ms`), dream
CPU time and replay GPU time; device memory per decision is not recorded.

## Reward

Cost per decision interval, lower is better:
request-seconds in the system + 1.0 x summed time to first content + a penalty per
failed, cancelled, timed-out or rejected request. The penalty is
(1 + 1.0) x the request deadline + 1 s, so refusing or failing a request always costs
more than serving it at its deadline. Rejections (`429`) during an interval are charged
to that interval. The reward is the negative cost, scaled by 10 s, discounted by
`exp(-interval / 30 s)`.

## The three learning parts

- **Cost model (AI).** Per action, a Bayesian linear regression predicts the group's
  service time and the time from dispatch to a member's first content, each with a
  predictive variance. Summed first content is not predicted as one opaque number: it is
  what is known at the decision (the ages of the waiting requests the action would take,
  the gather wait if the group is not full, half of it for arrivals expected during that
  wait) plus the expected members times the predicted dispatch-to-first-content time.
  The features use only decision-time facts: the expected member count and the summed
  prompt size of the requests that are already waiting (a per-member prefill term), never
  the group composition known afterwards. Every forecast is made before its transition is
  learned and scored: absolute error, relative error (denominator max(actual, 1 ms)), the
  same with a 50 ms floor, the share of actuals below 50 ms and two-sigma coverage, overall,
  by actual group size (1, 2, 3-4, 5-8, 9+) and per segment of 100 transitions. A forecast
  for a state and action with fewer than three real decisions, or fewer than three that
  produced first content, is "unsupported": not used and not scored. The criterion's threshold from the backlog is unchanged (relative error below
  1.0); its basis changed with the model: P1 reported a running average over every
  observation, P2 judges the latest complete segment of supported forecasts for groups
  without a failed, cancelled or rejected request (such a group has no complete first
  content to compare), so the overload regime is not in it (`model.criterion`). A group with
  no successful member does not train the cost model. When
  fewer than 70 % of recent observations fall inside two predictive standard deviations, the
  model is retired and its support cleared (drift reset).
- **Sequential RL.** Tabular Q-learning over 30 state bins (queue 1/2/3-4/5-8/9+, arrival
  rate, prompt size) with time-based discounting. The target bootstraps from the next
  decision state's value, so an action that leaves a long queue pays for it. While
  learning, the reference (or a qualified policy) serves; 25 % of decisions explore the
  least-tried action in that state, except for three decisions after a failed, rejected
  or slow (first content over 10 s) request and when six or more requests wait.
- **Dreaming.** While nothing waits, the same learner runs rollouts of up to six steps
  through the cost model from real start states (Dyna). A rollout stops where a state
  and action have fewer than three real observations, where the model's relative
  uncertainty exceeds 0.5, when the simulated queue empties, or when the slice (10 ms)
  ends. Synthetic updates are counted separately, use half the step size, never count
  as support and are never stored as experience. Budget: 60 CPU-seconds and 20,000
  rollout steps per hour (a rollout that stops before its first step costs no step).
  Planning follows new information: a real transition the model did not support, or whose
  service time it missed by more than one predictive standard deviation, adds 50 steps to an
  allowance; one it forecast within that adds 10; one without first content adds nothing (cap 2,000). Dreams
  draw on the allowance, and the hour is paced: steps so far may not exceed the hourly cap
  times the elapsed share of the hour plus 2,000 (the allowance is spent per slice, so a
  slice may overshoot it by up to 32 rollouts). Counters and allowance survive pause and
  restart. `dreaming.state`: dreaming, not run since start, no experience yet, model too
  uncertain (also shown when a slice made no step for another reason), no state with enough
  real support, waiting for new experience, paced over the hour, paused for demand, hourly
  budget exhausted, dreaming disabled, learning paused. A second learner (the twin) receives
  the same real transitions and never dreams; it is the "RL without dreaming" arm.

## Candidates and qualification

After every 40 real updates the main learner's greedy policy is frozen: per state the
best action with at least three real observations, else the reference. An action that
behaves like the reference over a state's whole queue range counts as the reference.
A candidate needs to act differently on at least 20 % of recent real decisions. It is
versioned and bound to its checkpoint digest, real updates, experience size, model
observations, synthetic updates and whether it dreamed.

Qualification runs while the server is idle, starting a window only after 5 s without
an arrival or a finished group. One window replays a held-out trace:
16 consecutive arrivals recorded **after** the freeze, with their arrival offsets
(gaps capped at 2 s), prompt lengths and output limits (capped at 128 tokens). Prompts
are a fixed filler text of the recorded length. Each triplet replays one trace three
times through the real backend and the scheduler's group execution (`_perform_batch`;
replay forms groups in its own loop with the same width and wait rule), A (reference), B (candidate) and AA
(reference), in the order A, B, AA or B, A, AA. Window cost is the mean per request of
total time + first-content time + the failure penalty. The optimizer's protocol 2
(`native/optimizer/src/qualify.rs`, unchanged) judges eight triplets with minimum gain
3 %, maximum regression 10 % and family alpha 0.05 over four trials. A triplet is
correct only if every replayed request produced identical text in A, B and AA and none
failed in B. Replay starts no window once 600 GPU-seconds were spent in the hour (a
started window may run past it).

**Real demand comes first.** A real request admitted during a window cancels it at once:
replayed requests not yet prefilled are skipped, those decoding stop at their next token
boundary (the engine stops a cancelled stream instead of decoding it to its limit; with
chained readback the boundary is the end of the current chain), and
the triplet is discarded, not resumed. The server records for that request the phase of
the window when it arrived (prestart: no replay group running; prefill; decode; drain:
cancellation already under way) and its timeline: detection, drain until the worker is
free, handoff until the scheduler takes it up, and, kept apart from the extra wait (take-up
minus arrival), the schedule's own gather wait before dispatch and its own first content
after dispatch. A request that displaced replay is served by the reference or the
qualified policy, never explored with. `replay` in the status shows samples, p95
and per-phase figures. Above 100 ms extra wait at the 95th percentile of the last 64 such
requests, replay qualification and `compare` are suspended for this identity, with the
reason in the status and chat line. One sample is enough, and the suspension stays for
this identity: only a new identity (changed code, contract, model, plan or library) or
removing its state lifts it. Only the first real group after a cancelled window is sampled;
arrivals during dream slices or before a window starts are not replay waits. No new replay
window starts while an earlier one has not provably ended. A prefill already in progress finishes (its length bounds the wait),
and like any cancellation, a cancelled stream whose drain fails retires the worker.

Observations, model predictions and dreams never qualify anything. A qualified
candidate serves until two consecutive monitoring windows (at least four decisions and
24 requests each) contain failed requests, or until an operator rollback. A qualified
policy that only becomes slower on live traffic is not withdrawn automatically. Both withdraw
it to the reference; the next candidate must qualify again. Four trials per identity;
after that the reference stays.

## Comparison arms and external qualification (RSI3-LIVE-P3)

Three native learners receive every real transition: RL with dreaming (the main learner),
the same RL without dreaming (the twin) and a contextual bandit, which is the same learner
with a vanishing time constant (an action is valued by its immediate cost alone). They share
one decision interface; no other controller exists. For an external, preregistered comparison:

- `freeze` fixes the three greedy tables with versions, their real and synthetic update
  counts, rollout depth histogram and rollout stop reasons;
- `evaluate` with `arm` = `reference`, `queue_rule`, `fixed:<action>`, `bandit`, `rl` or
  `rl_dream` makes that arm decide every group; nothing is learned or explored and no idle
  work runs beside it; `arm` null ends it;
- `activate` with `arm` and `evidence` installs a frozen learned arm through the active-policy
  path, only when the evidence names a qualified verdict, this server's binding digest and
  that arm's frozen version. Monitoring, failure withdrawal and rollback then work as for a
  replay-qualified policy.

`tools/rsi3_live_p3.py` in [IronMule-Research](https://github.com/Tobayko/IronMule-Research) runs the whole comparison (its `research/RSI3-LIVE-P3_PREREGISTRATION.md`).
Its result on one T4 with Gemma 3 1B (ledger RSI3-LIVE-P3): no learned arm beat the best simple
strategy, so nothing was activated and the reference serves.

## Comparison for a performance release

`POST {"action": "compare"}` freezes the twin's policy (RL) and the main learner's
(RL with dreaming) and then, while idle:

1. selects the best fixed catalogue action with one replay window each on the last
   trace recorded before the request;
2. waits for four traces recorded after the request (held-out);
3. replays each held-out trace under the reference, the best fixed action, a queue
   rule (width 1 alone, else min(4, capacity), no wait), RL and RL with dreaming, with
   the arm order rotated per trace.

Every learned arm learned from the same real transitions. The report gives per arm the
mean cost, the ratio to the reference, p95 first-content time, tokens per second,
failures, whether outputs matched the reference, how often RL and RL with dreaming beat
the better simple arm by 3 %, and dreaming against RL. `release` is true only if RL with
dreaming beats the better simple arm on every trace with identical outputs and no
failure. The report is descriptive over four traces and adopts nothing: only the
qualification above changes what serves, and a performance claim still needs the
backlog's preregistered gates. Budgets are equal in real experience and holdout
windows; dreaming's extra CPU time is reported, not matched.

## Persistence and identity

State lives in `<state-dir>/live-learning/state-<identity>.json` (owner-only), written
atomically and synced when changed state is 30 s old (after a group or while idle) and
on pause, resume and shutdown. Only a consistent snapshot is taken on the scheduler
thread; encoding, digest, write and fsync run in a writer thread. It binds the hardware (chip and GPU names), the worker's
device, the MLX and mlx-lm versions (and the IronMule version where it is installed as a
package; empty in a source checkout), the digest of every source file the server process
had loaded when its learner started and of every source file the model worker had loaded
when it became ready (the worker reports it in its ready event; runtime, plans, executor and
service included; any edit, a comment included, starts a fresh learner that inherits no
experience or qualification), model id
and revision, the execution
plan (variant, engine configuration and metadata, numeric plan, chat reuse, thinking),
capacity, queue limit, the action catalogue, the reward contract and the native
library's SHA-256. A restart with the same identity resumes learning, model, twin,
candidate and active policy; an interrupted qualification triplet is discarded. A
different identity starts fresh and never reads another identity's file. A file that
fails its digest or binding check is renamed `*.rejected-<time>` and the reference
serves.

## What it does not do

- It does not change model, precision, numeric plan, cache policy or output.
- It does not run a second model or an extra language model.
- It does not qualify anything from simulated or predicted outcomes.
- It does not claim a speedup. Whether a learned policy beats the best simple one on a
  given machine, model and workload is a measurement; see the ledger entry RSI3-LIVE.

## Acceptance smoke

`tools/rsi3_live_smoke.py` ([IronMule-Research](https://github.com/Tobayko/IronMule-Research)) starts the server itself and checks every phase: start and
streaming chats (grouped answers identical to the same prompts alone), model, RL and dream
updates, real requests sent while a replay window is in each phase, a replay window that
then runs to its end, one member of a running group cancelled while the others' answers
stay exact, pause and resume, a
restart (shut down paused, started paused, identity, experience, candidate, budgets and
preemption records compared, then resumed), a cancelled stream, overload with `429` and
service afterwards. Its JSON report carries the Git commit, the uncommitted diff digest,
the digests of the product sources and the server's model, plan and learner identity.

```bash
ironmule setup --mode desktop --state-dir /tmp/rsi3
ironmule models add mlx-community/gemma-3-1b-it-4bit --state-dir /tmp/rsi3
python tools/rsi3_live_smoke.py --model mlx-community/gemma-3-1b-it-4bit --state-dir /tmp/rsi3 \
    --out rsi3-smoke.json
```
