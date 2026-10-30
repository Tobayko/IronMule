# Speed backlog

Every hypothesis that could plausibly make IronMule faster, including the ones that
are probably wrong. Nothing here is a result. Nothing here has been measured unless
it says so and names an experiment.

This file exists because the alternative is holding twenty half-ideas in one head and
re-deriving them badly six weeks later. An idea written down with its kill criterion
costs nothing to keep and can be refuted by anyone.

## The main goal (maintainer decision, 2026-10-30)

The owner made one entry the project's main goal: a working IronMule agent. It decides on its own how the
model's requests run on the hardware, reaches a measured gain there, keeps learning by reinforcement learning
while it serves, and decides as fast as a small decision model. Every rule below still applies to it: it
changes no answer, a gain counts only when it is measured, and it serves nothing it has not proven on the
machine it runs on.

### AG1 — The IronMule agent: one online learner that decides, learns and proves its gain where it runs (opened 2026-10-30, before data)

**Mechanism.** `ironmule_product/agent.py`, opt-in (`ironmule serve --agent on`), one decision owner (refused
beside `--learning` and `--dispatch-policy`), one worker. It decides, at each group start, shortest output limit
first or arrival order and the group's width (1, 2, 4 or 8 within the backend's capacity), in the LD1 policy
format: 7.5 µs per group start in pure Python on this cloud container's CPU. Four steps repeat while the server runs:

1. *Observe:* every finished group, real or replayed, updates a world model by recursive least squares
   (forgetting 0.995): prefill seconds per member and per thousand prompt tokens, decode seconds per round and per
   further session in a round, the pool model RL4 and RL8 fitted by hand. Every finished request goes into a
   traffic memory of metadata (2,048 arrivals; no text).
2. *Plan:* while idle, a planner process (nice 19, its own process for the GIL; on a CPU it is stopped while
   requests run) trains REINFORCE on order and width in a one-card model priced by step 1, on windows of 16
   recorded arrivals with their gaps scaled by 0.5-1.5, harm weight 5 against the best fixed schedule in the
   model (RL8). It proposes the learned policy if the model prices it below the best fixed schedule, else that
   schedule, and nothing the model does not price below the incumbent. It plans only with at least 12 groups,
   32 arrivals and a median relative error of the model's group times below 25 % (RL4's fidelity bound).
3. *Qualify:* while idle (5 s after the last real group), arrivals recorded after the plan are replayed through
   the real backend with filler prompts and their recorded gaps (idle stretches skipped; output limits capped
   at 512): incumbent, candidate, incumbent (A, B, AA, or B, A, AA). A real request cancels the window; the arms
   already run are kept and the cut arm runs in the next idle spell; a window cut three times is skipped. A width the model has not priced (fewer than three groups that wide) is measured first, in
   at most two replay windows with every arrival queued at once, never on a user's request.
4. *Serve and watch:* a qualified policy decides every group. Three decision errors, two consecutive live
   windows of 24 requests with a failed request, or the operator (`POST /v1/ironmule/dispatch` `off`) withdraw
   it; its digest is recorded and never serves again under this binding. State (`<state-dir>/agent/`) is bound
   to the hardware, device, model and revision, plan and software versions.

**Qualification rule (fixed before data).** Per completed triplet the window costs (mean per request of time in
system + time to first token, failure penalty 2 x deadline + 1 s). Gain: 1 - sum(B) / sum((A + AA) / 2) over the
triplets, with a 2,000-resample bootstrap (seed 0). A/A floor: the size of the same estimator for AA against A.
**Pass:** after at least 6 triplets, the gain's 2.5 % bound is above zero and above the floor (KP1's "every resolved
gain" rule, no minimum effect size). **Fail:** the 97.5 % bound is at or below the floor, or 24 triplets without a
pass. **Veto:** any replayed answer that differs between A, B and AA, or a failed request under B.

**What earlier entries predict.** On the measured T4 setup the tuned engine's scheduling headroom over the best
fixed schedule is below the hardware's noise since H9 (RL8; 0.1-1 % for a learner), and the agent's planner on
RL8's H9 fit prices `w4/sjf` 5.2 % below `w8/fifo` (preview, below). The agent is therefore expected to find a fixed
schedule there, not to beat one. Its claim is autonomy: on a setup nobody studied it finds a schedule, proves it
and re-learns when the machine changes. **Evidence against:** idle time may be too short to qualify anything;
the pool model may misprice an engine (the dense plan's slot reuse is not modelled and is refused); the RSI3-LIVE
learner found no gain on a T4 (P3).

**Tests.**
- **S, the loop in simulation (`tools/ag1_online.py`, local, no GPU).** The agent unchanged against a simulated
  server priced by RL8's H9 fit (a window slowdown of 3 % and a round slowdown of 2 %, lognormal), 300 live traces
  cycling through RL2-H2's quiet, bursty and spike states (seed 90,000, 240 s pauses), 1,500 planner episodes per
  seed; and the same with the machine changed at trace 150 to a price where a further session costs a whole step
  (a CPU). Every window is also served by the reference schedule (`w8/fifo`) alone on the same arrivals.
  **Decision S (the loop works):** (a) every world-model parameter within 10 % of the machine's after 50 traces;
  (b) on the T4 price a candidate qualifies by the rule above within the 300 traces; (c) with the drift, a
  candidate planned on the new price qualifies within 150 traces after it. Gains per phase against the
  reference are recorded, not decided: S is a simulation.
- **C, mechanics on a real CPU (this container, `Qwen/Qwen3-0.6B` rev `c1899de`, `--agent on`).** Recorded: groups
  observed, plans, explorations, replayed triplets and that every replayed answer matched. No gain is decided: a
  CPU makes about one token a second here.
- **V, the gain on hardware (Kaggle T4, after the weekly GPU quota resets).** *(Fixed 2026-10-30, before its data;
  the pass rule amended from LD1-V's to KP1's, the rule qualification uses.)* `tools/rl2_headroom.py --plan ag1`,
  kernel `ironmule-rl2-headroom-ag1` pinned to a pushed commit, "GPU T4 x2" with one worker on the first card,
  `mlx-community/Qwen3-1.7B-bf16`, the product's own `ProductService` with `--engine tuned`, the agent attached as
  `--agent on` attaches it, reference schedule `w8/fifo` (the default where no schedule is qualified). About
  **2.5 session hours**.
  - *Online phase:* 36 traces of RL2-H2's states in turn (indices 107-142, seeds nothing earlier used), at the
    calibrated capacity, each followed by 120 s of idle time. The agent decides, learns and qualifies by itself;
    its status is recorded after each trace.
  - *Comparison:* the policy serving at the end is frozen and the agent detached. On 8 held-out traces per state
    (indices 143-150), each block runs reference, frozen policy, reference (LD1-V's order), the policy deciding
    through the product's LD1 hooks.
  - *Decisions:* noise is the nearest-rank 95th percentile of the A/A pairs.
    - **(1)** The gain over the reference (the mean of its two runs), paired by trace with the states weighted
      equally and a 2,000-resample bootstrap, has a 2.5 % bound above zero and above the A/A floor.
    - **(2) No harm:** each state's bound is above minus the noise.
    - **Validity:** every request completes, with the same tokens in all three cells of its block.
  - If nothing qualified in the online phase, V reports "nothing qualified online" and compares nothing. That
    is a result about idle time and noise on this setup, not a gain.
- **V-CPU, the same on this cloud container's CPU (fixed 2026-10-30, before its data).** While the GPU quota is spent,
  V's harness runs here with `--profile cpu`, differing from V only in this:
  - model `mlx-community/SmolLM2-135M-Instruct-8bit` (rev `0f0d9b8`), the reference engine with weights dequantized
    to float32 (`--compute-dtype dequantize`, about 6 tokens a second here). Its capacity is 1, so the agent
    decides the order only;
  - output limits 4 or 32 tokens, a request timeout of 1,800 s;
  - 15 online traces (indices 107-121) with 360 s idle after each, 4 held-out traces per state (143-146).
  - every prompt short *(amended 2026-10-30 before any of its data was read: the first start was stopped during its
    first online trace, because a 1,030-token prompt takes an estimated 25-75 s to prefill on this CPU (3.7 GFLOP/s
    measured on a 1,024-square product), which would stretch the run past 10 hours and make a replay window longer
    than an idle spell)*.
  Decisions, validity and analysis are V's. It takes about 3 hours of this CPU and no GPU. A pass is a gain of the
  agent measured on real hardware, bounded to this CPU, this model and this traffic; it says nothing about a T4.
  It runs on the commit whose S and S2 reruns pass (the decision table below changed the agent).
  - **Result (2026-10-30, run at `1db94a7`): nothing qualified online, so there is no hardware gain and no evaluation.**
    The agent planned one candidate, `w1/sjf` (`633ceda2cdce`). The 15 online traces produced 8 replay triplets. At
    the look after 6 triplets the candidate neither qualified nor was rejected, and the online phase ended before the
    look at 12. The agent served the reference throughout. Its world model fit 864 groups (median relative error
    7.4 %, never reopened). The analysis reads `valid: false, verdict: nothing qualified online` with no problems,
    which is the protocol's stop "nothing to compare". By the Kill rule, `--agent` stays opt-in and experimental.
    *Descriptive, not part of the verdict:* the 15-trace budget bought too few triplets for the group-sequential
    test to resolve a small order-only effect at capacity 1. A rerun with more online traces is a new protocol and
    has to be fixed here before its data. Ledger entry owed.

**Kill.** S failing (a)-(c) stops V until the loop is fixed. V failing ships no agent default: `--agent` stays opt-in
and experimental, as every learned path here. A pass still ships only opt-in, bound to the setup it was measured on.

**Preview (2026-10-30, before S's data; descriptive, not a decision).** The planner on RL8's H9 fit (one card,
40 recorded traces of RL2-H2's states at half RL2-H's two-card rate, 2 x 3,000 episodes, 14 CPU-seconds) chose
the fixed `w4/sjf` (21.80 on 80 held-out windows) over the learned policy (the same 21.80) and `w8/fifo`
(23.00): 5.2 % in the model. During development the simulated loop rejected every candidate under LD1-V's
noise rule (the gain's bound against the 95th percentile of single A/A pairs); the entry's rule above is KP1's,
fixed now, before S runs.

**Result S, run 1 (2026-10-30, `AG1-S-run1-20261030`, local; the agent before the fix below, otherwise as
committed with this entry): (a) and (b) pass, (c) fails, so S fails and the loop was fixed.**
- **(a)** After 50 traces the world model read prefill 0.140 s + 2.23 s per thousand tokens and a round
  18.9 ms + 15.6 ms per further session, against the machine's 0.146, 2.22, 18.7 and 15.7: within 4 % each.
- **(b)** The agent explored widths 2 and 4 in two replay windows and planned the fixed `w4/sjf`-class schedule
  `5aaabf717ed4`. It qualified after 6 triplets: +4.1 % over the reference schedule (95 % 1.9-6.5 %, A/A floor 1.7 %).
  From trace 19 on, a learned schedule served 281 of the 300 windows. On those windows it cost 5.2 % less than the
  reference alone on the same arrivals (28.88 against 30.46 per request). A second candidate `1c205576ee02`
  failed three times and then qualified over the first: +3.4 % (95 % 0.6-6.5 %, floor 0.5 %). Seven plans took
  6-8 CPU-seconds each.
- **(c) fails.** After the change to the CPU price the decode fit followed (59.6 ms + 59.6 ms against 60 and 60),
  but nothing was planned again: the model's median error stayed at 9.8 and the fidelity bound held. The prefill
  fit had already broken at trace 113-116, before the change, in the run without drift too. It went from 2.3 to
  1,824 s per thousand tokens. Replaying the run with that code located it: after about 2,600 groups, many of them
  eight long prompts, the covariance lost its positive diagonal. With cumulative prompt tokens and member index as
  features it was ill-conditioned. The fidelity bound kept the broken model from planning, so nothing worse than
  the incumbent served; after the change the incumbent cost 0.7 % more than the reference.
- **Fix (before run 2):**
  - Each member's own prefill is fitted on a constant and its prompt tokens.
  - The covariance is kept symmetric with a positive diagonal, else it is reopened keeping the fit, and its
    trace is held at its start.
  - When the median error stays above 25 % for a full window of 50 groups, both fits are reopened.
  - `test_thousands_of_groups_keep_the_fit_well_conditioned` fails with the old estimator and passes with the
    new one.

**Result S, run 2 (2026-10-30, `AG1-S-run2-20261030`, at `55ae021`): (a) and (b) pass, (c) fails again, now for
lack of idle time; the loop was fixed again.**
- **(a)** At trace 50: 0.147 s + 2.22 s per thousand tokens, 18.9 ms + 15.6 ms (within 1 % of the machine's). The
  prefill fit stayed put through all 300 traces this time.
- **(b)** The same two qualifications as run 1: +4.1 % (95 % 1.9-6.5 %, floor 1.7 %), then +4.4 % over that (95 %
  0.7-7.8 %, floor 0.5 %). Against the reference on the same arrivals, the learned schedules cost 5.6 % less over the
  131 windows before the change (28.99 against 30.71 per request).
- **(c)** After the change to the CPU price the model re-learned it within 20 traces (0.39 s + 5.99 s per thousand
  tokens, 59.9 ms + 59.7 ms against 0.40, 6.0, 60 and 60; one reopening). At trace 170 the planner proposed a
  learned candidate priced 2.6 % below the incumbent. It never completed a triplet: on the slower machine one
  replay window takes about two of the four idle minutes, so demand cut the third window every time, and a cut
  triplet was discarded. Over the 150 windows after the change, the incumbent cost 1.2 % more than the reference.
- **Fix (before run 3):** the arms already run in a triplet are kept (their cost, whether a request failed, and a
  digest per answer), and only an arm that demand cut runs again in the next idle spell. The arm order still
  alternates; the A/A floor measures any drift between spells.

**Result S, run 3 (2026-10-30, `AG1-S-run3-20261030`, at `cd1206d`): (a) and (b) pass, (c) fails, and a replay
defect was found; the loop was fixed again.**
- **(a)** and **(b)** as in run 2. Without drift, learned schedules served 281 of 300 windows at 5.3 % below the
  reference on the same arrivals.
- **(c)** After the change, triplets now completed (127 in all, 69 before the change). The agent compared four
  candidates on the CPU price and resolved none: a learned one measured +1.7 % (95 % -0.6 to 4.6 %, floor
  0.9 %), and `w8/sjf` +1.4 % (95 % -0.7 to 3.7 %).
- **Post hoc, a diagnosis after the data that changes no verdict:**
  - The run's live traffic was served without noise under every fixed schedule. `w8/sjf` was best on both
    prices: +7.1 % over the reference on the T4 price, against +5.9 % for `w4/sjf`, and +3.9 % on the CPU price,
    against -0.7 % for `w4/sjf`.
  - The agent's second qualification had replaced `w8/sjf` with `w4/sjf` (replay: +4.4 %), the opposite of
    what the live traffic gives. After the change the incumbent was therefore 4.5 % behind the best schedule.
  - The cause is the replay window, inherited from RSI3-LIVE:
    - gaps capped at 2 s turn quiet traffic into bursts;
    - output limits capped at 128 halve the 256-token requests.
    So qualification judged a different workload from the one served.
- **Fix (before run 4):**
  - A replay window keeps the recorded gaps. The service's replay skips stretches in which nothing waits and
    nothing runs, releasing the next arrival at once and moving the rest up. That changes no cost, because every
    time is measured from a request's own scheduled arrival.
  - Output limits are capped at 512.
  - The simulator's replay does the same.

**Result S, run 4 (2026-10-30, `AG1-S-run4-20261030`, at `7558cdb`): (a) and (b) pass, (c) fails, and a second defect
was found.**
- **(b)** With faithful replays the agent qualified `w8/sjf`, the schedule the noise-free check calls best. From
  trace 9 on a learned schedule served 291 of 300 windows, 5.8 % below the reference on the same arrivals.
- **The defect.** The planner priced `w4/sjf` below `w8/sjf` and proposed it ten times; nine comparisons rejected
  it (-9.5 % to +3.9 %), and the tenth passed (+7.3 %, 95 % 2.6-10.6 %, floor 0.75 %). Served without noise,
  `w4/sjf` is 1.3 % worse than `w8/sjf` on this traffic, so the tenth pass was the multiple-testing error the
  rule cannot see when the same loser is tried again and again.
- **(c)** With the drift to the CPU price, `w8/sjf` kept serving. Four candidates were rejected, and the learned
  schedules cost 2.6 % less than the reference after the change (runs 2 and 3: 1.2 % more).
- **Post hoc, changing no verdict:** the noise-free check says `w8/sjf` is also the best fixed schedule on the CPU
  price (+3.9 % over the reference). (c) as written could only pass if the agent replaced the best schedule. It
  was ill-posed for this drift. S stays failed as recorded.
- **Fix (before run 5):** a candidate rejected against the serving schedule is not proposed against it again
  until the world model reopens (the machine changed); the planner then proposes the next schedule the model
  prices below the incumbent, or nothing.

**S2, a drift that changes the best schedule (fixed 2026-10-30, after S run 4's data and before its own).** S's
(c) cannot pass when the best schedule survives the change. S2 is S with the drift at trace 150 going to a price
with the T4's prefill and rounds of 10 ms + 40 ms per further session. It was chosen with no agent run on it, by
serving 150 traces of the live traffic (seed 95,000) noise-free under every fixed schedule. There `w1/sjf` costs
20.00 against 38.24 for `w8/sjf` (47.7 % less); on the T4 price `w8/sjf` stays best.
- **Decision S2:**
  - (a') every world-model parameter within 10 % of the new price 50 traces after the change;
  - (c') a candidate planned on the new price qualifies within 150 traces after it.
- **Gate (amended with S2, before its data, stricter than before):** V runs once S run 5 passes (a) and (b) and S2
  run 1 passes (a') and (c'), on the same commit. S's (c) stays recorded as failed.

**Result S run 5 and S2 run 1 (2026-10-30, `AG1-S-run5-20261030`, `AG1-S2-run1-20261030`, both at `3d6e5c3`): all
four criteria pass. The gate does not open, though: a code review found ways to qualify without a gain, and one
of them happened in run 5.**
- **S (a)** At trace 50: 0.147 s + 2.24 s per thousand tokens and 18.9 ms + 15.7 ms (within 1.5 %).
- **S (b)** `w8/sjf` qualified at trace 9: +9.4 % (95 % 2.0-15.5 %, floor 1.3 %). Learned schedules served 291 of 300
  windows, 5.3 % below the reference.
- **S2 (a')** After the change at trace 150 the model read 10.0 ms + 40.1 ms per round (truth 10 and 40) within
  50 traces, after one reopening.
- **S2 (c')** At trace 168, 18 traces after the change, `w1/sjf` qualified over `w8/sjf`: +46.5 % (95 % 29.6-66.0 %,
  floor 1.2 %). After the change the agent's windows cost 46.6 % less than the reference's (22.34 against 41.83).
- **But:**
  - In S run 5 a learned policy (`4a68dcb5eeef`) qualified at trace 188 over `w8/sjf`: +4.7 % (95 % 1.9-7.3 %,
    floor 1.4 %). Served without noise on the same 300 traces, it is 1.7 % *worse* than `w8/sjf` (post hoc check).
  - The review (sparring agent) traced it, and further holes, to these causes:
    - the verdict was read after every triplet from 6 to 24 against an uncorrected 2.5 % bound;
    - a learned twin of a rejected schedule comes back under a new digest;
    - a failed request under the incumbent (A or AA) inflated the candidate's gain;
    - kept arms could outlive a change of incumbent;
    - alternating arm orders put B early under drift;
    - replay showed the policy at most eight waiting requests where the live schedule shows all;
    - nothing caught a serving schedule that becomes slower without failing;
    - `close` raced the scheduler's idle work and the state file had one shared temporary name;
    - the binding left out the reference schedule and capacity;
    - a width-less policy raised in `width`;
    - the prefill fit had no per-group term.
- **Fix (before S run 6 and S2 run 2, which replace run 5 and run 1 for the gate):**
  - The verdict is group-sequential, read only at 6, 12, 18 and 24 triplets with the lower bound at 0.625 %
    (2.5 % split over the four looks). The futility bound stays at 97.5 %.
  - Exclusion goes by behaviour: a digest of the decisions on every queue of 1-16 requests. What was withdrawn
    stays excluded.
  - A failure under A or AA drops the triplet. A triplet runs A, B, AA and is dropped if the incumbent or the
    pause changed under it.
  - Replay shows the whole queue.
  - While no candidate waits, an audit replays the serving schedule against the reference (up to 24 triplets).
    It withdraws the schedule when it loses with the corrected bound; comparative evidence may withdraw.
  - `close` stops idle work and plan starts, saves are serialised, and the harness detaches the agent before
    closing it.
  - The binding names the reference schedule and capacity, and a width-less policy keeps the service's width.
  - Prefill gains a per-group term (state schema v2).

**Result S run 6 and S2 run 2 (2026-10-30, at `f850704`): S passes (a) and (b); S2 fails (a') by 0.4 points. The gate
stays shut.**
- **No false qualification this time.** Learned policy `115547b60687` qualified at trace 134 over the reference:
  +6.2 % (95 % 1.8-10.0 %, floor 0.5 %). Served without noise it is +4.9 % over the reference, a real gain (post hoc
  check). Its 24-triplet audit against the reference ended `confirmed`.
- **The cost:** `w8/sjf` (+6.2 % without noise) was rejected against the reference with +6.3 % (95 % 2.9-9.2 %,
  floor 1.8 %). Its 95 % bound cleared the floor, but the evenly split 0.625 % bound did not at any look.
- **S2 (c')** `w1/sjf` qualified 10 traces after the change: +45.2 % (95 % 35.1-58.9 %). Its audit confirmed it,
  and the agent's windows after the change cost 49.4 % less than the reference's.
- **S2 (a') fails:** 50 traces after the change the prefill intercept read 0.161 s against 0.146 s (+10.4 %); the
  other parameters were within 1.3 %. The decode-only change had reopened both fits.
- **Fix (before S run 7 and S2 run 3):**
  - Prefill and decode errors are tracked apart, and only the part that stops fitting is reopened
    (`test_a_decode_only_change_reopens_only_the_decode_fit` fails under the old rule).
  - The per-group prefill term starts sure of zero, since groups of one cannot tell it apart.
  - Alpha is spent strict-early, lenient-late (0.1, 0.2, 0.4 and 1.8 % at the four looks; 2.5 % in all).
  - The planner scales recorded gaps log-symmetrically, 0.8-1.25 (0.5-1.5 weighted heavy load).

**Result S run 7 and S2 run 3 (2026-10-30, `AG1-S-run7-20261030`, `AG1-S2-run3-20261030`, both at `a353e41`): all
four criteria pass on one commit, so V's gate is open.**
- **S (a)** At trace 50 every parameter was within 1.5 % of the machine's.
- **S (b)** At trace 26 `w8/sjf` qualified over the reference: +6.3 % (95 % 2.9-9.2 %, floor 1.85 %). By the
  noise-free check it is the best fixed schedule, and its 24-triplet audit confirmed it (+6.4 %, lower 95 %
  bound 1.6 %). Nine later candidates were rejected; none of them qualified. A learned schedule served 274 of
  300 windows, 6.4 % below the reference on the same arrivals (28.54 against 30.48 per request).
- **S2 (a')** 50 traces after the change every parameter was within 2.4 % of the new price.
- **S2 (c')** At trace 186, 36 traces after the change, `w1/sjf` qualified over `w8/sjf`: +44.5 % (95 %
  35.4-50.5 %, floor 4.1 %). Its audit confirmed it (+55.6 % over the reference). After the change the agent's
  windows cost 42.3 % less than the reference's (24.12 against 41.83).
- **Recorded, not decided:**
  - A decision inside the agent (view, lock, policy) took 11.6 µs on this cloud CPU, in pure Python; order
    and width at a group start together 25 µs.
  - The planner preview (2 x 3,000 episodes) took 15 CPU-seconds.
  - Simulated wall time per 300-trace run: 3-7 minutes.

**V's harness as a CPU smoke (2026-10-30, mechanics only, no gain claim).** `tools/rl2_headroom.py --plan ag1 --smoke` ran
on this cloud container's CPU with `Qwen/Qwen3-0.6B` (rev `c1899de`, bf16, about one token a second), through the
product's own service and worker. The harness file was loaded at `7558cdb` and the agent module after `f850704`, so
mixed code ran; this checks the tool, nothing more.
- Calibration: 16 of 16 requests completed.
- Online phase (3 tiny traces): the agent observed 7 real groups of width 1 and 2. It planned nothing, being below
  its 12-group minimum.
- Comparison with the smoke's fixed stand-in policy (`w2/sjf`, used only when nothing qualified): 9 cells, every
  request completed with the same tokens in all three cells of its block, and the dispatch made 26 decisions with
  no error.
- The analysis reads "invalid", as the rule requires two complete traces and a smoke has one.
- With 7 groups of near-equal prompts the fitted prices were meaningless (14.9 s per group), and the agent was right
  not to plan on them.

## PROD1 — Approved product implementation (2026-09-05)

The user approved the product-wrapper plan, real-hardware validation of every
locally available Gemma model, and publishing completed work to GitHub. The
cross-tree work list is `BACKLOG.md` PROD1. Mechanism: portable model/backend
contracts, corrected population-aware evaluation, measured configuration
interactions, and conservative RL outside the token path. Gate: installed-product
correctness plus paired end-to-end validation against the unchanged product and
stock mlx_lm references. Kill: missing evidence, resource/quality failures or no
net gain leaves the reference active. Historical sealed artifacts stay unchanged.

## Serving stability follow-ups (2026-10-15)

The serving-stop fix (STAB1) and the CUDA memory diagnosis (MEM2-D) shipped; these stay open.

### MEM2 — CUDA memory of a long-running tuned server grows to the card's limit (2026-10-15)

**Mechanism (diagnosed, ledger MEM2).** MLX 0.32.2's buffer cache of freed buffers grows to 11 GiB on a 15 GiB T4
(its limit is 0.95 of the device) beside about 3 GiB MLX does not count; the device fills and a prefill or CUDA graph
fails. Live arrays return to 703 MiB at every group end; the CUDA graph cache is not the grower. **Candidate built**
(`e94e229`): a 1 GiB cache bound on CUDA. **Test.** Run the preregistered [MEM2-C](https://github.com/Tobayko/IronMule-Research/blob/main/research/MEM2-C_PREREGISTRATION.md)
as fixed (two notebooks, about 6.5 GPU hours); its first run was stopped by the user before any result. **Kill.** As
MEM2-C fixes it: not fewer worker losses, candidate memory above 8 GiB, a tokens/s or first-content regression beyond
the A/A noise, less answered work, or any changed output.

### HOL1 — A queued request times out behind a group that holds the worker until its own deadline (2026-10-15)

**Mechanism.** Groups run until their slowest member ends and joining is off, so requests queued behind a long group
wait for it; in STAB1 run 2 four short prompts waited 9.04 s behind two requests that could not finish, started
0.96 s before their own deadline, got no first token and ended with `504` when that deadline expired; their group
finished 0.07-0.08 s later. Inside a group, the worker also holds each member's last token and
completion until the engine run ends (it checks the streamed tokens against the result). **Test.** Deadline-aware
admission with the live learner's existing cost model (shed or reorder a request that cannot meet its deadline), and
releasing a finished member's completion once its tokens are verified, on the STAB1 deadline scenario and overload
replay. **Kill.** No fewer timeouts at equal throughput, any output or token-check change, or a learned rule that is
not qualified like every other arm.

### HTTP1 — Past the accept limit most refused clients see a reset, not the 429 (2026-10-15)

**Mechanism.** The accept loop writes the 429 and closes at once while the client is still sending its request, so
the client's write fails: in STAB1 run 2's burst of 192, 95 of the 128 refused clients saw `BrokenPipe` or
`ConnectionReset` (never accepted, so no request was lost, but the backpressure signal is lost). **Test.** Half-close
and a short bounded drain before closing, or answering only after the request head arrived, on the same burst.
**Kill.** No fewer resets, more threads than the 64 workers, or a slower accept loop under the burst.
**Not reproduced locally (2026-10-28).** On Linux x86_64 loopback (a 4-core cloud container, Python 3.13), 64 held
handlers and 128 refused clients that each send a POST and read: all 128 got the 429 in every one of 27 bursts,
across bodies of 200 B to 400 KB and sends delayed 0 to 100 ms after connect (the kernel kept the queued 429
readable after the reset). So the resets need STAB1's own client pattern or host; reproduce with that client
before building a drain, or the fix cannot be shown to work.

## The whole machine, unquantized checkpoints, NVIDIA on Kaggle (2026-10-28)

Opened from the maintainer's direction of 2026-10-28: the target is a Kaggle NVIDIA session (Tesla T4, also
offered as two T4s), the learner should use and tune the whole machine (every GPU, the host's cores, memory) for
whatever model runs, and checkpoints are not quantized. Every number published so far is a 4-bit checkpoint on one
GPU, so none of it reaches this box; each entry below needs its own Kaggle run. Nothing here is measured yet.

### HW1 — Serve from every GPU the machine has

**Mechanism.** The server drives one worker on one device; on a "GPU T4 x2" session half the card memory and
bandwidth idles. One worker per GPU (pinned with `CUDA_VISIBLE_DEVICES`), each with its own unchanged scheduler,
grouping, recovery and learning, behind join-the-shortest-queue dispatch, which is optimal for identical
servers with exponential service (Winston 1977, J. Appl. Prob. 14, 181-189; extended by Weber 1978 to service
times with non-decreasing hazard rate) and a strong heuristic beyond that. A
request never spans workers, so per-request tokens and plans are what one worker returns today. **Test.** Kaggle T4
x2, the P3/STAB1 synthetic traffic, one worker against two, paired by trace; the same prompts sent to both workers
must return the same tokens (same card model, same kernels). **Kill.** Aggregate tokens/s at four or more
concurrent clients under 1.5x one worker, any token difference between the two workers, or first-content p95 at
one client worse than one worker beyond the A/A noise. *Seen so far (RL2-H run 1, below):* a two-worker pool on
two T4s served 6,048 requests of Qwen 3 1.7B bfloat16 without a failure, and each request returned the same
answer digest in all 21 replays, which placed it on either card; throughput against one worker is still unmeasured.

### HW2 — Split one unquantized model across both T4s (tensor parallel decode)

**Mechanism.** Decode reads every weight once per token, so it is bandwidth-bound: an unquantized 4B checkpoint
(about 8 GB of bfloat16) needs about 25 ms per token at the T4's 320 GB/s, an 8B one does not fit a 15 GB card at
all. Sharding each projection across two cards halves the bytes per card at the cost of two all-reduces per
layer over PCIe (Kaggle's T4s have no NVLink). MLX 0.32's CUDA build loads NCCL at run time from the
`nvidia-nccl-cu12` wheel and mlx-lm shards supported architectures for `mx.distributed`. The all-reduce changes the
summation order, so this is a numeric plan (quality gate), never exact. **Test.** First whether
`mx.distributed.init(backend="nccl")` works on Kaggle's two T4s with the pip wheels; then Qwen 3 4B and 8B bfloat16,
stock single card (4B) against two-card decode, plus the perplexity gate. **Kill.** NCCL unavailable from the
wheels, under 1.2x on the 4B, or the gate fails.

### U1 — Unquantized bfloat16 checkpoints on CUDA without native bfloat16

**Mechanism.** Turing and Volta have no bfloat16 arithmetic. For a dense checkpoint the float16 plan casts the
weights, and a bfloat16 value (8 significant bits) is exactly representable in float16 (11 significant bits)
whenever its magnitude lies within float16's range, so the plan changes only activation arithmetic when every
weight is in range, unlike a 4-bit checkpoint whose scales are what gets cast. Stock MLX may already be close to
bandwidth on single-row decode (its CUDA `gemv` accumulates bfloat16 in float32), so the gain may sit in prefill
and grouped rows, which go to cuBLAS. **Test.** Kaggle T4: unquantized Qwen 3 0.6B/1.7B/4B and Gemma 3 1B, stock
against the exact knobs, `float16` and `float32`, plus each plan's quality gate; offline, count the weights a
float16 cast does not keep exactly. **Kill.** Per plan and checkpoint: not faster than stock, or the gate fails
(Gemma 3's activations already broke float16 on its 4-bit checkpoint, see the quality table in `BENCHMARKS.md`).

**Offline, 2026-10-28 (weights only, no hardware claim).** Every bfloat16 weight of five unquantized checkpoints
cast to float16 and back, compared with its float32 value: none overflowed, and 0.013-0.024 % changed. A bfloat16
value has 8 significant bits and float16 holds 11, so only magnitudes below float16's normal range (2^-14) can
change at all, each by at most 2^-25 (half a subnormal step); 708 to 5,558 per model flushed to zero. The largest
magnitude was 500 (Gemma 3 1B), far below float16's 65,504.

| Checkpoint (revision) | bfloat16 weights | inexact | flushed to zero | max abs |
| :-- | --: | --: | --: | --: |
| `Qwen/Qwen3-0.6B` (`c1899de2`) | 751,632,384 | 120,378 (0.016 %) | 708 | 192 |
| `mlx-community/gemma-3-1b-it-bf16` (`60444cd9`) | 1,301,875,840 | 261,355 (0.020 %) | 1,524 | 500 |
| `mlx-community/Llama-3.2-1B-Instruct-bf16` (`863c846a`) | 1,235,814,400 | 290,708 (0.024 %) | 1,689 | 2.92 |
| `mlx-community/Qwen3-1.7B-bf16` (`9cd66928`) | 1,720,574,976 | 226,844 (0.013 %) | 1,356 | 153 |
| `mlx-community/Qwen3-4B-bf16` (`3639b521`) | 4,022,468,096 | 936,720 (0.023 %) | 5,558 | 44 |

So for these checkpoints the float16 plan is, to within those subnormal roundings, the same weights with
different activation arithmetic; whether the activations fit is the gate's question. Autopilot's numeric choice
used to pick `native` for any Qwen 3 or Gemma 3 checkpoint on such a card, which an unquantized one cannot load;
it now treats `native` as a no-op there (commit `375df61`). **Open for the maintainer:** that choice still takes a
plan the table recommends per *architecture* without gating it, so an unquantized Qwen 3 would get `float16` on the
evidence of 4-bit checkpoints (`test_recommended_plan_is_taken_without_a_gate_and_remembered` pins this), while
`numeric_plans.recommend` limits a recommendation to the checkpoints measured (NEXT1-C). The two rules disagree;
this entry's gate result should decide which one applies to unquantized checkpoints.

### U2 — Float16 matmuls with the checkpoint's own activations, for Gemma 3 on a T4

**Mechanism.** The float16 plan casts everything, and Gemma 3's activations leave float16's range (its float16 gate
doubled perplexity on the 4-bit checkpoint). Keeping activations, norms and the residual stream in the checkpoint's
bfloat16 and running only the matmuls in float16, on the tensor cores Turing has, is how Unsloth runs Gemma 3 on T4s
(an engineering report, not a peer-reviewed one: unsloth.ai/blog/gemma3); IronMule's own `native` plan already
runs its 4-bit prefill GEMM in float16 and passed Gemma 3 12B's gate. By U1, the weights cast exactly. A matmul's own
output can still overflow float16. **Test.** `tools/kaggle_whole_machine.py` measures it as a fourth U1 arm
(`mixed16`, defined only in that tool) with the same gate and speed check; offline first, whether it removes the
float16 plan's non-finite or doubled perplexity on Gemma 3 1B on a CPU. **Kill.** Not faster than stock on the T4,
or its gate fails; only then is it worth a product plan.

### RL2 — One controller over the whole machine, where a decision has state-dependent headroom

**Mechanism.** The learner should decide for whatever model runs: which GPU takes a request (HW1), how wide a
group is and how long it waits, how much freed memory MLX may cache (MEM2), how many host threads detokenize.
**Evidence against a learner by default.** RSI3-LIVE-P3 (contextual bandit, RL, RL with dreaming) and OPT1
(sequential RL, Dream-RSI) lost to the best fixed setting: where one setting is best in every state, a learner
can only match it. In input-driven systems, exogenous traffic also inflates policy-gradient variance unless
policies are compared on the same input sequence (Mao et al., ICLR 2019, arXiv:1807.02264), which the replay
traces already do. **Test.** After HW1 and MEM2 exist: measure headroom first, the per-state best against the best
single setting on the same recorded traces (as OPT2-CAL did); train and qualify a learner only where that headroom
exceeds the A/A noise. **Kill.** Headroom inside the A/A noise for every decision: a fixed policy ships, no learner.

**RL2-H, the headroom measurement (protocol fixed 2026-10-28, before data).** `tools/rl2_headroom.py`, one
Kaggle "GPU T4 x2" kernel, `mlx-community/Qwen3-1.7B-bf16` on the tuned engine, one HW1 worker per card.
*Decisions measured:* placement (`shortest`, HW1's join-the-shortest-queue, against `fill`, the lowest-index card
with fewer than `width` requests outstanding, else the shortest) and the live learner's schedule catalogue (width
1, 2, 4, 8 x gather wait 0, 50, 150 ms, width 1 never waiting): 20 fixed arms, reference `shortest/w8/g0`.
The MEM2 cache bound waits for MEM2-C's own comparison, and no product setting for detokenizing threads exists
yet, so neither is in this run. *States:* offered load at 0.2, 0.6 and 0.95 of the reference's capacity (requests
per second from a burst of 16 per card), equal weights. *Traffic:* per level six fixed-seed traces of 16 Poisson
arrivals (generic prompts, 64 or 128 tokens), replayed identically under every arm. *Order:* per (trace, level)
block the reference, the other arms rotated per block and reversed on odd blocks, the reference again (A/A).
*Cost:* the live learner's reward per request (time in the system from the scheduled arrival + time to first
content, or 2 x deadline + 1 s for a failed, timed-out or refused request), mean per cell. *Headroom:*
H = 1 - oracle / fixed, the oracle taking the cheapest arm per level and fixed the cheapest single arm; reported
in sample and held out (arms chosen on five traces, scored on the sixth), jointly and per decision (the oracle
varies only that decision). *Noise:* the 95th percentile of the A/A relative differences. *Release rule:* a
decision gets a learner only if the 2.5 % bootstrap bound of its held-out H (2000 resamples, seed 0) exceeds
that noise. *Validity:* any answer digest that differs between arms, a non-finite cost, a failed calibration or
fewer than two complete traces per level makes the run invalid; an invalid run releases nothing. H is an upper
bound for a policy that picks one of these fixed arms from the load level (such a policy sees only the
arrival-rate estimate and pays switches), so passing it is necessary, not sufficient: a learner must still pass the
unchanged protocol-2 qualification. *(Corrected 2026-10-28, after run 1:* the text first said "an upper bound"
without that scope. A policy that reacts to each request, to its length and to the queue at that moment, is not
bounded by H; see RL3 and RL4.) 378 cells, an estimated three to four session hours.

**RL2-H run 1 (2026-10-28, valid): no headroom for either decision; no learner.** Kaggle "GPU T4 x2" (two Tesla
T4, 15 GiB, driver 580.178.04, 4 CPUs), MLX 0.32.4, mlx-lm 0.31.3, `mlx-community/Qwen3-1.7B-bf16` (`9cd66928`),
tuned engine, one worker per card, commit `258ccbe`, kernel `tobayko/ironmule-rl2-headroom` version 1 (4.75 session
hours). Calibration: 32 of 32 burst requests, capacity 0.96 requests/s, so the levels were 0.19, 0.58 and 0.91
requests/s. All 378 cells and 6,048 requests completed, none failed, timed out or was refused, and every request's
answer digest was identical across all 21 replays (arms and A/A runs, served by either card). A/A noise 3.5 %.
Joint headroom 0.12 % in sample and -0.05 % held out (95 % bootstrap -0.30 % to +0.13 %); placement 0 %; schedule
the same as joint. The reference `shortest/w8/g0` is within 0.4 % of the cheapest arm at every level (best arm
over all levels `shortest/w2/g0`, 0.03 % cheaper, far inside the noise), so the reference stays. The decisions do
matter, just never in a state-dependent way: `fill` placement at width 2 or more cost 1.23-2.83x the reference
(worst at width 8 under load; at width 1 it fills an idle card first and behaves like the shortest queue), and
gather waits of 50 ms and 150 ms cost 4-5 % and 14-18 % more. The tool's verdict line first read "the fixed
policy `shortest/w2/g0` ships"; after this run it was reworded to keep the reference when the best fixed arm is
inside the noise (the release rule is unchanged). Not measured: other models, quantized checkpoints, longer outputs
or prompts, bursty or real traffic, one card, the MEM2 cache bound and detokenizing threads. The ledger entry belongs
in the research repository, which this session cannot reach.

**RL2-H2, run 2 with queues and long requests (protocol fixed 2026-10-28, before its data).** Re-opens the Tier 0
entry "RL2 learner for GPU placement and group schedule" (`RL2-H-run1-20261028`), whose kill reads "Headroom inside
the A/A noise for every decision: a fixed policy ships, no learner". Condition met: a changed mechanism. Run 1's
Poisson arrivals with 64-128-token answers kept about 2.4 requests in the system even at its heaviest level, so groups
rarely formed and no request held a group long; the states where width, wait and placement could matter differently
(bursts that build a queue, long answers holding a group while short ones wait, as in HOL1, and long prompts that make
prefill matter) never occurred. Same model, cards, engine, pool, cost, order, headroom, noise, release rule and
validity as RL2-H (`tools/rl2_headroom.py --plan h2`). *States:* `quiet` (run 1's light level: Poisson at 0.2 of
capacity, short prompts, 64 or 128 tokens), `bursts` (four bursts of four arrivals inside 0.2 s each, burst starts
Poisson at 0.6 of capacity / 4) and `spike` (all 16 arrivals inside 1 s); in `bursts` and `spike` half the prompts
carry a fixed generic context of about 1,000 tokens and the output limit is 32 or 256 tokens; equal weights.
*Arms:* the ten `shortest` schedules and `fill` without a wait at every width (14); `fill` with a wait lost by 27 % or
more at every level of run 1. *Traces:* five per state, seeds distinct from run 1's. 225 cells, an estimated four
session hours. **Kill:** as RL2's, per decision: the held-out headroom's 2.5 % bootstrap bound at or below the A/A
noise means no learner for that decision under this traffic either.

**Result, RL2-H2 run 1 (2026-10-29, `RL2-H2-run1-20261028`, kernel `tobayko/ironmule-rl2-headroom-h2` at
`517bacee4068c0dc080d75f4aece2065959ab11b`, Kaggle T4 x2, Qwen3 1.7B bf16 revision `9cd6692`, MLX 0.32.4): invalid,
no learner is released.** All 225 cells and 3,600 requests completed; none failed. Seven requests were answered
differently across arms. All seven are short prompts 0 or 5 with 32-token limits, the two prompts X1 found to flip
at a long-prompt group's capacity. Each split the arms into the same two answers, with width-1 arms always on one
side; the A/A repeat agreed with its reference. That is X1's mechanism, so this run has the same cause as RL3-H's.
3.5 session hours. *Post hoc, not a verdict:* with those seven digests masked, the timing analysis reads an A/A
noise of 4.8 %, and the joint held-out headroom is -0.3 %. No grouping was cheapest: `fill/w1/g0` cost 7.6 % less
than the reference, more than the noise, mostly in `bursts` and `spike`, where a group holds every member until
its slowest ends. Groups also decoded at the long prompts' capacity then, which X1 has since removed, so a re-run
on the fix could rank the widths differently.

### RL3 — Length-aware placement and order on two GPUs (protocol fixed 2026-10-28, before data)

**Mechanism.** With requests of mixed length, a long answer holds a group and its card while short ones wait
behind it (head-of-line blocking, HOL1). Two decisions can avoid that with what is known when a request arrives:
its output limit. *Placement by length* sends long requests to one card and short ones to the other; *order by
length* starts the waiting request with the smallest limit first (shortest-job-first, which minimises mean time in
the system for a single server). The scheduler today does neither: placement is the shortest queue, and order is
first in, first out. **Evidence.** Learning-to-rank ordering approximated shortest-job-first and cut chatbot latency
2.8x in vLLM (Fu et al., NeurIPS 2024, arXiv:2408.15792). In a simulator-only, single-author preprint, REINFORCE
routing across GPUs beat the shortest-queue heuristic by 48 % by learning to segregate fast and slow requests,
while on one GPU it gained only 0.1-1.0 % over a tuned static batch size (Sharifullin, arXiv:2607.05272, 2026). The
second finding is what RL2-H run 1 measured on real T4s. **Evidence against.** That gain needed heterogeneous
requests and a reward weighting fast requests ten times more. Here the cost weighs every request equally.
Shortest-job-first can starve long requests, so it ages: past 10 s a request starts by arrival. Placement by length
can leave one card idle while the other queues, which `spill` guards against.

**Test (RL3-H, `tools/rl2_headroom.py --plan h3`).** The same pool, model, engine, traffic (run 2's `quiet`,
`bursts` and `spike` traces, the same seeds), cost, order of cells, headroom, noise, release rule and validity
as RL2-H2. A request is `long` with an output limit of at least 128 tokens. Arms: placement `shortest`
(reference), `segregate` (long to the last card, short to the first) or `spill` (`segregate` unless the preferred
card has at least four more requests outstanding than the other), times order `fifo` (reference) or `sjf`. All six
use the reference schedule (width 8, no wait). Both new decisions live in the tool only (a pool and service
subclass); the product is unchanged until a result says otherwise. 105 cells, an estimated two session hours.
**Kill.** Per decision, the held-out headroom's 2.5 % bootstrap bound at or below the A/A noise: no learner for
it. If one fixed arm beats the reference by more than the noise in every state, that fixed policy is a product
candidate with its own paired validation, and no learner is built. Only headroom that depends on the state
releases a learner. That learner would be trained offline on the recorded costs (every arm replayed the same
traces, so each state has a full-information cost per action), then qualified like every other arm.

**Result, RL3-H run 1 (2026-10-28, `RL3-H-run1-20261028`, kernel `tobayko/ironmule-rl2-headroom-h3` at
`35844a8ac29303e19682242c9ee6fba1da1e9c5c`, Kaggle T4 x2, Qwen3 1.7B bf16 revision `9cd6692`, MLX 0.32.4): invalid,
no learner is released.** All 105 cells and 1,680 requests completed; none failed, timed out or was refused. But
request 14 of `bursts` trace 1, a short prompt with a 32-token limit, got one answer under both `shortest` arms
(three replays, the A/A repeat included) and another under all four `segregate` and `spill` arms. The second answer
is also what every arm gave the same prompt in `spike` trace 1. The other 88 distinct requests agreed across every
arm. The validity rule fixed before the data makes the run invalid; X1 below looks for the cause. 1.7 session
hours. *Post hoc, not a verdict:* with that one digest masked, the timing analysis reads an A/A noise of 7.9 %. The
best fixed arm is `shortest/w8/g0/sjf`, 1.2 % cheaper than the reference and well inside that noise. The joint
held-out headroom is -1.6 %, and segregating by length cost 21 % more in `bursts`. RL4's gates are defined on
RL3-H's data, so RL4 does not go to hardware on this run. A post hoc preview of RL4 on these timings is not its
gate either. The model refitted to them meets the fidelity criteria: every cell within 21 %, and at least 90 % of
the resolved pairs ordered right. The trained policy beats the best fixed arm by 4.1 % in the simulator. That is
below this run's 7.9 % noise, so the simulator gate would fail.

**Result, RL3-H run 2 (2026-10-29, `RL3-H-run2-20261029`, kernel `tobayko/ironmule-rl2-headroom-h3` version 2 at
`6eaa4b4906d8ee5ab5ef8a15f8b8efdfe71b08d9`, X1's fix; otherwise as run 1): valid, no headroom, the reference
stays and no learner is built for placement or order.** All 105 cells and 1,680 requests completed, and every
request got the same answer under all six arms, so X1's fix holds on the T4s. The A/A noise is 18.7 %. That is
the nearest-rank 95th percentile of 15 pairs, which is their largest. The best fixed arm, `shortest/w8/g0/sjf`,
is within 0.02 % of the reference. Held out, the joint decision's headroom is -0.7 % (95 % bootstrap -3.0 % to
+1.1 %), placement's is -1.2 % and order's is +0.6 % (+0.2 % to +1.2 %); every bound is below the noise.
Segregating by length cost 15-24 % more in `bursts` (17.3 for the reference, 19.9-21.5 for the other placements).
1.7 session hours. *Post hoc:* one A/A pair (`bursts` trace 0, 17.9 against 14.8) sets that noise; the next
largest is 6.6 %, and every headroom bound is below that too. The rule stays as fixed.

### X1 — A short request's answer depended on its group on the T4s (opened 2026-10-28, before data)

**Finding.** RL3-H above. A group shares one KV capacity, sized to its longest prompt plus its largest output limit
(`MLXBackend.capacity_for`), and a request that joins a running group must fit it (SERVE8). Under `shortest`,
request 14 could share its card's group with the three long prompts that arrived with it. It then decoded at a
capacity of about 1,100 slots instead of 64; under `segregate` it could not. Decode attends over the whole
fixed-shape cache through a mask, and EXACT2 holds that the padding length does not move a token. On the MLX CPU
build that holds for this prompt at capacities 64 and 1,088: the logits were bit-identical over 32 steps (Qwen3
0.6B). It was never checked on CUDA, and LIMITS lists ragged prompt lengths within one group as untested. MLX
0.32.4's CUDA backend skips its fused decode attention whenever a mask array is passed (`supports_sdpa_vector`),
and `FixedKVCache` always passes one. Decode therefore runs the unfused path over the whole cache: matmul, softmax,
matmul. The reductions in that path can be ordered differently for different cache lengths, and that can flip a
near-tied greedy token.

**Test (`tools/rl2_headroom.py --plan x1`, Kaggle T4 x2, about half a session hour).** The model and revision are
RL3-H's. Each card gets its own child process, pinned as HW1 pins a worker. The engine is the one a worker without
a tuned profile runs (BASELINE knobs). (a) Request 14 is served alone and in its RL3-H burst (requests 12-15)
through `Runtime.serve`, three times each. (b) Each of the 16 short prompts is decoded directly for 32 tokens at
its own capacity and at 1,024 and 1,088. For each, the test records whether the tokens agree, the first differing
step, and the largest logit difference before it. **Decision (fixed before data), first match wins:**
- A device failed: incomplete.
- The same request was answered differently on the same device: nondeterminism, investigated first.
- Any capacity changed a token: the capacity moves tokens on CUDA. A request's capacity must stop depending on its
  group: each session is sized to its own prompt and limit, and the runtime contract tests gain the case. RL3-H is
  then re-run on that fix.
- The group changed the probe and no capacity did: grouped batch-1 is not exact here, investigated before any further
  RL run.
- The cards disagree: card identity joins the validity rule.
- Nothing reproduces: the product path (worker, admission, streaming) is reproduced next.

Until X1 is resolved, no RL run on CUDA counts, and LIMITS states the defect.

**Result (2026-10-29, `X1-run1-20261029`, kernel `tobayko/ironmule-rl2-headroom-x1` at
`cdc44937f96d1a9764be0e6b41ce9b488b995a67`, Kaggle T4 x2, Qwen3 1.7B bf16 revision `9cd6692`, MLX 0.32.4, 0.05
session hours): the KV capacity moves tokens on CUDA.** Both cards gave identical answers, and each repeat on a
card gave the same answer. Request 14 alone decoded at capacity 64 and gave exactly the answer RL3-H's `shortest`
arms recorded. In its burst it decoded at the group's 1,088 and gave exactly the answer of the `segregate` and
`spill` arms. Decoding that prompt directly at 1,088 gives the same second answer, so the capacity explains the
mismatch fully. Before their first differing token, all 16 short prompts' logits moved by 0.44-0.75 at capacity
1,024 or 1,088. Two of them (prompts 0 and 5) flipped a token within 32 (at steps 23 and 28). Under
`segregate`, request 14 had shared card 0 with the three long prompts of its burst, since all four had 32-token
limits. As the rule fixed before the data says, each session now prefills and decodes at the capacity it would
get alone (`build_sessions`). The group's capacity still bounds it, and a new runtime contract test, with a
backend whose tokens depend on the capacity, fails without the fix. The engine keeps one decode body per
capacity (at most eight), and the opt-in paired path pairs only equal capacities. The RL3-H re-run on the fix
(run 2, in RL3 above) answered all 1,680 requests identically across its six arms. Still owed: the ledger entry.

### RL4 — A model-based RL agent that places and orders each request (opened 2026-10-28, before data)

**Mechanism.** RL2-H's headroom bounds only policies that choose a fixed arm per load level. A policy that decides
per request can react to things a fixed arm ignores: that request's output limit and prompt length, and the
outstanding work on each card at that moment. Under bursts with mixed lengths, it can send a long request to the
card that will be free soonest, or keep short ones away from a card that a long group holds. This is what the
REINFORCE agent in arXiv:2607.05272 learned against the shortest queue. Training such a policy on hardware costs
GPU hours per episode, so it is trained in a discrete-event simulator of the HW1 pool instead (Dyna-style, as the
live learner already does). The pool model has per-card groups that run until their slowest member ends, no
joining, grouped prefill one member at a time, and a decode round time that depends on the group's width. It is
fitted to the per-request times RL2-H2 and RL3-H record on the T4s. The agent is a softmax policy over two
decisions. The first is the card for each arriving request, from that request's output limit and prompt length
and the outstanding requests on each card. The second is `fifo` or `sjf` at each group start, from the waiting
queue's composition. It is trained by REINFORCE with a running-mean baseline, the episode return being minus the
mean of the live learner's request cost, on traffic drawn from RL2-H2's state generators with seeds disjoint
from the hardware traces. **Evidence against.** The simulator can be wrong, the per-request decision can do
no better than RL3's best fixed arm, and every learned arm so far (P3, P4, OPT1) lost on hardware.

**Gates and test.**
- *Fidelity gate:* the fitted simulator must predict each of RL3-H's cells' mean cost within 25 %. In each
  state it must also order at least 90 % of the arm pairs that the hardware separated by more than RL3-H's A/A
  noise. Otherwise no policy trained in it goes to hardware. *(Amended 2026-10-28, before RL3-H's data, after a
  preview on RL2-H run 1:* the first wording asked for a Spearman rho of 0.8 over all arms per state. That
  failed on run 1 only because arms within 0.1 % of each other, far inside its 3.5 % noise, have no measured
  order to agree with. On run 1, after fitting to it, the model was within 3.4 % in every cell and ordered all
  53 resolvable pairs as the hardware did; that preview is not RL4's gate, which is RL3-H.)
- *Simulator gate:* on 200 held-out simulated traces per state, the learned policy must beat the best fixed arm
  by more than RL3-H's A/A noise.
- *Hardware (RL4-H):* `tools/rl2_headroom.py` gains a `learned` arm that applies the frozen policy. It runs
  against the reference and RL3-H's best fixed arm on RL3-H's states and traces, with the same cost, order, noise
  and validity as RL3-H. The policy is frozen and its digest recorded before the run.

**Kill.** The learned arm's held-out advantage over the best fixed arm has a 2.5 % bootstrap bound at or below
the A/A noise: no learner ships, and the best fixed arm stays as RL3 decides. Failing either gate stops RL4
before hardware. A learned arm that passes still needs the product integration (opt-in, off by default,
qualification and rollback as `docs/LIVE_LEARNING.md` describes) and its own paired validation.

**Result (2026-10-29, `RL4-SIM-20261029`, on RL3-H run 2's data; local, no GPU): the fidelity gate passes, the
simulator gate fails, and RL4 stops before hardware.** The model was fitted to RL3-H run 2 only (prefill 0.145 s +
2.65 ms per prompt token, decode round 19.8 ms + 13.4 ms per extra session; loss 0.0015). RL2-H2's times predate
X1's fix and were left out. It predicted every one of run 2's 18 cells within 6.1 % and ordered every pair the
hardware resolved; at 18.7 % noise that was only 6 pairs. REINFORCE ran 12,000 episodes on each of 4 seeds, and
seed 3 had the lowest validation cost. Over training its episode cost fell from 20.0 to 19.3. On 200 held-out
traces per state the policy (digest `e3f07b833e90`) cost 18.97 against 19.51 for the best fixed arm
(`shortest/w8/g0/sjf`), a gain of 2.8 %. Against the best rule outside the arms (`leastwork/w8/g0/sjf`) the gain
was 2.9 %. Per state: 2.44, 19.82 and 34.64 against 2.42, 19.94 and 36.15. The gate needs more than run 2's
18.7 % A/A noise, so it fails by a wide margin. It would also fail against the post hoc 6.6 %. The agent learns,
but what it learns is smaller than this testbed can measure. RL4's kill holds.

### RL4-S — The frozen RL4 policy on spike traffic, where the testbed is quiet (opened 2026-10-29, before data)

**Re-opens** the Tier 0 entry "RL4 model-based RL agent for per-request placement and order"
(`RL4-SIM-20261029`). Its kill reads: "In the simulator it beat the best fixed arm by only 2.8 %, against an
18.7 % A/A noise, so the simulator gate failed". Its re-open clause asks for "a testbed whose noise is several
times smaller than a few per cent". **New hardware evidence.** The 18.7 % came from one `bursts` pair. In
`spike`, all 15 A/A pairs of the three T4 runs differed by 0.2-4.5 %, and the five of RL3-H run 2 by at most
1.4 %. A spike delivers all 16 arrivals within a second, so the queue it builds is nearly the same on every
replay. The frozen policy (digest `e3f07b833e90`, trained before this entry) has its largest simulated gain
there. On RL4's 200 held-out spike traces it cost 34.64, against 36.15 for RL3-H run 2's best fixed arm
`shortest/w8/g0/sjf` and 35.99 for the best simulator-only rule: a gain of 4.2 %, three times run 2's spike noise.
Nothing is retrained or refitted for this entry.

**Test (`tools/rl2_headroom.py --plan h5`, Kaggle T4 x2, about 1.5 session hours).** RL3-H run 2's pool, model,
revision, engine, cost, cell order and validity. Traffic: `spike` only, from run 2's generator. It uses 30 traces
(indices 5-34) whose seeds no earlier run and no training, validation or held-out simulation used. Arms: the
reference `shortest/w8/g0`, `shortest/w8/g0/sjf`, and `learned` with the frozen policy, whose digest the kernel
verifies.

**Decision (fixed before data).**
- **Noise:** the nearest-rank 95th percentile of the 30 A/A pairs, which is their second largest.
- **Rival:** the cheaper of the two fixed arms over the 30 traces.
- **Gain:** one minus the learned arm's summed cost over the rival's, paired by trace, with a bootstrap over the
  traces (2,000 resamples, seed 0).
- **Pass:** the learned policy beats the rival only if the gain's 2.5 % bound exceeds the noise.
- **Validity:** every request's answer must be identical in all arms.

**Kill.** A 2.5 % bound at or below the noise: RL4 stays in Tier 0, now for `spike` as well. **A pass ships
nothing by itself.** It makes the policy a candidate for the product's opt-in learner, which is off by default,
needs its own qualification and rollback (`docs/LIVE_LEARNING.md`), and needs a paired validation on traffic that
mixes states.

**Result (2026-10-29, `RL4-S-run1-20261029`, kernel `tobayko/ironmule-rl2-headroom-h5` at
`8e41a53c1efedb3ba4f487f652d941a0e8c20c3e`, Kaggle T4 x2, Qwen3 1.7B bf16 revision `9cd6692`, MLX 0.32.4, 1.2 session
hours): valid, and the learned policy beats the best fixed arm.** All 120 cells and 1,920 requests completed, and
every request got the same answer under all three arms. The A/A noise was 2.9 % (second largest of 30 pairs). The
frozen policy (digest `e3f07b833e90`, verified by the kernel) cost 32.95 per request on average. The reference cost
36.59 and the rival `shortest/w8/g0/sjf` 36.23. Its gain over the rival is 9.0 %, with a 95 % bootstrap interval
of 6.9-11.1 %. The 2.5 % bound is far above the noise, so the pass rule fixed before the data is met. The policy
was cheapest on 29 of the 30 traces. Mean time to first content fell from 10.24 s to 9.80 s and mean total time
from 25.99 s to 23.15 s, with p95 total time 40.5 s against 41.1 s. The hardware gain is about twice the simulator's
4.2 %. The tool's generic verdict line names the learned arm as the cheapest "fixed" arm; that line is not this
entry's decision. As fixed, this ships nothing. It makes the policy a candidate for the product's opt-in learner,
and that candidate still needs its qualification, rollback and a paired validation on traffic that mixes states.
Still owed: the ledger entry.

### LD1 — The learned dispatch in the product, opt-in and bound to its evidence (2026-10-29)

**What it is.** RL4-S passed, so its frozen policy may become a candidate for an opt-in learner. LD1 is that
product path. `ironmule_product/learned_dispatch.py` holds the policy class, and a parity test shows it decides
exactly as the harness policy the hardware measured, on 500 random states. Through hooks it chooses each request's
card in the HW1 pool, shortest-job-first per group start in each card's scheduler, and, for a v2 policy, the group
width. It activates only with evidence that names the policy's digest, a passed comparison, and this server's
model, revision, GPU name and GPU count; anything else keeps the reference and says why in `/health`. After three
errors while deciding, or on the operator's `POST /v1/ironmule/dispatch` `{"action": "off"}`, it is withdrawn at
once; the withdrawal is recorded in the state directory and keeps the same policy under the same evidence off
across restarts, until new passed evidence or deleting the record qualifies it again. It is off by default,
refused together with `--learning`, and changes no answer. The RL4-S policy and its evidence ship in
`ironmule_product/dispatch/`.

**Still owed before it may be called better on real traffic.**
- A paired validation through this product path, on traffic that mixes quiet, bursty and spike states. In
  simulation the policy is neutral in `quiet` and `bursts`, so the test is that it costs nothing there and gains in
  spikes.
- The ledger entry. (The operator rollback without a restart is done; see above.)

### LD1-V — The product's learned dispatch on mixed traffic (opened 2026-10-29, before data)

**Question.** Does the product's own learned dispatch (LD1), running RL6's width-learning policy (digest
`fef74e9d1077`), beat the product's reference when quiet, bursty and spike traffic all count? RL6 measured that
policy only in `bursts` and `spike`. **Test (`tools/rl2_headroom.py --plan h7`, Kaggle T4 x2, about 2 session
hours).** Same pool, model, revision, engine, cost, cell order and validity as RL6. All three states are weighted
equally, with 8 traces each (indices 47-54) whose seeds no earlier run, training or simulation used.

**Arms.**
- The product's reference `shortest/w8/g0`: join-the-shortest-queue, first in, first out, width 8.
- `product/w8/g0`: LD1's `LearnedDispatch` with that policy, deciding through the product's own pool and scheduler
  hooks. This arm is a measurement, so it activates the policy without evidence.
- RL6's fixed schedules `shortest/w2/g0/sjf` and `shortest/w1/g0/sjf`.

**Decisions (fixed before data).**
- **Noise:** the nearest-rank 95th percentile of the 24 A/A pairs, which is their second largest.
- **(1) LD1 beats the reference** if its weighted gain over the reference, paired by trace with a 2,000-resample
  bootstrap, has a 2.5 % bound above the noise. Then evidence for this policy is written for all three states and
  shipped, and `--dispatch-policy` may use it.
- **(2) No harm.** In each state, LD1's gain over the run's cheapest fixed arm has a 2.5 % bound above minus the
  noise.
- **(3) The fixed candidate.** `shortest/w2/g0/sjf`'s gain over the reference, quiet traffic included, has a
  2.5 % bound above the noise. That confirms RL6's candidate for a product default.
- **Validity:** every answer identical across arms, and every product cell decided by an active dispatch.

**Kill.** If (1) or (2) fails, no evidence is written for this policy and LD1 keeps only RL4-S's spike-bound
evidence. If (3) fails, the default schedule stays.

**Result (2026-10-29, `LD1-V-run1-20261029`, kernel `tobayko/ironmule-rl2-headroom-h7` at
`bfe7ee65703b255beef8e37782af894dd083dd64`, Kaggle T4 x2, Qwen3 1.7B bf16 revision `9cd6692`, MLX 0.32.4, 1.7 session
hours): valid.** All 120 cells and 1,920 requests completed, and every answer was identical across the four arms.
Every product cell was decided by an active dispatch with no decision error (384 placements, 283 widths and 11,327
order decisions in total). The A/A noise was 8.1 % (second largest of 24 pairs), larger than RL6's 3.0 %.

- **(1) passes.** LD1 cost 21.2 % less than the reference (95 % bootstrap 18.3-24.3 %).
- **(2) fails in `bursts`.** Against the cheapest fixed arm, `shortest/w1/g0/sjf`, LD1 gained -6.9 % in `bursts`
  (95 % -12.9 to -1.1 %); the lower bound is below minus the noise. `quiet` (-2.4 %, 95 % -6.2 to -0.3 %) and
  `spike` (+7.0 %, 95 % 1.9-10.8 %) pass.
- **(3) passes.** `shortest/w2/g0/sjf` cost 18.4 % less than the reference with quiet traffic included (95 %
  14.9-22.2 %); `shortest/w1/g0/sjf` 19.5 % (16.1-23.4 %).

**Verdict, as fixed.** No evidence is written for RL6's policy, and LD1 keeps only RL4-S's spike-bound evidence
(Tier 0). RL6's fixed candidate is confirmed for a product default on this setup. Descriptive, not a rule: over all
three states LD1 gained 2.1 % on `shortest/w1/g0/sjf` (95 % -1.0 to 5.1 %), as in RL6. It wins in spikes and loses
in bursts. In `quiet` every arm is within 2.6 % of the others, LD1 the most expensive.

Measured costs per state:

| Arm | `quiet` | `bursts` | `spike` |
| --- | --- | --- | --- |
| reference `shortest/w8/g0` | 2.300 | 17.42 | 37.40 |
| LD1 `product/w8/g0` (RL6 policy) | 2.360 | 15.47 | 27.15 |
| `shortest/w2/g0/sjf` | 2.314 | 15.57 | 28.75 |
| `shortest/w1/g0/sjf` | 2.305 | 14.47 | 29.18 |

Still owed: the ledger entry. The fixed candidate's product change is done (BACKLOG RL6).

### RL6 — Group width and shortest-job-first on CUDA after X1, found by the RL agent (opened 2026-10-29, before data)

**Mechanism.** Before X1's fix a group decoded at its longest prompt's capacity. An extra session then cost a full
decode step: 20.0 ms against 19.8 ms alone, fitted to RL2-H2's width arms. That model predicted all 24 of that run's
cells within 4.1 % and ordered every resolved pair. With each request at its own capacity, the model fitted to
RL3-H run 2 charges 13.4 ms per extra session, so a width-4 round makes a token about 1.3x cheaper. A group still
holds every member until its slowest ends. Narrow groups that start the shortest requests first should therefore
beat the reference's width 8 in first-in, first-out order under load.

**Found by the agent.** REINFORCE was given a width decision (`train --widths`: a softmax over widths 1, 2, 4 and 8
per group start, beside card and order). Trained in the RL3-H run 2 model, its episode cost fell from 18.5 to 17.5.
The frozen policy (digest `fef74e9d1077`, schema v2) cost 17.08 on 200 held-out simulated traces per state, against
19.53 for the reference: 12.5 % less. It did not beat a fixed rule. `shortest/w2/g0/sjf` cost 17.14, so the
learned policy's simulator gate (0.4 %) fails. The agent mostly rediscovered that rule.

**Evidence against.** The run 2 model saw only width-8 arms after the fix, so its prices for widths 1-4 come from
rounds inside those groups, not from measured narrow arms. RL2-H2 measured width 1 best, but before the fix.

**Simulated per-state costs** (`bursts`, `spike`), recorded before the run for the fidelity check:

| Arm | `bursts` | `spike` |
| --- | --- | --- |
| reference | 19.95 | 36.23 |
| `shortest/w1/g0/sjf` | 18.31 | 32.14 |
| `shortest/w2/g0/sjf` | 18.32 | 30.69 |
| `shortest/w4/g0/sjf` | 19.46 | 31.40 |
| learned | 18.25 | 30.56 |

**Test (`tools/rl2_headroom.py --plan h6`, Kaggle T4 x2, about 2 session hours).** RL3-H run 2's pool, model,
revision, engine, cost, cell order and validity. Traffic: `bursts` and `spike`, weighted equally, from run 2's
generator. Each state gets 12 traces, indices 35-46, whose seeds no earlier run, training or simulation used. Arms:
the reference `shortest/w8/g0`, `shortest/w1/g0/sjf`, `shortest/w2/g0/sjf`, `shortest/w4/g0/sjf`, and `learned`
with the frozen v2 policy, whose digest the kernel verifies.

**Decisions (fixed before data).**
- **Noise:** the nearest-rank 95th percentile of the 24 A/A pairs, which is their second largest.
- **The fixed candidate, named now:** `shortest/w2/g0/sjf`. Its gain over the reference is paired by trace, with a
  bootstrap over traces (2,000 resamples). If the 2.5 % bound exceeds the noise, it becomes the candidate default
  schedule for CUDA. It then still needs its own product change and a paired validation that includes quiet
  traffic; Apple is unchanged.
- **The learned arm:** it beats every fixed arm only if its gain's 2.5 % bound over the cheapest fixed arm of the
  run exceeds the noise. The simulator predicts it will not.
- **Fidelity, descriptive:** the per-arm predictions above against the measured costs.
- **Validity:** every request's answer identical across all arms.

**Kill.** For the candidate, a bound at or below the noise: the reference schedule stays for these states and the
result goes to Tier 0. For the learned arm, the same: no width learner. That is also what its simulator gate
predicts.

**Result (2026-10-29, `RL6-run1-20261029`, kernel `tobayko/ironmule-rl2-headroom-h6` at
`323a9b82ec056c433c5f393c3dbdd4616ecb1dd0`, Kaggle T4 x2, Qwen3 1.7B bf16 revision `9cd6692`, MLX 0.32.4, 1.6 session
hours): valid.** All 144 cells and 2,304 requests completed, and every answer was identical across the five arms.
The A/A noise was 3.0 % (second largest of 24 pairs).

**The candidate passes.** `shortest/w2/g0/sjf` cost 18.0 % less than the reference (95 % bootstrap 15.0-20.5 %).
That makes it the candidate default schedule for CUDA, pending its product change and a paired validation that
includes quiet traffic. `shortest/w1/g0/sjf` did slightly better at 18.6 % (15.8-21.5 %), and
`shortest/w4/g0/sjf` reached 11.9 % (9.4-14.6 %).

**The learned arm does not pass.** Against the cheapest fixed arm, `shortest/w1/g0/sjf`, it gained 0.8 %
(-1.9 to 3.4 %), as its simulator gate predicted. Against the reference it gained 19.2 %, a descriptive
comparison that was not the rule. The agent found the narrow, shortest-first schedule; it adds nothing beyond it.

**Fidelity (descriptive).** In `bursts`, every arm was predicted within 4.9 %. In `spike` the model overpriced
narrow groups, by 24.4 % at width 1, 17.7 % at width 2 and 17.8 % for the learned arm. So it ranked width 2 ahead
of width 1, while the hardware ranked them the other way round, within 1 %.

Measured costs per state:

| Arm | `bursts` | `spike` |
| --- | --- | --- |
| reference | 20.98 | 34.08 |
| `shortest/w1/g0/sjf` | 18.99 | 25.83 |
| `shortest/w2/g0/sjf` | 19.11 | 26.07 |
| `shortest/w4/g0/sjf` | 20.40 | 28.12 |
| learned | 18.53 | 25.94 |

The paired validation with quiet traffic is done: LD1-V's (3) passed (18.4 %, 95 % 14.9-22.2 %). **Product change
(2026-10-29):** `ServingPolicy` gained `order` (`fifo` or `sjf`), and the scheduler starts shortest-first under
`sjf` exactly as the harness's arm did. The schedule ships with its comparison in
`ironmule_product/dispatch/rl6-t4x2-qwen3-1.7b/schedule.json` and serves by default only on an exact match of
engine (`tuned`), model, revision, GPU name and GPU count, when no serving policy is stored (`--schedule reference`
opts out). Still owed: the ledger entry. TB1 can change the best width: if true batching is built, this schedule
is re-derived, not carried over.

### HW3 — What a decode step costs on the T4, and which optimisations and combinations keep every token (opened 2026-10-29, before data)

**Why.** The owner's list of 2026-10-29 puts measuring the hardware first and asks for combinations of
optimisations, not single ones. On Metal, `B24` priced host time (17-32 % of a step) and `B10` found fusing the
small kernels worth under 0.7 % (Tier 0). Neither number exists for CUDA; the T4 is new hardware under the
re-opening rule, so `B10`'s kill ("fusing all of them away returns under 0.7 %") is re-measured here, not assumed.
X1 showed that on CUDA a change in the order of the arithmetic can move a token, so every candidate is checked for
exactness before its speed means anything.

**Test (`tools/cuda_probe.py`, Kaggle T4 x2, first card, about 1.5 session hours).** Qwen3 1.7B bf16 revision
`9cd6692`, MLX 0.32.4, greedy.
- **Knob factorial.** The engine's existing knobs `fuse_projections`, `compiled_fixed_cache`, `fused_argmax`,
  `readback_every` 1 or 4 and `speculate_k` 0 or 3 (prompt lookup), all 32 combinations, through `Engine.generate`
  on six prompts (three short questions, three behind about 1,000 tokens of context, chat template), 64 tokens
  each. One untimed pass, then five timed rounds; in each round the baseline runs first and last (an A/A pair) and
  the other 31 in an order rotated by the round. Fused combinations run on a second copy of the model.
- **Anatomy.** mlx-lm's own decode loop after a short and a long prompt, synchronous against pipelined (the next
  step built while the card runs this one), five alternating pairs each. Attention at the model's shapes, masked
  as `FixedKVCache` masks it against unmasked on keys cut to the context (the fused path X1 found disabled), timed
  alternately, at 1 and 8 rows and 64, 1,024 and 2,048 tokens. Every linear layer of block 0 and the output
  projection at one row, for the bandwidth floor.

**Decisions (fixed before data).**
- **Exact:** a combination's tokens equal the baseline's on every prompt in every pass.
- **Cost:** decode milliseconds per token; per (prompt, round) the ratio to that round's mean baseline; the
  geometric mean with a 2,000-resample bootstrap. **Noise:** the largest distance from 1 of the five A/A ratios.
- **A combination qualifies** for the CUDA tuned profile if it is exact and its ratio's upper 95 % bound is below
  1 minus the noise. The tuner already searches these knobs; this says which combinations it may use on this card.
- **Combinations:** the main effect and two-factor interaction of each knob on log decode time, from the full
  factorial.
- **(8) compile:** `compiled_fixed_cache` alone qualifies, or compiling the step is closed on this card.
- **(10) speculation:** recorded as exact or not, with its acceptance. This is the CUDA check of the claim that
  verifying drafts keeps the target's tokens; on CUDA the verify forward has other shapes than a decode step.
- **(16) overlap:** worth building if pipelining saved time in all five pairs at either context.
- **(6) fused attention:** worth building if the cut attention is faster with a 95 % bound above zero at any
  measured shape; its exactness and its share of a long-prompt step are recorded. It changes the order of the
  arithmetic, so it is class B either way.
- **(5)/(7):** the model's weight bytes over the best measured bandwidth is the floor; its share of a step bounds
  what any kernel or layout work can return. Recorded, no decision.
- **Validity:** a combination that errors on any prompt is invalid; the others stand. A crash keeps the rows so
  far.

**Result (2026-10-29, `HW3-run1-20261029`, kernel `tobayko/ironmule-cuda-probe` at
`c5c5104bfd8673b45c1105f6b2c4209dae1581c9`, Kaggle T4 x2 (first card), Qwen3 1.7B bf16 revision `9cd6692`, MLX
0.32.4, 0.8 session hours): valid.** All 32 combinations ran on all six prompts in the untimed pass and in five
rounds without an error, the baseline reproduced its tokens, and the A/A noise was 0.85 % (largest of five).
- **12 combinations qualify:** every combination with `compiled_fixed_cache` or `readback_every` 4 and without
  speculation. The best is `compiled_fixed_cache` + `fused_argmax` + `readback_every` 4 at 0.918 of the baseline's
  decode time (95 % 0.914-0.921). Without `fused_argmax` it is also 0.918 (0.915-0.921). By the HW4 rule this
  fixes HW4's decode winner: `compiled_fixed_cache`, `fused_argmax`, `readback_every` 4.
- **(8) compile qualifies:** 0.940 alone (0.936-0.946). `readback_every` 4 alone: 0.961 (0.958-0.965).
  `fuse_projections` is slower (1.010, 1.007-1.012), and `fused_argmax` alone changes nothing (1.002).
- **(10) speculation is not exact on this card.** Every combination with `speculate_k` 3 changed tokens, the
  first at steps 23 to 58 of 64. Acceptance was 0.42-0.47, and the ratio ran from 0.986 to 1.144.
- **Combinations (main effects on decode time):** compile 0.948, readback 0.940, speculation 1.107, fusion
  1.003, fused argmax 1.000. Every two-factor interaction is within 0.7 % of 1, except readback x speculation
  at 0.969. Compile x readback is 1.007: they share host time, so together they return 0.7 % less than the
  product of their single gains.
- **(16) overlap is worth building.** On mlx-lm's own loop, pipelining saved 13.5 % of a step after a short
  prompt and 12.5 % after a long one. That held in all five pairs, with the same tokens.
- **(6) fused attention is worth building.** Unmasked attention on keys cut to the context was faster at every
  shape: 194 to 79 us at 64 tokens and 276 to 157 us at 1,024, one row. Over 28 layers that is about 3.3 ms, 15 %
  of a long-prompt step. It is not identical (largest difference 0.001-0.006), so it is class B. On keys cut to
  the context the logits would no longer depend on the KV capacity (X1).
- **(5)/(7) floor:** the best bandwidth was 252 GB/s, the output projection at one row. 3.44 GB of weights then
  need 13.7 ms per token, 70 % of a short step (19.5 ms).
- **Post hoc phase table (RL7):** one combination wins every context, and the table's headroom is 0.0 % (95 %
  -0.05 to 0.01 %). For these five knobs one profile serves: no table and no learner. Prefill took 106 ms for
  17 prompt tokens and 1,726 ms for 1,026, and none of the five knobs moved it by more than +1.4 %.

### TB1 — True tensor batching on CUDA: exact, batch-invariant, and what a row costs (opened 2026-10-29, before data)

**Mechanism.** Grouped batch-1 reads every weight once per request: RL6's fitted model charges 20.0 ms for a step
alone and 13.4 ms for each further session. Stacking the rows into one forward reads the weights once per round,
so a further row should cost little, and a token at width 8 several times less. **Against:** `E14b`'s true
batching diverged by one token at batch 8 on Metal, `B28` failed its correctness gate, and on CUDA even the KV
length moves logits (X1). Old entries: LIMITS "No true tensor batching. It diverged at batch 8 in E14b", `B12`'s
kill "Divergence at 16 or beyond". Re-opened for new hardware (T4, MLX 0.32.4 CUDA) and a new question: not only
whether stacking matches the batch-1 path, but whether the kernels are batch-invariant, so a versioned batched
plan could be exact by itself.

**Test (in HW3's kernel).**
- **Kernel level.** Each linear layer of block 0 and the output projection, with its real weight, on M = 1, 2, 3,
  4, 8, 16, 32 rows of fixed random input: is each row bit-identical to the same row alone, and how long it takes.
- **Model level.** 16 distinct rows of 64 and of 1,024 tokens, each prefilled alone with mlx-lm's own cache, then
  decoded 32 steps alone and stacked at B = 2, 4, 8, 16. Per row: tokens equal to its decode alone, the largest
  logit difference at the first step (same inputs), the time per step. Equal lengths keep masks and padding out,
  so only the row count differs.

**Decisions (fixed before data).**
- **Batch-invariant (class A)** if every stacked row's first-step logits are bit-identical to its decode alone at
  every B and length: a batched path then keeps every answer and is built next.
- **Token-exact, not invariant** if all tokens agree but some logits do not: class B; a batched path needs
  batch-invariant kernels or a quality gate, and is not built as an exact path.
- **Not exact** if any token differs.
- **The lever:** 8 x the step time alone over the step time at B = 8, per length. Any lever above 1 counts. The
  per-row cost curve then replaces the 13.4 ms per session in RL4's simulator, which re-derives width and order
  for a batched path: RL6's narrow groups were best only because each further session cost 13.4 ms.

**Result (2026-10-29, in `HW3-run1-20261029`): not exact, and a small lever.**
- **Not exact.** Stacked rows never gave bit-identical logits (largest difference 0.25-0.31 at every B). Tokens
  changed in 2 of 8 rows at B = 8 and 2 of 16 at B = 16 for 64-token rows. Rows of 1,024 tokens kept their tokens
  over 32 steps.
- **The lever at 8 rows is 1.34x at 64 tokens and 1.27x at 1,024.** Any lever above 1 counts, so it "holds", but
  a stacked step costs 102-121 ms at B = 2 to 16, against 17.4 ms for one row.
- **The cause is one kernel path.** MLX's bf16 product at 2 to 32 rows is 5-7x slower than at one row, and flat in
  M. The output projection takes 2.47 ms at one row and 13.8-15.3 ms at 2 to 32 (252 against 41-45 GB/s). Rows
  were bit-identical to the row alone only at one row (`v_proj` up to 4, `q_proj` up to 2).
- **Combination with scheduling.** With this cost curve a stacked group is dearer than grouped batch-1 (20.0 ms
  plus 13.4 ms per session) at every width up to 7. Scheduling stays grouped batch-1, and RL6's narrow schedule
  stands.
- **Verdict, as fixed.** Not exact, so a batched path is not built as an exact path (Tier 0). The gap names one
  remedy for both the inexactness and the slowness: a small-M kernel that reads each weight once for every row
  and reduces each row in one fixed order (KB1, to be opened).

### HW4 — The prefill side, bit-exact knob sets, and what a further session costs per knob set (opened 2026-10-29, before data)

**Why.** The owner asked on 2026-10-29 for a learner that credits every improvement and learns to combine: "if
the start is strong but decoding then takes longer, another option may help there". On a ~1,000-token prompt,
prefill (about 2.8 s by RL4's fit) outweighs 64 decode steps (about 1.3 s), yet HW3 measures decode almost only
and leaves out both prefill knobs. Switching knobs per request also needs more than token equality on a sample: a
request's answer may depend on load only if its knob sets are bit-identical (class A). Finally, the width a
schedule should use depends on each knob set's cost per further session, which nothing has measured (RL6's model
mispriced narrow groups by 17-24 % in spikes). Designed with a second agent as sparring partner.

**Test (`tools/cuda_probe.py --plan hw4`, Kaggle T4 x2, first card, about 1.2 session hours).** Same model,
revision, MLX, prompts and output length as HW3.
- **Class A.** Every combination of `fuse_projections`, `compiled_fixed_cache`, `fused_argmax`, `speculate_k` (0
  or 3) and `head_skip_prefill` (32 sets; `readback_every` only moves the host's reads), at the product's capacity
  (`MLXBackend.capacity_for`). Per prompt: the prefill's last logits (read through `_project`) and state, then 63
  decode steps through the engine's own body fed the baseline's tokens, every step's logits (picks under
  `fused_argmax`), the final state; with speculation, the k + 1 verify window at positions 0, 16 and 32 against
  the sequential steps. All compared bit for bit with the baseline.
- **Separability.** `head_skip_prefill` x `fuse_projections` x HW3's decode winner (2 x 2 x 2) through
  `Engine.generate`, plus `prefill_into_fixed` alone, in HW3's round design (five rounds, A/A pair).
  `prefill_into_fixed` is re-measured on new hardware against `E1`'s kill ("the phase it would optimise costs
  1.47 ms of 537 ms"). **HW3's decode winner** is fixed by this rule before HW4 runs: the decode knobs of HW3's
  qualifying combination with the lowest ratio, without `fuse_projections` (crossed here); `compiled_fixed_cache`
  alone if none qualifies.
- **Widths.** Every class-A decode set, with `readback_every` 1 and 4, on the product's grouped path
  (`Runtime.serve`, `ThroughputMode`) at widths 1, 2 and 4, short prompts, five rounds.
- **Compile cost.** Per decode set, the first request at a new capacity minus the same request again, at three
  capacities.
- **Prefill floor.** Every linear layer of block 0 and the output projection at 256 and 1,024 rows: achieved
  TFLOP/s.

**Decisions (fixed before data).**
- **Class A:** the prefill logits and state, every decode step and the final state are bit-identical to the
  baseline's on all six prompts, and, with speculation, every verify window equals the sequential steps.
  `fused_argmax` returns no logits, so it is class A only if its picks and state match and its twin without it is
  class A; that assumption is recorded.
- **Separability:** the interaction of each prefill knob with the decode winner on log total cost (2 x prefill +
  n x decode, n = 16, 64, 256), bootstrapped over (prompt, round) cells. If every 95 % interval contains 1, the
  best combination is the best prefill set plus the best decode set; otherwise they couple and are optimised
  together.
- **A prefill set qualifies** when it is class A and its prefill ratio's upper 95 % bound is below 1 minus the
  noise (HW3's rule, on prefill).
- **Cost per further session:** the slope of seconds per decode round over widths 1, 2 and 4, per set; it feeds
  RL7's simulator.
- **Gap report:** the measured long-prompt prefill against the floor (weight bytes as parameters x 2 FLOP per
  token over the best TFLOP/s), with the decode gap from HW3; each gap is tagged with the exactness class of its
  remedy (float16 tensor cores on the T4, U2, are class B: a quality-gated plan, not an exact knob).
- **Validity:** a set that errors is invalid and the others stand; the width part's answers must be identical
  across sets for each width (they are all class A).

### KB1 — A batch-invariant bfloat16 row kernel for up to eight rows on pre-Ampere CUDA (opened 2026-10-29, before data)

**Mechanism.** TB1 found that MLX's bf16 product at 2 to 32 rows runs at 41-45 GB/s against 252 at one row on the
T4, and that its rows differ bit for bit from the row alone. That one path makes true batching both slow and
inexact. `ironmule/cuda_native.py` already solves this for 4-bit weights: one warp per few output rows, up to eight
activation rows, a float32 chain per (row, activation row) with pinned roundings, so a row's answer does not
depend on how many others share the launch. KB1 is the same kernel for an unquantized bf16 weight: it reads each
weight once for every row and is batch-invariant by construction. Designed with a second agent as sparring
partner, which corrected the comparison: the lever is measured against grouped batch-1, which already makes a
width-8 round 1.39x cheaper per token (RL6's fit), not against one request alone.

**KB1-0 (`tools/cuda_probe.py --plan kb1`, Kaggle T4 x2, first card, about 0.3 session hours).** Qwen3 1.7B bf16
revision `9cd6692`. Block 0's seven linear layers and the output head with their real weights, on the real inputs
of one 64-token forward. The kernel at one row is compared bit for bit with MLX's own product on all 64 rows. Each
row at M = 2, 4, 8 is compared with the same kernel at one row. Both are timed alternately at every M, 20 times.

**Kill criteria (fixed before data).**
- **(i)** At one row the kernel is not bit-identical to MLX on some shape. Then the class-A route (stacked rows on
  today's answers) is closed, and KB1 proceeds only inside one plan version together with a fixed-shape attention
  kernel (FA2), so answers change once and pass one quality gate.
- **(ii)** A row at M > 1 differs from the same kernel at one row. That is a bug, fixed before anything else.
- **(iii)** The upper 95 % bound of a step's linear time at eight rows (28 x the block's layers plus the head),
  over MLX's at one row, is at or above 8 / 1.39 = 5.74. Then KB1 has no lever over grouped batch-1, and it stops.

If KB1-0 passes, the model-level test is a stacked step of eight rows (KB1 plus per-row attention) against a
measured width-8 grouped round, both on the product path. A win there changes the width optimum, and RL6's
schedule is re-derived from the new per-row cost curve.

**KB1-0 result (2026-10-29, `KB1-0-run1-20261029`, kernel `tobayko/ironmule-cuda-probe-kb1` at
`4743f947b25ebf0470b706860ec31f67b5750fc6`, Kaggle T4, Qwen3 1.7B bf16 revision `9cd6692`, MLX 0.32.4, 0.03 session
hours): kill (i) fires, (ii) and (iii) pass.**
- **(ii) Batch-invariant.** Every row at M = 2, 4 and 8 was bit-identical to the same kernel at one row, on all
  eight shapes.
- **(iii) The lever holds.** A step's linear time at eight rows was 1.80x MLX's at one row (95 % 1.76-1.83),
  against the 5.74 limit. Per token that makes the linear layers about 4.4x cheaper at width 8, against grouped
  batch-1's 1.39x. At eight rows the kernel beat MLX's own eight-row product 3.7-8x on every shape (output head
  3.03 against 16.2 ms). At one row it was 7-70 % slower than MLX (k_proj 0.144 against 0.084 ms, head 2.61
  against 2.44 ms), so it is not a single-request win.
- **(i) Not bit-identical to MLX at one row.** Rows matched on 35-58 of 64 for the block's layers and 0 of 64 for
  the head (largest difference 0.0005-0.008 in the block, 0.0625 on the head's logits).
- **Verdict, as fixed.** The class-A route is closed. KB1 proceeds only inside one plan version together with
  FA2, so answers change once and pass one quality gate. Next: FA2's kernel, then the model-level test. That test
  runs eight stacked rows at their own offsets (KB1 and FA2) against the same plan at one row, which must be bit
  for bit, and against a measured width-8 grouped round on the product path, which must be faster. A quality
  gate against today's answers decides the plan.

**KB1-1 (fixed 2026-10-29 before its data; `tools/cuda_probe.py --plan kb1m`, Kaggle T4, first card, about 0.3
session hours).** The dense plan (`DensePlan`) is one Qwen 3 decode step for up to eight stacked rows at their own
positions, used at every B. It runs linears through KB1 and attention through FA2. FA2 reads only up to each row's
length, split into 64-key chunks at absolute positions and combined in index order, so it is invariant to batch and
capacity. It uses MLX's own RoPE (one position per row), `rms_norm` and `silu`, and writes the KV cache through a
pure scatter. Prefill stays MLX's own, one request at a time, so the plan touches the decode path only. Eight
distinct prompts, alternately short and behind about 1,000 tokens of context, 64 steps. Worked through with the
second agent, which caught that a full-sequence gate would never run the new code and that FA2 must split the
keys from the start.
- **(a) Exact within the plan:** every row's logits at every step are bit-identical between the plan at one row
  (at the row's own capacity) and stacked at 3 and at 8 rows (a shared capacity), and again at twice that
  capacity.
- **(b) Faster than today's grouping:** the stacked 8-row step against a width-8 grouped batch-1 round of today's
  engine, run as the executor runs one (each member's step at its own capacity, one wait), with the baseline and
  with the compiled step, the faster of the two counting. Five rounds; the ratio's 95 % bootstrap bound must be
  below 1.
- **(c) The numeric plans' quality gate on the decode path:** `ironmule/quality_reference.txt`, 16 chunks of 512
  tokens, teacher-forced through today's decode step and through the plan's (8 chunks stacked), perplexity ratio
  by `numeric_choice.ratio_interval`, passing below 1.005. A long variant scores 128 tokens after a 1,024-token
  prefix (8 chunks), so FA2 is scored far from the start; it must pass too.
- **Recorded, no decision:** (d) the plan at one row against today's best exact decode (compile + readback 4),
  since quiet traffic would regress if it is slower; the step at twice the capacity, which shows whether the KV
  write copies the whole cache; the plan's greedy tokens against today's engine.
- **Verdict:** (a), (b) and (c) together make the plan a candidate for an opt-in numeric plan with a stacked
  executor (a slot pool of fixed capacity, one slot per request). Before it is recommended anywhere, an LD1-V-style
  no-harm check on mixed traffic follows. A failed (a) is a bug, fixed before anything else.

**KB1-1 result (2026-10-29, `KB1-1-run1-20261029`, kernel `tobayko/ironmule-cuda-probe-kb1m` at
`5fde175f6c46fe94f134e265a7f82dcc595eb6db`, Kaggle T4, Qwen3 1.7B bf16 revision `9cd6692`, MLX 0.32.4, 0.2 session
hours): all three decisions pass.** KB1-0 re-ran in the same session at 1.81x.
- **(a) Exact within the plan.** Every row's logits at every step were bit-identical between the plan at one row
  (at the row's own capacity) and stacked at 3 and at 8 rows, and again at twice the capacity.
- **(b) 3.8x faster than today's grouping.** A stacked step of eight took 38.0 ms. Today's width-8 grouped round
  took 144.4 ms with the compiled step and 154.6 ms without. The ratio is 0.263 (95 % 0.262-0.264), which is 4.75
  ms per token at width 8 against 18.1 ms.
- **(c) The quality gate passes on the decode path.** On the 16 x 512-token chunks the perplexity ratio was
  0.9992 (95 % 0.9980-1.0005). Scored after a 1,024-token prefix it was 1.0018 (0.9989-1.0048). Both are below
  1.005.
- **Recorded.** (d) One request alone is slower under the plan: 22.0 ms per token, 1.18x today's best exact decode
  (compile + readback 4), so quiet traffic would regress as built. Twice the capacity changed the step time by
  0.94x, so the KV write does not copy the cache. Over 64 tokens, 6 of 8 prompts gave today's engine's tokens; two
  diverged at steps 23 and 58.
- **Verdict, as fixed.** The dense plan is a candidate for an opt-in numeric plan with a stacked executor (a slot
  pool of fixed capacity, one slot per request). Before it is recommended anywhere, two things are owed: the
  single-request regression (d) is closed or bounded, and an LD1-V-style no-harm check runs on mixed traffic
  through the product path. Next, KB1-2: the plan's single-row speed (rows per warp chosen per M, which leaves
  each row's arithmetic unchanged; readback and pipelining in its decode loop) and its cost per further row at
  widths 1-8, which RL7 needs and which re-derives RL6's width.

**KB1-2 (fixed 2026-10-29 before its data; `tools/cuda_probe.py --plan kb12`, Kaggle T4, first card, about 0.3
session hours).** The plan's single-row speed and its cost per further row, with the decode loop as a server
would run it: the pick stays on the card, the next step is built before the host reads, and the host reads every
4 steps. Three variants:
- **(i)** KB1-1's plan as it ran.
- **(ii)** One output row per warp at every M. That spreads rows over more warps without changing any row's
  arithmetic, so it must be bit-identical to (i) and inherits its gate.
- **(iii)** (ii) with the whole step compiled per shape.

Eight prompts as in KB1-1, 32 steps, widths 1 to 8, five alternating rounds, with today's best exact decode
(compile + readback 4) re-measured in every round.
- **Exact:** a variant is eligible only if every row's logits are bit-identical between one row (at its own
  capacity) and 3 or 8 stacked rows. A variant whose bits differ from (i) must pass the decode-path quality gate
  itself (main and long variants), run in the same session.
- **(d) The single-row regression** is closed when the fastest eligible variant at one row is faster than
  today's tuned decode with a 95 % bound below 1, for the short and for the long-context prompt each against the
  same prompt under today's engine. *Amended 2026-10-29 before run 1's data was read:* run 1 compared the plan's
  short prompt with today's mean over short and long prompts, which favours the plan, so run 1 does not decide
  (d) and run 2 measures it as written here. It is an early warning; LD1-V's no-harm rule on the product cost
  decides. Otherwise it stands at the measured ratio, and the plan cannot
  pass a no-harm check in quiet traffic: a request cannot change plans in mid-life, and choosing the plan by load
  would make answers depend on load.
- **Recorded:** each variant's milliseconds per step and per token at widths 1-8. The cost per further row is
  what RL7's simulator and the re-derivation of RL6's width need.

**Result, KB1-2 run 1 (2026-10-29, kernel `tobayko/ironmule-cuda-probe-kb12` at
`8bbe93a9eb658ce4aa0210d27c459265e185a06d`): descriptive only, as amended above.** All three variants were exact
across widths, and (ii) and (iii) bit-identical to (i). In the server-style loop one row took 16.7-17.5 ms and
eight rows 38.9-40.8 ms.

**Result, KB1-2 run 2 (2026-10-29, `KB1-2-run2-20261029`, kernel `tobayko/ironmule-cuda-probe-kb12` version 2 at
`aea9a2a65169063275f8c64b132490f6dd5e64be`, 0.17 session hours): (d) is closed; every variant is exact.** Same
machine, model and revision as KB1-1 (MLX 0.32.4, mlx-lm 0.31.3). Every variant was bit-identical across one,
three and eight rows, and (ii) and (iii) matched (i) on all 256 compared logit rows, so none needed a gate of its
own. One row against today's tuned decode on the same prompt (18.61 ms short, 21.10 ms long):

| variant | short, ratio [95 %] | long, ratio [95 %] | ms per step at widths 1 / 2 / 4 / 8 |
| --- | --- | --- | --- |
| (i) KB1-1's plan | 0.931 [0.922, 0.937] | 0.988 [0.984, 0.992] | 17.31 / 21.31 / 26.07 / 37.07 |
| (ii) one row per warp | 0.934 [0.921, 0.950] | 0.987 [0.982, 0.991] | 17.28 / 21.30 / 26.59 / 38.54 |
| (iii) (ii) compiled | 0.894 [0.887, 0.901] | 0.964 [0.961, 0.967] | 16.63 / 21.03 / 26.00 / 38.64 |

KB1-1's 18 % single-row loss was its synchronised loop: in the server-style loop (pick on the card, host read
every 4 steps) even (i) beats today's tuned decode at one row. Each further row costs about 3 ms (16.6 to 38.6 ms
from one row to eight), against 17-19 ms per extra session for grouped batch-1 (HW4). The library's backend now
runs (i) with that loop (`DenseBackend.steps_group`, 4 steps per host read); compiling it, (iii)'s further 4 %,
needs a functional pool update first, since `DensePlan.forward` writes its slots in place. It is open as a next
step and is not shipped. Still owed: the ledger entry.

**H8: the dense plan through the product (fixed 2026-10-29 before its data; `tools/rl2_headroom.py --plan h8`,
Kaggle T4 x2, about 2 session hours).** `--engine dense` serves the plan on the tuned engine's worker path. Its
slot-pool executor and bridge are in `ironmule/dense_plan.py`, `ironmule_product/engine_bridge.py` and the worker,
with contract tests on CPU stand-ins. The arms are:
- the product's default on this setup, the tuned engine with RL6's qualified schedule `shortest/w2/g0/sjf`;
- the dense plan at `shortest/w8/g0`;
- the dense plan at `shortest/w8/g0/sjf`.

All three states are weighted equally, with 8 traces on fresh seeds (indices 55-62). Each of four rounds (2 traces
each) starts both engines' HW1 pools in turn, the first one alternating by round, so drift falls on both. Within an
engine, its first arm runs at the start and end of every block (A/A). The cost is LD1-V's.
- **Noise:** the nearest-rank 95th percentile of all A/A pairs, both engines'.
- **(1) The dense plan beats today's default** if its best arm's weighted gain over the default, paired by trace
  with a 2,000-resample bootstrap, has a 2.5 % bound above the noise.
- **(2) No harm:** in each state, quiet included, that arm's gain over the default has a 2.5 % bound above minus
  the noise.
- **Validity:** answers identical across arms and runs within each engine (the dense plan is invariant to its
  group, the tuned engine to its schedule); every dense cell reports `numeric@dense`; at least two complete
  traces. Answers are not compared across engines; the plan changes them by design (recorded).
- **Verdict:** (1) and (2) together let the dense plan be recommended for this setup as an opt-in numeric plan.
  The recommendation names the gain and the answer change. If (2) fails, it stays opt-in for throughput with its
  cost documented, and the single-row speed (KB1-2) comes first.

**Result, H8 run 1 (2026-10-29, kernel `tobayko/ironmule-rl2-headroom-h8` at
`aea9a2a65169063275f8c64b132490f6dd5e64be`, 1.46 session hours): invalid; the dense plan never served.** All 1,152
dense requests failed with `backend_unavailable`, the warm-ups included; all 768 of the tuned engine succeeded. The
cause was in the product path, not the plan: `MLXWorkerClient` sends the dense engine lone streamed batches (it
streams batches, like the tuned engine), but the worker treated only the tuned engine's batches as streamed and
refused the rest as invalid; the client reports a refused batch as `backend_unavailable`. A worker that failed to
start would report the same code, so run 1 does not show whether the plan installed on the cards; run 2's preflight
checks that first. The worker now streams both (`STREAMED_BATCH_CONFIGURATIONS`). The test that
reproduces the refusal runs the worker's batch path on the dense bridge. The harness's fake worker served such
batches anyway, which is why the H8 harness test missed it. The analysis read the run as valid: it charged failed
requests as a cost, and the 14 % "loss" it reported is that cost, not a measurement.

*Amended 2026-10-29 after run 1's data and before run 2's; post hoc for run 1. No decision rule changes. The
second agent reviewed it before run 2.*
- **Validity:** a request that fails, or ends without an outcome, invalidates the run; only timeouts and rejections
  are costs. The analysis reports failures per engine and error code.
- **Stop rules in the harness:** before any pool starts, the plan installs on one card in a process of its own and
  serves three requests together as it serves them alone, or the run stops with the error and the install time.
  A pool that fails a warm-up request stops the run with its workers' health.
- **Equal warm-up and startup:** both engines warm with the same burst of eight mixed-length requests per card,
  and both get the same 600 s startup limit; startup is not in the cost. The plan's load probe now runs every
  width from one row to eight, so every kernel shape compiles before the worker is ready, not inside a timed cell.
- **The comparator, stated:** the default is the tuned engine as a fresh install runs it. Run 2 records each
  card's knobs and their source. Without a stored profile those are the baseline knobs: no compiled step and a host
  read every token, not HW4's class-A compile + readback 4 (8.2 % faster decode), which the default does not use.
  A gain over it is a gain over the product as shipped, not over the best exact engine.
- **Changed implementation:** run 2 measures the plan as it now serves. Its backend chains 4 steps per host read
  (KB1-2's loop; the same forwards, so the same tokens). A stream cancelled inside a chain, or inside a
  speculative draft, still ends at its cancel, in both engines.
- **Known, symmetric, recorded:** the service lets at most eight requests join a group over its whole life, so
  slots the plan frees mid-group are not reused; this understates the plan in spikes. A request longer than a slot
  (prompt plus answer above 2,048 tokens) fails its whole batch under the plan. H8's longest is about 1,300.

Run 2 uses the same arms, traces, rounds, cost, noise and decisions, at about 2 session hours.

**Result, H8 run 2 (2026-10-29, `H8-run2-20261029`, kernel `tobayko/ironmule-rl2-headroom-h8` version 2 at
`ca7353daa0ed6df12303e3cae6cbbfc546fc78d7`, 1.83 session hours): valid. (2) passes, (1) fails as fixed, so the
dense plan is not recommended and stays opt-in.** Kaggle T4 x2, Qwen3 1.7B bf16 at `9cd6692`, MLX 0.32.4,
mlx-lm 0.31.3.
- **Validity:**
  - the preflight installed the plan in 131 s (it would not have started within the old 120 s limit) and served
    three requests together as alone;
  - every warm-up and all 1,920 requests succeeded;
  - answers were identical across arms and runs within each engine;
  - both engines ran on both cards in every round, the tuned engine with the baseline knobs (no stored profile).
- **Noise:** 5.05 % over 48 A/A pairs. Calibration: 0.98 requests per second.
- **(1) fails as fixed.** The best arm, `dense:shortest/w8/g0/sjf`, gained 7.9 % weighted over the default (95 %
  3.5-12.5 %). That resolves it from zero, but not from the 5.05 % noise the entry fixed. Arrival order gained 7.2 %
  (2.2-12.6 %). Shortest first within the plan added 0.7 % (-0.2 to 1.6 %).
- **(2) passes.** Per state:
  - `quiet`: +5.2 % (4.8-5.6 %), as KB1-2's single-row speed predicted;
  - `bursts`: +10.9 % (3.3-19.0 %);
  - `spike`: +5.6 % (-2.6 to 14.5 %).

  Every bound is above minus the noise.
- **Recorded:** 163 of 384 compared answers were the same as today's; the plan changes answers by design. Mean
  cost per state, default against the best dense arm: 2.33 against 2.20, 22.30 against 19.87, 27.62 against
  26.06.
- **What it means.** On this setup the plan was faster than the shipped default in every state, quiet included.
  The gain is too small for this run to tell apart from the A/A noise, and smaller gains need more repeats, not a
  lower bar. The default it beat runs without the class-A compile + readback 4 (KP1), which would narrow the gap.
  The comparison worth a run 3 is against the default as it ships after KP1, with more traces. Still owed: the
  ledger entry.
- **Since run 2 (not measured yet):** a dense group now lets a newcomer take a slot a member frees. The service
  counts the group's running members, not every member over its life, up to 32. A joiner that finds every slot
  taken waits in the worker for the first one freed, instead of running after the whole group. Both apply only to
  the plan, and the other engines are unchanged. Run 3 measures the plan with this. A request longer than a slot
  still fails its batch.
- **Plan version 2 (not measured yet):** the prefill's first token now comes from the plan's own head at one row,
  the same head every decode step uses, which passed the quality gate. Before, MLX's head ran over every prompt
  position. HW4 timed skipping that head at 1,814 to 1,504 ms for a 1,026-token prompt (17 %) and 107 to 98 ms
  for a short one. The tuned engine cannot take this, because it is not class A. The plan changes answers anyway,
  so it is one more change inside its own version (`PLAN_VERSION`, reported with the kernel digest). Each
  request's answer still depends on nothing but itself.

### KP1 — The class-A knob profile as the tuned engine's default on the measured setup (opened 2026-10-29, before data)

**Mechanism.** HW4 found compile + readback 4 class A on the T4: prefill, every decode step and the final state
bit-identical to the baseline, and 0.918 of its decode time (HW3, one card, engine bodies). RL7's result, fixed
before data, says to combine that profile with the best schedule. It is not shipped. A fresh install has no stored
tune profile, so the tuned engine runs the baseline knobs: no compiled step, and a host read every token. The
product-level gain is unknown, because of what the probe did not see:
- with X1 each request decodes at its own capacity;
- the engine keeps eight decode bodies;
- a new compiled body costs 13-22 ms (HW4), and it delays a request's second token, not its first.

**Product mechanism.** `ironmule_product/knob_profile.py`:
- A shipped `knobs.json` with its comparison, digest-checked like the qualified schedule, applies only to exactly
  the model, revision, GPU name, engine and MLX and mlx-lm versions it was measured on. Class A was shown only on
  MLX 0.32.4, and the compiled body traces mlx-lm's model code.
- It may set only knobs that leave every answer unchanged: compile, projection fusion and fused argmax (class A
  in HW4) and the readback interval (host reads only). Everything else must stay at the baseline, and the worker
  refuses anything else.
- A stored, compatible tune profile still comes first. `/health` names the knobs' source.

**Second agent's review, applied before data:**
- the set is exactly what RL7 named, compile + readback 4, without fused argmax (HW3: 1.00);
- logits are checked at many capacities, not only token digests;
- no extra width arm: the step gets cheaper at every width, and RL6 had widths 1 and 2 within 1 %;
- more traces instead, for power;
- compile churn is recorded.

**Test (`tools/rl2_headroom.py --plan kp1`, Kaggle T4 x2, about 2.3 session hours).**
- **Preflight (kill 1):** in a process of its own on one card, the baseline and the KP knobs are compared bit for
  bit, HW4's method: prefill logits, then 24 decode steps fed the baseline's tokens. It runs on a short and a long
  prompt, at 16 capacities from 64 to 8,192 where the prompt fits, plus every capacity this run's traffic produces.
  Any difference stops the run: KP1 is not class A on this stack.
- **Traffic:** two engine configurations on LD1-V's pool, model and revision: today's default (the tuned engine as
  a fresh install runs it, with RL6's qualified schedule `shortest/w2/g0/sjf`) and the same with the KP knobs. All
  three states are weighted equally, with 12 traces on fresh seeds (63-74) in four rounds of three. Each round
  starts both engines' pools in turn, the first one alternating by round. Within a block each engine's arm runs at
  the start and end (A/A for both). Both engines get the same warm-up and startup limit as H8 run 2. The cost is
  LD1-V's.
- **Expected effect (fixed now, for power):** about 7 % in `quiet`, where decode is most of the cost, and 5-10 % in
  `bursts` and `spike`.
- **Kill 2, exactness on the product path:** every request's token ids, finish reason and completion count must
  be identical across all four cells of a trace and state, both engines included. Any difference kills KP1.
- **Validity:**
  - no request fails or ends without an outcome;
  - at least two complete traces;
  - each card's knobs and their source are recorded and match the arm.
- **Noise:** the nearest-rank 95th percentile of all A/A pairs, both engines'.
- **Decisions:**
  - **(1) KP beats the default:** its weighted gain, paired by trace with a 2,000-resample bootstrap, has a 2.5 %
    bound above 0 and above the A/A control. A cell's cost is the mean of its two runs. The A/A control applies the
    same estimator to each engine's first run against its second, and the larger of the two gains in size is the
    floor. This is the "every resolved gain" rule against what the estimator finds between identical runs: the
    answers are identical by kill 2, and the engine order alternates.
  - **(2) No harm:** in each state the bound is above minus the noise.
- **Recorded, no decision:**
  - per-state gains;
  - first-to-second-token time per request;
  - decode bodies built and evicted per card;
  - one stress trace per round with 16 distinct capacities (quiet arrivals, each engine once), since the main
    traffic's few capacities fit the body cache.
- **Verdict:** (1) and (2) with kill 1, kill 2 and validity let `knobs.json` ship. Its domain is this setup, and
  traffic whose active capacities fit the eight cached bodies; the stress trace shows what happens beyond that.
  Otherwise the default stays, and the result goes to Tier 0 with this entry's number.

**Result, KP1 run 1 (2026-10-29, `KP1-run1-20261029`, kernel `tobayko/ironmule-rl2-headroom-kp1` at
`f497264b27d5db38de33871cd61ac749eff7253d`, 2.50 session hours): valid; kills 1 and 2 pass, (1) and (2) fail, so
the knobs do not ship.** Kaggle T4 x2, Qwen3 1.7B bf16 at `9cd6692`, MLX 0.32.4, mlx-lm 0.31.3.
- **Kill 1 passes.** The preflight compared 38 prompt-and-capacity pairs at 27 capacities from 64 to 8,192, the
  run's own included, in 118 s. Every prefill logit, decode step and final state was bit-identical.
- **Kill 2 passes.** All 2,304 requests succeeded, and every answer (token ids, finish reason, count) was
  identical across all four cells of a trace and state, both engines included.
- **Validity:** each card ran its arm's knobs: the baseline (no stored profile) for the default, KP's for KP.
- **Noise** 4.68 % over 72 A/A pairs. The A/A floor was 0.37 % (tuned -0.08 %, KP -0.37 %).
- **(1) fails.** KP's weighted gain was +0.2 % (95 % -1.9 to +1.6 %).
- **(2) fails.** Per state:
  - `quiet`: +3.6 % (3.2-4.1 %);
  - `bursts`: +2.1 % (0.3-4.2 %);
  - `spike`: -1.3 % (-5.4 to +1.5 %), whose bound is below minus the noise.
- **Recorded:**
  - decode bodies built per card: 10-15, 2-7 of them evicted, alike for both engines;
  - the stress cell with 16 capacities: KP 6.5 % cheaper;
  - the time from first to second token: 23.6 ms (median) by default against 0.15 ms with KP.
- **Why, found in the data (post hoc).** In `quiet`, KP finished each answer 115 ms sooner (paired median) but
  showed its first token 56 ms later. The worker held every token until the next one arrived, so the last could
  carry the final text. At readback 4 the next token comes a whole chain later, so the first token waited for the
  chain. The default paid one step on every request too. The worker now sends each token as it is read back.
  Only the token that ends the answer (EOS or the limit, the only ways the engine stops) carries the final text
  and finish reason; tests pin that each token leaves before the engine reads the next. KP1 as measured goes to
  Tier 0. The combination is measured again, with the fix, in H9. Still owed: the ledger entry.

### H9 — The default, KP's knobs and the dense plan on the same traffic, after the first-token fix (opened 2026-10-29, before data)

**Mechanism.** KP1 run 1 and H8 run 2 both measured with the worker holding every token for the next one. That
costs the first token one step on today's default, and a whole 4-step chain under KP's readback and the dense
plan's. Since then:
- the worker sends each token as it is read back;
- a grouped member's `done` follows its last token at once. Before, every member's `done` waited for the whole
  run, so a short request grouped with a long one ended with it. This changes every grouped engine's cost, the
  default's included. It is also what lets the service see a member finish, which the dense plan's slot reuse
  needs: before it, slot reuse never ran through the real service (found by the second agent's review; an
  in-process test now drives the real service and the real worker together);
- the dense plan prefills with its own head on the last position only (version 2);
- the dense plan reuses freed slots within a group.

**Re-opens** KP1 run 1 from Tier 0. Its kill rule (fixed before its data): "(1) and (2) with kill 1, kill 2 and
validity let `knobs.json` ship". What changed is the implementation: the first-token hold the run exposed is gone.

H9 measures all three engines in one run, so every combination meets the same traffic.

**Test (`tools/rl2_headroom.py --plan h9`, Kaggle T4 x2).**
- **Setup:** LD1-V's pool, model, revision, cost and states, weighted equally. 12 fresh traces (75-86) in three
  rounds of four.
- **Engines:** `tuned` (the default: baseline knobs, RL6's qualified schedule `shortest/w2/g0/sjf`), `kp` (the
  same with compile + readback 4) and `dense` (the plan at `shortest/w8/g0/sjf`).
- **Order:** each round starts the three engines' pools in turn, the order rotating by one engine per round. Over
  the three rounds each engine starts once in each position (a cyclic Latin square), so card heat later in a
  round falls on all three alike. Within a block each engine's arm runs at the start and end (A/A for all three).
- **Before any pool:** KP1's bit-for-bit preflight (kill 1), whose failure stops the run, and H8's dense preflight,
  whose failure drops only the dense arms.
- **Warm-up and startup:** the same for all engines, as in H8 run 2 and KP1. The recorded stress cell runs for
  every engine.
- **KP part, KP1's rules unchanged:**
  - kill 2: identical answers across `tuned` and `kp`;
  - (1): a bound above 0 and above the larger A/A control of the two;
  - (2): no state's bound below minus their A/A noise.
  - Pass: `knobs.json` ships for exactly this setup.
- **Dense part, H8's rules:**
  - the comparator is the default as it would ship after the KP part: `kp` if the KP part passes, else `tuned`;
  - (1): the plan's weighted gain over it has a 2.5 % bound above the A/A noise of those two engines;
  - (2): no state's bound is below minus that noise;
  - validity also needs the plan's answers identical across its runs and every dense cell `numeric@dense`.
  - Pass: the plan is recommended for this setup as an opt-in numeric plan; the recommendation names the gain and
    the answer change.
- **Validity, per part:** no request fails or ends without an outcome, and at least two complete traces. A
  dense failure invalidates only the dense part; a failure of `tuned` or `kp` invalidates both.
- **Recorded:**
  - first-token and first-to-second-token times per engine;
  - decode bodies and stress costs;
  - the plan's answers that match today's;
  - the plan against both `tuned` and `kp`, so the switch of reference can be seen not to decide the result.
- **Two notes, fixed now:** the reference switches to `kp` partly because KP measured well on this same data,
  which is conservative against the plan. The two parts are two one-sided decisions at 2.5 % each on one run.
- **Cost:** about 4.2 session hours, about 10 minutes of it the plan's installs at each dense pool start.
- **Found while it ran, before any of its data was read (2026-10-29, 22:20):** the harness sets each arm's
  serving policy without `join`, and `join` defaults to off (SERVE8 killed joining for grouped batch-1). So in
  H9 the dense plan's freed slots were never offered to a newcomer. H9 measures the plan with version 2 and each
  member ending at its own last token, not with slot reuse. The service now lets a slot-reusing backend take
  newcomers whatever `join` says; the dense plan's next comparison measures that.
- **And a race that joining exposed (fixed before H9's data was read):** a join the client sends while the
  worker is already ending the run reaches the worker after the run. The worker named only the others in
  `batch_done`, the client treated that as a broken batch and retired the worker, and the stale join then failed
  the worker's next batch. It was latent since SERVE8, where `join` is off by default; with slot reuse always on
  it struck. Now the worker drops a join for an ended batch, the client hands back a joiner that streamed
  nothing, and the service puts it at the front of the queue for the next group. Tests cover each side; the
  harness's fake worker now speaks the current protocol (tokens as produced, `done` with the last, joins).

**Result, H9 run 1 (2026-10-29/30, `H9-run1-20261029`, kernel `tobayko/ironmule-rl2-headroom-h9` at
`843a865374a9da83e7f8d88ab8ecefd46304b718`, 3.74 session hours): valid. The dense plan passes both decisions and is
recommended for this setup as an opt-in numeric plan; KP's knobs fail (1) narrowly and do not ship.** Kaggle T4
x2, Qwen3 1.7B bf16 at `9cd6692`, MLX 0.32.4, mlx-lm 0.31.3.
- **Validity:**
  - both preflights passed: KP bit-for-bit at 27 capacities (38 pairs); the plan installed in 116 s and served
    three requests together as alone;
  - all 3,456 requests succeeded, and every warm-up;
  - each engine ran its knobs on both cards in every round (baseline; `class_a:KP1-candidate`; the dense plan);
  - answers were identical across `tuned` and `kp` (kill 2) and within the plan.
- **KP part** (noise 5.33 %, A/A floor 0.81 %):
  - (1) fails: the weighted gain was +1.7 % (95 % 0.39-3.15 %), whose bound does not clear the floor;
  - (2) passes: `quiet` +6.1 % (5.7-6.7 %), `bursts` +0.8 % (-1.1 to +2.4 %), `spike` +1.9 % (0.4-3.6 %);
  - recorded: the stress cell 8.7 % cheaper; median first-to-second token 94 ms for KP against 27 ms by default,
    since KP's second token comes with its first 4-step chain.

  Compared with KP1 run 1, the first-token fix turned spike's -1.3 % into +1.9 % and lifted `quiet` from 3.6 %
  to 6.1 %. The weighted gain still sits within the A/A control's reach.
- **Dense part, against `tuned` (the KP part failed; noise 4.52 %):**
  - (1) passes: the plan's weighted cost was 24.0 % lower (95 % 20.8-26.7 %);
  - (2) passes: `quiet` 10.0 % (9.4-10.7 %), `bursts` 22.6 % (15.9-27.3 %), `spike` 26.1 % (22.0-30.5 %);
  - against `kp`, descriptive: 22.7 % (19.9-25.2 %).
  - Mean cost per state, default against the plan: 2.30 against 2.07, 16.73 against 12.95, 26.59 against 19.65.
  - 271 of 576 compared answers were the same as today's; the plan changes answers by design and passed the
    numeric plans' quality gate (KB1-1).
- **What it measured:** the plan at version 2, with every member ending at its own last token. Slot reuse did not
  engage (see the note above). Compared with H8 run 2's 7.9 %, the first-token fix, the per-member end and
  version 2 together tripled the plan's lead.
- **Verdict, as fixed:** the dense plan is recommended on exactly this setup (two Tesla T4s, Qwen3 1.7B bf16 at
  `9cd66928`), opt-in, with the gain and the answer change named. The product has since changed (slot reuse
  always on), so that configuration is checked in H10 before the recommendation covers it. KP's knobs go to
  Tier 0 again. Still owed: the ledger entry.


### RL8 — Width and order again, now that a grouped request ends with its own last token (opened 2026-10-29)

**Mechanism.** RL6 and LD1-V chose `shortest/w2/g0/sjf` when every member's completion waited for the group's
longest member. RL4's pool model says so too ("every member's last token and completion are released when the
whole group ends"). Since H9's change, each member ends at its own last token. A wider group then no longer costs a
short request its neighbours' length, only the per-session step cost (17-19 ms per extra session on the T4, HW4).
So the best width may rise, and shortest-first may matter less. The same holds under the dense plan, where a
further row costs about 3 ms (KB1-2).

**Evidence against.** Wider grouped batch-1 rounds still slow every member's steps. RL6 found widths 1 and 2
within 1 %. H9 measures only width 2 (default, KP) and width 8 (dense).

**Test, fixed before its data, after H9:**
1. Refit RL4's pool model with per-member release on H9's data (the release rule is a model change, gated by
   RL4's fidelity rule).
2. Rank widths 1, 2, 4 and 8, with arrival order and shortest-first, for the default as it ships after H9, and
   widths up to 8 for the dense plan.
3. Run on hardware only the arms the refitted model cannot separate from the shipped schedule, with LD1-V's cost
   and rules, as RL6 did.

**Kill:** the shipped schedule stays if no arm beats it with a 2.5 % bound above the noise and no harm in any
state.

**Preview (2026-10-29, before H9's data; descriptive, not the decision).**
- **Tools.** The pool model gained a release rule (`release`, "group" or "member") and fits runs through the
  product (`h6`, `h7`, `kp1`, `h9`, by engine), each under the rule the product had then. RL6's run numbered its
  two states 0 and 1; the model now maps them to `bursts` and `spike`. An unmapped first try fitted garbage, at
  65-227 % error.
- **Fit.** On RL6, LD1-V and KP1 (the tuned engine, group release) the fit had a squared log error of 0.002:
  prefill 0.151 s + 2.2 ms per prompt token, a step 18.5 ms + 16.2 ms per further session (HW4 measured 17-19
  ms). It passes the fidelity gate on all three runs: every cell within 7.6 %, every resolved pair ordered.
- **With member release simulated** (200 held-out traces per state), `shortest/w2/g0/sjf` stays the best fixed
  arm: 15.44 weighted, against 15.57 for w1/sjf, 15.69 for w4/sjf and 16.34 for w8/sjf. Wider groups gain the most
  (w8 moves from 17 % to 5.8 % behind), but none overtakes w2. A per-state oracle over these arms would gain
  0.14 %.
- **The RL agent** (placement, order and width, 6,000 episodes, 3 seeds):
  - unconstrained, it ends 0.1-1.0 % behind the best fixed arm;
  - with a harm weight of 5 against each state's best fixed arm (`guard_arms`: w2/fifo, w1/sjf, w4/sjf), it ends
    0.04-0.37 % ahead and loses at most 0.9 % in any state. The constraint removes the per-state loss that
    stopped LD1-V's policy.
  - Its headroom is far below the 3-8 % hardware noise, so by RL4's simulator gate no learner goes to hardware
    for the tuned engine's schedule.
- **What is left for RL8:** the refit on H9's member-release data, and the dense plan, whose further row costs
  about 3 ms against 16 ms. That needs its own step model (stacked rounds of up to eight rows, slots refilled
  as members end), fitted to H9's dense cells.

**Result, steps 1 and 2 (2026-10-30, on H9's data; local, no GPU).**
- **The tuned engine.** Refitted on RL6, LD1-V, KP1 and H9 (each under its release rule), the model passes the
  fidelity gate on all four (squared log error 0.0021). With member release, `shortest/w2/g0/sjf` stays best
  (15.51 weighted, against 15.74 for w4/sjf and w1/sjf, and 16.35 for w8/sjf). The shipped schedule stays: no
  hardware run for the tuned engine.
- **The dense plan.** A model with eight stacked rows per round, fitted to H9's dense cells (error 0.0005),
  passes the fidelity gate. Prefill is 0.123 s + 1.6 ms per token (version 2, cheaper than the tuned engine's
  2.2), and a round 18.5 ms + 3.6 ms per further row (KB1-2: about 3 ms).
  - Without slot reuse, as measured: w8/sjf is best, at 11.30.
  - With slot reuse (the product now): w4/sjf at 8.88 and w8/sjf at 8.98. Slot reuse is predicted to take
    another 21 % off the plan's cost.
- **The RL agent on the dense plan** (harm-guarded, 3 seeds): -1.0 to +0.8 % against the best fixed arm, well
  inside the noise. No learner; the fixed rule is the answer here too.
- **Next: H10** checks the plan as the product now serves it (slot reuse), at w8/sjf and w4/sjf.

### H10 — The dense plan as the product now serves it, with slot reuse (opened 2026-10-30, before data)

**Mechanism.** H9 recommended the dense plan, measured without slot reuse. The product now always lets a
newcomer take a slot a finished member freed, and a join that arrives as a run ends goes to the next group
instead of failing it. RL8's dense model, fitted to H9 and passing its fidelity gate, predicts another 21 %
off the plan's cost from that: w8/sjf 8.98 and w4/sjf 8.88, against 11.30 without joins. The recommendation
covers the product only after this configuration is checked on hardware.

**Test (`tools/rl2_headroom.py --plan h10`, Kaggle T4 x2, about 3 session hours).**
- **Setup:** LD1-V's pool, model, revision, cost and states, weighted equally; 12 fresh traces (87-98) in four
  rounds of three. Each round starts both engines' pools in turn, the first one alternating.
- **Arms:** `tuned:shortest/w2/g0/sjf` (the default, A/A per block), `dense:shortest/w8/g0/sjf` (H9's arm, A/A
  per block) and `dense:shortest/w4/g0/sjf` (RL8's pick with joins). Equal warm-up and startup as in H9; the plan's
  preflight before any pool.
- **Validity:**
  - no request fails or ends without an outcome;
  - answers identical within each engine;
  - every dense cell `numeric@dense`;
  - at least two complete traces;
  - slot reuse engaged: the dense cells of `bursts` and `spike` together report joined requests (`/health`'s
    `joined_requests`, counted per cell). Without that, the run measures the wrong thing and is invalid.
- **Decisions (H8's rules):**
  - (1) the better dense arm's weighted gain over the default has a 2.5 % bound above the noise;
  - (2) no state's bound is below minus the noise.
  - Pass: the recommendation covers the plan with slot reuse.
- **The plan's schedule (RL8's kill):** w4/sjf replaces w8/sjf only if its paired gain over w8/sjf has a 2.5 %
  bound above the noise; otherwise w8/sjf stays the recommended schedule.
- **Recorded:** joined requests per state; cost per state; first-token and end times; the plan's gain over its
  H9 level, read against the default it shares a run with.

**Result, H10 run 1 (2026-10-30, `H10-run1-20261030`, kernel `tobayko/ironmule-rl2-headroom-h10` at
`2ffcbb8533a65124d17a9c7ed05ab97957d2dbf6`, 2.40 session hours): valid. Both decisions pass, so the recommendation
covers the plan with slot reuse; w8/sjf stays its schedule.** Kaggle T4 x2, Qwen3 1.7B bf16 at `9cd6692`, MLX
0.32.4, mlx-lm 0.31.3.
- **Validity:**
  - the plan's preflight passed (installed in 117 s);
  - all 2,880 requests and every warm-up succeeded;
  - answers were identical within each engine, and every dense cell ran `numeric@dense`;
  - slot reuse engaged: 839 requests joined a running dense group (32 in `quiet`, 363 in `bursts`, 444 in
    `spike`), and none joined a tuned group (join stays off there).
- **Noise:** 8.69 % over 72 A/A pairs.
- **(1) passes.** `dense:shortest/w8/g0/sjf` cost 48.3 % less than the default, weighted (95 % 45.0-51.4 %).
- **(2) passes.** Per state:
  - `quiet` 13.6 % (11.2-15.8 %);
  - `bursts` 53.0 % (48.8-56.9 %);
  - `spike` 48.0 % (42.1-53.3 %).
- **The schedule:** w4/sjf cost 8.3 % more than w8/sjf (95 % 4.7-12.3 %), so w8/sjf stays. RL8's dense model,
  fitted on H9's runs without joins, had put w4 1.1 % ahead: it extrapolated the joining mode and got the order
  wrong. The hardware decides.
- **Refit on H9 and H10 (each run simulated with the join mode it had), recorded:** the dense model passes the
  fidelity gate on H9, and on H10 its costs are within 10 %, but in `spike` it still orders w4 ahead of w8,
  where the hardware measured w8 12 % cheaper. So it fails on H10, and it does not decide the plan's width;
  w8/sjf, as measured, does. What the joining model misses is not known yet.
- **Recorded:**
  - mean cost per state, default against w8/sjf: 2.44 against 2.11, 20.26 against 9.53, 25.30 against 13.16;
  - 258 of 576 compared answers were the same as today's.
- **Verdict, as fixed:** on exactly two Tesla T4s with Qwen3 1.7B bf16 at `9cd66928`, `--engine dense` (version
  2, slot reuse, its default schedule `shortest/w8/g0/sjf`) is recommended as an opt-in numeric plan. It costs
  about half of the shipped default on mixed traffic, and about half its answers differ from `--engine tuned`.
  Still owed: the ledger entry.

### H11 — Dense slots that hold the whole context (opened 2026-10-30, before data)

**Mechanism.** The dense plan's slots held 2,048 tokens. A longer request failed its whole group, and the client
then retired the worker. Since commit `562ccd2` such a request is rejected alone with a `400`, but `--engine
dense` still refuses everything between 2,048 and the product's 8,192-token context limit, which `--engine tuned`
serves.

The plan can take the largest slot in (8,192, 4,096, 2,048) whose pool fits beside the weights and a 3 GiB
prefill reserve in 90 % of the card (`H11_TIERS`); until (a) passes, `install` keeps 2,048 (`SLOT_TIERS`). The reserve covers a 2,048-position chunk's attention at 8,192 keys and the
prompt's own cache before it is copied into its slot. That is 8,192 on a T4 with Qwen3 1.7B bf16: 3.44 GB of
weights and a 7.5 GB pool. It also prefills a prompt longer than 2,048 positions in chunks of 2,048, as mlx-lm
does (plan version 3). Prompts up to 2,048 are prefilled exactly as in version 2.

By construction the slot size moves no answer: FA2's combine reads a row's chunks only up to its position. What
the larger slot can cost is FA2's partial kernel. It launches warps for every 64-key chunk of the slot, and the
warps past a row's position exit at once: four times as many at 8,192 as at 2,048.

**Amended before data (2026-10-30), after a review of the protocol as first written:**
- (a1) now also checks a row far past 2,048 beside short ones.
- (a2)'s noise is now measured on the gain's own statistic, with a verdict per width as well.
- (a3)'s bar now counts the worker's cache and the CUDA context.
- (a4), quality at long context, is new.
- In (b), a long request that times out is now a cost, not an invalid run.

**Test, part (a): `tools/cuda_probe.py --plan h11`, one Kaggle T4, about 0.5 session hours.**
- **Setup:**
  - The reference's long-context likelihoods (a4) are taken first, with nothing else held.
  - The product's plan is then installed twice through `install` (load probe included), with 2,048- and
    8,192-token slots. The same eight requests (H10's mix: short questions and the same questions after about
    1,000 tokens of notes) are prefilled into both.
  - MLX's cache limit is the worker's (1 GiB).
- **(a1) Answers, kill.** Everything below must be bit-identical:
  - between the two slot sizes, every row's tokens over 32 steps at widths 1 and 8, and every row's logits at
    every step of a width-8 decode fed the same tokens;
  - in the 8,192 pool, one 8,000-token row decoded beside seven short rows (tokens and every step's logits)
    against the seven decoded without it and the long row decoded alone.

  If not, the slot size moves answers against the construction, and the plan keeps 2,048.
- **(a2) Cost of the larger slot.**
  - Measurement: 32 steps per width (1, 2, 4, 8), decoded as the product's executor does (chains of four, one
    host read each), in 10 ABBA rounds. Each size runs twice per round; those two runs are its A/A pair.
  - Statistic: per round, the 8,192 slot's time over the 2,048 slot's, averaged over widths; gain = 1 minus
    their mean, with a 2,000-resample bootstrap over rounds.
  - Noise: the nearest-rank 95th percentile, over rounds and sizes, of the same statistic's A/A value (the
    second run over the first, averaged over widths).
  - Per width: the same, each width on its own with its own noise.
  - Verdict:
    - `harm` if the pooled gain's or any width's 97.5 % bound is below minus its noise;
    - `gain` if the pooled 2.5 % bound is above the noise and no width shows harm;
    - otherwise no resolved difference.
  - On `harm`: before (b) runs, FA2's grid is cut to the longest live row's chunks, so a short-only step again
    launches exactly H10's kernel, and (a) is repeated.
- **(a3) Memory, kill.** Only the 8,192 pool is held. An 8,000-token prompt is prefilled while seven short rows
  hold their slots, then decoded with them. MLX's peak active memory, plus the 1 GiB cache limit, plus the CUDA
  context, must stay within 97 % of the card. The context is `nvidia-smi`'s memory in use minus MLX's active and
  cached memory, measured idle with the pool held. If not, the rule is wrong for the T4 and its tier drops to
  4,096.
- **(a4) Quality at long context, kill.**
  - Material: the numeric plans' quality text (11,756 Qwen 3 tokens), teacher-forced on the decode path.
    - Group `3k`: eight windows of 3,000 prefix tokens and 256 scored.
    - Group `6k`: four windows of 6,000 prefix tokens and 128 scored.
    - The scored spans do not overlap.
  - The plan, prefilled in chunks into its 8,192 slots and decoded stacked, against the engine's baseline.
  - Pass: in each group, the bootstrap ratio of perplexities passes the numeric plans' bound (97.5 % below 1.005,
    `ratio_interval`).
  - If not, the plan keeps 2,048-token slots: long requests are refused, not served by a plan whose quality is
    not shown there.
- **Ship rule:** 8,192-token slots stay only if (a1), (a3) and (a4) pass and (a2) shows no harm.
- **Recorded:**
  - prefill time of 2,000-, 4,000- and 8,000-token prompts;
  - the step time of eight rows with one 8,000-token row against eight short rows;
  - `nvidia-smi`'s memory in use;
  - the tier `slot_capacity` picks on the card.

**Test, part (b): `tools/rl2_headroom.py --plan h11`, Kaggle T4 x2, about 1.8 session hours. It runs only after
(a) passes, at the commit that then admits the 8,192 tier (`SLOT_TIERS = H11_TIERS`) and changes nothing else.**
- **Setup:** H10's pool, model, revision, cost and states, weighted equally; 8 fresh traces (99-106) in two
  rounds of four. Each round starts both engines' pools in turn, the first one alternating.
- **Traffic:** H10's traffic per state, where in each trace one request in eight (at least one) sits behind the
  background notes repeated 2, 4 or 7 times: about 2,050, 4,060 or 7,090 prompt tokens, chosen from the trace's
  seed. The output limits are H10's, so every request fits the context limit. Calibration still uses short
  traffic, so the states carry more work than in H10; the offered work per state is recorded.
- **Warm-up:** as in H10, plus one 7,090-token request per card, so that neither engine pays its first long
  prefill in a timed cell.
- **Arms:** `tuned:shortest/w2/g0/sjf` (the default) and `dense:shortest/w8/g0/sjf` (the plan as recommended),
  each with an A/A pair per block.
- **Validity:**
  - H8's conditions: no request fails or ends without an outcome; answers identical within each engine; every
    dense cell `numeric@dense`; at least two complete traces;
  - the plan's preflight passed;
  - every dense worker reports 8,192-token slots;
  - both engines served long requests, and none of those ended without a costed outcome. A timeout counts as a
    cost; a rejection for length means the fit rule is broken and invalidates the run.
- **Decisions (H8's rules):**
  - (1) the dense arm's weighted gain over the default has a 2.5 % bound above the noise;
  - (2) no state's bound is below minus the noise.
  - Pass: the recommendation covers prompts plus answers up to the 8,192-token context limit. Otherwise it stays
    bounded to 2,048 tokens, and longer requests are served but not recommended.
- **Recorded:**
  - cost per state;
  - per engine and state, short and long requests apart: mean cost, mean time to first content, timeouts;
  - the long requests' outcomes;
  - joined requests;
  - answers the same as `tuned`.
- **Ledger:** owed for both parts once they run.
- **Status (2026-10-30):** (a)'s kernel (`tobayko/ironmule-cuda-probe-h11`, commit `35d082e` for the tools) was
  refused by Kaggle: the weekly 30-hour GPU quota is used up. It runs when the quota resets, at the commit that
  carries this status.

### OV1 — Build the next decode step before reading the token (opened 2026-10-29)

**Mechanism.** HW3: on mlx-lm's loop, pipelining saved 12.5-13.5 % of a step with the same tokens. The engine
already reads the host every 4 tokens (`readback_every`) and compiles the step, which together reached 0.918. So
what pipelining can still hide is likely about a millisecond. It is class A by construction: same graph, only
`mx.async_eval` scheduling. **Test:** `compiled_fixed_cache` x `readback_every` {1, 4} x pipelined {off, on}
through the engine, HW3's round design. **Kill:** no resolved gain on top of compile + readback 4.

### FA2 — A fixed-shape decode attention that reads only up to the offset (opened 2026-10-29)

**Mechanism.** HW3: attention over keys cut to the context saves about 15 % of a long-prompt step against
`FixedKVCache`'s masked attention, and its logits no longer depend on the KV capacity (X1). Slicing the keys in
the engine would give a new shape every step and recompile the step, so FA2 is an `mx.fast.cuda_kernel` over the
fixed-capacity cache that loops to the offset read from an array, with a float32 online softmax in one fixed
order. That makes it capacity- and batch-invariant by construction, but it changes answers once (class B): it
ships only inside a plan version with a quality gate, together with KB1 if KB1's kill (i) fires. Budget for it
being slower than MLX's own fused kernel. **Kill:** not faster than the masked attention at 64 and 1,024 tokens,
or the plan's quality gate fails.

**HW4 result (2026-10-29, `HW4-run1-20261029`, kernel `tobayko/ironmule-cuda-probe-hw4` at
`6194401961f7dd93560bf430d92e6776749cdecc`, decode winner `compiled_fixed_cache` + `fused_argmax` + `readback_every`
4 by HW3's rule, Kaggle T4, Qwen3 1.7B bf16 revision `9cd6692`, MLX 0.32.4, 0.8 session hours): valid.** Every part
ran without an error.
- **Class A:** 7 knob sets are bit-identical to the baseline in prefill logits and state, every decode step and
  the final state: `compiled_fixed_cache`, `fused_argmax`, `fuse_projections` and every combination of the three.
  `readback_every` moves only host reads, so HW3's best combination, compile + readback 4 (0.918), is class A:
  8.2 % faster decode with today's answers bit for bit.
- **Not class A:** `head_skip_prefill` changes the prefill's last logits (the head at one row against the head
  over the whole prompt). Prompt-lookup speculation's k + 1 verify windows differ from the sequential steps.
- **Separability:** `head_skip_prefill` and the decode winner couple slightly (interaction 0.998, 95 % bound below
  1, at n = 16, 64 and 256). Fusion and the decode winner are within 0.06 % of separable. Because
  `head_skip_prefill` is not class A, no exact prefill knob qualifies.
- **`prefill_into_fixed`** kept its tokens but made prefill 7.8 % slower (1.078). `E1`'s kill holds on CUDA.
- **Cost per further session** on the grouped path: 17.2-18.9 ms per set against 19.6-21.6 ms for one session.
  Grouped batch-1 saves little per extra member on this card (RL6's fit had 13.4 ms; this run's prompts were short).
  No pair of sets changed order across widths 1, 2 and 4, and every width's answers were identical across sets.
- **Compile cost:** a new compiled body costs 13-22 ms at a new capacity.
- **Gap report:** a long prompt's prefill took 1,815 ms against a 1,213 ms floor (2.9 TFLOP/s, the best measured
  linear at 1,024 rows), a 33 % gap. Decode reaches 70 % of its bandwidth floor (HW3). Both remedies are class B:
  float16 tensor cores for prefill (U2) and the dense plan for decode (KB1-1).

### RL7 — An RL agent over scheduling and execution knobs together (opened 2026-10-29, before data)

**Mechanism.** The product decides scheduling (card, order, width; RL6) and execution knobs (prefill set, decode
set), and they interact. Speculation and readback act only on the sequential path, so "width 1 with speculation"
against "width 2 without" is a real trade-off. A knob set that is cheaper alone but dearer per further session
moves the best width. A long prompt's prefill under shortest-first delays short requests. That joint space
cannot be enumerated on hardware. **The objective stays the product's request cost** (time in system plus time
to first content). Every phase's improvement is credited and reported per phase (prefill, decode, queueing).
Clipping each phase at zero is rejected, because it would reward arms that are slower overall.

**Plan, in `tools/rl4_agent.py`.**
1. Give the pool model parameters per knob set from HW3 and HW4: prefill ms per prompt length per prefill set,
   decode ms per round and per further session per decode set, compile seconds per new body. Speculation and
   readback apply only at width 1.
2. Enumerate the joint fixed arms in the simulator (class-A knob sets x widths 1, 2, 4, 8 x fifo, sjf) and
   compute the per-state oracle against the best single joint arm.
3. Only then add a knob head to the REINFORCE policy (as `--widths` added the width head) and train.

**Knob sets** may be chosen per request or by load only if they are class A (HW4); otherwise a request's answer
would depend on traffic, the defect X1 removed.

**Kill criteria (fixed before data).**
1. **Class A:** fewer than two class-A knob sets besides the baseline. The knob axis is empty and RL7 stops;
   scheduling stays RL6's.
2. **Dominance:** no pair of class-A sets resolves in opposite directions across short and long prompts or widths
   1, 2 and 4. The knobs then separate from scheduling: refit the pool model, re-rank the fixed schedules, and
   build no learner.
3. **Fidelity:** every simulated cell within 25 % of hardware, and at least 90 % concordance on at least 10 pairs
   the hardware resolved.
4. **Simulator headroom:** the held-out per-state joint oracle beats the best single joint arm with a 2.5 % bound
   above the hardware noise of the target states (3.0 % for bursts and spikes, RL6; 8.1 % mixed, LD1-V). The
   trained learner must then beat the per-state table by the same margin, or the table ships instead.
5. **Hardware:** LD1-V's rules unchanged. The pooled gain's 2.5 % bound exceeds the noise, and each state's bound
   exceeds minus the noise.

**Result (2026-10-29, from HW4): kill 1 passes and kill 2 fires, so no knob learner.** Seven knob sets are class A
(kill 1 passes). No pair of them changed order across widths 1, 2 and 4, so the knobs separate from scheduling
(kill 2). As fixed, the knob axis needs no learner: the best class-A profile (compile + readback 4, 0.918) is
combined with the best schedule, and the pool model is refitted per set. RL7's question still applies where the
cost curve does change shape: under the dense plan a further row costs about 2.3 ms against 17-19 ms on today's
grouped path. Re-deriving width and order for it is the dense plan's hardware comparison (KB1, product path), not a
knob learner.

### L2 — A LoRA-tuned small language model as the planner's prior

**Mechanism.** A small LM fine-tuned with LoRA (Hu et al., ICLR 2022) on stored tune traces could propose a better
first test for an unseen model than the planner's current prior. **Evidence against.** Cycles 14 and 15: Gemma 3 1B
and 4B as planners failed the output contract every time (Markdown, wrong keys). LLAMBO (Liu et al., ICLR 2024,
arXiv:2402.03921) found an LLM helps Bayesian optimisation mainly as an in-context warm start while observations
are sparse, without fine-tuning; and there are tens of tune traces per device, not the hundreds a fine-tune needs.
**Test.** Leave-one-model-out on the stored RSI1/RSI1b traces: tests until the confirmed best, current prior against
an LM warm start (in-context first, LoRA only if in-context helps). **Kill.** No fewer tests at equal confirmed
gain, or any malformed proposal reaching hardware.

## Read this before optimising anything

### RSI2 — Shorten the paired confirmation where the evidence is already clear (2026-10-05)

**Mechanism.** RSI1b: the planner cut screening to 3-6 tests, but Gemma 3 4B's tune still took 30
minutes, about 25 of them in the six-process paired confirmation. A sequential confirmation (stop
early when the interval already excludes 1 by a margin) could keep the same error rate with fewer
processes. **Test.** Kaggle T4: confirm the RSI1b winners with six processes and with the sequential
rule; compare decisions and time. **Kill.** Any changed decision, or under 30 % time saved from 1B up.

### CUDA1 — A deep knob combination crashes MLX's CUDA graph (2026-10-05)

**Mechanism.** RSI1, random order, Gemma 3 270M on a T4: one depth-5 configuration aborted the process
with `cudaGraphAddDependencies ... invalid argument`, so that tune failed. **Test.** Reproduce it
alone, then with `MLX_USE_CUDA_GRAPHS=0`. **Kill.** Not reproducible twice: record as transient.

### CPU5 — A visible answer sooner on a CPU (2026-10-05)

**Mechanism.** Qwen3 0.6B spends its first tokens in `<think>`; at under 1 token per second a readable
reply takes minutes. Turn thinking off for `start`'s CPU default, or pick a small non-thinking model.
**Test.** Kaggle CPU: time to the first visible token, paired against today. **Kill.** No reduction,
or worse answers on simple checks.

### MOLE1 — Remaining agent-level MCP comparison (2026-09-12)

**Mechanism.** Provenance-bound same-task plans may amortize model orchestration and
validation across repeated `repository_context_bundle.v1` requests. The local MVP is
complete in `ironmole_mcp/`; results moved to `research/LEDGER.md` MOLE1 and
`ironmole_mcp/docs/RESULTS.md`. IronMule inference remains unchanged.

**Remaining test.** Supply real direct-agent and actual Programmatic Tool Calling
adapters, record final responses, actual usage/pricing and handoff/continuation costs,
and evaluate whole held-out repositories against the strongest stored workflow.
Local synthetic MCP timings cannot replace these measurements.

**Kill.** Any contract/permission violation or no amortized advantage inside the
preregistered quality, cost and latency limits leaves adaptation unqualified. Missing
agent evidence leaves the full hypothesis unanswered. LOCAL-1/LOCAL-2 already found
no adaptive benefit in their local regime; do not repeat those unchanged experiments.

**Making the runtime faster usually makes the headline number smaller.** The shipped
gain is `+11.81%` to `+19.24%`, and it is a *ratio*: recovered overhead over total step
time. Grouped batch-1 works by overlapping device execution with host submission
(`E14b`). Anything that makes host submission cheaper — `B8`, `B9`, `B10` — removes
exactly the thing the gain is measured against. The product gets faster and the
percentage falls.

That is not a reason to avoid those entries. It is a reason to state, before running
them, which number is being optimised: **absolute tokens per second**, or the
**grouping gain over an unchanged baseline**. They are not the same target and a fix
can move them in opposite directions. Publish both or the report is misleading.

## How to read an entry

Each has a **mechanism** (why it could work), the **evidence** for and against it from
the ledger, a **test**, and a **kill** — what result closes the entry for good. An
entry with no kill criterion is not a hypothesis, it is a wish.

Effort is calendar-honest for one person on one machine. Payoff is a guess and is
labelled as one.

## Release blockers imported from the 2026-08-27 runtime review

These entries are code-review hypotheses until a failing regression test or current
source proof confirms them. They are deliberately separate from the speed tiers: the
goal is a trustworthy `0.1.1` runtime and benchmark, not a larger headline gain.

### `R2` — Make request arrival, prefill and decode one measurable lifecycle

**Mechanism.** `engine_start_ns` now begins before prefill and remains nonnegative. The
remaining defect is that all prefills still run before request arrival/admission, and
the lifecycle lacks separate phase timestamps for admission, prefill start/finish and
decode. Delayed arrivals can therefore receive model work before they exist in the
simulated service timeline.

**Test.** Inject a deterministic clock/backend and assert the ordered timestamps
`request_received <= queue_entered <= engine_start <= prefill_start <= prefill_finished
<= first_token <= finished`; assert no prefill starts before `arrival_ms`.

**Kill.** The service owns admission and prefill scheduling, all durations derive from
those timestamps, and sequential/grouped results remain token-identical. This changes
the runtime architecture and requires the project's explicit architecture approval
before implementation.

### `R3` — Make the public benchmark balanced and end-to-end

**Mechanism.** The public benchmark now uses complete service `outer_wall_ms`, balanced
AB/BA ordering, and independent plans. Remaining gaps are one shared loaded
process/model, no stock `mlx_lm` arm, and prefill/decode phase diagnostics that are
planned but not yet present in snapshots.

**Test.** Fresh cache/plan instances per arm, at least two warmups, alternating AB/BA
orders in fresh processes, raw samples plus median/spread/interval, and a deliberate
token mismatch that must produce a non-zero exit code and a structured diff.

**Kill.** Primary throughput uses complete service wall time; executor/prefill/decode/
queue times remain diagnostic; the protocol is order-balanced and fails closed on
wrong answers. A stock `mlx_lm` arm is added only after its exact prompt/stop contract
is defined and architecture approval is recorded.

### `R8` — Turn correctness and packaging into automated release gates

**Mechanism.** The macOS workflow now runs remotely and its clean installed-wheel job
is green. The real-model fixture no longer skips every exception. What is left is the
gap this entry keeps confusing with CI: nothing asserted that a *first* run works. The
first two commands a new clone runs, `ironmule models` and `ironmule benchmark`, both
raised a raw `CacheNotFound` traceback on a machine with no Hugging Face cache — the
exact machine every new user is on, and the exact machine CI is.

**Test.** CI builds and installs the wheel in a clean environment, runs unit and CLI
smokes, and checks dependency metadata. Integration setup skips only enumerated model,
access or unavailable-Metal failures; all other exceptions fail. Subprocess tests assert
that a cache-less machine gets an actionable message and a non-zero exit, never a
traceback, and that `--help`/`doctor` still start when the MLX import itself is broken.

**Kill.** Remote clean package/CLI job green (done), first-run behaviour covered by
tests (done), and synthetic regressions covering `R1`–`R7` (done, checked 2026-09-26:
`R1` in `tests/engine/test_ironmule_runtime.py` — prefill EOS counted, `max_tokens=1`,
the grouped all-EOS round and `Runtime.serve`'s result; `R4` in `tests/engine/test_ironmule.py`
`test_mlx_backend_step_honours_fused_argmax_contract`; `R5` in
`test_telemetry_does_not_present_zero_as_a_correctness_check` and the benchmark's
structured mismatch exit in `tests/engine/test_benchmark.py`; `R6`/`R7` in
`tests/engine/test_tune_policy.py`; `R2` and `R3` are open entries of their own). What remains
is Apple-Silicon model CI. GitHub's hosted `macos-14` runner does run Metal (CI's doctor
step prints `MLX Metal device: Metal GPU operation verified`), so the integration suite can
run there against a cached, pinned snapshot; open until that job exists and is green.

### `C1` — Safe cache, chat and sampling expansion

**Mechanism.** Capacity buckets and an LRU prefix-cache budget can reduce mixed-prompt
memory, while multi-message templates and custom stops make the API useful. Sampling
must remain a separate seeded mode because exact-greedy guarantees do not transfer.

**Test.** Prefix mismatch rejects by default; per-bucket peak memory and hit rate;
model-family chat-template corpus; deterministic seeded-sampling distribution tests.

**Kill.** Any cross-tenant/prompt cache reuse, unbounded cache growth, exact-mode token
change, or no material memory reduction from bucketing. Architecture approval required.

### `Q1` — Expand the evidence matrix and run sustained-load gates

**Mechanism.** Current evidence is narrow in chip, model family, quantisation, context,
output length, concurrency and workload. A matrix plus smoke/1 h/6 h/24 h and burst
profiles can separate compatibility from performance and stability.

**Test.** Record exact model revision, hardware, framework, power, RSS/MLX peak/swap,
TTFT p50/p95/p99, throughput, queue depth, fallbacks and cache hits for every cell.

**Kill.** A cell without raw data, repeats, correctness gate or a comparable baseline
cannot extend the validity domain. Hardware/model acquisition and long runs require
explicit resource approval.

### `D1` — Reproducible community bundles and release supply chain

**Mechanism.** A local bundle with console output, raw JSON, fingerprints, checksums and
privacy preview enables external replication; clean build/install, dependency scanning,
SBOM and signed artifacts make releases inspectable.

**Test.** Offline bundle round-trip and redaction tests; clean wheel install/CLI smoke;
release dry run with tag/version equality. Submission is always opt-in.

**Kill.** Hidden upload, private prompt leakage, non-reproducible metadata, or unsigned/
unverifiable final artifacts. Publishing a tag/release requires separate user approval.

| | idea | effort | guessed payoff | correctness risk |
| :-- | :-- | :-- | :-- | :-- |
| `B1` | Width sweep at 27B | hours | 0 – 5% | low, gated by token identity |
| `B2` | Group the `lm_head` only | days | 1 – 3% at 4B, less at 27B | medium |
| `B3` | Unroll k decode steps into one graph | days | 2 – 8% | medium, this axis broke tokens once |
| `B4` | Wire the weights against page pressure | hours | 0 – 10% under load | none |
| `B5` | Fill the group on purpose | hours | 0 – 5% at short answers | none, latency cost |
| `B6` | Cost ratio of `M=4` vs `M=1` against model size | hours | 0, it is a precondition | none |
| `B26` | Qwen3.8 27B: same size, different family | hours | 0, it separates two explanations | none |
| `B30` | Widen Qwen grouped batch-1 groups to 5/6 | days | 0 – 10% | medium, throughput/correctness |
| `B8` | Native decode loop, no Python per operation | weeks | **at most 32/25/17% at 1B/4B/12B** (`B24`) | low |
| `B9` | Record the decode step once, replay it | weeks | **at most 32/25/17% at 1B/4B/12B** (`B24`) | low |
| `B11` | Layer-level pipelining across the group | weeks | 5 – 15% | medium |
| `B12` | Jump the `M=8` valley to width 16 | days | up to 40% throughput | **high** |
| `B13` | Speculative decoding with a real draft model | weeks | 1.5 – 3x at 27B, **rejected once at 4B** | low if verified greedily |
| `B14` | A draft head on the target model | months | 2 – 3x | low if verified greedily |
| `B16` | Lower or mixed weight precision | days | 20 – 40% | **high**, quality |
| `B17` | KV cache quantisation | days | 0 – 5% | medium |
| `B18` | Adaptive layer skip | weeks | 10 – 40% | **high**, quality |
| `B19` | Exploit Gemma's 5:1 local/global layers | days | 0 – 10% memory and bandwidth | medium |
| `B20` | `lm_head` on the Neural Engine | months | 10 – 16% at 4B | medium |
| `B21` | Small projections on the CPU while the GPU runs | weeks | 0 – 10% | medium |
| `B22` | Two processes, one GPU | days | 0, it is a control | none |
| `B23` | Weight layout tuned for `M=1` | weeks | 0 – 20% | low |
| `B41` | Pad the reduction dimension to 512 | **measured 2026-09-09** | **3.4% at 12B**, -5.8% at 1B, 0 at 4B | medium, logit difference |

---

## Opened 2026-08-29, from making the repository usable by strangers

These are not release blockers. They came out of running the project the way someone
who just cloned it would, and out of the review that followed.

### `Q3` — Adaptive optimizer method selection and replay

**Mechanism.** Reuse `tune.Knobs`/`SEARCH` and the evidence-bound execution
surfaces from B27 to make optimizer-method choice itself evidence-bound. Start
offline with no runtime import: define a durable state/action/outcome/failure/
uncertainty schema and an information-gain signal, then replay recorded studies
before allowing any method to influence tuning. Treat reinforcement learning as a
hypothesis, not an assumption; it is justified only if the data show a sequential
benefit that simpler search methods cannot provide.

**Test.** At equal evaluation budgets, compare the current coordinate descent and
`BASE` against seeded Random, a Bayesian/surrogate method, and a contextual bandit.
Use disjoint `TRAIN`/`VALIDATION`/sealed `HOLDOUT` group splits by study, model,
hardware and time, and report best-known outcome, regret, uncertainty calibration,
failure recovery and replay determinism. Keep all safety gates external to the
optimizer: exact output/token/stop/count identity, resource and leak checks,
fingerprint/domain validity, timeouts and rollback must pass before a candidate is
considered. No runtime import, routing, persistence or activation is part of this
experiment.

**Kill.** Insufficient real coverage or sealed-holdout evidence means data
collection only. If a simple method is equal or better at the same budget, do not
introduce RL. Any output divergence, correctness failure, resource leak, unsafe
failure handling or non-reproducible replay stops the experiment and leaves the
current deterministic coordinate descent unchanged.

**Current result (2026-08-31).** The real-data replay is `DATA_INSUFFICIENT` for
adaptive method comparison and `NOT_APPLICABLE` for offline RL. The frozen dataset
has SHA-256 `f67d975788763e4238019a3be7afa5394efbe2f2faea3a96a927e7cf522f2e33`,
dataset ID `d4ae0c148e826de85c7aa5338f892b5571481a105f558d463e9d041f63dc82b7`,
14 observations, 12 actions and 160 B36 raw timing samples. Q2 contributes a
validation trajectory, B36 a sealed holdout, and there are no training rows; the
counts are inventory facts, not statistical qualification. Q3 remains open. The
next missing evidence is separated into (1) a complete raw counterfactual action
panel for coordinate/random/BO-surrogate replay, (2) independent grouped contexts
with comparable panels for generalisation and contextual bandits, and (3) a measured
sequential horizon before RL can become applicable.

### `Q4` — Evidence-bound hierarchical RL and two-stage hybrid optimizer

**Mechanism.** The Q3 contract has a durable offline state/action/outcome schema, but
its ten-knob action space cannot express the execution/scheduling evidence that made
E14b and E16 useful. Q4 joins the existing Q2, B35, B36, B27, E14b, E16 and X1
knowledge through content-hash migration while keeping two closed spaces separate:
`KnobAction` for the ten runtime knobs and `ExecutionStrategyAction` for plan, mode,
grouping, synchronization, cache and workload policy. A hierarchical conservative
offline-RL policy can then learn an H=17 sequential trajectory (`11 KNOB_DELTA`, five
`STRATEGY_SELECT`, one `REVALIDATE`) before final revalidation, with evaluator-owned
correctness/resource gates, separate knob/strategy value heads and failure-risk critic,
behaviour prior, exact propensities, action masks and uncertainty/OPE checks. The
`HybridOptimizer` is two-stage and shadow-only until a later architecture decision.

**Test (remaining hardware/evaluation work).** The offline contracts, corpus importer,
replay methods, strict OPE, dataset gate, shadow envelope and foreign replay boundary
are complete; the durable implementation report is
`research/raw/Q4_implementation_report_20260901.md`. Remaining work is explicitly
named collection only: on the local M1 Max, collect 24 entirely new contexts spanning
Gemma 1B/4B/12B (`no27`) in new `Q4_TRAIN`/`Q4_VALIDATION`/`Q4_SEALED_HOLDOUT`
groups by study/model/manifest/workload/hardware/runtime/time; build 12-knob and
plan-matching five-strategy panels; collect 72 complete H17 trajectories (1224
transitions); then run the equal-budget BASELINE/current-coordinate/seeded-random/
BO/surrogate/contextual-bandit/OFFLINE_RL evaluation once on the direct sealed panels.
Historical `Q3_VALIDATION`/`Q3_SEALED_HOLDOUT` and E11 `LEDGER_ONLY` rows remain
outside Q4. The Stage-2 panel is 12 knob actions × 5 matching strategies = 60 exact
cells/context, in 12 anchor phases; S11/S12 are separate risk probes. Each collection
phase is separately preregistered and explicitly user-started, with 16 candidate
decisions per context (11 knob-delta + 5 strategy; BASE external), exact identities,
propensities, p95/request gates, grouped metrics, recovery and replay checks. The
runtime remains unable to import, select, write or activate from the shadow report.

For each context, Stage 2 uses only the five strategies matching its frozen plan:
S01--S05 for StrictOneShotPlan or S06--S10 for ReusableSessionPlan; the opposite plan
is not exact/safe. The Stage-2 interaction panel is the full 60-cell cross-product,
collected in 12 separate knob-anchor phases; each phase uses five fresh processes, one
per strategy under that knob anchor (five arm cells, two warmups and five measured
repeats), and any missing pair forces BASE fallback. The shared BASE reference is
external to the method budget. The H17
trajectory has explicit `KNOB_DELTA`/`STRATEGY_SELECT`/`REVALIDATE` stages; every delta
changes exactly one legal knob field, accepted targets update the current action and
rejected/failed targets leave it unchanged. Partial aborts are terminal at the current
step and complete only at step 16. Knob FQI and strategy-immediate value heads are
separate; the hybrid never scalar-adds them.

**Kill.** Keep the current deterministic coordinate path and mark Q4
`DATA_INSUFFICIENT` if any split, full panel, independent context or measured H=17
horizon is absent. Mark RL `NOT_APPLICABLE` until the sequential horizon exists. Reject RL
permanently for this decision when the best simpler method is equal or better at the
same budget, RL does not beat it on sealed-holdout reward and both time-to-best and
experiments-to-best, regression is worse, or uncertainty/OPE support is uncalibrated.
The direct grouped advantage must have a 95% lower bound `>+2pp`; the equivalence
margin is `1pp`. A DR estimate with opposite sign or absolute disagreement `>2pp`
invalidates `RL_WINS`. `time_to_best`/`experiments_to_best` use original cost
`c <= 1.01*c_oracle`; unsafe/censored and safe `c > 1.02*c_BASE` rates use separate
exact denominators.
Any token/output/stop/state divergence, safety/resource/cleanup failure, split
leakage, fabricated/imputed/summary-only label, nondeterministic replay, unauthorized
runtime import, profile write, routing or activation is terminal; BASE/current
coordinate remains unchanged. WIS uses ratio clip 10 and grouped five-fold DR by
complete context/group hash (all trajectories in a context co-fold); any overlap/support
failure yields `OPE_UNSUPPORTED`. Do not retry or pool Q3c/Q3d/Q3e/Q3f, promote X1/B35/
B27 summaries or exploratory true batching, combine E14b's `+18.02%` and `+20.05%`,
or invent foreign-Mac measurements.

### `Q3a` — Path interaction: final Q2 incumbent versus `fused_argmax`

**Mechanism.** Q2 evaluated `fused_argmax` early and then retained
`compiled_fixed_cache=True`, `head_skip_prefill=True`, and `readback_every=2`.
This six-process, balanced fresh-process A/B pilot asks the narrower path-dependence
question: does enabling `fused_argmax=True` change the result only after the final
incumbent has been assembled? Use only the locally cached
`mlx-community/gemma-3-4b-it-4bit`, two warmups and seven measured repeats per arm.
The dry-run is the default; execution is capped by the preregistered conservative
`6 * 35 s + 60 s = 270 s` bound and must stop before start if the bound exceeds
300 s. AC power, low-power off, nominal thermal state, no competing model process,
swap and resource ceilings, three-sample load average (`max <= 4.0`, spread `<= 1.0`),
strict process inventory, exact token/count/stop-reason/determinism, complete raw
evidence, and runtime/model/environment identity are hard gates. The parent owns a
monotone 300-second deadline; each child is capped at 35 seconds and the worker
phase at 240 seconds. A passing result is information gain about path interaction
only; there is no profile activation or promotion.

**Kill.** Refuse before start on any unknown preflight state, missing local model,
missing model revision/manifest, competing model process, or runtime bound above
300 s. During execution, timeout, crash, swap/resource breach, malformed process
inventory, load-gate breach, incomplete raw data, or any token/stop/count mismatch
records `FAILED` with `BASE` fallback and stops the pilot. A clean run still cannot
promote a profile; classify the paired total ratio before measurement as `GAIN` only
when `ci_high < 0.995`, `LOSS` only when `ci_low > 1.005`,
`PRACTICALLY_NEUTRAL` only for a complete CI inside `[0.995, 1.005]`, otherwise
`INCONCLUSIVE`. Do not claim a direct statistical comparison with early Q2.

**P2 safety debt (runtime lifecycle) — answered 2026-09-26 (agent decision).** `ab.run`
passes every completed child record as `ABRunError.partial_children` on timeout, start
failure, malformed output and non-zero exit, and a failed cleanup raises (`child ... and
cleanup failed`) rather than returning a short run (`tests/engine/test_ab.py`). A whole
process group needs no signal: the child runs under the q3f guard, which refuses and
records every process operation (`subprocess.Popen`, `os.system`, `os.fork`, `os.forkpty`,
...) before the package imports
(`test_q3f_guard_blocks_and_records_every_process_operation_in_isolated_child`), so a
timed-out child has no descendants to orphan. Reopen if the guard's operation set shrinks.

**P2 safety debt (evaluator-owned identity) — mostly answered, checked 2026-09-26.**
`ab._validate_child_record` requires the exact field set per child and per arm and
cross-checks it: per-repeat logical and physical tokens, their counts
(`{"logical": len, "physical": len}`), stop reasons in `{eos, length}`, positive
capacities, `decode_steps == len(physical[0]) - 1` and a recomputed determinism flag;
a missing or extra field fails. Still open: RSS is not recorded at all, and
`mlx_peak_bytes` is the child's own report, i.e. self-asserted. Kill when a new
execution path can pass validation with absent or self-asserted resource evidence.

**P2 safety debt (streaming worker output) — answered 2026-09-26 (agent decision).**
`ab._bounded_communicate` replaces `communicate()`: a selector drains both pipes as they
fill and stops once either passes 512 KiB, so at most the cap plus one 64 KiB read is held,
and `ab.run` then terminates and reaps the child (`output limit exceeded`). A real child
writing without end is stopped by the cap, not the timeout, and reaped; a real child that
puts 400 KiB on stderr before its `@@` marker keeps the marker (`tests/engine/test_ab.py`).
No process group is needed for the reason (a) gives. Reopen if the child may spawn
processes again.

---

## Tier 0 — measured and rejected. Re-open only under the rule below.

- **RL2 learner for GPU placement and group schedule (2026-10-28).** Experiment `RL2-H-run1-20261028` (kernel
  `tobayko/ironmule-rl2-headroom` v1, commit `258ccbe`; see RL2-H above): on two T4s with Qwen 3 1.7B bfloat16 at
  0.2-0.95 of capacity, the oracle over 20 fixed arms beat the best single arm by -0.05 % held out (95 % bootstrap
  to +0.13 %) against 3.5 % A/A noise, and the reference was within 0.4 % of the best arm at every level. RL2's kill
  holds for these two decisions on this fingerprint: no learner for them. Re-open with a new model, checkpoint,
  card, traffic shape or decision (the MEM2 bound and threads were not measured).

- **RL3 length-aware placement and order (2026-10-29).** Experiment `RL3-H-run2-20261029` (kernel
  `tobayko/ironmule-rl2-headroom-h3` v2, commit `6eaa4b4`; see RL3 above; run 1 was invalid, X1): on two T4s with
  Qwen 3 1.7B bfloat16 under quiet, bursty and spike traffic with long prompts and 32- or 256-token answers, no
  decision had held-out headroom above the 18.7 % A/A noise. The joint decision scored -0.7 %, placement -1.2 % and
  order +0.6 %. Shortest-job-first was within 0.02 % of the reference, and segregating long from short requests
  cost 15-24 % in bursts. RL3's kill holds: no learner for placement or order by output limit, and the reference
  stays. Re-open with a new model, card, traffic shape, cost weighting or decision. A noise estimate that one A/A
  pair does not dominate also counts: here the second-largest pair was 6.6 %.

- **RL4 model-based RL agent for per-request placement and order (2026-10-29).** Experiment `RL4-SIM-20261029`
  (see RL4 above): a REINFORCE policy trained in a pool simulator fitted to RL3-H run 2 passed the fidelity gate.
  In the simulator it beat the best fixed arm by only 2.8 %, against an 18.7 % A/A noise, so the simulator gate
  failed and no learned arm went to hardware. Re-open with a testbed whose noise is several times smaller than a
  few per cent, or a decision whose simulated gain is several times larger.
  *Re-opened for `spike` by RL4-S (above):* on 30 new spike traces the frozen policy beat the best fixed arm on the
  T4s by 9.0 % (95 % 6.9-11.1 %) against 2.9 % noise. Mixed traffic is still unvalidated.

- **RL6 width-learning policy against fixed widths (2026-10-29).** Experiment `RL6-run1-20261029` (see RL6 above):
  the v2 policy that also picks each group's width gained 0.8 % (95 % -1.9 to 3.4 %) over the best fixed arm,
  `shortest/w1/g0/sjf`, against 3.0 % noise. No width learner. The fixed narrow shortest-first schedule it found
  is the candidate default (RL6).

- **A knob learner over the engine's execution knobs on the T4 (RL7, 2026-10-29).** Experiment `HW4-run1-20261029`
  (see RL7 above): seven knob sets are class A but none changes order with the group width, so knobs and
  scheduling separate and the best fixed profile plus the best schedule is the policy. Re-open when a knob's cost
  per further session changes shape against another's.

- **`head_skip_prefill` as an exact knob on CUDA (2026-10-29).** Experiment `HW4-run1-20261029`: the output head at
  one row gives other bits than the head over the whole prompt, so the prefill's logits change. Re-open with a head
  kernel that is invariant to the row count.

- **`prefill_into_fixed` on CUDA (2026-10-29).** Experiment `HW4-run1-20261029`: tokens kept, prefill 7.8 % slower.
  `E1`'s kill ("the phase it would optimise costs 1.47 ms of 537 ms") holds on the T4.

- **Prompt-lookup speculation as an exact knob on CUDA (2026-10-29).** Experiment `HW3-run1-20261029` (see HW3
  above): on a T4 every combination with `speculate_k` 3 changed tokens (first at steps 23-58 of 64), acceptance
  0.42-0.47, 0.99-1.14x. The verify forward's k + 1 rows take MLX's slower, non-invariant multi-row path. Re-open
  with a verify path whose logits are bit-identical to the sequential step (KB1).

- **True tensor batching through MLX's bf16 matmul on the T4 (2026-10-29).** Experiment `HW3-run1-20261029` (see
  TB1 above): stacked rows changed tokens and never matched bit for bit, and eight rows made a token only 1.34x
  cheaper, because MLX's bf16 product at 2-32 rows runs at 41-45 GB/s against 252 at one row. Re-open with a
  small-M kernel (KB1) or another card.

- **`fuse_projections` and `fused_argmax` on CUDA decode (2026-10-29).** Experiment `HW3-run1-20261029`: fusion
  1.010 (slower) on decode and +0.5-1.4 % on prefill; fused argmax 1.002. Neither helps on a T4 with Qwen3 1.7B
  bf16.

- **RL6's width policy as the product's learned dispatch on mixed traffic (2026-10-29).** Experiment
  `LD1-V-run1-20261029` (see LD1-V above): through the product's own pool and scheduler it beat the reference by
  21.2 % (95 % 18.3-24.3 %), but in `bursts` it cost 6.9 % more than `shortest/w1/g0/sjf` (95 % bound -12.9 %
  against 8.1 % noise), so the no-harm rule failed. No evidence for it; LD1 ships only RL4-S's spike-bound policy.
  Re-open with a policy trained against a per-state harm constraint, or traffic whose bursts it was trained on.

- **KP1 run 1: compile + readback 4 as the tuned engine's default, through the product (2026-10-29).** Experiment
  `KP1-run1-20261029` (see KP1 above): answers identical and class A at 27 capacities, but +0.2 % weighted (95 %
  -1.9 to +1.6 %) and `spike` -1.3 % (bound -5.4 % against 4.7 % noise). The worker held each token for the next,
  so the first token waited a whole chain; fixed, and re-opened as H9.

- **KP's knobs after the first-token fix (2026-10-30).** Experiment `H9-run1-20261029` (see H9): +1.7 % weighted
  (95 % 0.39-3.15 %) against a 0.81 % A/A floor, no harm in any state, `quiet` +6.1 %. Too small to resolve in 12
  traces. Re-open with more traces, or once the default's own cost changes again.

- **MOLE1 local adaptive route and indexed preparation.** Experiments
  `MOLE1-LOCAL-1-20260912-attempt1` and `MOLE1-LOCAL-2-20260912-attempt1`:
  no adaptive benefit over C; LOCAL-2 preparation missed its 5% benefit gate.
  Full results: `ironmole_mcp/docs/RESULTS.md`. Agent-level comparisons remain open.

- **R15, narrowing the competing-process gate's substring match.** Opened and closed
  2026-09-26 on inspection, no experiment: its own kill holds, because the Q3c and Q3d
  records bind `research/q3b_residual_swap_canary.py` in `runtime_code_sha256`. The gate
  stays fail-closed; q3f's real cleanup test skips with the reason (ledger R14).

### Rejected is not forbidden (project rule, 2026-09-10)

Tier 0, `NO-GO` and every closed kill entry are **historical evidence, not a permanent
ban**. They record what a specific mechanism cost on a specific fingerprint, and that is
exactly how far they reach.

Any of them may be re-opened when one of three things is new:

* **new hardware evidence** — a different chip, memory size, MLX/mlx-lm build or model
  revision than the one the entry was measured on;
* **a changed mechanism** — the reason the entry died no longer applies, stated
  explicitly against the old entry's own kill criterion;
* **a new implementation** — a different code path, kernel or execution route, not a
  re-run of the same one.

Re-opening costs one thing: the new entry must name the old entry, quote the old kill
criterion, and say which of the three conditions above is met. Repeating an experiment
with no such statement is still forbidden, because that is what wastes GPU time. Refusing
a method *only* because an older attempt failed is equally forbidden, because that is what
freezes a runtime.

- **`B15` exact-but-pruned `lm_head` (2026-09-09).** Measured offline on
  `gemma-3-4b-it-4bit` over 72 real greedy decode steps: with a correct
  Cauchy-Schwarz cluster bound, `95.5%` of the 262,208 rows survive at `k=16384`
  and `98.5%` at `k=4096`, against a kill threshold of `25%`. The token was exact
  on 72 of 72 steps, so the bound is right and the geometry is the problem. The
  bound needs a cluster radius of `0.106`; k-means reaches `0.387`. It is not a
  clustering failure: sampling 256 rows, only `1.25` rows on average lie within
  the required radius of any row, and the mean nearest-neighbour distance is
  `0.235`. Achieving the radius would need roughly one cluster per row, and the
  centroids alone would then cost more than the rows they replace. Row norms are
  nearly uniform (`0.006` to `1.085`, median `1.000`), so the pure norm bound
  prunes `0.02%`. Experiment `B15_4B_offline_bound_20260909_attempt1`; do not
  re-run with a different clustering.

- **`B48` sharing the aligned projections too (2026-09-09).** Kernel GO for `K=4096` and
  `K=15360`, bit-identical, but the product gain against `PairedThroughputMode()` was
  `0.9903` and `0.9920` against a `5%` goal. `share_aligned` stays `False`. Full entry in
  `research/LEDGER.md`.

- **`B49` one weight sweep for four requests (2026-09-09).** Bit-identical, but four shared
  over two shared pairs is `1.1084`, CI `1.1014 – 1.1205`, slower in 20 of 20 blocks. Wider
  sharing is not the lever at `K=3840`. Full entry in `research/LEDGER.md`.

- **`B56` a per-request objective (2026-09-10).** Implemented, default unchanged, `NO_GO`
  in `B56` and `B56b` (`research/LEDGER.md`).

- **`B68` MLX command buffer limits (2026-09-10).** `MLX_MAX_OPS_PER_BUFFER` and
  `MLX_MAX_MB_PER_BUFFER` swept in 48 fresh processes on 4B and 12B: no interval below
  `1.0`, three outright losses, `DEFAULT_WINS` on both models, MLX `50/50` stays.
  Experiment `B68_command_buffer_sweep_20260910`. Proposed again on 2026-09-23 as a first
  step; not re-run, because nothing about the mechanism or the hardware changed.

- **`B10` fewer kernels per step (2026-09-09).** Bounded by `B24S`: norms, adds, rope,
  attention and copies together are `0.2` to `0.7%` of decode GPU time, so fusing all of
  them away returns under `0.7%`. Experiment `B24_metal_trace_series_20260909_attempt1`.

- **Fusing `lm_head` with argmax (proposed 2026-09-23; derived, not measured).** The
  saving is the logits write and re-read, about `0.5` to `1 MB` per token against
  `2.561 GB` of weights read per 4B token (`B24R`), and argmax sits inside the `0.40%`
  non-matmul share of 4B GPU time (`B24S`). That bound is below every equivalence margin in
  use. The `fused_argmax` knob already moves argmax into the compiled body; `E11` rejected
  it and `Q3a` asks whether that changes on the final incumbent. Re-open only with a
  profile that shows the head's epilogue as a cost.

- **Judging a module by searching its source text (`B82`).** Four studies wrote checks that
  matched the module's own prose rather than its code. Structural claims are checked on the
  syntax tree. See `research/LEDGER.md` B82.

- **`B76` a contextual state model for the `(4, 8)` stack ratio (2026-09-11).** Ridge over
  `load_1min`, `memory_free_percent` and `swap_used_gb`: prediction error `0.0087` against a
  constant model's `0.0058`, larger than the whole between-session spread. Those features
  carry no signal; a new feature set needs a mechanism first.

- **`B77` plain mean against a Bayesian posterior (2026-09-11).** Identical action sequence
  and regret over fourteen sessions; the data cannot tell them apart until the spread moves.

- **PROD8 12B 1077-token integration screen (2026-09-07).** First real stock-reference request hit the frozen 6 s host deadline (6.002551 s observed); owned worker reaped, no product/HTTP run. Experiment `PROD8_12B_long_context_20260907_attempt1`; no shorter-context or relaxed-limit retry. Phase-aware follow-up hypothesis: root `BACKLOG.md` PROD9.

- **PROD2/PROD6 12B short-prompt calibration (2026-09-07).** Complete installed 90-call protocol, exact outputs/resources and three normal worker exits; no qualified net gain at any cap under the frozen noise rule. Experiment `PROD6_12B_installed_20260907_attempt1`; no adoption or same-design retry. Results: `docs/PROD6_12B_RESULTS_2026-09-07.md`.

Listed so the next person does not spend a week rediscovering them.

- **PROD2/PROD6 1B short-prompt calibration (2026-09-07).** Complete installed 90-call protocol and clean worker exits, but no qualified signal above the frozen noise floor; no adoption. Experiment `PROD6_1B_installed_20260907_attempt1`; do not repeat for a luckier result.

- **PROD2/PROD3 4B short-prompt calibration (2026-09-07).** Complete installed 90-call protocol, exact outputs and resources valid, but all three caps remain below the frozen noise qualification; no adoption. See `PROD3_4B_installed_20260907_attempt1` and `docs/PROD3_RESULTS_2026-09-07.md`; do not rerun the same design for a luckier result.

- **`Q3c` direct replication attempts (2026-08-31).** Run 1 was refused before
  a phase because load `8.294921875 > 8` (raw SHA-256
  `5270c0f38e50984cd26223aa2a9817982fc5a1861ddbe2caa3cff98393c9e8d5`); run 2
  aborted after `105` samples / `27.395 s`, swap `2,353,654,661 B` to
  `2,625,172,930 B`, delta `271,518,269 B` (`258.94 MiB > 128 MiB`), with
  cleanup unverified. No timings, identity, performance result or promotion
  exists. Raw run 2 SHA-256 is
  `d94db80402254c87c0e4a0128cf802e1eaa59d42c4459c2f208077f48c38b8df`; retain
  both records and do not rerun Q3c.

- **`Q3d` model-free recovery gate (2026-08-31).** The gate itself passed:
  `61` samples over `60.020192667 s`, maximum adjacent gap
  `1.013944625 s`, swap `2,651,722,874 B` throughout and delta `0 B`.
  Raw SHA-256 is
  `4699a49b174db31580a9701ef2075f8b1964d309b0f857dd7779fb230cfccb83`
  (`34,144` bytes); the summary SHA-256 is
  `3b43e267000ba15b9d9079d9f118e59c1cd51dbcdfecc067c20995b01a0a1c3e`
  (`970` bytes). The one permitted Q3c invocation was then refused before
  `Popen` because macOS `/bin/ps` rejected `sid` (`rc=1`, `ps: sid: keyword not
  found`). No model, inference process, timing, identity or performance data
  exists. Final state is `Q3C_FAILED` with `BASE/current incumbent` fallback;
  do not repeat Q3d or pool its gate with Q3c. Q3e is the separately frozen
  portability-repair path.

- **`Q3e` portable process probe and one Q3c invocation (2026-08-31).** The
  repair removed unsupported macOS `ps sid` and the exact Phase-R run completed
  with token/resource identity intact. Its descriptive incumbent/BASE total
  ratio was `0.857466859207542`, CI
  `[0.8551668079699586, 0.8611021999710893]` (`14.2533140792%` faster), but
  cleanup conservatively rejected four stable unrelated same-UID launchd
  services that appeared after the worker baseline. The raw result is
  `research/raw/Q3e_q3c_final_20260831.json`, SHA-256
  `1df6c81dc824911016e687883c535f1ec314f3e03b51303b04c38ae71bb6f4ea`, size
  `2,205,857` bytes; the terminal note is
  `research/raw/Q3e_terminal_result_20260831.md`. Status is `FAILED`, Phase N
  did not run, and fallback remains `BASE/current incumbent`. Do not pool or
  rerun Q3e; Q3f is the separately frozen attribution path.

- **`Q3f` terminal attribution path (2026-09-01).** The single permitted Q3f
  execution reached the exact local Gemma 4B worker only far enough to fail with
  `ABRunError: child 0 start callback failed`; no child marker, ledger, timing,
  token, Phase-N or accepted performance evidence exists. All 14 preflight
  checks passed, AC/thermal/low-power gates were green, and swap stayed at
  `2,609,643,520 B` across 25 samples with zero delta. Cleanup reaped the
  worker and rejected the run closed because guard/ledger evidence was
  unavailable and same-UID Spotify Helper PID `52017` could not be proven
  unrelated. Raw SHA-256 is
  `e82accdbd52857e6201fa2b34984765e61658ecfd3956d5c903a49c1e6de70a9`
  (`2,487,533` bytes); terminal note is
  `research/raw/Q3f_terminal_result_20260901.md`. Status is `FAILED`, with
  `BASE/current incumbent` fallback. Do not retry Q3f, pool its record or say
  that historical Q2 speed was reproduced; any child-visibility race fix needs
  new authorization and preregistration.

- **`R11/R12/E15` fork-per-block memory-integrity path (b700377).** Closed by the
  complete four-block E15 after-file (SHA-256
  `d14875e43ee800d8f1a29af966b8adad56245a414dd204f202a48b81d1f91b5c`): four fresh
  PIDs, flat per-block peaks, no swap growth, no correctness divergence. This is an
  engineering memory result only, not a speed claim; do not repeat the exact attempt.

- **`B25` KV cache reallocation during decode.** Nothing reallocates. The fixed-shape
  cache is allocated once per `serve()`: `mx.zeros` appears only in
  `_empty_fixed_state`, and `_caches_from_state` wraps the existing arrays rather than
  copying them. Measured, not just read: writing 56 tokens through `FixedKVCache`
  moved active memory from `65,644 B` to `32,876 B` — it *fell* by exactly one full
  keys+values copy (`32,768 B`), which is the warmup's double buffer being released
  once MLX takes the `slice_update` donation. Shape constant throughout.
  `tests/test_cache_allocation.py` holds both properties. The predecessor's `4.4263%`
  from cache growth copies (candidate 21) is gone with the growing cache it was
  measured on, so there is nothing here to claim.

  **One caveat worth carrying, because it shapes how the tuned gain reads.**
  `FixedKVCache.update_and_fetch` builds `mx.array(0)` and `mx.stack(...)` on every
  call, and `make_mask` an `mx.arange(capacity)` — per layer, per decode step. Under
  `compiled_fixed_cache=True` these are bound once when `mx.compile` traces the body,
  so they cost nothing. Under the untuned `BASELINE` they are rebuilt every step. Part
  of that knob's `0.9679` is therefore host work that stops happening, not GPU work
  that got faster — which is the documented purpose of `mx.compile`, and `E5` already
  put `3.3 ms` of host work on each step. How the ratio splits between constant rebuild
  and kernel fusion is not measured. The tuned gain against `BASELINE` is real, because
  `BASELINE` is what an untuned install actually runs; it is not a gain against a
  well-optimised floor.

- **`R11` measurement is gated on swap, not on a byte count.** The literal
  `12 * 1024**3` was wrong in both directions: it refused `gemma-3-12b-it-4bit` at a
  `17.51 GB` block that never swapped, and it passed a run that had to be discarded
  because swap climbed `2816 MB`. `ironmule/bench.py` now carries `MemoryGate`, which
  aborts on a *delta* against the run's own swap baseline — an absolute value would
  refuse every run on a machine already carrying backlog — with a coarse peak backstop
  derived from `hw.memsize` rather than typed in. Wired into `e14b_arms.py`,
  `e15_service.py` and `e12_window_falsification.py`; `e16_replication.py` keeps its
  per-child ceiling, which was always coherent because it forks. The self-check replays
  both reference runs from `B7`'s recorded numbers, so the gate is tested against what
  actually happened rather than against what sounds reasonable. A gate that can read
  neither swap nor installed memory reports itself `inert` instead of silently passing.

- **`R9` `ironmule.tune` resolves to the function, not the module.** `__init__.py:35`
  re-exports `tune` from `.tune`, rebinding the name in the package namespace, so
  `import ironmule.tune as m` yields the function and the module is reachable only via
  `importlib.import_module("ironmule.tune")`. Renaming would break the public API, so
  this is documented rather than changed — the decision its kill criterion allowed for.
  Checked across the package: exactly one of fifteen submodules is shadowed this way.
  `tests/engine/test_ironmule.py` asserts that, so a second one cannot appear unnoticed; the
  quirk stays a footnote instead of becoming a pattern.

- **B27e mirrored cross-commit control.** Four fresh 4B processes, source-surface
  digest `ec242c…`, all correctness/resource gates green. OLD/D1 block ratios were
  within 2%; mirrored D1/OLD made D1 appear 5.8–7.9% faster. Final
  `ORDER_OR_TEMPORAL_DRIFT`; a consistent D1 slowdown was not reproduced, but B27d
  remains formally inconclusive. Do not rerun the same two-block unconditioned design.

**Two lists, one of them closed.** The entries below marked `E*` come from this
project's ledger. The ones marked `cycle *` come from the predecessor project, whose
candidate catalogue is `EXPERIMENT_BACKLOG.md` in the source repository — 24 candidates
over 17 sealed cycles, most of them closed. That file is history and takes no new
entries; everything current goes here.

- **Prompt-lookup speculation.** `E0c`: `2.9x` slower, acceptance `0.17` per drafted
  token. The workload does not repeat itself enough. This does **not** kill `B13`,
  which uses a different draft source.
- **Projection fusion as a decode win.** `E5`: paired across six processes, decode
  ratio `0.9990` with a CI containing 1.0. Kept for its `-1.10%` prefill effect only.
  `E0b`'s `+2.4%` was noise.
- **A dedicated `M=1` fast path.** `E2`: the apparent `M=1` cost was a `~0.49 ms`
  `eval`+`synchronize` round trip, not kernel behaviour.
- **`prefill_into_fixed`.** `E1`: the phase it would optimise costs `1.47 ms` of
  `537 ms`. Killed before it was benchmarked.
- **Padding `M=322` to 384.** `E2`: `~4.74 ms` predicted against `4.43 ms` measured.
- **An adaptive width controller.** Realised width already adapts; nothing measured
  beats a fixed 4. Revisit only if `B1` finds the optimum is model-dependent.
- **A draft model drafting for a larger target** (predecessor candidate 11, 1B for 4B).
  Measured and rejected, recorded as `0.560x`. **The sign convention on that row is
  ambiguous** — elsewhere in that file a ratio below 1.0 marks a win — so re-derive the
  direction from the cycle's raw data before trusting it either way. See `B13`, which is
  the same idea at a target/draft ratio the predecessor never tried.
- **`mx.compile` over decode subgraphs on a growing cache** (predecessor candidate 12).
  `-23.8%` dispatch and **wrong tokens from position 2**. Rejected, and the older
  device-model compile numbers were invalidated with it. The same idea on a *fixed-shape*
  cache is what this runtime ships: cycle 16 measured `0.9296` with identical tokens
  across 18 arm executions, and `E0a` reproduced it. The lesson is about the cache, not
  about compiling. See `B3`.
- **A custom Metal kernel chosen without profiler evidence** (predecessor candidate 14).
  Locked because cycle 9 localised no single kernel hotspot. Still the right rule:
  `B24` before `B10` or `B23`, or it is guesswork with a compiler attached.
- **Bundled host readback**, reading the stop token only every `N` steps (predecessor
  candidate 18, cycle 17). Readback 8 was faster in every pair, but `0.9581` missed a
  preregistered 5% threshold, so the recorded decision is
  `no_clear_speedup_baseline_retained` and the `4.19%` is calculated rather than claimed.
  A valid negative result. It also points the wrong way for the scaling problem: a fixed
  per-step host cost is a *smaller* share of a longer step, so 27B would show less, not
  more. Do not re-run it hoping for a better draw.
- **B28 / `qwen_native_true_batch_v1` rejected at the correctness gate.** Widths 2, 3
  and 4 produced exact visible tokens and stop reasons with no fallbacks, but the final
  hybrid `kv_hash` differed from the sequential reference. Swap delta was `0 B`; no
  token-rate or other performance measurement is valid. Do not route this path.
- **B29c / `qwen_native_b1_v1` below target.** Widths 2, 3 and 4 passed exact
  correctness, final state and 16-token continuation with no fallback and zero swap
  delta. Candidate `16.0722` versus Interactive `15.6740` (`1.02541x`) and versus
  Throughput `16.0687` (`1.000219x`) remain below the `1.10` gate. No route.

- **B40 width sweep (experiment B40, `INCONCLUSIVE`).** W2/W4 and W3/W4 were
  directionally slower on all six blocks, but material epoch drift prevented a
  valid selection; all 18 children and safety gates were clean. No retry is
  authorized, and no W2/W3/W4 timing may be cherry-picked or treated as a
  selected profile. W4 remains the unchanged operational baseline.

---

## Known consequence of the 2026-09-11 repository layout change

The research packages moved from the repository root into `research/`. Two sealed
studies name a file by its old path inside their own provenance input list, so they
can still be **replayed and verified** but can no longer **re-seal**:

- `experiments/head_skip_formal/study.py` hashes `friday_h1/statistics.py`. Collecting
  provenance now raises `unavailable provenance input`. Its 16 records verify, its
  script and preregistration hashes match, and its verdict stands.
- `docs/F1_INTEGRATION_VORREGISTRIERUNG.md` links `friday_optimizer/integration.py`.
  The document is a hashed provenance input and must not be edited, so the link stays
  as the record of the tree it was written against.

Neither is fixed by editing a sealed file. Re-running either study on the current tree
would be a new study with its own preregistration, which is what a moved source tree
means anyway.

## Tier 1 — cheap, grounded, worth doing first

### `CHAT1` — Is IronMule's engine faster than mlx-lm's own generation in the chat? (2026-09-27)

**Observation, not a measurement.** Recording the chat page on an M1 Max with Gemma 3 4B (one
warm process each, one prompt, "Explain in about 300 words how a refrigerator moves heat out of
its interior"): `ironmule serve --engine stock` (mlx-lm `stream_generate`) answered 493 tokens in
5.97 s, 82.6 tok/s, first text after 0.43 s; `--engine ironmule` (the engine with this machine's
tuned profile, `compatible_tuned_profile`) 445 tokens in 6.00 s, 74.1 tok/s, all text at the end.
A scripted pair on a 256-token cap gave 75.5 against 70.5 tok/s. The two answers differ from
token 44: mlx-lm prefills the last prompt token alone, `StrictOneShotPlan` with the rest; each
server matches its own greedy computation exactly.

**Mechanism.** The README's Apple numbers compare IronMule's tuned engine with IronMule's own
baseline engine, not with mlx-lm's `stream_generate`, which the product serves by default and
which overlaps the next step with the readback (async eval). The engine's wins (compiled fixed
cache, head skip, readback every 2) may not cover that gap on long chat answers.

**Test.** Preregistered, fresh interleaved processes, the product HTTP path both ways, several
prompts and answer lengths, completion tokens per second from send to last token.
**Kill:** engine below 1.0x `stream_generate` -> `--engine ironmule` is not offered as a speed-up
and the README says what its "stock" is; above 1.05x -> the chat may default to it only after a
separate decision on the differing answers.

### `B57` — Qualify a kernel per shape, through the profile

**Mechanism.** `k3840_matvec` is admitted at load, against one `K`, by
`Engine.admit_k3840`. `ironmule/kernel_registry.py` already derives one MLX kernel name
per whole specification, so several specialisations can coexist without the
ml-explore/mlx#3832 collision that `B54` closed. What is missing is the record that says
*which shapes* a kernel is qualified for on this fingerprint, so the router can select
one per dispatch instead of per load. The tuned profile already carries exactly this
shape of record for the service strategy (`ironmule/service_strategy.py`): strategy,
validity range, correctness contract, evidence run IDs.

**Test.** Extend that record to kernels, qualify one shape end to end against the library
path with bit-exact outputs, and measure the routed selection against a fixed
library-only arm under the `B55` protocol. Only shapes with stored evidence may activate.

**Kill.** If per-shape selection cannot be made bit-exact against the library path for
the shapes it would select, or if the selection cost exceeds the kernel's gain at every
qualified shape, kernels stay load-time knobs.

**Since written (2026-09-10).** The record exists: `ironmule/silicon_profile.py` (`B70`)
loads, matches and reports a fail-closed per-class profile and may not change a route. `B69`
confirmed one geometry, `(4, 8)`, against the stack; it is not activated. What stays open is
this entry's question: letting a qualified per-shape record select a kernel at dispatch.

**What the kernel set may grow into (external proposal, 2026-09-23).** A small library of
variants, each qualified for one shape family and described by shape, dtype, bits and group
size, layout, hardware fingerprint, MLX version and compile options; `B54`'s name digest lets
them coexist. A variant may change only how independent outputs are mapped to threads and
threadgroups. The order in which one output's partial sums are reduced stays the library's,
which is what made `B42` and `B69` provable. Selection is by fixed rules over stored evidence
first; a learned selector enters only if it beats that table including its own cost, which
`B76`'s contextual model did not manage on this machine. A bounded template search over the
same axes (`docs/PRODUCT_RESEARCH_2026-09-05.md`, section 3) comes only once this entry has
more than one qualified variant for some shape: new candidates start locked, run through the
controlled kernel worker, and follow the rule from `B66` that only a shape the profile points
at gets a run.

### `B58` — Does the paired route survive real concurrency?

**Mechanism.** `B45` to `B50` measured `PairedThroughputMode` on a request *pair*:
`12.7-12.8%` earlier completion. `B52` measured that choosing it from the profile costs
nothing against naming it. Neither measured it under a stream of arrivals, where the
number of ready requests moves between rounds and a partner may not exist when the round
starts. `B55` never exercised it at all — this machine's profile carries no
service-strategy record, so `AutomaticMode` correctly kept the established mode
throughout, which means the routed paired path is currently untested end to end.

**Test.** Write a service-strategy record for this fingerprint from the `B45`/`B46`
evidence, then run the `B55` protocol with staggered arrivals so ready counts vary within
a dispatch. Gate on token identity per request, `paired_steps` versus `solo_steps`, and
per-request latency against the grouped path.

**Kill.** If the paired route wins only when both requests are ready at dispatch — a
condition a real arrival stream rarely meets — it stays an explicitly named mode and is
never routed to automatically.

### `B59` — Is a native Metal path faster than `mx.fast.metal_kernel`?

**Mechanism.** Custom kernels currently reach the GPU through
`mx.fast.metal_kernel`, which compiles from source at first use and dispatches through
MLX's own primitive. A native MLX extension or a prebuilt `.metallib` removes the
source-compile step and part of the dispatch wrapper. Whether that is measurable at
decode shapes is unknown; the reasonable prior is that it is not, because decode at width
one is bandwidth bound (`E4`: `104` to `324 GB/s` over `1.4` to `360 MB`) and dispatch is
not the binding constraint. This entry exists to measure that rather than assume it in
either direction.

**Test.** Build the same `k3840` matvec three ways — `mx.fast.metal_kernel`, a native
extension, a prebuilt `.metallib` — assert byte-identical outputs against the library
path on the captured inputs, and compare dispatch and end-to-end decode time under the
`B55` protocol.

**Kill.** If the three paths are within the `2%` equivalence margin at decode shapes,
`mx.fast.metal_kernel` stays the only kernel path and this entry closes. "Closer to the
hardware" is not a result.

### `B64` — Now that `wired_fraction` can be confirmed, does it pay?

**Mechanism.** `B62` removed the trap: the knob no longer starts a subprocess in a guarded
child, so a candidate that keeps it can now reach and survive a confirmation. Nothing about
that says it is worth keeping. The only evidence that it buys anything is one screening
ratio, `0.9034`, from the `12B` run that then died — a single-process screening number,
never confirmed, and taken on a machine whose memory behaviour under a wired limit is
exactly what is in question. `B61` re-screened without it and reached `0.9218` on other
knobs, so the two numbers are not comparable and must not be subtracted.

**Test.** Its own preregistered study, on the `12B` where the screening once kept it: the
standard paired confirmation of the current confirmed candidate against the same candidate
plus `wired_fraction=0.6`, with the wired limit's effect on free memory, swap and MLX active
and peak bytes recorded around every child, and the existing swapout and free-memory gates
unchanged. A wired limit of `20.6 GB` on a `34.36 GB` machine is the thing being measured,
not a side condition, so the resource trace is a result and not a diagnostic.

**Kill.** If the interval against the unmodified candidate does not lie entirely below `1.0`,
or if the run cannot pass the resource gates it is measured under, the knob is removed from
`ironmule.tune.SEARCH` rather than left in it. A knob that survives confirmation but never
wins one does not belong in a search that feeds a profile.

### `B73` — A second Mac, which is what most of this now needs

**Mechanism.** Four separate entries have reached the same wall. `B71`'s `H1` cannot be
tested on one machine. `B72`'s hardware axis does not vary. `B69`'s confirmed geometry is
bound to one fingerprint and nobody knows whether it is a property of this chip or of Apple
GPUs. `B66`'s width step at `16` and its submission-boundary cost are the same. One machine
has produced everything it can.

**Test.** Run, on a Mac with a materially different memory system, exactly what already
exists and in this order: `B71`'s probe set, then a tune, then `B57`'s composition study, then
`B69`'s stack proof for `(4, 8)`. Nothing new is written for it. Every harness already fails
closed on an unknown fingerprint, so the second machine's results cannot contaminate this
one's.

**Kill.** If the second machine's vector and its stack results agree in sign with this one's,
`H1` survives its first real test and a third machine becomes worth arguing about. If they
disagree, the vector is a description of one machine and the entries built on it say so. Either
outcome is a result; not having a second machine is not.

**Sharpened by `B76` and `B77`.** The comparison target is no longer one confirmation. Fourteen
sessions on this M1 Max put `single_short` at `0.9647` with a between-session `SD` of `0.0069`,
so a second machine is measured against a distribution and a disagreement has a scale. `B77`
then found that what separates `B69`'s `0.8469` from those sessions is dispersion, not speed:
`B69`'s A/A control ran at a half width of `0.0848` against `B76`'s `SD` of `0.0069`, and its
candidate interval overlaps its own control. So the second machine reports its A/A dispersion
and its machine state **before** any magnitude it produces is quoted, and a magnitude whose A/A
arm fails the `B75` gate is recorded without being quoted at all.

### `B41` — Pad the reduction dimension so the fast quantised kernel runs

**Mechanism.** `B24S` measured that 99.3 to 99.8% of a decode step's GPU time is one
kernel family, the quantised matrix-vector product, and that MLX picks between two
variants of it. `affine_qmv_fast` runs only when the reduction dimension is a multiple
of 512. The three local Gemmas fall differently on that boundary, and the measured time
shares follow it exactly:

| model | hidden | intermediate | plain `qmv` | `qmv_fast` |
| :-- | --: | --: | --: | --: |
| 1B | `1152` (no) | `6912` (no) | `98.6%` | `0.8%` |
| 4B | `2560` (yes) | `10240` (yes) | `0.0%` | `99.6%` |
| 12B | `3840` (no) | `15360` (yes) | `74.1%` | `25.7%` |

**Price of the boundary (`B41K`).** A microbenchmark on the device, eight interleaved
blocks with alternating direction, `N=4096`, batch 1: `4096` reaches `362 GB/s` while
`3840`, `4032` and `4160` all sit at `306` to `310`. The jump is `17.6%` and it sits on
the boundary alone, not on size. Those absolute numbers are cache-assisted: at `N=4096`
the weights are `8.4 MB` and a chain rereads them inside the `48 MB` system level cache
(`B41C`). The comparison is unaffected, both arms ran under the same conditions, and the
model-level result below reads `7.2 GB` per token, far past any cache.

**The bandwidth ceiling this competes against (`B41C`).** Same kernel, `K=4096`, `N`
swept so the weight set crosses the cache, five interleaved blocks:

| weights | resident in | median | best |
| --: | :-- | --: | --: |
| `9.4 MB` | cache | `264.5 GB/s` | `315.8` |
| `18.9 MB` | cache | `358.0` | `367.2` |
| `37.7 MB` | cache | `406.2` | `409.2` |
| `75.5 MB` | memory | `319.9` | `324.6` |
| `151.0 MB` | memory | `334.1` | `338.5` |
| `302.0 MB` | memory | `342.2` | `343.7` |

`409 GB/s` is the highest this project has ever measured and it is a cache read. Out of
memory, where a model's weights actually live, the ceiling is `334` to `344 GB/s`. That
confirms the `324 GB/s` floor `B24R` borrowed from `E4`: the roofline it derived stands,
and a 12B step's device time is `80%` of a ceiling that is real.

**The padding is arithmetically exact.** Appending zero packed weights with zero scales
and zero biases makes `dequantize` return exactly `0.0` for the new columns, and the
input is zero there, so every added product is `0 * 0`. No existing byte is touched and
nothing is requantised. `tests/test_padded_qmv.py` asserts the zero.

**Result on a real model (`B41P`).** 12B, 241 padded linears, all `3840 -> 4096`, six
interleaved order-balanced blocks of 20 greedy steps:

| | median step | paired ratio |
| :-- | --: | --: |
| reference | `31.34 ms` | |
| padded | `30.29 ms` | `0.9657` |

Faster in six blocks of six, ratios `0.9485` to `0.9831`, and the two arms do not
overlap: reference `31.14` to `31.82 ms`, candidate `30.09` to `30.62`. **`3.4%` off a
12B decode step.** The 4B null control padded nothing, as every 4B shape is already
aligned, and measured `-0.44%` with a logit difference of exactly `0.0`: the harness
carries no bias of its own.

**What it costs, and the open gate.** Tokens were identical over 20 steps, but the
maximum absolute logit difference is `0.53`, and in the microbenchmark `2.1e-07` on a
128-row case. The added columns contribute nothing; the difference is the other kernel
reducing in a different order. Under `B2`'s rule, any nonzero logit difference makes
this a plan change, not a drop-in optimisation. It also costs `6.7%` more weight bytes
and one pad kernel per distinct input vector.

**Where it does not work.** 1B measured `-5.8%`, slower. Its `1152` pads to `1536`, a
`33%` byte increase that the `17.6%` bandwidth gain cannot repay, and its host share is
the largest of the three. The entry is size-dependent: it wins only where the distance
to the next boundary is small relative to the gain.

**Next.** Qualify it as an explicit opt-in plan against the exact-output contract, or
hold the padded weights so that the per-call pad disappears. Do not activate it while
the logit difference is unqualified.

### `B38` — Exact Gemma 12B core-profile activation/canary

**Mechanism.** B36 qualifies the non-mutating core profile only for the exact
Gemma 3 12B revision, prompt, host and full-hash/prefault protocol. A separate
activation/canary decision must preserve that fingerprint scope rather than
turning the exploratory result into automatic routing.

**Test.** Obtain explicit architecture approval first. Then canary only the
exact revision/quantisation/hardware identity, with fail-closed model/code/
environment/workload fingerprints, correctness, memory, swap, crash and
rollback gates. Keep activation opt-in and compare against the unchanged
baseline under a preregistered rollout budget.

**Kill.** Any identity drift, missing gate evidence, token/stop divergence,
resource regression, inconclusive canary or unapproved architecture change
blocks activation. No activation is allowed now.

### `B1` — Is width 4 still the ceiling at 27B?

**Mechanism.** `W=4` was fixed by `E2`, `E3` and `E14b`, all three measured on 4B.
The pathological `M=8` regime is a property of a matrix shape, and 27B's shapes are
not close (hidden `2560 -> 5376`, intermediate `10240 -> 21504`). `E4` showed larger
matrices reach much higher achieved bandwidth.

**Test.** Sweep 2, 4, 6, 8, 12 at 27B under the standard protocol; repeat the winner
three times; check token identity at every width.

**Kill.** Width 4 wins at 27B too. Then the ceiling belongs to the kernel rather than
the shape, `LIMITS.md` gains a sentence saying it was checked at two scales, and the
`M=8` story is settled. Full detail in [`SCALING.md`](https://github.com/Tobayko/IronMule-Research/blob/main/docs/SCALING.md).

### `B2` — Group the `lm_head`, and only the `lm_head`

**Mechanism.** Grouping deliberately never changes a tensor shape. But the four
sessions in a group each run `lm_head` separately at `M=1`, and `E3` measured
`1.307 ms` at width 1 against `1.126 ms` per token at width 4. One matmul is stacked;
the trunk, the cache and the per-session state stay untouched.

**Evidence against.** Rows of a matmul are arithmetically independent, but bit
identity across a shape change is not guaranteed — tiling can alter reduction order.
`E9` is the precedent: plans that differ by `4.31` logits still chose the same token,
and `E10` recorded that as luck rather than a guarantee.

**Test.** Stack only the `lm_head` input across the group. Compare logits bitwise
against the ungrouped run, not just the tokens, across at least the 756-request
corpus `E12` used.

**Kill.** Any nonzero logit difference. Then it is a plan change, not an optimisation,
and belongs behind an explicit plan the way `ReusableSessionPlan` is. Note the payoff
shrinks with model size: `lm_head` is roughly 16% of a 4B step and about 5% of a 27B
step, so this is the wrong direction for the scaling problem.

### `B3` — Unroll k decode steps into one compiled graph

**Mechanism.** `E5` measured `dispatch_us = 6.41` and ~510 kernels per decode step,
which is `3.3 ms` of host work per step — and it accounts for the whole gap between
the trunk's achieved `195 GB/s` and the `324 GB/s` ceiling `E4` measured. Compiling
four sequential steps into one graph submits once instead of four times.

**Evidence against, and it has already bitten once.** Greedy decoding is
data-dependent: step `n+1` needs step `n`'s token. Unrolling requires the argmax and the
cache write to live inside the compiled body, which `E0a` already did for one step. EOS
handling has to become a mask rather than a branch.

The predecessor project compiled decode subgraphs on a *growing* cache and got
**wrong tokens from position 2** (candidate 12), which invalidated a set of earlier
compile measurements along with it. Compiling on a *fixed-shape* cache then worked —
cycle 16, `0.9296`, identical tokens over 18 arm executions — and that is what this
runtime ships. Unrolling several steps is the next move along the same axis, and it is
the axis where a shape assumption has already produced silently wrong output once.

**Test.** Unroll 2 and 4 steps, paired A/B across fresh processes like `E5`, with
token identity as a gate.

**Kill.** No improvement beyond noise, or the compile cost per shape makes it a loss
for short answers.

**B3-U2 pilot (2026-08-28).** The fixed-shape two-step correctness pilot completed
all eight isolated processes and 240 measured requests with identical tokens and
final states, zero fallback, zero Swap delta and no relevant crash. This is not a
performance result. The parent did not persist a separate pre/post host-state record
for every child, so confirmation is blocked and the pilot is not retried or pooled.
Public evidence: `research/raw/B3-U2_public_summary_20260828.json`.

**B3a next gate.** Persist real parent pre/post host state and strictly validate
request IDs and checkpoint events in a newly authorised pilot. Kill on any missing
state/event, token or state mismatch, fallback, cache growth, resource failure or
incomplete pair. No confirmation or activation follows automatically.

### `B4` — Wire the weights against page pressure

**Mechanism.** 27B peaks at `16.78 GB` on a 32 GB machine that also runs an editor, a
browser and, in this project's case, a second agent. If the OS evicts weight pages
between steps, decode pays a page fault at 4 KB granularity. MLX exposes a wired
memory limit; nothing in this repo has ever set it.

**Test.** Standard protocol at 27B with and without a raised wired limit, under a
deliberately loaded machine, three runs each. Record swap activity, not just time.

**Kill.** No difference on an idle machine and no difference under load. Cheap enough
to be worth the certainty either way — and if it *is* the explanation for some of the
27B spread (`0.73pp`, twice 4B's), that matters for every other measurement here.

### `B5` — Fill the group on purpose

**Mechanism.** `LIMITS.md`: with 2–3 token answers the realised width fell to `1.83`
and the gain dropped below threshold in `E15` and `E16`. The executor never waits to
fill a group, by design. A bounded wait — a few milliseconds — would raise realised
width where it collapses.

**Evidence against.** Median latency already worsens by 26% to 54% under grouping.
This makes it worse on purpose, and `E15`'s workload is where it would apply.

**Test.** A wait bounded at 2, 5 and 10 ms on the `E15` short-answer workload. Report
realised width, throughput **and** the full latency distribution, not the median.

**Kill.** Throughput gain smaller than the added tail latency at every bound. Likely
outcome, and the entry exists so the trade is measured once instead of argued.

### `B6` — How does the `M=4` cost ratio move with model size?

**Mechanism.** Not an optimisation — a precondition for `B12`, `B13` and `B14`. At 4B,
`E3` measured a width-4 forward costing `2.545x` a width-1 forward for four tokens.
A larger model is more bandwidth bound and its weight traffic is identical at `M=4`,
so the ratio should *fall* toward 1.0 as the model grows. If it does, every
multi-token-per-forward idea gets cheaper exactly where the grouping gain got worse.

**Test.** `E3`'s isolated forward sweep repeated at 12B and 27B. Two hours of GPU time.

**Kill.** The ratio does not fall. Then `B13` and `B14` lose most of their predicted
advantage at scale and drop several tiers.

### `B26` — Qwen3.8 27B, to separate model size from model family

**Mechanism.** 4B, 12B and 27B are all Gemma 3, so size and family are fully
confounded and the falling gain in [`SCALING.md`](https://github.com/Tobayko/IronMule-Research/blob/main/docs/SCALING.md) has two live explanations.
`mlx-community/Qwen3.8-27B-4bit` is the cleanest available discriminator: the same
parameter count as the Gemma 3 27B already measured, at **4 bit, group size 64** —
identical quantisation, so the validity box changes in one dimension instead of three.

The shapes are close enough to compare and different enough to matter:

| | hidden | intermediate | layers | kv heads | vocab |
| :-- | --: | --: | --: | --: | --: |
| Gemma 3 27B | 5376 | 21504 | 62 | 16 | 262144 |
| Qwen3.8 27B | 5120 | 17408 | 64 | **4** | 248320 |

Near-identical depth and width, and **a quarter of the KV heads**. If the gain lands on
the Gemma line, the trend belongs to size and is worth preregistering. If it does not,
it belongs to architecture, and the four-fold difference in KV traffic is the first
place to look.

**Test.** The standard protocol — strict plan, 6 requests, 48 max tokens, three runs —
with no knob touched, exactly as X1 was run. Escalate the way the Gemma 27B run did: a
two-request probe first, then half size, then full. Report peak memory separately; this
checkpoint carries a vision tower the text path never uses, so it is not comparable with
Gemma's `16.78 GB` without saying so.

**Before starting.** The checkpoint is roughly 16 GB on disk and needs comparable
headroom in unified memory. The Gemma 3 27B weights were deleted to make room, which
costs nothing evidential — X1's nine raw 27B result files are in `research/raw/` — but
it does mean a re-measurement of the Gemma side is a 16 GB download away, not a command
away.

**Kill.** Nothing. Either answer closes a confound that currently limits every
conclusion in `SCALING.md`, which is why this sits in Tier 1 despite a guessed payoff of
zero percent.

**Compatibility result.** The separate X2 gate qualified the type-preserving hybrid
adapter and leaves this performance/family study open for its preregistered,
three-repeat measurements.

### `B30` — Widen Qwen grouped batch-1 groups to five and six

**Mechanism.** The simultaneous six-request Qwen workload currently forms width-4
plus width-2 independent fixed-state batch-1 groups. A Qwen-only scheduler ceiling
of 5/6 can submit more independent B=1 forwards before the existing barrier while
leaving fixed cache tensors, model calls and grouping semantics unchanged. Gemma
stays at width 4.

**Success gate.** On the strict six-request, 48-token workload, candidate versus
Interactive paired median executor/decode rate must be at least `1.10`, with no
regression versus current Throughput. Physical/visible tokens, stops, counts and
final `kv_hash` must match fresh Sequential and current Throughput controls; memory,
swap and fallback gates must pass.

**Test.** Run widths 4, 5 and 6 through the mandatory pre-timing gate, then the
standard balanced two-warmup/three-repeat pilot. Keep model calls at independent
B=1 and do not add native conversion, cache merging or compilation.

**Kill.** Any divergence, state/hash mismatch, fallback, memory/swap pressure,
candidate below `1.10`, regression versus current Throughput, or Gemma regression.

**B30 correctness/resource event (2026-08-27).** The real integration gate passed
at widths 4/5/6 with exact tokens, stops, counts and state hashes, zero fallbacks,
exit `0`, and `152.16 s` wall time. No token-ID artifact exists, so this is not a
product or performance claim. Swap rose from `505.75 MiB` to `2676.69 MiB`
(`+2170.94 MiB`); free-memory telemetry moved `82%` to `87%`, with no crash.
B30a is therefore not run until reboot/resource reset; all preregistered gates and
thresholds remain unchanged. Raw: `B30_correctness_gate_20260827.json`.

**B30a pilot result (2026-08-27).** The mandatory correctness gate again passed
at widths 4/5/6 with exact output/state and zero fallback. Warmup 0 produced
Interactive `15.8483`, Throughput `16.4353`, and candidate `16.4240` executor
tokens/s. The safety checkpoint then stopped the pilot: swap delta was
`314111427 B` and peak MLX memory `23882126950 B`. No measured repeat ran;
classification is `INCONCLUSIVE` and no performance claim is made. B30b is
required after reboot/resource reset; gates and thresholds remain unchanged.

---

## Tier 2 — structural. Weeks, and they change the shape of the runtime.

### `B8` — A decode loop that does not go through Python per operation

**Mechanism.** `6.41 µs` per dispatch is not a GPU number. ~510 kernels per step at
that rate is `3.3 ms` of host time per decode step, against a `~10.1 ms` trunk. A
loop written against MLX's C++ API, with no interpreter and no GIL between kernels,
should submit the same graph in a fraction of that.

**Evidence for.** `E14b`'s clearest result: arm B's host submission per request *rises*
from `6.24` to `9.16 ms` as the group grows, while completion wait collapses to
`1.27 ms`. **Host submission is the saturating resource**, not the GPU.

**The catch.** This is the entry that most directly shrinks the published gain. If
host submission stops being the bottleneck, there is much less left to overlap. Decide
which number is being reported before starting — see the warning at the top.

**Test.** Port one decode step, measure it standalone against the Python path first.
No integration until that microbenchmark says the premise is right.

**`B24` discharged this dependency on 2026-09-09.** The four-way wall-clock split
could not separate host from device cost; the Metal System Trace can. Measured on an
ungrouped batch-1 step, host time is `2.72 ms` at 1B, `3.27` at 4B and `5.50` at 12B,
against steps of `8.48`, `13.23` and `32.86 ms`. That is the ceiling for anything here:
at most `32% / 25% / 17%`, falling as the model grows. Encoders per step are `16 / 22 /
35`, but encoders are not dispatches, so the `~510 kernels` premise is still unmeasured.

**Kill.** Under `2x` improvement on the isolated step. Then `6.41 µs` is Metal's
enqueue cost rather than Python's, and `B9` becomes the only remaining route.

**Sharpened 2026-09-23, from an external proposal.** The hypothesis is not "a faster
language" but less host work around the same GPU operations: same kernels, same dtypes, same
weights, only the submission path changes. Two routes, both native:

- MLX `0.32.0` ships `mx.export_function` and `mx.import_function` (checked 2026-09-23). An
  exported compiled decode step could be replayed from C++ (`mlx::core::import_function`) or
  from Rust through `mlx-c`, without re-implementing the model. First gate: a Gemma decode
  step with its fixed KV state exports, imports and produces the same logit bits and KV bytes
  as the Python path. If it does not, the only route left is re-writing the model forward
  natively, which is a different and far larger entry.
- C++ and Rust are equal candidates; C++ sits closer to MLX and keeps a first prototype
  smaller. A Python extension is acceptable only at the outer boundary: one call per decode
  step, or per `k` steps, never a call back into Python per layer. A native loop that
  re-enters Python per layer does not test this hypothesis.

Measure host time with the `B24` trace protocol, not the wall clock alone:
`compiled_fixed_cache` already removes much per-step Python work (the `B25` caveat in
Tier 0), so what remains of the `2.72` to `5.50 ms` may be MLX's own encoding rather than
the interpreter. The `~510 kernels` premise is still unmeasured; a run with
`Shader Timeline: Enabled` is what would count them.

### `B9` — Record the decode step once, replay it every step

**Mechanism.** The decode step is a fixed sequence of kernels over fixed shapes with a
fixed cache — the exact case Metal's indirect command buffers exist for. Encode once,
replay per step, and the per-dispatch host cost goes away almost entirely.

**Evidence against.** MLX does not expose this today. It means patching MLX or
building alongside it, and it welds the runtime to one framework version — which the
fingerprint mechanism was built to police, not to encourage.

**Test.** Prototype outside IronMule: a hand-built ICB replaying one transformer
block, timed against the same block through MLX.

**`B24` discharged this dependency on 2026-09-09.** The four-way wall-clock split
could not separate host from device cost; the Metal System Trace can. Measured on an
ungrouped batch-1 step, host time is `2.72 ms` at 1B, `3.27` at 4B and `5.50` at 12B,
against steps of `8.48`, `13.23` and `32.86 ms`. That is the ceiling for anything here:
at most `32% / 25% / 17%`, falling as the model grows. Encoders per step are `16 / 22 /
35`, but encoders are not dispatches, so the `~510 kernels` premise is still unmeasured.

**Kill.** The replay is not meaningfully faster, or the shapes turn out not to be
stable enough across steps to reuse an encoding. Highest ceiling of anything in Tier 2
and the highest chance of being abandoned halfway.

### `B11` — Pipeline layers across the group instead of whole steps

**Mechanism.** Grouping currently submits four whole steps and waits once. Interleaving
at layer granularity — layer `i` of session A, then layer `i` of session B — keeps the
same shapes but gives the scheduler a finer-grained stream and lets one session's layer
`i+1` overlap another's layer `i` synchronisation.

**Evidence against.** `E14b` showed the gain already comes from overlap that MLX's
async submission achieves on its own. This may be re-implementing what the runtime
already does, one level up and slower.

**Test.** A microbenchmark on two sessions before any executor change.

**Kill.** No improvement over the current whole-step grouping at width 4. Likely, and
cheap to establish.

### `B12` — Jump over the `M=8` valley

**Mechanism.** `E3`'s own table, which nobody has acted on:

| width | full | ms/token |
| --: | --: | --: |
| 1 | 11.909 | 11.909 |
| 4 | 30.307 | **7.577** |
| 8 | 66.856 | 8.357 |
| 16 | 71.807 | **4.488** |

Width 16 is the best point in the entire sweep, by a wide margin. `M=8` is a valley,
not a ceiling — and every conclusion in this project stopped at the near side of it.

**Evidence against, and it is serious.** This requires true tensor batching, which
`E14b` measured producing a reproducible one-token divergence at `b = 8` — row 3,
index 6, `1437` against `1580`, in all four blocks (blocks, not OS processes — see
`E15` limitation `M2`). `E14b` also measured `C8` as
*worse* than `C4` on throughput. Correctness decides before speed does.

**Test.** `E14b`'s arm C extended to `b = 16` and `b = 32`, correctness first: if
divergence grows with batch size, the entry closes on that alone. Only if token
identity holds at 16 does the throughput number mean anything.

**Kill.** Divergence at 16 or beyond. Then `LIMITS.md`'s "no true tensor batching"
becomes permanent rather than provisional, and `E3`'s width-16 row is a curiosity
about kernels rather than a route.

---

## Tier 3 — algorithmic. Real speedups with a quality cost that must be bounded.

### `B13` — Speculative decoding with an actual draft model

**Mechanism.** `E0c` rejected speculation, but it drafted with prompt-lookup n-grams
and got `0.17` acceptance. A small model drafting for a large one is a different
proposition, and `E3` already measured the verification side: four tokens verified in
one `M=4` forward cost `2.545x` a single step, so break-even sits at about `2.6`
accepted tokens out of four.

**This has been tried once already and rejected.** The predecessor project ran a 1B
draft against a 4B target and recorded `0.560x` (candidate 11, and see the sign-convention
warning in Tier 0). What is different here is only the ratio: a draft costs roughly the
same either way, while the target it saves grows `6.75x` from 4B to 27B, and per `B6` the
`M=4` verification premium should shrink at the same time. **That is a reason to
re-measure, not a reason to assume the earlier result does not apply.** Read the cycle's
raw numbers first; if the rejection was about acceptance rather than about the target
being too small, none of this changes it.

**Why it points the right way.** The bigger the target model, the cheaper the draft is
in relative terms, and per `B6` the `M=4` verification ratio should fall as well. This
is the one entry whose payoff *grows* exactly where the grouping gain shrank. Drafting
`k=3` and verifying at `M=4` also lands on `E3`'s optimum by construction.

**Correctness.** Greedy verification is exact: a speculative step either produces the
token the target would have produced or falls back. That is a structural guarantee of
the same kind `E9` found for chunked prefill, not a statistical one.

**Test.** Measure acceptance first, on this project's own workloads, before any
integration. Acceptance is the whole entry.

**Kill.** Acceptance below `0.65` at `k=3`. Then the arithmetic never closes.

**Measured once on other hardware (`PERF1`, 2026-09-23).** On a Kaggle T4 a draft model
reached acceptance `0.61`, below this kill (`perf1-run3-1d84848f`). That closes the entry for
that CUDA cell only; Apple and a 27B target remain unmeasured.

### `B14` — A draft head trained on the target's own hidden state

**Mechanism.** The EAGLE/Medusa family: instead of a separate draft model, a small head
on the target's last hidden state proposes the next few tokens. No second set of
weights to hold, much higher acceptance than an independent draft, same exact greedy
verification.

**A shortcut worth checking first.** `mlx-community/Qwen3.8-27B-MTP-4bit` ships a
multi-token-prediction head already trained. If `B26` is run anyway, that variant makes
this entry testable without training anything — measure acceptance on it before deciding
whether the idea is worth building.

**Evidence against.** It needs training, which nothing in this project currently does,
and the head is model-specific — a new head per model, per quantisation. That fights
the fingerprint discipline directly.

**Test.** Only after `B13` shows the verification arithmetic closes. Sequence matters.

**Kill.** `B13` fails, or the head cannot be trained to acceptance meaningfully above
an independent draft's.

### `B16` — Lower or mixed precision

**Mechanism.** Decode is bandwidth bound. Fewer bits is directly fewer bytes: 3-bit
weights would be ~25% less traffic. Mixed precision — 3-bit for the large MLP
matrices, 4-bit for attention — is the usual compromise.

**Evidence against.** This changes the model, not the runtime, and every performance
number in this repo is scoped to `4 bit, group size 64` by the fingerprint. Quality
would need `E13`'s paired design, on a set that is not contaminated the way SQuAD is.

**Kill.** Quality loss beyond a preregistered bound. It should be preregistered
*before* the speed is measured, or the temptation to accept the loss is obvious.

### `B17` — Quantise the KV cache

**Mechanism.** Decode reads the KV cache as well as the weights. At 2048 context this
is small next to 13.5 GB of 27B weights; at 8192 it is not, and `LIMITS.md` puts the
hard refusal there.

**Test.** Measure the KV share of per-step traffic at 2048 and 8192 first. If it is
under 10%, close the entry without implementing anything.

**Kill.** KV traffic is a small share at every context this runtime allows.

### `B18` — Skip layers that do not change the answer

**Mechanism.** 27B has 62 layers. Early-exit and layer-skip work claims many decode
steps converge well before the last layer.

**Evidence against.** No exactness guarantee exists, unlike `B13` and `B15`. This is a
quality/speed trade dressed as an optimisation, and it interacts with the fixed cache:
a skipped layer still needs its KV written or the next step's attention is wrong.

**Kill.** Any implementation that cannot state what it costs on `E13`'s design.
Listed for completeness and deliberately ranked below everything exact.

### `B19` — Use Gemma's 5:1 local-to-global layer pattern

**Mechanism.** Gemma 3 alternates sliding-window local attention with global layers.
The local layers never need KV beyond 1024 tokens. If the fixed cache allocates full
capacity for every layer, five sixths of the layers are holding — and reading — KV
they cannot attend to.

**Evidence.** `E12` deliberately spanned the 1024 boundary and found reuse bit-exact
across it, so the boundary is understood. Whether the allocation exploits it is a
different question and one nobody has checked.

**Test.** Read what the fixed cache actually allocates per layer. Possibly a
five-minute answer with no code at all.

**Kill.** MLX already handles it. Quite likely, and worth the five minutes.

---

## Tier 4 — the wild ones

Written down because the cost of recording a bad idea is one paragraph, and because
two of these are only wild in effort, not in mechanism.

### `B20` — Put `lm_head` on the Neural Engine

**Mechanism.** The ANE sits completely idle during every measurement in this repo. It
is a second compute unit on the same unified memory. `lm_head` is one large,
self-contained matmul with no data dependency on the rest of the block until the end
of the step — the one piece of the model that could plausibly run somewhere else
*while the GPU works on the next step's trunk*.

**Evidence against.** Reachable only through CoreML, which means a converted model
graph, its own quantisation story, and a bridge on the critical path. The ANE's own
latency for a `262144 x 2560` matmul is unknown and could easily exceed the GPU's
`1.3 ms`.

**Test.** Time the isolated matmul through CoreML on the ANE before anything else. One
number decides the entry.

**Kill.** ANE slower than `1.3 ms`, or the transfer cost exceeds the overlap gained.

### `B21` — Small projections on the CPU while the GPU does the big ones

**Mechanism.** `E4`'s most underused finding: `k_proj` and `v_proj` are `1.4 MB` and
achieve `103.7 GB/s`, a third of what the same hardware reaches on large matrices. At
`M=1` these are matrix-vector products — the case Apple's AMX units are built for —
and `E2` measured a `~0.49 ms` fixed cost just to dispatch to the GPU at all.

**Evidence against.** Synchronising CPU and GPU per layer would cost far more than
`1.4 MB` of traffic, so this only works if a whole set of small operations moves
together and stays there.

**Test.** Time a `1.4 MB` 4-bit matvec on the CPU against `E4`'s GPU number. Again,
one microbenchmark decides it.

**Kill.** CPU not faster on the isolated operation, which is the likely result.

### `B22` — Two processes, one GPU

**Mechanism.** Not a speedup — a control. If two independent processes each running
batch-1 achieve overlap comparable to grouping inside one process, then the gain is
the GPU scheduler's and the executor is taking credit for the operating system's work.

**Evidence for running it anyway.** `E16` already runs across forty OS processes and
found the gain replicates *within* each. This asks the complementary question and it
is the cheapest possible attack on the project's central claim.

**Test.** Two processes at 4B (`2.78 GB` each, comfortable), same total request load,
against one process grouping at width 2.

**Kill.** Nothing — either answer is worth knowing, and a project that publishes
negative results should be willing to aim one at itself.

### `B23` — A weight layout built for `M=1`

**Mechanism.** Quantised weights are stored for general matmul. At `M=1` every kernel
reads a full tile to use one row of activations. A layout swizzled for matrix-vector
access might reach closer to `E4`'s `324 GB/s` on the small matrices that currently
manage `104`.

**Evidence against.** It means a custom Metal kernel competing with a tuned library
implementation, and a weight conversion step at load time.

**Kill.** A hand-written matvec kernel does not beat `quantized_matmul` at `M=1` on the
small shapes. Expect it not to.

**Sharpened 2026-09-23.** Three things changed since this was written. `B24R` prices the
prize: device slack above the bandwidth floor is `15.5%` of a 4B step and `16.1%` of a 12B
step, but `41.9%` at 1B, so a layout that helps small matrices has most room on the smallest
model. `B42` showed a hand-written kernel can beat the library at one shape without a new
layout (`7` to `10%` kernel, `1.0` to `1.3%` model), and `B66`/`B69` found a byte-identical
`(4, 8)` geometry for `K=3840, N=15360`. A layout candidate is measured against the best of
those at the same shape, not against `quantized_matmul`.

The layout is a reversible repacking of the existing quantised bits, scales and biases,
never a requantisation: unpacking returns the original bytes, and the reduction order stays
the library's so identity can be proven the way `B42` proved it. The artifact is bound to
model revision, quantisation, kernel version and layout version, and fails closed on any
mismatch. On a 32 GB machine it replaces one projection family, not a second copy of the
model. Repacking pays only after `repack cost / saving per token` tokens, stated per workload
class. Hidden copies count as cost: `mx.fast.metal_kernel(..., ensure_row_contiguous=True)`
copies a non-contiguous input.

**Kill, added.** No gain over the best existing kernel for the same shape, or a break-even
beyond the token counts the workload classes actually produce.

---

## How these compose

`B13` drafts three tokens and verifies at `M=4` — which is `E3`'s measured optimum, and
the same width the executor already groups at. Grouped batch-1 and speculation are not
alternatives: four grouped sessions each verifying a four-token draft is the same
scheduler with a different unit of work.

`B8` and `B9` shrink the published gain while making the product faster. `B13` and
`B14` raise both. If the goal is a number that goes up *and* means something, the
speculative branch is the honest one to push.

`B24` and `B24R` now put numbers on that sentence. Every host entry competes for
`24.7%` of a 4B step and `16.7%` of a 12B one, and that share falls as the model grows.
Every kernel entry competes for the `15.5%` and `16.1%` of device slack above a
bandwidth floor the device already reaches four fifths of. The entries that read fewer
bytes per token are the only ones not bounded by either number, which puts `B13`, `B14`
and `B19` ahead of `B8`, `B9` and `B10` on measured grounds rather than on taste.

## Rules that apply to every entry here

**Preregister before measuring anything that could ship.** `X1` is exploratory because
it was not preregistered, and it stays labelled that way. An entry graduating from this
file to the ledger needs a sealed preregistration, a threshold fixed in advance, and
repeats across separate OS processes, the way `E16` was run.

**Every resolved gain counts (maintainer decision, 2026-10-29).** No entry opened from this date sets a minimum
effect size, and the effort or run time a gain costs is not a reason to drop it. A gain counts where it shows, in
one traffic state, one context length or one card, as long as the measurement tells it apart from noise: its
confidence bound clears the A/A noise or zero, as the entry fixes before data. Small gains need more repeats, not
a lower bar. Old entries that died only on a fixed minimum (Tier 0's readback 8 at `0.9581` against 5 %, `B29c`
against `1.10`, `B48` against 5 %) may be re-opened on this rule, with a new measurement and its own noise; their
recorded data is not re-read to a new verdict.

**Measure optimisations in combination.** Two optimisations that each help can share one resource and return less
together (host time, the same memory reads), or enable each other. An entry that ships more than one measures the
combination itself, not the product of the single gains, and a factorial design is preferred where the knobs are
few (HW3).

**Correctness gates come first and they are gates, not tie-breakers.** `B12`'s ceiling
is the highest in Tier 2 and it is still ranked behind entries with a third of the
payoff, because `E14b` found a reproducible token divergence and that is not a rounding
error to be negotiated with.

**A finished entry leaves this file.** Delete it from its tier the moment it is
answered — this is a list of open work, and an entry that survives its own answer makes
the list lie about how much is left. The result does not disappear, it moves:

- **It worked and shipped** -> a full entry in [`research/LEDGER.md`](https://github.com/Tobayko/IronMule-Research/blob/main/research/LEDGER.md),
  and a line in [`LIMITS.md`](LIMITS.md) if it changed the validity domain.
- **It failed, or it was measured and rejected** -> one line in Tier 0 above, with the
  number and the experiment ID. Tier 0 is the only part of this file that grows.
- **It turned out to be someone else's problem** -> delete it and say so in the commit.

**Add what you learn while you are here.** Anything discovered mid-experiment that
could make the runtime faster belongs in this file the same day, even half-formed,
even if it is probably wrong — with a mechanism and a kill criterion, because an idea
without a kill criterion is a wish. The same goes for a route that turned out to be
blocked: write it into Tier 0 so nobody walks into it twice. This file is only worth
reading if it is the current state of what is known, not a snapshot of one afternoon.
