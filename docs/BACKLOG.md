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

**What "gain" means (owner, 2026-11-03).** Tokens per second come first. Waiting time counts too, but less, and
the time to the first token must not get worse. AG1 and AG2 so far judge a schedule by each request's time in
the system plus its time to the first token. Ordering lowers that, but at a fixed capacity it moves no tokens per
second: the same work runs on the same engine. Tokens per second move only through what the engine can run at
once (width, true batching) and how fast its kernels are (AG3).

**The agent sets up what was found by hand (owner, 2026-11-03).** "Things we already found out, the AI should
learn and set on its own." The settings found by hand runs so far are the numeric plan (`native` about 6x on 4-bit
checkpoints of an emulating GPU), the stacked engine (H10: 48 % cheaper), the width (w8 on the dense plan, w2 on
the tuned engine), the order and the class-A knobs. The owner was asked whether the agent may also choose plans
that change answers slightly. The answer was **yes, always**: under `--agent on`, a plan that passes the agent's
own on-machine gate may serve without a further switch. The gate is the numeric plans' perplexity gate against
the reference path (bound 1.005), the plan's load probe, and a speed check. This replaces "it changes no answer"
for plan choice inside agent mode (AG4). It is never silent: the chosen plan, its gate and its reason are in
`/health` and the agent's status. The agent stays opt-in. Without `--agent on`, the caller's plan is kept as
before.

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
- **S, the loop in simulation (`training/ag1_online.py`, local, no GPU).** The agent unchanged against a simulated
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
  the pass rule amended from LD1-V's to KP1's, the rule qualification uses.)* `training/rl2_headroom.py --plan ag1`,
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
  - **Result (2026-11-01, kernel `tobayko/ironmule-rl2-headroom-ag1` v1 at `3cb1e8f`, about 2.2 session hours):
    nothing qualified online, so there is no hardware gain and no comparison.** The run is valid as far as it went:
    all 36 online traces completed, 0 answers mismatched, and the analysis found no problems.
    - The agent planned one candidate, the best fixed schedule (`c63bf219337e`), and explored twice.
    - Its 36 traces gave 17 replay triplets of that candidate. At 17 triplets the candidate's gain was +3.5 % (95 %
      -6.5 to +15.5 %) against 5.2 % replay noise. The looks at 6 and 12 were inconclusive, and the run never
      reached the look at 18.
    - The reference served throughout. The world model fit 1,138 groups at widths 1-8 to a median relative error of
      4.0 %, and never reopened.
    - By the Kill rule, `--agent` stays opt-in and experimental.
    - *Descriptive, not part of the verdict.* V-CPU and V both gave about 0.47 triplets per trace. On the T4, one
      triplet's noise (5.2 %) is wider than the candidate's point gain. Enough triplets to resolve a few-percent
      gain needs either far more idle replay per trace or a lower-noise replay, such as pairing at the group level
      instead of the window. Either is a mechanism change, to be fixed here before any rerun.
    - Ledger entry owed.
- **V2, the gain on hardware with enough triplets (fixed 2026-11-01, after V's result and before any of its own
  data).** This protocol is shaped by V's data, and is labelled so. V and V-CPU each gave one replay window per
  16-request trace. On the T4 a triplet's gain spread about 23 %, which you can read back from the 95 % interval
  at 17 triplets, so the looks up to 24 could not resolve a gain of a few percent. Two things change:
  - **The agent's judge** (`agent.py`). The looks move from (6, 12, 18, 24) to (12, 24, 48, 96), and the one-sided
    alpha spent at them from (0.1, 0.2, 0.4, 1.8 %) to (0.05, 0.2, 0.6, 1.65 %), 2.5 % in all. `MAX_TRIPLETS` is
    96, and the futility rule and the audit's looks follow. Nothing else in the agent changes.
  - **The harness**, as `--profile t4b`. Each online trace has 64 requests, so four replay windows. There are 24
    online traces on fresh seeds 195-218, with 480 s idle after each so that four triplets fit. The held-out
    comparison uses 4 traces per state on seeds 219-222, also of 64 requests. Everything else is V's: model,
    engine, one worker, reference `w8/fifo`, states, cost, decisions (1) and (2), and validity.
  - **Three tiers (maintainer request, 2026-11-01).** Each held-out block runs the default (`w8/fifo`) twice as
    its A/A pair, plus the best fixed schedule earlier runs found (RL6's `w2/sjf`, through the product's dispatch
    hooks) and the agent, in rotated order. The fixed tier gives no answer of its own, and its tokens join the
    validity check. Recorded, not decided:
    - fixed over the default;
    - the agent over the default;
    - the agent over fixed.

    Any gain of the agent over the fixed tier is what RL added. If the agent qualified nothing online, the tiers
    are still measured, and its tier is the default schedule it served. The verdict stays "nothing qualified
    online".
  - **Gate:** S and S2 are rerun at the commit, and must pass (a)-(c) and (a')-(c') as before.
    - Passed: S run 9 and S2 run 5, local.
    - S: at trace 50, every world-model parameter is within 1.1 %. At trace 50 `w8/sjf` qualified (+6.6 %, 95 %
      3.7-9.6 %, floor 0.8 %), and two later candidates were rejected. Learned windows cost 6.2 % less than the
      reference.
    - S2: 50 traces after the change, every parameter is within 2.9 %. At trace 182, 32 traces after the change,
      `w1/sjf` qualified (+44.7 %, 95 % 37.2-50.7 %), and its audit confirmed it (+53.3 %). After the change,
      windows cost 43.0 % less than the reference.
  - **Cost:** about 8.5 session hours on one Kaggle T4, within the 12-hour session limit.
  - **Kill and ship:** as V. A pass is the agent's gain measured on real hardware, bounded to this setup. It
    ships only opt-in.
  - **Result (2026-11-02, kernel `tobayko/ironmule-rl2-headroom-ag1-t4b` v1 at `e18cd96`, 9.75 session hours):
    invalid, and nothing qualified online.**
    - **Why invalid.** The fixed validity rule says every held-out request completes. In `bursts` and `spike`, 10 to
      27 of a trace's 64 requests hit the 120 s request deadline, in every arm, on all four held-out traces. Sixty-four
      requests per trace overload the T4 in those states. A spike puts all 64 arrivals inside one second; V's 16 never
      timed out. Sixteen requests differ across arms, all of them only in whether they finished before the deadline.
      Every request that completed in all arms answered with the same tokens.
    - **Online.** 24 traces, 617 groups, 3 plans, 0 explorations, 36 triplets, 0 qualified, 0 rejected; the reference
      served throughout. The world model reached a 2.1 % median relative error over 1,684 groups and reopened 3
      times. At arrival 192 (trace 198) the planner proposed `w4/sjf` (`1c205576ee02`, model gain +3.0 %). Not the
      RL6 schedule `w2/sjf`: its model priced that at 17.20 against `w4/sjf`'s 16.52 and the reference's 17.06.
    - At the look at 12 triplets the corrected bound was below the floor. At 24 it did not pass either: the bound
      at the look's 0.2 % is not in the report, which shows the bound at 1.65 % between looks. At the last trace,
      36 triplets gave +3.0 % (95 % +0.7 to +5.6 %, floor 0.03 %, noise 3.0 %). The run ended before the look at
      48.
    - **Three tiers**, *descriptive only, since the run is invalid*. They are bootstrap 95 % intervals over 4 traces
      per state with the states weighted equally, and the A/A control is +0.2 % (95 % -0.3 to +0.6 %):
      - fixed `w2/sjf` over the default `w8/fifo`: -2.7 % (95 % -3.9 to -1.7 %);
      - the agent over the default: -0.03 % (95 % -0.4 to +0.3 %). The agent served the default, as the protocol
        says when nothing qualified;
      - the agent over fixed: +2.6 % (95 % +1.6 to +3.8 %). That is the default against `w2/sjf`, not anything RL
        added.
    - *Descriptive.* The agent made 1.5 triplets per trace, not the four the protocol assumed. A replay arm of an
      overloaded window takes about 90 s, so a triplet takes about 270 s, and 480 s of idle holds one or two. The
      triplets were about three times less noisy than V's: the 95 % width at 17 triplets was 6.7 points, against
      V's 22.
    - By the Kill rule, `--agent` stays opt-in and experimental. Ledger entry owed.
- **V3, V2 with completable traces and room for the look at 48 (fixed 2026-11-02, after V2's result and before any
  of its own data).** Shaped by V2's data, and labelled so. The agent does not change: `agent.py` is the code that
  passed V2's gate (S run 9, S2 run 5). The harness runs as `--profile t4c`, which is V2's `t4b` with four changes:
  - **The request deadline is 900 s**, through the product's own `request_timeout_s`, for the whole run. A spike of
    64 requests takes about 250 to 300 s to drain on the T4, so every request can complete. Validity, the cost and
    its failure penalty are unchanged, priced at the new deadline. The agent's replays already ran without a
    deadline failure in V2.
  - **1,200 s idle after each online trace**, so the four windows of a trace fit at about 270 s a triplet.
  - **At most 16 online traces on fresh seeds 223-238.** The online phase ends after the trace in which the agent
    qualified a schedule, since the held-out comparison needs nothing more from it.
  - **Held-out: 3 traces per state on fresh seeds 239-241.** Each block runs the same four arms as V2: the default
    twice, fixed `w2/sjf` and the agent, in rotated order.
  - Everything else is V2's: model, engine, worker, states, cost, decisions (1) and (2), validity, the three
    recorded tiers and "nothing qualified online".
  - **Gate:** the harness tests and the CPU smoke of `--profile t4c` pass. The agent is unchanged, so S and S2
    need no rerun.
    - Passed 2026-11-02, before any V3 data. On this container the T4 profile's worker (`current_engine`) did not
      start within its startup deadline, so the smoke ran `t4c` with V-CPU's worker (reference engine,
      dequantized) and SmolLM2-135M.
    - Every step ran: deadline 900 s, warmup, calibration, three online traces, then one held-out block per state
      with all four arms. All 36 requests completed and none differed across arms. The smoke reads "invalid" only
      because it has one held-out trace and the rule needs two.
  - **Cost:** at most about 10.5 session hours on one Kaggle T4. That is 16 traces of about 320 s serving and
    1,200 s idle (6.8 h), held-out blocks at V2's rate of about 330 s a cell (3.3 h) and setup. The 12-hour limit
    is the binding constraint, so there is no slack for more traces.
  - **Kill and ship:** as V. A pass is the agent's gain measured on real hardware, bounded to this setup: the T4,
    Qwen3-1.7B bf16, these states and this cost. It ships only opt-in.
  - **Result (2026-11-02, kernel `tobayko/ironmule-rl2-headroom-ag1-t4c` v1 at `2762ef8`, 10.3 session hours):
    valid, and nothing qualified online.** Every held-out request completed in every arm, 0 answers mismatched,
    3 complete traces per state, no problems.
    - **Online.** All 16 traces ran (no early stop): 481 groups, 2 plans, 0 explorations, 41 triplets, 0 qualified,
      0 rejected; the reference served throughout. The world model reached a 4.6 % median relative error over 1,551
      groups and reopened 4 times. The first plan (32 arrivals) proposed nothing. The second (108 arrivals)
      proposed the best fixed schedule, here `w2/sjf` (`c63bf219337e`, model gain +2.9 %; model costs `w2/sjf`
      7.18, `w4/sjf` 7.20, `w8/fifo` 7.41).
    - Neither look reached passed: at 24 triplets the bound at the look's 0.2 % was -0.1 %, below the floor of
      0.1 %. At the last trace, 41 triplets gave the candidate +6.8 % over the reference (95 % +3.6 to +9.7 %, floor
      0.1 %, noise 0.7 %). The run ended 7 triplets before the look at 48.
    - **Decisions.** (1) fails: +0.1 % (95 % +0.03 to +0.2 %), with the bound below the A/A floor of 0.1 %. The
      agent served the default, so this is an A/A difference. (2) passes. A/A control +0.1 % (95 % -0.06 to +0.2 %),
      noise 1.2 %.
    - **Three tiers** (recorded, not decided; 3 traces per state, states weighted equally):
      - fixed `w2/sjf` over the default `w8/fifo`: -6.5 % (95 % -7.7 to -4.7 %);
      - the agent over the default: +0.1 % (95 % +0.03 to +0.2 %). It served the default;
      - the agent over fixed: +6.3 % (95 % +4.5 to +7.4 %). That is the default against `w2/sjf`, not anything RL
        added.
    - *Descriptive, the finding that matters:* the agent's replay judge and the served comparison disagree in
      sign. On replay, `w2/sjf` beat `w8/fifo` by +6.8 %. Served end to end on held-out traffic, it lost by 6.5 %.
      Per state, the mean request cost was: quiet even (2.87-3.07 s in every arm); bursts 22.8 / 108.4 / 68.0 s
      under `w2/sjf` against 22.5 / 92.0 / 60.6 s under the default; spike 214.5 / 275.3 / 231.0 s against
      203.9 / 263.9 / 219.9 s. The world model mispriced it the same way (+3.1 % in the model).
    - Had the run reached the look at 48, the candidate would probably have qualified, and the agent would have
      served a schedule that costs more on the traffic it was proposed for. Decision (2) would then have been
      the guard. The replay is what the planner's proposal is checked against, so this is a defect in the
      mechanism (step 3), not in the noise.
    - Candidate causes, not tested:
      - filler prompts instead of the recorded prompt lengths' content;
      - output limits capped at 512;
      - idle stretches skipped;
      - windows of 16 arrivals cut from 64-request bursts, so the queue a schedule leaves behind for the next window
        is never paid for.
    - Fixing it is a mechanism change. It needs its own entry before any rerun.
    - **Diagnosis (2026-11-02, post hoc, descriptive; it changes no verdict above).** The script
      `v3_window_sim.py`, not committed, rebuilds the held-out traces from their seeds with each request's
      measured prompt and output tokens. It prices them with the world model the agent ended V3 with. Gains are
      `w2/sjf` against `w8/fifo`, as the mean over the 9 traces:

      | Priced | Gain |
      | --- | --- |
      | Served | -5.1 % |
      | Model, on whole 64-request traces | -4.8 % |
      | Model, on the same traces cut into 16-arrival windows | +3.5 % |
      | Model, 16-arrival windows with outputs capped at 512 | +3.5 % (the cap is irrelevant here) |

      Per trace, the whole-trace model tracks the served result: bursts 240 -16.2 % against -17.8 %, spike 239
      -4.9 % against -5.2 %. The 16-arrival window turns spike from about -5 % to +6 to +14 %.

      So the world model was right, and the window was wrong. Both the planner (`resample`, windows of `TRACE`)
      and the replay judge (`_trace`, windows of `TRACE`) score a schedule on 16 of a burst's 64 queued
      requests. That leaves out the backlog a low-throughput schedule builds up for the requests behind them,
      which is where `w2/sjf` loses.

      On whole traces the model ranks the fixed schedules by gain over `w8/fifo` (quiet / bursts / spike /
      equal weight):

      | Schedule | Quiet | Bursts | Spike | Equal weight |
      | --- | --- | --- | --- | --- |
      | `w8/sjf` | 0.0 % | +0.3 % | +1.1 % | +0.5 % |
      | `w4/sjf` | 0.0 % | -5.2 % | -1.5 % | -2.2 % |
      | `w2/sjf` | 0.0 % | -9.7 % | -4.5 % | -4.8 % |

      The scheduling headroom over the default on this setup is therefore about half a percent, which matches
      RL8's 0.1-1 %. The fix is entry V4 below.
    - By the Kill rule, `--agent` stays opt-in and experimental. Ledger entry owed.
- **V4, busy-period windows (fixed 2026-11-02, after V3's result and its post hoc diagnosis, before any V4 data).**
  Shaped by V3's data, and labelled so.
  - **Mechanism change** (`agent.py`, commit `6b2d661` and the covariance bound below):
    - The planner's windows (`resample`) and the replay judge's windows (`_trace`) start at an arrival that finds
      the server empty, under the reference schedule in the world model.
    - Each window runs to the next such arrival: at least 16 arrivals, at most 128.
    - A replay window stays fixed in the triplet's state until its triplet ends.
    - Exploration keeps its 16 arrivals.
    - Everything else in the agent is unchanged: looks, alphas, floor, veto and decisions.
  - **Covariance bound** (`_RLS.update`): each parameter's variance stays at or below its starting spread. S2 run 6,
    the first gate run with busy windows, failed (a'): at trace 200 the prefill term read -11.8 % and the per-group
    term +0.021 s against a truth of 0, while their sum was within 2.3 %. Groups of one cannot separate the two
    terms. Forgetting regrew the per-group term's variance up to the whole trace bound, so the split wandered.
  - **Harness, `--profile t4d`:** V3's `t4c` with these changes:
    - at most 12 online traces on fresh seeds 242-253, with 800 s of idle after each;
    - 2 held-out traces per state on fresh seeds 254-255;
    - the fixed tier is `w8/sjf`, the best fixed schedule on whole traces in V3's world model, instead of
      `w2/sjf`.

    Everything else is V3's: model, engine, worker, the 900 s deadline, states, cost, decisions (1) and (2),
    validity (two complete traces per state suffice), the three recorded tiers, early stop on qualification and
    "nothing qualified online".
  - **Gate:**
    - the agent and harness tests pass;
    - S passes (a) and (b);
    - S2 passes (a') and (c').

    All three run on the commit the kernel pins.
    - **Passed 2026-11-02** (S run 11, S2 run 7, local, on the tree committed with this entry). S2 run 6 failed
      (a'), as above.
    - S: at trace 50, every world-model parameter is within 1.2 %. At trace 50 `w8/sjf` qualified (+6.6 %, floor
      0.8 %). Later, two learned candidates were rejected.
    - S2: at trace 200, 50 traces after the change, every parameter is within 0.8 %, and the per-group term reads
      0.0004 s. At trace 207 a learned candidate qualified (+2.9 %). At trace 223, 73 traces after the change,
      `w1/sjf` qualified (+42.3 %, floor 1.1 %). After the change, windows cost 31.5 % less than the reference.
    - *Descriptive:* over traces 200-299 the largest single parameter error was 12.2 %, so the split still moves
      at times while groups of one serve.
  - **Cost:** at most about 6.2 session hours on one Kaggle T4. That is about 0.3 h of setup, 12 traces of about
    300 s serving and 800 s idle (3.7 h), and 24 held-out cells at V3's rate of about 330 s (2.2 h). It fits the
    7.3 h of weekly quota left after V3.
  - **What V3 predicts, descriptively:**
    - In the model, `w8/sjf` gains about +0.5 % over `w8/fifo` on whole traces (+1.1 % in spike), and every
      narrower schedule loses.
    - Replay windows are about 16 arrivals in quiet traffic and up to the whole 64 in bursts and spikes, about 2
      triplets a trace, so the look at 24 is reachable only near the end of the 12 traces.
    - A gain this small may not resolve. If the agent qualifies `w8/sjf`, its tier and the fixed tier are the same
      schedule. That shows the agent finding the best fixed schedule on its own, not RL adding to it.
  - **Kill and ship:** as V. A pass is the agent's gain measured on real hardware, bounded to this setup: the T4,
    Qwen3-1.7B bf16, these states and this cost. It ships only opt-in.
  - **Result (2026-11-03, kernel `tobayko/ironmule-rl2-headroom-ag1-t4d` v1 at `a7dc097`, 6.1 session hours):
    valid, and nothing qualified online.** Every held-out request completed in every arm, 0 answers mismatched,
    2 complete traces per state, no problems.
    - **Online.** All 12 traces ran (no early stop): 324 groups, 2 plans, 1 exploration, 9 triplets, 0 qualified,
      0 rejected; the reference served throughout. The world model reached a 1.5 % median relative error over 813
      groups and reopened 6 times. The first plan (32 arrivals) proposed nothing. The second (128 arrivals)
      proposed the best fixed schedule, here `w8/sjf` (`5aaabf717ed4`, model gain +2.5 %; model costs `w8/sjf`
      16.93, `w8/fifo` 17.39, `w4/sjf` 17.83, `w2/sjf` 19.46).
    - No look was reached: the run ended with 9 triplets, 3 before the look at 12. At that point the candidate read
      +1.4 % over the reference (95 % +0.7 to +2.1 %, bound 0.7 %, floor 0.03 %, noise 2.2 %), verdict open.
    - **Decisions.** (1) fails: -1.5 % (95 % -2.9 to -0.03 %). The agent served the default, so this is an A/A
      difference. (2) passes (quiet -0.1 %, bursts -4.0 %, spike -0.6 %; bursts' interval reaches -7.1 %, within
      the rule because the agent served the default). A/A control -0.8 % (95 % -1.7 to +0.2 %), noise 9.9 %.
    - **Three tiers** (recorded, not decided; 2 traces per state, states weighted equally):
      - fixed `w8/sjf` over the default `w8/fifo`: +0.9 % (95 % +0.6 to +1.2 %);
      - the agent over the default: -1.5 % (95 % -2.9 to -0.03 %). It served the default;
      - the agent over fixed: -2.4 % (95 % -3.5 to -1.2 %). That is the default against `w8/sjf`, not anything RL
        took away.
    - *Descriptive.* V3's defect is gone in this run: replay (+1.4 % at 9 triplets) and the served fixed tier
      (+0.9 %) agree in sign and roughly in size, and the model's +0.5 % prediction on whole traces is in the same
      range. The run was short of triplets, not of signal. Busy windows in bursts and spike span most of a
      64-request trace, so a triplet replays about 3 windows of 100-250 s each, and 800 s of idle held 0.75
      triplets a trace instead of the predicted 2.
    - *Descriptive.* The agent arm's bursts deficit is noise between two runs of the same schedule: in bursts block
      4 the two reference arms read 48.3 and 52.6 s mean request cost and the agent, also serving the reference,
      53.8 s.
    - By the Kill rule, `--agent` stays opt-in and experimental. Ledger entry owed.
- **V5-CPU, V4's agent on a Kaggle CPU session (fixed 2026-11-03, after V4's result and before any of its own
  data).** V4 left about 1.2 of the week's GPU hours, too few for a T4 rerun with enough triplets. A Kaggle CPU
  session spends no GPU quota and, unlike this cloud container (whose restart lost V-CPU-2), runs to its end.
  - **Agent:** `agent.py` exactly as V4 ran it, which passed S run 11 and S2 run 7. No mechanism change.
  - **Harness, `--profile cpu3`:** V-CPU's `cpu` (SmolLM2-135M 8 bit rev `0f0d9b8`, the reference engine with
    weights dequantized to float32, capacity 1, outputs of 4 or 32 tokens, short prompts, 1,800 s deadline, 360 s of
    idle after each online trace) with these changes:
    - V4's agent and V3's early stop: at most 30 online traces on fresh seeds 256-285, ending after the trace in
      which the agent qualified a schedule;
    - V2's three tiers on 4 held-out traces per state, fresh seeds 286-289: the default (twice), the fixed `w1/sjf`
      and the agent, in rotated order. At capacity 1 the order is the only decision, so `w1/sjf` is the best fixed
      schedule the agent's action space holds;
    - the kernel is a Kaggle CPU session that installs `ironmule[cpu]` (`write_kernel(..., cpu=True)`).
  - Everything else is V's and V3's: states, cost, decisions (1) and (2), validity (two complete traces per state
    suffice), the recorded tiers and "nothing qualified online".
  - **Gate:**
    - the harness tests pass;
    - a CPU smoke of `--profile cpu3` passes, both locally and as a Kaggle CPU kernel. The Kaggle smoke checks the
      install and the session, and measures nothing.
    - **Passed 2026-11-03, before any V5-CPU data**, at `a608921`, both smokes with SmolLM2-135M:
      - harness tests green;
      - local: every step ran, all 36 held-out requests completed and none differed across the four arms;
      - Kaggle CPU kernel `tobayko/ironmule-rl2-headroom-ag1-cpu3-smoke` v1: x86_64, 4 CPUs, Python 3.13, mlx 0.32.4,
        35 minutes, the same outcome.
      - Both read "invalid" only because a smoke has one held-out trace.
      - *Cost check:* the Kaggle CPU is about 2x slower than this container today (about 7 s against 3.5 s per
        smoke request, calibrated capacity 0.13 against 0.27 requests a second). Priced from the smoke rather than
        from V-CPU-2, the run takes about 30 x 560 s (4.7 h) online plus 48 held-out cells of about 180 s (2.4 h)
        plus setup, about 7.5 h. That is inside the 12 h session.
  - **Cost:** no GPU hours. V-CPU-2 ran about 580 s of this container's CPU per online trace, and V-CPU about 50 s
    per held-out cell. So at most about 30 x 580 s (4.8 h) plus 48 cells (0.7 h) plus setup (0.3 h), about 6 h.
    That is inside a Kaggle session's 12 h if its CPU is no more than about 2x slower than this container's.
  - **What earlier runs predict, descriptively:**
    - V-CPU and V-CPU-2 planned the same candidate, `w1/sjf`.
    - V-CPU-2's looks at 6 and 12 triplets were inconclusive. So the order-only gain on this traffic may be small
      against the noise, and the run may again end before a look passes.
    - If the agent qualifies `w1/sjf`, its tier and the fixed tier are the same schedule. A pass then shows the
      agent finding, proving and serving that schedule on its own, not RL adding to it.
  - **Kill and ship:** as V. A pass is the agent's gain measured on real hardware, bounded to that Kaggle CPU, this
    model, this traffic and this cost; it says nothing about a GPU. It ships only opt-in.
  - **Result (2026-11-03, kernel `tobayko/ironmule-rl2-headroom-ag1-cpu3` at `451e282`, no GPU hours): the
    session was cut by Kaggle's 12 h limit, the analysis reads invalid, and nothing qualified online.**
    - **Cut by the time limit.** 43 of 48 held-out cells ran. Block quiet/289 is complete. Bursts/289 lacks the
      reference's second run, and spike/289 never started. The online phase took about 6 h, not the 4.7 h priced:
      about 730 s a trace against 560 s. The held-out cells took about 480 s each, not 180 s.
    - **Verdict by the fixed analyser: invalid** (`block 1/289 is incomplete`). 0 answers mismatched, every run
      request completed, and traces 286-288 are complete in every state.
    - *Discrepancy, reported and not resolved here.* The protocol's prose says two complete traces per state
      suffice. `analyse_ag1` counts any incomplete block as a problem. The verdict above is the analyser's. Read by
      the prose instead, the run would be valid with 3 traces per state. The outcome would not change: (1) fails at
      +0.05 % (95 % -0.5 to +0.6 %), and (2) passes (quiet +0.5 %, bursts +0.1 %, spike -0.04 %; every bound above
      minus the noise of 3.3 %). The agent served the default throughout, so all of this is an A/A difference.
    - **Online.** All 30 traces ran (no early stop): 480 groups, 2 plans, 0 explorations, 7 triplets, 0 qualified,
      0 rejected. The reference served throughout. The world model reached a 2.7 % median relative error over 1,290
      groups and never reopened. The first plan (32 arrivals) proposed nothing. The second (96 arrivals, after
      about 1.4 h) proposed the best fixed schedule, `w1/sjf` (`633ceda2cdce`, model gain +0.5 %; model costs
      `w1/sjf` 59.38, `w1/fifo` 59.74).
    - No look was reached: the run ended with 7 triplets, 5 short of the look at 12. At that point the candidate
      read +0.6 % over the reference (95 % +0.1 to +1.2 %, bound 0.05 %, floor 1.2 %, noise 4.4 %), verdict open.
    - **Three tiers** (recorded, not decided; traces 286-288, states weighted equally):
      - fixed `w1/sjf` over the default: +1.0 % (95 % -0.1 to +1.6 %);
      - the agent over the default: +0.05 % (95 % -0.5 to +0.6 %). It served the default;
      - the agent over fixed: -1.0 % (95 % -2.1 to +0.2 %). That is the default against `w1/sjf`, not anything RL
        took away.
    - *Descriptive.* As on the T4 (V4), replay (+0.6 % at 7 triplets) and the served fixed tier (+1.0 %) agree in
      sign and roughly in size with the model's +0.5 %. The order-only gain on this traffic is about 0.5-1 %. The
      noise of a single trace is 3-4 %. At 7 triplets in the 23 traces after the plan, the 12-triplet look needs
      about 40 traces. Like V4, the run was short of triplets, not of signal. This action space (order by limit, at
      capacity 1) holds too little gain to qualify within one session, which is why AG2 changes the key the order
      sorts by instead.
    - By the Kill rule, `--agent` stays opt-in and experimental. Ledger entry owed.
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

- **V-CPU-2 (fixed 2026-10-30, after V-CPU's result and before any of its own data).** V-CPU with `--profile cpu2`,
  differing only in its seeds and budget: 40 online traces on fresh seeds 151-190 and 4 held-out traces per state on
  seeds 191-194 (no earlier run, training or simulation used them). V-CPU gave 8 triplets in 15 traces, so 40 traces
  should reach the look at 18. The agent code, decisions, alpha spending, validity and analysis are V's and V-CPU's,
  unchanged. It takes about 7 hours of this CPU and no GPU. Pass and kill are V-CPU's. Its motive, V-CPU's too-short
  budget, is named here, and no rule was changed after V-CPU's data.
  - **Run 1 (2026-10-30, at `3cb1e8f`): lost, no verdict.** A restart of the cloud container ended the process
    after 25 of 40 online traces. By then it had 17 triplets of its one candidate, `w1/sjf`. The looks at 6 and 12
    were inconclusive, nothing qualified or was rejected, and no evaluation ran. It is not rerun on these seeds,
    because a second attempt after reading its interim looks would be a repeated test. The T4 run V comes first.

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

**V's harness as a CPU smoke (2026-10-30, mechanics only, no gain claim).** `training/rl2_headroom.py --plan ag1 --smoke` ran
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

### AG2 — The agent learns how long an answer will be, and orders by it (opened 2026-11-03, before hardware data)

**Why (owner request, 2026-11-03: beat the fixed rules, with ideas from the literature).** Through V4, the agent never
beat the best fixed schedule. Its actions so far are a group's order (output limit first, or arrival) and width,
and in every measured setup the fixed rule is close to optimal for them: RL8 put the headroom at 0.1-1 %, below
replay noise. Two exploratory simulations on 2026-11-03 asked where a learner could do better. Both ran after V4's
data and are labelled so; they decide nothing on hardware. Script: the AG1 simulator's machine (`training/ag1_online.py`),
600 traces of RL2-H2's states per cell, run outside the repository.
- **A schedule per traffic state** against the best single fixed schedule: +0.02 % on the T4 price with 16-request
  traces, +0.18 % with 64, 0.00 % on the CPU price. *Closed by this simulation:* a state-dependent schedule has no
  headroom worth an experiment here.
- **Ordering by a predicted answer length.** Our test traffic so far sets an output limit that every answer
  reaches: of V4's 768 online requests, 761 ended at their limit. Then "shortest limit first" is the true shortest
  job first, and nothing is left to learn. In chat traffic, clients often send one limit (or none), and answers end
  at EOS, so "shortest limit first" falls back to arrival order. The literature's largest scheduling gains come from
  predicting the answer length instead: Fu et al. (NeurIPS 2024, arXiv 2408.15792) rank by a learned predictor, and
  so do SSJF (arXiv 2404.08509), ELIS (arXiv 2505.09142) and TIE (arXiv 2604.00499). In the simulator, with every
  limit 512 and lognormal answers (median 150 tokens, sigma 1, capped at the limit), ordering by a predictor whose
  correlation with the log length is rho, against ordering by the limit, at load 1.0:

  | rho | width 1 (T4 / CPU price) | width 4 | width 8 |
  | --: | --: | --: | --: |
  | 0.0 | +0.7 % / +1.0 % | -0.2 % / +0.3 % | -0.2 % / -0.2 % |
  | 0.3 | +8.2 % / +9.2 % | +5.1 % / +6.5 % | +3.1 % / +3.3 % |
  | 0.5 | +12.6 % / +14.3 % | +8.9 % / +10.6 % | +4.8 % / +6.0 % |
  | 1.0 (oracle) | +24.4 % / +27.5 % | +16.4 % / +18.8 % | +8.8 % / +10.1 % |

  At load 0.6 the gains are within 1 point of these.
- **How large rho gets online.** An online learner with no prior, run as the agent would run it: SGD on hashed word
  unigrams and bigrams of the prompt's first 64 words plus its log length, predict then learn, 6,000 shuffled
  requests. Spearman correlation with the answer's log length, over requests 500-2,000 and then 2,000-6,000:
  - Dolly 15k (human answers, CC BY-SA 3.0): 0.30, then 0.45;
  - Alpaca (model answers, CC BY-NC 4.0): 0.45, then 0.60;
  - prompt length alone: -0.10 to -0.24.

  A prediction cost 16-54 µs in pure Python. It is made once when a request arrives, so the decision at a group
  start still only reads a number.

**Mechanism.** The agent gains a second order key: the predicted answer length, capped at the request's limit, in
place of the limit itself.
- A length model is learned online from every finished request (label: completion tokens). Its features are
  hashed word unigrams and bigrams of the last user message (at most its first 64 words), log prompt characters
  and log output limit. It is updated by SGD, with weights only, and stores no text.
- The agent's policy carries which key it orders by. The planner proposes "shortest predicted first" as a
  candidate when the model is fit, and the candidate is proven by the same replay triplets and judge as AG1. A
  replayed arrival keeps its recorded prediction and replays its recorded answer length as its limit.
- Like every AG1 action, it changes the order of requests only, never an answer. State stays bound to the
  machine, model and plan, and the length model's weights live under `<state-dir>/agent/`.

**Kill criterion.**
- **S-LP in simulation:** the agent, unchanged except for this key, on the simulator with EOS-ended answers
  (Alpaca-like rho, learned online, not given) qualifies the predicted order within 150 traces, and its windows cost
  at least 5 % less than ordering by the limit. Anything less stops AG2 before hardware.
- **On hardware:** AG1 V's qualification rule and decisions. The workload uses real instruction prompts (Dolly 15k,
  fetched at run time, not committed), one limit for all, and answers ending at EOS. The protocol is fixed here
  before its data. Planned order:
  1. a Kaggle CPU session, which spends no GPU quota;
  2. Gemma 3 12B under the owner's `--compute-dtype native` plan on a T4 once the weekly quota resets. There the
     plan runs one request at a time, which is where this lever is largest in the simulation.

  If the online model's Spearman correlation after 500 requests of the hardware workload is below 0.2, the run
  reports "no signal" and AG2 is closed for that workload.
- A pass ships opt-in only, bound to the measured setup.

**Implementation (2026-11-03, `ec78d03`).**
- `ironmule_product/length_model.py` holds the length model.
- An order key in the LD1 policy format (`order_key`, absent means the limit, so every earlier digest holds). LD1
  refuses a predicted policy, because only the agent serves one.
- The planner gains `w{1,2,4,8}/pred` arms once at least 32 recorded arrivals, and 90 % of them, carry a prediction.
- Replays make each answer its recorded length and sort by the arm's recorded key. *(This also changes AG1's
  replays, which ran every answer to its capped limit. In V4 that made no difference: 761 of 768 answers ended at
  their limit.)*
- `training/ag1_online.py --prompts` replays real prompts with one limit for all, and adds hindsight comparators.
- `--profile cpu-lp` in the harness.
- Tests in `test_agent.py` and `test_rl2_headroom.py`.

**S-LP result (2026-11-03, at `ec78d03`'s agent and simulator, local): passed on both prices.**
- *Setup:* 300 traces of RL2-H2's states, seed 90,000, limit 512 for every request. Prompts and answer lengths
  come from 20,000 Alpaca rows (answer tokens taken as 1.3 per word: median 39, mean 57, 90th percentile 129). The
  length model learns online from nothing.
- *T4 price:*
  - The agent planned `w8/pred` at trace 34 and qualified it at trace 130 (replay gain +4.6 %), inside the 150
    allowed.
  - While it served (170 windows), windows cost 6.74 s against 7.24 s for the best limit-ordered fixed schedule in
    hindsight (`w8/fifo`): **6.9 % less**.
  - Before it served, the gap was 0.2 %.
- *CPU price:*
  - The agent planned `w8/pred` at trace 2 and qualified it at trace 99 (replay gain +6.6 %).
  - While it served (201 windows), windows cost **8.2 % less** than the best limit-ordered fixed schedule in
    hindsight (`w8/fifo`).
- *The length model:* Spearman 0.63 over its last 512 predictions on both prices.
- Both runs clear "at least 5 % less than ordering by the limit". AG2 goes to hardware.
- This is a simulation, so no hardware claim follows from it.

**The Kaggle CPU run is dropped (2026-11-03, from a pre-check of the workload, before any gain data).**
- *The pre-check:* SmolLM2-135M (dequantized, this container's CPU, greedy), the CPU profile's model, answered 16
  Dolly prompts under a limit of 128. Eleven answers reached the limit; the others took 42 to 113 tokens.
- That gives the length model almost nothing to separate. Each answer also took about 25 s here, about 50 s on a
  Kaggle CPU.
- So `--profile cpu-lp` would spend a session where little answer-length variance exists, and the CPU step of the
  planned order is dropped. The profile stays for smokes.
- The hardware test of AG2 is a T4 with a model whose answers end at EOS.

**Multi-GPU (owner request, 2026-11-03).**
- *Exploratory simulation, after V4's data:* two cards on the AG1 simulator's T4 price, the pool placing each
  request at arrival, 300 traces of 32 requests at the two cards' capacity, one limit, lognormal answers. The table
  gives each policy's gain over today's pool (least outstanding requests, arrival order):

  | Correlation rho | Order by prediction, width 1 | Also place by predicted work, width 1 | Order by prediction, width 8 | Also place by predicted work, width 8 |
  | --: | --: | --: | --: | --: |
  | 0.5 | +15.0 % | +15.3 % | +7.2 % | +8.1 % |
  | 0.63 | +18.7 % | +19.2 % | +8.7 % | +10.4 % |
  | 1.0 (oracle) | +30.5 % | +31.5 % | +13.2 % | +15.4 % |

  Placement by predicted work alone gains +0.7 to +2.8 %, about what RL2 and RL3 found for placement. The ordering
  carries the lever on two cards as on one. So AG2 keeps the pool's placement and orders each card's queue.
- *Change:* `ironmule serve --agent on --gpus all` was refused on several GPUs. It now attaches one agent per card
  (bound to its card, so its learning and evidence never mix). The pool reports and controls all of them, and keeps
  join-the-shortest-queue placement. The harness gains `cards`: one agent per card, and each card's frozen policy and
  length model in the held-out arm.

**T4 gate: a Gemma 3 12B smoke (fixed 2026-11-03, before its data).** This is the owner's requested model, under
the owner's `--compute-dtype native` plan.
- *Profile:* `--profile t4-g12-lp`, a `--smoke` run on both T4s of a Kaggle "GPU T4 x2" session, one agent per card:
  - `mlx-community/gemma-3-12b-it-4bit` (its `native` plan is gated at rev `86cc6a8`; the run records the revision
    it downloads);
  - the reference worker with `--compute-dtype native`, at capacity 1;
  - Dolly prompts and one limit of 512;
  - a 1,800 s deadline;
  - the fixed tier `w1/sjf`.
- *What it records* (it measures no gain):
  - whether the plan runs through the product and the agent's harness;
  - the calibration burst's 16 answer lengths and how they ended;
  - the calibrated capacity and the tokens per second.
- These fix the full T4 protocol's limit, trace count, idle time and session hours, written here before that
  run's data.
- *Cost:* about 0.7 session hours (an 8 GB download loaded once per card, a 32-request calibration burst and the
  smoke's small traces). That fits the roughly 1.2 GPU hours V4 left this week. A session cut by the quota is
  reported as such, and nothing is inferred from it.
- *Local smoke first (2026-11-03, at `e6b5d82`, `--profile cpu-lp --smoke` with SmolLM2-135M on this container's
  CPU):* the harness's AG2 path ran end to end:
  - Dolly's 15,011 prompts were fetched (sha256 `2df90833...`);
  - the 16-request calibration burst on Dolly prompts completed;
  - three online traces fed the length model;
  - all 36 held-out requests completed, with 0 answers differing across the four arms and no problems.
  - It reads "invalid" only because a smoke has one held-out trace.
- **Gate result (2026-11-03): passed.** Kaggle kernel `tobayko/ironmule-rl2-headroom-ag1-t4-g12-lp-smoke` v1, pinned
  to `e6b5d82`.
  - *Setup:* two Tesla T4s (driver 580.178.04), Python 3.13, mlx 0.32.4, mlx-lm 0.31.3, Gemma 3 12B at rev
    `86cc6a8` (the gated revision), and one agent per card. The harness took 16.4 minutes.
  - *Calibration:* all 32 requests completed in 237 s, a capacity of 0.135 requests a second for both cards. That is
    about 15 s per answer on one card, at about 21 tokens a second (the fitted 47 ms per decode step).
  - *Answers:* the 45 answers to the smoke's five distinct Dolly prompts took 40, 121, 129, 305 and 512 tokens; the
    longest reached the limit.
  - *Validity:* all 36 held-out requests completed, with 0 answers differing across the four arms. The run reads
    "invalid" only because a smoke has one held-out trace.
  - It measures no gain.
  - Ledger entry owed.

**V-G12, AG2's gain on hardware (fixed 2026-11-03, after the gate and before any of its own data).**
- *Profile:* `--profile t4-g12-v`. It is the gate's setup (Gemma 3 12B, `native`, both T4s with an agent per card,
  Dolly prompts, a 1,800 s deadline, the fixed tier `w1/sjf`), with these sizes, set from the gate:
  - **one limit of 256**, which halves the time per answer and keeps 40 to 256 tokens of spread;
  - **32 requests a trace**, about 16 per card, so one replay window per card per trace;
  - **360 s of idle time** after each online trace, enough for a triplet of about 16 answers of about 7 s;
  - at most **40 online traces** on fresh seeds 358-397, ending after the trace in which every card qualified;
  - **2 held-out traces per state** on seeds 398-399.
- *One change, made before any of this run's data* (`fcefb29`): the cards' agents share one length model
  (`agent.share_lengths`), so each card learns from both cards' answers. An answer's length depends on the model
  and the prompt, not on the card. Each card's policy, world model, replays and evidence stay its own.
- *Decisions:* AG1 V's decisions (1) and (2), its validity rule, and the three tiers, recorded and not decided. Under
  one limit the fixed `w1/sjf` orders by arrival, so it is the default's schedule. That is the point of the test: the
  hard rule has no information that the agent learns.
- *No signal:* if the shared length model's Spearman correlation is below 0.2 once it has seen 500 requests, the run
  reports "no signal" for this workload.
- *Cost:* about 0.4 h of setup, 40 traces of about 5 min of serving plus 6 min of idle (7.3 h), and 24 held-out
  cells of about 5 min (2.0 h). That is about **9.7 session hours** on one "GPU T4 x2" session, inside the 12-hour
  limit.
- It runs after the weekly GPU quota resets. The gate left about 0.9 hours this week.
- **Replaced before any of its data (owner, 2026-11-03): V-G12 does not run.** The owner set tokens per second as
  the agent's first measure ("The main goal"). At capacity 1 the order moves no tokens per second, so V-G12 could
  show none by construction. A dry run of its online phase in the simulator then found it would most likely not
  qualify in time either:
  - *Setup:* `training/ag1_online.py --profile v-g12`, per card: the gate's Gemma 3 12B price on a T4, capacity 1,
    16 requests a trace at half the two cards' calibrated rate, limit 256, 360 s idle, 40 traces, and a twin
    card's answers teaching the shared length model. Answer lengths are a proxy (an assumption): Dolly's
    reference answers in tokens, scaled by 3 or by 6. Scaled by 6, the mean at a limit of 512 is 277 tokens,
    near the gate's calibrated 300.
  - *Result, 3 seeds per scale:* the agent qualified nothing in all 6 runs. The length model reached a Spearman
    correlation of 0.31-0.44, against S-LP's 0.63. Two runs reached 12 triplets, and each was rejected.
  - The weekend's GPU hours go to KB2-0 instead.
- *Kill and ship:* as AG1 V. A pass is the agent's gain measured on two T4s, bounded to this model, plan, workload and
  cost. It ships opt-in only.

**Other routes from the same search, not yet entries (kept so they are not re-derived):**
- *preemption with a re-predicted remaining length* (TRAIL; ELIS's iterative shortest-remaining-time-first). This
  needs the engine to pause a session and resume it exactly.
- *prefix-aware grouping* (arXiv 2605.06046), grouping requests that share a cached prefix. Its gain depends on
  shared system prompts in the traffic.
- *load-adaptive prefill chunking* (DRLServe). The engine has no chunked prefill to adapt.

### AG3 — The agent knows the machine's architecture (opened 2026-11-03, owner request)

**Question (owner, 2026-11-03).** Would the agent adapt better if it knew the computer's hardware architecture, and
what else could then be optimised? The agent already learns the machine empirically: its world model fits what
prefill, a decode round and a further session cost on this card. Knowing the architecture can add three things:
1. a starting point for that fit (a prior from the hardware), so the agent plans sooner;
2. the gap between the measured costs and the hardware's limit, which shows where speed is left;
3. a guard on what the card can hold (memory). This one is not built, because no measured run has hit it.

**Mechanism (2026-11-03).** `ironmule_product/hardware_profile.py` reads the device's name (`nvidia-smi` per card,
or the Apple chip). It looks the name up in a table of published peak memory bandwidths; an unknown device gets no
value, never a guess. With the checkpoint's weight bytes it gives the decode floor: a memory-bound step reads every
weight once, so it takes at least `weight bytes / bandwidth`. The agent's status (`/health`, `agent.hardware`) reports:
- the profile;
- once the world model has fitted 12 groups:
  - the share of bandwidth a measured decode step uses;
  - the most a bandwidth-bound kernel could gain;
  - how much of a whole step a further session in a round adds.

It changes no decision and no answer.

**What the measured fits already say** (derived from earlier runs' world models, not a new measurement):

| Setup | Weights | Floor | Measured step | Bandwidth used | Kernel headroom at most | Further session costs |
| :-- | --: | --: | --: | --: | --: | --: |
| T4, Qwen3 1.7B bf16, tuned engine (V4, 813 groups) | 3.44 GB | 10.8 ms | 19.6 ms | 55 % | 1.8x | 0.91 of a step |
| T4, Gemma 3 12B 4 bit, `native` (gate, 5 groups only) | 8.03 GB | 25.1 ms | 47.2 ms | 53 % | 1.9x | not fitted |

- *Kernels:* decode on the T4 uses about half the card's memory bandwidth. Up to about 1.8x is left in kernels,
  not in scheduling.
- *Batching:* a further session in a grouped batch-1 round costs 91 % of a step, so every session reads the
  weights again. That is why the dense plan, which shares a round's weight reads, served 48 % faster (H10).
- Both levers sit outside the agent's actions. Its actions change only the order and width of exact grouped work.
  A kernel change goes through the controlled worker, with its own correctness and evidence gates.

**The prior, tested in simulation (2026-11-03, exploratory, after AG2's data).** The setup is S-LP's (300 traces,
Alpaca prompts, one limit of 512), with the world model starting from the machine's own price instead of the
generic prior.
- *Kept if:* it qualifies the agent's schedule at least 10 traces earlier than S-LP run 1 on both prices.
- *Otherwise:* AG3 stays report-only.
- **Result (2026-11-03): not kept; AG3 stays report-only.**
  - *T4 price:* the agent planned `w8/pred` at trace 10 (S-LP run 1: trace 34), but replay rejected it at trace
    106. Two later candidates were also rejected, nothing qualified in the 300 traces, and the reference served
    throughout. S-LP run 1, with the generic prior, qualified at trace 130.
  - *CPU price:* it qualified `w8/pred` at trace 99, as S-LP run 1 did, and served 8.2 % below the best fixed
    schedule.
  - Neither price qualified 10 traces earlier.
  - *Descriptive:* the world model leaves its generic prior within a few groups. What holds the agent back is how
    fast the length model learns and how many replay triplets it collects, not what it knows about the machine.
    The T4 rejection, one seed each, shows how noisy a single simulated run's verdict is, not that the prior
    harms.

### AG4 — The agent chooses the plan and the engine itself (opened 2026-11-03, owner decision, before data)

**Why.** Every large tokens-per-second gain so far was a setting a person found by hand on Kaggle:
- the `native` plan for a 4-bit checkpoint on an emulating GPU (about 6x);
- the stacked engine (H10: 48 % cheaper on two T4s);
- its width (w8);
- for Gemma 3 at 4 bit, KB2.

The agent (AG1, AG2) decides only order and width, and it starts from whatever plan and engine the operator
named. The owner wants the agent to learn and set these itself ("The main goal"), and allowed output-changing plans
that pass its own gate.

**Mechanism (step 1, at startup).** `ironmule serve --agent on` with no `--engine` and no `--compute-dtype` chooses
both itself on this machine, before it serves. The choice is made once per hardware fingerprint and model identity,
and stored with its evidence like NUM1's.
- *Candidates:*
  - the reference path (always eligible, the incumbent);
  - NUM1's numeric plans (`ironmule/numeric_choice.py`, which already gates each plan on the device with the
    perplexity gate and a speed check; consent is now implied by agent mode);
  - a stacked engine where its `refuse` lets the model through:
    - `dense` for Qwen 3 bf16 on pre-Ampere CUDA;
    - KB2's Gemma plan for Gemma 3 at 4 bit on the `native` plan.
- *A stacked engine's gate, all measured in a child process on this card:*
  - its load probe (every stacked width bit-identical to one row);
  - the perplexity gate against the reference path (16 x 512 tokens, the 97.5 % bound of the ratio below 1.005);
  - tokens per second at width 1 and width 8 against the reference path's single-request decode;
  - the time to the first token of a short and a long prompt.
- *Decision rule (fixed now):* the candidate with the most tokens per second at width 8 among those that pass
  their gate, if:
  - its single-request tokens per second are at least 0.9x the reference path's (quiet traffic must not
    regress);
  - and its time to first token is no worse than the reference path's by more than 10 %.
  Otherwise the reference path serves. The online agent then learns width and order inside the chosen engine,
  as before.
- *Visible:* `/health` and the agent's status carry the chosen plan and engine, every candidate's gate and speeds,
  and the reason.

**Kill (fixed now).** On a Kaggle T4 with Gemma 3 12B at 4 bit, the agent's own choice must reach at least 2x the
tokens per second of today's default (`--engine reference` without a plan), with time to first token no worse,
measured through the product. Otherwise AG4 does not ship as the agent's default setup. The product run's full
protocol (traffic, arms, cost) is fixed here after KB2-0 and before that run's data.

**Step 2 (later).** Re-choose while serving, when the measured traffic or the software changes. This needs a
second worker to switch engines without dropping requests.

**Step 2a: both paths on one card, routed per request (built 2026-11-03, after AG4-Q run 2; labelled as shaped by
that run's data).** AG4-Q run 2 showed the stacked engine winning under load (5x at width 8) and losing alone
(0.82x, a later first token). So the choice depends on the load, and a fixed choice at startup leaves one side's
gain unused.
- *Mechanism.* `ironmule_product/path_router.py`. Where the stacked engine passes its gate and wins at width 8 but
  fails the quiet floor or the first-token rule, and two copies fit the card (two copies of the weights, the pool
  and 3 GiB in 90 % of the card), the agent's setup is `route`. Both paths serve on one card, each its own worker:
  NUM1's single path and the stacked engine.
- For each request the router predicts its finish time on each path from what is outstanding there, and sends it
  to the earlier one. Ties go to the single path, so a quiet server behaves as the single path alone.
- *Online learning:* recursive least squares on each path's seconds per token against how many requests share it
  (forgetting 0.995), plus running means of the first token and of the tokens per request. It starts from the
  AG4 measurement on this card and learns from every finished request. A decision takes a few microseconds (test:
  under 100 µs in pure Python, measured as `decision_us` in `/health`'s `paths` block).
- *Simulation (`training/ag4_routing.py`, priced by AG4-Q run 2's T4 numbers; descriptive, no hardware claim):*
  - Traffic: RL2-H2's three states, 30 traces of 32 requests per load, answers from Dolly's lengths (scale 3,
    capped at 256).
  - The table gives tokens per second and the mean and 95th-percentile first token.

  | Load (over the single path's capacity) | Single path alone | Stacked alone | Router |
  | --: | --: | --: | --: |
  | 0.3 | 7.5, 13.8 s, 63.5 s | 7.7, 3.1 s, 16.1 s | 7.8, 2.4 s, 12.0 s |
  | 0.8 | 19.1, 15.4 s, 65.7 s | 20.9, 3.2 s, 16.2 s | 21.1, 2.5 s, 12.1 s |
  | 1.5 | 30.9, 16.5 s, 64.4 s | 37.2, 3.1 s, 15.9 s | 37.4, 2.7 s, 12.2 s |
  | 3.0 | 46.0, 19.6 s, 61.6 s | 67.0, 3.3 s, 15.6 s | 68.7, 2.7 s, 11.8 s |

  - The router is at least as fast as the better path at every load, with the earliest first tokens. At load 3 it
    makes 49 % more tokens per second than the single path the agent chose alone.
  - RL2-H2's bursts and spike queue the single path even at low mean load, which is why its first token waits.
- **AG4-R, the hardware test (fixed now, before its data; it runs after KB2-0, when the quota allows).**
  - *Setup:* one Kaggle T4, `mlx-community/Qwen3-1.7B-bf16` at `9cd6692`, served by `ironmule serve --agent on`
    with the agent's own setup. It is `route` if AG4's rule says so; otherwise the run reports that and stops.
  - *Arms, on the same traces through the product:* the router; the single path alone (NUM1's `float16`, what the
    agent chose before step 2a); the stacked engine alone.
  - *Traffic:* RL2-H2's states at loads 0.8 and 3.0, on fresh seeds, 6 traces per state and load.
  - *Pass:* both, on the pooled traces:
    - the router's tokens per second over the single path's has a 2.5 % bound above 0 (paired by trace, 2,000
      resamples);
    - its mean and 95th-percentile first token are no worse than the single path's beyond the A/A noise of two
      runs of the single path.
  - *Otherwise:* the router does not ship as the agent's setup.
  - *Harness (built 2026-11-03):* `training/rl2_headroom.py --plan ag4r`. It runs the product's `choose_setup`, then
    both paths in their own workers on one card, and all three arms through the same two loaded workers. The
    single path and the stacked engine are forced by a fixed route, and the router keeps learning across blocks.
    Each block runs single, router and stacked (rotated), then single again, on the same arrivals. A calibration
    burst on the single path sets the loads. `analyse_ag4r` applies the rule above.
  - *Cost:* about 3.5 session hours (36 blocks of 4 cells of about 80 s, plus setup). It runs after KB2-0 in the
    week's quota.
  - *Smoke first:* a `--smoke` kernel checks that both workers fit and run together on one T4, and measures
    nothing.
  - **Amended 2026-11-05, before any AG4-R data: a fourth arm, the router frozen at its startup prices.** The
    pass rule above decides whether routing ships. It cannot say whether *learning while serving* adds anything,
    because a router with AG4's startup prices alone might route as well. That is the main goal's open part (LIMITS:
    online learning has no hardware gain yet).
    - *Arm:* `frozen`, a copy of the router taken before its first finished request. It decides exactly as the router
      does but never learns. Each block runs single, then router, stacked and frozen in cyclic rotation, then single
      again, on the same arrivals.
    - *Learning decision (recorded, it does not change the router's ship rule):* learning while serving shows a
      measured gain on this card if the router's tokens per second over the frozen router's, paired by block, has a
      2.5 % bound above 1 (2,000 resamples), and its mean and 95th-percentile first token are no worse than the
      frozen router's beyond the same A/A noise. Otherwise: "no measured gain from learning while serving" on this
      setup, and LIMITS keeps saying so. The later half of the blocks is reported too, descriptively, since the
      router keeps learning across blocks.
    - *Expected:* if AG4's startup prices are already accurate, the two routers decide alike and the learning
      decision fails. That would be an honest result, not a fault of the run.
    - *Cost:* 36 blocks of 5 cells of about 80 s, plus setup, about **4.4 session hours** instead of 3.5.
  - **Smoke 1 (2026-11-03, kernel `tobayko/ironmule-rl2-headroom-ag4r-smoke` v1 at `526934d`, Kaggle T4, 9
    minutes): stopped before serving, on a product gap.**
    - The agent's own setup, made again from scratch on the card, was `route`: NUM1's `float16`, and the dense
      engine 0.82x alone.
    - The product's worker then refused `float16`. It knew only `float32`, `native` and `dequantize`, although
      NUM1 can choose `float16`.
    - *Fixed:* the worker, its client and `--compute-dtype` take `float16` (the checkpoint's floating weights in
      float16, as `ironmule.tune` already loads that plan). Smoke 2 repeats it.
  - **Smoke 2 (2026-11-03, the same kernel v2 at `321d116`, Kaggle T4, 14 minutes): the tool works end to end; it
    measures nothing.**
    - The agent chose `route` again on its own.
    - Both workers (`float16` and the dense engine) loaded and served together on one T4.
    - All 3 blocks of 4 cells completed with no failed request.
    - The router split its 12 requests 6 and 6, in 17 µs per decision.
    - *Descriptive, a smoke measures nothing (4 requests of up to 16 tokens):* the stacked engine's learned
      per-row cost was 106 ms. In answers this short, a newcomer's prefill stalling the running rows dominates the
      per-token time. The full run's 256-token answers price that stall per token, as it should be.
  - **Run 1 (2026-11-07, kernel `tobayko/ironmule-rl2-headroom-ag4r` v1 at `b1c2382`, one Kaggle T4, Qwen 3 1.7B
    bf16 at `9cd6692`): cut by Kaggle's 12-hour limit after 22 of 36 blocks, invalid by the fixed rule.**
    - The agent chose `route` on its own again: NUM1's `float16`, and the stacked engine at 0.82x for one request
      alone. Setup took 8 minutes.
    - **Why it ran out of time.** The cells measured all 11.8 hours; nothing waited idle. The protocol priced a cell
      at about 80 s, but a cell lasts as long as its 32 arrivals are spread. The median cell took 791 s in the quiet
      state at load 0.8, 301 s in bursts, 62 s in a spike, and 215 s quiet at load 3.0. All 36 blocks would have
      needed about 25 hours. Load 0.8 ran in full; load 3.0 reached only its quiet state, and its bursts and spike
      never ran.
    - **Problems (the rule voids the run):** block 22 is incomplete, and in blocks 13 and 14 (spike, load 0.8)
      the stacked engine rejected 8 of 32 requests alone, and the frozen router's cell lost 7. The rejection reads
      "the prompt and max_tokens exceed the ...-token limit of this server's engine": some Dolly prompts plus 256
      tokens do not fit the stacked engine's slot. The forced stacked arm cannot avoid them. The frozen router sent
      such requests there, so **the router does not check whether a path can hold a request**. That is a product
      defect, fixed before run 2.
    - *Descriptive only (20 complete blocks, no verdict; pooled ratios against the single path, 95 % in brackets):*
      - the router: 1.44x the tokens per second (1.13-1.68), its mean first token 0.35x (0.32-0.37), its 95th
        percentile 0.43x. By load, 1.57x at 0.8 and 1.000x at 3.0, where only the quiet state ran;
      - **the stacked engine alone: 2.10x the tokens per second (1.32-2.70), its mean first token 0.19x
        (0.14-0.21)**, although AG4's rule set it aside for its 0.82x on one request alone;
      - learning while serving: the router over the frozen router 1.007x the tokens per second (0.998-1.013), its
        first token later (mean 1.16x, 95th percentile 1.30x), and the late half 1.011x (1.003-1.017). No gain from
        learning is shown, and it cost first-token time;
      - A/A noise of the single path's two runs: 0.06 % tokens per second, 4.7 % mean first token.
    - *What it says (post hoc, it changes no verdict):* on this traffic the agent's own measurement of one request
      alone misled it. The stacked engine it set aside made 2.10x where its routed setup made 1.44x. Its router
      learned too little, too slowly, to find that out while serving. That is where a learning agent should gain.
      Run 2 needs a protocol that fits a session, and the router has to respect each path's request limit.
    - The ledger entry is still owed. About 12 session hours were spent.
**AG4-C, the agent's own setup against today's default on a CPU (fixed 2026-11-03, before its data).** No GPU
hours are left this week, and a Kaggle CPU session spends none. Does the agent's own setup, chosen on that machine
with nothing given to it, serve more tokens per second than today's default through the product?
- *Setup:* `training/rl2_headroom.py --plan ag4c` on a Kaggle CPU session at a pushed commit, with
  `mlx-community/SmolLM2-135M-Instruct-8bit` (V-CPU's model). The product's `choose_setup` runs first. NUM1's CPU
  check decides between the checkpoint's 4-bit weights and `dequantize` (CPU3 found `dequantize` about 6x faster
  by hand). If the agent keeps the default, the run reports that and stops.
- *Arms, both paths loaded side by side through the product:* the default (the reference worker, no plan) and the
  agent's chosen path. Each block runs default, agent, default on the same arrivals.
- *Traffic:* RL2-H2's three states at 0.8 of the default path's calibrated capacity, fresh seeds 410-413 (4 traces
  per state), 8 requests per trace, an output limit of 32.
- *Sized after a local tool smoke, before any data:* on this container `Qwen3-0.6B-4bit` took about 9 s a token
  on its 4-bit weights. The agent's CPU check chose `dequantize` there by itself (0.124 of stock time, logits
  within 1.45 % of the float32 plan). At that speed the run would not fit a 12-hour session, so it uses V-CPU's
  smaller model and an output limit of 32 instead of 64.
- *Amended again before any data (2026-11-03), after a second local tool smoke:*
  - On this container the agent's own CPU check, on its first sight of `SmolLM2-135M-Instruct-8bit`, measured
    260.4 s for a 16-token answer on the 8-bit weights and 7.3 s with `dequantize` (0.028 of stock time, logits
    within 0.28 % of the float32 plan, the same top token). It chose `dequantize` by itself.
  - At 16 s a token on the default path, the sizes above would take more than a day.
  - So the run uses 2 traces per state (seeds 410-411), 4 requests per trace, an output limit of 16, and a 7,200 s
    request deadline.
  - It runs on this cloud container's CPU. A Kaggle CPU was about 2x slower in V5-CPU, and the run would not fit
    a 12-hour session there. A container restart loses the run, and is reported as such.
  - Expected: about 4.5 hours, nearly all of it the default path's cells.
- **Run 1 (2026-11-03, this container's CPU, x86_64 with 4 cores, at `b0030fa`): lost to a container restart after
  1 of 6 blocks. Invalid by the rule (fewer than two complete blocks).**
  - *The agent's own setup:* `dequantize` on the reference engine (no stacked engine on a CPU).
  - Calibration on the default path: 0.0044 requests a second, 918 s for 4 answers of 16 tokens.
  - *The one complete block (quiet, trace 410), descriptive only:*
    - first token: 3.8 s on the agent's path against 155.7 s and 156.8 s on the default's two runs;
    - a whole answer: 5-8 s against 222-254 s;
    - tokens per second over the block's span: 0.019 against 0.018;
    - 3 of the 4 answers were bit-identical to the default's.
  - *Why the tokens per second did not move (descriptive):* in RL2-H2's quiet state the arrivals are about 20
    minutes apart here, so a block's span is set by the arrivals, not by how fast the path computes. This metric can
    show a speed-up only in bursts and spike.
- **Run 2, labelled post hoc to run 1 (fixed 2026-11-03, before its own data).**
  - *What changes:* only RL2-H2's bursts and spike states, where requests overlap and the span is set by
    computing, on fresh seeds 412-413 (2 traces per state, 4 blocks).
  - *What stays:* everything else, the pass rule included.
  - *Expected:* about 2.5 hours on this container. Another restart is reported as such.
  - *Duration, recorded while it runs (2026-11-04):* the estimate was wrong. A 1,034-token prompt in bursts
    takes about 70 minutes to its first token on the default path, and every block runs that path twice (the A/A
    control). Block 0 took about 2.6 hours, so the four blocks need about 10-13 hours. The run and its rule are
    unchanged; only the time estimate was wrong.
  - *Result (2026-11-04, this container's CPU, at `745b6f7`, 15.2 hours): invalid by the fixed rule.* The
    default path missed the 7,200 s deadline in blocks 1 and 2 (bursts trace 413: 2 of 4 requests in both of its
    runs; spike trace 412: 3 of 4 in both), so only blocks 0 and 3 are complete, and a block with a failed request
    makes the run invalid. The agent's path answered every request in every block.
  - *The agent's own setup:* `dequantize` on the reference engine again (the stored choice, not re-measured).
  - *On the two complete blocks, descriptive only:* the agent's tokens per second were 37.8x the default's
    (95 % 37.3x to 38.2x), its mean first token 0.026x (95 % 0.026x to 0.027x), and the A/A noise of the default's
    two runs 1.2 %. Both parts of the pass rule would hold there.
  - *What went wrong (a flaw of the protocol, not of either path):* a block whose baseline is too slow for the
    deadline is dropped, so the slower the default, the less the run can show. The rule was fixed before the data
    and stands for this run.
  - *Post hoc, descriptive only, not a verdict:* with run 3's censoring (below) applied to run 2 at its 7,200 s
    deadline, all 4 blocks count (10 default requests censored): 30.5x tokens per second (95 % 19.8x to 37.8x),
    mean first token 0.035x.
- **Run 3, labelled post hoc to run 2 (fixed 2026-11-04, before its own data).**
  - *What changes:*
    - a default-path request that misses the deadline counts as if it had produced its whole token limit, its
      first token and its last, at the deadline. That is never later than it really would have, so it can only
      flatter the default (`_cell`'s `censor`, `censor_default` in the report);
    - an agent-path failure still makes its block incomplete, and an incomplete block still makes the run invalid;
    - the deadline is 3,600 s instead of 7,200 s, which shortens the run and flatters the default further;
    - fresh seeds 414-415, bursts and spike again (4 blocks).
  - *What stays:* the agent's setup is chosen by `choose_setup` as before, the pass rule (tokens per second
    with a 2.5 % bound above 1, first token no worse beyond the A/A noise), 2,000 resamples, the model and the
    sizes.
  - *Expected:* 7-10 hours on this container (a default run is at most 3,600 s plus the time its worker needs to
    finish a request it was working on). A restart loses the run and is reported as such.
  - *Result (2026-11-04, this container's CPU, x86_64 with 4 cores, at `19c3d0a`, 8.0 hours): valid, pass.*
    - *The agent's own setup:* on a fresh store it measured again and chose `dequantize` by itself (0.0247 of stock
      time for a 16-token answer, logits within 0.28 % of the float32 plan), on the reference engine.
    - *All 4 blocks complete:* the agent's path answered all 16 of its requests; 22 of the default path's 32 missed
      the 3,600 s deadline and were censored in its favour.
    - *Tokens per second:* the agent's path 24.5x the default's (95 % 12.3x to 31.3x), paired by block, 2,000
      resamples. The bound is far above 1.
    - *First token:* mean 0.048x the default's (95 % 0.042x to 0.052x), 95th percentile 0.068x; the A/A noise of
      the default's two runs was 1.5 % and 1.1 %. No worse, by a wide margin.
    - *Answers:* `dequantize` is an output-changing plan (the owner allows it under `--agent on` behind its own
      quality gate, which it passed); of the 5 requests both paths finished, 1 answer was bit-identical.
    - *What it shows:* the agent, unasked, on a machine and model it had not seen, picks its own numeric plan, and
      through the product that choice serves about 24x the tokens per second of today's default with first
      tokens about 20x sooner. The setting itself is NUM1's, found by hand first; the gain is the agent's to
      choose and is measured on this CPU, this model (`SmolLM2-135M-Instruct-8bit`) and RL2-H2's bursts and spike
      traffic only. The ledger entry is still owed.
- *Pass (both):*
  - the agent's tokens per second over the default's (the mean of its two runs), paired by block, has a 2.5 %
    bound above 1 (2,000 resamples);
  - its mean and 95th-percentile first token are no worse than the default's beyond the A/A noise of the
    default's two runs.
- *A pass* is the agent's own measured gain on that CPU, model and traffic. The setting itself (`dequantize`) is
  NUM1's, found by hand first. What AG4-C shows is the agent choosing it, unasked, on a machine it had not seen,
  and that the choice pays through the product. *Otherwise* AG4's CPU setup does not ship.
- *Cost:* no GPU hours; about 2-3 hours of a Kaggle CPU session (12 blocks of 3 cells).

- **Step 3 (later).** The online agent inside the stacked engine. Its world model prices one group at a time, so
`_agent` refuses an engine with slot reuse. Until it models joins, the stacked engine serves its measured
schedule (w8 with joins, H10), and `/health` says so.

**Implementation (2026-11-03).**
- `ironmule/plan_choice.py` (`choose_setup`, `measure_stacked`, `decide`).
- `ironmule serve --agent on` without `--engine` and `--compute-dtype` applies the choice. A named one is kept.
- `--engine dense` serves a 4-bit Gemma 3 through KB2's plan (`engine_bridge.install_dense`).
- `/health` carries `agent_setup`.
- Tests: `tests/engine/test_plan_choice.py`, and the CLI wiring in `tests/agent/test_agent.py`.
- No hardware run yet: the stacked Gemma engine waits for KB2-0, and AG4's kill run for its protocol.

**AG4-Q, the agent's own setup on a T4 (fixed 2026-11-03, before its data).** Does the agent find on its own, on
the hardware, the setup that people found by hand for Qwen 3 1.7B bf16 on a T4 (H10: the dense engine, 48.3 %
cheaper than the default)?
- *Run:* `tools/cuda_probe.py --plan ag4` on one Kaggle T4 at the pushed commit. It runs the product's own
  `choose_setup` for `mlx-community/Qwen3-1.7B-bf16` at `9cd6692`, with nothing given to it, and records the
  decision with every gate, speed and reason.
- *Expected (descriptive, from H9/H10 and NUM1's table):* NUM1 keeps the checkpoint's bf16 or takes a plan its
  table recommends. The stacked engine passes its gate, at about 1x alone and several times the tokens per second
  at width 8 (KB1-2: 0.93-0.99x alone, about 3 ms per further row).
- *Pass:* the agent chooses the dense engine by AG4's rule, and its gate passed on the decode path. Then the
  product-level gain of that choice on this setup is H10's measurement (a hand-chosen run of the same engine).
  That holds for the setup only and is no new measurement.
- *Otherwise:* the decision and its reason are recorded, and AG4's rule or gate is re-examined before KB2.
- *Cost:* about 0.5 session hours (a 3.4 GB download, NUM1's candidates in children, the gate's 16 x 512 decode
  steps). That fits the roughly 0.75 hours left this week. A session cut by the quota is reported as such.
- **Run 1 (2026-11-03, kernel `tobayko/ironmule-cuda-probe-ag4` v1 at `29fd91a`, Kaggle T4, 3 minutes): the
  chooser had a bug, so no stacked engine was compared.**
  - NUM1 took `float16` from its measured table (0.310 of stock time, quality inside 1.005).
  - `choose_setup` then measured the stacked engine on top of that plan. The dense plan refused it ("serves bfloat16
    weights only"), so the agent kept the reference engine with `float16`.
  - *Implementation fix, not an analysis change:* a stacked engine now runs on the plan it was built on (KB1's on
    the checkpoint's bf16, KB2's on `native`, `stacked_base`). It is then compared against the path that would
    serve instead: NUM1's choice, measured alone in its own child (`measure_base`, `against`). The rule and gate are
    unchanged. Run 2 repeats AG4-Q at the fixed commit.
- **Run 2 (2026-11-03, kernel `tobayko/ironmule-cuda-probe-ag4` v2 at `cd31095`, Kaggle T4, 8 minutes): the agent
  chose by itself, on the hardware, and by AG4's rule it kept `float16` on the reference engine. That is not the
  dense engine.**
  - *NUM1:* `float16` from its measured table (0.310 of the stock bf16 time). Alone it made 57.7 tokens a second
    here.
  - *The dense engine (on the checkpoint's bf16):*
    - load probe passed: stacked rows identical, row kernel within 0.0025;
    - **gate passed on the decode path**, ratio 0.9997 (95 % 0.9983-1.0011);
    - **290.9 tokens a second at width 8, 5.0x the `float16` path's single request**;
    - but 47.3 tokens a second alone, 0.82x. That fails the quiet floor of 0.9, so the agent did not take it.
  - *First token, recorded:* on the 300-token prompt the dense engine took 0.36 s against `float16`'s 0.06 s. The
    dense plan prefills with MLX's bf16, which the T4 emulates. On the short prompt the base path's 7.8 s
    included its first compile; the timing had no warm-up. *Fixed after this run:* both paths are warmed before
    the clock. This does not change the decision, which failed on the quiet floor alone.
  - *Reading.* Neither engine wins everywhere on this card, and the agent's rule saw that:
    - the dense engine makes 5x the tokens per second under load;
    - it is 18 % slower for a request alone, and its first token comes much later on longer prompts.
    That makes the choice load-dependent. Two levers follow:
    - serve both paths and route by queue length (AG4 step 2, with two workers on one card: 2 x 3.4 GB fits a
      T4);
    - prefill the dense plan in float16 (a numeric change with its own gate), so its first token stops waiting
      on bf16 emulation.
    Neither is built yet. Ledger entry owed.

### AG5 — The agent learns the stacked engine's width, and keeps learning it while it serves (opened 2026-11-07, owner request, before data)

**Why (owner, 2026-11-07).** The owner wants a self-learning AI that makes the decisions, pursued with dreaming
and RL, and fast visible gains in tokens per second. The decision with the largest measured lever that nothing
learns yet is the stacked engine's width.
- AG4's rule judges the engine at width 8 only (`plan_choice._plan_speed` times widths 1 and 8), and the product
  always stacks up to 8 (`engine_bridge.DENSE_WIDTH`).
- KB2-0 run 2 measured the Gemma plan's step: 48.8 ms at 1 row, 67.8, 85.1, 113.0, 122.1, 141.3 at 6, 190.3 at 7
  and 394.1 at 8. Today's native step is 47.3 ms.
- At width 8 the plan makes 20.3 tokens per second, below today's 21.1, so AG4's rule would refuse it. At width 6
  it makes 42.5, which is 2.006x.
- The best width depends on the model and the card: Qwen 3's dense plan is best at 8 (AG4-Q: 290.9 tokens/s).

Recent work points the same way. In batching, RL's edge over hand-tuned heuristics comes from combinatorial,
multi-resource decisions rather than single-resource timing (arXiv 2607.05272). Nightjar (arXiv 2512.22420) uses a
contextual bandit to pick a decode-side setting per batch size, scored by tokens per second.

**Mechanism.**
1. *Startup, the agent measures:* `plan_choice` times the stacked step at every width from 1 to the pool's slots,
   and judges the engine at its best width, the one with the most tokens per second on that curve. AG4's other
   conditions are unchanged. The setup records the curve and the width.
2. *Serving at the learned width:* the product builds the stacked engine with that many slots (`--dense-width`),
   so neither a starting group nor a joiner exceeds it, and its pool is smaller.
3. *Online, AG5-L (fixed before its data, once 1 and 2 have a hardware result):* the agent keeps a step-cost
   model per width from the steps it serves and re-chooses the width by replaying its logged traffic in that
   model (dreaming, as RSI4's `dream` does for group width) and by trying a neighbouring width now and then (a
   bandit). This is the part that keeps learning.

Width changes no answer: within the plan every row is bit-identical alone or stacked (KB1-1, KB2 (a)), so this
decision needs no quality gate of its own.

**Amendment to AG4's rule (2026-11-07, before AG5-G's data).** "More tokens per second at its best width"
replaces "at width 8". AG4-Q, AG4-R and AG4-C measured under the width-8 rule and stay as recorded; AG4-R runs at
`b1c2382` and is not touched.

**Kill.** AG5 is closed, and the agent keeps width 8, if AG5-G fails by its rule below or the agent at its own width
makes fewer tokens per second than at width 8 on the same card.

**AG5-G, the hardware test (fixed now, before its data; it runs only if KB2-G passes).** *Not run:* KB2-G did not
pass (2026-11-07), so the Gemma plan has no gate to serve under. The width rule and its wiring (1 and 2) stand for
every stacked engine; AG5-G waits for a stacked engine whose gate passes. *KB2-L passed the same day, and KB2's
verdict is met, so AG5-G runs as fixed, at a pushed commit that carries the path fallback.*
- **AG5-G run 1 (2026-11-07, kernel `tobayko/ironmule-rl2-headroom-ag5g` v1 at `b8231b7`, two T4s, 20 minutes):
  stopped before serving, on a defect in the agent's own measurement. No data.**
  - The agent's `measure_stacked` installed the stacked plan, with its 3.2 GB pool, and only then timed today's
    native path. That path prefills a whole prompt through a 2 GB float16 copy of the head, and the card ran out
    of memory. KB2-0 run 1 had found the same in the probe. The measurement's child process failed, so the agent
    saw "no stacked engine", kept the native path, and the run stopped as its rule says.
  - *Fixed (no protocol change):* the base path is timed before the plan exists, and freed buffers are returned
    first. A model no stacked plan serves is now refused before anything is timed (`stacked_refusal`).
  - Run 2 is the same AG5-G at the fixed commit.
- **AG5-G run 2 (2026-11-07, kernel v2 at `fd576f4`, two T4s, 2.1 session hours): stopped before serving, again
  in the agent's own measurement. No data.**
  - NUM1 chose `native` within its gate 3 minutes into the session. The stacked measurement's child process then
    ran for its whole 7,200 s timeout without a result, and the timeout crashed the run. It printed nothing, so
    where the time went is not recorded.
  - *Likely cause (an inference, not a measurement):* the gate's reference still ran on the native path after the
    plan's pool existed. That path's prefill of each of the 16 chunks builds the 2 GB float16 head copy beside the
    weights and the 3.2 GB pool. KB2-G's probe computed the same reference before any pool, and it gated the same
    plan, weights and card in 26 minutes. For scale: AG6-H's measurement of Qwen 3 1.7B on a T4 took 355 s.
  - *Fixed (no protocol change):*
    - the gate's reference on the base path now also runs before the plan exists, as in the probe;
    - the child logs each stage with its elapsed seconds to the session log;
    - a measurement past `MEASURE_TIMEOUT_S` (3,600 s) is killed and ends as "no stacked engine", with the reason;
    - a failed measurement of the competing path is refused instead of raising a `KeyError`.
  - Run 3 is the same AG5-G at the fixed commit. If its measurement again passes 3,600 s, the stage log shows
    where, and the run stops by its rule.
- **AG5-G run 3 (2026-11-07, kernel v3 at `f737494`, two T4s, 0.35 session hours): stopped before serving, at
  the measurement's last stage. No data.**
  - The reordering worked. The stage log read: base path timed at 30 s, gate reference on the native path
    (16 chunks) at 448 s, plan installed at 595 s, gate on the plan at 995 s, and every width timed on the short
    prompt at 1,031 s.
  - Then the long prompt's first prefill on the plan (about 300 tokens) failed in
    `cudaMallocAsync ... out of memory`. The child exited, the agent read "no stacked engine", and the run
    stopped by its rule. The gate's numbers were lost with the child.
  - *Cause:* not identified. The log carried no allocator figures, and the plan writes every prefill into its
    pool at once, so nothing piled up there.
  - *Fixed (no protocol change):*
    - freed buffers go back to the device at every stage boundary, as KB2's probe does;
    - a step that runs out of memory runs once more after that and is recorded (`retried_out_of_memory`), and a
      second failure is raised as before;
    - each stage's log line carries the allocator's active, cached and peak gigabytes.
  - Run 4 is the same AG5-G at the fixed commit.
- **AG5-G run 4 (2026-11-07, kernel v4 at `b076efd`, two T4s, 0.4 session hours): valid stop by the rule. The
  agent measured everything and chose not to stack; no serving data.**
  - The measurement ran to its end in 1,115 s, with no step retried and a peak of 13.28 GB. The stage log read:
    | stage | seconds | active | cached | peak |
    | :-- | --: | --: | --: | --: |
    | base path timed | 32 | 7.19 | 6.11 | 11.61 |
    | gate reference (16 chunks) | 463 | 7.19 | 6.65 | 11.61 |
    | plan installed | 616 | 10.41 | 0.90 | 11.61 |
    | gate on the plan | 1,029 | 10.41 | 1.01 | 11.61 |
    | every width, short prompt | 1,067 | 10.41 | 3.05 | 13.28 |
    | every width, long prompt | 1,115 | 10.41 | 2.80 | 13.28 |
  - The gate passed: ratio 0.9988, 95 % 0.9967-1.0008, as in KB2-G.
  - The width curve, tokens per second: 18.6 at 1 row, 29.2, 33.0, 36.2, 40.7, **43.9 at 6**, 37.2 at 7 and
    20.4 at 8. The native path alone made 20.7, so AG5's rule picked width 6 (2.12x).
  - AG4's rule then refused the engine. One request alone on the plan would run at 0.897x the native path, below
    the 0.9 floor. Two paths do not fit the card either: 2 x 7.19 GB of weights plus the 3.2 GB pool and the
    reserve exceed it. So the agent kept the native path, and AG5-G stopped as its rule says.
  - *Where the 10 % goes:* the plan's one-row step was 50.4 ms on the short prompt and 57.8 ms on the long one
    (about 300 tokens), against the native path's 48.2 and 48.4 ms. The rise with context, 7.4 ms against 0.2 ms
    natively, is the plan's attention (KB3).
  - AG5 stays open. The width rule chose a width, but a stacked engine the agent chooses on this card is still
    missing. KB3 targets that.
- **AG5-G run 5 (2026-11-07, kernel v5 at `91610fb`, two T4s, 0.4 session hours): valid stop by the rule, on
  the other condition. No serving data.** It was identical to run 4 except for the FA2 kernel: only
  `ironmule/dense_gemma.py` differed, and the harness file was the same.
  - The plan served FA2 version 3 (kernel `622a8bfe...`, plan version 2). The gate read 0.9988020931047558, the
    same value to the last digit as run 4, so the plan's answers were unchanged.
  - One request alone now made 0.914x the native path (19.14 tokens per second against 20.94), above the 0.9
    floor. Width 6 made 44.18 tokens per second (2.11x).
  - The rule refused on the first token instead. On the long prompt the plan's first token took 0.810 s against
    the native path's 0.685 s, 1.18x, above 1.1. In run 4 the same two prefills had read 0.781 s and 0.742 s
    (1.05x). FA2 does not run in a prefill, so nothing in the kernel change touched these times.
  - *Post hoc (after run 5's data, labelled so; run 5's verdict stands):* both first-token times were single
    timings. The native path's own prefill moved by 8 % between the two sessions, enough to flip the rule.
    - Each side is now the median of several prefills: 5 fresh ones on the native path, and the plan's 8 slot
      prefills, which it ran anyway and of which it kept only the first.
    - Every sample is recorded (`base_first_samples_s`, `plan_first_samples_s`).
    - The rule itself is unchanged: 1.1x, on the worse prompt.
  - Run 6 is the same AG5-G at the commit with that measurement, run once. If the plan's prefill is really about
    15 % slower, the medians show it, and the rule refuses again.
- *Setup:* one Kaggle session with two T4s, `mlx-community/gemma-3-12b-it-4bit` at `86cc6a8`,
  `training/rl2_headroom.py --plan ag5g` at a pushed commit, about 1.6 session hours.
- *The agent decides:* the product's own `choose_setup` on card 0 (NUM1, then AG4's amended rule on gate 2's
  chunks). If it does not choose the stacked engine, the run stops and records why.
- *Arms, through the product, as AG4-C:*
  - `default`: today's reference worker with `--compute-dtype native`, one request at a time;
  - `agent`: the agent's setup, the native plan with the Gemma stacked engine at the agent's width.
  Each block (state, load, trace) runs default, agent, default on the same arrivals.
- *Traffic:* AG4-C's bursts and spike states at AG4-R's loads, 0.8 and 3.0 of the default path's calibrated
  capacity, one fresh trace each, 12 requests, a 256-token limit. Prompts are short (the question alone): a
  1,024-token slot cannot hold the long variant's roughly 1,200-token context plus 256 tokens.
- *Cards:* phase 1 runs default on card 0 and the agent on card 1; phase 2 swaps them, on the same traces. That
  makes 8 blocks.
- **Decision (AG4-C's analysis, unchanged):** pass if the agent's tokens per second over the default's (the mean of
  its two runs), paired by block, has its 2.5 % bound above 1, and the mean and 95th-percentile first token are no
  worse than the default's beyond the default's A/A noise.
  - A failed agent request voids its block. The default path's failures are censored at the 3,600 s deadline, as
    in AG4-C run 3. Fewer than two complete blocks make the run invalid.
- *Recorded, no decision:* the width the agent chose and its curve, the ratios per load and per phase, and the
  requests that joined a running group.
- A pass is the agent's gain on this card, model and traffic. It ships opt-in (`--agent on`).

### AG6 — The agent learns its serving policy from the traffic it serves, by dreaming (opened 2026-11-07, owner request, before data)

**Why (AG4-R run 1, descriptive).** On one T4 with Qwen 3 1.7B, the stacked engine alone made 2.10x the single
path's tokens per second and the agent's router 1.44x. The router decides one request at a time: each goes where
it would finish first. A request alone finishes sooner on the single path, so the router sends it there while that
path is free. Requests arriving meanwhile are left to the stacked engine, which then runs with few rows, so its
stacking advantage goes unused. Its online learning (recursive least squares on each path's prices) moved its
tokens per second by 1.007x over a frozen copy. The owner wants a self-learning agent that decides, pursued with
dreaming and RL (2026-11-07).

**Mechanism (`ironmule_product/path_router.py`, `PolicyRouter`).**
- *Modes:* `route` (today's router), `stacked` (every request to the stacked engine; one it cannot hold goes to
  the single path, as the pool now does) and `single`.
- *World model:* the router's learned prices for each path, plus an event replay of each path:
  - the single path serves one request at a time, its first token then a fixed price per token;
  - the stacked engine steps all its rows together at the learned price for that many rows, up to its width, and a
    newcomer waits for a free row and pays its first-token time;
  - both run on one card and share its time (repaired after the check below).
- *Dreaming:* every 16 finished requests, with at least 32 logged, the agent replays the arrivals and token counts
  of its last 64 finished requests in that model under each mode. It moves to the mode with the most replayed tokens
  per second whose replayed mean first token is at most 1.1x the current mode's. It moves only if the gain is at
  least 5 %. The replay runs after a request finishes, never inside a routing decision; a decision stays a lookup
  plus, in `route`, the router's arithmetic, a few microseconds.
- *No answer changes:* each path serves its own answers, and both passed their gates (AG4).

**Check in simulation (fixed now, before it runs; AG4-R run 1's data already exists, so this is a model check,
not evidence for a claim).**
- On run 1's 20 complete blocks, the dreamer starts from the prices the agent measured at startup.
- For each block it replays the block's arrivals and token counts under the three modes and names its best mode
  by the rule above. That mode is compared with the measured arms: `single`, `router` and `stacked` on the same
  arrivals.
- **Pass:** the named mode is the measured best arm by tokens per second, or within 5 % of it, in at least 80 % of
  the blocks. Its replayed stacked-over-single tokens per second, pooled, must lie within 25 % of the measured 2.10.
- If it fails, the world model is repaired before any hardware run, as V3's replay was before V4.
- **Check, first result (2026-11-07): a pass by the rule, hollow where it counts.** Sixteen of 20 blocks hit, and
  the replayed stacked-over-single ratio was 2.07 against the measured 2.10. Only the 4 spike blocks separate the
  modes, though: there the stacked engine measured 228-236 tokens per second and the router 131-136. On all 4 the
  dreamer named `route`, which it replayed at 236-244. The other 16 blocks are arrival-bound, and every mode
  serves them alike.
- **Post hoc repair (after the check's data, labelled so; no hardware data yet).** The frozen router, which kept
  its startup prices, also made only 129-133 in those spikes, so learning was not the cause. The replay had
  treated the two paths as two free servers, but both run on one card. While both have work, they share its time.
  The world model now gives the card to the two paths a unit each in turn: a newcomer's first token, or one decode
  step of all of a path's rows.
- **Check, re-run on the repaired model:** 20 of 20 blocks hit, and in all 4 spike blocks the dreamer names
  `stacked`. The replay there reads route 124-126, stacked 211-212 and single 57, against measured 131-136, 228-236
  and 65. Pooled, stacked over single replays at 2.07 against the measured 2.10. Since the repair followed the data,
  this shows only that the repaired model fits run 1. The evidence is AG6-H.

**AG6-H, the hardware test (fixed now, before its data).**
- *Setup:* one Kaggle T4, `mlx-community/Qwen3-1.7B-bf16` at `9cd6692`, `training/rl2_headroom.py --plan ag6h`
  at a pushed commit. The product's `choose_setup` runs first; if it is not `route`, the run stops and records why.
- *Arms, through the product on the same two loaded workers:*
  - `single`, forced;
  - `router`, today's agent, learning as in AG4-R;
  - `agent`, the `PolicyRouter`, starting in `route` from the same prices, learning and dreaming across blocks;
  - `stacked`, forced, with the pool's fallback;
  - then `single` again, for the A/A noise.
  The middle three rotate by block.
- *Traffic:* RL2-H2's bursts and spike states at loads 0.8 and 3.0 over the single path's calibrated capacity, 4
  fresh traces each (16 blocks), 32 Dolly requests, a 256-token limit, as AG4-R.
- *Cost, from run 1's measured cells:* about 300 s a cell in bursts and 62 s in a spike at load 0.8, and an assumed
  150 s at load 3.0. That gives about 4 session hours. The harness starts no new block after 10 hours and records
  a time cut as such.
- **Decision:** the agent passes if its tokens per second over the router's, paired by block, have a 2.5 % bound
  above 1 (2,000 resamples), and its mean and 95th-percentile first token are no worse than the router's beyond
  the A/A noise of the two single runs. A failed request voids its block; fewer than 4 complete blocks make the run
  invalid.
- *Recorded, no decision:* the agent over the single path and over the forced stacked engine, both with 95 %
  intervals; which mode the agent held in each block, and when it switched; the pool's fallbacks.
- A pass is a hardware gain from learning while serving on this card, model and traffic. LIMITS' agent section
  then says so, and the policy learner ships opt-in under `--agent on`.
- *Harness (built 2026-11-07, before any AG6-H data):* `measure_ag6h` and `analyse_ag6h` in
  `training/rl2_headroom.py`. Blocks run trace by trace, each trace's four load-and-state blocks before the next
  trace, so a time cut leaves every load and state measured. The stacked worker serves at the width AG5's rule
  chose. Before the hardware run, the tests cover the arms, the rotation, the mode record, the time guard and the
  decision rule with stand-in workers.
- **AG6-H result: passes (2026-11-07; kernel `tobayko/ironmule-rl2-headroom-ag6h` v1 at `c58fb4b`, one Kaggle T4,
  3.2 session hours).** The agent's own setup was `route`: the float16 single path beside the stacked engine at
  width 8, because one request alone ran at 0.81x on the stacked engine. All 16 blocks completed, no request
  failed, and no time cut was needed. The A/A noise of the two single runs was 1.3 % in tokens per second and 2.7 %
  in the mean first token.
  - **Decision:** the agent over the router, paired by block, made 1.512x the tokens per second (95 %
    1.335-1.639). Its mean first token was 0.553x the router's (95 % 0.445-0.622), its 95th percentile 0.487x
    (95 % 0.336-0.613). Both conditions hold, so AG6 passes by its rule.
  - *Recorded:*
    - the agent over the single path, 2.58x (95 % 2.08-3.03), with first tokens at 0.19x (mean) and 0.21x (p95);
    - the agent over the forced stacked engine, 0.9925x (95 % 0.9877-0.9967);
    - the router over the single path, 1.70x, and the stacked engine over it, 2.59x;
    - the agent over the router by load and state: 1.019x (0.8, bursts), 1.551x (0.8, spike), 1.294x (3.0, bursts)
      and 1.734x (3.0, spike).
  - *What it learned:* its first dream, after block 0's 32 requests, moved it from `route` to `stacked`, and its 30
    later dreams kept `stacked`. Its gain over the router is that switch, learned from the traffic it served.
    Afterwards it served as the forced stacked engine does. Block 0, served wholly in `route`, read 0.965x the
    forced arm, and blocks 3 and 13 read 0.980x and 0.967x; the other 13 blocks were 0.993-1.002x. The run does not
    show it moving back where another mode wins, because on this card, model and traffic `stacked` won in every
    load and state. A decision took 0.85 us, against the router's 11.7 us.
  - *Correctness:* each of the 128 prompts was answered 20 times (4 blocks by 5 arm runs). Every prompt had at
    most two distinct answers, the single path's and the stacked engine's, both gated (AG4); 8 prompts had one. The
    single path's answers were identical in every block and in both of its runs.
  - *A product defect, found in this run's data:* the pool fell back to the single path 15 times. In block 13 the
    forced stacked arm handed 8 short requests to the single path, which served them one after another. The
    worker's result loop rebound the dense plan's slot limit to the last member's output limit
    (`ironmule_product/worker.py`, since `562ccd25` on 2026-10-30). So from a batch's second run on, a request
    joining it was rejected as too long for the engine (prompt plus 256 tokens against 256). In this route setup
    the single path then served it. Under `--engine dense` alone the client would have got a 400, and under
    `current_profile` the same happened once a batch had a second run. It is fixed with tests after this run (`359f8d6`). All
    arms ran on the same worker, so the comparison stands. The fix can only keep more rows on the stacked engine.
  - *Ships:* the policy learner serves opt-in under `--agent on` wherever the agent's setup is `route`. The claim
    is bounded to one T4, Qwen 3 1.7B bf16 at `9cd6692`, these four load-and-state cells, 32-request blocks with
    256-token answers and the product at `c58fb4b`. The ledger entry is still owed.

### AG6-Q — Quiet, a spike, quiet again: does the agent follow a load change on hardware? (owner request 2026-11-08; fixed now, before its data)

**Why.** AG6-H measured blocks of one state each, and REV1's first-token rule for quiet traffic is unmeasured on
hardware. The owner asked for one continuous load change: does the repaired agent decide right in each phase,
and do its guards hold?

**Setup.** One Kaggle T4, `mlx-community/Qwen3-1.7B-bf16` at `9cd6692`, `training/rl2_headroom.py --plan ag6q`
at a pushed commit carrying REV1, the cursor fix and the observed quiet guard (`f2d82e5` or later). The product's
`choose_setup` runs first; if it is not `route`, the run stops and records why. Both paths are loaded once, and
every arm uses the same two workers. Before the first repetition, 16 calibration prompts go at once through each
path to warm it; that is not measured.

**Trace** (one per repetition, seed 450 + r; held-out Dolly prompts as AG4-R; `max_tokens` 64):
1. *Quiet:* 32 requests at a constant 3.0 s. Alone, one takes about 1.2 s on the single path and 2 s on the
   stacked engine, so in every mode no request overlaps another.
2. *Spike:* 48 requests 3.0 s after the last quiet one, all within 0.5 s.
3. *Quiet again:* 80 requests at 3.0 s, from 90 s after the spike's first arrival. By then even the forced single
   path has served the spike, about 48 x 1.2 s.

**Arms**, each run once per repetition on the same trace, in an order rotated by repetition, the card idle between
them:
- `agent`: the `PolicyRouter`, fresh from the startup prices, starting in `route`;
- `route`: the `PathRouter`, fresh, per-request routing (a fixed mode);
- `stacked`, forced;
- `single`, forced;
- `busy`, a simple adaptive rule: the stacked engine while any request is outstanding on the card, else the single
  path.

There are 3 repetitions, and the harness starts none after 3 hours.

**Measured per phase and arm** (pooled over repetitions, and each repetition's value):
- tokens per second over the phase (from its first arrival to the end of its last request);
- mean and 95th-percentile first token;
- requests, failures and cancellations;
- the agent's mode at each request;
- its switches (time, request, from, to and the rule);
- the time to adapt: from a phase's first arrival to the agent's first switch into `stacked` in the spike and back
  into `route` or `single` in the second quiet phase, censored at the phase's end;
- the cost of a decision (mean and maximum µs), apart from the cost of the dreams (count, mean and maximum ms).

**Uncertainty.**
- First-token ratios come with a paired bootstrap over requests: the same request of the same repetition under
  both arms, 2,000 resamples, 95 %.
- Tokens per second come as the ratio in each repetition, with their mean.

**Decision (fixed now).** The agent passes if, in each of the three phases:
1. its mean first token is at most 1.1x that of the best fixed mode in that phase, the one with the earliest mean
   first token among `single`, `stacked` and `route`; and
2. its tokens per second are at least 0.95x the best fixed mode's in that phase, the one with the most.

Both are pooled over the valid repetitions: the mean first token over all their requests in the phase, and tokens
per second as all their tokens over the sum of their spans. A phase's span runs from its first arrival to the end of
its last request.

A failed request voids its repetition, and fewer than 2 valid repetitions make the run invalid. Every answer must
be one of the two gated paths' own: per prompt at most two distinct answers across arms.

**Recorded, no decision:** the simple rule's same ratios; the best fixed mode per phase and over the whole trace;
the adaptation times; the switches; the decision and dream costs.

**Expected risk, stated before the data.** The agent dreams after 32 finished requests and then every 16, on its
last 64, and it scores tokens per second over the window's whole span.
- In the spike, a window that still holds most of the first quiet phase may dilute the spike's gain below the 5 %
  switch, so the agent may serve the spike in `route`.
- If it does switch to `stacked`, it can return only once the spike has left its 64-request window. That is at the
  144th finished request, the second quiet phase's 64th: about 190 s after that phase's first arrival, with about
  64 of its 80 requests served stacked. Its mean first token there would then be far above the single path's.
- If a fixed mode or the simple rule wins a phase, that is the result, and it names the next step.

**Offline prediction (world model only; made 2026-11-08 at 02:15 UTC, after the kernel started and before any of its
data was read; a scratch script, not a claim).** The world model in `path_router.replay` served AG6-Q's trace in
closed loop, with AG6-H's startup prices as both the card and the agent's model and the mode in force at each
arrival. All three seeds gave the same result:
- The agent serves the quiet phase in `route`, as the single path does (first token 0.063 s).
- It serves the spike in `route`, at 0.81x the stacked engine's tokens per second, with mean first tokens 1.12x
  later. It switches to `stacked` only at the 80th finished request, the spike's last, after the spike.
- It returns to `route` at the 144th finished request, 191 s into the second quiet phase. There its mean first token
  is 3.8x the single path's.
- So the prediction is a fail in the spike (both conditions) and in the second quiet phase (first tokens).
- The simple rule passes every phase: 0.97x the stacked engine's tokens per second in the spike, first tokens 1.04x
  later, and the single path's first tokens when it is quiet.

**Cost:** about 0.15 session hours of setup and warm-up, and 7.1 minutes per arm and repetition (the trace spans
425 s), about 2 session hours in all.

*Harness (built 2026-11-08, before any AG6-Q data):* `trace_ag6q`, `BusyRule`, `measure_ag6q`, `analyse_ag6q`
and `run_ag6q` in `training/rl2_headroom.py`; AG6-H and AG6-Q now share one start (`_route_paths`). The router
reports what the run records: each switch's time and rule, the mean and maximum cost of a decision, and the count,
mean and maximum cost of its dreams. Before the hardware run, the tests cover the trace, the simple rule, the arms
and their rotation, a fresh agent per repetition, the decision record, the time guard, the decision rule and the
adaptation times on stand-in workers. The dream runs on the thread of the request that finished, before that
request's last event goes out, so its cost also delays that request's end; the run records it.

### STAB2 — The agent never stops the server or fails a request (owner decision 2026-11-07, shipped)

The owner asked that aborts stop and that nothing crash. Today's runs had shown how the agent could stop work.
In AG5-G run 1 its own measurement ran out of memory, so it lost the stacked engine; in run 2 the measurement
timed out, and the timeout raised through `choose_setup` and ended the run. A chosen engine that did not start, or
`route` on several cards, stopped `serve`, and a router or dream fault would have broken the response stream it ran
in. None of it is a performance change, and no
decision moves, so it needs no hardware evidence; each case has a test that injects the failure.
- *Setup:* a failure anywhere in the agent's measurement (`cli._agent_setup`, `plan_choice.choose_setup`: the
  numeric gate, the stacked measurement and its child, the rule, an unreadable child result, a store of another
  shape or on a read-only disk) leaves the reference engine on the checkpoint's dtype, with the reason.
- *Start:* when the engine the agent chose does not start (a card out of memory, a kernel that fails), or it chose
  `route` on several cards, its gated numeric plan serves on the reference engine, else the checkpoint's dtype. It
  is printed and shown as `agent_setup.fallback_from`. A plan the caller named is kept and its failure reported.
- *Serving:* a router decision that fails serves the request in placement order (`router_faults`); a learning or
  dream fault keeps the mode and lets the answer go out (`errors`, `last_error` in `/health`'s `paths`). A
  rerouted request is now priced at its new path's load as it stands, not as it was when the request came: AG6-H
  had taught the single path a 7.3 s first token from requests that waited there behind earlier refusals.
- *Harness:* a Kaggle run that raises records the error and its traceback in its report, keeps what it measured
  and ends in order (`training/rl2_headroom.py`), so no session ends without a result to read.
- *Not covered:* a worker that dies mid-request is STAB1's (it is replaced, and the request fails as before); a
  driver fault that takes the whole process down is outside Python's reach.

### REV1 — An external review of 2026-11-08: eight findings, all reproduced and fixed (shipped)

A review of PRs 21 to 23 and of this branch at `5f064c3` came with four read-only scripts. They run the
repository's own functions on synthetic inputs, with no model, GPU or network. All four reproduced their
findings here (each script exits 0 while its fault is present). Each fix below has a regression test that fails on
the old code, and each script now fails its assertion. The serving faults are on `main` as well.
- **P1, the agent's qualification stalled for good.** The triplet cursor counted used arrivals, but the traffic
  memory keeps 2,048. Once 2,048 were used, no window was found again, and the open candidate blocked replanning.
  *Fix:* the cursor is the last window's recorded end time (`Agent._next_window`).
- **P1, a candidate that loses could be promoted.** Eleven triplets against a schedule that was then withdrawn were
  judged together with one against the reference that served next. That promoted a candidate costing 50 % more than
  the reference. *Fix:* a candidate's evidence is bound to the incumbent it ran against, and starts over when the
  incumbent changes.
- **P1, early failures never withdrew a policy.** `_agent_observe` dropped any group with a request that failed
  before its first token, so the live monitor never saw those failures. *Fix:* such a group's outcomes still
  reach the monitor (`Agent.observe_outcomes`); so do those of groups a request joined.
- **P1, cancelled joiners kept computing.** A joined row's stream had no cancel check, so a joiner whose client
  had gone computed its whole limit. *Fix:* the worker's admission hands each joiner its own check (a fifth field),
  and the bridge passes it to the executor.
- **P1/P2, AG6 stayed stacked on quiet traffic.** On quiet traffic every mode's tokens per second is the arrival
  rate, so the 5 % rule never left a mode whose first tokens came 10x later. *Fix (an amendment after AG6-H,
  labelled so):* where no mode wins on throughput, the earliest mean first token wins among the modes with at
  least 98 % of the current mode's tokens per second, if it is at most 0.8x the current mode's. This applies on
  quiet traffic only: at most 1 in 16 requests may arrive while another is being served (`DREAM_QUIET`).
  - Without that limit the rule would have broken AG6-H. In its bursts at load 0.8, with the startup prices, the
    replay priced the router's first tokens 25 % earlier than stacking's, while the card measured stacking's 5x
    earlier (0.36-0.40 s against 1.7-2.2 s). Under load the replay's first tokens are not reliable.
  - *Checks, all replays:* AG4-R run 1's model check is unchanged (20 of 20 blocks, the same named modes). Replayed
    on AG6-H's 16 blocks, no block leaves `stacked`, with the startup prices or with the agent's learned ones. The
    review's quiet trace now moves to `route`.
  - This changes nothing AG6-H measured. On quiet traffic the choice is a replay of learned single-request prices,
    not measured on hardware (LIMITS).
  - **Correction, the guard's data source (owner check, 2026-11-08).** As shipped in `4ba5550`, "quiet" counted
    overlaps inside the replay. Whether a path was busy at a replayed arrival came from the model's prices, so the
    guard rested on the very estimate it was meant to guard against. It now rests on observation alone
    (`path_router.observed_quiet`).
    - *Source:* the router's own served intervals. `PolicyRouter.watch` records each request's arrival (when the
      pool hands it the stream) and its end (when its `done` passes), both on `time.monotonic()`. It also counts
      requests in flight, and notes the arrival of any that ended without its answer: failed or cancelled, but
      not rerouted on an engine refusal, which the other path serves and the router watches again.
    - *Definition:* the window is the last 64 finished requests a dream replays, at least 32. Up to 64 earlier
      finished ones are kept as possible overlappers. A request of the window overlapped if it arrived while
      another observed request was being served, that is arrived_i <= arrived_j < ended_i (half-open: arriving
      the instant another ends is not an overlap; two that arrive together both overlap).
    - *Quiet:* at most floor(n / 16) of the window's n requests overlapped (4 of 64, 2 of 32). The observation must
      be complete: at least 32 requests, none in flight, and none in the window's time span that ended without
      its answer. A caller without observed intervals gets no first-token move.
    - *Checks, rerun:* AG4-R run 1's model check, now fed the router arm's served intervals, is unchanged at 20 of
      20 with the same named modes. None of its blocks is quiet by observation, not even the quiet state at load
      0.8. Replayed on AG6-H's 16 blocks, no block is observed quiet and none leaves `stacked`, with the startup
      prices, the learned ones, or prices made to favour routing 500x.
    - The threshold of 1 in 16 is a heuristic for the hardware test (AG6-Q) to check.
- **P2, a technical measurement failure was cached for good.** `choose_setup` stored the base path after a child
  had crashed or run out of memory, and reused it after the cause was gone. *Fix:* such a setup serves once, marked
  `measurement_failed`, and is measured again on the next start; a refusal by a rule or a gate is still stored.
- **P2, constant predictions scored a perfect rank correlation.** Tied values got consecutive ranks. *Fix:* ties
  share their average rank, and a constant input is undefined (None), as in SciPy. RL4's own copy of the function
  had the same fault and now uses the product's.
- **P2, the queue bound could be exceeded.** Admission bounded only the queue, while a shortest-first order moves
  waiting work to the carry. *Fix:* the bound is on queue plus carry.
- The review also read AG6-H as a limited positive selection result: 1.512x over the router it replaces, but
  0.9925x the forced stacked engine, one switch and no switch back. A lasting advantage over the best fixed mode
  is not shown. That agrees with AG6-H's record and LIMITS.

### TP1 — Kaggle TPUs as a further device the agent can choose (opened 2026-11-05, owner suggestion, before data)

The owner offered TPUs. Kaggle gives this account 20 TPU hours a week (TPU v5e-8 by default), apart from the 30
GPU hours, and none of them has been used yet.
- *The gap:* MLX, IronMule's runtime, has no TPU backend, so nothing in the product can use one today. A TPU path
  would be a second runtime (JAX), with the model's own bf16 weights, never changed or shrunk (owner boundary,
  HQ1). It would sit beside the T4 paths as one more path for the agent to measure and choose (AG4).
- *Mechanism:* a TPU v5e chip does bf16 matrix products natively, with far more memory bandwidth than a T4 (which
  emulates bf16, U1). Decode is bound by streaming the weights, so a decode step could be much shorter there.
- *Kill:* TP1 is closed unless a JAX decode path for `Qwen3-1.7B` on one v5e chip makes more tokens per second
  than the T4's best path measured so far (57.7 alone with float16, 290.9 at width 8 with the dense plan, AG4-Q
  run 2), at the agent's quality gate. It is never built if TP0 shows that even its matrix products alone cannot
  reach that.
- **TP0, a probe before building anything (fixed 2026-11-05, before its data; descriptive only, no claim).**
  `tools/tpu_probe.py` on one Kaggle TPU session records:
  - the devices JAX sees, the host CPU and its memory;
  - whether MLX's CPU build installs on the host;
  - the median of 20 timed bf16 products, after 3 warm-ups, at one decode step's shapes of `Qwen3-1.7B` (q, k, v,
    o, gate, up, down; 28 layers), for 1, 8 and 64 rows, on one TPU chip with JAX and on the host CPU with MLX.
  - *Read as:* each runtime's matrix time for one decode step and the tokens per second that time alone would
    allow (a ceiling, not a speed: attention, norms, sampling and the host are left out).
  - *Decides:* if the TPU's ceiling at 1 row is not above 57.7 and at 8 rows not above 290.9, TP1 is closed in
    Tier 0. Otherwise the next step is a JAX decode path, fixed here before it is built.
  - *Cost:* about 0.3 TPU hours of this week's 20, which lapse on 2026-11-07 unused.
  - **Queued, not run (2026-11-05).** The kernel `tobayko/ironmule-tpu-probe` at `08c96d4` was pushed at 04:58 UTC
    and was still waiting at 16:35 UTC. It spent no TPU hours. When the owner tried a TPU session in Kaggle's web
    editor that afternoon, Kaggle answered "TPUs are popular right now. You are #789 in the queue." So the wait is
    Kaggle's shared TPU queue, not the API submission. The kernel stays queued; if it has not run by 2026-11-07
    00:00 UTC, this week's TPU hours lapse and TP0 waits for the next week.
  - **Still queued after more than 24 hours (2026-11-06, 07:00 UTC).** The kernel submitted through the API had not
    been given a TPU 26 hours after its push. A status poll every 3 minutes from 23:00 to 01:01 UTC read `QUEUED`
    all 40 times, as did the check at 07:00 UTC. It has still spent no TPU hours. The owner is waiting in the web
    editor's TPU queue as well. No further check-in is scheduled this week; if neither run gets a TPU before
    2026-11-07 00:00 UTC, TP0 moves to the next week's quota.
  - **Result (ran 2026-11-06, about 20:35 UTC; kernel `tobayko/ironmule-tpu-probe` v1 at `08c96d4`; about 110 s,
    roughly 0.03 TPU hours).** The session had a TPU v5 lite with 8 chips (JAX 0.11.0) and an x86_64 host with 224
    CPUs and 377.8 GiB. MLX 0.32.4's CPU build installed there. Matrix time for one decode step and the ceiling it
    allows:

    | Runtime | 1 row | 8 rows | 64 rows |
    | --- | --- | --- | --- |
    | JAX, one v5e chip | 48.1 ms, 20.8 tokens/s | 44.2 ms, 180.8 tokens/s | 46.1 ms, 1,388 tokens/s |
    | MLX, host CPU | 1,225 ms, 0.8 tokens/s | 5,049 ms, 1.6 tokens/s | 37,382 ms, 1.7 tokens/s |

    *Verdict by the fixed rule:* the TPU's ceiling is 20.8 at 1 row (not above 57.7) and 180.8 at 8 rows (not
    above 290.9), so **TP1 is closed in Tier 0**. The ledger entry in the research repository is still owed.
  - *Post hoc reading (after the data; it does not change the verdict):* every JAX product took 0.20-0.29 ms
    whatever its size, from 2048x1024 to 6144x2048 and from 1 to 64 rows, which is 58-64 GB/s of weights against
    the chip's much higher memory bandwidth. Each product was jitted and timed alone to completion, so this is a
    per-call floor (dispatch and host sync), not the chip's streaming speed, and the probe cannot say what a
    decode step compiled as one program would reach. A reopening needs that changed implementation: a probe that
    times a whole jitted decode step, fixed before its data.

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

**RL2-H, the headroom measurement (protocol fixed 2026-10-28, before data).** `training/rl2_headroom.py`, one
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
validity as RL2-H (`training/rl2_headroom.py --plan h2`). *States:* `quiet` (run 1's light level: Poisson at 0.2 of
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

**Test (RL3-H, `training/rl2_headroom.py --plan h3`).** The same pool, model, engine, traffic (run 2's `quiet`,
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

**Test (`training/rl2_headroom.py --plan x1`, Kaggle T4 x2, about half a session hour).** The model and revision are
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
- *Hardware (RL4-H):* `training/rl2_headroom.py` gains a `learned` arm that applies the frozen policy. It runs
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

**Test (`training/rl2_headroom.py --plan h5`, Kaggle T4 x2, about 1.5 session hours).** RL3-H run 2's pool, model,
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
policy only in `bursts` and `spike`. **Test (`training/rl2_headroom.py --plan h7`, Kaggle T4 x2, about 2 session
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

**Test (`training/rl2_headroom.py --plan h6`, Kaggle T4 x2, about 2 session hours).** RL3-H run 2's pool, model,
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

### HQ1 — Levers read from HyperQwen, Strata and vLLM, kept for the agent to use (opened 2026-11-03, owner request, before data)

**Owner's boundary (2026-11-03): the model is never changed or made smaller.** No new quantisation of weights, no
pruned or distilled models, no drafter trained in the model's place. Only how the given model runs on the hardware:
kernels, batching, scheduling, memory layout, speculation that the model itself verifies. Every route below stays
inside that boundary. The int8 head from the first version of this entry re-quantised weights, so it is dropped.

**Source.** [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) (read 2026-11-03). It is a patch series on
vLLM for Qwen3.8-27B on one RTX 3090. By its own measurements it takes one request from 46 to 127 tokens a second
(2.8x), and 64 concurrent requests from 370 to 1,035 (2.8x). Every setting was tuned by hand for that one model and
card. The owner asked that these levers be remembered, applied, and brought into the agent. The agent's lever list
(`ironmule.plan_choice.LEVERS`, shown in `/health` under `agent_setup.levers`) names each one, built or not. A built
lever passes the same gate and rule as every plan there (AG4).

**Why it is fast, in its own numbers:**
1. *Speculative decoding* (most of the single-request gain). A small drafter proposes 4-7 tokens, and the model
   verifies them in one memory-bound step, so about 3 tokens land per step.
   - The drafter was made cheap: int4, and a 40k-token draft vocabulary counted over the model's own outputs
     (+10 % on its own).
   - Drafts are also copied from the prompt when the answer repeats it: up to 381 tokens a second.
2. *Tensor cores for batched decode* (most of the concurrent gain). At 40-64 rows a decode step is bound by
   arithmetic, not by memory. int8 activations on int8 tensor cores run 4x the math.
3. *Memory for more concurrent requests.* The embedding and head go to int8 (2.6 GB freed), and the recurrent state
   goes to fp16: 37 resident requests became 64.
4. *Verify-step kernels.* Split-KV attention for a few query rows (FA2 used 24 of 82 SMs there), and a sampler that
   never sorts.

**The routes for IronMule, in this order (none is built yet; each needs its protocol here before its data):**
- **TC1, tensor-core rows for the stacked plans.**
  - *Why:* KB2's smoke showed the same wall. The native row kernel's scalar float32 FMAs add about 15 ms per further
    Gemma 3 12B row on a T4, so 8 rows give about 2x instead of the bandwidth bound's 8x.
  - *Mechanism:* the T4 (sm75) has fp16 (65 TFLOPS) and int8 (130 TOPS) tensor cores. A row kernel on `mma.sync`
    keeps each row's arithmetic independent of the others: fixed tile order, rows zero-padded to the tile. So the
    plan stays batch-invariant.
  - *Gate:* it changes numerics, so it is a numeric plan with the perplexity gate, which the agent may then choose.
  - *Kill:* the stacked 8-row step must fall below half of KB2-0's measured 8-row step, with every row still
    bit-identical alone and stacked.
- **SP1, speculation from the context.**
  - *Why:* AG4-Q showed the stacked engine losing for one request alone. Speculation is the single-request lever.
  - *Mechanism:* drafts copied from the request's own prompt and earlier answer (no drafter to train), verified by
    the stacked plan's rows. KB1's rows verify up to 8 positions bit-identically to one row, at 1.8x one row's cost
    (KB1-0).
  - *Why it is allowed now:* prompt-lookup speculation was dropped earlier because it changed tokens on the T4
    (HW3). Under the owner's AG4 decision a plan may change answers if it passes the gate.
  - *Kill:* a single request through the product must reach at least 1.3x the base path's tokens per second on
    Dolly prompts, with time to first token no worse.
- **KV8, an 8-bit KV cache in the stacked plans.**
  - *Mechanism:* keys and values stored in int8 per token and head, and dequantized inside FA2's chunk loop. That
    halves the pool, so Gemma 3 12B fits 16 slots instead of 8 on a T4, or longer slots.
  - *Kill:* the perplexity gate, and AG4's rule on the larger pool.
  - The cache holds the model's activations at run time, not its weights; the model is unchanged.
- *Not carried over:*
  - the int8 head (it re-quantises weights; the owner's boundary);
  - HyperQwen's requantised drafter and its draft vocabulary (a second model);
  - the recurrent-state precision (no such layer in our models);
  - vLLM-specific patches.

**From Strata ([mach10x/strata](https://github.com/mach10x/strata), read 2026-11-03).** It is llama.cpp-based, and
runs a 125B mixture-of-experts model on a 12 GB card. What concerns running the model, not shrinking it:
- *An online draft policy, which is a decision model of exactly the agent's kind* (`src/spec/draft_policy.cpp`):
  - for each verify window size, an exponential moving average of the measured round time;
  - for prompt-lookup drafts, an acceptance rate per match-length bucket, with a prior worth 4 observations that
    real windows soon override;
  - each round it picks the window size with the most expected tokens per millisecond, and drafts only where that
    beats plain decoding by a margin;
  - a size that has never been measured is probed a few times before its guessed cost may veto it.
  - Measured there: code edits 6-11 % faster, other text unchanged, output the same (the model verifies).
  - **SP1 takes this policy as its decision**, learned online per machine like the router's costs.
- *An expert cache:* the most-used experts of a mixture-of-experts model fill spare VRAM, and the CPU computes the
  rest in place at the same time; the cache adapts to the conversation.
  - **EC1** (a route for when a mixture-of-experts model is served; `cuda_native` already runs Gemma 4 26B-A4B's
    experts): the agent learns which experts are hot on this traffic and keeps them on the card.
  - *Kill:* tokens per second over the plan without the cache, through the product.
- *Prompts in chunks of 2,048-8,192 tokens* (see CP1).

**From vLLM ([vllm-project/vllm](https://github.com/vllm-project/vllm) docs, read 2026-11-03).**
- **CP1, chunked prefill with decode first** (`docs/configuration/optimization.md`). Long prefills are cut into
  chunks and batched with the running decode steps, decodes scheduled first. A token budget per step trades
  inter-token latency (small) against time to first token (large).
  - *Why for us:* AG4-Q measured the dense engine's first token at 0.36 s on a 300-token prompt. A newcomer's
    prefill stalls every running row (H10's joins), and AG4-R's smoke saw the stall dominate short answers.
  - *Mechanism:* the stacked plan prefills a joiner in chunks between decode steps. The chunk size is a knob the
    agent learns, like vLLM's `max_num_batched_tokens`, from measured step times.
  - *Kill:* AG4-R-style traffic through the product: tokens per second no worse and running rows' inter-token
    time lower, with time to first token no worse than one prefill at once.
- **PK1, a paged KV cache** (`docs/design/paged_attention.md`). Keys and values live in fixed-size blocks allocated
  as a request grows, not in a fixed slot per request. More requests and longer contexts fit the same memory, at the
  same precision. This also answers H11's slot limit.
  - *Kill:* concurrent requests that fit, and tokens per second at the product's context limit, no worse per step.
- *Batch invariance* (`docs/features/batch_invariance.md`): vLLM offers it only from compute capability 8.0. KB1 and
  KB2 give it on a T4 (7.5), which is why the stacked plans may take any width without moving an answer.
- *Already here:* CUDA graphs (MLX's CUDA backend captures graphs) and prefix caching (the product's prefix reuse).

### KB2 — Gemma 3 at 4 bit stacked into one decode step (opened 2026-11-03, owner decision, before data)

**Why.** The owner's measure is tokens per second, with time to first token no worse. A 4-bit Gemma 3 under
the `native` plan runs one request at a time on the reference worker. Gemma 3 12B makes about 21 tokens a second
on a T4, because every step reads all 8 GB of weights for one token (AG3: about 53 % of the bandwidth). The
dense plan showed the lever on Qwen 3 1.7B bf16: eight stacked rows decoded 3.8x faster than today's grouping
(KB1-1), and through the product it cost 48.3 % less (H10). Order and width over batch-1 work cannot add tokens
per second (AG1, AG2). Stacking can, and it gives the agent a width that moves them.

**Mechanism.** `ironmule/dense_gemma.py`, a numeric plan of its own:
- The linears run `ironmule/cuda_native.py`'s 4-bit row kernel. It already takes up to eight activation rows, each
  in its own float32 chain with pinned roundings, and it is the native plan's decode path at one row.
- Attention runs FA2 from `dense_plan.py`, with three changes:
  - E = D / 32 values per lane, for Gemma 3's head dimension of 256;
  - the score scale as a constant (`query_pre_attn_scalar`);
  - a sliding window: a row reads [max(0, length - window), length).
  A row reads only its own slot and length, so the other rows and the pool's capacity cannot move its answer.
- RoPE runs per layer kind, with one position per row (local base, or global base with linear scaling). The
  embedding, the RMS norms with `1 + weight`, and GELU are the model's own.
- Prefill is MLX's own, one request at a time, with plain caches and mlx-lm's masks. Its keys come out in
  position order and go straight into the request's slot.
- *Off CUDA:* `tests/engine/test_dense_gemma.py` checks the plan against mlx-lm's Gemma 3 step by step, with
  FA2's semantics in plain MLX on a tiny random 4-bit model and prompts shorter and longer than its window. It
  catches a wrong window, RoPE scale or embedding scale (each mutation fails the test).

**KB2-0 (`tools/cuda_probe.py --plan kb2`, one Kaggle T4, about 1 session hour).**
- *Setup:* `mlx-community/gemma-3-12b-it-4bit` at `86cc6a8`, with the native plan installed as the product
  installs it.
- *Pools:*
  - eight slots of 1,024 tokens (3.2 GB beside the weights), with eight chat prompts, alternately short and with
    700 tokens of context;
  - then four slots of 2,048 tokens for rows of 1,300 tokens, past the 1,024-token window.
- *Runs:* 48 decode steps.
- **(a) Exact within the plan (a failure is a bug, fixed before anything else).** Every row's logits at every step
  are bit-identical between the row alone and stacked at 3 and 8 rows (main pool), and at 4 rows (long pool).
- **(b) The lever, kill.** The plan's stacked step at eight rows, over eight of today's single-request native
  steps:
  - five rounds, alternating, each a median over 48 steps;
  - a 2,000-resample bootstrap over the rounds;
  - it must have its 97.5 % bound below 0.5, so stacking at least doubles tokens per second at width eight.
  - Otherwise KB2 stops: the row kernel's weight reads do not amortise on this model.
- **(c) The numeric plans' quality gate, kill.** `ironmule/quality_reference.txt` is teacher-forced through
  today's native decode (mlx-lm's own model and caches) and through the plan, eight rows stacked. The perplexity
  ratio by `numeric_choice.ratio_interval` must pass below 1.005 on both paths:
  - the gate's 16 x 512-token chunks;
  - a long variant, 4 chunks scoring 128 tokens after a 1,536-token prefix, so past the window. It runs in the
    2,048-token pool, four rows stacked.
- **Recorded, no decision:**
  - (d) the plan at one row over today's native step. Quiet traffic would regress if it is slower;
  - the step time at every width 1-8, the cost curve the agent's width decision will price;
  - where the plan's greedy tokens first differ from today's.
- **Verdict:** (a), (b) and (c) together make the plan a candidate opt-in numeric plan for 4-bit Gemma 3. The next
  step is its stacked executor in the product (as `--engine dense` for Qwen 3 bf16), with the agent deciding its
  width. A product run on two T4s then measures tokens per second and time to first token against today's
  native reference worker, under a protocol fixed here before that run.
- **Smoke first:** a `--smoke` kernel on a T4 before the run checks the install, the kernels' compilation and the
  path. It measures nothing, and costs about 0.25 of the week's remaining 0.9 GPU hours.
- **Smoke result (2026-11-03, kernel `tobayko/ironmule-cuda-probe-kb2-smoke` v1 at `54f1894`, Kaggle T4, 7 minutes):
  the tool works end to end.**
  - The native plan installed (338 modules, probe error 0.0028), and the FA2 kernels compiled.
  - Every stacked row was bit-identical to the row alone, in both pools.
  - The plan's greedy tokens matched today's on all 8 prompts over the smoke's 5 tokens.
  - *Descriptive, a smoke measures nothing:*
    - one round of 4 steps gave 52.9 ms a step at one row against 52.1 ms for today's native step;
    - about 15 ms more for each further row up to 6 rows, 162 ms at 7 rows, and 354 ms at 8 rows;
    - the gate on 4 chunks of 12 tokens read 1.20 and 1.01, too few tokens to say anything.
- **Amended after the smoke, before KB2-0's data (post hoc to the smoke, not to the run):** (b) is read at the width
  with the most tokens per second on the median curve over the five rounds, fixed before the bootstrap, instead of
  at 8 rows. The ratio at 8 rows is recorded beside it. The agent would serve the best width, and the smoke's step
  jumped at 8 rows (register pressure in the row kernel at eight activation rows is the suspect, not yet shown).
  The bound (below 0.5) and (a), (c) are unchanged.
- **Run 1 failed, no data (2026-11-07; kernel `tobayko/ironmule-cuda-probe-kb2` v1 at `03617a0`, Kaggle T4, about
  6 minutes).** It stopped with `cudaMallocAsync ... out of memory` before any measurement, and the tool's report
  kept only the error. The likely cause, from the code (the report recorded no stage): the probe ran today's native
  path, for (d) and for the gate's references, while the plan's 3.2 GB pool was already allocated. That path
  computes the head at every prompt position through a float16 copy of the 262,144-row head (about 2 GB), which
  `cuda_native.head_skip_needed` lets fit beside the 8 GB of weights alone, not beside a pool as well. The smoke's
  pools were 64 and 128 tokens, so it could not show this.
  - *Tool fix, no change to the protocol:* every native reference that needs the head runs before any pool exists,
    and (b)'s native step fills its cache with the trunk alone (the same keys and values; the timed decode steps
    are unchanged). The report now keeps the stage reached, the allocator at each stage and everything measured
    before a failure.
  - *Run 2* is the same KB2-0 under the same rules, (b) as amended above.
- **Run 2: valid, does not pass; KB2 stops at its quality gate (2026-11-07; kernel `tobayko/ironmule-cuda-probe-kb2`
  v2 at `9d388e9`, Kaggle T4, MLX 0.32.4, mlx-lm 0.31.3, `mlx-community/gemma-3-12b-it-4bit` at `86cc6a8`, about
  26 minutes).** The run reached its end; the allocator peaked at 12.6 GB (11.1 GB during the native references
  before any pool), so the fix held.
  - **(a) passes.** Every row's logits were bit-identical alone and stacked: 144/144 steps at 3 rows, 384/384 at 8
    (main pool) and 192/192 at 4 rows (long pool, past the window). The install probe matched as well.
  - **(b) passes, narrowly.** The best width on the median curve is 6: six stacked rows cost 0.4985 of six of today's
    single native steps (95 % 0.4979-0.4991, below 0.5), so **2.006x the tokens per second**. Five rounds agreed
    within 1 ms. Step time by width, median ms: 48.8, 67.8, 85.1, 113.0, 122.1, 141.3, 190.3, 394.1 (widths 1-8);
    today's native step 47.3. Recorded: at 8 rows the ratio is 1.042, slower than eight single steps (the smoke's
    jump at 8 rows, now measured).
  - **(c) fails on the main path.** Main, 16 x 512 tokens: ratio 0.984, 95 % 0.825-1.177, not below 1.005. Long, 4
    chunks after a 1,536-token prefix: ratio 0.9994, 95 % 0.9950-1.0033, passes. The gate needs both.
  - **(d), recorded:** at one row the plan's step is 1.033x today's native step (3.3 % slower). Its greedy tokens
    matched today's for all 48 steps on 5 of 8 prompts and first differed at steps 14, 20 and 42 on the others.
  - **Verdict by the fixed rule: does not pass as a whole. KB2 stops at (c) and goes to Tier 0.** The ledger entry
    in the research repository is still owed.
  - *Post hoc reading (after the data; it changes no verdict):* the main path's failure comes from 3 of 16 chunks.
    13 differ from the reference by at most 0.12 nats a token, while chunk 1 is 0.95 worse, chunk 2 0.99 better and
    chunk 4 0.25 better. Both signs is not a uniform loss of precision. The main chunks start mid-text with no prefix
    and no BOS token, where Gemma 3's first position is known to carry outsized activations. The data cannot tell
    whether that start makes both paths unstable, or whether the plan has a bug at a one-token prefill. Telling
    them apart needs a diagnostic, not a re-gate: the per-token NLL of chunks 1, 2 and 4 under the plan, today's
    native path and mlx-lm's unmodified bf16 path. A bug found there would be a new implementation, and that would
    need a new gate, fixed before its data. The gate's text and chunking stay as fixed.
- **KB2-D, the diagnostic of those chunks (owner approved 2026-11-07; fixed now, before its data; descriptive only,
  it changes no verdict).** `tools/cuda_probe.py --plan kb2d` on one Kaggle T4, about 0.3 session hours, same model
  and revision. It records the per-token NLL of the gate's main chunks 0-4 (1, 2 and 4 diverged; 0 and 3 are
  controls) on seven paths:
  - `stock_oneshot` and `stock_oneshot_bos`: mlx-lm unmodified (MLX's own quantized matmul), the whole chunk in one
    forward, run before the native plan is installed;
  - `native_oneshot`: the native plan, the whole chunk in one forward;
  - `native_decode` and `native_decode_bos`: the native plan teacher-forced one token a step, the gate's reference;
  - `plan_decode` and `plan_decode_bos`: the Gemma plan, one row, teacher-forced the same way, the gate's plan side.
  The `_bos` paths put the tokenizer's BOS token before the chunk and score the same tokens.
  - *Agreement:* two paths agree on a chunk when their mean NLL differs by at most 0.12 nats a token (the largest
    gap among run 2's 13 close chunks).
  - *Check first:* `native_decode` and `plan_decode` reproduce run 2's chunk means (same code); if they do not, the
    run says nothing about run 2.
  - *Read as, on chunks 1, 2 and 4:*
    - **a plan bug is suspected** if `plan_decode` disagrees with `native_decode` while `native_decode` agrees with
      `stock_oneshot` or `native_oneshot`. The first position where the plan's per-token NLL leaves the native one
      by more than 1 nat is where to look. A fix is a new implementation and needs a new gate, fixed before its
      data;
    - **the starts are unstable for every path** if `native_decode` disagrees with both one-shot paths too, while
      the controls agree. No plan bug is shown then;
    - **the missing BOS is the trigger** if, on all five chunks, `plan_decode_bos` agrees with `native_decode_bos`.
  - KB2 stays in Tier 0 whatever this shows. The gate's text and chunking are not changed after the fact.
- **KB2-D result (2026-11-07; kernel `tobayko/ironmule-cuda-probe-kb2d` v1 at `e763e31`, Kaggle T4, 20 minutes,
  peak 11.1 GB).** The check holds: `native_decode` and `plan_decode` reproduce run 2's chunk means to four places.
  Mean NLL a token:

  | Chunk | stock one-shot | native one-shot | native decode | plan decode | stock + BOS | native + BOS | plan + BOS |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 0 | 2.923 | 2.914 | 2.921 | 2.917 | 3.037 | 3.049 | 3.042 |
  | 1 | 5.246 | 5.419 | 5.317 | 6.267 | 2.712 | 2.712 | 2.706 |
  | 2 | 6.445 | 5.977 | 6.688 | 5.696 | 2.748 | 2.738 | 2.739 |
  | 3 | 4.222 | 4.337 | 4.297 | 4.280 | 2.476 | 2.475 | 2.478 |
  | 4 | 4.992 | 4.907 | 5.142 | 4.897 | 2.669 | 2.675 | 2.670 |

  - *Readings by the fixed rule:*
    - **The missing BOS is the trigger: yes.** With BOS, the plan agrees with today's native path on all five
      chunks, within 0.000-0.007 nats a token, and no token differs by more than 1 nat on chunks 1, 2 and 4.
      Every path's NLL on chunks 1-4 falls from 4.2-6.7 to 2.5-2.7.
    - **Chunks 2 and 4: unstable for every path.** Without BOS, `native_decode` disagrees with both one-shot paths
      as well. On chunk 2, today's own one-shot and decoded paths differ by 0.71 nats a token.
    - **Chunk 1: a plan bug is suspected by the rule.** The plan is 0.95 above `native_decode`, which agrees with
      both one-shot paths (0.07 and 0.10). The plan first leaves it by more than 1 nat at token 14.
  - *Post hoc reading (after the data; it changes no verdict):* without BOS the independent paths themselves spread
    by up to 0.47 nats a token (stock one-shot against native one-shot, chunk 2), so chunk 1 may be the same
    instability with the plan landing furthest out. With BOS the plan sits within 0.006 of the native path on
    chunk 1 too. This run cannot rule a plan bug in or out on chunk 1. What it shows is that the gate's main
    chunks, which start mid-text with no BOS, put Gemma 3 where every path, today's included, is unstable. Chat
    prompts always start with BOS.
  - *What follows, for the owner to decide:* the gate's text and chunking are shared by every numeric plan and by
    the agent's on-machine gate (`ironmule/numeric_choice.py`), so changing them is a decision across plans, not
    a fix for KB2. If the gate gets BOS-led chunks, KB2 can be re-gated under a protocol fixed before that run.
    KB2 stays in Tier 0 until then. The ledger entry is still owed.

- **KB2-G, KB2 reopened under gate 2 (owner decision 2026-11-07; fixed now, before its data).** It reopens the Tier 0
  entry "KB2 Gemma 3 4-bit stacked decode plan", whose kill was (c): "The perplexity ratio by
  `numeric_choice.ratio_interval` must pass below 1.005 on both paths." The new hardware evidence is KB2-D: without
  BOS every path is unstable on the gate's chunks, and with BOS the plan is within 0.007 nats a token of today's
  native path on all five chunks tested. The gate itself changes for every plan (GATE2 below), not for KB2 alone.
  - *Run:* `tools/cuda_probe.py --plan kb2g` on one Kaggle T4, the same model and revision, about 0.5 session hours.
    The plan's code is unchanged since run 2. The run checks this: its install must report run 2's kernel SHA-256
    (`54b1d3a2...`) and plan version 1, or the run is invalid.
  - **(c) decides, as fixed for KB2-0, on gate 2's chunks:** the main 16 x 512-token chunks, each after the BOS
    (chunk 0 already starts with the text's own BOS), eight rows stacked; the long 4 x 128 tokens after a
    1,536-token prefix that starts with the BOS, four rows stacked. Both paths' upper bounds must be below 1.005.
  - **(a) decides again:** every row bit-identical alone and stacked, in both pools. A failure is a bug.
  - **(b) stays decided by run 2** (width 6, ratio 0.4985, 95 % 0.4979-0.4991). This run records (b) and (d) again,
    descriptive only. If the recorded ratio reads at or above 0.5, that is noted as a limitation, and the product
    run decides, since it measures tokens per second itself.
  - *Verdict:* (a) and (c) here, with run 2's (b), make the plan a candidate opt-in numeric plan for 4-bit Gemma 3.
    The next step is then the one fixed under KB2's verdict above. Otherwise KB2 returns to Tier 0 with this run's
    numbers.

- **KB2-G result: valid, does not pass; KB2 returns to Tier 0 (2026-11-07; kernel `tobayko/ironmule-cuda-probe-kb2g`
  v1 at `4a01f49`, Kaggle T4, 26 minutes, peak 12.6 GB).** The plan was run 2's: kernel `54b1d3a2...`, version 1,
  eight 1,024-token slots. Gate 2 led 15 of 16 main chunks with BOS; chunk 0 starts with the text's own.
  - **(a) passes again:** 144/144, 384/384 and 192/192 steps bit-identical.
  - **(c) main passes:** 16 x 512 tokens, ratio 0.9988, 95 % 0.9967-1.0008. Every chunk is within 0.010 nats a
    token of the native path.
  - **(c) long fails:** 4 x 128 tokens after a 1,536-token prefix, ratio 1.0003, 95 % 0.9944-1.0080, not below
    1.005. The gate needs both paths.
  - *(b) and (d), recorded:* at width 6 the ratio read 0.5074 (95 % 0.5022-0.5121, 1.97x tokens per second) on
    this card, whose native step was 48.8 ms against run 2's 47.3. Run 2's (b) decided, so this changes nothing.
    It is noted as a limitation: the margin over 2x does not hold on every T4. The ratio at width 8 read 1.039. At
    one row the plan was 1.040x today's step.
  - **Verdict by the fixed rule: does not pass. KB2 returns to Tier 0, and AG5-G, which needed KB2-G to pass, does
    not run.** The ledger entry is still owed.
  - *Post hoc reading (after the data; it changes no verdict):* the long path's chunk differences were -0.0073,
    -0.0006, -0.0022 and +0.0114 nats a token, a mean of +0.0003. The bound fails on one chunk whose gap is as large
    as the main path's widest (-0.0100). With 4 chunks of 128 scored tokens, 512 in all against the main path's
    8,192, the bootstrap cannot clear a 0.5 % bound with this spread. So the long path's failure is a matter of power
    as much as of quality: neither a loss nor its absence is shown past the window. In the product the plan at
    width 6 gets 1,024-token slots, which never pass the 1,024-token window. Whether the plan may serve there, scoped
    to the window, is a new claim; it needs its own protocol and fresh data, and the owner's decision.

- **KB2-L, the long path with power (owner decision 2026-11-07; fixed now, before its data).** It reopens KB2's
  Tier 0 entry once more. The kill it failed was (c), long path: "the perplexity ratio by
  `numeric_choice.ratio_interval` must pass below 1.005" on 4 chunks of 128 tokens past the window. The new
  evidence is KB2-G itself. Its main path passed, and its long path's mean was 1.0003, but 4 chunks, 512 scored
  tokens, could not clear the bound with the per-chunk spread the main path shows (up to 0.010 nats).
  - *Power, fixed before the data:* the bound is 0.005 nats a token (ln 1.005). The 95 % half-width of a mean over
    n windows is about 1.96 x SD / sqrt(n). At KB2-G's long spread (SD 0.0079 nats) that is 0.0078 for 4 windows,
    wider than the bound even at a true ratio of 1, and 0.0039 for 16. At the main path's spread (SD 0.0043) it is
    0.0021 for 16.
  - *Run:* `tools/cuda_probe.py --plan kb2l` on one Kaggle T4, the same model and revision, at a pushed commit,
    about 0.2 session hours. The plan is run 2's (`dense_gemma.py` unchanged since `9d388e9`).
  - *Windows:* 16, each a 1,536-token prefix led by exactly one BOS (gate 2), then 128 scored tokens past the
    1,024-token window, spread over the quality text as H11 spreads its windows (scored tokens disjoint). The
    plan runs four rows stacked in 2,048-token slots, against today's native decode teacher-forced on the same
    windows.
  - **Decides:** the long path passes if its ratio's 95 % upper bound is below 1.005 over all 16 windows, and the
    long pool's rows are bit-identical alone and stacked. Fewer than 16 windows, or any exactness miss, fails it.
  - *Verdict:* a pass, with KB2-G's (a) and main path and run 2's (b), makes KB2 a candidate opt-in numeric plan
    for 4-bit Gemma 3, and AG5-G may then run as fixed. A fail keeps KB2 in Tier 0 with no further re-gate.

- **KB2-L result: passes (2026-11-07; kernel `tobayko/ironmule-cuda-probe-kb2l` v1 at `7f05b80`, Kaggle T4, 8
  minutes, peak 12.6 GB). KB2 is a candidate opt-in numeric plan for 4-bit Gemma 3.**
  - Long path, 16 windows of 128 tokens after a BOS-led 1,536-token prefix: ratio 0.9973, 95 % 0.9935-1.0012,
    below 1.005. The differences per window ranged from -0.0161 to +0.0123 nats a token, mean -0.0027.
  - The long pool stayed bit-identical, 192 of 192 steps alone and stacked at 4 rows.
  - With KB2-G's (a) and main path (0.9988, 95 % up to 1.0008) and run 2's (b) (2.006x at width 6), KB2's verdict
    is met. AG5-G may now run as fixed. The ledger entry is still owed.

### KB3 — The Gemma plan's attention reads its keys coalesced, one warp per query head, arithmetic unchanged (opened 2026-11-07, before data)

**Why (AG5-G run 4).** On a T4 with Gemma 3 12B at 4 bit, the agent's own measurement found the stacked plan 2.12x
faster at width 6, but 0.897x for one request alone. AG4's floor is 0.9, so the agent refused the plan. The plan's
one-row step rose from 50.4 ms to 57.8 ms between the short and the long prompt (about 300 tokens), while the
native path's rose by 0.2 ms. That rise is FA2 (`dense_gemma.PARTIAL`):
- at one row only one warp per KV head and 64-key chunk runs: 40 warps for 300 tokens on a 40-SM card;
- each lane reads its own 512-byte key row in 2-byte loads, and does so again for each of the G query heads
  that share a KV head (G = 2 for Gemma 3 12B).

**Mechanism (FA2 version 2; it ran in KB3-0 and is removed since KB3-1, its source at `09db8ad`).**
1. One warp per (row, KV head, query head, chunk) instead of per (row, KV head, chunk) looping over the G query
   heads. That gives G times the warps at one row, and each warp does exactly the former loop body for its head.
2. Each warp stages its chunk's keys in shared memory, 32 dimensions at a time, with coalesced loads: the 32 lanes
   read 64 contiguous bytes of one key row. Each lane then forms the same products in the same order (dimension
   0 to D-1) from shared memory.
The combine kernel and every other step are unchanged. Every floating-point operation and its order stay the
same, so v2's outputs must be bit-identical to v1's. That is why KB2's gate and exactness can carry over, and why
the test below demands bit-identity.

**KB3-0, the hardware test (fixed now, before its data).** One Kaggle T4, `tools/cuda_probe.py --plan kb3`, about
0.4 session hours. The model, revision and plan are KB2's: native plan first, eight 1,024-token slots.
- *(a) Exactness, bit for bit, v2 against v1 on the same inputs:*
  - random pools at widths 1 to 8, with lengths from 1 to the capacity, with and without a window, and for 1, 2
    and 4 query heads per KV head;
  - and 64 real decode steps of the plan at widths 1, 6 and 8, run once with each version from the same
    prefills: every step's logits identical.
- *(b) Speed:* plan_choice's own timing, the native path's one-request step and the plan's step at every width
  on its short and long prompts, with v1 and v2 alternated over 3 rounds (medians).
- **Decision:**
  - if any bit differs in (a), v2 is rejected (Tier 0);
  - if (a) holds, v2's one-request ratio by AG4's formula (`stacked_one / base_one`) is at least 0.92, and its
    best-width tokens per second are within 2 % of v1's or better, v2 becomes the plan's kernel (`PLAN_VERSION`
    2). KB2's verdict then carries over by (a), and AG5-G runs again as fixed (run 5);
  - if (a) holds but (b) falls short, the result is recorded and v1 stays.
- *Recorded, no decision:* v1's ratio in the same session, each width's step under both versions, and the
  attention's share of the one-row step.
- **KB3-0 result: (a) passes, (b) does not; v1 stays (2026-11-07; kernel `tobayko/ironmule-cuda-probe-kb3` v1 at
  `09db8ad`, Kaggle T4, 20.5 minutes, peak 12.3 GB).** The plan was KB2's: kernel `54b1d3a2...`, version 1,
  eight 1,024-token slots, at revision `86cc6a8`.
  - **(a) Bit-identical:** all 48 random cases (G of 1, 2 and 4; D of 256 and 128; widths 1 to 8; windows),
    and every one of 64 real decode steps at widths 1, 6 and 8 (logits hashes and tokens).
  - **(b) Not fast enough:** v2's one-request ratio by AG4's formula was 0.904, below 0.92 (v1 in the same
    session: 0.889). The native path made 20.57 tokens per second alone.
  - *Recorded:*
    | step at one row | v1 | v2 |
    | :-- | --: | --: |
    | long prompt (314 tokens) | 60.28 ms | 54.29 ms |
    | short prompt (12 tokens) | 50.04 ms | 53.25 ms |
    - v2 helped the long prompt by 6.0 ms but cost the short one 3.2 ms.
    - At the best width, 6, v2 made 47.81 tokens per second against v1's 45.93 (+4.1 %).
  - *Why the short prompt lost (an inference from the kernels, not measured):* v2 stages all 64 key rows of a
    chunk and forms both keys' products in every lane, valid or not. v1 skips a lane's invalid key. With 12
    tokens, v2 does about five times v1's key work, and its second warp per KV head adds launch work that a short
    context cannot use.
  - By the rule, v1 stays and `FA2_VERSION` remains 1. AG5-G does not run again on KB3-0.
- **KB3-1, version 3 (fixed now, before its data; a new implementation, so this is not a re-analysis of KB3-0).**
  - *Mechanism:* v2, plus three changes that keep every operation and its order:
    - the tile loads only the chunk's valid rows (its `count`, the same for the whole warp);
    - the second half of a chunk's keys is computed only when the chunk has more than 32 of them (also the same
      for the whole warp);
    - the row loop is unrolled by 8, so more loads are in flight.
  - *Test:* KB3-0's protocol unchanged, with v3 in v2's place. (a) is v3 against v1, bit for bit, on the same
    random pools and decode steps. (b) uses the same thresholds: a one-request ratio of at least 0.92, and the
    best width within 2 % of v1's.
  - *Decision:* as KB3-0. If v3 passes, it becomes the plan's kernel (`FA2_VERSION` 3, `PLAN_VERSION` 2), and
    AG5-G runs again (run 5). If not, v1 stays and KB3 closes in Tier 0 with both results.
- **KB3-1 result: passes; v3 is the plan's kernel (2026-11-07; kernel `tobayko/ironmule-cuda-probe-kb31` v1 at
  `191484c`, Kaggle T4, 19.9 minutes, peak 12.3 GB).** The plan was KB2's: kernel `54b1d3a2...`, version 1, eight
  1,024-token slots, at revision `86cc6a8`. The protocol was KB3-0's, with v3 in v2's place.
  - **(a) Bit-identical:** all 48 random cases, and every one of 64 decode steps at widths 1, 6 and 8.
  - **(b) Fast enough:** v3's one-request ratio by AG4's formula was 0.928, at least 0.92 (v1 in the same
    session: 0.884). Its best width, 6, made 46.09 tokens per second against v1's 44.87 (+2.7 %). The native path
    made 20.93 alone.
  - *Recorded:*
    | step at one row | v1 | v3 |
    | :-- | --: | --: |
    | long prompt (314 tokens) | 57.98 ms | 52.61 ms |
    | short prompt (12 tokens) | 50.56 ms | 50.36 ms |
  - *Shipped:* `dense_gemma.FA2_VERSION` is 3 and `PLAN_VERSION` 2. By (a), the plan's answers are v1's, so
    KB2's gate and exactness verdicts carry over, and the agent still gates the plan on its own card at startup.
    v1 stays as the reference for this comparison. v2 is removed; its source is at `09db8ad`, where KB3-0 ran it.
  - *Next:* AG5-G run 5, the same AG5-G at the commit that ships v3, run once. The ledger entry is still owed.

### GATE2 — Every quality-gate chunk starts with the tokenizer's BOS (owner decision 2026-11-07)

**Why.** KB2-D showed that the gate's main chunks, cut from mid-text with no BOS, put Gemma 3 where every path,
today's included, is unstable: without BOS today's own one-shot and decoded paths differed by up to 0.71 nats a
token on one chunk. With BOS all seven paths agreed. Chat requests always start with BOS. The research gate has put
BOS on every chunk since PORT2-K (2026-09-26), after Gemma 4's tokenizer, which adds none when it encodes, left the
chunks without it. The product's gate (`ironmule/numeric_choice.py`) had not followed.

**Rule (`GATE_VERSION` 2).** Each chunk is scored on the same tokens as before, after the tokenizer's BOS token as
unscored context. A chunk that already starts with BOS gets no second one: Gemma 3's tokenizer puts BOS before the
text, so chunk 0 is unchanged. The bound (1.005), the bootstrap, the text and the chunk sizes are unchanged. It
applies everywhere the gate's chunks are built:
- NUM1's one-shot gate (`nll_per_chunk`);
- the agent's stacked-engine gate (`plan_choice.measure_stacked`);
- `tools/kaggle_whole_machine.py`;
- the CUDA probe's gate items for KB1, KB2 and H11, including the long prefixes.

KB2-D's own tool keeps the lead-free chunks it ran on.

**Scope, checked on the tokenizers in use:**
- Qwen 3 declares no BOS (`bos_token` null), so its gate is unchanged. Every Qwen result stands as measured,
  AG4-R's included.
- Gemma 3 (`<bos>`, added when it encodes) gets BOS on chunks 1-15.
- SmolLM2-135M-Instruct declares `<|im_start|>` and adds none when it encodes. All its chunks now start with it,
  as each of its chat requests does.

**Old evidence.** Results measured under gate 1 stay as recorded, scoped to that gate. A decision the product or the
agent stored under gate 1 (`numeric.json`, `plan_choice.json`) no longer authorises anything. It is measured again
on the next start, because both stores now record the gate version and read only their own.

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

**H8: the dense plan through the product (fixed 2026-10-29 before its data; `training/rl2_headroom.py --plan h8`,
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

**Test (`training/rl2_headroom.py --plan kp1`, Kaggle T4 x2, about 2.3 session hours).**
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

**Test (`training/rl2_headroom.py --plan h9`, Kaggle T4 x2).**
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

**Test (`training/rl2_headroom.py --plan h10`, Kaggle T4 x2, about 3 session hours).**
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

**Test, part (b): `training/rl2_headroom.py --plan h11`, Kaggle T4 x2, about 1.8 session hours. It runs only after
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
- **Result, part (a) (2026-11-01, kernel `tobayko/ironmule-cuda-probe-h11` v1 at `3cb1e8f`, one Kaggle T4, 0.22
  session hours): the 8,192-token slots do not ship. The plan keeps 2,048-token slots, and (b) does not run.**
  - (a1) passes. Every row's tokens and logits are bit-identical between the slot sizes. The 8,000-token row
    matched at all 256 logit steps beside seven short rows.
  - (a2) `harm`. The larger slot costs 8.1 % per step (95 % 7.8-8.4 %, noise 0.9 %), with harm at every width:
    7.2 % at width 1, 19.4 % at 2, 4.2 % at 4 and 1.7 % at 8. This is the empty-warp cost the mechanism named.
  - (a3) passes. The peak was 13.68 GB, against a bar of 15.62 GB, with a CUDA context of 0.64 GB.
  - (a4) fails, which is a kill. At 3,000-token prefixes the perplexity ratio is 1.0003 (95 % 0.9965-1.0035, a
    pass). At 6,000 it is 1.0070 (95 % 1.0059-1.0078), above the 1.005 bound.
  - Recorded: prefill takes 2.77 s, 5.84 s and 13.34 s at 2,000, 4,000 and 8,000 tokens. One 8,000-token row
    makes an eight-row step 1.44 times as long as eight short rows.
  - By the ship rule, `install` keeps `SLOT_TIERS` at 2,048, and `--engine dense` keeps refusing longer requests.
    The FA2 grid cut that (a2)'s harm calls for is moot while (a4) fails. Reopening needs a changed mechanism
    for long-context quality, not a rerun.
  - Ledger entry owed.

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

**Plan, in `training/rl4_agent.py`.**
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

- **KB2 Gemma 3 4-bit stacked decode plan (2026-11-07).** KB2-0 run 2 (kernel `tobayko/ironmule-cuda-probe-kb2`
  v2, commit `9d388e9`; see KB2 above): on one T4 with `gemma-3-12b-it-4bit` at `86cc6a8`, stacked rows were bit-exact
  and six of them made 2.006x today's tokens per second (95 % bound 0.4991 against 0.5), but the quality gate's main
  path read 0.984 with 95 % up to 1.177, not below 1.005 (three of 16 chunks diverge by up to 1 nat a token, in both
  directions). KB2's rule stops it at (c). Re-open with a fixed implementation found by a diagnostic of those chunks,
  re-gated under a protocol fixed before its data, or with a new model or card. **Reopened 2026-11-07 as KB2-G**
  (see KB2): KB2-D's new evidence that the gate's lead-free chunks are unstable for every path, and GATE2.
  **KB2-G (commit `4a01f49`): back in Tier 0.** Gate 2's main path passed (0.9988, 95 % up to 1.0008), but its long
  path, 4 chunks past the window, read 1.0003 with 95 % up to 1.0080. Re-open only with a new mechanism or new
  evidence, for example serving scoped to the window under its own protocol. **Reopened again as KB2-L (owner
  decision, 2026-11-07) and passed:** the long path on 16 windows read 0.9973, 95 % up to 1.0012. KB2 is no longer
  rejected; this entry stays as its history.

- **TP1 Kaggle TPU as a decode device (2026-11-07).** Probe TP0 (kernel `tobayko/ironmule-tpu-probe` v1, commit
  `08c96d4`; see TP1 above): on one TPU v5e chip, the bf16 matrix products of one `Qwen3-1.7B` decode step,
  each jitted and timed alone, allowed at most 20.8 tokens/s at 1 row and 180.8 at 8 rows, below the T4's 57.7 and
  290.9. TP1's rule closes it. Re-open with a probe that times a whole decode step compiled as one program (each
  product cost a flat 0.2-0.3 ms, a per-call floor), or with a new model or TPU generation.

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

- **H11 dense slots of 8,192 tokens (2026-11-01).** Experiment `H11a-run1-20261101` (kernel
  `tobayko/ironmule-cuda-probe-h11` v1, commit `3cb1e8f`; see H11). Tested on one T4 with Qwen 3 1.7B bf16.
  Answers were bit-identical and memory fit. At 6,000-token prefixes, though, the plan's perplexity ratio was
  1.0070 (95 % to 1.0078), above the 1.005 bound. The larger slot also cost 8.1 % per step. The dense plan keeps
  2,048-token slots. Re-open with a changed mechanism for long-context accuracy, such as float32 accumulation in
  the long-key combine, and with FA2's grid cut to live chunks.

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
