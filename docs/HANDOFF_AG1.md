# Handoff: the IronMule agent (AG1), state on 2026-10-30

For the next agent. Read `AGENTS.md`, then BACKLOG AG1 (`docs/BACKLOG.md`, top). This file says where
things stand and what to do next; BACKLOG holds every protocol and result.

## Goal (maintainer decision, 2026-10-30)

A working IronMule agent that decides on its own, reaches a measured gain on real hardware, learns
continuously by RL, and decides as fast as a decision model. Simulated gains do not count as the
hardware gain.

## Done

- `ironmule_product/agent.py`, opt-in via `ironmule serve --agent on`; hooks are in `service.py`
  (`_agent_observe`, idle, `_replay_window(skip_idle, whole_queue)`), `cli.py` (`_agent`).
  - It learns a world model by RLS and trains REINFORCE in a planner subprocess.
  - It qualifies by replay A/B/AA under KP1's rule, read group-sequentially.
  - It audits the serving schedule against the reference, excludes losers by behaviour, and decides
    from a compiled table (1.3 us per decision).
- Simulation S/S2 (`tools/ag1_online.py`): S run 8 and S2 run 4 at `217fd2a` pass every criterion,
  so V's gate is open.
- Tests: `tests/engine/test_agent.py`, `test_ag1_online.py`, and the ag1 tests in
  `test_rl2_headroom.py`. The engine suite is green, and so is CI on every pushed head.

## Running or scheduled

- **V-CPU (real hardware, this container's CPU).**
  - Command: `python3 tools/rl2_headroom.py --plan ag1 --profile cpu --out <scratchpad>/ag1/vcpu_report.json`.
  - It started about 14:46 UTC at `1db94a7` and takes about 5 hours. The protocol is in BACKLOG AG1
    (V-CPU), amended before data to short prompts only.
  - Last seen: 3 online traces done. The agent planned `w1/sjf` (`633ceda2cdce`) as its candidate; no
    triplets yet.
  - When it ends, run `python3 tools/rl2_headroom.py --analyse <report>`. Record the result under V-CPU
    in BACKLOG by its fixed rules (pass: the gain's 2.5 % bound is above 0 and above the A/A floor, and
    no state is harmed beyond the noise). Commit and push.
  - The scratchpad does not survive the container. If the run was lost, restart it at the current HEAD;
    the agent code is unchanged since `217fd2a`.
- **V on a Kaggle T4 and H11 (a).**
  - Scheduled task `trig_01LoEUJyHcJcK3z8FsEmexvN` fires at 2026-10-31T00:30Z. Pushes were refused
    because the weekly 30-hour GPU quota was used up.
  - Kernel: `python3 tools/rl2_headroom.py --write-kernel DIR --owner tobayko --commit <pushed HEAD> --plan ag1`,
    then `kaggle kernels push -p DIR --accelerator NvidiaTeslaT4`, using `KAGGLE_USERNAME=tobayko` and a
    dummy `KAGGLE_KEY` (the proxy injects the real one).
  - Before pushing, state about 2.5 session hours. Poll at most every 2 minutes, and archive to
    `tobayko/ironmule-notebook-archive` before deleting anything.
- PR: https://github.com/Tobayko/IronMule/pull/23 (branch `claude/blissful-goldberg-kiwkyw`), green
  with no reviews. A safety-net check-in fires at 18:04Z.

## Next steps

1. Record the V-CPU result. On a pass, update LIMITS and README with the measured, bounded gain.
2. Record V (T4) once it has run. A pass ships only opt-in, bound to that setup.
3. Open work toward the goal:
   - model the dense plan's slot reuse, so the agent can run there too (currently refused);
   - placement across several GPUs;
   - the ledger entries in IronMule-Research for AG1, H8-H11, KP1 and RL8 (owed, never written).
4. Results page (dashboard): https://claude.ai/artifact/MRqZZVQceJBo6nEbsAvS1z

## Rules not to break

- Fix each protocol in BACKLOG before its data, and label any later change post hoc.
- No minimum effect size: every resolved gain counts.
- Commits carry only `Co-Authored-By: Claude <noreply@anthropic.com>`, with no model names or session links.
- Push only to `claude/blissful-goldberg-kiwkyw`.
