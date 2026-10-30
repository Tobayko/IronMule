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

- **V-CPU (real hardware, this container's CPU): done, no gain.** The run at `1db94a7` qualified nothing online:
  8 triplets in 15 traces, the look at 6 was inconclusive, and the phase ended before the look at 12. The result is
  in BACKLOG AG1 (V-CPU). A rerun needs a new protocol fixed before its data, with more online traces (more
  triplets) or traffic where order matters more.
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

1. V-CPU is recorded (no gain). Optionally, fix and run a longer V-CPU-2 protocol.
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
