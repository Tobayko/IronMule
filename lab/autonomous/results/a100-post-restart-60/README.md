# Planning the next source experiment on the merged PR 60 worker

The two numbered files are complete A100 observations captured in separate restricted
processes. The control host used their actual complete source and hardware bindings.
The deterministic decision and the archive rows it cited are provided alongside them.

Replay requires the merged checkout and this framework with NumPy installed.

```sh
git fetch origin qualification/d938522e9c93ef05
git worktree add --detach /tmp/ironmule-pr60-replay 63db7745998383509602601d46ae4bbff0076648
d=lab/autonomous/results/a100-post-restart-60
r=lab/autonomous/results/a100-restart/d938522e9c93ef05d746bb030631675ccd3c76b3c8292be7f0ad6dba5b3c31b3
OPENBLAS_NUM_THREADS=1 python -m training.post_restart_planner \
  --observation "$d/96751.json" --observation "$d/96752.json" \
  --control "$r/control.json" --weights "$r/startup-head.npz" \
  --contract training/post_restart_contract.json --checkout /tmp/ironmule-pr60-replay \
  --archive "$d/archive-rows.jsonl" --hourly-usd 2.201904761904762 \
  --out /tmp/ironmule-post-restart-plan.json
cmp "$d/decision.json" /tmp/ironmule-post-restart-plan.json
```

On the active original target, `python -m lab.pk1.post_restart_observe --help` lists the
bound checkpoint, active release, cache and worker inputs needed to repeat each observation.
The broker requires the pinned startup head to select the active cache on the actual
target. A different card, source, TTL, device binding or selected release needs its own
qualification. Use the shared GPU measurement lock when repeating target runs.
