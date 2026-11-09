"""STAB4: run 6's arrivals as explicit event plans through the product's own path (control logic only).

The same world as `test_dense_slot_flow.py` (the product's service, client, transport, worker process, bridge,
Runtime and slot pool; a tiny random Gemma 3 with the CPU reference attention in place of CUDA). The worker pauses
before each stacked step at which a planned request arrives; the test queues it there and lets the step run once
the service and the worker agree on the open members. `IRONMULE_STAB4_SEEDS=1-20` adds seeded variations of both
blocks to the four kept here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from tests.engine.test_dense_slot_flow import SLOTS, World, _paused, _tokens, _until, alone, check_ends, check_pool  # noqa: E402


# AG5-R's first two blocks are run 6's phase 0 blocks 0 (bursts) and 1 (spike), rebuilt from trace 430
# (`training/rl2_headroom.py:ag5r_blocks`). Here their arrival times only order events against the worker's stacked
# steps: an arrival at offset t comes before step 1 + floor(t / STEP_S), where STEP_S is one chained readback
# (4 tokens) of the 256-token answers scaled to LOCAL_LIMIT, at run 6's 0.128 s per step at width 6 (AG5-G's
# measured curve, 46.8 tokens/s). Prompts, lengths and the clock are local stand-ins: nothing here shows what the
# T4 did inside run 6, and no time or rate from these runs is a measurement.

LOCAL_LIMIT = 32
STEP_S = 4 * 0.128 * 256 / LOCAL_LIMIT
FIRST_S = 0.05  # arrivals this close to the first one are queued when the scheduler takes it


def run6_plan(block: str, seed: int | None) -> tuple[list[tuple[str, str, int]], dict[int, list], list]:
    """(the first group, arrivals by step, the requests after the run): run 6's order, or with `seed` a nearby
    variation -- each arrival moved by up to one step, arrivals at one step shuffled, lengths varied."""
    import random

    from training.rl2_headroom import ag5r_blocks

    items = dict(ag5r_blocks())[block]
    rng = random.Random(seed)
    first, by_step, after = [], {}, []
    # Later than twice a whole answer: the run has ended (run 6's third burst of block 0, at 96 s, came after the
    # agent's answers had ended at 25 to 45 s); such requests start a new group, which pauses cannot order.
    run_end_s = items[0]["offset_s"] + 2 * STEP_S * LOCAL_LIMIT / 4
    for index, item in enumerate(items):
        rid, text = f"{block[-1]}-{index:02d}", f"r6 {block[-1]} #{index:02d} " + "klmnopqrst"[index % 10] * 3
        limit = LOCAL_LIMIT if seed is None else rng.randint(LOCAL_LIMIT // 4, LOCAL_LIMIT + 8)
        offset = item["offset_s"] - items[0]["offset_s"]
        if offset <= FIRST_S:
            first.append((rid, text, limit))
        elif item["offset_s"] > run_end_s:
            after.append((rid, text, limit))
        else:
            step = 1 + int(offset // STEP_S) + (0 if seed is None else rng.choice((-1, 0, 1)))
            by_step.setdefault(max(1, step), []).append((rid, text, limit))
    if seed is not None:
        for arrivals in by_step.values():
            rng.shuffle(arrivals)
    return first, by_step, after


def _sync(world: World, step: int, open_ids: list[str], queued: list[str],
          arrivals: list[tuple[str, str, int]]) -> tuple[list[str], list[str]]:
    """At the pause before `step`: every done the worker sent has reached its caller, so the service's count of
    open members is the worker's; the step's arrivals are queued in order; the queue joins the group while it has
    room, and the worker holds those joins before the step runs."""
    paused = _paused(world, step)
    sent = [row["id"] for row in world.slots() if row["op"] == "done_sent"]
    assert _until(lambda: all(any(e.get("type") == "done" for e in world.live.get(rid, ())) for rid in sent))
    open_ids = [rid for rid in open_ids if rid not in sent]
    assert paused["free"] == SLOTS - len(open_ids), (paused, open_ids)  # the worker's pool agrees with the service
    for rid, text, limit in arrivals:
        world.submit(rid, text, limit)
        assert _until(lambda rid=rid: rid in [row["id"] for row in world.log() if row["event"] == "admit"])
        queued = queued + [rid]
    joining = queued[:max(0, SLOTS - len(open_ids))]
    if joining:
        assert _until(lambda: set(joining) <= {row["id"] for row in world.log() if row["event"] == "join"})
        assert _until(lambda: set(joining) <= {i for row in world.slots() if row["op"] == "join_received"
                                               for i in row["ids"]})
    assert _until(lambda: world.service.health()["queued_requests"] == len(queued) - len(joining))
    return open_ids + joining, queued[len(joining):]


def play(world: World, first, by_step, after, holds: dict[int, Path]) -> list[str]:
    """Run the plan; returns every request id in arrival order."""
    world.offer_together(first)
    open_ids, queued, ids = [rid for rid, _, _ in first], [], [rid for rid, _, _ in first]
    for step in sorted(holds):
        ids += [rid for rid, _, _ in by_step.get(step, ())]
        open_ids, queued = _sync(world, step, open_ids, queued, by_step.get(step, []))
        holds[step].write_text("go")
    world.wait()
    if after:
        assert _until(lambda: world.service.health()["in_flight"] is None)
        world.offer_together(after)
        ids += [rid for rid, _, _ in after]
        world.wait()
    return ids


def _seeds() -> list[int]:
    first, _, last = os.environ.get("IRONMULE_STAB4_SEEDS", "").partition("-")
    return list(range(int(first), int(last or first) + 1)) if first else []


PLANS = ([("phase0-block0", None), ("phase0-block1", None), ("phase0-block0", 1), ("phase0-block1", 2)]
         + [(block, seed) for seed in _seeds() for block in ("phase0-block0", "phase0-block1")
            if (block, seed) not in (("phase0-block0", 1), ("phase0-block1", 2))])


@pytest.mark.parametrize("block,seed", PLANS)
def test_run_6s_arrivals_in_explicit_event_plans_keep_every_slot_rule(tmp_path, monkeypatch, block, seed):
    """Request runs, newcomers arrive, members end, freed slots are taken again: in run 6's own order and in
    seeded variations of nearby events. Every plan: never a seventh slot, each request once to its end with its
    own answer, the waiting ones served from freed slots, nothing left taken, no worker lost."""
    first, by_step, after = run6_plan(block, seed)
    holds = {step: tmp_path / f"resume-{step}" for step in by_step}
    world = World(tmp_path, monkeypatch, faults={"pause": [[step, str(path)] for step, path in holds.items()]})
    try:
        ids = play(world, first, by_step, after, holds)
        texts_of = {rid: (text, limit) for rid, text, limit in first + after + [a for v in by_step.values() for a in v]}
        ends = check_ends(world, ids)
        for rid in ids:
            assert ends[rid]["outcome"] == "ok" and _tokens(world.results[rid]) == alone(*texts_of[rid]), rid
        pool = check_pool(world)
        assert pool["widest"] <= SLOTS and pool["takes"] == len(ids)
        rows = world.slots()
        # A slot another request freed, taken while the run goes on (at least one other slot still held).
        reused = [row for i, row in enumerate(rows) if row["op"] == "take" and row["free"] <= SLOTS - 2
                  and any(r["op"] == "free" and r["slot"] == row["slot"] for r in rows[:i])]
        if len(ids) - len(after) > SLOTS:
            assert reused, "more requests than slots, so some had to take a slot another freed mid-run"
        assert not [row for row in world.log() if row["event"] in ("worker_exit", "worker_restart_start")]
    finally:
        world.close()


def test_the_same_event_plan_gives_the_same_admissions_twice(tmp_path, monkeypatch):
    """Reproducible: run 6's spike as an event plan, twice; the slots taken and freed, and when, are identical."""
    traces = []
    for attempt in ("a", "b"):
        first, by_step, after = run6_plan("phase0-block1", 3)
        root = tmp_path / attempt
        root.mkdir()
        holds = {step: root / f"resume-{step}" for step in by_step}
        with pytest.MonkeyPatch.context() as patch:
            world = World(root, patch, faults={"pause": [[step, str(path)] for step, path in holds.items()]})
            try:
                play(world, first, by_step, after, holds)
                traces.append([(row["op"], row.get("slot"), row.get("prompt"), row.get("step"))
                               for row in world.slots() if row["op"] in ("take", "free", "paused")])
            finally:
                world.close()
    assert traces[0] == traces[1] and len(traces[0]) > 20
