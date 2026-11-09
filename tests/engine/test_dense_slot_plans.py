"""STAB4: run 6's arrivals as explicit event plans through the product's own path (control logic only).

The same world as `test_dense_slot_flow.py` (the product's service, client, transport, worker process, bridge,
Runtime and slot pool; a tiny random Gemma 3 with the CPU reference attention in place of CUDA). The plans run in
lockstep: the worker holds at the start of every executor round (finished requests have returned their slots,
newcomers not yet admitted); the test queues that round's planned arrivals, waits until the service and the pool
agree on the open members and every join is in the worker's hands, and lets the round run. `IRONMULE_STAB4_SEEDS=1-20` adds seeded variations of both
blocks to the four kept here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from tests.engine.test_dense_slot_flow import SLOTS, World, _tokens, _until, alone, check_ends, check_pool  # noqa: E402


# AG5-R's first two blocks are run 6's phase 0 blocks 0 (bursts) and 1 (spike), rebuilt from trace 430
# (`training/rl2_headroom.py:ag5r_blocks`). Here their arrival times only order events against the worker's rounds
# (one stacked step each): an arrival at offset t comes at round 1 + floor(t / STEP_S), where STEP_S is one chained readback
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
    paused = _round(world, step)
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


def _round(world: World, number: int) -> dict:
    assert _until(lambda: any(row["op"] == "paused_round" and row["round"] == number for row in world.slots()))
    return next(row for row in world.slots() if row["op"] == "paused_round" and row["round"] == number)


ROUNDS = 400  # holds for every executor round the plans can take, in lockstep


def play(world: World, first, by_step, after, holds: dict[int, Path]) -> list[str]:
    """Run the plan in lockstep: the worker holds at the start of every executor round; there the test lets the
    service see every done the worker sent, queues the round's planned arrivals, waits until the service has
    handed every place the group has over and the worker holds those joins, and lets the round run. Requests
    after the load (`after`) start a new group once the first run has ended. Returns every id in arrival order."""
    world.offer_together(first)
    open_ids, queued, ids = [rid for rid, _, _ in first], [], [rid for rid, _, _ in first]
    later = list(after)
    for number in sorted(holds):
        def reached(number=number) -> bool:
            ended = {row["id"] for row in world.log() if row["event"] == "end"}
            return (any(row["op"] == "paused_round" and row["round"] == number for row in world.slots())
                    or (set(ids) <= ended and not queued))
        assert _until(reached)
        if not any(row["op"] == "paused_round" and row["round"] == number for row in world.slots()):
            if not later:
                break  # the last run has ended
            assert _until(lambda: world.service.health()["in_flight"] is None)
            world.offer_together(later)
            ids += [rid for rid, _, _ in later]
            open_ids, later = [rid for rid, _, _ in later], []
        arrivals = by_step.get(number, []) if not after or later else []
        ids += [rid for rid, _, _ in arrivals]
        open_ids, queued = _sync(world, number, open_ids, queued, arrivals)
        holds[number].write_text("go")
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
    holds = {number: tmp_path / f"resume-{number}" for number in range(1, ROUNDS + 1)}
    world = World(tmp_path, monkeypatch, faults={"pause_rounds": [[n, str(path)] for n, path in holds.items()]})
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
    """Reproducible: run 6's spike as an event plan, twice; the slots taken and freed, and in which round, are
    identical."""
    traces = []
    for attempt in ("a", "b"):
        first, by_step, after = run6_plan("phase0-block1", 3)
        root = tmp_path / attempt
        root.mkdir()
        holds = {number: root / f"resume-{number}" for number in range(1, ROUNDS + 1)}
        with pytest.MonkeyPatch.context() as patch:
            world = World(root, patch, faults={"pause_rounds": [[n, str(path)] for n, path in holds.items()]})
            try:
                play(world, first, by_step, after, holds)
                traces.append([(row["op"], row.get("slot"), row.get("prompt"), row.get("round"))
                               for row in world.slots() if row["op"] in ("take", "free", "paused_round")])
            finally:
                world.close()
    assert traces[0] == traces[1] and len(traces[0]) > 20


# --- AG5-R rev 2 through the real path: the harness's own blocks, measurement and analysis ---------------------------

def _ag5r_pool(tmp_path: Path, monkeypatch):
    """The product's pool (`start_paths`, one shard) on the CPU Gemma worker, as `run_ag5r` builds it on the card."""
    import subprocess
    import sys

    import ironmule_product.backend as backend_module
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.device_pool import start_paths
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore
    from ironmule_product.types import ModelSpec
    from tests.engine import dense_cpu_worker as cpu
    from training import rl2_headroom as rl2

    real = subprocess.Popen

    def popen(args, *a, **k):
        if len(args) > 3 and str(args[3]).endswith("worker.py"):
            args = [sys.executable, "-u", cpu.__file__, *args[4:]]
        return real(args, *a, **k)

    monkeypatch.setattr(backend_module.subprocess, "Popen", popen)
    monkeypatch.setenv("DENSE_CPU_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("DENSE_CPU_CAPACITY", "96")
    monkeypatch.setenv("DENSE_CPU_FAULTS", "{}")
    store = ProductStore(tmp_path / "state")
    store.setup("server")
    store.set_request_timeout(120)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    spec = ModelSpec("local/test", "revision", str(snapshot.resolve()), 1)
    store.register_model(spec)
    maker = lambda: MLXWorkerClient(spec, execution_variant="current_engine",  # noqa: E731
                                    engine_configuration="dense_stacked", dense_width=SLOTS)
    return start_paths(store, spec, [maker], ["cpu:agent"], ProductService, rl2.ForcedRoute(0)), store, spec


def test_ag5r_rev2_start_section_and_acceptance_through_the_real_path(tmp_path, monkeypatch):
    """Rev 2 on today's product: the harness's six blocks (small), its measurement and its analysis on the real
    stability log. The start block's 8 requests are all queued before the scheduler takes one (a controlled hold:
    the scheduler waits at a gate until the log shows all 8 admitted), with the harness's 500 ms gather in force;
    the repaired service starts 6 and the other 2 join. Accepted; A1's precondition (8 start members) is not met
    here, which is R2's case and is not required of it."""
    import threading

    from ironmule_product.service import STABILITY_LOG
    from training import rl2_headroom as rl2

    pool, store, spec = _ag5r_pool(tmp_path, monkeypatch)
    shard = pool.shards[0]
    gate, parked, seen = threading.Event(), threading.Event(), {}
    gate.set()
    original = shard._next_session

    def gated():
        if not gate.is_set():
            parked.set()
            gate.wait()
            seen["gather_ms"] = shard.policy.gather_ms  # in force when the held group is taken
        return original()

    shard._next_session = gated
    replay = rl2.replay

    def held(pool_, model, items, deadline_s, ids=None):
        if ids and ids[0].startswith("ag5r-eight-at-start"):
            parked.clear()
            gate.clear()
            assert parked.wait(30)

            def release():
                log = store.root / STABILITY_LOG
                _until(lambda: {r.get("id") for r in _log(log) if r.get("event") == "admit"} >= set(ids))
                gate.set()

            threading.Thread(target=release, daemon=True).start()
        return replay(pool_, model, items, deadline_s, ids=ids)

    monkeypatch.setattr(rl2, "replay", held)
    texts = {name: [{"offset_s": 0.0 if name != "after-the-load" else 0.3 * k, "max_tokens": 8, "prompt": 0,
                     "long": False, "text": f"{name} #{k}"} for k in range(n)]
             for name, n in (("phase0-block0", 2), ("phase0-block1", 3), ("phase0-block1-again-1", 3),
                             ("eight-at-once", 8), ("eight-at-start", 8), ("after-the-load", 2))}
    blocks = list(texts.items())
    try:
        report = rl2.measure_ag5r(pool, spec.model_id, blocks, 120.0)
    finally:
        pool.close()
    log = _log(store.root / STABILITY_LOG)
    analysis = rl2.analyse_ag5r(report, log, rl2.ag5r_expected(blocks))
    assert analysis["acceptance"]["verdict"] == "accepted", analysis["acceptance"]
    start = analysis["preconditions"]["eight-at-start"]
    assert seen["gather_ms"] == rl2.AG5R_START_GATHER_MS and start["gather_ms_after"] == [0]
    assert start["offered"] == rl2.ag5r_ids("eight-at-start", 8) and start["service_capacity"] == SLOTS
    assert len(start["start_members"]) == SLOTS and not start["foreign_members"] and start["ready_at_start"]
    assert sorted(start["joined"]) == sorted(set(start["offered"]) - set(start["start_members"]))
    assert start["engine_width"][0]["slots"] == SLOTS and analysis["log_checks"]["widest_group"] <= SLOTS
    assert analysis["cause"]["a1"].startswith("precondition not met: the group started with 6 of 8")
    assert analysis["preconditions"]["eight-at-once"]["joined"], "the old section still tests dynamic admission"
    assert max(row.get("realised_width") or 0 for row in report["blocks"][4]["requests"]) <= SLOTS


def _log(path: Path) -> list[dict]:
    import json

    if not path.exists():
        return []
    text = path.read_text()
    return [json.loads(line) for line in text[:text.rfind("\n") + 1].splitlines()]
