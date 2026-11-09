"""STAB4: the stacked engine's slots, admission, joining, cancellation and failure, through the product's own path.

The service (`ProductService`), the client (`MLXWorkerClient`) and its JSON-lines transport, the worker process
(`ironmule_product.worker.main`: its command loop, `_run_engine_batch`'s admission), the bridge, the Runtime and
the plan's slot pool (`DenseBackend`) are the product's own. `dense_cpu_worker.py` replaces only what needs CUDA or
a checkpoint: a tiny random 4-bit Gemma 3, KB2's plan with the CPU reference attention (`native=False`), a stand-in
identity. Order is set by explicit holds, never by sleeps: the scheduler waits behind a gate until the requests a
test offers are all queued, and the worker pauses before a named stacked step until the test has acted.

What this checks: the control logic (slot count against admission, waiting, reuse of a freed slot, one terminal
state per request, no slot left taken, per-request isolation of answers and caches). What it cannot check: CUDA's
kernels, card memory, timing or throughput; `tools/cuda_probe.py` and AG5-R measure those on the hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

import ironmule_product.backend as backend_module  # noqa: E402
import ironmule_product.service as service_module  # noqa: E402
from ironmule_product.backend import MLXWorkerClient  # noqa: E402
from ironmule_product.errors import ProductError  # noqa: E402
from ironmule_product.service import STABILITY_LOG, ProductService  # noqa: E402
from ironmule_product.state import ProductStore  # noqa: E402
from ironmule_product.types import GenerationRequest, ModelSpec  # noqa: E402
from tests.engine import dense_cpu_worker as cpu  # noqa: E402

LAUNCHER = Path(cpu.__file__)
SLOTS = 6  # AG5-G run 6: the agent's width, so the plan's pool held six requests
CAPACITY = 96  # a slot's context: prompt plus output tokens of one request (not the number of slots)


class World:
    def __init__(self, tmp_path: Path, monkeypatch, *, client_width: int | None = SLOTS, slots: int = SLOTS,
                 faults: dict | None = None):
        real_popen = subprocess.Popen

        def popen(args, *popen_args, **popen_kwargs):
            if len(args) > 3 and str(args[3]).endswith("worker.py"):
                args = [sys.executable, "-u", str(LAUNCHER), *args[4:]]
            return real_popen(args, *popen_args, **popen_kwargs)

        monkeypatch.setattr(backend_module.subprocess, "Popen", popen)
        monkeypatch.setattr(service_module, "RESTART_BACKOFF_S", (0.0, 0.2, 0.4))
        self.ledger = tmp_path / "ledger.jsonl"
        monkeypatch.setenv("DENSE_CPU_LEDGER", str(self.ledger))
        monkeypatch.setenv("DENSE_CPU_CAPACITY", str(CAPACITY))
        monkeypatch.setenv("DENSE_CPU_FAULTS", json.dumps(faults or {}))
        if slots != (client_width or 8):
            monkeypatch.setenv("DENSE_CPU_SLOTS", str(slots))  # the pool the client does not know of (run 6)
        self.tmp = tmp_path
        self.store = ProductStore(tmp_path / "state")
        self.store.setup("server")
        self.store.set_request_timeout(120)
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        spec = ModelSpec("local/test", "revision", str(snapshot.resolve()), 1)
        self.store.register_model(spec)

        def client() -> MLXWorkerClient:
            return MLXWorkerClient(spec, execution_variant="current_engine", engine_configuration="dense_stacked",
                                   dense_width=client_width)

        backend = client()
        backend.start()
        assert backend._ready_payload["engine"]["dense_plan"]["kernels"] == "cpu_reference"
        self.service = ProductService(self.store, backend=backend, spec=spec)
        self.service.make_backend = client
        self.gate, self.parked = threading.Event(), threading.Event()
        self.gate.set()
        original = self.service._next_session

        def gated():
            if not self.gate.is_set():
                self.parked.set()  # the scheduler waits here, holding no request
            self.gate.wait()
            return original()

        self.service._next_session = gated
        self.results: dict[str, list | ProductError] = {}
        self.live: dict[str, list] = {}
        self.threads: list[threading.Thread] = []

    def submit(self, rid: str, text: str, max_tokens: int) -> None:
        request = GenerationRequest("local/test", (("user", text),), max_tokens=max_tokens, request_id=rid,
                                    stream=True)

        events = self.live.setdefault(rid, [])  # what its caller has received so far

        def run() -> None:
            try:
                for event in self.service.stream(request):
                    events.append(event)
            except ProductError as exc:
                exc.events = events
                self.results[rid] = exc
                return
            self.results[rid] = events

        self.threads.append(threading.Thread(target=run, name=rid))
        self.threads[-1].start()

    def offer_together(self, rows: list[tuple[str, str, int]]) -> None:
        """Every request queued before the scheduler takes any of them (8 at once, as AG5-R's case)."""
        self.parked.clear()
        self.gate.clear()
        assert self.parked.wait(30), "the scheduler did not reach the gate"  # its last pending wait has returned
        queued = self.service.health()["queued_requests"]
        for k, (rid, text, limit) in enumerate(rows, 1):
            self.submit(rid, text, limit)  # each queued before the next is sent, so the queue holds them in order
            assert _until(lambda k=k: self.service.health()["queued_requests"] == queued + k), self.service.health()
        self.gate.set()

    def wait(self) -> None:
        for thread in self.threads:
            thread.join(timeout=60)
        assert not any(thread.is_alive() for thread in self.threads)

    def log(self) -> list[dict]:
        return _rows(self.store.root / STABILITY_LOG)

    def slots(self) -> list[dict]:
        return _rows(self.ledger)

    def close(self) -> None:
        self.gate.set()
        self.service.close()


def _rows(path: Path) -> list[dict]:
    """A JSON-lines file another process or thread is appending to: its complete lines only."""
    if not path.exists():
        return []
    text = path.read_text()
    return [json.loads(line) for line in text[:text.rfind("\n") + 1].splitlines()]


def _until(predicate, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


_ALONE: dict = {}


def alone(text: str, max_tokens: int) -> list[int]:
    """The request on its own in a fresh pool of the same plan: what isolation means for its answer."""
    if "bridge" not in _ALONE:
        from ironmule.runtime import Engine, Knobs
        from ironmule.service import DenseStackedMode, Runtime
        from ironmule_product.engine_bridge import CurrentEngineBridge

        model, tokenizer = cpu.tiny_gemma(), cpu.ChatTokenizer()
        runtime = Runtime(Engine(model, tokenizer, Knobs()), tokenizer,
                          mode=DenseStackedMode(cpu.cpu_plan(model, SLOTS, CAPACITY), max_width=SLOTS))
        _ALONE["bridge"] = CurrentEngineBridge(
            runtime, identity=SimpleNamespace(to_dict=lambda: {"identity_sha256": "c" * 64}), knobs=Knobs(),
            profile_available=False, profile_source="dense_numeric_plan", configuration="dense_stacked",
            historical_configuration_candidate=None, dense_plan={"kernels": "cpu_reference"})
        _ALONE["encode"] = tokenizer.encode
    key = (text, max_tokens)
    if key not in _ALONE:
        tokens = list(_ALONE["bridge"].generate(_ALONE["encode"](text), max_tokens).tokens)
        # The service streams no stop token (the worker's `stop_handling` is the parent's): drop a final EOS.
        _ALONE[key] = tokens[:-1] if tokens and tokens[-1] in cpu.ChatTokenizer.eos_token_ids else tokens
    return _ALONE[key]


def _tokens(events: list[dict]) -> list[int]:
    return [event["token_id"] for event in events if event.get("type") == "token"]


def check_pool(world: World, slots: int = SLOTS) -> dict:
    """The ledger's invariants: never more taken than the pool has, never a slot taken twice, all returned."""
    rows = [row for row in world.slots() if row["op"] in ("take", "free")]
    assert not [row for row in rows if "violation" in row], rows
    taken: dict[int, str] = {}
    widest = 0
    for row in rows:
        if row["op"] == "take":
            assert row["slot"] not in taken and 0 <= row["slot"] < slots, row
            taken[row["slot"]] = row["prompt"]
        else:
            assert taken.pop(row["slot"]) == row["prompt"], row
        widest = max(widest, len(taken))
        assert row["free"] == slots - len(taken), row  # the pool's own count agrees with the ledger
    assert taken == {}, f"slots left taken: {taken}"
    return {"widest": widest, "takes": sum(row["op"] == "take" for row in rows)}


def check_ends(world: World, ids: list[str]) -> dict[str, dict]:
    """Each accepted request ends exactly once, in the stability log and for its caller."""
    assert _until(lambda: {row["id"] for row in world.log() if row["event"] == "end"} >= set(ids))
    ends = [row for row in world.log() if row["event"] == "end"]
    for rid in ids:
        assert [row["id"] for row in ends].count(rid) == 1, rid
        result = world.results[rid]
        if isinstance(result, ProductError):
            assert not any(event.get("type") == "done" for event in result.events), rid
        else:
            assert [event["type"] for event in result].count("done") == 1 and result[-1]["type"] == "done", rid
    health = world.service.health()
    assert health["active_requests"] == 0 and health["queued_requests"] == 0, health
    return {row["id"]: row for row in ends}


def texts(n: int) -> list[str]:
    return [f"request {i}: " + "abcdefghij"[i % 10] * (3 + i) for i in range(n)]


LIMITS = [6, 22, 10, 30, 8, 14, 12, 9]  # different lengths, so members end at different steps


@pytest.mark.parametrize("client_width", [SLOTS, None], ids=["service-width-6", "run-6-service-width-8"])
def test_eight_offered_at_once_to_six_slots_wait_and_take_the_slots_members_free(tmp_path, monkeypatch,
                                                                                    client_width):
    """AG5-R's eight-at-once case on six slots. With the service told the width (STAB3), six start and two join
    as members end; with run 6's constellation (the service admits eight into a group), the worker starts no
    more rows than slots are free and serves the rest after. Either way: never a seventh slot, every answer the
    one it gets alone, every slot back."""
    world = World(tmp_path, monkeypatch, client_width=client_width)
    try:
        prompts = texts(8)
        world.offer_together([(f"r{i}", prompts[i], LIMITS[i]) for i in range(8)])
        world.wait()
        ends = check_ends(world, [f"r{i}" for i in range(8)])
        for i in range(8):
            assert not isinstance(world.results[f"r{i}"], ProductError), world.results[f"r{i}"]
            assert _tokens(world.results[f"r{i}"]) == alone(prompts[i], LIMITS[i]), i
            assert ends[f"r{i}"]["outcome"] == "ok"
        pool = check_pool(world)
        assert pool["widest"] == SLOTS and pool["takes"] == 8  # the pool was full, and no request ran twice
        rows = world.slots()
        takes = [row for row in rows if row["op"] == "take"]
        first_free = next(i for i, row in enumerate(rows) if row["op"] == "free")
        late = [i for i, row in enumerate(rows) if row["op"] == "take" and i > first_free]
        assert len(late) == 2  # the two that waited took slots only after members had freed them
        freed_before = {row["slot"] for row in rows[:late[0]] if row["op"] == "free"}
        assert takes[6]["slot"] in freed_before  # a freed slot, used again
        log = world.log()
        groups = [row for row in log if row["event"] == "group"]
        joins = [row["id"] for row in log if row["event"] == "join"]
        from training.rl2_headroom import ag5r_log_checks

        checks = ag5r_log_checks(log, SLOTS)  # AG5-R R2's own checks, on this real stability log
        assert checks["ended_once"] and checks["worker_events"] == [] and checks["errors"] == []
        if client_width == SLOTS:
            assert [len(row["ids"]) for row in groups] == [6] and joins == ["r6", "r7"]
            assert checks["within_width"] and checks["widest_group"] == SLOTS and checks["joins"] == 2
        else:
            assert [len(row["ids"]) for row in groups] == [8] and joins == []  # the worker held two back itself
            assert not checks["within_width"] and checks["widest_group"] == 8  # the check sees run 6's constellation
        assert not [row for row in log if row["event"] in ("worker_exit", "worker_restart_start")]
        assert world.service.health()["worker"]["restarts"] == 0
    finally:
        world.close()


def _paused(world: World, step: int) -> dict:
    assert _until(lambda: any(row["op"] == "paused" and row["step"] == step for row in world.slots()))
    return next(row for row in world.slots() if row["op"] == "paused" and row["step"] == step)


def test_joiners_arrive_mid_run_one_is_cancelled_and_a_waiting_one_takes_the_slot_it_frees(tmp_path, monkeypatch):
    """A long and a short member run; five arrive mid-run, four take the free slots and the fifth waits (the pool
    is full); a joined request is cancelled, and the waiting one takes the slot it frees. Each pause is a point in
    the worker's run the test reaches by observation."""
    hold = [tmp_path / "resume-1", tmp_path / "resume-2", tmp_path / "resume-3"]
    world = World(tmp_path, monkeypatch, faults={"pause": [[2, str(hold[0])], [5, str(hold[1])], [7, str(hold[2])]]})
    try:
        prompts = texts(7)
        world.offer_together([("long", prompts[0], 80), ("short", prompts[1], 44)])
        first = _paused(world, 2)  # inside the run: both members have their slots, the group streams
        assert first["free"] == SLOTS - 2
        joiners = [(f"j{k}", prompts[1 + k], 40) for k in range(1, 6)]
        for rid, text, limit in joiners:  # each admitted before the next is sent: the queue holds them in order
            world.submit(rid, text, limit)
            assert _until(lambda rid=rid: rid in [row["id"] for row in world.log() if row["event"] == "admit"])
        assert _until(lambda: [row["id"] for row in world.log() if row["event"] == "join"] == ["j1", "j2", "j3", "j4"])
        assert _until(lambda: world.service.health()["queued_requests"] == 1)  # j5: the group is at its width
        hold[0].write_text("go")
        second = _paused(world, 5)
        assert second["free"] == 0 and second["rows"] == SLOTS  # four joined; every slot taken
        assert world.service.cancel("j2")
        hold[1].write_text("go")
        _paused(world, 7)  # j2 has ended; the service hands j5 the place, and the worker holds it before its slot
        assert _until(lambda: [row["id"] for row in world.log() if row["event"] == "join"][-1:] == ["j5"])
        assert _until(lambda: any(row["op"] == "join_received" and row["ids"] == ["j5"] for row in world.slots()))
        hold[2].write_text("go")
        world.wait()
        ends = check_ends(world, ["long", "short", "j1", "j2", "j3", "j4", "j5"])
        assert ends["j2"]["outcome"] == "cancelled" and isinstance(world.results["j2"], ProductError)
        for rid, text, limit in [("long", prompts[0], 80), ("short", prompts[1], 44)] + joiners:
            if rid != "j2":
                assert _tokens(world.results[rid]) == alone(text, limit), rid
        streamed = _tokens(world.results["j2"].events)
        assert streamed == alone(prompts[3], 40)[:len(streamed)]  # what it got before the cancel was its own
        check_pool(world)
        rows = world.slots()
        j2_free = next(i for i, row in enumerate(rows) if row["op"] == "free" and row["prompt"] == _key(prompts[3]))
        j5_take = next(i for i, row in enumerate(rows) if row["op"] == "take" and row["prompt"] == _key(prompts[6]))
        assert j5_take > j2_free and rows[j5_take]["free"] == 0  # it waited, then took the one slot j2 freed
        assert rows[j5_take]["slot"] == rows[j2_free]["slot"]
        assert ends["j5"]["outcome"] == "ok" and world.service.health()["worker"]["restarts"] == 0
    finally:
        world.close()


def _key(text: str) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(cpu.ChatTokenizer().encode(text)).encode()).hexdigest()[:12]


def test_a_request_beyond_one_slots_context_is_refused_alone_and_takes_no_slot(tmp_path, monkeypatch):
    """A slot's context (prompt plus output tokens) is a limit per request, not the pool's width: a request
    that needs more is refused on its own, takes no slot, and the others run, one exactly at the limit."""
    world = World(tmp_path, monkeypatch)
    try:
        exact = "x" * 16
        world.offer_together([("fits", exact, CAPACITY - 16), ("too-long", "y" * 16, CAPACITY - 15),
                              ("other", texts(1)[0], 12)])
        world.wait()
        ends = check_ends(world, ["fits", "too-long", "other"])
        refused = world.results["too-long"]
        assert isinstance(refused, ProductError) and refused.code == "invalid_request" and refused.events == []
        assert ends["fits"]["outcome"] == ends["other"]["outcome"] == "ok"
        assert _tokens(world.results["fits"]) == alone(exact, CAPACITY - 16)
        assert check_pool(world)["takes"] == 2
    finally:
        world.close()


def test_a_failed_admission_ends_every_member_once_returns_its_slots_and_the_worker_serves_again(tmp_path,
                                                                                                 monkeypatch):
    """Two requests join in one admission round and the second one's prefill fails. The Runtime replays the
    group's members on the sequential path (no admission there) and they keep their own answers; the joiners of
    that round end failed, each once; the slot the first joiner had already taken comes back with the rest (it
    stayed taken until the worker's end before STAB4); the worker is replaced and the next request is answered as
    it is alone. Run 6's pattern, with the first joiner failing, at `5f064c3`: request 0 answered at one row's
    speed, every joiner `backend_unavailable` as it ended."""
    hold = tmp_path / "resume"
    bad = "this prefill fails"
    world = World(tmp_path, monkeypatch, faults={"prefill_fail": bad, "pause": [[2, str(hold)]]})
    try:
        prompts = texts(4)
        world.offer_together([("a", prompts[0], 30), ("b", prompts[1], 30)])
        _paused(world, 2)
        for rid, text in (("c", prompts[3]), ("bad", bad)):
            world.submit(rid, text, 10)
            assert _until(lambda rid=rid: rid in [row["id"] for row in world.log() if row["event"] == "join"])
        assert _until(lambda: any(row["op"] == "join_received" and "bad" in row["ids"] for row in world.slots()))
        hold.write_text("go")
        world.wait()
        ends = check_ends(world, ["a", "b", "c", "bad"])
        assert [row["op"] for row in world.slots() if row["op"] in ("take", "prefill_fault")][-2:] == [
            "take", "prefill_fault"]  # c took a slot, then bad's prefill failed in the same round
        for rid, text in (("a", prompts[0]), ("b", prompts[1])):
            assert ends[rid]["outcome"] == "ok" and _tokens(world.results[rid]) == alone(text, 30), rid
        assert ends["c"]["outcome"] == ends["bad"]["outcome"] == "failed"
        check_pool(world)  # every slot the failed run held came back before the worker was replaced
        # The worker's note names what the Runtime's safety net caught, not only the later result-count error.
        assert _until(lambda: any(row["event"] == "worker_exit" for row in world.log()))
        tail = next(row for row in world.log() if row["event"] == "worker_exit")["stderr_tail"]
        note = json.loads(next(line for line in tail.splitlines() if '"worker_error"' in line))
        assert note["fallbacks"] == [{"where": "executor", "type": "RuntimeError", "out_of_memory": False,
                                      "illegal_memory_access": False}], note
        assert "injected" not in tail and bad not in tail  # types and flags only, never a message
        assert _until(lambda: world.service.health()["worker"]["state"] == "ready")
        world.submit("after", prompts[2], 12)
        world.wait()
        assert _tokens(world.results["after"]) == alone(prompts[2], 12)
        check_ends(world, ["a", "b", "c", "bad", "after"])
        check_pool(world)
    finally:
        world.close()


@pytest.mark.parametrize("fault", ["step_fail_once", "step_fail"])
def test_a_failed_stacked_step_is_replayed_once_or_ends_each_member_once_and_frees_every_slot(tmp_path, monkeypatch,
                                                                                              fault):
    """Execution: a stacked step that fails once is replayed inside the plan with no answer changed; a step that
    keeps failing for one member ends the group, each member once, with every slot back and the worker replaced."""
    poisoned = "this member's steps fail"
    world = World(tmp_path, monkeypatch, faults={fault: poisoned})
    try:
        prompts = texts(3)
        world.offer_together([("a", prompts[0], 20), ("p", poisoned, 20), ("b", prompts[1], 20)])
        world.wait()
        ends = check_ends(world, ["a", "p", "b"])
        assert any(row["op"] == "step_fault" for row in world.slots())
        check_pool(world)
        if fault == "step_fail_once":
            for rid, text in (("a", prompts[0]), ("p", poisoned), ("b", prompts[1])):
                assert _tokens(world.results[rid]) == alone(text, 20), rid
            assert world.service.health()["worker"]["restarts"] == 0
        else:
            # The plan's own fallback finishes the healthy members one by one; the poisoned one fails there and
            # again in the Runtime's replay, and the worker reports the group failed. Each ends once: the healthy
            # ones with their own answers (their done went out with their last token), the poisoned one failed.
            assert ends["p"]["outcome"] == "failed" and isinstance(world.results["p"], ProductError)
            for rid, text in (("a", prompts[0]), ("b", prompts[1])):
                if ends[rid]["outcome"] == "ok":
                    assert _tokens(world.results[rid]) == alone(text, 20), rid
            assert _until(lambda: world.service.health()["worker"]["state"] == "ready")
            world.submit("after", prompts[2], 12)
            world.wait()
            assert _tokens(world.results["after"]) == alone(prompts[2], 12)
            check_pool(world)
    finally:
        world.close()

