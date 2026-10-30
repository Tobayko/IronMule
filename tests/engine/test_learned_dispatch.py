"""LD1: the learned dispatch in the product -- parity with the tested policy, evidence binding, and
the hooks in the device pool and the scheduler (no model, no GPU)."""

from __future__ import annotations

import collections
import json
from pathlib import Path
import queue
import random
import threading
import time
from types import SimpleNamespace

import pytest

from ironmule_product import cli
from ironmule_product import learned_dispatch as ld
from ironmule_product.device_pool import DevicePool, gpu_name
from ironmule_product.service import ProductService
from ironmule_product.serving_policy import ServingPolicy
from ironmule_product.types import GenerationRequest
from tools import rl2_headroom as rl2

BINDING = {"model_id": "local/test", "revision": "abc", "gpu": "Tesla T4", "devices": 2}
MEASURED = {"model_id": "mlx-community/Qwen3-1.7B-bf16", "revision": "9cd6692855d3e06772228e9a962b2606359b2d24",
            "gpu": "Tesla T4", "devices": 2, "engine": "tuned"}  # RL4-S, RL6 and LD1-V's setup


def _request(max_tokens: int, long_prompt: bool = False, request_id: str | None = None) -> GenerationRequest:
    content = "x" * (ld.LONG_PROMPT_CHARS + 1) if long_prompt else "hello"
    return GenerationRequest("local/test", (("user", content),), max_tokens=max_tokens,
                             **({"request_id": request_id} if request_id else {}))


def _policy(seed: int, widths: bool = False) -> rl2.LearnedPolicy:
    rng = random.Random(seed)
    policy = rl2.LearnedPolicy([rng.uniform(-3, 3) for _ in range(rl2.PLACEMENT_FEATURES)],
                               [rng.uniform(-3, 3) for _ in range(rl2.ORDER_FEATURES)])
    if widths:
        policy.width = [rng.uniform(-3, 3) for _ in range(len(rl2.WIDTH_CHOICES) * rl2.ORDER_FEATURES)]
    return policy


def _files(tmp_path, policy: rl2.LearnedPolicy, verdict_low: float = 0.07, binding=BINDING):
    data = policy.to_json()
    learned = {"rival": "shortest/w8/g0/sjf", "gain": 0.09, "gain_ci95": [verdict_low, 0.11]}
    evidence = ld.evidence_for(data["sha256"], "RL4-S-run1-20261029", binding, ["spike"], learned, noise=0.029)
    policy_path, evidence_path = tmp_path / "policy.json", tmp_path / "evidence.json"
    policy_path.write_text(json.dumps(data))
    evidence_path.write_text(json.dumps(evidence))
    return policy_path, evidence_path


def test_the_product_decides_exactly_as_the_policy_the_hardware_tested():
    rng = random.Random(7)
    for seed in range(20):
        tested = _policy(seed, widths=seed % 2 == 0)
        product = ld.Policy.from_json(tested.to_json())
        for _ in range(25):
            cards = [[rl2._View(rng.choice((32, 256)), rng.random() < 0.5) for _ in range(rng.randrange(0, 7))]
                     for _ in range(2)]
            request = rl2._View(rng.choice((32, 256)), rng.random() < 0.5)
            waiting = [rl2._View(rng.choice((32, 256)), rng.random() < 0.5) for _ in range(rng.randrange(1, 9))]
            views = [[ld.View(r.max_tokens, r.long_prompt) for r in card] for card in cards]
            assert product.best_card(ld.View(request.max_tokens, request.long_prompt), views) == \
                tested.best_card(request, [rl2._CardView(card) for card in cards])
            mine = [ld.View(r.max_tokens, r.long_prompt) for r in waiting]
            assert product.uses_sjf(mine) == tested.uses_sjf(waiting)
            if tested.width is not None:
                assert product.best_width(mine) == tested.best_width(waiting)


def test_evidence_for_this_policy_and_setup_activates_it(tmp_path):
    policy = _policy(1)
    dispatch = ld.LearnedDispatch.load(*_files(tmp_path, policy), BINDING)
    assert dispatch.active and dispatch.reason is None
    status = dispatch.status()
    assert status["policy_sha256"] == policy.to_json()["sha256"] and status["domain"] == ["spike"]
    assert status["experiment"] == "RL4-S-run1-20261029"


@pytest.mark.parametrize("breaks,reason", [
    ("policy", "digest"), ("evidence", "digest"), ("other_policy", "different policy"),
    ("failed", "passed comparison"), ("revision", "revision"), ("gpu", "gpu"), ("devices", "devices"),
    ("missing", "No such file"),
])
def test_anything_that_does_not_prove_the_policy_here_keeps_the_reference(tmp_path, breaks, reason):
    policy_path, evidence_path = _files(tmp_path, _policy(2), verdict_low=0.01 if breaks == "failed" else 0.07)
    binding = dict(BINDING)
    if breaks == "policy":
        data = json.loads(policy_path.read_text())
        data["order"][0] += 1.0
        policy_path.write_text(json.dumps(data))
    elif breaks == "evidence":
        data = json.loads(evidence_path.read_text())
        data["gain"] = 0.5
        evidence_path.write_text(json.dumps(data))
    elif breaks == "other_policy":
        (tmp_path / "other").mkdir()
        _, evidence_path = _files(tmp_path / "other", _policy(3))
    elif breaks in ("revision", "gpu", "devices"):
        binding[breaks] = {"revision": "def", "gpu": "Tesla P100", "devices": 1}[breaks]
    elif breaks == "missing":
        evidence_path = tmp_path / "absent.json"
    dispatch = ld.LearnedDispatch.load(policy_path, evidence_path, binding)
    assert not dispatch.active and reason in dispatch.reason and "reference dispatch serves" in dispatch.reason
    assert dispatch.card(_request(32), [[], []]) is None and dispatch.sjf([_request(32)]) is None


def test_a_policy_that_fails_while_deciding_falls_back_and_is_withdrawn(tmp_path):
    dispatch = ld.LearnedDispatch.load(*_files(tmp_path, _policy(4)), BINDING)
    broken = SimpleNamespace(max_tokens="not a number", messages=(("user", "hi"),))
    for _ in range(ld.MAX_ERRORS):
        assert dispatch.card(broken, [[], []]) is None
    assert not dispatch.active and "withdrawn: 3 errors while deciding" in dispatch.reason


def test_the_operator_turns_it_off_and_it_stays_off_across_a_restart_until_requalified(tmp_path):
    policy = _policy(5)
    files, kill = _files(tmp_path, policy), tmp_path / "state" / "dispatch-withdrawals.json"
    dispatch = ld.LearnedDispatch.load(*files, BINDING, kill_path=kill)
    assert dispatch.active and ld.status_of(None)["reason"] == ld.NOT_CONFIGURED
    with pytest.raises(ValueError, match="off"):
        ld.control(dispatch, "on")
    with pytest.raises(ValueError, match="configured"):
        ld.control(None, "off")
    status = ld.control(dispatch, "off")
    assert not status["active"] and "operator turned it off" in status["reason"]
    assert dispatch.card(_request(32), [[], []]) is None  # the reference decides from now on
    restarted = ld.LearnedDispatch.load(*files, BINDING, kill_path=kill)
    assert not restarted.active and "withdrawn" in restarted.reason and kill.name in restarted.reason
    (tmp_path / "rerun").mkdir()
    rerun = policy.to_json()
    learned = {"rival": "shortest/w8/g0/sjf", "gain": 0.08, "gain_ci95": [0.06, 0.10]}
    (tmp_path / "rerun" / "evidence.json").write_text(json.dumps(
        ld.evidence_for(rerun["sha256"], "RL4-S-run2", BINDING, ["spike"], learned, noise=0.029)))
    requalified = ld.LearnedDispatch.load(files[0], tmp_path / "rerun" / "evidence.json", BINDING, kill_path=kill)
    assert requalified.active  # new passed evidence for the same policy qualifies it again
    broken = ld.LearnedDispatch.load(*_files(tmp_path / "rerun", _policy(6)), BINDING, kill_path=kill)
    assert broken.active  # another policy is not covered by this withdrawal
    kill.write_text("{not json")
    assert not ld.LearnedDispatch.load(*files, BINDING, kill_path=kill).active  # unreadable: fail closed


def test_an_error_withdrawal_is_recorded_and_survives_a_restart(tmp_path):
    files, kill = _files(tmp_path, _policy(7)), tmp_path / "dispatch-withdrawals.json"
    dispatch = ld.LearnedDispatch.load(*files, BINDING, kill_path=kill)
    broken = SimpleNamespace(max_tokens="not a number", messages=(("user", "hi"),))
    for _ in range(ld.MAX_ERRORS):
        dispatch.card(broken, [[], []])
    record = json.loads(kill.read_text())
    assert record["schema"] == ld.WITHDRAWALS_SCHEMA and "errors while deciding" in record["withdrawn"][0]["reason"]
    assert not ld.LearnedDispatch.load(*files, BINDING, kill_path=kill).active


def test_the_http_route_reports_and_withdraws_the_learned_dispatch(tmp_path):
    import http.client

    from ironmule_product.http_server import create_server

    pool = DevicePool([_Shard([]), _Shard([])], ["0", "1"])
    try:
        server = create_server(pool, port=0)
    except PermissionError as exc:
        pytest.skip(f"loopback bind denied by test environment: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def call(method, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        connection.request(method, "/v1/ironmule/dispatch", None if body is None else json.dumps(body).encode(),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    try:
        assert call("GET") == (200, ld.status_of(None))
        assert call("POST", {"action": "off"})[0] == 400  # nothing to withdraw
        pool.attach(ld.LearnedDispatch.load(*_files(tmp_path, _policy(8)), BINDING))
        status, body = call("GET")
        assert status == 200 and body["active"] is True
        assert call("POST", {"action": "on"})[0] == 400 and call("POST", {"off": True})[0] == 400
        status, body = call("POST", {"action": "off"})
        assert status == 200 and body["active"] is False and "operator" in body["reason"]
        assert pool.health()["dispatch"] == "join_shortest_queue"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _Shard:
    """The part of ProductService the pool's placement uses."""

    def __init__(self, admitted):
        self.spec, self.store = "spec", SimpleNamespace(root=None, models=lambda: [])
        self.stability_log = self.serving_log = self.dispatch = None
        self._admitted = list(admitted)
        self.streamed = []

    def accepting(self):
        return True

    def outstanding(self):
        return len(self._admitted)

    def holds(self, request_id):
        return False

    def admitted_requests(self):
        return list(self._admitted)

    def stream(self, request):
        self.streamed.append(request)
        self._admitted.append(request)
        return iter([])

    def health(self):
        return {"service": "ironmule", "ready": True, "queued_requests": 0, "active_requests": 0,
                "completed_requests": 0, "failed_requests": 0, "cancelled_requests": 0, "backend": "x",
                "loaded_model": "local/test"}


def test_the_pool_places_by_the_learned_policy_and_by_the_shortest_queue_without_it(tmp_path):
    long_policy = rl2.LearnedPolicy([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, -5.0, 0.0, 0.0, 0.0], [0.0] * 4)
    first, second = _Shard([]), _Shard([_request(32)])  # the second GPU already holds one request
    pool = DevicePool([first, second], ["0", "1"])
    list(pool.stream(_request(256, request_id="plain")))
    assert first.streamed and not second.streamed  # the shortest queue
    pool.attach(ld.LearnedDispatch.load(*_files(tmp_path, long_policy), BINDING))
    assert first.dispatch is pool.dispatch and second.dispatch is pool.dispatch
    list(pool.stream(_request(256, request_id="long")))  # the policy sends long requests to the last GPU
    assert [r.request_id for r in second.streamed] == ["long"]
    health = pool.health()
    assert health["dispatch"] == "learned" and health["learned_dispatch"]["decisions"]["placement"] == 1
    pool.attach(ld.LearnedDispatch(None, None, "learned dispatch is off: test"))
    assert pool.health()["dispatch"] == "join_shortest_queue"


def _scheduler(dispatch):
    service = object.__new__(ProductService)
    service._lock, service._carry, service._pending = threading.RLock(), collections.deque(), queue.Queue()
    service.policy = ServingPolicy(max_width=8, source="test")
    service.backend = SimpleNamespace(batch_capacity=8, can_batch_requests=lambda requests: True)
    service.dispatch = dispatch
    return service


def _sessions(limits):
    return [SimpleNamespace(request=_request(limit), submitted_ns=time.monotonic_ns() + i)
            for i, limit in enumerate(limits)]


def test_the_scheduler_orders_and_sizes_groups_by_the_learned_policy(tmp_path):
    always = rl2.LearnedPolicy([0.0] * 10, [10.0, 0.0, 0.0, 0.0])
    never = rl2.LearnedPolicy([0.0] * 10, [-10.0, 0.0, 0.0, 0.0])
    for policy, expected in ((always, [3, 6, 9]), (never, [9, 3, 6])):
        service = _scheduler(ld.LearnedDispatch.load(*_files(tmp_path, policy), BINDING))
        for session in _sessions([9, 3, 6]):
            service._pending.put(session)
        assert [service._next_session().request.max_tokens for _ in range(3)] == expected
    narrow = rl2.LearnedPolicy.zero(widths=True)
    narrow.width[rl2.WIDTH_CHOICES.index(2) * rl2.ORDER_FEATURES] = 5.0
    service = _scheduler(ld.LearnedDispatch.load(*_files(tmp_path, narrow), BINDING))
    first, *rest = _sessions([32] * 5)
    service._carry.extend(rest)
    assert len(service._group_for(first)) == 2
    reference = _scheduler(None)  # no dispatch: first in, first out, the policy's width
    first, *rest = _sessions([9, 3, 6, 4])
    reference._carry.extend(rest)
    assert [s.request.max_tokens for s in reference._group_for(first)] == [9, 3, 6, 4]


def test_the_gpu_name_binding_comes_from_nvidia_smi():
    listed = SimpleNamespace(returncode=0, stdout="0, GPU-aa, Tesla T4\n1, GPU-bb, Tesla T4\n")
    assert gpu_name(["0", "1"], lambda *a, **k: listed) == "Tesla T4"
    assert gpu_name(["GPU-bb"], lambda *a, **k: listed) == "Tesla T4"
    assert gpu_name(["2"], lambda *a, **k: listed) is None
    assert gpu_name(["0"], lambda *a, **k: SimpleNamespace(returncode=1, stdout="")) is None


def test_serve_takes_the_policy_and_its_evidence_together_and_never_with_live_learning(tmp_path):
    state = str(tmp_path / "state")
    for argv in (["--dispatch-policy", "p.json"], ["--dispatch-evidence", "e.json"],
                 ["--dispatch-policy", "p.json", "--dispatch-evidence", "e.json", "--learning", "active"]):
        with pytest.raises(SystemExit):
            cli.serve(["--no-model", "--state-dir", state, *argv])


def test_the_shipped_rl4_s_policy_loads_only_on_the_setup_it_was_measured_on():
    folder = Path(ld.__file__).with_name("dispatch") / "rl4s-t4x2-qwen3-1.7b"
    dispatch = ld.LearnedDispatch.load(folder / "policy.json", folder / "evidence.json", MEASURED)
    assert dispatch.active and dispatch.policy.sha256.startswith("e3f07b833e90")
    assert dispatch.status()["domain"] == ["spike"] and dispatch.status()["gain_ci95"][0] > 0.029
    for key, other in (("devices", 1), ("gpu", "NVIDIA A100"), ("revision", "main"), ("engine", "reference")):
        assert not ld.LearnedDispatch.load(folder / "policy.json", folder / "evidence.json",
                                           {**MEASURED, key: other}).active


def test_a_stored_policy_keeps_its_settings_and_may_start_shortest_first(tmp_path):
    from ironmule_product.serving_policy import POLICY_FILE, load_policy, save_policy

    (tmp_path / POLICY_FILE).write_text(json.dumps({"max_width": 4, "join": False, "gather_ms": 50, "source": "dream"}))
    assert load_policy(tmp_path) == ServingPolicy(max_width=4, gather_ms=50, source="dream")  # stored before RL6
    save_policy(tmp_path, ServingPolicy(max_width=2, order="sjf", source="test"))
    assert load_policy(tmp_path).order == "sjf"
    with pytest.raises(ValueError):
        ServingPolicy(order="random")


def test_the_qualified_schedule_serves_only_on_the_setup_it_was_measured_on(tmp_path):
    import shutil

    from ironmule_product.serving_policy import qualified_schedule

    policy, record = qualified_schedule(MEASURED)
    assert (policy.max_width, policy.order, policy.gather_ms) == (2, "sjf", 0)
    assert record["experiment"] == "LD1-V-run1-20261029" and record["gain_ci95"][0] > record["noise"]
    assert record["domain"] == ["quiet", "bursts", "spike"]
    for key, other in (("devices", 1), ("gpu", "NVIDIA A100"), ("revision", "main"), ("engine", "reference"),
                       ("model_id", "mlx-community/Qwen3-4B-bf16")):
        assert qualified_schedule({**MEASURED, key: other}) is None
    shipped = Path(ld.__file__).with_name("dispatch") / "rl6-t4x2-qwen3-1.7b" / "schedule.json"
    (tmp_path / "copy").mkdir()
    tampered = json.loads(shipped.read_text())
    tampered["policy"]["max_width"] = 8
    (tmp_path / "copy" / "schedule.json").write_text(json.dumps(tampered))
    assert qualified_schedule(MEASURED, folder=tmp_path) is None  # the digest no longer holds
    shutil.copy(shipped, tmp_path / "copy" / "schedule.json")
    assert qualified_schedule(MEASURED, folder=tmp_path) is not None


def test_a_shortest_first_policy_orders_and_narrows_groups_without_any_learned_dispatch():
    service = _scheduler(None)
    service.policy = ServingPolicy(max_width=2, order="sjf", source="test")
    for session in _sessions([9, 3, 6]):
        service._pending.put(session)
    assert [service._next_session().request.max_tokens for _ in range(3)] == [3, 6, 9]
    first, *rest = _sessions([9, 3, 6, 4])
    service._carry.extend(rest)
    assert [s.request.max_tokens for s in service._group_for(first)] == [9, 3]  # width 2, the shortest joins
