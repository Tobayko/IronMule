"""Scripted-backend integration checks; these are not hardware measurements."""

from __future__ import annotations

import copy
import threading
from types import SimpleNamespace

import pytest

from ironmule import service
from ironmule.model_identity import ModelIdentityError
from ironmule.plans import ReusableSessionPlan, StrictOneShotPlan
from ironmule.runtime import Knobs
from ironmule.service import InteractiveMode, Request, Runtime, ThroughputMode
from tests.engine.test_ironmule_runtime import FakeBackend, PROMPTS, SCRIPTS


class ScriptedController:
    def __init__(self, identity, actions=(0,), return_arm=-1):
        self.identity = identity
        self.actions = actions
        self.return_arm = return_arm
        self.contexts = []
        self.outcomes = []

    def run(self, context, execute):
        self.contexts.append(context)
        outcomes = [execute(action) for action in self.actions]
        self.outcomes.extend(outcomes)
        return outcomes[self.return_arm].value


@pytest.fixture
def runtime(monkeypatch):
    item = Runtime.__new__(Runtime)
    item.engine = SimpleNamespace(knobs=Knobs(), compute_dtype=None)
    item.model_identity = SimpleNamespace(identity_sha256="a" * 64)
    item.backend = FakeBackend(SCRIPTS)
    item.mode = InteractiveMode()
    item.tokenizer = SimpleNamespace(decode=lambda tokens: " ".join(map(str, tokens)))
    identity = {
        "schema": "explicit_scripted_backend_identity.v1", "backend": "scripted",
        "runtime": {"model_identity_sha256": "a" * 64,
                    "hardware_fingerprint": "scripted-machine"},
        "compute_dtype": None, "grouping_supported": True,
        "eos_ids": list(item.backend.eos_ids),
    }
    # Only this explicit scripted fixture supplies a replacement identity helper.
    # The production helper refuses unavailable exact model/backend identity.
    def scripted_identity(_, *, action_contract=None, hardware_binding=None):
        result = copy.deepcopy(identity)
        if action_contract is not None:
            result["action_contract"] = copy.deepcopy(action_contract)
            result["profiles"] = {str(i): name for i, name in enumerate(action_contract["profiles"])}
        if hardware_binding is not None:
            result["hardware_probe_binding"] = copy.deepcopy(hardware_binding)
        return result
    monkeypatch.setattr(service, "runtime_identity", scripted_identity)
    item.test_identity = identity
    return item


def requests():
    return [Request(prompt_ids=prompt, rid=f"request-{index}", max_tokens=8)
            for index, prompt in enumerate(PROMPTS[:4])]


def signature(rows):
    return [(row.rid, row.tokens, row.stop_reason,
             row.metrics["physical_generated_tokens"],
             row.metrics["visible_generated_tokens"]) for row in rows]


def attach(runtime, actions=(0,), return_arm=-1):
    controller = ScriptedController(copy.deepcopy(runtime.test_identity), actions, return_arm)
    runtime.attach_online_controller(controller)
    return controller


def test_decision_reaches_distinct_executors_with_identical_outputs(runtime):
    reference = runtime.serve(requests())
    controller = attach(runtime, (1,))
    runtime.backend.completed_widths.clear()
    grouped = runtime.serve(requests())
    assert signature(grouped) == signature(reference)
    assert max(runtime.backend.completed_widths) == 4
    assert runtime.telemetry.mode == "throughput"
    assert runtime.telemetry.routing["online_controller"]["profile"] == "grouped4.v1"
    assert controller.outcomes[-1].generated_tokens == sum(len(row.tokens) for row in grouped)
    assert controller.outcomes[-1].elapsed_s > 0
    assert runtime.telemetry.correctness_check_performed is False
    assert runtime.telemetry.plan_switch_attempts == 0


def test_comparison_builds_fresh_sessions_and_restores_returned_arm(runtime):
    controller = attach(runtime, (0, 0, 1), return_arm=0)
    work = requests()
    plans = [request.plan for request in work]
    rows = runtime.serve(work)
    assert len(controller.outcomes) == 3
    assert len({outcome.signature for outcome in controller.outcomes}) == 1
    assert runtime.telemetry is controller.outcomes[0].value[1]
    assert runtime.telemetry.mode == "interactive"
    assert rows is controller.outcomes[0].value[0]
    assert [request.plan for request in work] == plans
    assert all(request.plan is plan for request, plan in zip(work, plans))
    outer = runtime.telemetry.routing["online_controller"]["outer_wall_ns"]
    assert outer >= sum(outcome.elapsed_s for outcome in controller.outcomes) * 1e9
    assert all(row.metrics["caller_return_latency_ms"] == outer / 1e6 for row in rows)


def test_each_controlled_arm_uses_one_request_arrival_timestamp(runtime):
    controller = attach(runtime, (0, 0, 1), return_arm=0)
    runtime.serve(requests())
    arrivals = []
    for outcome in controller.outcomes:
        metrics = outcome.value[1].requests
        assert len({item.arrival_ns for item in metrics}) == 1
        assert all(item.engine_start_ns >= item.arrival_ns for item in metrics)
        arrivals.append(metrics[0].arrival_ns)
    assert arrivals == sorted(set(arrivals))


def test_caller_supplied_dispatch_timestamp_keeps_original_path(runtime):
    controller = attach(runtime, (1,))
    runtime.serve(requests(), dispatch_ns=1)
    assert controller.contexts == []
    assert all(item.arrival_ns == 1 for item in runtime.telemetry.requests)


@pytest.mark.parametrize("case", ["reusable", "custom", "arrival", "latency", "mode"])
def test_unsupported_contracts_bypass_control(runtime, case):
    controller = attach(runtime, (1,))
    work = requests()
    if case == "reusable":
        work[0].plan = ReusableSessionPlan([1])
    elif case == "custom":
        class CustomStrictPlan(StrictOneShotPlan):
            pass
        work[0].plan = CustomStrictPlan()
    elif case == "arrival":
        work[0].arrival_ms = 0.1
    elif case == "latency":
        work[0].objective = "latency"
    else:
        runtime.mode = ThroughputMode()
    original_plans = [request.plan for request in work]
    rows = runtime.serve(work)
    assert len(rows) == len(work)
    assert controller.contexts == []
    assert [request.plan for request in work] == original_plans
    assert "online_controller" not in runtime.telemetry.routing


@pytest.mark.parametrize("case", ["knobs", "precision", "model", "eos"])
def test_mutated_engine_contract_cannot_reuse_bound_controller(runtime, case):
    controller = attach(runtime, (1,))
    if case == "knobs":
        runtime.engine.knobs = Knobs(readback_every=2)
    elif case == "precision":
        runtime.engine.compute_dtype = "float32"
    elif case == "model":
        runtime.model_identity = SimpleNamespace(identity_sha256="b" * 64)
    else:
        runtime.backend.eos_ids = (99, 98)
    runtime.serve(requests())
    assert controller.contexts == []


def test_single_request_and_hybrid_contract_only_offer_reference(runtime):
    controller = attach(runtime)
    runtime.serve(requests()[:1])
    assert controller.contexts[-1].eligible_mask == 1
    runtime.online_controller = None
    runtime.test_identity["grouping_supported"] = False
    controller = attach(runtime)
    runtime.serve(requests())
    assert controller.contexts[-1].eligible_mask == 1
    assert max(runtime.backend.completed_widths) == 1


def test_controller_cannot_execute_masked_grouped_profile(runtime):
    attach(runtime, (1,))
    with pytest.raises(ValueError, match="inadmissible"):
        runtime.serve(requests()[:1])
    assert runtime.backend.completed_widths == []


def test_workload_digest_binds_contents_order_and_caps_without_prompt_storage(runtime):
    controller = attach(runtime)
    work = requests()
    runtime.serve(work)
    first = controller.contexts[-1].workload
    work[0].prompt_ids = [1, 42] + [1] * 8  # same length, different content
    runtime.serve(work)
    second = controller.contexts[-1].workload
    assert first["prompt_tokens"] == second["prompt_tokens"]
    assert first["request_sha256"] != second["request_sha256"]
    runtime.serve(list(reversed(work)))
    assert second["request_sha256"] != controller.contexts[-1].workload["request_sha256"]
    work[0].max_tokens = 7
    runtime.serve(work)
    assert second["request_sha256"] != controller.contexts[-1].workload["request_sha256"]
    assert set(first) == {"schema", "plan", "prompt_tokens", "token_caps", "capacity",
                          "eos_ids", "request_sha256"}


def test_group_failure_preserves_tokens_and_is_reported_to_controller(runtime):
    reference = signature(runtime.serve(requests()))
    runtime.backend = FakeBackend(SCRIPTS, fail_on_group_call=2)
    controller = attach(runtime, (1,))
    assert signature(runtime.serve(requests())) == reference
    assert controller.outcomes[-1].fallback_count == 1


def test_reference_failure_propagates_without_hidden_retry(runtime, monkeypatch):
    attach(runtime)
    calls = []
    def fail(*_):
        calls.append(1)
        raise RuntimeError("scripted prefill failure")
    monkeypatch.setattr(runtime.backend, "prefill", fail)
    with pytest.raises(RuntimeError, match="prefill failure"):
        runtime.serve(requests())
    assert len(calls) == 1


def test_identity_mismatch_and_explicit_modes_refuse_attachment(runtime):
    foreign = copy.deepcopy(runtime.test_identity)
    foreign["runtime"]["model_identity_sha256"] = "b" * 64
    with pytest.raises(ModelIdentityError, match="differs"):
        runtime.attach_online_controller(ScriptedController(foreign))
    runtime.mode = ThroughputMode()
    with pytest.raises(ValueError, match="reference mode"):
        runtime.attach_online_controller(ScriptedController(runtime.test_identity))


def test_production_identity_refuses_unidentified_execution_only_runtime():
    item = SimpleNamespace(model_identity=None, backend=object())
    with pytest.raises(ModelIdentityError, match="identified MLX"):
        service.runtime_identity(item)


def test_shutdown_closes_controller_and_engine_even_after_controller_error(runtime):
    controller = attach(runtime)
    closed = []
    def fail_close():
        closed.append("controller")
        raise RuntimeError("scripted shutdown failure")
    controller.close = fail_close
    runtime.engine.close = lambda: closed.append("engine")
    with pytest.raises(RuntimeError, match="shutdown failure"):
        runtime.close()
    assert closed == ["controller", "engine"]


def test_explicit_successful_comparison_is_visible_in_runtime_telemetry(runtime):
    controller = attach(runtime, (0, 0, 1), return_arm=0)
    controller.last_decision = {"comparison": True, "correctness": "matched"}
    rows = runtime.serve(requests())
    assert runtime.telemetry.correctness_check_performed is True
    assert runtime.telemetry.correctness_checked_requests == len(rows)
    assert runtime.telemetry.correctness_errors == 0


def test_attached_runtime_serializes_model_calls_including_bypasses(runtime, monkeypatch):
    controller = attach(runtime)
    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    original = runtime._serve_group
    counts = []
    def held_group(work, stamp, mode):
        counts.append(threading.current_thread().name)
        if len(counts) == 1:
            entered.set()
            assert release.wait(3)
        return original(work, stamp, mode)
    monkeypatch.setattr(runtime, "_serve_group", held_group)
    errors = []
    def run(work, signal=None):
        try:
            if signal:
                signal.set()
            runtime.serve(work)
        except BaseException as error:
            errors.append(error)
    first = threading.Thread(target=run, args=(requests(),), name="first")
    bypass = requests()
    bypass[0].objective = "latency"
    second = threading.Thread(target=run, args=(bypass, second_started), name="second")
    first.start()
    try:
        assert entered.wait(3)
        second.start()
        assert second_started.wait(3)
        assert counts == ["first"]
    finally:
        release.set()
        first.join(3)
        if second.ident is not None:
            second.join(3)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert counts == ["first", "second"]
    assert len(controller.contexts) == 1


def attach_profiles(runtime, actions=(0,), return_arm=-1):
    from ironmule_product.engine_bridge import serving_profile_contract
    contract = serving_profile_contract(runtime.engine.knobs.as_dict(),
                                        grouping_supported=runtime.test_identity["grouping_supported"])
    identity = service.runtime_identity(runtime, action_contract=contract)
    controller = ScriptedController(identity, actions, return_arm)
    controller.action_contract = contract
    runtime.attach_online_controller(controller)
    return controller


def test_effective_profiles_change_only_allowed_knobs_and_restore_caller_state(runtime, monkeypatch):
    original_knobs = Knobs(fused_argmax=True, readback_every=8, capacity_slack=128)
    runtime.engine.knobs = original_knobs
    original_compiled = object()
    runtime.engine._compiled = original_compiled
    runtime.engine._compiled_capacity = (256, 1)
    controller = attach_profiles(runtime, (2, 3, 2))
    original_group = runtime._serve_group
    seen = []
    compiled = object()
    def group(work, stamp, mode):
        selected = runtime.engine.knobs
        seen.append((selected.as_dict(), runtime.engine._compiled, mode.name))
        runtime.engine._compiled = compiled
        runtime.engine._compiled_capacity = (384, 1)
        return original_group(work, stamp, mode)
    monkeypatch.setattr(runtime, "_serve_group", group)
    rows = runtime.serve(requests())
    assert all(values["compiled_fixed_cache"] and values["head_skip_prefill"] for values, _, _ in seen)
    assert all(values["fused_argmax"] and values["readback_every"] == 8
               and values["capacity_slack"] == 128 for values, _, _ in seen)
    assert seen[0][1] is None
    assert seen[1][1] is compiled and seen[2][1] is compiled
    assert [mode for _, _, mode in seen] == ["interactive", "throughput", "interactive"]
    assert len({outcome.signature for outcome in controller.outcomes}) == 1
    assert runtime.engine.knobs is original_knobs
    assert runtime.engine._compiled is original_compiled
    assert runtime.engine._compiled_capacity == (256, 1)
    assert len(runtime._online_compiled_cache) == 1
    assert runtime.telemetry.routing["online_controller"]["profile_id"] == "core.sequential.v1"
    assert all(row.metrics["caller_return_latency_ms"] > 0 for row in rows)


def test_profile_switch_failure_restores_exact_state_and_discards_partial_compiler(runtime, monkeypatch):
    original_knobs = runtime.engine.knobs
    original_compiled = object()
    runtime.engine._compiled = original_compiled
    runtime.engine._compiled_capacity = (256, 1)
    attach_profiles(runtime, (2,))
    def fail(*_):
        runtime.engine._compiled = object()
        runtime.engine._compiled_capacity = (512, 1)
        raise RuntimeError("scripted profile failure")
    monkeypatch.setattr(runtime, "_serve_group", fail)
    with pytest.raises(RuntimeError, match="profile failure"):
        runtime.serve(requests())
    assert runtime.engine.knobs is original_knobs
    assert runtime.engine._compiled is original_compiled
    assert runtime.engine._compiled_capacity == (256, 1)
    assert runtime._online_compiled_cache == {}


def test_profiles_filter_grouping_without_disabling_sequential_alternatives(runtime):
    controller = attach_profiles(runtime, (2,))
    runtime.serve(requests()[:1])
    assert controller.contexts[-1].eligible_mask == 0b0101
    assert max(runtime.backend.completed_widths) == 1
    runtime.online_controller = None
    runtime.test_identity["grouping_supported"] = False
    controller = attach_profiles(runtime, (1,))
    runtime.serve(requests())
    assert controller.contexts[-1].eligible_mask == 0b11
    assert len(controller.action_contract["definitions"]) == 2
    assert max(runtime.backend.completed_widths) == 1


def test_profile_contract_cannot_change_unallowed_caller_knobs(runtime):
    from ironmule_product.engine_bridge import serving_profile_contract
    contract = serving_profile_contract(runtime.engine.knobs.as_dict())
    contract["definitions"][2]["knobs"]["fused_argmax"] = True
    controller = ScriptedController(service.runtime_identity(runtime, action_contract=contract))
    controller.action_contract = contract
    with pytest.raises(ModelIdentityError, match="profiles differ"):
        runtime.attach_online_controller(controller)


def test_profile_verification_captures_terminal_state_only_when_requested(runtime, monkeypatch):
    from friday_evidence.canonical import canonical_sha256
    controller = attach_profiles(runtime, (0, 2, 3), return_arm=0)
    hashes = []
    def hasher(state, offset):
        hashes.append(offset)
        return canonical_sha256({"state": state, "offset": offset})
    runtime.backend.kv_hash = hasher
    runtime.serve(requests())
    assert hashes == []
    controller.last_decision = {"verification": True}
    runtime.serve(requests())
    assert len(hashes) == 3 * len(requests())
    assert len({item.signature for item in controller.outcomes[-3:]}) == 1
    assert all(len(item.signature[1]) == len(requests()) for item in controller.outcomes[-3:])
    for item in controller.outcomes[-3:]:
        for request, (_, offset, digest) in zip(requests(), item.signature[1]):
            row = next(row for row in item.value[0] if row.rid == request.rid)
            assert offset == len(request.prompt_ids) + len(row.tokens) - 1
            assert len(digest) == 64


def test_changed_terminal_state_is_visible_despite_identical_tokens(runtime):
    from friday_evidence.canonical import canonical_sha256
    controller = attach_profiles(runtime, (0, 2), return_arm=0)
    controller.last_decision = {"comparison": True}
    runtime.backend.kv_hash = lambda state, offset: canonical_sha256({
        "state": state, "offset": offset,
        "scripted_state_variant": runtime.engine.knobs.compiled_fixed_cache})
    runtime.serve(requests())
    a, b = controller.outcomes
    assert a.signature[0] == b.signature[0]
    assert a.signature[1] != b.signature[1]
    assert a.signature != b.signature


def test_unsupported_terminal_hash_cannot_be_reported_as_a_checked_comparison(runtime):
    controller = attach_profiles(runtime, (2,))
    controller.last_decision = {"comparison": True}
    before = runtime.engine.knobs
    with pytest.raises(RuntimeError, match="hashing is unsupported"):
        runtime.serve(requests())
    assert controller.outcomes == []
    assert runtime.engine.knobs is before
    assert runtime._online_compiled_cache == {}


def test_hardware_learning_reuses_probe_catalog_controller_and_stable_identity(runtime, monkeypatch, tmp_path):
    from ironmule import hw, online_controller
    calls = []
    record = {"static": {"chip": "scripted"}, "fingerprint": "scripted-machine",
              "binding": {"schema": "explicit_scripted_binding.v1", "fingerprint": "scripted-machine"},
              "measured": {"dispatch_us": 1.0}, "observed_unix_ns": 1}
    def probe(**kwargs):
        calls.append(("probe", kwargs))
        return copy.deepcopy(record)
    def validate(value, **kwargs):
        calls.append(("validate", kwargs))
        assert value["binding"] == record["binding"]
        return True, []
    created = []
    def create(identity, **kwargs):
        controller = ScriptedController(identity)
        controller.action_contract = kwargs["action_contract"]
        controller.close = lambda: None
        created.append((controller, kwargs))
        return controller
    monkeypatch.setattr(hw, "probe", probe)
    monkeypatch.setattr(hw, "validate_probe", validate)
    monkeypatch.setattr(online_controller, "OnlineController", create)
    config = online_controller.ControllerConfig(verify_exploration=False)
    controller = runtime.enable_hardware_learning(tmp_path / "controller.json", config=config,
                                                 probe_cache_dir=tmp_path / "probe")
    assert controller is runtime.online_controller
    assert created[0][1]["config"].verify_exploration is True
    assert config.verify_exploration is False
    assert [name for name, _ in calls] == ["probe", "validate"]
    assert calls[0][1]["cache_dir"] == tmp_path / "probe"
    assert calls[0][1]["allow_measure"] is False
    assert created[0][1]["hardware_record"] == record
    assert "measured" not in controller.identity
    first_identity = copy.deepcopy(controller.identity)
    runtime.online_controller = None
    record["measured"]["dispatch_us"] = 2.0
    record["observed_unix_ns"] = 2
    second = runtime.enable_hardware_learning(tmp_path / "controller.json")
    assert second.identity == first_identity
    with pytest.raises(ValueError, match="already owns"):
        runtime.enable_hardware_learning(tmp_path / "other.json")


def test_hardware_learning_refuses_invalid_or_foreign_probe_before_controller(runtime, monkeypatch, tmp_path):
    from ironmule import hw
    record = {"static": {}, "fingerprint": "foreign", "binding": {"schema": "scripted"}}
    monkeypatch.setattr(hw, "probe", lambda **_: record)
    monkeypatch.setattr(hw, "validate_probe", lambda *_args, **_kwargs: (False, ["corrupt_probe"]))
    with pytest.raises(ModelIdentityError, match="probe is invalid"):
        runtime.enable_hardware_learning(tmp_path / "controller.json")
    monkeypatch.setattr(hw, "validate_probe", lambda *_args, **_kwargs: (True, []))
    with pytest.raises(ModelIdentityError, match="fingerprint differs"):
        runtime.enable_hardware_learning(tmp_path / "controller.json")


def test_load_characterizes_hardware_before_model_load_and_reuses_enable(monkeypatch, tmp_path):
    import importlib
    from ironmule import hw
    tune = importlib.import_module("ironmule.tune")
    events = []
    identity = SimpleNamespace(model_id="scripted/model")
    resolved = SimpleNamespace(identity=identity)
    monkeypatch.setattr(hw, "probe", lambda **kwargs: events.append(("probe", kwargs)))
    monkeypatch.setattr(tune, "resolve_local_model", lambda *_: resolved)
    engine = SimpleNamespace(close=lambda: events.append(("close", {})))
    def load(*_args, **kwargs):
        events.append(("model_load", kwargs))
        return engine, object()
    monkeypatch.setattr(tune, "load_engine", load)
    class ScriptedRuntime(Runtime):
        def __init__(self, engine, _tokenizer, **_kwargs):
            self.engine = engine
        def enable_hardware_learning(self, checkpoint, **kwargs):
            events.append(("enable", {"checkpoint": checkpoint, **kwargs}))
    checkpoint = tmp_path / "controller.json"
    item = ScriptedRuntime.load("scripted/model", use_tuned_profile=False,
                                learning_checkpoint=checkpoint, probe_cache_dir=tmp_path / "probe")
    assert item.engine is engine
    assert [name for name, _ in events] == ["probe", "model_load", "enable"]
    assert events[0][1]["cache_dir"] == tmp_path / "probe"
    assert events[2][1]["checkpoint"] == checkpoint
    events.clear()
    ScriptedRuntime.load("scripted/model", use_tuned_profile=False)
    assert [name for name, _ in events] == ["model_load"]


@pytest.mark.parametrize("mode,automatic", [(ThroughputMode(), False), (None, True)])
def test_learning_load_rejects_conflicting_modes_before_probe(monkeypatch, tmp_path, mode, automatic):
    from ironmule import hw
    monkeypatch.setattr(hw, "probe", lambda **_: pytest.fail("conflict must precede probe"))
    with pytest.raises(ValueError, match="interactive reference"):
        Runtime.load("scripted/model", mode=mode, automatic_service_mode=automatic,
                     learning_checkpoint=tmp_path / "controller.json")


def test_learning_load_closes_loaded_model_after_setup_failure(monkeypatch, tmp_path):
    import importlib
    from ironmule import hw
    tune = importlib.import_module("ironmule.tune")
    closed = []
    monkeypatch.setattr(hw, "probe", lambda **_: {})
    monkeypatch.setattr(tune, "resolve_local_model", lambda *_: SimpleNamespace(
        identity=SimpleNamespace(model_id="scripted/model")))
    engine = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(tune, "load_engine", lambda *_args, **_kwargs: (engine, object()))
    class ScriptedRuntime(Runtime):
        def __init__(self, engine, _tokenizer, **_kwargs):
            self.engine = engine
        def enable_hardware_learning(self, *_args, **_kwargs):
            raise RuntimeError("scripted setup failure")
    with pytest.raises(RuntimeError, match="setup failure"):
        ScriptedRuntime.load("scripted/model", use_tuned_profile=False,
                             learning_checkpoint=tmp_path / "controller.json")
    assert closed == [True]


@pytest.fixture
def scripted_device_identity(monkeypatch):
    """Exercise production identity construction with metadata-only doubles."""
    import importlib
    import mlx.core as mx
    engine_runtime = importlib.import_module("ironmule.runtime")
    fingerprint = {
        "hardware_fingerprint": "scripted-exact-hardware", "chip": "scripted CUDA device",
        "memory_bytes": 16 * 1024**3, "os": "scripted Linux", "mlx": "scripted-mlx",
        "mlx_lm": "scripted-mlx-lm", "model_identity_sha256": "a" * 64,
        "model_revision": "scripted-revision", "model_manifest_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
    }
    info = {"device_name": "scripted CUDA device", "architecture": "scripted Turing",
            "total_memory": 16 * 1024**3, "free_memory": 12 * 1024**3,
            "compute_capability_major": 7, "compute_capability_minor": 5,
            "uuid": "GPU-scripted-one", "pci_bus_id": "0000:01:00.0"}
    device = ["Device(gpu, 0)"]
    item = Runtime.__new__(Runtime)
    item.engine = SimpleNamespace(knobs=Knobs(), model=object(), compute_dtype=None)
    item.backend = service.MLXBackend(item.engine, (2,))
    item.model_identity = SimpleNamespace(identity_sha256="a" * 64)
    item.fingerprint = lambda _: copy.deepcopy(fingerprint)
    monkeypatch.setattr(engine_runtime, "_new_cache", lambda _: object())
    monkeypatch.setattr(engine_runtime, "_cache_kinds", lambda _: ["kv"])
    monkeypatch.setattr(mx, "device_info", lambda: copy.deepcopy(info))
    monkeypatch.setattr(mx, "default_device", lambda: device[0])
    return item, info, device, fingerprint


def test_runtime_identity_ignores_cuda_free_memory_but_uses_shared_projection(scripted_device_identity):
    from ironmule.hw import stable_device_info
    item, info, _, _ = scripted_device_identity
    first = service.runtime_identity(item)
    info["free_memory"] -= 1024**3
    info["utilization"] = 90
    second = service.runtime_identity(item)
    assert first == second
    assert second["backend_device"] == stable_device_info(info)
    assert "free_memory" not in second["backend_device"]
    assert "utilization" not in second["backend_device"]
    assert "hw.py" in second["source_files"]


@pytest.mark.parametrize("field,replacement", [
    ("uuid", "GPU-scripted-two"), ("pci_bus_id", "0000:02:00.0"),
    ("device_name", "another scripted device"), ("architecture", "another architecture"),
    ("total_memory", 32 * 1024**3), ("compute_capability_major", 8),
])
def test_runtime_identity_invalidates_changed_static_device_metadata(scripted_device_identity, field, replacement):
    item, info, _, _ = scripted_device_identity
    first = service.runtime_identity(item)
    info[field] = replacement
    assert service.runtime_identity(item)["identity_sha256"] != first["identity_sha256"]


def test_runtime_identity_invalidates_execution_device_and_model_changes(scripted_device_identity):
    item, _, device, fingerprint = scripted_device_identity
    first = service.runtime_identity(item)
    device[0] = "Device(cpu, 0)"
    assert service.runtime_identity(item)["identity_sha256"] != first["identity_sha256"]
    device[0] = "Device(gpu, 0)"
    fingerprint["model_identity_sha256"] = "d" * 64
    assert service.runtime_identity(item)["identity_sha256"] != first["identity_sha256"]
    item.backend = object()
    with pytest.raises(ModelIdentityError, match="identified MLX backend"):
        service.runtime_identity(item)


def test_runtime_identity_still_binds_execution_source_code(scripted_device_identity, monkeypatch):
    from pathlib import Path
    item, _, _, _ = scripted_device_identity
    first = service.runtime_identity(item)
    original = Path.read_bytes
    def changed(path):
        value = original(path)
        return value + b"\n# explicit scripted source change\n" if path.name == "executor.py" else value
    monkeypatch.setattr(Path, "read_bytes", changed)
    changed_identity = service.runtime_identity(item)
    assert changed_identity["source_files"]["executor.py"] != first["source_files"]["executor.py"]
    assert changed_identity["identity_sha256"] != first["identity_sha256"]


def test_offset_signature_ignores_cache_bits_but_keeps_tokens_and_offsets(runtime):
    from friday_evidence.canonical import canonical_sha256
    controller = attach_profiles(runtime, (0, 2), return_arm=0)
    controller.config = SimpleNamespace(state_signature="offset")
    controller.last_decision = {"comparison": True}
    runtime.backend.kv_hash = lambda state, offset: canonical_sha256({
        "state": state, "offset": offset,
        "scripted_state_variant": runtime.engine.knobs.compiled_fixed_cache})
    runtime.serve(requests())
    a, b = controller.outcomes
    assert a.signature == b.signature, "differing cache bits alone no longer separate the arms"
    for request, (rid, offset) in zip(requests(), a.signature[1]):
        assert rid == request.rid and offset > len(request.prompt_ids) - 1
