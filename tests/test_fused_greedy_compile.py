from __future__ import annotations

import ast
import base64
import copy
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "fused_greedy_compile"
WORKER_PATH = EXP / "worker.py"
HARNESS_PATH = EXP / "measure_fused_greedy_compile.py"
PREREGISTRATION = EXP / "PREREGISTRATION.md"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = _load("fused_greedy_worker_tests", WORKER_PATH)
harness = _load("fused_greedy_harness_tests", HARNESS_PATH)
_REAL_REQUIRE_TARGET = harness._require_target


def _canonical(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


@pytest.fixture(autouse=True)
def _isolate_all_parent_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """No test may observe or change the real Cycle-18 evidence paths."""
    attempt_dir = tmp_path / "private-attempt"
    monkeypatch.setattr(harness, "ATTEMPT_DIR", attempt_dir)
    monkeypatch.setattr(harness, "ATTEMPT_PATH", attempt_dir / "attempt.json")
    monkeypatch.setattr(harness, "RESULT_PATH", tmp_path / "results.json")


@pytest.fixture
def synthetic_bindings(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    prompt_ids = [10_000 + index for index in range(322)]
    rendered = b"synthetic rendered planner prompt"
    prompt_token_sha = _sha_bytes(_canonical(prompt_ids))
    rendered_sha = _sha_bytes(rendered)
    monkeypatch.setattr(harness, "EXPECTED_PROMPT_TOKEN_SHA256", prompt_token_sha)
    monkeypatch.setattr(harness, "EXPECTED_RENDERED_PROMPT_SHA256", rendered_sha)
    return {"prompt_ids": prompt_ids, "prompt_token_sha256": prompt_token_sha,
            "rendered": rendered, "rendered_sha256": rendered_sha}


def _arm(name: str, *, tokens: list[int] | None = None, text: str = "same visible text") -> dict[str, Any]:
    physical = list(tokens or [501, 502, 1])
    normalized = worker.normalize_tokens(physical)
    forwards = len(physical) - 1
    return {
        "arm": name, "fixed_cache": True, "fixed_compile": True,
        "fused_selection": name == harness.ARM_NAMES[1], "cache_capacity": 512,
        "warmup_forwards": 8, "decode_forwards": forwards, "physical_forwards": forwards,
        "finish_reason": normalized["finish_reason"], "cache_discarded": True,
        "physical_tokens": physical, "logical_tokens": normalized["logical_tokens"],
        "visible_tokens": normalized["visible_tokens"],
        "physical_token_count": normalized["physical_token_count"],
        "logical_token_count": normalized["logical_token_count"],
        "visible_token_count": normalized["visible_token_count"],
        "overproduced_tokens": normalized["overproduced_tokens"],
        "eos_found": normalized["eos_found"], "eos_position": normalized["eos_position"],
        "eos_token_id": normalized["eos_token_id"],
        "physical_token_sha256": _sha_bytes(_canonical(physical)),
        "logical_token_sha256": _sha_bytes(_canonical(normalized["logical_tokens"])),
        "visible_token_sha256": _sha_bytes(_canonical(normalized["visible_tokens"])),
        "visible_text": text, "text_sha256": _sha_bytes(text.encode()),
        "prompt_sha256": harness.EXPECTED_PROMPT_SHA256,
        "prompt_token_sha256": harness.EXPECTED_PROMPT_TOKEN_SHA256,
        "rendered_prompt_sha256": harness.EXPECTED_RENDERED_PROMPT_SHA256,
        "ttft_ns": 300_000_000, "prefill_ns": 250_000_000,
        "measurement_prefill_ns": 250_000_000, "cache_conversion_ns": 10_000_000,
        "model_work_ns": 30_000_000, "decode_critical_path_ns": 40_000_000,
        "host_readback_ns": 3_000_000, "host_boundary_count": len(physical),
        "host_transfer_api_call_count": len(physical), "intertoken_ns": [15_000_000] * forwards,
        "token_rate": 75.0, "compile_wrapper_ns": 1_000, "compile_cold_ns": 5_000_000,
        "first_token_boundary_ns": 1_000_000, "warmup_boundary_ns": [2_000_000] * 8,
        "warmup_prefill_ns": 200_000_000, "warmup_conversion_ns": 10_000_000,
        "warmup_preparation_total_ns": 210_000_000,
        "measurement_preparation_total_ns": 260_000_000,
        "timing_scopes": {"ttft": "prefill-through-first-token", "primary": "full-decode-critical-path",
                          "model_work": "whole-arm-before-budget-charge"},
    }


def _budget_summary() -> dict[str, Any]:
    return {"candidate_cooldown_seconds": 0.0, "continuous_gpu_limit_seconds": 6.0,
            "cooldown_seconds": 104.0, "duty_cycle_limit": 0.15,
            "gpu_work_limit_seconds": 120.0, "gpu_work_seconds": 2.0,
            "max_continuous_gpu_seconds": 1.0, "required_break_limit_seconds": 4.0,
            "required_break_seconds": 104.0, "wall_limit_seconds": 1200.0, "wall_seconds": 110.0}


def _arm_budget() -> dict[str, Any]:
    return {"observed_model_work_ns": 1_000_000_000, "charged_model_work_ns": 1_000_000_000,
            "guard_recorded_model_work_ns": 1_000_000_000, "charge_accepted": True,
            "required_break_blocks": 13, "required_break_seconds": 52.0}


def _resources() -> dict[str, int]:
    return {"rss_peak_bytes": 2 * 1024**3, "mlx_peak_bytes": 2 * 1024**3,
            "swap_before_bytes": 0, "swap_after_bytes": 0, "swap_delta_bytes": 0,
            "swap_available": True}


def _snapshot_files() -> dict[str, str]:
    return {"config.json": "1" * 64, "tokenizer_config.json": "2" * 64,
            "tokenizer.json": "3" * 64, "model.safetensors": harness.EXPECTED_WEIGHT_SHA256}


def _execution_files() -> dict[str, str]:
    return {**_snapshot_files(), "generation_config.json": "4" * 64}


def _stat_manifest() -> dict[str, dict[str, Any]]:
    names = [*_snapshot_files(), "generation_config.json"]
    return {name: {"dev": 1, "inode": position + 10, "mtime_ns": 123,
                   "path": f"snapshot/{name}", "size": 10}
            for position, name in enumerate(names)}


def _expected_identity() -> dict[str, Any]:
    return {"snapshot_path": "snapshot/revision",
            "weight_sha256": {"model.safetensors": harness.EXPECTED_WEIGHT_SHA256},
            "snapshot_files_sha256": _snapshot_files(),
            "execution_files_sha256": _execution_files(),
            "execution_stat_manifest": _stat_manifest()}


def _complete_event(bindings: dict[str, Any], *, index: int = 1) -> dict[str, Any]:
    order = harness.PAIR_ORDERS[index - 1]
    arms = {name: _arm(name) for name in harness.ARM_NAMES}
    code = harness.code_fingerprints()
    snapshot_files = _snapshot_files()
    stat_manifest = _stat_manifest()
    marker_token = "synthetic-marker-token"
    return {
        "event": "complete", "status": "complete", "terminal_status": "complete",
        "partial_result": False, "study_id": harness.STUDY_ID, "run_id": harness.RUN_ID,
        "candidate_id": harness.CANDIDATE_ID, "formal_claim": False,
        "protocol_version": harness.PROTOCOL_VERSION, "process_index": index, "block": index,
        "arm_order": list(order), "arms": arms,
        "arm_budget": {name: _arm_budget() for name in harness.ARM_NAMES},
        "arm_resources": {name: _resources() for name in harness.ARM_NAMES},
        "correctness": {"pass": True, "physical_token_identity": True,
                        "logical_token_identity": True, "visible_token_identity": True,
                        "token_identity": True, "text_identity": True},
        "error": None, "pid": 4321 + index, "load_count": 1, "model_key": harness.MODEL_KEY,
        "model_id": harness.MODEL_ID, "snapshot_revision": harness.MODEL_REVISION,
        "snapshot_path": "snapshot/revision", "snapshot_sha256": harness.EXPECTED_SNAPSHOT_SHA256,
        "weight_sha256": {"model.safetensors": harness.EXPECTED_WEIGHT_SHA256},
        "snapshot_integrity": {"before_load_stat_manifest": stat_manifest,
                               "after_load_stat_manifest": copy.deepcopy(stat_manifest),
                               "post_arm_stat_manifest": copy.deepcopy(stat_manifest),
                               "snapshot_files_sha256": snapshot_files,
                               "execution_files_sha256": _execution_files()},
        "execution_files_sha256": _execution_files(),
        "model_load_ns": 100_000_000, "cache_capacity": 512, "max_physical_tokens": 32,
        "warmup_forwards_per_arm": 8, "prompt_sha256": harness.EXPECTED_PROMPT_SHA256,
        "prompt_token_ids": bindings["prompt_ids"],
        "prompt_token_sha256": bindings["prompt_token_sha256"],
        "rendered_prompt_b64": base64.b64encode(bindings["rendered"]).decode(),
        "rendered_prompt_sha256": bindings["rendered_sha256"], "prompt_tokens": 322,
        "eos_token_ids": [1, 106], "sampler_temperature": 0.0, "greedy": True,
        "device": "Device(gpu, 0)", "power_source": "AC Power",
        "model_work_ns": 2_000_000_000, "observed_model_work_ns": 2_000_000_000,
        "charged_model_work_ns": 2_000_000_000, "guard_recorded_model_work_ns": 2_000_000_000,
        "budget": _budget_summary(), **_resources(), "worker_watchdog_seconds": 6.0,
        "determinism": {"greedy": True, "within_arm_checked_by_parent": True},
        "git_revision": "a" * 40, "dirty_state": "clean",
        "marker_token_sha256": _sha_bytes(marker_token.encode()),
        "preregistration_sha256": harness.FROZEN_PREREGISTRATION_SHA256,
        "code_fingerprints": code, "code_sha256": _sha_bytes(_canonical(code)),
        "environment_sha256": harness.environment_fingerprint(), "process_wall_ns": 110_000_000_000,
    }


def _terminal_event(bindings: dict[str, Any], status: str, *, index: int = 1) -> dict[str, Any]:
    event = _complete_event(bindings, index=index)
    event["status"] = event["terminal_status"] = status
    event["event"] = "error" if status == "error" else "terminal"
    event["partial_result"] = True
    event["error"] = {"type": "SyntheticFailure", "message": status}
    if status == "candidate_not_runnable":
        failed = harness.ARM_NAMES[1]
        event["arms"].pop(failed); event["arm_resources"].pop(failed)
        event["arm_budget"][failed] = {**_arm_budget(), "observed_model_work_ns": 0,
                                       "charged_model_work_ns": 0, "guard_recorded_model_work_ns": 0,
                                       "charge_accepted": False, "required_break_blocks": 0,
                                       "required_break_seconds": 0.0}
        event["correctness"] = {"pass": False}
    elif status == "correctness_failed":
        event["arms"][harness.ARM_NAMES[1]] = _arm(harness.ARM_NAMES[1], tokens=[501, 503, 1], text="different")
        event["correctness"] = {"pass": False, "physical_token_identity": False,
                                "logical_token_identity": False, "visible_token_identity": False,
                                "token_identity": False, "text_identity": False}
    elif status == "resource_or_budget_failed":
        event["rss_peak_bytes"] = 6 * 1024**3
        failed = event["arm_order"][-1]
        event["arm_resources"][failed]["rss_peak_bytes"] = 6 * 1024**3
        event["arm_budget"][failed]["charge_accepted"] = False
        event["arm_budget"][failed]["charged_model_work_ns"] = 0
    elif status == "error":
        event["arms"] = {}; event["arm_resources"] = {}; event["arm_budget"] = {}
        event["load_count"] = 0; event["correctness"] = {"pass": False}
    return event


def _assert_rejected(event: dict[str, Any], *, index: int = 1) -> None:
    with pytest.raises((harness.WorkerError, ValueError, TypeError, KeyError)):
        harness._validate_event(event, index, harness.PAIR_ORDERS[index - 1],
                                identity=_expected_identity(), git_revision="a" * 40,
                                marker_token="synthetic-marker-token", expected_pid=4321 + index)


def _validate(event: dict[str, Any], *, index: int = 1) -> dict[str, Any]:
    return harness._validate_event(event, index, harness.PAIR_ORDERS[index - 1],
                                   identity=_expected_identity(), git_revision="a" * 40,
                                   marker_token="synthetic-marker-token", expected_pid=4321 + index)


def test_modules_import_without_loading_mlx_or_mlx_lm():
    for path in (WORKER_PATH, HARNESS_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import): names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom): names.add(node.module or "")
        assert not any(name == "mlx" or name.startswith("mlx.") or name.startswith("mlx_lm") for name in names)


def test_default_worker_and_parent_do_not_enter_execute(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(worker, "_run_worker", lambda _key: pytest.fail("worker execution reached"))
    monkeypatch.setattr(harness, "execute", lambda: pytest.fail("harness execution reached"))
    assert worker.main([]) == 78
    assert harness.main([]) == 78
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(item.get("formal_claim") is False for item in lines)


def test_direct_unauthorised_worker_fails_before_any_model_import(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    imported: list[str] = []
    original_import = __import__
    def guarded_import(name: str, *args: Any, **kwargs: Any):
        if name == "mlx" or name.startswith("mlx.") or name.startswith("mlx_lm"):
            imported.append(name); raise AssertionError("MLX/model import reached before authorisation")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", guarded_import)
    monkeypatch.setattr(worker, "PROJECT_ROOT", tmp_path)
    for key in list(os.environ):
        if key.startswith(worker.AUTH_PREFIX): monkeypatch.delenv(key, raising=False)
    with pytest.raises(worker.WorkerError, match="authorisation failed"):
        worker._run_worker(worker.MODEL_KEY)
    assert imported == []


@pytest.mark.parametrize("parser", [worker.parse_one_json, harness._strict_json])
def test_strict_json_accepts_one_object_and_rejects_ambiguous_frames(parser: Any):
    assert parser(b'{"ok":true}') == {"ok": True}
    assert parser(b'{"ok":true}\n') == {"ok": True}
    invalid = (b'', b'[]', b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}',
               b'{"x":1}\n{"y":2}', b'{"x":1}\r\n', b'{"x":1}\r', b'{"x":1} trailing',
               b'\xff', b'{"x":"' + b'x' * 1_000_001 + b'"}')
    for payload in invalid:
        with pytest.raises((ValueError, TypeError, harness.WorkerError, UnicodeError, json.JSONDecodeError)):
            parser(payload)


def test_fused_compiled_wrapper_changes_only_location_of_exact_argmax(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[Any, int]] = []; fixed_calls: list[tuple[Any, Any]] = []
    selected_logits = object(); next_state = {"position": {"offset": 323}, "layers": []}
    token = SimpleNamespace(reshape=lambda shape: ("selected-token", shape))
    class FakeMX:
        @staticmethod
        def compile(function: Any, *, shapeless: bool): assert shapeless is False; return function
        @staticmethod
        def argmax(value: Any, *, axis: int): calls.append((value, axis)); return token
    class FakeLogits:
        def __getitem__(self, index: Any):
            assert index == (slice(None), -1, slice(None)); return selected_logits
    fake_logits = FakeLogits()
    def fixed_forward(model: Any, input_ids: Any, state: Any, mx: Any):
        assert model == "same-model" and mx is FakeMX; fixed_calls.append((input_ids, state)); return fake_logits, next_state
    monkeypatch.setattr(worker, "_fixed_forward", fixed_forward)
    baseline = worker._make_compiled("same-model", FakeMX, False)
    candidate = worker._make_compiled("same-model", FakeMX, True)
    state = {"position": {"offset": 322}, "layers": []}
    assert baseline("same-input", state) == (fake_logits, next_state)
    assert candidate("same-input", state) == (("selected-token", (1,)), next_state)
    assert fixed_calls == [("same-input", state), ("same-input", state)]
    assert calls == [(selected_logits, -1)]
    source = ast.unparse(ast.parse(WORKER_PATH.read_text(encoding="utf-8")))
    assert "mx.argmax(logits[:, -1, :], axis=-1).reshape((1,))" in source
    assert "mx.compile(body, shapeless=False)" in source
    assert "mx.concatenate" not in source


def test_fixed_kv_state_offset_is_explicit_and_chains_without_outer_mutation():
    class Tensor:
        def __init__(self, shape: tuple[int, ...], label: str): self.shape = shape; self.label = label; self.dtype = "dtype"
    class Scalar:
        dtype = "int32"
        def __init__(self, value: int): self.value = value
        def __add__(self, other: int): return Scalar(self.value + other)
        def __eq__(self, other: object): return self.value == (other.value if isinstance(other, Scalar) else other)
        def __repr__(self): return str(self.value)
    class Input: shape = (1, 1)
    class FakeMX:
        @staticmethod
        def array(value: Any, dtype: Any = None): del dtype; return value
        @staticmethod
        def stack(values: Any): return tuple(values)
        @staticmethod
        def slice_update(target: Tensor, update: Tensor, *, start_indices: Any, axes: Any):
            assert axes == (0, 1, 2, 3); return Tensor(target.shape, f"{target.label}+{update.label}@{tuple(start_indices)}")
    seen_offsets: list[int] = []
    class Model:
        def __call__(self, _input: Any, *, cache: list[Any]):
            seen_offsets.append(cache[0].offset)
            cache[0].update_and_fetch(Tensor((1, 2, 1, 4), "new-k"), Tensor((1, 2, 1, 4), "new-v"))
            return "same-logits"
    original = {"position": {"offset": Scalar(322)},
                "layers": [{"keys": Tensor((1, 2, 512, 4), "k"), "values": Tensor((1, 2, 512, 4), "v")}]}
    logits, state_323 = worker._fixed_forward(Model(), Input(), original, FakeMX)
    assert logits == "same-logits" and original["position"]["offset"] == 322
    assert state_323["position"]["offset"] == 323
    _, state_324 = worker._fixed_forward(Model(), Input(), state_323, FakeMX)
    assert state_324["position"]["offset"] == 324 and seen_offsets == [322, 323]


def test_token_normalization_keeps_physical_tail_and_exact_visible_contract():
    normalized = worker.normalize_tokens([7, 8, 1, 99, 100])
    assert normalized["physical_tokens"] == [7, 8, 1, 99, 100]
    assert normalized["logical_tokens"] == [7, 8, 1] and normalized["visible_tokens"] == [7, 8]
    assert normalized["finish_reason"] == "stop" and normalized["overproduced_tokens"] == 2
    assert worker.normalize_tokens(list(range(40)))["physical_token_count"] == 32
    with pytest.raises(ValueError): worker.normalize_tokens([1, -1])
    with pytest.raises(ValueError): worker.normalize_tokens([True])


def test_complete_synthetic_event_is_strictly_accepted(synthetic_bindings: dict[str, Any]):
    event = _complete_event(synthetic_bindings)
    assert _validate(copy.deepcopy(event))["status"] == "complete"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda event: event.__setitem__("formal_claim", True),
        lambda event: event.__setitem__("protocol_version", 999),
        lambda event: event.__setitem__("process_index", 2),
        lambda event: event.__setitem__("arm_order", list(reversed(event["arm_order"]))),
        lambda event: event.__setitem__("load_count", 2),
        lambda event: event.__setitem__("dirty_state", " M changed.py"),
        lambda event: event.__setitem__("git_revision", "not-a-revision"),
        lambda event: event.__setitem__("marker_token_sha256", "0" * 64),
        lambda event: event.__setitem__("preregistration_sha256", "0" * 64),
        lambda event: event.__setitem__("environment_sha256", "0" * 64),
        lambda event: event.__setitem__("code_sha256", "0" * 64),
        lambda event: event.__setitem__("code_fingerprints", {}),
        lambda event: event.__setitem__("snapshot_revision", "wrong"),
        lambda event: event.__setitem__("snapshot_sha256", "0" * 64),
        lambda event: event.__setitem__("prompt_sha256", "0" * 64),
        lambda event: event.__setitem__("prompt_tokens", 321),
        lambda event: event.__setitem__("prompt_token_ids", event["prompt_token_ids"][:-1]),
        lambda event: event.__setitem__("prompt_token_sha256", "0" * 64),
        lambda event: event.__setitem__("rendered_prompt_b64", base64.b64encode(b"mutated").decode()),
        lambda event: event["snapshot_integrity"].__setitem__("post_arm_stat_manifest", {}),
    ],
)
def test_complete_event_rejects_identity_or_provenance_mutation(
    synthetic_bindings: dict[str, Any], mutation: Any
):
    event = _complete_event(synthetic_bindings)
    mutation(event)
    _assert_rejected(event)


@pytest.mark.parametrize(
    "field",
    ["physical_tokens", "logical_tokens", "visible_tokens", "physical_token_sha256",
     "logical_token_sha256", "visible_token_sha256", "visible_text", "text_sha256",
     "prompt_sha256", "prompt_token_sha256", "rendered_prompt_sha256"],
)
def test_arm_contract_rejects_token_text_or_prompt_mutation(
    synthetic_bindings: dict[str, Any], field: str
):
    event = _complete_event(synthetic_bindings)
    candidate = event["arms"][harness.ARM_NAMES[1]]
    candidate[field] = "mutated" if isinstance(candidate[field], str) else [999]
    _assert_rejected(event)


def test_cross_arm_identity_includes_physical_tail_not_only_logical_tokens(
    synthetic_bindings: dict[str, Any]
):
    event = _complete_event(synthetic_bindings)
    candidate = _arm(harness.ARM_NAMES[1], tokens=[501, 502, 1, 777])
    baseline = event["arms"][harness.ARM_NAMES[0]]
    assert candidate["logical_tokens"] == baseline["logical_tokens"]
    assert candidate["visible_tokens"] == baseline["visible_tokens"]
    event["arms"][harness.ARM_NAMES[1]] = candidate
    _assert_rejected(event)


@pytest.mark.parametrize(
    "status", ["candidate_not_runnable", "correctness_failed", "resource_or_budget_failed", "error"]
)
def test_all_well_formed_terminal_event_schemas_are_preserved(
    synthetic_bindings: dict[str, Any], status: str
):
    event = _terminal_event(synthetic_bindings, status)
    validated = _validate(copy.deepcopy(event))
    assert validated["status"] == status and validated["partial_result"] is True
    assert validated["formal_claim"] is False


@pytest.mark.parametrize(
    "missing", ["error", "protocol_version", "preregistration_sha256", "environment_sha256", "git_revision"]
)
def test_terminal_events_are_not_allowed_to_bypass_provenance_schema(
    synthetic_bindings: dict[str, Any], missing: str
):
    event = _terminal_event(synthetic_bindings, "candidate_not_runnable")
    event.pop(missing)
    _assert_rejected(event)


def test_complete_resource_limits_are_hard_gates_but_resource_terminal_is_preserved(
    synthetic_bindings: dict[str, Any]
):
    for field, bad in (("rss_peak_bytes", 5 * 1024**3 + 1),
                       ("mlx_peak_bytes", 5 * 1024**3 + 1), ("swap_delta_bytes", 1)):
        event = _complete_event(synthetic_bindings); event[field] = bad; _assert_rejected(event)
    terminal = _terminal_event(synthetic_bindings, "resource_or_budget_failed")
    assert _validate(terminal)["status"] == "resource_or_budget_failed"


def test_per_arm_resource_and_budget_failures_are_terminal_not_complete(synthetic_bindings: dict[str, Any]):
    for mutation in (
        lambda event: event["arm_resources"][harness.ARM_NAMES[0]].__setitem__("rss_peak_bytes", 5 * 1024**3 + 1),
        lambda event: event["arm_resources"][harness.ARM_NAMES[0]].__setitem__("mlx_peak_bytes", 5 * 1024**3 + 1),
        lambda event: event["arm_resources"][harness.ARM_NAMES[0]].__setitem__("swap_delta_bytes", 1),
        lambda event: event["arm_budget"][harness.ARM_NAMES[0]].__setitem__("charge_accepted", False),
        lambda event: event["arm_budget"][harness.ARM_NAMES[0]].__setitem__("observed_model_work_ns", 6_000_000_001),
    ):
        event = _complete_event(synthetic_bindings); mutation(event); _assert_rejected(event)


def test_budget_rejection_is_terminal_and_timer_stops_before_guard_charge():
    source = WORKER_PATH.read_text(encoding="utf-8")
    run_worker = source[source.index("def _run_worker"):]
    stopped = run_worker.index("stopped = time.perf_counter_ns()")
    elapsed = run_worker.index("arm_ns = stopped - started")
    record = run_worker.index("guard.record_gpu")
    resource = run_worker.index("_resource_evidence", record)
    pause = run_worker.index("guard.required_break", resource)
    assert stopped < elapsed < record < resource < pause
    assert 'status = "resource_or_budget_failed"' in run_worker[record:pause]
    assert "continuous_gpu_limit_s=CONTINUOUS_GPU_LIMIT_SECONDS" in source
    assert "duty_cycle_limit=0.15" in source and "wall_limit_s=WALL_LIMIT_SECONDS" in source


def test_decision_precedence_matches_frozen_table():
    common = {"paired": {"median": 0.9, "lower": 0.8, "upper": 0.95}, "complete": True}
    decide = harness.decision_for
    assert decide(resource_pass=False, budget_pass=True, correctness_pass=False,
                  candidate_runnable=False, **common) == "resource_or_budget_failed"
    assert decide(resource_pass=True, budget_pass=True, correctness_pass=False,
                  candidate_runnable=False, **common) == "correctness_failed"
    assert decide(resource_pass=True, budget_pass=True, correctness_pass=True,
                  candidate_runnable=False, **common) == "candidate_not_runnable"
    assert decide(resource_pass=True, budget_pass=True, correctness_pass=True,
                  candidate_runnable=True, paired={}, complete=False) == "incomplete_evidence"
    assert decide(resource_pass=True, budget_pass=True, correctness_pass=True,
                  candidate_runnable=True, **common) == "fused_greedy_compile_wins_exact_scope"
    assert decide(resource_pass=True, budget_pass=True, correctness_pass=True, candidate_runnable=True,
                  paired={"median": 1.01, "lower": 1.001, "upper": 1.02}, complete=True) == "fused_greedy_compile_regression_baseline_retained"
    assert decide(resource_pass=True, budget_pass=True, correctness_pass=True, candidate_runnable=True,
                  paired={"median": 1.01, "lower": 0.99, "upper": 1.03}, complete=True) == "fused_greedy_compile_inconclusive"


def test_bootstrap_is_frozen_deterministic_10k_and_keeps_every_outlier():
    assert worker.BOOTSTRAP_SEED == harness.BOOTSTRAP_SEED == 20260825
    assert harness.BOOTSTRAP_RESAMPLES == 10_000
    prereg = PREREGISTRATION.read_text(encoding="utf-8")
    assert "`20260825`" in prereg
    baseline = [1.0] * 6; candidate = [0.1, 1.0, 1.0, 1.0, 1.0, 10.0]
    first = harness.paired_bootstrap(baseline, candidate); second = harness.paired_bootstrap(baseline, candidate)
    assert first == second and first["seed"] == 20260825 and first["iterations"] == 10_000
    assert first["ratios"] == candidate


def test_preregistration_hash_and_protocol_are_bound_consistently():
    digest = _sha_file(PREREGISTRATION)
    assert digest == worker.FROZEN_PREREGISTRATION_SHA256 == harness.FROZEN_PREREGISTRATION_SHA256
    assert worker.PROMPT_SHA256 == worker.EXPECTED_PROMPT_SHA256 == harness.EXPECTED_PROMPT_SHA256
    contract = worker.protocol_contract()
    assert contract["version"] == harness.PROTOCOL_VERSION
    assert contract["study_id"] == harness.STUDY_ID and contract["run_id"] == harness.RUN_ID
    assert tuple(contract["arms"]) == harness.ARM_NAMES and contract["capacity"] == 512
    assert contract["max_physical_tokens"] == 32 and contract["warmups"] == 8
    assert contract["nonce"] == harness.AUTH_NONCE and contract["prompt_sha256"] == harness.EXPECTED_PROMPT_SHA256
    assert worker.PROTOCOL_SHA256 == _sha_bytes(_canonical(contract))


def test_snapshot_identity_hashes_execution_files_and_generation_config(tmp_path: Path):
    repository = tmp_path / "models--local"
    snapshot_root = repository / "snapshots" / "revision"
    snapshot_root.mkdir(parents=True)
    for name, payload in {"config.json": b"config", "tokenizer_config.json": b"tokenizer-config",
                          "tokenizer.json": b"tokenizer", "generation_config.json": b'{"eos_token_id":[1,106]}',
                          "model.safetensors": b"weights"}.items():
        (snapshot_root / name).write_bytes(payload)
    snapshot = SimpleNamespace(path=str(snapshot_root), revision=harness.MODEL_REVISION,
                               weight_files=("model.safetensors",), weight_bytes=7)
    identity = harness._snapshot_identity(snapshot)
    assert "generation_config.json" not in identity["snapshot_files_sha256"]
    assert "generation_config.json" in identity["execution_files_sha256"]
    original = identity["execution_files_sha256"]["generation_config.json"]
    (snapshot_root / "generation_config.json").write_bytes(b'{"eos_token_id":[106,1]}')
    assert harness._snapshot_identity(snapshot)["execution_files_sha256"]["generation_config.json"] != original


def _configure_preflight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[SimpleNamespace, dict[str, Any]]:
    prereg = tmp_path / "PREREGISTRATION.md"; prereg.write_text("sealed", encoding="utf-8")
    monkeypatch.setattr(harness, "PREREGISTRATION", prereg)
    monkeypatch.setattr(harness, "FROZEN_PREREGISTRATION_SHA256", _sha_file(prereg))
    fake_worker = SimpleNamespace(PROMPT_SHA256=harness.EXPECTED_PROMPT_SHA256, PROTOCOL_SHA256="c" * 64,
        protocol_contract=lambda: {"study_id": harness.STUDY_ID, "run_id": harness.RUN_ID,
                                   "arms": list(harness.ARM_NAMES), "capacity": 512, "warmups": 8})
    monkeypatch.setattr(harness, "_module", lambda: fake_worker)
    monkeypatch.setattr(harness, "_clean_worktree", lambda: ("a" * 40, ""))
    monkeypatch.setattr(harness, "_require_target", lambda: None)
    monkeypatch.setattr(harness, "_swap_used_bytes", lambda: 0)
    snapshot = SimpleNamespace(revision=harness.MODEL_REVISION)
    identity = {"model_id": harness.MODEL_ID, "model_revision": harness.MODEL_REVISION,
                "snapshot_path": "/local/snapshot",
                "snapshot_files_sha256": {"model.safetensors": harness.EXPECTED_WEIGHT_SHA256},
                "execution_files_sha256": {"model.safetensors": harness.EXPECTED_WEIGHT_SHA256,
                                             "generation_config.json": "4" * 64},
                "snapshot_sha256": harness.EXPECTED_SNAPSHOT_SHA256,
                "weight_sha256": {"model.safetensors": harness.EXPECTED_WEIGHT_SHA256},
                "execution_stat_manifest": {"model.safetensors": {"size": 1}}}
    monkeypatch.setattr(harness, "_snapshot_identity", lambda _snapshot: copy.deepcopy(identity))
    bench = types.ModuleType("_bench"); bench.require_ac_power = lambda: "AC Power"
    bench.resolve_local_model_snapshot = lambda _model: snapshot
    monkeypatch.setitem(sys.modules, "_bench", bench)
    return snapshot, identity


def test_preflight_enforces_clean_snapshot_hash_revision_power_and_swap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    snapshot, identity = _configure_preflight(monkeypatch, tmp_path)
    assert harness._preflight() == ("a" * 40, "", "AC Power", identity, 0)
    snapshot.revision = "wrong"
    with pytest.raises(harness.StudyError, match="revision"): harness._preflight()
    snapshot.revision = harness.MODEL_REVISION
    monkeypatch.setattr(harness, "_snapshot_identity", lambda _snapshot: {**identity, "snapshot_sha256": "0" * 64})
    with pytest.raises(harness.StudyError, match="snapshot|hash"): harness._preflight()


def test_clean_dirty_hardware_package_power_and_swap_gates_are_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _configure_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(harness, "_clean_worktree", lambda: (_ for _ in ()).throw(harness.StudyError("dirty")))
    with pytest.raises(harness.StudyError, match="dirty"): harness._preflight()
    _configure_preflight(monkeypatch, tmp_path); monkeypatch.setattr(harness, "_swap_used_bytes", lambda: None)
    with pytest.raises(harness.StudyError, match="swap"): harness._preflight()
    _configure_preflight(monkeypatch, tmp_path)
    sys.modules["_bench"].require_ac_power = lambda: (_ for _ in ()).throw(RuntimeError("battery"))
    with pytest.raises(RuntimeError, match="battery"): harness._preflight()
    monkeypatch.setattr(harness.platform, "machine", lambda: "x86_64")
    with pytest.raises(harness.StudyError, match="target"): _REAL_REQUIRE_TARGET()


def test_hardware_device_and_package_gates_use_exact_registered_values(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(harness.platform, "machine", lambda: harness.EXPECTED_MACHINE)
    values = {"machdep.cpu.brand_string": harness.EXPECTED_CPU_BRAND,
              "hw.memsize": str(harness.EXPECTED_MEMORY_BYTES)}
    monkeypatch.setattr(harness, "_sysctl", values.get)
    monkeypatch.setattr(harness.importlib.metadata, "version", lambda package: harness.REQUIRED_PACKAGES[package])
    fake_core = types.ModuleType("mlx.core"); fake_core.default_device = lambda: "Device(gpu, 0)"
    fake_mlx = types.ModuleType("mlx"); fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx); monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    harness._require_target()
    fake_core.default_device = lambda: "Device(cpu, 0)"
    with pytest.raises(harness.StudyError, match="device"): harness._require_target()
    fake_core.default_device = lambda: "Device(gpu, 0)"
    monkeypatch.setattr(harness.importlib.metadata, "version", lambda _package: "wrong")
    with pytest.raises(harness.StudyError, match="package"): harness._require_target()


def test_pipe_caps_and_process_group_timeout_are_enforced(monkeypatch: pytest.MonkeyPatch):
    capture: dict[str, Any] = {}; capped_kills: list[int] = []
    cap_process = SimpleNamespace(pid=111, kill=lambda: None)
    monkeypatch.setattr(harness, "_kill_process_group", lambda child: capped_kills.append(child.pid))
    harness._read_pipe(io.BytesIO(b"x" * 20), 10, capture, "stdout", cap_process)
    assert capture["overflow_stdout"] is True and len(capture["stdout"]) == 11
    assert capped_kills == [111]
    class TimedOutProcess:
        pid = 999; returncode = None; stdout = io.BytesIO(b""); stderr = io.BytesIO(b"")
        def __init__(self): self.waits = 0
        def wait(self, timeout: float):
            assert timeout > 0; self.waits += 1
            if self.waits == 1: raise subprocess.TimeoutExpired("worker", timeout)
            self.returncode = -9; return self.returncode
        def kill(self): self.returncode = -9
    process = TimedOutProcess(); killed: list[int] = []
    monkeypatch.setattr(harness.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(harness, "_kill_process_group", lambda child: killed.append(child.pid))
    monkeypatch.setattr(harness, "_module", lambda: SimpleNamespace(PROTOCOL_SHA256="c" * 64))
    identity = {"snapshot_path": "/snapshot", "snapshot_sha256": harness.EXPECTED_SNAPSHOT_SHA256,
                "weight_sha256": {"model.safetensors": harness.EXPECTED_WEIGHT_SHA256},
                "snapshot_files_sha256": {"model.safetensors": harness.EXPECTED_WEIGHT_SHA256},
                "execution_files_sha256": {"model.safetensors": harness.EXPECTED_WEIGHT_SHA256,
                                             "generation_config.json": "4" * 64},
                "execution_stat_manifest": {}}
    with pytest.raises(harness.WorkerError, match="timeout"):
        harness._run_child(1, harness.PAIR_ORDERS[0], harness.time.monotonic() + 1,
                           identity=identity, git_revision="a" * 40, marker_token="token")
    assert killed == [999]
    source = HARNESS_PATH.read_text(encoding="utf-8")
    assert "start_new_session=True" in source and "MAX_STDOUT_BYTES" in source and "MAX_STDERR_BYTES" in source


def test_private_marker_modes_exclusive_creation_and_symlink_rejection(tmp_path: Path):
    marker = tmp_path / "marker.json"; harness._write_exclusive(marker, {"formal_claim": False}, 0o600)
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError): harness._write_exclusive(marker, {}, 0o600)
    link = tmp_path / "result-link"; link.symlink_to(marker); harness.RESULT_PATH = link
    state = harness._evidence_state(); assert state["result"]["symlink"] is True
    with pytest.raises(harness.StudyError): harness._validate_evidence_state(state)


def test_existing_marker_or_result_blocks_before_preflight(monkeypatch: pytest.MonkeyPatch):
    entered: list[bool] = []; monkeypatch.setattr(harness, "_preflight", lambda: entered.append(True))
    harness.ATTEMPT_DIR.mkdir(mode=0o700); harness.ATTEMPT_PATH.write_text("{}", encoding="utf-8")
    os.chmod(harness.ATTEMPT_PATH, 0o600)
    with pytest.raises(harness.StudyError, match="existing"): harness.execute()
    assert entered == []


def _configure_fake_execute(
    monkeypatch: pytest.MonkeyPatch, child: Any, *, postflight_snapshot_mutates: bool = False
) -> list[dict[str, Any]]:
    identity = {"snapshot_path": "/snapshot", "snapshot_sha256": harness.EXPECTED_SNAPSHOT_SHA256,
                "weight_sha256": {"model.safetensors": harness.EXPECTED_WEIGHT_SHA256},
                "snapshot_files_sha256": {"model.safetensors": harness.EXPECTED_WEIGHT_SHA256},
                "execution_files_sha256": {"model.safetensors": harness.EXPECTED_WEIGHT_SHA256,
                                             "generation_config.json": "4" * 64},
                "execution_stat_manifest": {}}
    monkeypatch.setattr(harness, "_preflight", lambda: ("a" * 40, "", "AC Power", copy.deepcopy(identity), 0))
    monkeypatch.setattr(harness, "code_fingerprints", lambda: {"worker.py": "c" * 64})
    monkeypatch.setattr(harness, "_sha256_file", lambda path: harness.FROZEN_PREREGISTRATION_SHA256
                        if path == harness.PREREGISTRATION else "d" * 64)
    monkeypatch.setattr(harness, "_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(harness, "_git", lambda *args: "a" * 40 if args == ("rev-parse", "HEAD") else "")
    target = {"machine": harness.EXPECTED_MACHINE, "cpu_brand": harness.EXPECTED_CPU_BRAND,
              "memory_bytes": str(harness.EXPECTED_MEMORY_BYTES), "packages": harness.REQUIRED_PACKAGES,
              "python": str((ROOT / ".venv" / "bin" / "python").resolve()), "device": "Device(gpu, 0)"}
    monkeypatch.setattr(harness, "_target_info", lambda: copy.deepcopy(target))
    monkeypatch.setattr(harness, "_run_child", child)
    monkeypatch.setattr(harness, "_aggregate", lambda runs: {
        "arms": {}, "paired": {"median": 0.98, "lower": 0.97, "upper": 0.99}})
    post_identity = copy.deepcopy(identity)
    if postflight_snapshot_mutates: post_identity["snapshot_sha256"] = "0" * 64
    monkeypatch.setattr(harness, "_snapshot_identity", lambda snapshot: copy.deepcopy(post_identity))
    bench = types.ModuleType("_bench"); bench.require_ac_power = lambda: "AC Power"
    bench.resolve_local_model_snapshot = lambda _model: object()
    monkeypatch.setitem(sys.modules, "_bench", bench)
    checkpoints: list[dict[str, Any]] = []; real_atomic = harness._atomic_result
    def capture(value: dict[str, Any], *, replace: bool = False):
        checkpoints.append(copy.deepcopy(value)); real_atomic(value, replace=replace)
    monkeypatch.setattr(harness, "_atomic_result", capture)
    return checkpoints


def test_execute_checkpoints_initially_and_after_every_pair(
    monkeypatch: pytest.MonkeyPatch, synthetic_bindings: dict[str, Any]
):
    calls: list[int] = []
    def child(index: int, order: tuple[str, str], deadline: float, **kwargs: Any):
        assert order == harness.PAIR_ORDERS[index - 1] and deadline > harness.time.monotonic()
        calls.append(index); return _complete_event(synthetic_bindings, index=index)
    checkpoints = _configure_fake_execute(monkeypatch, child); report = harness.execute()
    assert calls == [1, 2, 3, 4, 5, 6]
    assert checkpoints[0]["status"] == "running" and checkpoints[0]["runs"] == []
    assert [item.get("checkpoint_pair") for item in checkpoints[1:7]] == [1, 2, 3, 4, 5, 6]
    assert checkpoints[-1]["status"] == report["status"]
    assert report["formal_claim"] is False and report["thresholds"]["no_outlier_removal"] is True


def test_terminal_child_stops_schedule_without_retry_and_preserves_partial(
    monkeypatch: pytest.MonkeyPatch, synthetic_bindings: dict[str, Any]
):
    calls: list[int] = []
    def child(index: int, _order: tuple[str, str], _deadline: float, **_kwargs: Any):
        calls.append(index); return _terminal_event(synthetic_bindings, "candidate_not_runnable", index=index)
    checkpoints = _configure_fake_execute(monkeypatch, child); report = harness.execute()
    assert calls == [1]
    assert report["decision"] == "candidate_not_runnable" and report["partial_result"] is True
    assert checkpoints[1]["checkpoint_pair"] == 1 and checkpoints[-1]["decision"] == "candidate_not_runnable"


def test_child_exception_stops_schedule_and_writes_fail_safe_final_result(monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []
    def child(index: int, *_args: Any, **_kwargs: Any):
        calls.append(index); raise harness.WorkerError("synthetic child failure")
    checkpoints = _configure_fake_execute(monkeypatch, child); report = harness.execute()
    assert calls == [1] and report["decision"] == "resource_or_budget_failed" and report["partial_result"] is True
    assert report["error"]["type"] == "WorkerError"
    assert checkpoints[0]["status"] == "running" and checkpoints[-1]["decision"] == "resource_or_budget_failed"


def test_postflight_snapshot_mutation_forces_resource_failure(
    monkeypatch: pytest.MonkeyPatch, synthetic_bindings: dict[str, Any]
):
    def child(index: int, _order: tuple[str, str], _deadline: float, **_kwargs: Any):
        return _complete_event(synthetic_bindings, index=index)
    _configure_fake_execute(monkeypatch, child, postflight_snapshot_mutates=True)
    report = harness.execute()
    assert report["decision"] == "resource_or_budget_failed" and report["gates"]["resource_pass"] is False
    assert "snapshot" in report["error"]["message"].lower()


def test_show_absent_and_existing_result_are_read_only(capsys: pytest.CaptureFixture[str]):
    assert harness._show() == 78
    unavailable = json.loads(capsys.readouterr().out)
    assert unavailable == {"formal_claim": False, "status": "unavailable", "study_id": harness.STUDY_ID}
    result = {"study_id": harness.STUDY_ID, "run_id": harness.RUN_ID,
              "decision": "fused_greedy_compile_inconclusive", "formal_claim": False, "runs": [{}, {}]}
    harness.RESULT_PATH.write_bytes(_canonical(result) + b"\n"); os.chmod(harness.RESULT_PATH, 0o644)
    before = (_sha_file(harness.RESULT_PATH), harness.RESULT_PATH.stat().st_mtime_ns)
    assert harness._show() == 0; shown = json.loads(capsys.readouterr().out)
    after = (_sha_file(harness.RESULT_PATH), harness.RESULT_PATH.stat().st_mtime_ns)
    assert shown["decision"] == result["decision"] and shown["runs_completed"] == 2
    assert shown["formal_claim"] is False and after == before


def test_evidence_state_hash_detects_mutation_without_touching_real_evidence():
    harness.ATTEMPT_DIR.mkdir(mode=0o700); harness.ATTEMPT_PATH.write_bytes(b'{"marker":1}\n')
    os.chmod(harness.ATTEMPT_PATH, 0o600); harness.RESULT_PATH.write_bytes(b'{"result":1}\n')
    os.chmod(harness.RESULT_PATH, 0o644); before = harness._evidence_state()
    harness.RESULT_PATH.write_bytes(b'{"result":2}\n'); after = harness._evidence_state()
    assert before["marker"]["sha256"] == after["marker"]["sha256"]
    assert before["result"]["sha256"] != after["result"]["sha256"]


def test_source_has_no_quantization_weight_model_or_matmul_change():
    prereg = PREREGISTRATION.read_text(encoding="utf-8"); source = WORKER_PATH.read_text(encoding="utf-8")
    assert "Matmul mathematics are unchanged" in prereg and "no Matmul-off arm" in prereg
    assert worker.MODEL_ID == harness.MODEL_ID == "mlx-community/gemma-3-4b-it-4bit"
    assert worker.MODEL_REVISION == harness.MODEL_REVISION and worker.ARM_NAMES == harness.ARM_NAMES
    assert source.count("load(snapshot_path)") == 1
    assert "quantize(" not in source and "dequantize(" not in source
    assert "formal_claim=false" in prereg and "no outlier removal" in prereg
