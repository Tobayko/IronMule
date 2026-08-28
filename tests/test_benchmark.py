"""CPU-only tests for the public benchmark protocol and correctness gate."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

_BENCHMARK_PATH = Path(__file__).parents[1] / "ironmule" / "benchmark.py"
_SPEC = importlib.util.spec_from_file_location("ironmule_benchmark_test_target", _BENCHMARK_PATH)
benchmark = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(benchmark)


class FakePlan:
    def __init__(self, name):
        self.name = name


class FakeRuntime:
    model_id = "fake-model"

    def __init__(self, mismatch=False):
        self.plans = []
        self.mismatch = mismatch
        self.mode = None
        self.telemetry = SimpleNamespace(snapshot=self.snapshot)

    def session_plan(self, _document, name):
        plan = FakePlan(name)
        self.plans.append(plan)
        return plan

    def encode(self, prompt):
        return list(range(len(prompt) % 7 + 1))

    def fingerprint(self, plan, workload):
        return {"plan": plan.name, "workload": workload}

    def serve(self, requests):
        token = 8 if self.mode.__class__.__name__ == "InteractiveMode" else 9
        if not self.mismatch:
            token = 8
        stop_reason = "eos" if self.mismatch and token == 9 else "length"
        return [SimpleNamespace(
            rid=request.rid, tokens=[token], stop_reason=stop_reason, text="answer",
            metrics={"generated_tokens": 1, "physical_tokens": 1,
                     "visible_generated_tokens": 1},
        ) for request in requests]

    def snapshot(self):
        # Deliberately disagree with the outer clock: this is executor diagnostic data.
        return {"wall_ms": 999.0}


class FakeIronMule:
    class Request:
        def __init__(self, prompt_ids, max_tokens, plan, rid):
            self.prompt_ids = prompt_ids
            self.max_tokens = max_tokens
            self.plan = plan
            self.rid = rid

    class InteractiveMode:
        pass

    class ThroughputMode:
        pass

    class StrictOneShotPlan:
        pass


def _clock():
    current = 0

    def tick():
        nonlocal current
        current += 1_000_000
        return current

    return tick


def test_output_diff_reports_tokens_stop_and_missing_results():
    expected = [SimpleNamespace(rid="q0", tokens=[1], stop_reason="eos", text="one")]
    actual = [SimpleNamespace(rid="q0", tokens=[2], stop_reason="length", text="two")]
    diff = benchmark.output_diff(expected, actual)
    assert diff[0]["kind"] == "field_mismatch"
    assert diff[0]["fields"]["tokens"] == {"expected": [1], "actual": [2]}
    assert diff[0]["fields"]["first_token_difference"] == {
        "position": 0, "expected": 1, "actual": 2,
    }
    assert "token_counts" in diff[0]["fields"]
    assert {entry["kind"] for entry in benchmark.output_diff(expected, [])} == {"missing_result"}


def test_protocol_is_balanced_has_fresh_arm_plans_and_raw_samples(monkeypatch):
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", _clock())
    runtime = FakeRuntime()
    result = benchmark.run_protocol(
        runtime, FakeIronMule, requests=2, max_tokens=3, plan_name="reusable",
        warmup=2, repeats=4,
    )
    assert len(runtime.plans) == 2
    assert runtime.plans[0] is not runtime.plans[1]
    assert result["protocol"]["primary_metric"] == "outer_wall_ms"
    assert result["protocol"]["primary_rate_metric"] == "physical_tokens_per_second"
    assert "outer_wall_ms" in result["protocol"]["available_diagnostic_metrics"]
    assert "prefill_ns" in result["protocol"]["planned_diagnostic_metrics"]
    assert "prefill_ns" not in result["protocol"]["available_diagnostic_metrics"]
    assert result["protocol"]["warmup"] == 2
    for arm in benchmark.ARM_NAMES:
        raw = result["arms"][arm]["raw"]
        assert [entry["phase"] for entry in raw].count("warmup") == 2
        assert [entry["phase"] for entry in raw].count("measure") == 4
        assert all(entry["snapshot"]["outer_wall_ms"] == 1.0 for entry in raw)
        assert result["arms"][arm]["summary"]["executor_wall_ms"]["median"] == 999.0
        assert result["arms"][arm]["summary"]["physical_tokens_per_second"]["median"] == 2000.0
        assert result["arms"][arm]["summary"]["visible_tokens_per_second"]["median"] == 2000.0
        assert "runtime_fingerprint" in result["arms"][arm]
        assert result["arms"][arm]["workload"]["repeats"] == 4
    assert result["comparison"]["token_identity"]
    assert result["limitations"]["fresh_process_per_arm"] is False
    assert result["limitations"]["r3_status"] == "open_shared_process_protocol"


def test_default_protocol_repeats_are_even():
    assert inspect.signature(benchmark.run_protocol).parameters["repeats"].default == 6


def test_protocol_returns_structured_difference_for_wrong_tokens(monkeypatch):
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", _clock())
    result = benchmark.run_protocol(
        FakeRuntime(mismatch=True), FakeIronMule, requests=1, max_tokens=3,
        warmup=2, repeats=2,
    )
    assert not result["comparison"]["token_identity"]
    assert result["comparison"]["token_differences"]
    assert result["comparison"]["token_differences"][0]["repeat"] in {0, 1}


def test_main_mismatch_exits_nonzero_and_persists_structured_diff(tmp_path, monkeypatch):
    runtime = FakeRuntime(mismatch=True)
    fake_ironmule = SimpleNamespace(
        Runtime=SimpleNamespace(load=lambda **_kwargs: runtime),
        Request=FakeIronMule.Request,
        InteractiveMode=FakeIronMule.InteractiveMode,
        ThroughputMode=FakeIronMule.ThroughputMode,
        StrictOneShotPlan=FakeIronMule.StrictOneShotPlan,
    )
    monkeypatch.setitem(sys.modules, "ironmule", fake_ironmule)
    output = tmp_path / "result.json"
    assert benchmark.main([
        "--requests", "1", "--max-tokens", "3", "--warmup", "2", "--repeats", "2",
        "--json", str(output),
    ]) == 2
    payload = json.loads(output.read_text())
    difference = payload["comparison"]["token_differences"][0]
    assert difference["fields"]["first_token_difference"] == {
        "position": 0, "expected": 8, "actual": 9,
    }
    assert "stop_reason" in difference["fields"]
    assert difference["fields"]["token_counts"]["physical"] == {
        "expected": 1, "actual": 1,
    }


@pytest.mark.parametrize("warmup,repeats", [(0, 2), (1, 2), (2, 0), (2, 1), (2, 3)])
def test_protocol_requires_repeated_warmups_and_measurements(warmup, repeats):
    with pytest.raises(ValueError):
        benchmark.run_protocol(
            FakeRuntime(), FakeIronMule, requests=1, max_tokens=3,
            warmup=warmup, repeats=repeats,
        )


def _roofline_inputs():
    return {
        "prefill_ns": 2_000_000,
        "decode_ns": 5_000_000,
        "decode_steps": 10,
        "effective_bandwidth_gbps": 300.0,
        "active_weight_bytes_per_token": 100_000_000,
        "kv_read_bytes_per_token": 1_000,
        "kv_write_bytes_per_token": 2_000,
        "extra_bytes_per_token": 3_000,
        "bandwidth_source": "measured_fixture",
        "bandwidth_source_kind": "measured_effective",
    }


def test_phase_roofline_diagnostic_reports_units_and_excludes_prefill_token():
    result = benchmark.phase_roofline_diagnostic(**_roofline_inputs())
    assert result["schema"] == "ironmule.phase_roofline.v1"
    assert result["diagnostic_only"] is True
    assert result["status"] == "ok"
    assert result["phases"]["prefill"] == {
        "duration_ns": 2_000_000, "duration_ms": 2.0, "status": "measured",
    }
    decode = result["phases"]["decode"]
    assert decode["decode_steps"] == 10
    assert decode["tokens_per_second"] == pytest.approx(2_000.0)
    roofline = result["roofline"]
    assert roofline["bandwidth_source"] == {
        "kind": "measured_effective", "label": "measured_fixture",
    }
    assert roofline["bytes_per_token"] == {
        "weights": 100_000_000, "kv_read": 1_000, "kv_write": 2_000,
        "extra": 3_000, "total": 100_006_000,
    }
    assert roofline["ideal_tokens_per_second"] == pytest.approx(
        300.0 * 1e9 / 100_006_000,
    )
    assert roofline["efficiency"] == pytest.approx(
        2_000.0 / roofline["ideal_tokens_per_second"],
    )
    assert "bandwidth_bound" not in result and "compute_bound" not in result
    assert "bandwidth_bound" not in roofline and "compute_bound" not in roofline


def test_phase_roofline_missing_traffic_is_inconclusive_but_keeps_phase_values():
    inputs = _roofline_inputs()
    inputs["kv_read_bytes_per_token"] = None
    result = benchmark.phase_roofline_diagnostic(**inputs)
    assert result["status"] == "inconclusive"
    assert result["phases"]["prefill"]["status"] == "measured"
    assert result["phases"]["decode"]["tokens_per_second"] == pytest.approx(2_000.0)
    assert "kv_read_bytes_per_token" in result["missing"]
    assert result["roofline"]["status"] == "inconclusive"
    assert "ideal_tokens_per_second" not in result["roofline"]
    assert "efficiency" not in result["roofline"]


@pytest.mark.parametrize("field", ["prefill_ns", "decode_steps"])
def test_phase_roofline_missing_phase_is_inconclusive(field):
    inputs = _roofline_inputs()
    inputs[field] = None
    result = benchmark.phase_roofline_diagnostic(**inputs)
    assert result["status"] == "inconclusive"
    assert field in result["missing"]
    assert result["roofline"]["status"] == "inconclusive"
    assert "ideal_tokens_per_second" not in result["roofline"]


def test_phase_roofline_missing_bandwidth_source_cannot_be_ok():
    inputs = _roofline_inputs()
    inputs["bandwidth_source"] = None
    result = benchmark.phase_roofline_diagnostic(**inputs)
    assert result["status"] == "inconclusive"
    assert result["roofline"]["status"] == "inconclusive"
    assert "bandwidth_source" in result["missing"]


def test_phase_roofline_empty_bandwidth_source_is_invalid():
    inputs = _roofline_inputs()
    inputs["bandwidth_source"] = " "
    result = benchmark.phase_roofline_diagnostic(**inputs)
    assert result["status"] == "invalid"
    assert result["roofline"]["status"] == "invalid"
    assert "bandwidth_source" in result["roofline"]["invalid"][0]


def test_phase_roofline_zero_step_decode_is_not_applicable():
    inputs = _roofline_inputs()
    inputs.update(decode_ns=0, decode_steps=0)
    result = benchmark.phase_roofline_diagnostic(**inputs)
    assert result["status"] == "inconclusive"
    assert result["phases"]["decode"] == {
        "duration_ns": 0, "decode_steps": 0,
        "tokens_per_second": None, "status": "not_applicable",
    }
    assert result["roofline"]["status"] == "not_applicable"


def test_phase_roofline_zero_step_boolean_duration_is_invalid():
    inputs = _roofline_inputs()
    inputs.update(decode_ns=False, decode_steps=0)
    result = benchmark.phase_roofline_diagnostic(**inputs)
    assert result["status"] == "invalid"
    assert result["phases"]["decode"]["duration_ns"] is None
    assert result["phases"]["decode"]["status"] == "invalid"


@pytest.mark.parametrize(
    "field,value",
    [
        ("prefill_ns", True),
        ("decode_ns", float("nan")),
        ("prefill_ns", 10**400),
        ("decode_steps", True),
        ("effective_bandwidth_gbps", -1.0),
        ("active_weight_bytes_per_token", float("inf")),
        ("kv_read_bytes_per_token", -1),
    ],
)
def test_phase_roofline_invalid_inputs_fail_closed(field, value):
    inputs = _roofline_inputs()
    inputs[field] = value
    result = benchmark.phase_roofline_diagnostic(**inputs)
    assert result["status"] == "invalid"
    assert result["roofline"]["status"] == "invalid"
    if field == "decode_ns":
        assert result["phases"]["decode"]["duration_ns"] is None
    json.dumps(result, allow_nan=False)
    assert "ideal_tokens_per_second" not in result["roofline"]
    assert "efficiency" not in result["roofline"]


def test_phase_roofline_zero_denominator_is_invalid():
    inputs = _roofline_inputs()
    inputs.update(active_weight_bytes_per_token=0, kv_read_bytes_per_token=0,
                  kv_write_bytes_per_token=0, extra_bytes_per_token=0)
    result = benchmark.phase_roofline_diagnostic(**inputs)
    assert result["status"] == "invalid"
    assert any("denominator" in message for message in result["roofline"]["invalid"])


def test_phase_roofline_subnormal_prefill_duration_fails_closed():
    inputs = _roofline_inputs()
    inputs["prefill_ns"] = 5e-324
    result = benchmark.phase_roofline_diagnostic(**inputs)
    assert result["status"] == "invalid"
    assert result["phases"]["prefill"]["duration_ms"] is None
    json.dumps(result, allow_nan=False)


def test_phase_roofline_efficiency_above_one_is_not_clamped():
    inputs = _roofline_inputs()
    inputs.update(prefill_ns=1, decode_ns=1_000_000_000, decode_steps=1,
                  effective_bandwidth_gbps=1e-10,
                  active_weight_bytes_per_token=1, kv_read_bytes_per_token=0,
                  kv_write_bytes_per_token=0, extra_bytes_per_token=0)
    result = benchmark.phase_roofline_diagnostic(**inputs)
    assert result["status"] == "ok"
    assert result["roofline"]["efficiency"] > 1
    assert result["roofline"]["warnings"] == [
        "efficiency_above_one_input_consistency",
    ]


def test_phase_roofline_nominal_peak_never_produces_efficiency():
    inputs = _roofline_inputs()
    inputs["bandwidth_source_kind"] = "nominal_peak"
    result = benchmark.phase_roofline_diagnostic(**inputs)
    assert result["status"] == "inconclusive"
    assert result["roofline"]["status"] == "inconclusive"
    assert result["roofline"]["reason"] == "nominal_peak_not_valid_for_efficiency"
    assert result["roofline"]["bandwidth_source"] == {
        "kind": "nominal_peak", "label": "measured_fixture",
    }
    assert "ideal_tokens_per_second" not in result["roofline"]
    assert "efficiency" not in result["roofline"]


@pytest.mark.parametrize("kind", [None, "estimated", 42])
def test_phase_roofline_bandwidth_provenance_fails_closed(kind):
    inputs = _roofline_inputs()
    inputs["bandwidth_source_kind"] = kind
    result = benchmark.phase_roofline_diagnostic(**inputs)
    assert result["status"] in {"inconclusive", "invalid"}
    assert result["roofline"]["status"] in {"inconclusive", "invalid"}
    assert "ideal_tokens_per_second" not in result["roofline"]
    assert "efficiency" not in result["roofline"]


def test_phase_roofline_rejects_huge_steps_and_sanitizes_json():
    inputs = _roofline_inputs()
    inputs["decode_steps"] = 2**63
    result = benchmark.phase_roofline_diagnostic(**inputs)
    assert result["status"] == "invalid"
    assert result["phases"]["decode"]["decode_steps"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "updates,needle",
    [
        ({"decode_ns": 5e-324, "decode_steps": 1}, "decode_tokens_per_second"),
        ({"active_weight_bytes_per_token": 1e308,
          "kv_read_bytes_per_token": 1e308,
          "kv_write_bytes_per_token": 1e308,
          "extra_bytes_per_token": 1e308}, "bytes_per_token"),
        ({"effective_bandwidth_gbps": 1e308,
          "active_weight_bytes_per_token": 1}, "ideal_tokens_per_second"),
        ({"effective_bandwidth_gbps": 5e-324,
          "active_weight_bytes_per_token": 1e308,
          "kv_read_bytes_per_token": 0,
          "kv_write_bytes_per_token": 0,
          "extra_bytes_per_token": 0}, "ideal_tokens_per_second"),
    ],
)
def test_phase_roofline_derived_overflow_or_underflow_fails_closed(updates, needle):
    inputs = _roofline_inputs()
    inputs.update(updates)
    result = benchmark.phase_roofline_diagnostic(**inputs)
    assert result["status"] == "invalid"
    assert result["roofline"]["status"] == "invalid"
    assert any(needle in message for message in result["roofline"]["invalid"])
    json.dumps(result, allow_nan=False)
