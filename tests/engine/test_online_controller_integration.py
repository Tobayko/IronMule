"""Tiny real-model/native-controller correctness check, without a speed claim."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def controller_library():
    from ironmule_controller import default_library_path
    library = default_library_path()
    if not library.is_file():
        pytest.skip("native controller library is not built; run cargo build --release")
    return library


@pytest.fixture
def cached_runtime(controller_library):
    # Preserve the existing fixture's enumerated environment-only skip policy.
    # No model downloads or broad exception-to-skip conversions are permitted.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from tests.engine.test_ironmule_runtime_integration import _runtime
    runtime, _ = _runtime(None)
    try:
        yield runtime
    finally:
        runtime.close()


def _signature(results):
    return [(item.rid, tuple(item.tokens), item.stop_reason,
             item.metrics["physical_generated_tokens"],
             item.metrics["visible_generated_tokens"]) for item in results]


def test_native_budgeted_profile_executes_on_cached_model(cached_runtime, controller_library):
    from ironmule_controller import ControllerConfig, OnlineController
    from ironmule.service import Request, runtime_identity

    runtime = cached_runtime
    ids = runtime.encode("Count from one to four.")

    def pair():
        return [Request(prompt_ids=ids, max_tokens=4, rid=f"check-{index}")
                for index in range(2)]

    reference = runtime.serve(pair())
    identity = runtime_identity(runtime)
    controller = OnlineController(
        identity, library_path=controller_library,
        config=ControllerConfig(exploration_ppm=50_000, seed=15),
    )
    try:
        runtime.attach_online_controller(controller)
    except BaseException:
        controller.close()
        raise
    # Nineteen actual singleton executions accrue 0.95 trials of budget. Their
    # reference-only mask prevents random draws and no alternate is trained.
    for index in range(19):
        result = runtime.serve([Request(prompt_ids=ids, max_tokens=1,
                                        rid=f"budget-{index}")])[0]
        assert len(result.tokens) == 1
        assert controller.last_decision["actual_action"] == 0
        assert controller.flush()
    before = controller.snapshot()
    assert before["decisions"] == 19
    assert before["promoted"] == 0

    # The twentieth decision supplies one credit. Seed 15's first xorshift
    # draw has residue 46,415, below the configured 50,000 exploration cutoff.
    selected = runtime.serve(pair())
    decision = controller.last_decision
    assert decision["kind"] == 2
    assert decision["action"] == decision["actual_action"] == 1
    assert decision["comparison"] is False
    assert decision["correctness"] == "unchecked"
    assert _signature(selected) == _signature(reference)
    assert runtime.telemetry.mode == "throughput"
    assert max(runtime.telemetry.realised_widths) == 2
    assert runtime.telemetry.fallbacks == 0
    assert runtime.telemetry.plan_switch_attempts == 0
    assert runtime.telemetry.correctness_check_performed is False
    assert all(result.metrics["caller_return_latency_ms"] > 0 for result in selected)
    assert controller.flush()
    after = controller.snapshot()
    assert after["updates"] > before["updates"]
    assert after["promoted"] == 0
    assert after["service"]["completed_requests"] == 21
    assert after["pending"] == 0
    assert after["service"]["comparisons"] == 0


def test_native_hardware_learning_selects_compiler_profile_without_new_probe(controller_library, tmp_path, monkeypatch):
    """Use the existing cache and one model to check the new served-knob route."""
    from ironmule import hw
    from ironmule.service import Request, Runtime
    from ironmule_controller import ControllerConfig
    from tests.engine.test_ironmule_runtime_integration import _is_expected_unavailable

    def forbidden_measure(*_args, **_kwargs):
        pytest.fail("hardware integration must reuse the existing probe cache")
    monkeypatch.setattr(hw, "measure", forbidden_measure)
    try:
        hw.probe(allow_measure=False)
    except hw.HardwareProbeUnavailable as exc:
        pytest.skip(f"matching hardware probe cache unavailable: {exc}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        runtime = Runtime.load("mlx-community/gemma-3-1b-it-4bit",
                               revision="2d44e83dc9e80843d22fb941d3d699a0b1351aa6",
                               use_tuned_profile=False)
    except Exception as exc:
        if _is_expected_unavailable(exc):
            pytest.skip(f"local model unavailable: {type(exc).__name__}: {exc}")
        raise
    controller = None
    try:
        ids = runtime.encode("Count from one to four.")
        original_knobs = runtime.engine.knobs
        reference = runtime.serve([Request(prompt_ids=ids, max_tokens=8, rid="compiled-check")])
        original_compiled = getattr(runtime.engine, "_compiled", None)
        original_capacity = getattr(runtime.engine, "_compiled_capacity", None)
        seen = []
        original_prefill = runtime.backend.prefill
        def prefill(prompt_ids, plan, capacity):
            seen.append((runtime.engine.knobs.compiled_fixed_cache,
                         runtime.engine.knobs.head_skip_prefill))
            return original_prefill(prompt_ids, plan, capacity)
        monkeypatch.setattr(runtime.backend, "prefill", prefill)
        controller = runtime.enable_hardware_learning(
            tmp_path / "hardware-learning.json", library_path=controller_library,
            config=ControllerConfig(exploration_ppm=50_000, seed=15))
        for index in range(19):
            row = runtime.serve([Request(prompt_ids=ids, max_tokens=1,
                                        rid=f"compiler-budget-{index}")])[0]
            assert len(row.tokens) == 1
            assert controller.last_decision["actual_action"] == 0
            assert controller.flush()
        before = controller.snapshot()
        selected = runtime.serve([Request(prompt_ids=ids, max_tokens=8, rid="compiled-check")])
        decision = controller.last_decision
        assert decision["eligible_mask"] == 0b0101
        assert decision["kind"] == 2
        assert decision["proposed_action"] == decision["observation_action"] == 2
        assert decision["executed_actions"] == [0, 2]
        assert decision["returned_action"] == decision["actual_action"] == 0
        assert decision["correctness"] == "matched"
        assert _signature(selected) == _signature(reference)
        assert selected[0].text == reference[0].text
        assert (False, False) in seen and (True, True) in seen
        assert runtime.engine.knobs is original_knobs
        assert runtime.engine._compiled is original_compiled
        assert runtime.engine._compiled_capacity == original_capacity
        assert runtime.telemetry.correctness_check_performed
        assert runtime.telemetry.correctness_checked_requests == 1
        assert runtime.telemetry.fallbacks == 0
        assert runtime.telemetry.plan_switch_attempts == 0
        assert controller.flush()
        after = controller.snapshot()
        assert after["updates"] > before["updates"]
        assert after["working_counts"][2] >= 1
        assert after["promoted"] == 0
        assert after["service"]["completed_requests"] == 20
    finally:
        try:
            if controller:
                controller.close()
        finally:
            runtime.close()
