"""The diagnosis harness must decode a killed child correctly, or it repeats the gap it exists to close."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "b60", ROOT / "tools" / "b60_12b_child_diagnosis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_child_bootstrap_compiles():
    compile(_module().CHILD_BOOTSTRAP, "<bootstrap>", "exec")


def test_a_signalled_child_is_reported_as_signalled():
    decode = _module()._decode_status
    killed = decode(-9)
    assert killed["signalled"] and not killed["exited"]
    assert killed["signal"] == 9 and killed["signal_name"] == "SIGKILL"
    assert decode(0) == {"returncode": 0, "exited": True, "signalled": False,
                         "signal": None, "signal_name": None,
                         "interpretation": "exited normally with status 0"}
    assert decode(1)["exited"] and not decode(1)["signalled"]
    assert decode(None)["returncode"] is None and not decode(None)["exited"]


def test_the_two_conditions_differ_in_one_knob_only():
    module = _module()
    differing = [k for k in module.CONDITION_A
                 if module.CONDITION_A[k] != module.CONDITION_B[k]]
    assert differing == ["wired_fraction"]
    assert module.CONDITION_A["wired_fraction"] == 0.6
    assert module.CONDITION_B["wired_fraction"] == 0.0


def test_the_budget_is_a_fraction_of_a_confirmation():
    module = _module()
    assert module.MAX_TOKENS <= 8 and module.REPEATS == 1 and module.WARMUP <= 1
    assert "not_a_performance_study" in module.PREREGISTRATION
    assert module.SCREENING_RECORD["unknown"], "the unrecorded search steps must stay named"
