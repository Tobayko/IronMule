import json
from pathlib import Path

import pytest

from training.experiment_planner import EVALUATOR, OBSERVATION, observe_wave, plan


def wave(i, workload, host_a, host_b):
    return {"schema": OBSERVATION, "kind": "measured", "base_commit": "test-source", "evaluator": EVALUATOR,
            "binding": {"hardware": "test-cpu", "gguf": "test-weights"}, "workload": workload,
            "captured_ns": i, "correct": True, "requests": [[12, 2], [8, 8]], "wall_s": 2,
            "functions": [{"path": "ironmule/gguf_tokenizer.py", "function": name, "line": n,
                           "self_s": host, "calls": 10}
                          for n, (name, host) in enumerate([("encode", host_a), ("piece", host_b)], 1)]}


def test_replays_load_transition_selects_different_work_orders_and_excludes_submitted_target():
    quiet = wave(1, "short", 0.001, 0.4)
    document = wave(2, "document", 1, 0.001)
    order = plan([quiet, document, dict(quiet, captured_ns=3)])
    assert order["target"]["function"] == "encode"
    assert all(r["source_sha256"] and r["locator"] for r in order["historical_experience"])
    assert order["historical_experience"]
    assert order["next_workloads"] == ["document"]
    assert order["simulation"]["kind"] == "simulated"
    assert all(a["sensitivity_saved_s"][0] == 0 for a in order["dreams"])
    later_order = plan([quiet, document, dict(quiet, captured_ns=3), dict(document, captured_ns=4)])
    assert later_order["target"]["function"] == "piece"
    next_order = plan([quiet, document, dict(quiet, captured_ns=3)], [order])
    assert next_order["target"]["function"] == "piece"


def test_refuses_mixed_foreign_unsuccessful_or_simulated_profiles():
    quiet = wave(1, "short", .1, .2)
    for change in ({"binding": {"hardware": "other"}}, {"kind": "simulated"}, {"correct": False},
                   {"base_commit": "other"}, {"requests": [[2, 0]]}, {"width": 0}, {"wall_s": float("nan")}):
        with pytest.raises(ValueError):
            plan([quiet, dict(quiet, captured_ns=2, **change)])


def test_captures_only_completed_measured_python_work(tmp_path):
    from ironmule.gguf_tokenizer import _complete_prefix
    root = Path(__file__).resolve().parents[2]
    with observe_wave(tmp_path, "short", {"hardware": "test-cpu"}, root) as record:
        assert _complete_prefix(b"hello") == 5
        record.update(requests=[[2, 1]], correct=True)
    saved = json.loads(next(tmp_path.glob("wave-*.json")).read_text())
    assert saved["kind"] == "measured"
    assert any(f["function"] == "_complete_prefix" for f in saved["functions"])
    with pytest.raises(RuntimeError), observe_wave(tmp_path, "failed", {"hardware": "test-cpu"}, root):
        raise RuntimeError("interrupted trial")
    assert len(list(tmp_path.glob("wave-*.json"))) == 1
