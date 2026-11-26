import json
import subprocess

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
    root = tmp_path / "repo"
    root.mkdir()
    module = root / "ironmule/gguf_tokenizer.py"
    module.parent.mkdir()
    module.write_text("def _complete_prefix(data):\n    return len(data)\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.org", "commit", "-qm", "base"],
                   cwd=root, check=True)
    from runpy import run_path
    _complete_prefix = run_path(str(module))["_complete_prefix"]
    with observe_wave(tmp_path, "short", {"hardware": "test-cpu"}, root) as record:
        assert _complete_prefix(b"hello") == 5
        record.update(requests=[[2, 1]], correct=True)
    saved = json.loads(next(tmp_path.glob("wave-*.json")).read_text())
    assert saved["kind"] == "measured"
    assert any(f["function"] == "_complete_prefix" for f in saved["functions"])
    with pytest.raises(RuntimeError), observe_wave(tmp_path, "failed", {"hardware": "test-cpu"}, root):
        raise RuntimeError("interrupted trial")
    assert len(list(tmp_path.glob("wave-*.json"))) == 1


def test_followup_can_save_the_cost_of_an_uninformative_experiment():
    from training.experiment_planner import DEFERRED
    observations = [wave(i, "document", .03, .0002) for i in (1, 2)]
    first = plan(observations)
    decision = plan(observations, [first])
    assert decision["schema"] == DEFERRED and "order_id" not in decision
    assert decision["simulation"]["highest_remaining_measured_self_fraction"] == .0001
    changed = [wave(i, "document", .03, .5) for i in (3, 4)]
    assert plan(changed, [first])["target"]["function"] == "piece"
