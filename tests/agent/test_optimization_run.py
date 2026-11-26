"""Bounded orchestration and crash recovery without allocating a server or contacting Codex."""
from types import SimpleNamespace

import pytest

from training import optimization_run as op


def test_resume_preserves_order_and_freezes_inputs(tmp_path, monkeypatch):
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"immutable model")
    args = SimpleNamespace(state=tmp_path / "state", run_id="first", gguf=model, provider="cpu", base=None, publish=False)
    events = []
    monkeypatch.setattr(op, "git", lambda *a, **kw: "" if a[0] == "status" else "a" * 40)
    monkeypatch.setattr(op.code_trial, "prepare", lambda run, base, name: tmp_path)
    monkeypatch.setattr(op.experience_archive, "build", lambda tree: [])

    def discover(tree, run, model, provider):
        if (run / "discovery").exists():
            return
        (run / "discovery/waves").mkdir(parents=True)
        op.save(run / "discovery/trial.json", {"workload": {"width": 4}})
        events.append("measure")

    def plan(rows, past, **kw):
        events.append("plan")
        return {"base_commit": "a" * 40, "binding": {"model_sha256": op.sha(model)}, "order_id": "order",
                "target": {"function": "example"}}

    def cycle(options):
        if "interrupt" not in events:
            events.append("interrupt")
            raise InterruptedError()
        events.append("judge")
        options.state.mkdir(exist_ok=True)
        op.save(options.state / "result.json", {"passed": False, "reason": "slower"})

    monkeypatch.setattr(op, "discover", discover)
    monkeypatch.setattr(op.experiment_planner, "plan", plan)
    monkeypatch.setattr(op.code_trial, "cycle", cycle)
    with pytest.raises(InterruptedError):
        op.run(args)
    result = op.run(args)
    assert op.run(args) == result
    assert not result["historical_qualification"]["passed"] and events == ["measure", "plan", "interrupt", "judge"]
    run = args.state / "runs/first"
    assert op.read(run / "experience.json")["decision"] is False
    failure = op.read(next(run.glob("failure-*.json")))
    assert failure["reason"].endswith("InterruptedError")
    assert failure["source_sha256"] == op.sha(run.parent / failure["source"])
    assert result["deployment"] == {"active": None, "withdrawn": False}
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="different model"):
        op.run(args)


def test_break_even_uses_completed_work_and_full_experiment_time(tmp_path, monkeypatch):
    monkeypatch.setattr(op.code_trial, "PAIRS", 2)
    assert op.paid_back_after(tmp_path, 100) is None
    for name in ("pair-0.json", "pair-1.json"):
        op.save(tmp_path / name, {"reference": {"wall_s": 10, "completed": 10}, "candidate": {"wall_s": 8, "completed": 10}})
    assert op.paid_back_after(tmp_path, 100) == 500


def test_cached_result_with_changed_active_link_withdraws(tmp_path, monkeypatch):
    d = tmp_path / "deployment"
    d.mkdir()
    op.save(d / "release.json", {"base": "a" * 40, "candidate": "b" * 40, "order_id": "o", "pr": 123})
    (d / "active").symlink_to("c" * 40)
    monkeypatch.setattr(op.trial_release, "checkout", lambda root, commit: root / commit)
    state = op.current_deployment(tmp_path)
    assert state["withdrawn"]
    assert state["active"] == "a" * 40
    assert (d / "active").readlink().name == "a" * 40
    assert op.current_deployment(tmp_path) == state
