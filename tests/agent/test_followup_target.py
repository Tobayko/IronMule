from types import SimpleNamespace

from lab.pk1 import followup_target as target
from training.code_trial import save


def test_followup_regression_withdraws_durably_and_invokes_preceding_release(tmp_path, monkeypatch):
    state = tmp_path / "active"
    d = state / "deployment"
    d.mkdir(parents=True)
    save(tmp_path / "order", {})
    save(d / "target-check.json", {"metrics": {"reference": {"cold_s": 10.0}}})
    args = SimpleNamespace(order=tmp_path / "order", checkout=tmp_path, receipt=tmp_path / "r", receipt_sha256="r")
    monkeypatch.setattr(target.followup_release, "register_receipt", lambda *_: "test-fixture")
    monkeypatch.setattr(target.release, "pinned", lambda *_: {"qualification": {"reference_sha256": "prior"}})
    monkeypatch.setattr(target.release, "selected", lambda *_: "reference" if (d / "selection.json").exists() else "candidate")
    monkeypatch.setattr(target, "serving_metrics", lambda _: {"cold_s": 11.0})
    monkeypatch.setattr(target, "checked", lambda *_, **kw: {})
    monkeypatch.setattr(target, "preceding", lambda _, seed, expected: {"from": expected, "seed": seed})
    assert target.run(args, state, 100) == {"from": "prior", "seed": 100}
    assert target.read(d / "selection.json")["withdrawn"] is True
    assert target.run(args, state, 101) == {"from": "prior", "seed": 101}


def test_fresh_gate_requires_new_counterbalanced_mixed_and_factual_measurements(tmp_path, monkeypatch):
    observed = []
    def checked(_, receipt, arm, seed, path, *, factual):
        observed.append((seed, arm, factual))
        save(path, {"arm": arm})
        return {"arm": arm}
    monkeypatch.setattr(target, "checked", checked)
    monkeypatch.setattr(target, "serving_metrics", lambda _: {"cold_s": 10.0})
    monkeypatch.setattr(target.trial, "scores", lambda ref, cand: {"pair": (ref["arm"], cand["arm"])})
    monkeypatch.setattr(target.trial, "gate", lambda mixed, facts, aa: len(mixed) == len(facts) == 1 and aa == .002)
    control = {"qualification": {**{k: "pinned" for k in target.release.BOUND}, "metrics": {"aa_spread": .002}}}
    out = target.fresh(None, control, tmp_path)
    assert out["passed"] and len(out["metrics"]["raw_sha256"]) == 4
    assert observed == [(97051, "reference", False), (97051, "candidate", False), (97052, "candidate", True), (97052, "reference", True)]
