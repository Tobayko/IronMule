"""OPT1 adapter 1 (IronMule) speaks the coordinator's protocol. Model-free: tune's model, engine,
confirmation and profile store are replaced by fakes; the real cycle runs on hardware (OPT1 run 2)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import importlib

from ironmule import opt_adapter
from ironmule.runtime import BASELINE, Knobs

tune = importlib.import_module("ironmule.tune")


def _args(**kw):
    base = dict(model="m/x", prompt="p", max_tokens=8, repeats=2, action=None, seed=None, previous=None)
    return SimpleNamespace(**{**base, **kw})


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr("ironmule.bench.environment", lambda: {"mlx": "0.32", "mlx_lm": "0.31"})
    store = {"profiles": {}}
    monkeypatch.setenv("IMOPT_IRONMULE_STATE", str(tmp_path / "state"))
    monkeypatch.delenv("IRONMULE_RUST2_BINARY", raising=False)
    monkeypatch.setattr(opt_adapter, "fingerprint", lambda: "fp")
    monkeypatch.setattr(tune, "_all_profiles", lambda: dict(store["profiles"]))
    monkeypatch.setattr(tune, "PROFILES", tmp_path / "profiles.json")
    monkeypatch.setattr(opt_adapter, "STORE", tmp_path)

    def restore_reads():
        p = tmp_path / "profiles.json"
        if p.exists():
            store["profiles"] = json.loads(p.read_text())
    monkeypatch.setattr(tune, "_profile_key", lambda fp, identity, dtype=None: f"{fp}/{identity.identity_sha256}")
    monkeypatch.setattr(tune, "resolve_local_model",
                        lambda model, revision=None: SimpleNamespace(identity=SimpleNamespace(identity_sha256="id", model_id=model), path=tmp_path))
    store["restore_reads"] = restore_reads
    return store


def test_describe_lists_the_reference_and_one_knob_change_per_action(fake, monkeypatch):
    monkeypatch.setattr("ironmule.bench.environment", lambda: {"mlx": "0.32", "mlx_lm": "0.31"})
    d = opt_adapter.describe(_args())
    ids = [a["id"] for a in d["actions"]]
    assert d["reference"] == "baseline" and ids[0] == "baseline" and d["evaluator"] == "adapter"
    assert "readback_every=2" in ids and "compiled_fixed_cache=True" in ids
    assert all(a["class"] == "auto" for a in d["actions"]) and opt_adapter.RUST2 not in ids
    monkeypatch.setenv("IRONMULE_RUST2_BINARY", "/bin/rust2")
    monkeypatch.setattr(opt_adapter, "_cuda", lambda: True)
    rust2 = [a for a in opt_adapter.describe(_args())["actions"] if a["id"] == opt_adapter.RUST2]
    assert rust2 and rust2[0]["class"] == "tested_only"


def test_run_reports_cost_and_a_token_digest_and_a_failure_is_not_a_measurement(fake, monkeypatch):
    closed = []
    monkeypatch.setattr(tune, "load_engine", lambda model, knobs, offline=True: (SimpleNamespace(knobs=knobs), object()))
    monkeypatch.setattr(tune, "prompt_ids", lambda tok, prompt: [1, 2, 3])
    monkeypatch.setattr(tune, "_eos_ids", lambda tok: (0,))
    monkeypatch.setattr(tune, "_close_engine", lambda e: closed.append(e))
    monkeypatch.setattr(tune, "measure", lambda engine, ids, n, eos, repeats=5: {
        "total_ns": 2e9, "prefill_ns": 1e8, "decode_ns": 1.9e9, "logical_tokens": [5, 6], "deterministic": True, "capacity": 64})
    r = opt_adapter.run(_args(action="readback_every=2"))
    assert r["ok"] and r["cost"] == 2.0 and len(r["correctness"]) == 64 and closed
    assert opt_adapter.run(_args(action="nonsense"))["ok"] is False

    def broken(model, knobs, offline=True):
        raise ValueError("fusion unsupported here")
    monkeypatch.setattr(tune, "load_engine", broken)
    failed = opt_adapter.run(_args(action="fuse_projections=True"))
    assert failed["ok"] is False and "unsupported" in failed["error"]


def test_only_an_accepted_confirmation_activates_and_revert_restores_the_previous_entry(fake, monkeypatch):
    knobs = Knobs(**{**BASELINE.as_dict(), "readback_every": 2})
    saved = []
    monkeypatch.setattr(tune, "confirm", lambda *a, **k: {"ratios": {"candidate/baseline": {"total_ns": {"median_ratio": 0.8}}},
                                                          "processes": 2})
    monkeypatch.setattr(tune, "_confirmation_evidence_sha256", lambda raw: "e" * 64)
    monkeypatch.setattr(tune, "_confirmation_decision", lambda raw, expected_baseline, expected_candidate: (False, "ci_not_below_one"))
    assert opt_adapter.qualify(_args(action="readback_every=2"))["qualified"] is False
    assert opt_adapter.apply(_args(action="readback_every=2"))["ok"] is False  # rejected evidence never activates
    assert opt_adapter.qualify(_args(action=opt_adapter.RUST2))["qualified"] is False  # not auto-qualifiable

    monkeypatch.setattr(tune, "_confirmation_decision", lambda raw, expected_baseline, expected_candidate: (True, None))
    assert opt_adapter.qualify(_args(action="readback_every=2"))["qualified"] is True
    # a qualification is bound to its model, machine, software and workload
    assert opt_adapter.apply(_args(action="readback_every=2", max_tokens=64))["ok"] is False
    fake["profiles"] = {"fp/id": {"knobs": BASELINE.as_dict(), "marker": "old"}}
    state = {"active": None}
    monkeypatch.setattr(opt_adapter, "_profile_for", lambda args, k, raw: {"knobs": k.as_dict(), "marker": "new"})

    def save(profile):
        saved.append(profile)
        state["active"] = profile
    monkeypatch.setattr(tune, "save_profile", save)
    monkeypatch.setattr(tune, "load_profile", lambda model, expected_candidate=None: state["active"])
    answer = opt_adapter.apply(_args(action="readback_every=2"))
    assert answer["ok"] and saved[0]["knobs"] == knobs.as_dict()
    assert answer["previous"] == {"key": "fp/id", "profile": {"knobs": BASELINE.as_dict(), "marker": "old"}, "action": None}
    assert opt_adapter.active(_args())["action"] == "readback_every=2"
    opt_adapter.revert(_args(previous=json.dumps(answer["previous"])))
    fake["restore_reads"]()
    assert fake["profiles"] == {"fp/id": {"knobs": BASELINE.as_dict(), "marker": "old"}}
    opt_adapter.revert(_args(previous=json.dumps({"key": "fp/id", "profile": None})))
    fake["restore_reads"]()
    assert fake["profiles"] == {}


def test_main_prints_exactly_one_protocol_line(fake, monkeypatch, capsys):
    monkeypatch.setattr(tune, "load_profile", lambda model, expected_candidate=None: None)
    assert opt_adapter.main(["active", "--model", "m/x"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert json.loads(out[-1]) == {"ok": True, "action": None}
