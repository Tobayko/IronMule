"""Campaign-level money limits and cleanup survive another control process."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

OPERATOR = Path(__file__).resolve().parents[2] / "lab" / "rp"


def cli(state, *args):
    return subprocess.run(
        [sys.executable, str(OPERATOR / "rp.py"), *args],
        env={**os.environ, "RP_STATE": str(state), "RP_BUDGET": "9000"},
        check=False, capture_output=True, text=True,
    )


def test_external_reservations_share_budget_across_processes(tmp_path):
    state = tmp_path / "state.json"
    assert cli(state, "reserve-external", "code", "12").returncode == 0
    assert cli(state, "reserve-external", "next", "4").returncode != 0
    assert cli(state, "reserve-external", "bad", "nan").returncode != 0
    assert cli(state, "settle-external", "code", "3").returncode == 0
    assert cli(state, "reserve-external", "next", "12").returncode == 0
    assert cli(state, "extend", "another target needs a bounded test").returncode == 0
    assert cli(state, "reserve-external", "last", "5").returncode == 0
    assert cli(state, "reserve-external", "over", "0.01").returncode != 0
    assert json.loads(cli(state, "resume").stdout)["budget"] == 20
    assert state.stat().st_mode & 0o077 == 0


def test_lost_create_result_reconciles_only_owned_pod(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(OPERATOR))
    import rp
    monkeypatch.setattr(rp, "STATE", tmp_path / "rp.json")
    state = {"spent": 0., "pods": [], "pod": {"name": "owned", "id": None, "disk_rate": .02,
            "started": time.time(), "gpu": "fixture", "balance_before": 22., "reserved": 1.,
            "deadline": time.time() - 1}}
    rp.save(state)
    other = {"name": "other", "id": "unrelated", "costPerHr": 2}
    account = {"pods": [other], "clientBalance": 21.9}
    monkeypatch.setattr(rp, "account", lambda: account)
    with pytest.raises(ValueError, match="unresolved"):
        rp.cleanup(rp.load())
    assert rp.load()["pod"]["reserved"] == 1.
    account["pods"].append({"name": "owned", "id": "provider-id", "costPerHr": 2})
    terminated = []

    def terminate(query, variables):
        terminated.append(variables["i"]["podId"])
        account["pods"] = [other]
    monkeypatch.setattr(rp, "gql", terminate)
    rp.watch("owned")
    assert terminated == ["provider-id"]
    recovered = rp.load()
    assert recovered["pod"] is None
    assert recovered["spent"] >= rp.CLEANUP_SECONDS / 3600 * 2.02
    assert recovered["events"][-1]["kind"] == "removed"


def test_create_timeout_leaves_reservation_and_independent_watch(tmp_path, monkeypatch):
    import argparse

    monkeypatch.syspath_prepend(str(OPERATOR))
    import rp
    monkeypatch.setattr(rp, "STATE", tmp_path / "rp.json")
    monkeypatch.setattr(rp, "account", lambda: {"clientBalance": 32., "pods": []})
    calls = []
    def provider(query, variables):
        if "gpuTypes" in query:
            return {"gpuTypes": [{"id": "gpu", "lowestPrice": {
                "uninterruptablePrice": 2., "stockStatus": "Available"}}]}
        saved = rp.load()["pod"]
        assert calls == [saved["name"]]
        assert saved["reserved"] > 2.
        env = {v["key"]: v["value"] for v in variables["i"]["env"]}
        assert env["RP_DEADLINE"] == str(saved["deadline"])
        assert "RUNPOD_API_KEY" not in env
        raise TimeoutError()
    monkeypatch.setattr(rp, "gql", provider)
    monkeypatch.setattr(rp, "launch_watch", calls.append)
    args = argparse.Namespace(count=1, disk=80, gpu="gpu", minutes=60, max_hourly=2.5,
                              cloud="SECURE", image="test", vcpu=4, mem=16)
    with pytest.raises(TimeoutError):
        rp.book(args)
    with pytest.raises(ValueError, match="unresolved"):
        rp.book(args)
    assert len(calls) == 1
