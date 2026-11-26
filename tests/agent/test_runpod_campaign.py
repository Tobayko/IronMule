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


def test_two_phase_renewal_reserves_once_and_keeps_original_cutoff_on_failure(tmp_path, monkeypatch):
    from lab.rp import rp

    monkeypatch.setattr(rp, "STATE", tmp_path / "rp.json")
    old = int(time.time()) + 1200
    pod = {"id": "owned-id", "name": "owned", "started": old - 3600, "deadline": old,
           "reserved": 2.5 * 70 / 60, "cost": 2.2, "disk_rate": .01}
    state = {"pod": pod, "pods": [], "spent": 0.}
    provider = {"clientBalance": 32., "pods": [{"id": "owned-id", "name": "owned", "costPerHr": 2.19}]}
    monkeypatch.setattr(rp, "account", lambda: provider)
    monkeypatch.setattr(rp, "launch_watch", lambda _: None)
    calls = []
    def no_ack(pod, action):
        calls.append(action)
        if action == "commit":
            raise TimeoutError()
    monkeypatch.setattr(rp, "remote_handoff", no_ack)
    state["spent"] = 10.
    with pytest.raises(ValueError, match="insufficient"):
        rp.renew(state, 90, "same GPU")
    state["spent"] = 0.
    with pytest.raises(TimeoutError):
        rp.renew(state, 90, "same GPU")
    pending = rp.load()
    assert calls == ["commit", "revert"] and pending["pod"]["deadline"] == old
    assert pending["pod"]["reserved"] == pytest.approx(2.5 * 160 / 60)
    monkeypatch.setattr(rp, "remote_handoff", lambda *a: {"effective": old + 5400, "live": True})
    assert rp.renew(pending, 90, "same GPU")["deadline"] == old + 5400
    assert rp.renew(rp.load(), 90, "same GPU")["reserved"] == pytest.approx(2.5 * 160 / 60)
    with pytest.raises(ValueError, match="one renewal"):
        rp.renew(rp.load(), 5, "again")


def test_report_requires_verified_cleanup_and_excludes_private_data(tmp_path, monkeypatch):
    from lab.rp import rp

    pod = {"id": "private-id", "gpu": "GPU", "hours": .5, "usd": 1.5, "account_delta": .98}
    state = {"spent": 1.5, "pods": [pod], "events": [{"kind": "reserve", "balance": 32., "name": "private-name"}]}
    provider = {"clientBalance": 31., "pods": [{"id": "private-id"}]}
    monkeypatch.setattr(rp, "account", lambda: provider)
    dest = tmp_path / "cost.json"
    with pytest.raises(ValueError, match="still lists"):
        rp.cost_report(state, dest)
    provider["pods"] = []
    monkeypatch.setattr(rp, "posted_billing", lambda _: {"posted_records": 0, "posted_usd": None})
    result = rp.cost_report(state, dest)
    assert result["leases"][0]["posted_usd"] is None and result["provider_account_delta_since_first_reservation_usd"] == 1.
    assert "private" not in dest.read_text()


def test_real_process_handoff_arms_replacement_before_original_exits(tmp_path, monkeypatch):
    """Exercise the handoff with isolated local processes and a harmless sleeping fixture."""
    import base64
    import importlib.util
    import signal

    helper = tmp_path / "handoff.py"
    helper.write_bytes((OPERATOR / "renew_watch.py").read_bytes())
    spec = importlib.util.spec_from_file_location("renew_watch_fixture", helper)
    watcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watcher)
    original = tmp_path / "original.py"
    original.write_text("import time\ntime.sleep(600)\n")
    old = int(time.time()) + 300
    new = old + 60
    monkeypatch.setenv("WATCH_B64", base64.b64encode(original.read_bytes()).decode())
    monkeypatch.setenv("RP_DEADLINE", str(old))
    monkeypatch.setenv("RUNPOD_API_KEY", "local-fixture-no-network")
    monkeypatch.setenv("RUNPOD_POD_ID", "fixture")
    monkeypatch.setattr(watcher, "ORIGINAL", original)
    monkeypatch.setattr(watcher.os, "geteuid", lambda: 0)
    proc = subprocess.Popen([sys.executable, str(original)], start_new_session=True)
    try:
        watcher.handoff(old, new, "commit")
        assert proc.wait(timeout=2) == -signal.SIGTERM
        assert watcher.live(old, new) and json.loads(watcher.MARKER.read_text()) == new
        watcher.handoff(old, new, "revert")
        assert watcher.live(old, new) and not watcher.MARKER.exists()
    finally:
        if watcher.READY.exists():
            pid = json.loads(watcher.READY.read_text())["pid"]
            if watcher.live(old, new):
                os.kill(pid, signal.SIGTERM)
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=2)
