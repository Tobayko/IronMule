import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab.rp import native_campaign as campaign


@pytest.mark.parametrize("late_rejection", [False, True])
def test_source_order_resumes_into_a_single_remote_qualification(tmp_path, monkeypatch, late_rejection):
    cfg = {k: "/" + k for k in campaign.CONFIG_KEYS}
    cfg.update(pod_id="existing", table="/cache/dispatch.table")
    config, order = tmp_path / "target.json", tmp_path / "order.json"
    config.write_text(json.dumps(cfg))
    profile = order.with_suffix(".profile.json")
    profile.write_text('{"trace":true}')
    table_hash = hashlib.sha256(b"table").hexdigest()
    row = {"base_commit": "a" * 40, "binding": {"table_sha256": table_hash},
           "profile_sha256": campaign.native_trial.checksum(profile)}
    row["order_id"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
    order.write_text(json.dumps(row))
    args = SimpleNamespace(config=config, order=order, state=tmp_path / "campaign", checkout=tmp_path,
                           author_cli=False, publish=False)
    launches = []
    result = {"passed": False, "context": {"order": row["order_id"]}, "reason": "paired throughput gate"}
    def author(author_args):
        if not author_args.output.parent.exists():
            author_args.output.parent.mkdir()
            raise campaign.PendingAuthor()
        author_args.output.write_text("kernel")
    def snapshot(pod, directory, names, local):
        local.mkdir(exist_ok=True)
        if "dispatch.table" in names:
            (local / "dispatch.table").write_bytes(b"table")
        if launches and "result.json" in names:
            if late_rejection:
                result["schema"] = campaign.native_trial.SCHEMA
                result["context"].update(base_commit=row["base_commit"], candidate_source=hashlib.sha256(b"kernel").hexdigest())
                (local / "context.json").write_text(json.dumps(result["context"]))
            (local / "result.json").write_text(json.dumps(result))
    monkeypatch.setattr(campaign.native_trial, "author", author)
    monkeypatch.setattr(campaign, "lease", lambda ident: {"id": ident, "started": 1, "reserved": 2, "cost": 1})
    monkeypatch.setattr(campaign, "upload", lambda *a: None)
    monkeypatch.setattr(campaign, "snapshot", snapshot)
    monkeypatch.setattr(campaign, "shell", lambda *a: campaign.native_trial.checksum(campaign.native_trial.__file__))
    def job(pod, job_name, command):
        launches.append(command)
        if late_rejection:
            raise RuntimeError("bounded evaluator exited nonzero after rejection")
        return "running"
    monkeypatch.setattr(campaign, "job", job)
    assert campaign.cycle(args)["stage"] == "awaiting_author"
    assert campaign.cycle(args)["stage"] == ("rejected" if late_rejection else "measuring")
    assert "--worker-root /worker_root" in launches[0] and "--order " in launches[0]
    saved = campaign.cycle(args)
    assert campaign.cycle(args) == saved
    assert saved["stage"] == "rejected" and saved["order"] == row["order_id"] and saved["result"] == result
    assert saved["cost_window"]["requests_to_amortize_lease_time"] is None
    assert len(launches) == 1
    order.write_text(json.dumps({**row, "base_commit": "b" * 40}))
    with pytest.raises(ValueError, match="order or adjacent"):
        campaign.cycle(args)


def test_checkpoint_rejects_transfer_corruption_without_overwriting_completed_bytes(tmp_path, monkeypatch):
    expected = hashlib.sha256(b'{"valid":1}').hexdigest()
    monkeypatch.setattr(campaign, "shell", lambda *_: expected + "  /trial/build.json")
    monkeypatch.setattr(campaign, "call", lambda *a, **k: b"bad transfer")
    with pytest.raises(OSError, match="changed during transfer"):
        campaign.snapshot({}, Path("/trial"), ["build.json"], tmp_path)
    assert not (tmp_path / "build.json").exists()
    monkeypatch.setattr(campaign, "call", lambda *a, **k: b'{"valid":1}')
    campaign.snapshot({}, Path("/trial"), ["build.json"], tmp_path)
    monkeypatch.setattr(campaign, "shell", lambda *_: pytest.fail("download repeated"))
    campaign.snapshot({}, Path("/trial"), ["build.json"], tmp_path)
    assert json.loads((tmp_path / "build.json").read_bytes()) == {"valid": 1}


def test_http_option_stops_at_negative_public_qualification(tmp_path, monkeypatch):
    cfg = {k: "/" + k for k in campaign.CONFIG_KEYS}
    key, binding = "a" * 64, "b" * 64
    args = SimpleNamespace(state=tmp_path, serve_qualified=True, http_reference_url=None, http_serve_seconds=5,
                           http_port=8230, http_timeout=600)
    native = {"stage": "active", "receipt_sha256": "receipt", "pr": 9}
    campaign.save(tmp_path / "http-framework.json", {"checked": True})
    jobs = []
    def snapshot(pod, remote, names, local):
        if jobs and "http-qualify.json" in names:
            campaign.save(local / "http-qualify.json", {"binding": binding, "control_receipt": "receipt",
                                                        "decision": {"passed": False}})
    monkeypatch.setattr(campaign, "snapshot", snapshot)
    monkeypatch.setattr(campaign, "collect_http", lambda *_: binding)
    monkeypatch.setattr(campaign, "lease", lambda *_: {"deadline": 9e12})
    monkeypatch.setattr(campaign, "cost_window", lambda *args: {})
    def job(pod, name, cmd):
        jobs.append(name)
        return "running"
    monkeypatch.setattr(campaign, "job", job)
    assert campaign.published_http(args, cfg, {"order_id": key}, native)["stage"] == "http_qualify"
    result = campaign.published_http(args, cfg, {"order_id": key}, native)
    assert result["stage"] == "http_declined"
    assert campaign.published_http(args, cfg, {"order_id": key}, native) == result
    assert len(jobs) == 1
def test_busy_cpu_or_gpu_broker_defers_until_shared_host_is_available(tmp_path):
    import fcntl
    import os
    import subprocess

    cfg = {"worker_root": str(tmp_path)}
    lock = tmp_path / ".ironmule-measurement.lock"
    lock.mkdir(mode=0o700)
    completed = tmp_path / "ran"
    cmd = ["/bin/sh", "-ec", campaign.exclusive_measurement(cfg, ["touch", str(completed)], 1)]
    fd = os.open(lock, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert subprocess.run(cmd, check=False).returncode == 75
        assert not completed.exists()
    finally:
        os.close(fd)
    subprocess.run(cmd, check=True)
    assert completed.is_file()
