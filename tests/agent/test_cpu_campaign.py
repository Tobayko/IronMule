import hashlib
import io
import tarfile
from types import SimpleNamespace

import pytest

from lab.rp import cpu_campaign as campaign
from lab.rp import cpu_campaign_stage as staging


def sig(data):
    return hashlib.sha256(data).hexdigest()


def test_complete_archives_are_immutable_on_replay(tmp_path):
    archive = tmp_path / "all.tar.gz"
    ref, new = b"def old(): pass", b"def new(): pass"
    entries = {staging.SOURCE: ref, "tools/gguf_from_mlx.py": b"helper", "metadata": b"full export"}
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in entries.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    candidate = tmp_path / "source.py"
    candidate.write_bytes(new)
    lane = tmp_path / "lane"
    lane.mkdir()
    args = {"reference_sha": sig(ref), "candidate_sha": sig(new)}
    staging.stage(archive, candidate, lane, **args)
    staging.stage(archive, candidate, lane, **args)
    assert (lane / "candidate" / staging.SOURCE).read_bytes() == new
    assert (lane / "base" / staging.SOURCE).read_bytes() == ref
    assert (lane / "candidate/tools/gguf_from_mlx.py").read_bytes() == b"helper"
    (lane / "candidate/surprise.py").touch()
    with pytest.raises(ValueError, match="complete archive"):
        staging.stage(archive, candidate, lane, **args)
    (lane / "candidate/surprise.py").unlink()
    updated = lane / "candidate/metadata"
    updated.chmod(0o644)
    updated.write_text("foreign")
    with pytest.raises(ValueError, match="source differs"):
        staging.stage(archive, candidate, lane, **args)
    assert updated.read_text() == "foreign"


def test_reject_symlink_in_archive(tmp_path):
    archive = tmp_path / "all.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        entry = tarfile.TarInfo("traversal")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "/etc"
        tar.addfile(entry)
    candidate = tmp_path / "new"
    candidate.write_bytes(b"src")
    with pytest.raises(ValueError, match="unsupported entry"):
        staging.stage(archive, candidate, tmp_path, reference_sha=sig(b""), candidate_sha=sig(b"src"))


@pytest.mark.parametrize("named_job_running", [True, False])
def test_adopted_job_or_existing_process_blocks_duplicate_dispatch(tmp_path, monkeypatch, named_job_running):
    key = "e" * 64
    state = tmp_path / "state"
    state.mkdir()
    cfg = {"lane": "/opt/ironmule-worker/cpu-eval", "pod_id": "pod", "framework": "/framework",
           "python": "/python", "gguf": "/model", "worker_root": "/workers"}
    order = {"schema": "measured", "order_id": key, "binding": {"affinity": [0, 1, 2, 3]}}
    trees = []
    for arm in ("reference", "candidate"):
        path = tmp_path / arm
        (path / "ironmule").mkdir(parents=True)
        (path / campaign.SOURCE).write_text(arm)
        trees.append(path)
    monkeypatch.setattr(campaign, "target", lambda _: cfg)
    monkeypatch.setattr(campaign, "load_order", lambda _: order)
    monkeypatch.setattr(campaign, "make_source", lambda *_: trees)
    monkeypatch.setattr(campaign.transport, "lease", lambda _: {})
    monkeypatch.setattr(campaign, "state_of_job", lambda *_: {"running": named_job_running, "rc": 1, "tail": ""})
    monkeypatch.setattr(campaign, "active_qualifier", lambda *_: True)
    monkeypatch.setattr(campaign, "snapshot_trial", lambda pod, remote, raw, suite: raw.mkdir())
    monkeypatch.setattr(campaign, "stage", lambda *_: pytest.fail("attempted a duplicate dispatch"))
    name = "epyc-qualify-" + key[:16] + "-quiet"
    args = SimpleNamespace(state=state, target=tmp_path, authorship=None, author_cli=False, job_name=name, matched=False, prepare_only=False)
    assert campaign.advance(args) == {"stage": "measuring", "job": name, "completed_arms": 0}


def test_lease_deadline_and_single_writer_single_core_set(monkeypatch):
    monkeypatch.setattr(campaign.time, "time", lambda: 50)
    cfg = {"env": "/etc/lane.env", "framework": "/repo", "worker_root": "/workers"}
    cmd = campaign.job_command(cfg, {"deadline": 100}, ["/python", "-m", "arm"], [0, 1, 2, 3], "/runs/one.lock", 120)
    assert "timeout -k 5 40" in cmd and "--wait 39" in cmd
    assert "/workers/.ironmule-measurement.lock flock -n /runs/one.lock taskset -c 0,1,2,3 /python" in cmd
    with pytest.raises(ValueError, match="deadline"):
        campaign.job_command(cfg, {"deadline": 52}, [], [], "/runs/one.lock", 120)


def test_actual_v1_confounded_arm_receipts_never_reach_a_target(tmp_path, monkeypatch):
    from shutil import copytree
    archived = campaign.ROOT / "lab/autonomous/results/runpod-epyc-pilot"
    raw = tmp_path / "qualification"
    copytree(archived / "prefill-code/qualification-v1", raw)
    # Recreate the frozen incident report solely from the two arm outputs.
    expected = campaign.read(raw / "comparison-rejected.json")
    (raw / "comparison-rejected.json").unlink()
    monkeypatch.setattr(campaign, "target", lambda _: {"lane": "/unused"})
    monkeypatch.setattr(campaign.transport, "lease", lambda *_: pytest.fail("restarted a closed experiment"))
    args = SimpleNamespace(state=tmp_path, target=None, matched=False, order=archived / "prefill-order.json")
    assert campaign.advance(args) == {"stage": "confounded", "result": expected}
    assert campaign.advance(args)["result"]["training_label"] is False


def test_materialized_inputs_compare_before_and_after_upload(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    monkeypatch.setattr(campaign.transport, "shell", lambda _, cmd: subprocess.check_output(["bash", "-ec", cmd], text=True).strip())
    monkeypatch.setattr(campaign.transport, "upload", lambda _, path, payload: Path(path).write_bytes(payload))
    dst = tmp_path / "input with spaces"
    campaign.put_once(None, dst, b"first")
    campaign.put_once(None, dst, b"first")
    with pytest.raises(ValueError, match="different target contents"):
        campaign.put_once(None, dst, b"second")
    assert dst.read_bytes() == b"first"


def test_v2_is_distinct_and_collects_matched_rule_seeded_receipts():
    v1, v2 = campaign.Suite(), campaign.Suite(True)
    assert v1.suffix != v2.suffix
    assert v2.seed == 18351
    assert "rule-18356" in v2.arms()
    assert len(v2.arms()) == 3 * campaign.PAIRS
    assert set(v2.arms()).isdisjoint(v1.arms())


def test_product_withdrawal_recovers_delivered_batch_without_restart(tmp_path, monkeypatch):
    from pathlib import Path, PurePosixPath

    from training.code_trial import save
    state = tmp_path / "campaign"
    output = state / "target-release"
    output.mkdir(parents=True)
    req = tmp_path / "test.json"
    req.write_text('{"prompts": ["x"], "max_tokens": 3}')
    key = "0"*64
    name = "epyc-product-run-" + key[:16] + "-" + sig(req.read_bytes())[:12] + ".json"
    save(output / name, {"arm": "reference", "work": {"answers": [{"text": "x"}]}})
    monkeypatch.setattr(campaign.transport, "snapshot", lambda *_: None)
    monkeypatch.setattr(campaign, "release_framework", lambda *_: pytest.fail("rescheduled withdrawn target"))
    result = campaign.product(None, {}, state, PurePosixPath("/tmp/lane"), key, [], [], {"stage": "withdrawn",
                              "selection": {"withdrawn": True}}, req, campaign.Suite(), Path("/unused"))
    assert result["selection"]["withdrawn"]
    assert result["product_arm"] == "reference"
    assert result["response"] == str(output / name)
