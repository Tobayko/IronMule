"""A production selector binds measured code, a reviewed tree and a fresh target observation."""
from copy import deepcopy
import json

import pytest

from training import scoped_release as release


def order(adapter):
    base, method = release.ADAPTERS[adapter]
    target = {"path": release.SOURCE, "class": "GgmlEngine", "function": method.split(".")[1]}
    obj = {"target": target, "allowed_paths": [release.SOURCE], "base_commit": base, "hypothesis": "Measured hotspot"}
    return {**obj, "order_id": release.digest(obj)}


def publication(obj):
    return {"order_id": obj["order_id"], "base": obj["base_commit"], "pr": 900,
            "branch": "qualification/" + obj["order_id"][:16]}


def observation():
    return {"passed": True, "reference_sha256": "1" * 64, "candidate_sha256": "2" * 64,
            "evaluator_sha256": "3" * 64, "binding": {"target": "fixture", "context": 512},
            "metrics": {"correct_per_arm": [96, 96]}}


def pinned(tmp_path):
    obs, obj = observation(), order("epyc-prefill")
    receipt = {"schema": release.SCHEMA, "adapter": "epyc-prefill", "order": obj,
               "publication": publication(obj), "qualification": obs}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return path, release.checksum(path), obs


def test_target_restart_precise_scope_and_withdrawal(tmp_path):
    path, sha, obs = pinned(tmp_path)
    state = tmp_path / "target"
    assert release.selected(state, sha) == "reference"
    assert release.activate(state, path, sha, lambda _: obs)["arm"] == "candidate"
    assert release.selected(state, sha) == "candidate"
    assert release.selected(state, "d" * 64) == "reference"
    # A new process sees the atomic state and its independently checksummed check.
    altered = deepcopy(obs)
    altered["binding"]["context"] = 1024
    assert release.activate(state, path, sha, lambda _: altered)["withdrawn"]
    assert release.selected(state, sha) == "reference"
    assert release.activate(state, path, sha, lambda _: pytest.fail("withdrew"))["arm"] == "reference"


def test_inflight_check_cannot_reenable_withdrawn_candidate(tmp_path):
    path, sha, obs = pinned(tmp_path)
    state = tmp_path / "target"
    def recheck(_):
        release.rollback(state, "runtime regression")
        return obs
    assert release.activate(state, path, sha, recheck)["withdrawn"]
    assert release.selected(state, sha) == "reference"


def test_scope_and_partial_target_receipt_fail_closed(tmp_path):
    path, sha, obs = pinned(tmp_path)
    state = tmp_path / "target"
    with pytest.raises(ValueError, match="transferred"):
        release.activate(state, path, "0" * 64, lambda _: obs)
    def failed(_):
        raise TimeoutError("failed probe")
    assert release.activate(state, path, sha, failed)["withdrawn"]
    another = tmp_path / "other"
    release.activate(another, path, sha, lambda _: obs)
    (another / "deployment/target-check.json").write_text("{}")
    assert release.selected(another, sha) == "reference"


def test_publish_reconstructs_exact_method_and_requires_merged_tree(tmp_path, monkeypatch):
    from training import code_trial
    repo = tmp_path / "repo"
    repo.mkdir()
    original = repo / release.SOURCE
    original.parent.mkdir()
    original.write_text("class GgmlEngine:\n    def __init__(self):\n        self.n = 1\n")
    release.git("init", "--quiet", cwd=repo)
    release.git("add", ".", cwd=repo)
    release.git("-c", "user.name=Fixture", "-c", "user.email=fixture@local", "commit", "-qm", "base", cwd=repo)
    base = release.git("rev-parse", "HEAD", cwd=repo)
    monkeypatch.setattr(code_trial, "ROOT", repo)
    adapter = "qwen-restart"
    monkeypatch.setitem(release.ADAPTERS, adapter, (base, "GgmlEngine.__init__"))
    obj = order(adapter)
    source = tmp_path / "candidate.py"
    source.write_text(original.read_text().replace("self.n = 1", "self.n = 2"))
    obs = {**observation(), "candidate_sha256": release.checksum(source), "reference_sha256": release.checksum(original)}
    kw = {"adapter": adapter, "verify": lambda: obs, "rationale": "Bound startup"}
    state = tmp_path / "control"
    before = release.publish(state, obj, source, **kw)
    assert before["status"] == "qualified, awaiting publication"
    monkeypatch.setattr(release, "submit_candidate", lambda *args: 900)
    monkeypatch.setattr(release, "merged_source", lambda *a, **kw: {**publication(obj), "candidate_tree": "other"})
    with pytest.raises(ValueError, match="merged tree"):
        release.publish(state, obj, source, **kw, submit=True)
    monkeypatch.setattr(release, "merged_source", lambda *a, **kw: {**publication(obj), "candidate_tree": before["tree"]})
    result = release.publish(state, obj, source, **kw, submit=True)
    assert release.pinned(state, result["receipt_sha256"])["publication"]["candidate_tree"] == before["tree"]
    source.write_text(source.read_text() + "other_function = True\n")
    obs["candidate_sha256"] = release.checksum(source)
    with pytest.raises(ValueError, match="ordered method"):
        release.publish(state, obj, source, **kw)
