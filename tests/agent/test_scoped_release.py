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


def test_target_rejects_different_runtime_and_orphaned_code(tmp_path):
    source = tmp_path / release.SOURCE
    source.parent.mkdir()
    source.write_bytes(b"base")
    other = source.with_name("pilot.py")
    other.write_bytes(b"frozen pilot")
    receipt = {"runtime_manifest": {release.SOURCE: release.checksum(source), "ironmule/pilot.py": release.checksum(other)},
               "qualification": {"reference_sha256": release.checksum(source), "candidate_sha256": "0" * 64}}
    release.check_tree(receipt, tmp_path, False)
    with pytest.raises(ValueError, match="changed release input"):
        release.check_tree(receipt, tmp_path, True)
    extra = other.with_name("extra.pyc")
    extra.write_bytes(b"orphaned")
    with pytest.raises(ValueError, match="untracked import"):
        release.check_tree(receipt, tmp_path, False)
    extra.unlink()
    other.write_bytes(b"changed")
    with pytest.raises(ValueError, match="pilot.py"):
        release.check_tree(receipt, tmp_path, False)


def test_shadow_withdrawal_persists_only_equivalent_bound_work(tmp_path):
    path, sha, obs = pinned(tmp_path)
    state = tmp_path / "target"
    release.activate(state, path, sha, lambda _: obs)
    def run(shadow):
        arm = release.selected(shadow, sha)
        return {"arm": arm, "source_sha256": obs[arm + "_sha256"], "work_sha256": "same"}
    proof = release.exercise(state, sha, run)
    assert proof["before"]["arm"] == "candidate" and proof["after"]["arm"] == "reference"
    assert release.selected(state, sha) == "candidate"


@pytest.mark.parametrize("module", ["cpu", "qwen"])
def test_failed_selected_request_persists_withdrawal_and_serves_reference(tmp_path, monkeypatch, module):
    from argparse import Namespace
    from lab.pk1 import restart_release
    from lab.rp import cpu_code_release

    adapter = cpu_code_release if module == "cpu" else restart_release
    path, sha, obs = pinned(tmp_path)
    state = tmp_path / "target"
    release.activate(state, path, sha, lambda _: obs)
    calls = []
    def actual(_, receipt, arm, seed, directory):
        calls.append(arm)
        if arm == "candidate":
            raise ValueError("tested target drift")
        return {"runtime_source_sha256": {"ggml_engine.py": "base"}, "work_sha256": "work", "output_sha256": "work",
                "correct": 16, "completed": 16, "tokens": 48, "scratch_measured_decisions": 4, "wall_s": 1., "ttft_s": .1,
                "source_sha256": "base", "reused": False, "cold_s": 2., "cold": {"output_sha256": "x", "completed": 32},
                "warm": {"output_sha256": "y", "completed": 32, "wall_s": 1.}, "factual": ["mars"], "facts": [True] * 18,
                "pilot_decisions": 4}, 3.
    monkeypatch.setattr(adapter, "measured", actual)
    result = adapter.run(Namespace(receipt_sha256=sha), state, 111)
    assert calls == ["candidate", "reference"] and result["arm"] == "reference"
    calls.clear()
    after = adapter.run(Namespace(receipt_sha256=sha), state, 111)
    assert calls == ["reference"] and result["work_sha256"] == after["work_sha256"]


def test_projection_preserves_only_approved_source_bytes(tmp_path):
    tree = tmp_path / "base"
    source = tree / release.SOURCE
    source.parent.mkdir(parents=True)
    source.write_bytes(b"reference")
    cache = source.parent / "__pycache__"
    cache.mkdir()
    (cache / "ggml_engine.pyc").write_bytes(b"stale input")
    proposal = tmp_path / "optimized.py"
    proposal.write_bytes(b"candidate")
    observed = {"reference_sha256": release.checksum(source), "candidate_sha256": release.checksum(proposal)}
    receipt = {"runtime_manifest": {release.SOURCE: release.checksum(source)}, "qualification": observed,
               "publication": {"reference_tree": "a" * 40, "candidate_tree": "b" * 40}}
    target = release.projected_tree(receipt, tree, tmp_path, proposal)
    assert (target / release.SOURCE).read_bytes() == b"candidate" and not list(target.rglob("*.pyc"))
    assert (release.projected_tree(receipt, tree, tmp_path) / release.SOURCE).read_bytes() == b"reference"


def test_failed_rollback_exercise_withdraws_real_selection(tmp_path):
    path, sha, obs = pinned(tmp_path)
    state = tmp_path / "target"
    release.activate(state, path, sha, lambda _: obs)
    def wrong(shadow):
        arm = release.selected(shadow, sha)
        return {"arm": arm, "source_sha256": obs[arm + "_sha256"], "work_sha256": arm}
    with pytest.raises(ValueError, match="rollback did not"):
        release.exercise(state, sha, wrong)
    assert release.selected(state, sha) == "reference"
