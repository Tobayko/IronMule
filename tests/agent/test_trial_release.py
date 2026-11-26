"""Exercise merged-tree verification, atomic checkout, regression withdrawal and restart offline."""

import hashlib
import json
from pathlib import Path

import pytest

from training import trial_release as rel


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    source = tmp_path / "remote"
    source.mkdir()
    def git(*args):
        return rel.git(*args, cwd=source)
    git("init", "-q")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Fixture")
    (source / "ironmule").mkdir()
    for name in rel.SOURCES:
        (source / "ironmule" / name).write_text("# reference\n")
    git("add", ".")
    git("commit", "-qm", "reference")
    base = git("rev-parse", "HEAD")
    target = "ironmule/ggml_engine.py"
    binding = {"provider": "cpu", "model_sha256": "model", "native": {"sha256": "native"}, "machine": "local-test",
               "cores": 4, "pilot_weights_sha256": "pilot"}
    order = {"schema": rel.ORDER, "evaluator": rel.EVALUATOR, "target": {"path": target},
             "base_commit": base, "binding": binding}
    order["order_id"] = rel.digest(order)
    branch = "qualification/" + order["order_id"][:16]
    git("switch", "-qc", branch)
    (source / target).write_text("# change\n")
    git("add", target)
    git("commit", "-qm", "candidate")
    candidate = git("rev-parse", "HEAD")
    git("update-ref", "refs/pull/9/head", candidate)
    meta = {"merged": True, "merged_at": "test", "merge_commit_sha": candidate, "head": {"sha": candidate},
            "base": {"repo": {"full_name": rel.REPOSITORY}, "ref": branch}}
    monkeypatch.setattr(rel, "URL", str(source))
    monkeypatch.setattr(rel, "run", lambda argv: json.dumps(meta))
    w = {"clients": 4, "waves": 4, "max_tokens": 3, "warmup_waves": 1, "context": 512, "phases": ["short", "document"]}
    sha = hashlib.sha256((rel.ROOT / "training/serving_trial.py").read_bytes()).hexdigest()
    runs = {}
    def row(commit, seed):
        return {"binding": binding, "schema": rel.EVALUATOR, "profiled": False, "evaluator_sha256": sha,
                "runtime_source_sha256": rel.source_hashes(source, commit), "pilot_decisions": 17,
                "workload": {**w, "seed": seed}, "completed": 16, "waves": [{}] * 4, "correct": 16, "tokens": 48,
                "output_sha256": "a" * 64, "wall_s": runs.get(commit, 1.1 if commit == base else 1),
                "ttft_s": 0.1, "peak_rss_kib": 1024}
    monkeypatch.setattr(rel, "measure", lambda tree, args, seed, parent: row(tree.name, seed))
    rel.save(state / "context.json", {"provider": "cpu", "base": base})
    rel.save(state / "result.json", {"passed": True})
    for i in range(rel.PAIRS):
        rel.save(state / f"pair-{i}.json", {"reference": row(base, 8351 + i), "candidate": row(candidate, 8351 + i),
                                          "reference_tree": git("rev-parse", base + "^{tree}"),
                                          "candidate_tree": git("rev-parse", candidate + "^{tree}")})
    return state, base, candidate, meta, order, runs


def test_real_git_promotion_regression_and_restart(campaign):
    state, base, candidate, _, order, runs = campaign
    model = Path("fixture")
    assert rel.activate(state, order, 9, model)["active"] == candidate
    active = state / "deployment/active"
    assert (active / "ironmule/ggml_engine.py").read_text() == "# change\n"
    assert rel.check(state, model)["active"] == candidate
    runs[candidate] = 1.5
    assert rel.check(state, model)["active"] == base
    # Simulate interruption after the durable withdrawal but before the pointer update.
    active.unlink()
    active.symlink_to(candidate, target_is_directory=True)
    assert rel.check(state, model)["withdrawn"]
    assert active.readlink() == Path(base)
    assert (active / "ironmule/ggml_engine.py").read_text() == "# reference\n"
    assert rel.activate(state, order, 9, model)["withdrawn"]


@pytest.mark.parametrize("mutation", ["different_machine", "different_tree", "wrong_branch", "other_source"])
def test_mixed_evidence_or_unreviewed_tree_cannot_activate(campaign, mutation):
    state, _, _, meta, order, _ = campaign
    pair = rel.read(state / "pair-3.json")
    if mutation == "different_machine":
        for arm in ("reference", "candidate"):
            pair[arm]["binding"]["machine"] = "other"
    elif mutation == "different_tree":
        pair["candidate_tree"] = "f" * 40
    elif mutation == "wrong_branch":
        meta["base"]["ref"] = "main"
    else:
        pair["candidate"]["runtime_source_sha256"]["ggml_engine.py"] = "a" * 64
    rel.save(state / "pair-3.json", pair)
    with pytest.raises(ValueError):
        rel.activate(state, order, 9, Path("fixture"))
    assert not (state / "deployment/active").exists()


def test_evaluator_failure_restores_working_reference(campaign, monkeypatch):
    state, base, _, _, order, _ = campaign
    rel.activate(state, order, 9, Path("fixture"))
    def fail(*args):
        raise TimeoutError("fixture timeout")
    monkeypatch.setattr(rel, "measure", fail)
    assert rel.check(state, Path("fixture"))["withdrawn"]
    assert (state / "deployment/active").readlink() == Path(base)


def test_qualified_workload_consumes_active_or_withdrawn_revision(campaign):
    state, base, candidate, _, order, _ = campaign
    rel.activate(state, order, 9, Path("fixture"))
    release = rel.read(state / "deployment/release.json")
    assert rel.run_active(state, Path("fixture"), 93)["runtime_source_sha256"] == release["candidate_hashes"]
    rel.rollback(state)
    assert rel.run_active(state, Path("fixture"), 94)["runtime_source_sha256"] == release["reference_hashes"]
