import ast
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location("q3_build_replay", ROOT / "research/q3_build_replay.py")
q3 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(q3)


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _q2_fixture(tmp_path):
    trials = []
    for name, values in q3._load_contracts()[1].SEARCH_VALUES:
        for value in values:
            trials.append({"knob": name, "value": value, "disposition": "accepted" if name in {"compiled_fixed_cache", "head_skip_prefill", "readback_every"} and value in {True, 2} else "rejected", "total_ns": 100, "prefill_ns": 60, "decode_ns": 40})
    # The sequential path only keeps accepted values; make the fixture match the real Q2 path.
    accepted = {"compiled_fixed_cache", "head_skip_prefill", "readback_every"}
    for trial in trials:
        if trial["knob"] == "readback_every":
            trial["disposition"] = "accepted" if trial["value"] == 2 else "rejected"
        elif trial["knob"] in accepted:
            trial["disposition"] = "accepted"
    profile = {
        "baseline_ns": 100, "baseline_prefill_ns": 60, "baseline_decode_ns": 40,
        "conditions": {"mlx": "0.32.0", "mlx_lm": "0.31.3", "runtime_version": "0.1.0", "os": "test", "prompt_tokens": 322, "max_tokens": 32},
        "hardware": {"chip": "test"}, "model_id": "test/model",
        "model_identity": {"model_manifest_sha256": "a" * 64},
        "knobs": {"fuse_projections": False, "compiled_fixed_cache": True, "fused_argmax": False, "head_skip_prefill": True, "prefill_into_fixed": False, "readback_every": 2, "speculate_k": 0, "speculate_ngram": 3, "capacity_slack": 0, "wired_fraction": 0.0},
        "trials": trials, "tuned_at": 1.0,
    }
    profile_path = tmp_path / "Q2_profiles.json"
    log_path = tmp_path / "Q2_run.log"
    profile_path.write_text(json.dumps({"profile": profile}))
    log_path.write_text("confirmed ratio tokens identical True\n")
    return profile_path, log_path, q3._sha256(profile_path), q3._sha256(log_path)


def _b36_fixture(tmp_path):
    model = "b" * 64
    workload = {"max_tokens": 32, "prompt_tokens": 322, "prompt_sha256": "c" * 64, "prompt_token_sha256": "d" * 64, "eos_ids": [1, 106]}
    environment = {"chip": "Test M1", "hardware_fingerprint": "test-hw", "memory_bytes": 32, "gpu_cores": 8, "mlx": "0.32.0", "mlx_lm": "0.31.3", "python": "3.12", "os": "test", "git_commit": "e" * 40, "power_source": "AC", "low_power_mode": False}
    measured = [{"total_ns": 100 + i, "prefill_ns": 60 + i, "decode_ns": 40 + i} for i in range(5)]
    warmups = [{"total_ns": 110 + i, "prefill_ns": 70 + i, "decode_ns": 40 + i} for i in range(2)]
    checkpoints = [{"timestamp_ns": 10 + i, "rss_bytes": 30, "mlx": {"active": 20, "peak": 25}} for i in range(2)]

    def child(arm, pair_index):
        return {"schema": "ironmule.b36.child.v1", "arm": arm, "pair_index": pair_index, "measured": measured, "warmups": warmups, "checkpoints": checkpoints, "environment": environment, "workload": workload, "model_manifest_digest": model, "preregistration_sha256": q3.B36_PREREGISTRATION_SHA256, "b36a_preregistration_sha256": q3.B36A_PREREGISTRATION_SHA256, "code_digest": q3.B36_CODE_DIGEST, "returncode": 0, "crashed": False, "no_crash": True, "identity_gate": True, "canonical_correctness_gate": True, "post_evidence_complete": True}

    gate = {"complete": True, "identity": True, "no_crash": True, "peak_memory": True, "swap": True, "timings": True, "token_identity": True}
    pairs = []
    for i in range(16):
        order = ["baseline", "candidate"] if i < 8 else ["candidate", "baseline"]
        pairs.append({"order": order, "children": [child(arm, i) for arm in order], "pair_result": {"status": "ok", "token_identity": True, "hard_gates": gate}})
    raw = {"schema": "ironmule.b36.v1", "status": "complete", "constants": {"max_tokens": 32, "no_retry": True, "pairs": 16, "repeats": 5, "warmups": 2}, "preregistration": {"sha256": q3.B36_PREREGISTRATION_SHA256}, "b36a_preregistration": {"sha256": q3.B36A_PREREGISTRATION_SHA256}, "code_identity": {"digest": q3.B36_CODE_DIGEST}, "model_binding": {"digest": model}, "summary": {"valid_pairs": 16}, "pairs": pairs}
    path = tmp_path / "B36.json"
    path.write_text(json.dumps(raw))
    return path, q3._sha256(path)


def test_real_shaped_fixture_builds_without_runtime_or_model_imports(tmp_path):
    profile, log, profile_sha, log_sha = _q2_fixture(tmp_path)
    b36, b36_sha = _b36_fixture(tmp_path)
    dataset = q3.build_dataset(profile, log, b36, profile_sha, log_sha, b36_sha)
    assert len(dataset.observations) == 14
    assert len(dataset.action_pool) == 12
    assert dataset.coverage_report()["raw_sample_count"] == 160
    methods = dataset.method_eligibility()["methods"]
    assert methods["BASELINE"]["status"] == "STRUCTURALLY_ELIGIBLE"
    for name in ("CURRENT_COORDINATE", "SEEDED_RANDOM", "BO", "SURROGATE", "CONTEXTUAL_BANDIT"):
        assert methods[name]["status"] == "DATA_INSUFFICIENT"
    assert methods["OFFLINE_RL"]["status"] == "NOT_APPLICABLE"


def test_script_has_no_forbidden_imports_and_hash_failures_are_fail_closed(tmp_path):
    tree = ast.parse((ROOT / "research/q3_build_replay.py").read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    rendered = "\n".join(ast.unparse(node) for node in imports)
    assert not any(word in rendered for word in ("mlx", "runtime", "tune"))
    profile, log, profile_sha, log_sha = _q2_fixture(tmp_path)
    b36, b36_sha = _b36_fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        q3.build_dataset(profile, log, b36, "0" * 64, log_sha, b36_sha)


def test_execute_is_explicit_private_and_non_overwriting(tmp_path):
    payload = b"{}\n"
    output = tmp_path / "replay.json"
    q3._write_exclusive(output, payload)
    assert output.read_bytes() == payload
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        q3._write_exclusive(output, payload)


def test_cli_dry_run_does_not_write_and_execute_refuses_overwrite(tmp_path, capsys):
    profile, log, profile_sha, log_sha = _q2_fixture(tmp_path)
    b36, b36_sha = _b36_fixture(tmp_path)
    output = tmp_path / "dataset.json"
    args = ["--q2-profile", str(profile), "--q2-log", str(log), "--b36", str(b36), "--q2-profile-sha256", profile_sha, "--q2-log-sha256", log_sha, "--b36-sha256", b36_sha, "--output", str(output)]
    assert q3.main(args) == 0
    assert not output.exists()
    capsys.readouterr()
    assert q3.main(args + ["--execute"]) == 0
    assert output.exists()
    assert q3.main(args + ["--execute"]) == 2
