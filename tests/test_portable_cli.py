"""Bounded DATA1 CLI, contract, credential, and bundle regressions.

These tests exercise control-plane code only.  The NumPy files are synthetic
control fixtures and do not support hardware, model, or performance claims.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from friday_evidence.portable.bundle import prepare_bundle
from friday_evidence.portable.cli import main
from friday_evidence.portable.contracts import ContractError, load_json, make_spec, validate_spec
from friday_evidence.portable.credentials import configured_kaggle_credentials
from friday_evidence.portable.kaggle_probe import _source


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(root: Path, *, partition: str = "train", suffix: str = "1") -> dict:
    left = root / f"a-{suffix}.npy"
    right = root / f"b-{suffix}.npy"
    np.save(left, np.eye(2, dtype=np.float32), allow_pickle=False)
    np.save(right, np.eye(2, dtype=np.float32), allow_pickle=False)
    return {
        "case_id": f"case-{suffix}", "a_file": left.name, "b_file": right.name,
        "a_sha256": _sha(left), "b_sha256": _sha(right), "shape": [2, 2, 2],
        "dtype": "float32", "lineage_id": f"lineage-{suffix}",
        "weight_sha256": "1" * 64, "prompt_family": f"prompt-{suffix}",
        "shape_family": "tiny-2", "capture_session": f"capture-{suffix}",
        "model_id": "org/model", "model_sha256": "2" * 64,
        "source_kind": "real_model_capture", "partition": partition,
        "export_policy": "private_public_workload_only",
    }


def test_capture_defaults_to_plan_without_hardware(tmp_path: Path, capsys) -> None:
    assert main(["--state-dir", str(tmp_path / "state"), "capture", "--partition", "train"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "planned"
    assert result["hardware_started"] is False
    assert not (tmp_path / "state").exists()


def test_run_defaults_to_plan_without_worker(tmp_path: Path, capsys) -> None:
    spec = make_spec([_case(tmp_path)], "mlx")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    assert main(["--state-dir", str(tmp_path / "state"), "run", "--spec", str(spec_path),
                 "--data-dir", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "planned" and result["hardware_started"] is False
    assert not (tmp_path / "state").exists()


def test_cli_help_is_available_without_dispatch() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0


@pytest.mark.parametrize(("field", "value"), [("warmup", 4), ("pairs", 11), ("server_seconds", 0),
                                                ("work_seconds", 180)])
def test_spec_bounds_are_fail_closed(tmp_path: Path, field: str, value: int) -> None:
    spec = make_spec([_case(tmp_path)], "mlx")
    spec[field] = value
    with pytest.raises(ContractError, match=f"invalid_{field}|cleanup_margin"):
        validate_spec(spec)


def test_duplicate_cases_and_mixed_partition_are_rejected(tmp_path: Path) -> None:
    first = _case(tmp_path, suffix="1")
    spec = make_spec([first], "mlx")
    spec["cases"] = [first, dict(first)]
    with pytest.raises(ContractError, match="duplicate_case"):
        validate_spec(spec)
    other = _case(tmp_path, partition="validation", suffix="2")
    spec["cases"] = [first, other]
    with pytest.raises(ContractError, match="duplicate_case_or_mixed_partition"):
        validate_spec(spec)


def test_duplicate_json_keys_and_symlink_inputs_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"ironmule.experiment.v1","schema":"other"}', encoding="utf-8")
    with pytest.raises(ContractError, match="duplicate_json_key"):
        load_json(duplicate)
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ContractError, match="json_invalid_or_too_large"):
        load_json(link)


def test_malformed_numpy_metadata_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    np.save(tmp_path / case["a_file"], np.eye(2, dtype=np.float64), allow_pickle=False)
    case["a_sha256"] = _sha(tmp_path / case["a_file"])
    from friday_evidence.portable.runner import RunnerError, _load_operands

    with pytest.raises(RunnerError, match="dtype|metadata|float32"):
        _load_operands(case, tmp_path)


def test_credentials_use_only_synthetic_token_and_restore_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[mcp_servers.kaggle]\nurl = "https://www.kaggle.com/mcp"\nbearer_token_env_var = "TEST_KGAT"\n', encoding="utf-8")
    token = "KGAT_" + "x" * 16
    monkeypatch.setenv("TEST_KGAT", token)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    with configured_kaggle_credentials(config):
        assert os.environ["KAGGLE_API_TOKEN"] == token
    assert "KAGGLE_API_TOKEN" not in os.environ
    assert token not in config.read_text(encoding="utf-8")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["config.toml"]


def test_bundle_is_private_offline_and_entry_bootstraps_inside_main(tmp_path: Path) -> None:
    case = _case(tmp_path, suffix="1")
    spec = make_spec([case], "cuda")
    destination = tmp_path / "bundle"
    result = prepare_bundle(spec, tmp_path, destination, owner="owner")
    metadata = load_json(Path(result["notebook_dir"]) / "kernel-metadata.json")
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == ["owner/data1-" + spec["run_id"]]
    entry = Path(result["notebook_dir"]) / "data1.py"
    tree = ast.parse(entry.read_text(encoding="utf-8"))
    guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and isinstance(node.test, ast.Compare)
              and any(isinstance(part, ast.Constant) and part.value == "__main__"
                      for part in ast.walk(node.test))]
    assert len(guards) == 1
    assert any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body)
    assert "subprocess" not in entry.read_text(encoding="utf-8")


def test_provider_smoke_defaults_to_plan_and_uploads_no_model_payload(tmp_path: Path, capsys) -> None:
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}", encoding="utf-8")
    assert main(["--state-dir", str(tmp_path / "state"), "provider-smoke",
                 "--backend", "cuda", "--owner", "owner",
                 "--accelerator", "NvidiaTeslaT4",
                 "--account-preflight", str(preflight)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "planned"
    assert result["hardware_started"] is False
    assert not (tmp_path / "state").exists()
    source = _source("cuda", "a" * 32)
    compile(source, "probe.py", "exec")
    assert "mlx-community" not in source
    assert "/kaggle/input" not in source
    assert "performance_claim\":False" in source
