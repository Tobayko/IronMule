"""Model-policy checks with synthetic metadata only; no inference is executed."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ironmule_product.model_policy import ModelPolicyError, validate_model_config


def _snapshot(tmp_path: Path, config: str = "{}") -> tuple[Path, str]:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "config.json").write_text(config, encoding="utf-8")
    return snapshot, "revision"


def test_safe_config_is_pinned_to_none(tmp_path: Path) -> None:
    snapshot, revision = _snapshot(tmp_path, '{"model_type":"gemma3","model_file":null}')
    assert validate_model_config(str(snapshot), revision) == {"model_file": None}


@pytest.mark.parametrize("value", [False, "", "custom.py", 0, [], {}])
def test_any_non_none_model_file_is_rejected(tmp_path: Path, value: object) -> None:
    snapshot, revision = _snapshot(tmp_path, json.dumps({"model_file": value}, separators=(",", ":")))
    with pytest.raises(ModelPolicyError, match="custom model Python"):
        validate_model_config(str(snapshot), revision)


@pytest.mark.parametrize("config", ["", "[]", "{\"model_file\": NaN}", '{"model_file":', '{"value":1e999}'])
def test_invalid_config_is_rejected_without_raw_body(tmp_path: Path, config: str) -> None:
    snapshot, revision = _snapshot(tmp_path, config)
    with pytest.raises(ModelPolicyError):
        validate_model_config(str(snapshot), revision)


def test_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    snapshot, revision = _snapshot(tmp_path, '{"model_file":null,"model_file":null}')
    with pytest.raises(ModelPolicyError):
        validate_model_config(str(snapshot), revision)


def test_oversized_config_is_rejected(tmp_path: Path) -> None:
    snapshot, revision = _snapshot(tmp_path)
    (snapshot / "config.json").write_bytes(b"{" + b"x" * (16 * 1024 * 1024) + b"}")
    with pytest.raises(ModelPolicyError):
        validate_model_config(str(snapshot), revision)


def test_deeply_nested_config_is_rejected_cleanly(tmp_path: Path) -> None:
    snapshot, revision = _snapshot(tmp_path, '{"x":' * 65 + "0" + "}" * 65)
    with pytest.raises(ModelPolicyError):
        validate_model_config(str(snapshot), revision)


def test_shallow_nested_config_is_accepted(tmp_path: Path) -> None:
    snapshot, revision = _snapshot(tmp_path, '{"x":{"y":1}}')
    assert validate_model_config(str(snapshot), revision) == {"model_file": None}


def test_hf_config_blob_symlink_inside_repository_root_is_allowed(tmp_path: Path) -> None:
    repository = tmp_path / "models--org--model"
    snapshot = repository / "snapshots" / "revision"
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (blobs / "config").write_text('{"model_type":"gemma3"}', encoding="utf-8")
    (blobs / "model").write_bytes(b"weights")
    (snapshot / "config.json").symlink_to(blobs / "config")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to(blobs / "model")
    (snapshot / "tokenizer.model").write_bytes(b"tok")
    assert validate_model_config(str(snapshot), "revision") == {"model_file": None}


def test_hf_config_symlink_escape_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "models--org--model"
    snapshot = repository / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (snapshot / "config.json").symlink_to(outside)
    with pytest.raises(ModelPolicyError):
        validate_model_config(str(snapshot), "revision")


def test_config_fifo_is_rejected_before_open(tmp_path: Path) -> None:
    snapshot, revision = _snapshot(tmp_path)
    (snapshot / "config.json").unlink()
    os.mkfifo(snapshot / "config.json")
    with pytest.raises(ModelPolicyError):
        validate_model_config(str(snapshot), revision)


def test_worker_reports_typed_policy_error_before_mlx_import(tmp_path: Path) -> None:
    snapshot, revision = _snapshot(tmp_path, '{"model_file":false}')
    spec = {"model_id": "local/test", "revision": revision, "snapshot_path": str(snapshot), "weight_bytes": 7}
    worker = Path(__file__).resolve().parents[2] / "ironmule_product" / "worker.py"
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-u", str(worker), "--spec", json.dumps(spec)],
        capture_output=True, timeout=3, check=False,
    )
    assert result.returncode == 1
    event = json.loads(result.stdout)
    assert event["code"] == "unsupported_model_code"
    assert b"mlx" not in result.stderr.lower()


def test_custom_model_marker_is_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    snapshot, revision = _snapshot(tmp_path, json.dumps({"model_file": "custom.py"}))
    (snapshot / "custom.py").write_text(
        f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    spec = {"model_id": "local/test", "revision": revision, "snapshot_path": str(snapshot), "weight_bytes": 7}
    worker = Path(__file__).resolve().parents[2] / "ironmule_product" / "worker.py"
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-u", str(worker), "--spec", json.dumps(spec)],
        capture_output=True, timeout=3, check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["code"] == "unsupported_model_code"
    assert not marker.exists()


def test_worker_load_pins_guarded_model_config_argument() -> None:
    worker_source = (Path(__file__).resolve().parents[2] / "ironmule_product" / "worker.py").read_text(encoding="utf-8")
    tree = ast.parse(worker_source)
    load_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "load"]
    assert load_calls
    assert any(any(keyword.arg == "model_config" and isinstance(keyword.value, ast.Name)
                   and keyword.value.id == "model_config" for keyword in call.keywords)
               for call in load_calls)
