"""Model-free identity contracts, synthetic boundary files and a cached-snapshot check."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess

import pytest

import friday_evidence.identity as identity_module
from friday_evidence.identity import (
    IdentityError,
    MAX_METADATA_FILE_BYTES,
    MAX_TOKENIZER_FILE_BYTES,
    assert_model_unchanged,
    runtime_identity,
)


def _snapshot(tmp_path: Path, *, model_bytes: bytes = b"weights") -> tuple[dict[str, object], Path]:
    path = tmp_path / "snapshot"
    path.mkdir()
    (path / "model.safetensors").write_bytes(model_bytes)
    (path / "config.json").write_text(json.dumps({"model_type": "test"}), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "test"}), encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return {"model_id": "local/test", "revision": "rev", "snapshot_path": str(path), "weight_bytes": len(model_bytes)}, path


def test_runtime_identity_binds_model_files_and_hides_absolute_paths(tmp_path: Path) -> None:
    spec, _ = _snapshot(tmp_path)
    identity = runtime_identity(spec, {"chip": "Test Chip", "memory_total_bytes": 1234})
    assert identity["model_id"] == "local/test"
    assert identity["revision"] == "rev"
    assert identity["model_files"][0]["name"] == "model.safetensors"
    assert identity["model_files"][0]["size"] == 7
    assert len(identity["model_sha256"]) == 64
    assert len(identity["identity_sha256"]) == 64
    assert str(tmp_path) not in json.dumps(identity)
    assert identity["hardware"]["chip"] == "Test Chip"


def test_assert_model_unchanged_detects_content_change(tmp_path: Path) -> None:
    spec, path = _snapshot(tmp_path)
    identity = runtime_identity(spec)
    assert_model_unchanged(spec, identity)
    (path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(IdentityError, match="model identity changed"):
        assert_model_unchanged(spec, identity)


def test_spec_and_weight_registration_are_exact(tmp_path: Path) -> None:
    spec, _ = _snapshot(tmp_path)
    with pytest.raises(IdentityError, match="exactly"):
        runtime_identity({**spec, "extra": 1})
    with pytest.raises(IdentityError, match="weight bytes"):
        runtime_identity({**spec, "weight_bytes": 99})


def test_hf_blob_symlink_inside_repository_root_is_allowed(tmp_path: Path) -> None:
    repository = tmp_path / "models--org--model"
    snapshot = repository / "snapshots" / "rev"
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (blobs / "model").write_bytes(b"blob")
    (snapshot / "model.safetensors").symlink_to(blobs / "model")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.model").write_bytes(b"tok")
    spec = {"model_id": "org/model", "revision": "rev", "snapshot_path": str(snapshot), "weight_bytes": 4}
    identity = runtime_identity(spec)
    assert identity["model_files"][0]["name"] == "model.safetensors"


def test_snapshot_symlink_escape_is_rejected(tmp_path: Path) -> None:
    spec, path = _snapshot(tmp_path)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"weights")
    (path / "model.safetensors").unlink()
    (path / "model.safetensors").symlink_to(outside)
    with pytest.raises(IdentityError, match="escapes"):
        runtime_identity(spec)


def test_metadata_file_size_is_bounded(tmp_path: Path) -> None:
    spec, path = _snapshot(tmp_path)
    with (path / "config.json").open("wb") as handle:
        handle.truncate(MAX_METADATA_FILE_BYTES + 1)
    with pytest.raises(IdentityError, match="too large"):
        runtime_identity(spec)


def test_tokenizer_model_content_is_bound_to_model_identity(tmp_path: Path) -> None:
    spec, path = _snapshot(tmp_path)
    (path / "tokenizer.model").write_bytes(b"tokenizer-v1")
    baseline = runtime_identity(spec)
    (path / "tokenizer.model").write_bytes(b"tokenizer-v2")
    changed = runtime_identity(spec)
    assert changed["model_sha256"] != baseline["model_sha256"]


def test_legitimate_large_tokenizer_json_is_stream_hashed_and_bound(tmp_path: Path) -> None:
    spec, path = _snapshot(tmp_path)
    tokenizer = path / "tokenizer.json"
    large_size = 33 * 1024 * 1024
    with tokenizer.open("wb") as handle:
        handle.truncate(large_size)
        handle.seek(large_size - 1)
        handle.write(b"x")
    baseline = runtime_identity(spec)
    with tokenizer.open("r+b") as handle:
        handle.seek(large_size - 1)
        handle.write(b"y")
    changed = runtime_identity(spec)
    assert changed["model_sha256"] != baseline["model_sha256"]


def test_tokenizer_artifact_over_limit_is_rejected(tmp_path: Path) -> None:
    spec, path = _snapshot(tmp_path)
    with (path / "tokenizer.json").open("wb") as handle:
        handle.truncate(MAX_TOKENIZER_FILE_BYTES + 1)
    with pytest.raises(IdentityError, match="too large"):
        runtime_identity(spec)


def test_code_manifest_uses_same_mlx_lm_label_for_installed_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = tmp_path / "site-packages" / "mlx_lm"
    installed.mkdir(parents=True)
    (installed / "generate.py").write_text("# installed", encoding="utf-8")

    def distribution(name: str):
        if name == "mlx-lm":
            return SimpleNamespace(locate_file=lambda _: installed)
        raise identity_module.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(identity_module.importlib.metadata, "distribution", distribution)
    _, files = identity_module._code_identity()
    assert "mlx_lm/generate.py" in files
    assert not any(label.startswith("installed/mlx_lm/") for label in files)


def test_code_symlink_outside_source_root_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "code"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("secret", encoding="utf-8")
    linked = root / "linked.py"
    linked.symlink_to(outside)
    with pytest.raises(IdentityError, match="escapes"):
        identity_module._safe_regular(linked, root, metadata=False)


def test_gpu_cores_change_identity_but_volatile_host_fields_do_not(tmp_path: Path) -> None:
    spec, _ = _snapshot(tmp_path)
    base_host = {
        "chip_name": "Test Chip",
        "gpu_devices": [{"model": "Test GPU", "cores": 10, "metal_support": "spdisplays_metal4"}],
        "load": 0.2,
        "temperature_c": 40,
        "memory_total_bytes": 1234,
    }
    baseline = runtime_identity(spec, base_host)
    volatile_changed = runtime_identity(spec, {**base_host, "load": 0.9, "temperature_c": 90})
    cores_changed = runtime_identity(spec, {
        **base_host,
        "gpu_devices": [{"model": "Test GPU", "cores": 12, "metal_support": "spdisplays_metal4"}],
    })
    assert volatile_changed["hardware_sha256"] == baseline["hardware_sha256"]
    assert volatile_changed["identity_sha256"] == baseline["identity_sha256"]
    assert cores_changed["hardware_sha256"] != baseline["hardware_sha256"]
    assert cores_changed["hardware"]["gpu_devices"][0]["cores"] == 12


@pytest.mark.parametrize("value", [True, 1, [], {"metal": "unsupported"}])
def test_gpu_metal_support_requires_technical_string_or_none(value: object) -> None:
    with pytest.raises(IdentityError):
        identity_module._host_identity({"gpu_devices": [{
            "model": "Test GPU", "cores": 10, "metal_support": value,
        }]})


def test_actual_cached_gemma_1b_identity_uses_native_hardware_without_model_load() -> None:
    from ironmule_inventory import discover_models

    rows = discover_models(
        [Path(__file__).resolve().parents[2] / ".friday-data" / "models" / "hub"],
        family="gemma", loader="mlx_lm",
    )
    row = next((item for item in rows if "1b" in item["model_id"].lower() and item["status"] == "available"), None)
    if row is None:
        pytest.skip("cached Gemma 3 1B snapshot is unavailable")
    spec = {key: row[key] for key in (
        "model_id", "revision", "snapshot_path", "weight_bytes"
    )}
    # Other engine tests legitimately import MLX. Verify this path in a clean
    # real child, rather than asserting against the suite's global module cache.
    script = """
import json, sys
sys.path.insert(0, sys.argv[1])
class DenyNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm', 'numpy'}:
            raise AssertionError('identity imported a native inference dependency')
sys.meta_path.insert(0, DenyNative())
from friday_evidence.identity import runtime_identity
from ironmule_product.readiness import hardware_identity
identity = runtime_identity(json.loads(sys.argv[2]), hardware_identity())
assert 'mlx' not in sys.modules and 'mlx_lm' not in sys.modules
assert json.loads(sys.argv[2])['snapshot_path'] not in json.dumps(identity)
print(json.dumps({key:identity[key] for key in ('model_sha256','identity_sha256')}))
"""
    result = subprocess.run([sys.executable, "-S", "-c", script,
                             str(Path(__file__).resolve().parents[2]), json.dumps(spec)],
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    identity = json.loads(result.stdout)
    assert all(len(value) == 64 for value in identity.values())
