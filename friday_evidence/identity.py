"""Portable, model-free identity binding for local model measurements."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import stat
from collections.abc import Mapping
from typing import Any

from .canonical import canonical_sha256


MAX_METADATA_FILE_BYTES = 16 * 1024 * 1024
MAX_TOKENIZER_FILE_BYTES = 64 * 1024 * 1024
_MODEL_SPEC_KEYS = frozenset(("model_id", "revision", "snapshot_path", "weight_bytes"))
_REQUIRED_PACKAGES = ("mlx", "mlx-lm", "numpy", "transformers")
_METADATA_SUFFIXES = frozenset((".json", ".jsonl", ".jinja", ".jinja2", ".model"))
_MAX_CODE_FILE_BYTES = 16 * 1024 * 1024
_MAX_CODE_FILES = 4096
_MAX_TREE_DEPTH = 32
_MAX_TREE_FILES = 4096


class IdentityError(ValueError):
    """A model snapshot or identity cannot be safely bound."""


def _validate_spec(model_spec: Mapping[str, Any]) -> tuple[str, str, Path, int]:
    if not isinstance(model_spec, Mapping) or set(model_spec) != _MODEL_SPEC_KEYS:
        raise IdentityError("model spec must contain exactly model_id, revision, snapshot_path, weight_bytes")
    model_id = model_spec["model_id"]
    revision = model_spec["revision"]
    snapshot_path = model_spec["snapshot_path"]
    weight_bytes = model_spec["weight_bytes"]
    if (not isinstance(model_id, str) or not model_id or len(model_id) > 4096
            or not isinstance(revision, str) or not revision or len(revision) > 4096
            or not isinstance(snapshot_path, str) or not snapshot_path or len(snapshot_path) > 4096):
        raise IdentityError("model identity fields must be bounded non-empty strings")
    if type(weight_bytes) is not int or weight_bytes <= 0:
        raise IdentityError("model weight_bytes must be a positive integer")
    path = Path(snapshot_path)
    if not path.is_absolute():
        raise IdentityError("model snapshot path must be absolute")
    return model_id, revision, path, weight_bytes


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _snapshot_root(raw_path: Path, revision: str) -> tuple[Path, Path]:
    """Resolve a snapshot and its allowed file root without accepting escapes."""
    try:
        lexical = raw_path
        if lexical.parent.name == "snapshots" and lexical.name == revision:
            repository_lexical = lexical.parent.parent
            repository_root = repository_lexical.resolve(strict=True)
            snapshot = lexical.resolve(strict=True)
            if not _inside(snapshot, repository_root) or not snapshot.is_dir():
                raise IdentityError("snapshot symlink escapes its known repository root")
            return snapshot, repository_root
        if raw_path.is_symlink():
            raise IdentityError("local snapshot symlink must use a known HF repository layout")
        snapshot = raw_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IdentityError("snapshot directory is unavailable") from exc
    if not snapshot.is_dir():
        raise IdentityError("snapshot path is not a directory")
    return snapshot, snapshot


def _safe_regular(path: Path, root: Path, *, metadata: bool, maximum: int | None = None) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IdentityError(f"snapshot file is missing or broken: {path.name}") from exc
    if not _inside(resolved, root):
        raise IdentityError(f"snapshot file escapes its root: {path.name}")
    try:
        info = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise IdentityError(f"snapshot file cannot be inspected: {path.name}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise IdentityError(f"snapshot file is not regular: {path.name}")
    bound = maximum if maximum is not None else (MAX_METADATA_FILE_BYTES if metadata else None)
    if bound is not None and info.st_size > bound:
        raise IdentityError(f"snapshot metadata file is too large: {path.name}")
    return resolved


def _read_hashed(path: Path, *, metadata: bool, maximum: int | None = None) -> tuple[int, str]:
    """Read a regular file and reject replacement or mutation during hashing."""
    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise IdentityError(f"snapshot file is not regular: {path.name}")
        bound = maximum if maximum is not None else (MAX_METADATA_FILE_BYTES if metadata else None)
        if bound is not None and before.st_size > bound:
            raise IdentityError(f"snapshot metadata file is too large: {path.name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
            ):
                raise IdentityError(f"snapshot file changed while opening: {path.name}")
            total = 0
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if bound is not None and total > bound:
                    raise IdentityError(f"identity file is too large: {path.name}")
                digest.update(chunk)
        after = os.stat(path, follow_symlinks=False)
    except IdentityError:
        raise
    except OSError as exc:
        raise IdentityError(f"snapshot file cannot be hashed: {path.name}") from exc
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ) or total != before.st_size:
        raise IdentityError(f"snapshot file changed while hashing: {path.name}")
    return total, digest.hexdigest()


def _tree_files(root: Path, *, suffixes: frozenset[str]) -> list[Path]:
    """Walk a bounded tree, failing closed on symlinked directories."""
    if not root.is_dir() or root.is_symlink():
        raise IdentityError("identity source root is unavailable")
    files: list[Path] = []
    visited_files = 0
    for current, directories, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        try:
            depth = len(current_path.relative_to(root).parts)
        except ValueError as exc:
            raise IdentityError("identity traversal escaped its root") from exc
        if depth > _MAX_TREE_DEPTH:
            raise IdentityError("identity tree is too deeply nested")
        for directory in directories:
            if (current_path / directory).is_symlink():
                raise IdentityError(f"identity tree contains a symlinked directory: {directory}")
        directories.sort()
        for name in sorted(names):
            path = current_path / name
            visited_files += 1
            if visited_files > _MAX_TREE_FILES:
                raise IdentityError("identity tree contains too many files")
            if path.suffix.lower() in suffixes:
                files.append(path)
                if len(files) > _MAX_CODE_FILES:
                    raise IdentityError("identity tree contains too many source files")
    return files


def _metadata_bound(path: Path) -> int:
    if path.name in {"tokenizer.json", "tokenizer.model"}:
        return MAX_TOKENIZER_FILE_BYTES
    return MAX_METADATA_FILE_BYTES


def _snapshot_files(snapshot: Path, root: Path, expected_weight_bytes: int) -> list[dict[str, Any]]:
    weight_paths = sorted(snapshot.glob("model*.safetensors"))
    if not weight_paths:
        raise IdentityError("snapshot has no direct model*.safetensors files")
    required = [snapshot / "config.json", snapshot / "tokenizer_config.json"]
    for path in required:
        _safe_regular(path, root, metadata=True)
    tokenizer_paths = [
        path for path in (snapshot / "tokenizer.json", snapshot / "tokenizer.model")
        if path.exists() or path.is_symlink()
    ]
    if not tokenizer_paths:
        raise IdentityError("snapshot has no tokenizer.json or tokenizer.model")
    metadata_paths: set[Path] = set(required + tokenizer_paths)
    metadata_paths.update(_tree_files(snapshot, suffixes=_METADATA_SUFFIXES))
    for path in weight_paths:
        _safe_regular(path, root, metadata=False)
    for path in metadata_paths:
        _safe_regular(path, root, metadata=True, maximum=_metadata_bound(path))
    total = 0
    files: list[dict[str, Any]] = []
    for path in weight_paths:
        resolved = _safe_regular(path, root, metadata=False)
        size, digest = _read_hashed(resolved, metadata=False)
        total += size
        files.append({"name": path.relative_to(snapshot).as_posix(), "size": size, "sha256": digest})
    if total != expected_weight_bytes:
        raise IdentityError("snapshot weight bytes differ from model registration")
    # Hash required and relevant metadata as part of the model identity, while
    # exposing only weight files in the public model_files contract.
    for path in sorted(metadata_paths):
        resolved = _safe_regular(path, root, metadata=True, maximum=_metadata_bound(path))
        _read_hashed(resolved, metadata=True, maximum=_metadata_bound(path))
    return files


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _REQUIRED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _code_identity() -> tuple[str, dict[str, str]]:
    project_root = Path(__file__).resolve().parents[1]
    paths: list[tuple[Path, Path, str]] = []
    python_suffix = frozenset((".py",))
    # The product can now execute the existing Engine as an explicit candidate.
    # Bind its files without importing the MLX-bearing package in the controller.
    for package in ("ironmule", "ironmule_product", "friday_evidence"):
        package_root = project_root / package
        paths.extend((path, package_root, "") for path in _tree_files(package_root, suffixes=python_suffix))
    try:
        mlx_root = Path(importlib.metadata.distribution("mlx-lm").locate_file("mlx_lm"))
    except importlib.metadata.PackageNotFoundError:
        mlx_root = None
    if mlx_root is not None and mlx_root.is_dir():
        paths.extend((path, mlx_root, "mlx_lm") for path in _tree_files(mlx_root, suffixes=python_suffix))
    files: dict[str, str] = {}
    for path, source_root, prefix in paths:
        resolved = _safe_regular(path, source_root, metadata=False)
        try:
            label = path.relative_to(project_root).as_posix()
        except ValueError:
            label = prefix + "/" + path.relative_to(source_root).as_posix()
        _, digest = _read_hashed(resolved, metadata=False, maximum=_MAX_CODE_FILE_BYTES)
        if label in files:
            raise IdentityError(f"duplicate code manifest label: {label}")
        files[label] = digest
    return canonical_sha256(files), files


def _host_identity(host: Mapping[str, Any] | None) -> dict[str, Any]:
    supplied = dict(host or {})
    chip = supplied.get("chip_name") or supplied.get("chip") or platform.processor() or None
    if chip is not None and (not isinstance(chip, str) or not chip or len(chip) > 256):
        raise IdentityError("host chip_name must be a bounded non-empty string")
    raw_gpus = supplied.get("gpu_devices", [])
    if raw_gpus is None:
        raw_gpus = []
    if not isinstance(raw_gpus, list):
        raise IdentityError("host gpu_devices must be a list")
    gpu_devices: list[dict[str, Any]] = []
    for device in raw_gpus:
        if not isinstance(device, Mapping) or set(device) != {"model", "cores", "metal_support"}:
            raise IdentityError("host gpu_devices entries must contain model, cores and metal_support")
        model, cores, metal_support = device["model"], device["cores"], device["metal_support"]
        if model is not None and (not isinstance(model, str) or not model or len(model) > 256):
            raise IdentityError("host GPU model must be a bounded string or None")
        if cores is not None and (type(cores) is not int or cores < 0):
            raise IdentityError("host GPU cores must be a non-negative integer or None")
        if metal_support is not None and (
            not isinstance(metal_support, str) or not metal_support or len(metal_support) > 256
        ):
            raise IdentityError("host GPU metal_support must be a bounded technical string or None")
        gpu_devices.append({"model": model, "cores": cores, "metal_support": metal_support})
    gpu_devices.sort(key=lambda item: (
        item["model"] or "", item["cores"] if item["cores"] is not None else -1,
        item["metal_support"] or "",
    ))
    memory = supplied.get("memory_total_bytes")
    if memory is None:
        try:
            memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, ValueError, OSError):
            memory = None
    return {
        "platform_supported": platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"},
        "platform": platform.system(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": memory if type(memory) is int and memory > 0 else None,
        "chip": chip,
        "chip_name": chip,
        "gpu_devices": gpu_devices,
    }


def runtime_identity(model_spec: Mapping[str, Any], host: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Bind model content, runtime code, packages, and stable host identity."""
    model_id, revision, raw_snapshot, weight_bytes = _validate_spec(model_spec)
    snapshot, root = _snapshot_root(raw_snapshot, revision)
    model_files = _snapshot_files(snapshot, root, weight_bytes)
    # Metadata content is bound by the complete metadata tree, not emitted as
    # absolute paths.  Re-read through the same safe traversal for the digest.
    metadata_hashes: dict[str, str] = {}
    for path in _tree_files(snapshot, suffixes=_METADATA_SUFFIXES):
        if path.is_file():
            bound = _metadata_bound(path)
            resolved = _safe_regular(path, root, metadata=True, maximum=bound)
            _, digest = _read_hashed(resolved, metadata=True, maximum=bound)
            metadata_hashes[path.relative_to(snapshot).as_posix()] = digest
    model_payload = {
        "model_id": model_id,
        "revision": revision,
        "weight_bytes": weight_bytes,
        "model_files": model_files,
        "metadata": metadata_hashes,
    }
    model_sha256 = canonical_sha256(model_payload)
    code_sha256, code_files = _code_identity()
    environment = {"python": platform.python_version(), "platform": platform.platform(), "packages": _package_versions()}
    environment_sha256 = canonical_sha256(environment)
    hardware = _host_identity(host)
    hardware_sha256 = canonical_sha256(hardware)
    identity = {
        "model_id": model_id,
        "revision": revision,
        "weight_bytes": weight_bytes,
        "model_files": model_files,
        "model_sha256": model_sha256,
        "environment_sha256": environment_sha256,
        "code_sha256": code_sha256,
        "hardware_sha256": hardware_sha256,
        "environment": environment,
        "hardware": hardware,
        "code_files": code_files,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def assert_model_unchanged(model_spec: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    """Re-hash the registered snapshot and fail if its content changed."""
    if not isinstance(identity, Mapping):
        raise IdentityError("identity must be a mapping")
    current = runtime_identity(model_spec)
    for key in ("model_id", "revision", "weight_bytes", "model_files", "model_sha256"):
        if current.get(key) != identity.get(key):
            raise IdentityError(f"model identity changed: {key}")


__all__ = [
    "IdentityError", "MAX_METADATA_FILE_BYTES", "MAX_TOKENIZER_FILE_BYTES",
    "assert_model_unchanged", "runtime_identity",
]
