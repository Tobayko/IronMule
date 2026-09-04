"""Deterministic, stdlib-only provenance for the closed H0.1 execution contract.

The three digests published here bind every paced session to an exact source
tree, an exact interpreter/platform/package environment, and the preregistered
study specification.  A study only aggregates sessions whose provenance is
byte-identical, so any edit to a listed file or a package upgrade produces a new,
separate study rather than a silently mixed one.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProvenanceError(ValueError):
    """Raised when a fixed provenance input is missing or unsafe."""


_MAX_SOURCE_BYTES = 2 * 1024 * 1024
_MAX_SPEC_BYTES = 2 * 1024 * 1024
_SPEC_RELATIVE = "docs/H01_PACED_TRAJECTORY_SPEC.md"

# This is deliberately a closed list.  It must not grow through globs or a
# caller-provided path: changing any of these inputs changes every session
# identity and therefore invalidates an in-flight six-session study.
_CODE_FILES = tuple(
    sorted(
        (
            "friday_h01/__init__.py",
            "friday_h01/analysis.py",
            "friday_h01/canonical.py",
            "friday_h01/cli.py",
            "friday_h01/constants.py",
            "friday_h01/dashboard.py",
            "friday_h01/import_h0.py",
            "friday_h01/protocol.py",
            "friday_h01/provenance.py",
            "friday_h01/runner.py",
            "friday_h01/schedule.py",
            "friday_h01/storage.py",
            "friday_h01/study.py",
            "friday_h01/migrations/0001_initial.sql",
        )
    )
)
# MLX and NumPy are execution-path dependencies only; the analysis core stays
# stdlib-only.  Their versions still belong to the environment identity because
# they determine what the measured durations mean.
_PACKAGE_METADATA = ("numpy", "mlx")
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Provenance:
    """The exact three provenance digests accepted by the v2 manifest."""

    code_sha256: str
    study_spec_sha256: str
    environment_sha256: str


def _regular_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ProvenanceError(f"provenance input is unavailable: {label}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ProvenanceError(f"provenance input is not a regular file: {label}")
    if info.st_size > maximum:
        raise ProvenanceError(f"provenance input exceeds its registered size: {label}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ProvenanceError(f"provenance input is unreadable: {label}") from exc
    if len(data) != info.st_size:
        raise ProvenanceError(f"provenance input changed while reading: {label}")
    return data


def _framed_code_hash(root: Path) -> str:
    digest = hashlib.sha256(b"friday_h01_code_v1\0")
    for relative in _CODE_FILES:
        data = _regular_bytes(root / relative, maximum=_MAX_SOURCE_BYTES, label=relative)
        path_bytes = relative.encode("utf-8")
        # Length-prefixing prevents concatenation/path ambiguity.
        digest.update(len(path_bytes).to_bytes(4, "big"))
        digest.update(path_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _package_metadata(name: str) -> dict[str, Any]:
    try:
        available = importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        available = False
    version: str | None = None
    if available:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
    return {"available": available, "version": version}


def _environment_hash() -> str:
    implementation = getattr(sys.implementation, "name", "unknown")
    version = sys.version_info
    material = {
        "python": {
            "implementation": str(implementation),
            "major": int(version.major),
            "minor": int(version.minor),
            "micro": int(version.micro),
            "cache_tag": str(getattr(sys.implementation, "cache_tag", "unknown")),
            "compiler": platform.python_compiler(),
            "build": list(platform.python_build()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        # This is an explicit, secret-free allowlist.  OS/kernel changes are
        # intentionally new environments; arbitrary environment variables are
        # never read or hashed.
        "packages": {name: _package_metadata(name) for name in _PACKAGE_METADATA},
    }
    encoded = json.dumps(
        material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _collect_from_root(root: Path) -> Provenance:
    spec = _regular_bytes(
        root / _SPEC_RELATIVE, maximum=_MAX_SPEC_BYTES, label="H0.1 paced trajectory spec"
    )
    return Provenance(
        code_sha256=_framed_code_hash(root),
        study_spec_sha256=hashlib.sha256(spec).hexdigest(),
        environment_sha256=_environment_hash(),
    )


def collect_provenance() -> Provenance:
    """Compute provenance for the real repository root only."""

    return _collect_from_root(_REPOSITORY_ROOT)


__all__ = ["Provenance", "ProvenanceError", "collect_provenance"]
