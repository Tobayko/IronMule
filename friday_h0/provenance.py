"""Deterministic, stdlib-only provenance for the closed H0 runner contract."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProvenanceError(ValueError):
    """Raised when a fixed provenance input is missing or unsafe."""


REVISION_MISSING_REASON = "project root is not a Git repository"
_MAX_SOURCE_BYTES = 2 * 1024 * 1024
_MAX_SPEC_BYTES = 2 * 1024 * 1024

# This is deliberately a closed list.  It must not grow through globs or a
# caller-provided path: changing any of these inputs changes the run identity.
_CODE_FILES = tuple(sorted((
    "friday_h0/__init__.py",
    "friday_h0/aggregation.py",
    "friday_h0/benchmark.py",
    "friday_h0/canonical.py",
    "friday_h0/cli.py",
    "friday_h0/constants.py",
    "friday_h0/dashboard.py",
    "friday_h0/dashboard_assets.py",
    "friday_h0/decision.py",
    "friday_h0/manifest.py",
    "friday_h0/protocol.py",
    "friday_h0/provenance.py",
    "friday_h0/runner.py",
    "friday_h0/statistics.py",
    "friday_h0/storage.py",
    "friday_h0/supervisor.py",
    "friday_h0/worker.py",
    "friday_h0/migrations/0001_initial.sql",
)))
_PACKAGE_METADATA = ("numpy", "mlx")
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_TEST_ROOT_TOKEN = object()


@dataclass(frozen=True)
class _TestRoot:
    """Private, non-CLI root handle used only by isolated unit tests."""

    root: Path
    token: object


def _test_root(root: str | Path) -> _TestRoot:
    candidate = Path(root).resolve()
    if not candidate.is_dir():
        raise ProvenanceError("test provenance root must be an existing directory")
    return _TestRoot(candidate, _TEST_ROOT_TOKEN)


@dataclass(frozen=True)
class Provenance:
    """The exact four provenance fields accepted by the v1 manifest."""

    code_sha256: str
    spec_sha256: str
    environment_sha256: str
    revision: dict[str, str | None]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "code_sha256": self.code_sha256,
            "spec_sha256": self.spec_sha256,
            "environment_sha256": self.environment_sha256,
            "revision": dict(self.revision),
        }


def _regular_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    """Read one bounded regular file through one no-follow descriptor.

    A descriptor closes the pathname replacement window between the type check
    and the read.  Python's file API cannot make a cryptographic snapshot of a
    file that another process mutates in place, so the descriptor identity and
    size are checked before and after the bounded read as a fail-closed guard.
    """
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ProvenanceError(f"{label} cannot be read safely on this platform")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    fd = -1
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ProvenanceError(f"{label} is unavailable") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ProvenanceError(f"{label} must be a regular non-symlink file")
        if before.st_size > maximum:
            raise ProvenanceError(f"{label} exceeds its bounded size")
        data = bytearray()
        while len(data) <= maximum:
            chunk = os.read(fd, min(64 * 1024, maximum + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise ProvenanceError(f"{label} changed while it was read")
        if len(data) > maximum:
            raise ProvenanceError(f"{label} exceeds its bounded size")
        return bytes(data)
    except ProvenanceError:
        raise
    except OSError as exc:
        raise ProvenanceError(f"{label} cannot be read") from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _framed_code_hash(root: Path) -> str:
    digest = hashlib.sha256()
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
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _collect_from_root(root: Path) -> Provenance:
    spec = _regular_bytes(root / "docs/PHASE1_MATMUL_SPEC.md", maximum=_MAX_SPEC_BYTES, label="phase-1 spec")
    return Provenance(
        code_sha256=_framed_code_hash(root),
        spec_sha256=hashlib.sha256(spec).hexdigest(),
        environment_sha256=_environment_hash(),
        revision={"value": None, "missing_reason": REVISION_MISSING_REASON},
    )


def collect_provenance(project_root: str | Path | None = None) -> Provenance:
    """Compute provenance for the real repository root only.

    Alternate roots are intentionally unavailable through the product API.  A
    private token-bound helper exists solely for isolated unit-test fixtures.
    """

    root = _REPOSITORY_ROOT if project_root is None else Path(project_root).resolve()
    if root != _REPOSITORY_ROOT:
        raise ProvenanceError("alternate provenance roots require the private test factory")
    return _collect_from_root(root)


def _collect_provenance_for_tests(context: _TestRoot) -> Provenance:
    if not isinstance(context, _TestRoot) or context.token is not _TEST_ROOT_TOKEN:
        raise ProvenanceError("invalid private provenance test context")
    return _collect_from_root(context.root)


__all__ = ["Provenance", "ProvenanceError", "REVISION_MISSING_REASON", "collect_provenance"]
