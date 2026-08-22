"""Repository, dependency and hardware identity for shadow-router evidence."""

from __future__ import annotations

import hashlib
import os
import platform
import stat
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from friday_n10_v2.canonical import canonical_sha256

from .constants import MAX_CANONICAL_BYTES, PROJECT_ROOT, ROUTER_ID, SCHEMA_VERSION


_CODE_FILES = (
    "friday_avo_router/__init__.py",
    "friday_avo_router/benchmark.py",
    "friday_avo_router/cli.py",
    "friday_avo_router/constants.py",
    "friday_avo_router/dashboard.py",
    "friday_avo_router/history.py",
    "friday_avo_router/migrations/001_init.sql",
    "friday_avo_router/provenance.py",
    "friday_avo_router/router.py",
    "tools/run_avo_router.py",
)
_SPEC_FILES = ("docs/AVO_SHADOW_ROUTER_SPEC.md",)


class ProvenanceError(RuntimeError):
    """A live router record cannot be bound to immutable local source."""


def _git(*args: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(PROJECT_ROOT), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10.0,
        check=False,
    )
    if completed.returncode != 0:
        raise ProvenanceError("Git could not establish router provenance")
    return completed.stdout


def _regular_bytes(relative: str) -> bytes:
    path = PROJECT_ROOT / relative
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_CANONICAL_BYTES:
            raise ProvenanceError(f"provenance input is unsafe or oversized: {relative}")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            data = handle.read(MAX_CANONICAL_BYTES + 1)
            after = os.fstat(handle.fileno())
        path_after = path.lstat()
    except OSError as exc:
        raise ProvenanceError(f"provenance input is unavailable: {relative}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        len(data) > MAX_CANONICAL_BYTES
        or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        != (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
        )
        or len(data) != info.st_size
    ):
        raise ProvenanceError(f"provenance input is unsafe or unstable: {relative}")
    return data


def _file_hashes(relatives: tuple[str, ...]) -> dict[str, str]:
    return {
        relative: hashlib.sha256(_regular_bytes(relative)).hexdigest()
        for relative in sorted(set(relatives))
    }


def _revision_file_hashes(revision: str, relatives: tuple[str, ...]) -> dict[str, str]:
    return {
        relative: hashlib.sha256(_git("show", f"{revision}:{relative}")).hexdigest()
        for relative in sorted(set(relatives))
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _sysctl(name: str) -> str | None:
    completed = subprocess.run(
        ["/usr/sbin/sysctl", "-n", name],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=3.0,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", errors="strict").strip()


def collect_provenance(*, require_clean: bool = True) -> dict[str, Any]:
    revision_before = _git("rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    status_before = _git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)ProjectAtlas",
    )
    dirty = bool(status_before.strip())
    if require_clean and dirty:
        raise ProvenanceError("router live work requires a clean root worktree")

    code_files = _file_hashes(_CODE_FILES)
    spec_files = _file_hashes(_SPEC_FILES)
    revision_after = _git("rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    status_after = _git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)ProjectAtlas",
    )
    if revision_after != revision_before or status_after != status_before:
        raise ProvenanceError("Git identity changed while collecting router provenance")
    if require_clean and (
        code_files != _revision_file_hashes(revision_before, _CODE_FILES)
        or spec_files != _revision_file_hashes(revision_before, _SPEC_FILES)
    ):
        raise ProvenanceError("router source differs from its clean Git revision")
    environment = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "mlx": _package_version("mlx"),
        "numpy": _package_version("numpy"),
    }
    hardware = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "model": _sysctl("hw.model"),
        "cpu_brand": _sysctl("machdep.cpu.brand_string"),
        "physical_memory_bytes": _sysctl("hw.memsize"),
    }
    payload: dict[str, Any] = {
        "router_id": ROUTER_ID,
        "schema_version": SCHEMA_VERSION,
        "git_revision": revision_before,
        "git_dirty": dirty,
        "git_diff_sha256": hashlib.sha256(status_before).hexdigest(),
        "code_files": code_files,
        "code_sha256": canonical_sha256(code_files),
        "spec_files": spec_files,
        "spec_sha256": canonical_sha256(spec_files),
        "environment": environment,
        "environment_sha256": canonical_sha256(environment),
        "hardware": hardware,
        "hardware_sha256": canonical_sha256(hardware),
    }
    payload["provenance_sha256"] = canonical_sha256(payload)
    return payload
