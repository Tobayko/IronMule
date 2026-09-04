"""Immutable repository, dependency, source, and hardware identity."""

from __future__ import annotations

import hashlib
import os
import platform
import stat
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_sha256
from .constants import (
    CONTRACT_ID,
    EXPERIMENT_ID,
    MAX_CANONICAL_BYTES,
    PROJECT_ROOT,
    SCHEMA_VERSION,
)
from .kernel_source import KERNEL_NAME, KERNEL_SOURCE_SHA256, validate_frozen_source


_CODE_FILES = (
    "friday_phase1b/__init__.py",
    "friday_phase1b/canonical.py",
    "friday_phase1b/cli.py",
    "friday_phase1b/constants.py",
    "friday_phase1b/dashboard.py",
    "friday_phase1b/experiment.py",
    "friday_phase1b/history.py",
    "friday_phase1b/kernel.py",
    "friday_phase1b/kernel_source.py",
    "friday_phase1b/migrations/001_init.sql",
    "friday_phase1b/provenance.py",
    "friday_phase1b/statistics.py",
    "friday_phase1b/supervisor.py",
    "friday_phase1b/worker.py",
    "friday_phase1b/workload.py",
    "tools/run_phase1b_rmsnorm.py",
)
_SPEC_FILES = ("docs/PHASE1B_RESIDUAL_RMSNORM_SPEC.md",)
_PROVENANCE_KEYS = frozenset(
    {
        "experiment_id",
        "contract_id",
        "schema_version",
        "git_revision",
        "git_dirty",
        "git_status_sha256",
        "code_files",
        "code_sha256",
        "spec_files",
        "spec_sha256",
        "source",
        "source_binding_sha256",
        "environment",
        "environment_sha256",
        "hardware",
        "hardware_sha256",
        "provenance_sha256",
    }
)


class ProvenanceError(RuntimeError):
    """Live Phase-1B evidence cannot be bound to immutable source."""


def _git(*args: str) -> bytes:
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    completed = subprocess.run(
        [
            "/usr/bin/git",
            "--no-pager",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "diff.external=",
            "-C",
            str(PROJECT_ROOT),
            *args,
        ],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10.0,
        check=False,
    )
    if completed.returncode != 0:
        raise ProvenanceError("Git could not establish Phase-1B provenance")
    return completed.stdout


def _regular_bytes(relative: str) -> bytes:
    path = PROJECT_ROOT / relative
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_CANONICAL_BYTES:
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
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if (
        len(data) > MAX_CANONICAL_BYTES
        or len(data) != before.st_size
        or identity_before
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or identity_before
        != (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
        )
    ):
        raise ProvenanceError(f"provenance input is unsafe or unstable: {relative}")
    return data


def _file_hashes(relatives: tuple[str, ...]) -> dict[str, str]:
    return {
        relative: hashlib.sha256(_regular_bytes(relative)).hexdigest()
        for relative in sorted(set(relatives))
    }


def _revision_hashes(revision: str, relatives: tuple[str, ...]) -> dict[str, str]:
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


def _xcode() -> str | None:
    completed = subprocess.run(
        ["/usr/bin/xcodebuild", "-version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=5.0,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _source_snapshot(*, require_clean: bool) -> dict[str, Any]:
    validate_frozen_source()
    revision_before = _git("rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    status_before = _git(
        "status", "--porcelain", "--untracked-files=all", "--", ".", ":(exclude)ProjectAtlas"
    )
    dirty = bool(status_before.strip())
    if require_clean and dirty:
        raise ProvenanceError("Phase-1B live work requires a clean root worktree")
    code_files = _file_hashes(_CODE_FILES)
    spec_files = _file_hashes(_SPEC_FILES)
    revision_after = _git("rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    status_after = _git(
        "status", "--porcelain", "--untracked-files=all", "--", ".", ":(exclude)ProjectAtlas"
    )
    if revision_before != revision_after or status_before != status_after:
        raise ProvenanceError("Git identity changed while collecting Phase-1B provenance")
    if require_clean and (
        code_files != _revision_hashes(revision_before, _CODE_FILES)
        or spec_files != _revision_hashes(revision_before, _SPEC_FILES)
    ):
        raise ProvenanceError("Phase-1B source differs from its clean Git revision")
    source = {"kernel_name": KERNEL_NAME, "source_sha256": KERNEL_SOURCE_SHA256}
    return {
        "git_revision": revision_before,
        "git_dirty": dirty,
        "git_status_sha256": hashlib.sha256(status_before).hexdigest(),
        "code_files": code_files,
        "code_sha256": canonical_sha256(code_files),
        "spec_files": spec_files,
        "spec_sha256": canonical_sha256(spec_files),
        "source": source,
        "source_binding_sha256": canonical_sha256(source),
    }


def verify_source_snapshot(expected: Mapping[str, Any]) -> None:
    """Fail closed if live worker inputs differ from collected provenance."""

    if not isinstance(expected, Mapping):
        raise ProvenanceError("expected provenance is unavailable")
    expected_payload = dict(expected)
    if set(expected_payload) != _PROVENANCE_KEYS:
        raise ProvenanceError("expected provenance schema differs")
    expected_digest = expected_payload.pop("provenance_sha256", None)
    if (
        not isinstance(expected_digest, str)
        or canonical_sha256(expected_payload) != expected_digest
        or expected.get("experiment_id") != EXPERIMENT_ID
        or expected.get("contract_id") != CONTRACT_ID
        or expected.get("schema_version") != SCHEMA_VERSION
        or expected.get("git_dirty") is not False
    ):
        raise ProvenanceError("expected provenance digest differs")
    observed = _source_snapshot(require_clean=True)
    if any(expected.get(key) != value for key, value in observed.items()):
        raise ProvenanceError("Phase-1B source snapshot changed around worker execution")


def collect_provenance(*, require_clean: bool = True) -> dict[str, Any]:
    snapshot = _source_snapshot(require_clean=require_clean)
    environment = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable_lexical": sys.executable,
        "executable_resolved": str(Path(sys.executable).resolve()),
        "mlx": _package_version("mlx"),
        "numpy": _package_version("numpy"),
        "psutil": _package_version("psutil"),
        "xcode": _xcode(),
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
        "experiment_id": EXPERIMENT_ID,
        "contract_id": CONTRACT_ID,
        "schema_version": SCHEMA_VERSION,
        **snapshot,
        "environment": environment,
        "environment_sha256": canonical_sha256(environment),
        "hardware": hardware,
        "hardware_sha256": canonical_sha256(hardware),
    }
    payload["provenance_sha256"] = canonical_sha256(payload)
    return payload
