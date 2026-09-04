"""Repository, dependency, environment, and hardware identity for N10 runtime evidence."""

from __future__ import annotations

import hashlib
import platform
import stat
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from friday_n10_v2.canonical import canonical_sha256

from .constants import PROJECT_ROOT, RUNTIME_ID, SCHEMA_VERSION

_MAX_FILE_BYTES = 4 * 1024 * 1024
_PACKAGES = ("mlx", "mlx-metal", "numpy", "scipy", "psutil", "mlx-lm")
_EXACT_CODE_FILES = (
    "tools/run_n10_runtime.py",
    "friday_evidence/budget.py",
    "friday_evidence/registry.py",
    "friday_h0/benchmark.py",
)
_SPEC_FILES = (
    "docs/N10_RUNTIME_PROTOTYPE_SPEC.md",
    "docs/N10_VORREGISTRIERUNG_V2.md",
    "docs/N10_VORREGISTRIERUNG_V1.md",
    "docs/PHASE1_MATMUL_SPEC.md",
    "docs/H1H2_EVIDENZ_ARCHITEKTUR.md",
    "requirements-apple-silicon.txt",
)


class ProvenanceError(RuntimeError):
    """A runtime measurement cannot be bound to immutable local source."""


def _git(*args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["/usr/bin/git", "-C", str(PROJECT_ROOT), *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProvenanceError("Git is unavailable") from exc
    if completed.returncode != 0:
        raise ProvenanceError("Git could not establish runtime provenance")
    return completed.stdout


def _regular_bytes(relative: str) -> bytes:
    path = PROJECT_ROOT / relative
    try:
        info = path.lstat()
        data = path.read_bytes()
    except OSError as exc:
        raise ProvenanceError(f"provenance input is unavailable: {relative}") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or info.st_size > _MAX_FILE_BYTES
        or len(data) != info.st_size
    ):
        raise ProvenanceError(f"provenance input is unsafe or unstable: {relative}")
    return data


def _file_hashes(relatives: list[str]) -> dict[str, str]:
    return {
        relative: hashlib.sha256(_regular_bytes(relative)).hexdigest()
        for relative in sorted(set(relatives))
    }


def _sysctl(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", errors="strict").strip() or None


def collect_provenance(*, require_clean: bool = True) -> dict[str, Any]:
    revision = _git("rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    status = _git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)ProjectAtlas",
    ).decode("utf-8", errors="strict")
    dirty = bool(status.strip())
    if require_clean and dirty:
        raise ProvenanceError("project worktree is dirty; commit before measuring")
    diff = _git("diff", "--binary", "HEAD") + _git("diff", "--cached", "--binary", "HEAD")

    code_relatives = list(_EXACT_CODE_FILES)
    for directory in ("friday_runtime_n10", "friday_n10_v2", "friday_n10", "friday_h0"):
        code_relatives.extend(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in (PROJECT_ROOT / directory).rglob("*")
            if path.suffix in {".py", ".sql"} and "__pycache__" not in path.parts
        )
    code_files = _file_hashes(code_relatives)
    spec_files = _file_hashes(list(_SPEC_FILES))
    packages: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    environment = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "packages": packages,
    }
    hardware = {
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0] or None,
        "model": _sysctl("hw.model"),
        "memory_bytes": _sysctl("hw.memsize"),
        "cpu_brand": _sysctl("machdep.cpu.brand_string"),
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runtime_id": RUNTIME_ID,
        "git_revision": revision,
        "git_dirty": dirty,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "code_files": code_files,
        "code_sha256": canonical_sha256(code_files),
        "spec_files": spec_files,
        "spec_sha256": canonical_sha256(spec_files),
        "environment": environment,
        "environment_sha256": canonical_sha256(environment),
        "hardware": hardware,
        "hardware_sha256": canonical_sha256(hardware),
    }
    result["provenance_sha256"] = canonical_sha256(result)
    return result


__all__ = ["ProvenanceError", "collect_provenance"]
