"""Repository, environment, hardware, code, and specification provenance."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .canonical import canonical_sha256
from .registry import REGISTERED_TOOLS, SCHEMA_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("friday_evidence", "friday_h0", "friday_h01", "tools")
SPEC_FILES = (
    "docs/PHASE1_MATMUL_SPEC.md",
    "docs/H1_VORREGISTRIERUNG_ENTWURF.md",
    "docs/H1H2_EVIDENZ_ARCHITEKTUR.md",
    "requirements-apple-silicon.txt",
    "pytest.ini",
)
PACKAGES = ("mlx", "mlx-metal", "numpy", "scipy", "psutil", "pytest", "pytest-xdist", "torch", "mlx-lm")


class ProvenanceError(RuntimeError):
    """Live source cannot be bound to a complete reproducible identity."""


def _run_git(*args: str) -> bytes:
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
        raise ProvenanceError(f"Git unavailable: {type(exc).__name__}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()[-200:]
        raise ProvenanceError(f"Git command failed: {detail}")
    return completed.stdout


def _file_hashes(paths: list[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(paths):
        if not path.is_file() or path.is_symlink():
            raise ProvenanceError(f"registered provenance file is not a regular file: {path}")
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _code_paths() -> list[Path]:
    paths: list[Path] = []
    for directory in SOURCE_DIRS:
        root = PROJECT_ROOT / directory
        paths.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
        paths.extend(root.rglob("*.sql"))
    return paths


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
    value = completed.stdout.decode("utf-8", errors="strict").strip()
    return value or None


def collect_provenance(tool: str, *, require_clean: bool = True) -> dict[str, object]:
    if tool not in REGISTERED_TOOLS:
        raise ProvenanceError(f"unregistered evidence tool: {tool}")

    revision = _run_git("rev-parse", "HEAD").decode("ascii").strip()
    status = _run_git(
        "status", "--porcelain", "--untracked-files=all", "--", ".", ":(exclude)ProjectAtlas"
    ).decode(
        "utf-8", errors="strict"
    )
    dirty = bool(status.strip())
    if require_clean and dirty:
        raise ProvenanceError("project worktree is dirty; commit before measuring")
    diff = _run_git("diff", "--binary", "HEAD") + _run_git("diff", "--cached", "--binary", "HEAD")

    code_files = _file_hashes(_code_paths())
    spec_paths = [PROJECT_ROOT / relative for relative in SPEC_FILES]
    spec_files = _file_hashes(spec_paths)
    packages: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None

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
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "tool": tool,
        "workload_key": REGISTERED_TOOLS[tool],
        "provenance_kind": "native",
        "git_revision": revision,
        "git_dirty": dirty,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "code_sha256": canonical_sha256(code_files),
        "code_files": code_files,
        "spec_sha256": canonical_sha256(spec_files),
        "spec_files": spec_files,
        "environment_sha256": canonical_sha256(environment),
        "environment": environment,
        "hardware_key": canonical_sha256(hardware),
        "hardware": hardware,
    }
    provenance["provenance_sha256"] = canonical_sha256(provenance)
    return provenance


__all__ = ["ProvenanceError", "collect_provenance"]
