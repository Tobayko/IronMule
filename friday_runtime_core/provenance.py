"""Parameterised runtime provenance: the same collector for every runtime package.

``friday_runtime``, ``friday_runtime_n10`` and ``friday_head_skip_runtime`` each
carry their own copy of this file; they differ only in *which* files they hash.
``ProvenanceSpec`` is that difference, made explicit.

What this deliberately does **not** do is compare the result against a frozen
constant. A frozen hardware hash pins evidence to one machine at one OS build;
on this very machine ``N10_HARDWARE_SHA256`` stopped matching when macOS
updated. Comparison belongs to the caller, and from Phase 1 on the counterpart
is a device profile measured here, not a constant baked in elsewhere.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from friday_evidence.canonical import canonical_sha256

from .files import UnsafeFile, file_hashes, source_relatives

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = frozenset({".py", ".sql"})
DEFAULT_PACKAGES = ("mlx", "mlx-metal", "numpy", "scipy", "psutil", "mlx-lm")


class ProvenanceError(RuntimeError):
    """A runtime measurement cannot be bound to immutable local source."""


@dataclass(frozen=True)
class ProvenanceSpec:
    """Which files, directories and packages make up one runtime's identity."""

    runtime_id: str
    schema_version: int = 1
    code_files: tuple[str, ...] = ()
    code_directories: tuple[str, ...] = ()
    spec_files: tuple[str, ...] = ()
    packages: tuple[str, ...] = DEFAULT_PACKAGES
    project_root: Path = field(default=PROJECT_ROOT)

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_id, str) or not self.runtime_id:
            raise ProvenanceError("runtime_id must be a non-empty string")
        if not self.code_files and not self.code_directories:
            raise ProvenanceError("a runtime identity must hash at least one code input")


def _git(root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["/usr/bin/git", "-C", str(root), *args],
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


def environment_facts(packages: tuple[str, ...]) -> dict[str, Any]:
    resolved: dict[str, str | None] = {}
    for name in packages:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "packages": resolved,
    }


def hardware_facts() -> dict[str, Any]:
    return {
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0] or None,
        "model": _sysctl("hw.model"),
        "memory_bytes": _sysctl("hw.memsize"),
        "cpu_brand": _sysctl("machdep.cpu.brand_string"),
    }


def machine_sha256() -> str:
    """A digest over the *stable* host identity: CPU, model, memory, arch.

    Deliberately excludes ``macos`` — a routine OS update changed a frozen
    hardware hash on the origin machine itself once, and the device profile
    exists precisely so that does not invalidate a calibration. This is enough
    to catch a ``.friday-data`` copied to a different Mac.
    """

    facts = hardware_facts()
    stable = {key: value for key, value in facts.items() if key != "macos"}
    return canonical_sha256(stable)


def collect_provenance(spec: ProvenanceSpec, *, require_clean: bool = True) -> dict[str, Any]:
    """Collect the full identity of one runtime: repository, code, spec, host."""

    root = spec.project_root
    revision = _git(root, "rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    status = _git(
        root,
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
    diff = _git(root, "diff", "--binary", "HEAD") + _git(
        root, "diff", "--cached", "--binary", "HEAD"
    )

    relatives = list(spec.code_files)
    for directory in spec.code_directories:
        relatives.extend(source_relatives(root, directory, SOURCE_SUFFIXES))
    try:
        code_files = file_hashes(root, relatives)
        spec_files = file_hashes(root, list(spec.spec_files))
    except UnsafeFile as exc:
        raise ProvenanceError(str(exc)) from exc

    environment = environment_facts(spec.packages)
    hardware = hardware_facts()
    result: dict[str, Any] = {
        "schema_version": spec.schema_version,
        "runtime_id": spec.runtime_id,
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


__all__ = [
    "PROJECT_ROOT",
    "ProvenanceError",
    "ProvenanceSpec",
    "collect_provenance",
    "environment_facts",
    "hardware_facts",
]
