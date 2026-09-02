"""Safe reads for anything that goes into a hash: no symlinks, no growth, bounded."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path

MAX_FILE_BYTES = 4 * 1024 * 1024


class UnsafeFile(RuntimeError):
    """A file cannot be hashed without weakening the identity it would carry."""


def regular_bytes(path: Path, *, maximum: int = MAX_FILE_BYTES) -> bytes:
    """Read a plain, bounded, non-symlink regular file or refuse."""

    try:
        info = path.lstat()
        data = path.read_bytes()
    except OSError as exc:
        raise UnsafeFile(f"provenance input is unavailable: {path}") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or info.st_size > maximum
        or len(data) != info.st_size
    ):
        raise UnsafeFile(f"provenance input is unsafe or unstable: {path}")
    return data


def regular_sha256(path: Path, *, maximum: int = MAX_FILE_BYTES) -> str:
    return hashlib.sha256(regular_bytes(path, maximum=maximum)).hexdigest()


def file_hashes(root: Path, relatives: list[str]) -> dict[str, str]:
    """Hash each relative path under ``root``, deduplicated and sorted."""

    return {
        relative: regular_sha256(root / relative)
        for relative in sorted(set(relatives))
    }


def source_relatives(root: Path, directory: str, suffixes: frozenset[str]) -> list[str]:
    """Every source file under ``directory``, excluding bytecode caches."""

    return [
        path.relative_to(root).as_posix()
        for path in (root / directory).rglob("*")
        if path.suffix in suffixes and "__pycache__" not in path.parts
    ]


__all__ = [
    "MAX_FILE_BYTES",
    "UnsafeFile",
    "file_hashes",
    "regular_bytes",
    "regular_sha256",
    "source_relatives",
]
