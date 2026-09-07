"""Fail-closed policy for model snapshot Python execution."""

from __future__ import annotations

import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from friday_evidence.identity import (
    MAX_METADATA_FILE_BYTES,
    _safe_regular,
    _snapshot_root,
)

MAX_CONFIG_DEPTH = 64


class ModelPolicyError(ValueError):
    """A model snapshot requests unsupported executable Python configuration."""

    code = "unsupported_model_code"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelPolicyError("config contains duplicate keys")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise ModelPolicyError("config contains a non-finite number")


def _finite_float(value: str) -> float:
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ModelPolicyError("config contains an invalid number") from exc
    if not math.isfinite(number):
        raise ModelPolicyError("config contains a non-finite number")
    return number


def _read_config(path: Path, root: Path) -> dict[str, Any]:
    resolved = _safe_regular(path, root, metadata=True, maximum=MAX_METADATA_FILE_BYTES)
    try:
        before = os.stat(resolved, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_METADATA_FILE_BYTES:
            raise ModelPolicyError("config is invalid")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(resolved, flags)
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
            ):
                raise ModelPolicyError("config changed while opening")
            chunks: list[bytes] = []
            total = 0
            while total <= MAX_METADATA_FILE_BYTES:
                chunk = os.read(fd, min(1024 * 1024, MAX_METADATA_FILE_BYTES + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if total > MAX_METADATA_FILE_BYTES or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
        ) or total != before.st_size:
            raise ModelPolicyError("config changed while reading")
        payload = b"".join(chunks).decode("utf-8")
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_keys,
                           parse_constant=_reject_nonfinite, parse_float=_finite_float)
    except ModelPolicyError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ModelPolicyError("config is invalid") from exc
    if not isinstance(value, dict):
        raise ModelPolicyError("config must be a JSON object")
    pending: list[tuple[Any, int]] = [(value, 1)]
    while pending:
        current, depth = pending.pop()
        if depth > MAX_CONFIG_DEPTH:
            raise ModelPolicyError("config nesting is too deep")
        if isinstance(current, dict):
            pending.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            pending.extend((child, depth + 1) for child in current)
    return value


def validate_model_config(snapshot_path: str, revision: Any) -> dict[str, Any]:
    """Validate ``config.json`` before any MLX import or model loading."""
    if not isinstance(snapshot_path, str) or not snapshot_path:
        raise ModelPolicyError("snapshot path is invalid")
    if not isinstance(revision, str) or not revision:
        raise ModelPolicyError("snapshot revision is invalid")
    path = Path(snapshot_path)
    if not path.is_absolute():
        raise ModelPolicyError("snapshot path is invalid")
    try:
        snapshot, root = _snapshot_root(path, revision)
        config = _read_config(snapshot / "config.json", root)
    except ModelPolicyError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise ModelPolicyError("snapshot config is unavailable") from exc
    # mlx_lm.utils.load_model executes this path whenever it is not None.
    if "model_file" in config and config["model_file"] is not None:
        raise ModelPolicyError("custom model Python is not supported")
    return {"model_file": None}


__all__ = ["MAX_CONFIG_DEPTH", "ModelPolicyError", "validate_model_config"]
