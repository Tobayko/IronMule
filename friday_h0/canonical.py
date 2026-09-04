"""Deterministic, finite, UTF-8 JSON and SHA-256 helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


class CanonicalizationError(TypeError):
    """Raised when a value cannot be represented by the closed JSON contract."""


def _validate_json_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(f"non-finite number at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(f"non-string object key at {path}")
            _validate_json_value(child, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
        return
    raise CanonicalizationError(f"unsupported JSON value at {path}: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact, sorted-key, finite UTF-8 JSON bytes."""

    _validate_json_value(value)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalizationError(str(exc)) from exc
    return encoded.encode("utf-8")


def canonical_json(value: Any) -> str:
    """Return the canonical JSON text corresponding to :func:`canonical_json_bytes`."""

    return canonical_json_bytes(value).decode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash canonical JSON values, or raw bytes when explicitly supplied."""

    payload = value if isinstance(value, bytes) else canonical_json_bytes(value)
    return hashlib.sha256(payload).hexdigest()

