"""Bounded canonical JSON helpers for Phase 1B."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .constants import MAX_CANONICAL_BYTES


INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


class CanonicalError(ValueError):
    """A value cannot enter the closed Phase-1B JSON contract."""


class _DuplicateKeyError(ValueError):
    pass


def _validate(value: Any, *, depth: int, nodes: list[int]) -> None:
    nodes[0] += 1
    if nodes[0] > 200_000 or depth > 32:
        raise CanonicalError("JSON structure exceeds its registered bounds")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        if not INT64_MIN <= value <= INT64_MAX:
            raise CanonicalError("JSON integer is outside signed-int64")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalError("JSON number must be finite")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or len(key) > 512:
                raise CanonicalError("JSON object key is invalid")
            _validate(child, depth=depth + 1, nodes=nodes)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _validate(child, depth=depth + 1, nodes=nodes)
        return
    raise CanonicalError(f"unsupported JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any, *, maximum: int = MAX_CANONICAL_BYTES) -> bytes:
    _validate(value, depth=0, nodes=[0])
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CanonicalError(str(exc)) from exc
    if len(payload) > maximum:
        raise CanonicalError("canonical JSON exceeds its registered byte limit")
    return payload


def canonical_sha256(value: Any) -> str:
    payload = value if isinstance(value, bytes) else canonical_json_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def strict_json_loads(payload: bytes, *, maximum: int = MAX_CANONICAL_BYTES) -> Any:
    if not isinstance(payload, bytes) or len(payload) > maximum:
        raise CanonicalError("JSON input exceeds its registered byte limit")

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise _DuplicateKeyError(key)
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, _DuplicateKeyError) as exc:
        raise CanonicalError("invalid strict JSON") from exc
    if canonical_json_bytes(value, maximum=maximum) != payload:
        raise CanonicalError("JSON input is not canonical")
    return value
