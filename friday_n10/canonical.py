"""Bounded canonical JSON and exact primitive validators for N10-v1."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .constants import INT64_MAX, INT64_MIN, MAX_CANONICAL_BYTES


class CanonicalError(ValueError):
    """A value cannot enter the closed N10-v1 JSON contract."""


class _DuplicateKeyError(ValueError):
    pass


def exact_keys(value: Any, expected: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CanonicalError(f"{name} must be an object with string keys")
    if set(value) != expected:
        raise CanonicalError(f"{name} has unknown or missing keys")
    return value


def int64(
    value: Any,
    name: str,
    *,
    minimum: int = INT64_MIN,
    maximum: int = INT64_MAX,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CanonicalError(f"{name} must be a signed-int64 integer, not bool")
    if not minimum <= value <= maximum:
        raise CanonicalError(f"{name} is outside the registered signed-int64 range")
    return value


def finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanonicalError(f"{name} must be numeric, not bool")
    if isinstance(value, int):
        int64(value, name)
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise CanonicalError(f"{name} must be finite and within its registered range")
    return result


def bounded_text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise CanonicalError(f"{name} must be bounded non-empty text")
    return value


def _validate(value: Any, *, depth: int, counter: list[int]) -> None:
    counter[0] += 1
    if counter[0] > 200_000 or depth > 32:
        raise CanonicalError("JSON structure exceeds its registered bounds")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        int64(value, "JSON integer")
        return
    if isinstance(value, float):
        finite_number(value, "JSON number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise CanonicalError("JSON object key must be text")
            _validate(child, depth=depth + 1, counter=counter)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _validate(child, depth=depth + 1, counter=counter)
        return
    raise CanonicalError(f"unsupported JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any, *, maximum: int = MAX_CANONICAL_BYTES) -> bytes:
    _validate(value, depth=0, counter=[0])
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


__all__ = [
    "CanonicalError",
    "bounded_text",
    "canonical_json_bytes",
    "canonical_sha256",
    "exact_keys",
    "finite_number",
    "int64",
    "strict_json_loads",
]
