"""Bounded canonical JSON and primitive validators for H0.1."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .constants import INT64_MAX, INT64_MIN

MAX_DEPTH = 16
MAX_NODES = 50_000
MAX_SEQUENCE = 10_000
MAX_STRING = 64 * 1024
MAX_CANONICAL_BYTES = 1 * 1024 * 1024


class CanonicalError(ValueError):
    """Raised when a value cannot enter the bounded canonical contract."""


class _DuplicateKeyError(ValueError):
    pass


def exact_keys(value: Any, expected: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CanonicalError(f"{name} must be an object with string keys")
    if set(value) != expected:
        raise CanonicalError(f"{name} has unknown or missing keys")
    return value


def int64(value: Any, name: str, *, minimum: int = INT64_MIN, maximum: int = INT64_MAX) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CanonicalError(f"{name} must be a signed-int64 integer, not bool")
    if value < minimum or value > maximum:
        raise CanonicalError(f"{name} is outside the registered signed-int64 range")
    return value


def nonnegative_int64(value: Any, name: str, *, maximum: int = INT64_MAX) -> int:
    """Validate a non-negative signed-int64; ``bool`` is never an integer here."""

    return int64(value, name, minimum=0, maximum=maximum)


def positive_int64(value: Any, name: str, *, maximum: int = INT64_MAX) -> int:
    """Validate a positive signed-int64; ``bool`` is never an integer here."""

    return int64(value, name, minimum=1, maximum=maximum)


def exact_int64(value: Any, name: str, expected: int) -> int:
    """Type-check before comparing an integer to its registered exact value."""

    checked_expected = int64(expected, f"registered {name}")
    checked = int64(value, name)
    if checked != checked_expected:
        raise CanonicalError(f"{name} must equal registered signed-int64 {checked_expected}")
    return checked


def finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanonicalError(f"{name} must be numeric, not bool")
    if isinstance(value, int):
        int64(value, name)
    numeric = float(value)
    if not math.isfinite(numeric) or (minimum is not None and numeric < minimum):
        raise CanonicalError(f"{name} is non-finite or outside its lower bound")
    return numeric


def bounded_text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise CanonicalError(f"{name} must be bounded non-empty text")
    return value


def _validate(value: Any, *, depth: int, counter: list[int]) -> None:
    counter[0] += 1
    if counter[0] > MAX_NODES or depth > MAX_DEPTH:
        raise CanonicalError("JSON structure exceeds its node or depth bound")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        int64(value, "JSON integer")
        return
    if isinstance(value, float):
        finite_number(value, "JSON number")
        return
    if isinstance(value, str):
        if len(value) > MAX_STRING or "\x00" in value:
            raise CanonicalError("JSON string exceeds its bound or contains NUL")
        return
    if isinstance(value, Mapping):
        if len(value) > MAX_SEQUENCE or any(not isinstance(key, str) for key in value):
            raise CanonicalError("JSON object exceeds its bound or has a non-string key")
        for key, child in value.items():
            _validate(key, depth=depth + 1, counter=counter)
            _validate(child, depth=depth + 1, counter=counter)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_SEQUENCE:
            raise CanonicalError("JSON array exceeds its bound")
        for child in value:
            _validate(child, depth=depth + 1, counter=counter)
        return
    raise CanonicalError(f"unsupported JSON value {type(value).__name__}")


def canonical_json_bytes(value: Any, *, maximum: int = MAX_CANONICAL_BYTES) -> bytes:
    _validate(value, depth=0, counter=[0])
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise CanonicalError(f"value is not canonical JSON: {exc}") from exc
    if len(encoded) > maximum:
        raise CanonicalError(f"canonical JSON exceeds {maximum} bytes")
    return encoded


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def strict_json_loads(payload: bytes, *, maximum: int = MAX_CANONICAL_BYTES) -> Any:
    if not isinstance(payload, bytes) or len(payload) > maximum:
        raise CanonicalError("JSON payload is not bytes or exceeds its bound")

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
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, _DuplicateKeyError) as exc:
        raise CanonicalError(f"invalid strict JSON: {exc}") from exc
    canonical_json_bytes(value, maximum=maximum)
    return value
