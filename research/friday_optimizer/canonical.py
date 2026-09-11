"""Strict, bounded canonical JSON used for evidence identities.

This is intentionally a small contract rather than a general JSON convenience
wrapper.  Objects have string keys, duplicate keys are rejected while loading,
non-finite numbers are rejected, and both nesting and encoded size are bounded.
The resulting UTF-8 bytes use sorted keys and compact separators, making hashes
stable across processes and machines running the same Python JSON semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, Final

DEFAULT_MAX_DEPTH: Final[int] = 32
DEFAULT_MAX_BYTES: Final[int] = 1_048_576
DEFAULT_MAX_ITEMS: Final[int] = 10_000
DEFAULT_MAX_STRING_BYTES: Final[int] = 262_144
DEFAULT_MAX_INTEGER_DIGITS: Final[int] = 4_096


class CanonicalJSONError(ValueError):
    """Raised when JSON is not safe or bounded enough for the memory store."""


def _check_value(
    value: Any,
    *,
    depth: int,
    max_depth: int,
    max_items: int,
    max_string_bytes: int,
    max_integer_digits: int,
    ancestors: set[int],
) -> None:
    if depth > max_depth:
        raise CanonicalJSONError(f"maximum JSON depth exceeded ({max_depth})")
    if value is None or isinstance(value, bool) or isinstance(value, str):
        if isinstance(value, str):
            if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
                raise CanonicalJSONError("surrogate code points are not valid UTF-8 JSON")
            try:
                string_bytes = len(value.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise CanonicalJSONError("JSON string is not valid UTF-8") from exc
            if string_bytes > max_string_bytes:
                raise CanonicalJSONError("maximum JSON string size exceeded")
        return
    # bool is checked above: Python otherwise considers it an int.
    if isinstance(value, int):
        # bit_length avoids converting an attacker-controlled giant integer to
        # a decimal string before the bound is known.
        if value and int(value.bit_length() * 0.302 + 1) > max_integer_digits:
            raise CanonicalJSONError("maximum JSON integer size exceeded")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJSONError("NaN and infinity are not valid canonical JSON")
        return
    if isinstance(value, (list, tuple)):
        if len(value) > max_items:
            raise CanonicalJSONError("maximum JSON array size exceeded")
        identity = id(value)
        if identity in ancestors:
            raise CanonicalJSONError("cyclic JSON value")
        ancestors.add(identity)
        for item in value:
            _check_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_bytes=max_string_bytes,
                max_integer_digits=max_integer_digits,
                ancestors=ancestors,
            )
        ancestors.remove(identity)
        return
    if isinstance(value, Mapping):
        if len(value) > max_items:
            raise CanonicalJSONError("maximum JSON object size exceeded")
        identity = id(value)
        if identity in ancestors:
            raise CanonicalJSONError("cyclic JSON value")
        ancestors.add(identity)
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJSONError("canonical JSON object keys must be strings")
            if any(0xD800 <= ord(character) <= 0xDFFF for character in key):
                raise CanonicalJSONError("surrogate code points are not valid UTF-8 JSON keys")
            try:
                key_bytes = len(key.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise CanonicalJSONError("JSON key is not valid UTF-8") from exc
            if key_bytes > max_string_bytes:
                raise CanonicalJSONError("maximum JSON key size exceeded")
            _check_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_bytes=max_string_bytes,
                max_integer_digits=max_integer_digits,
                ancestors=ancestors,
            )
        ancestors.remove(identity)
        return
    raise CanonicalJSONError(f"unsupported JSON value: {type(value).__name__}")


def _json_value(value: Any) -> Any:
    """Normalize JSON-compatible mapping/sequence subclasses for ``json``."""

    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def canonical_bytes(
    value: Any,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_string_bytes: int = DEFAULT_MAX_STRING_BYTES,
    max_integer_digits: int = DEFAULT_MAX_INTEGER_DIGITS,
) -> bytes:
    """Return bounded canonical UTF-8 JSON bytes for *value*.

    ``tuple`` values are accepted as a convenience and encoded as JSON arrays.
    Booleans remain JSON booleans; callers defining integer fields must validate
    those fields separately and must never use ``isinstance(x, int)`` alone.
    """

    for name, bound in (
        ("max_depth", max_depth),
        ("max_bytes", max_bytes),
        ("max_items", max_items),
        ("max_string_bytes", max_string_bytes),
        ("max_integer_digits", max_integer_digits),
    ):
        if not isinstance(bound, int) or isinstance(bound, bool) or bound <= 0:
            raise ValueError(f"{name} must be a positive integer")
    _check_value(
        value,
        depth=0,
        max_depth=max_depth,
        max_items=max_items,
        max_string_bytes=max_string_bytes,
        max_integer_digits=max_integer_digits,
        ancestors=set(),
    )
    try:
        encoded = json.dumps(
            _json_value(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CanonicalJSONError(str(exc)) from exc
    if len(encoded) > max_bytes:
        raise CanonicalJSONError(f"maximum canonical JSON size exceeded ({max_bytes})")
    return encoded


def canonical_dumps(value: Any, **kwargs: int) -> str:
    """Return :func:`canonical_bytes` decoded as UTF-8 text."""

    return canonical_bytes(value, **kwargs).decode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CanonicalJSONError(f"non-finite JSON number: {value}")


def _parse_int(value: str, *, max_digits: int = DEFAULT_MAX_INTEGER_DIGITS) -> int:
    if len(value.lstrip("-")) > max_digits:
        raise CanonicalJSONError("maximum JSON integer size exceeded")
    return int(value)


def loads_strict(
    data: str | bytes | bytearray,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_string_bytes: int = DEFAULT_MAX_STRING_BYTES,
    max_integer_digits: int = DEFAULT_MAX_INTEGER_DIGITS,
) -> Any:
    """Parse JSON with duplicate-key, numeric, depth, and size checks."""

    for name, bound in (
        ("max_depth", max_depth),
        ("max_bytes", max_bytes),
        ("max_items", max_items),
        ("max_string_bytes", max_string_bytes),
        ("max_integer_digits", max_integer_digits),
    ):
        if not isinstance(bound, int) or isinstance(bound, bool) or bound <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(data, str):
        try:
            encoded = data.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise CanonicalJSONError("JSON input is not valid UTF-8") from exc
    elif isinstance(data, (bytes, bytearray)):
        encoded = bytes(data)
    else:
        raise CanonicalJSONError("JSON input must be str, bytes, or bytearray")
    if len(encoded) > max_bytes:
        raise CanonicalJSONError(f"maximum JSON input size exceeded ({max_bytes})")
    try:
        text = encoded.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_int=lambda integer: _parse_int(integer, max_digits=max_integer_digits),
        )
    except CanonicalJSONError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise CanonicalJSONError(str(exc)) from exc
    # Revalidate parsed output to apply all recursive bounds.
    _check_value(
        value,
        depth=0,
        max_depth=max_depth,
        max_items=max_items,
        max_string_bytes=max_string_bytes,
        max_integer_digits=max_integer_digits,
        ancestors=set(),
    )
    return value


def sha256_hex(data: bytes | bytearray | str) -> str:
    """Return a lowercase SHA-256 digest for bytes or UTF-8 text."""

    if isinstance(data, str):
        data = data.encode("utf-8")
    elif isinstance(data, bytearray):
        data = bytes(data)
    if not isinstance(data, bytes):
        raise TypeError("sha256_hex expects bytes or str")
    return hashlib.sha256(data).hexdigest()


# Friendly aliases used by callers that prefer nouns over verbs.
canonical_json = canonical_bytes
canonical_json_bytes = canonical_bytes
strict_loads = loads_strict
strict_json_loads = loads_strict
