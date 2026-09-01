"""Bounded, read-only local dashboard for the Friday optimizer.

The dashboard is deliberately a thin observation surface.  It never starts a
model, changes a profile, writes to the memory database, or accepts a client
selected path or SQL statement.  Each request opens a fresh query-only view of
the bound sources and fails closed when an input is missing, corrupt, or too
large.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import re
import sqlite3
import stat
import threading
import time
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import parse_qs, urlsplit

from .canonical import canonical_bytes, loads_strict
from .decisions import DECISION_SCHEMA, OUTCOME_SCHEMA, SelectionPolicy
from .replay import DEFAULT_MIN_SAMPLES, ReplayEnv, ReplayError, evaluate as evaluate_offline, load_steps
from .memory import (
    APPLICATION_ID,
    MAX_PAYLOAD_BYTES,
    OptimizationMemoryV2,
    ReadOnlyMemoryView,
    USER_VERSION,
    _safe_path,
)
from .profiles import AtomicProfileStore, ProfileError
from .portfolio import MANIFEST_SCHEMA as PORTFOLIO_MANIFEST_SCHEMA, PortfolioError, SNAPSHOT_SCHEMA as PORTFOLIO_SNAPSHOT_SCHEMA, build_portfolio


HOST = LOOPBACK_HOST = "127.0.0.1"
DEFAULT_LIMIT = 100
MAX_HISTORY_ROWS = 500
MAX_VERIFY_ROWS = 10_000
MAX_RESPONSE_BYTES = 512 * 1024
MAX_TARGET_BYTES = 2048
MAX_ASSET_BYTES = 128 * 1024
MAX_DATASET_BYTES = 4 * 1024 * 1024
MAX_PROFILE_BYTES = 16 * 1024 * 1024
MAX_DATABASE_BYTES = 64 * 1024 * 1024
READ_TIMEOUT_SECONDS = 0.75
MAX_TEXT_BYTES = 256
MAX_JSON_DEPTH = 8
# Short aliases mirror the names used by the earlier local dashboards.
MAX_ASSET = MAX_ASSET_BYTES
MAX_RESPONSE = MAX_RESPONSE_BYTES
MAX_TARGET = MAX_TARGET_BYTES

CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)
_HEADERS = (
    ("Cache-Control", "no-store"),
    ("Content-Security-Policy", CSP),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Referrer-Policy", "no-referrer"),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Connection", "close"),
)


class DashboardError(ValueError):
    """A request is outside the bounded dashboard contract."""


class DashboardUnavailable(RuntimeError):
    """A bound source cannot be trusted for this request."""

    def __init__(self, reason: str = "unavailable") -> None:
        self.reason = reason if reason in _UNAVAILABLE_REASONS else "unavailable"
        super().__init__(self.reason)


class DatasetProvider(Protocol):
    """Optional immutable dataset provider used by the dashboard."""

    def as_dict(self) -> Mapping[str, Any]: ...


class ShadowProvider(Protocol):
    """Optional immutable shadow-decision provider."""

    def as_dict(self) -> Mapping[str, Any]: ...


class PortfolioProvider(Protocol):
    """Optional immutable portfolio manifest/snapshot provider."""

    def as_dict(self) -> Mapping[str, Any]: ...


_HASH_RE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,255}$", re.ASCII)
_SENSITIVE_WORDS = {"prompt", "text", "output", "response", "log", "trace", "source", "path"}
_UNAVAILABLE_REASONS = frozenset({"missing", "invalid", "integrity", "timeout", "bounded", "identity", "unavailable"})


def _path_signature(path: Path, *, sidecars: bool = False) -> tuple[Any, ...]:
    """Return a stable identity for a bound file and its existing ancestors."""

    try:
        absolute = Path(os.path.abspath(os.fspath(path)))
        components: list[tuple[str, int, int, int, int]] = []
        current = Path(absolute.anchor or os.sep)
        for part in absolute.parts[1:]:
            current /= part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise DashboardUnavailable("identity")
            is_leaf = current == absolute
            if not is_leaf and not stat.S_ISDIR(info.st_mode):
                raise DashboardUnavailable("identity")
            # Ancestor directory mtimes legitimately change when a sibling
            # temporary directory is created.  Bind ancestors by identity only;
            # retain size/mtime for the actual leaf file.
            components.append((part, int(info.st_dev), int(info.st_ino), int(info.st_size) if is_leaf else 0, int(info.st_mtime_ns) if is_leaf else 0))
        if not components or not stat.S_ISREG(absolute.stat().st_mode):
            raise DashboardUnavailable("missing")
        extra: list[tuple[str, int, int, int, int]] = []
        if sidecars:
            for suffix in ("-journal", "-wal", "-shm"):
                candidate = Path(f"{absolute}{suffix}")
                try:
                    info = candidate.lstat()
                except FileNotFoundError:
                    continue
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                    raise DashboardUnavailable("identity")
                extra.append((suffix, int(info.st_dev), int(info.st_ino), int(info.st_size), int(info.st_mtime_ns)))
        return tuple(components) + tuple(extra)
    except DashboardUnavailable:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise DashboardUnavailable("missing") from exc


def _jsonable(value: Any, *, max_depth: int = MAX_JSON_DEPTH) -> Any:
    """Parse/normalise provider values without ever exposing arbitrary objects."""

    if max_depth < 0:
        raise DashboardUnavailable("provider data is too deep")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DashboardUnavailable("provider data is non-finite")
        return value
    if isinstance(value, Mapping):
        if len(value) > 512:
            raise DashboardUnavailable("provider mapping is too large")
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise DashboardUnavailable("provider key is invalid")
            result[key] = _jsonable(child, max_depth=max_depth - 1)
        return result
    if isinstance(value, (tuple, list)):
        if len(value) > 512:
            raise DashboardUnavailable("provider sequence is too large")
        return [_jsonable(child, max_depth=max_depth - 1) for child in value]
    raise DashboardUnavailable("provider value is not JSON-compatible")


def _safe_text(value: Any, *, identifier: bool = False) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError:
        return None
    if len(encoded) > MAX_TEXT_BYTES or any(ord(char) < 0x20 for char in value):
        return None
    # A source/path-like value is never useful in this UI and can disclose the
    # machine layout.  Hashes are handled separately.
    lowered = value.lower()
    if value.startswith(("/", "~", "file:")) or "\\" in value:
        return None
    if any(token in lowered for token in ("prompt", "traceback", "model output", "raw log")):
        return None
    if identifier and not _ID_RE.fullmatch(value):
        return None
    return value


def _safe_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if abs(value) > 10**15:
        return None
    return value


def _safe_hash(value: Any) -> str | None:
    if isinstance(value, str) and _HASH_RE.fullmatch(value):
        return value
    return None


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


# Only these names can cross the payload-to-UI boundary.  In particular,
# prompt/model text, logs, paths, and arbitrary nested payloads are excluded.
_ALIASES: dict[str, tuple[str, ...]] = {
    "state": ("state", "systemstate", "readinessstate", "runtime_state"),
    "wait_reason": ("waitreason", "waitingreason", "reason", "blockreason"),
    "status": ("status", "resultstatus", "outcome"),
    "decision": ("decision", "finaldecision", "recommendation", "action"),
    "fingerprint": (
        "fingerprint", "environmentfingerprint", "hardwarefingerprint",
        "model_fingerprint", "workloadfingerprint", "fingerprinthash",
    ),
    "ood": ("ood", "outofdistribution", "oodstatus"),
    "ood_reason": ("oodreason", "ood_reason", "fingerprintoodreason"),
    "dataset_hash": ("datasethash", "datasetsha256", "snapshot_hash"),
    "candidate_hash": ("candidatehash", "candidatesha256"),
    "candidate": ("candidateid", "candidate"),
    "code_hash": ("codehash", "codesha256", "artifacthash", "artifactsha256", "sourcehash"),
    "profile_hash": ("profilehash",),
    "ttft_ms": ("ttftms", "ttft", "timeToFirstToken"),
    "decode_tps": ("decodetps", "tokspers", "tokenspersecond", "decode_tokens_per_second"),
    "mde": ("mde", "minimumdetectableeffect", "minimum_detectable_effect"),
    "ci_low": ("cilow", "confidenceinterval_low", "confidence_low", "lowerbound"),
    "ci_high": ("cihigh", "confidenceinterval_high", "confidence_high", "upperbound"),
    "correctness": ("correctness", "correctnessstatus", "correctnesspassed", "qualitygate"),
    "peak_memory_mb": ("peakmemorymb", "mlxpeakmemorymb", "memorypeakmb"),
    "peak_rss_mb": ("peakrssmb", "rsspeakmb", "processpeakmb"),
    "swap_before_mb": ("swapbeforemb", "swapbefore"),
    "swap_after_mb": ("swapaftermb", "swapafter"),
    "lease": ("lease", "lease_state", "leasevalid"),
    "pid": ("pid", "processid", "workerpid"),
    "fork": ("fork", "processmode", "executionmode"),
    "rollback": ("rollback", "rollbacklatched", "rollbackreason"),
}
_ALIASES = {name: tuple(_key(alias) for alias in aliases) for name, aliases in _ALIASES.items()}
_ALLOWED_NESTED_KEYS = {alias for aliases in _ALIASES.values() for alias in aliases}


def _find_allowed(value: Any, aliases: Sequence[str], *, depth: int = 0) -> Any:
    if depth > 5 or not isinstance(value, Mapping):
        return None
    for name, child in value.items():
        if not isinstance(name, str):
            continue
        normal = _key(name)
        if normal in aliases:
            return child
    for name, child in value.items():
        if not isinstance(name, str) or _key(name) in _SENSITIVE_WORDS:
            continue
        if isinstance(child, Mapping):
            found = _find_allowed(child, aliases, depth=depth + 1)
            if found is not None:
                return found
    return None


def _scalar(value: Any, *, identifier: bool = False) -> str | int | float | bool | None:
    if isinstance(value, bool):
        return value
    number = _safe_number(value)
    if number is not None:
        return number
    return _safe_text(value, identifier=identifier)


def _summary_from_payload(payload: Mapping[str, Any], *, row: sqlite3.Row) -> dict[str, Any]:
    def value(name: str, *, identifier: bool = False) -> Any:
        raw = _find_allowed(payload, _ALIASES[name])
        if name.endswith("_hash") or name == "profile_hash":
            return _safe_hash(raw)
        return _scalar(raw, identifier=identifier)

    status = value("status", identifier=True) or _safe_text(row["kind"], identifier=True) or "unknown"
    result: dict[str, Any] = {
        "seq": int(row["seq"]),
        "record_id": _safe_text(str(row["record_id"]), identifier=True) or "unknown",
        "kind": _safe_text(str(row["kind"]), identifier=True) or "unknown",
        "quality": _safe_text(str(row["quality"]), identifier=True) or "unknown",
        "phase": _safe_text(str(row["phase"]), identifier=True) or "unknown",
        "created_at": _safe_text(str(row["created_at"])) or "",
        "status": status,
    }
    for output, identifier in (
        ("state", True), ("wait_reason", False), ("fingerprint", True),
        ("ood_reason", False), ("dataset_hash", False), ("candidate_hash", False),
        ("code_hash", False), ("candidate", True), ("profile_hash", False),
        ("ttft_ms", False), ("decode_tps", False), ("mde", False),
        ("correctness", True), ("peak_memory_mb", False), ("peak_rss_mb", False),
        ("swap_before_mb", False), ("swap_after_mb", False), ("pid", False),
        ("fork", True), ("lease", True), ("rollback", False),
    ):
        item = value(output, identifier=identifier)
        if item is not None:
            result[output] = item
    ood = value("ood")
    if isinstance(ood, bool):
        result["ood"] = ood
    elif isinstance(ood, str):
        result["ood"] = ood
    low, high = value("ci_low"), value("ci_high")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        result["ci"] = {"low": low, "high": high}
    return result


def _canonical_hash(value: Any) -> str:
    try:
        return hashlib.sha256(canonical_bytes(value, max_bytes=MAX_DATASET_BYTES)).hexdigest()
    except (TypeError, ValueError):
        raise DashboardUnavailable("dataset is not canonical")


def _error_payload(reason: str = "unavailable") -> dict[str, Any]:
    bounded_reason = reason if reason in _UNAVAILABLE_REASONS else "unavailable"
    return {
        "schema_version": 1,
        "read_only": True,
        "data_state": "unavailable",
        "error": "unavailable",
        "reason": bounded_reason,
    }


def _portfolio_payload(data_state: str, *, value: Mapping[str, Any] | None = None, reason: str = "unavailable") -> dict[str, Any]:
    """Project only bounded portfolio status; never expose source paths."""
    result: dict[str, Any] = {
        "schema_version": 1,
        "read_only": True,
        "data_state": data_state,
        "snapshot_hash": None,
        "manifest_hash": None,
        "models": [],
        "model_count": 0,
        "next_action": None,
        "no_model_load": True,
        "no_download": True,
        "no_activation": True,
    }
    if data_state == "unavailable":
        result.update({"error": "unavailable", "reason": reason if reason in _UNAVAILABLE_REASONS else "unavailable"})
        return result
    if not isinstance(value, Mapping):
        return _portfolio_payload("unavailable", reason="invalid")
    snapshot_hash = _safe_hash(value.get("snapshot_sha256"))
    manifest_hash = _safe_hash(value.get("manifest_sha256"))
    entries = value.get("models")
    if snapshot_hash is None or manifest_hash is None or not isinstance(entries, (list, tuple)) or len(entries) != 4:
        return _portfolio_payload("unavailable", reason="invalid")
    body = {key: child for key, child in value.items() if key != "snapshot_sha256"}
    try:
        if _canonical_hash(body) != snapshot_hash:
            return _portfolio_payload("unavailable", reason="integrity")
    except DashboardUnavailable:
        return _portfolio_payload("unavailable", reason="invalid")
    models: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            return _portfolio_payload("unavailable", reason="invalid")
        size = _safe_text(entry.get("size"), identifier=True)
        model_id = _safe_text(entry.get("model_id"), identifier=True)
        status = _safe_text(entry.get("status"), identifier=True)
        identity_hash = _safe_hash(entry.get("identity_hash"))
        if size is None or model_id is None or status is None:
            return _portfolio_payload("unavailable", reason="invalid")
        counts = entry.get("evidence_counts")
        if not isinstance(counts, Mapping):
            return _portfolio_payload("unavailable", reason="invalid")
        safe_counts = {
            str(name): count
            for name, count in counts.items()
            if isinstance(name, str) and type(count) is int and 0 <= count <= MAX_VERIFY_ROWS
        }
        if len(safe_counts) != len(counts):
            return _portfolio_payload("unavailable", reason="invalid")
        usable = entry.get("usable_records")
        if type(usable) is not int or not 0 <= usable <= MAX_VERIFY_ROWS:
            return _portfolio_payload("unavailable", reason="invalid")
        row: dict[str, Any] = {
            "size": size,
            "model_id": model_id,
            "status": status,
            "identity_hash": None if identity_hash is None else identity_hash[:12],
            "identity_hash_short": None if identity_hash is None else identity_hash[:12],
            "evidence_counts": safe_counts,
            "usable_records": usable,
        }
        next_point = entry.get("next_safe_measurement")
        if isinstance(next_point, Mapping):
            action = _safe_text(next_point.get("action"), identifier=True)
            point_size = _safe_text(next_point.get("size"), identifier=True)
            requires_start = next_point.get("requires_user_start")
            if action is not None and point_size is not None and isinstance(requires_start, bool):
                row["next_action"] = {"action": action, "size": point_size, "requires_user_start": requires_start}
        models.append(row)
    next_point = value.get("next_safe_measurement")
    next_action = None
    if isinstance(next_point, Mapping):
        action = _safe_text(next_point.get("action"), identifier=True)
        point_size = _safe_text(next_point.get("size"), identifier=True)
        requires_start = next_point.get("requires_user_start")
        if action is not None and point_size is not None and isinstance(requires_start, bool):
            next_action = {"action": action, "size": point_size, "requires_user_start": requires_start}
    result.update({"data_state": "available", "snapshot_hash": snapshot_hash, "manifest_hash": manifest_hash, "models": models, "model_count": len(models), "next_action": next_action})
    return result


def _dataset_payload(
    data_state: str,
    *,
    dataset_hash: str | None = None,
    card: Mapping[str, Any] | None = None,
    record_count: int | None = None,
    splits: Mapping[str, int] | None = None,
    leakage: Mapping[str, Any] | None = None,
    claim: str | None = None,
    reason: str = "unavailable",
) -> dict[str, Any]:
    """Return the stable dataset endpoint envelope for every source state."""

    value: dict[str, Any] = {
        "schema_version": 1,
        "read_only": True,
        "data_state": data_state,
        "dataset_hash": dataset_hash if _safe_hash(dataset_hash) else None,
        "card": dict(card) if isinstance(card, Mapping) else None,
        "record_count": record_count if type(record_count) is int and 0 <= record_count <= MAX_VERIFY_ROWS else None,
        "splits": dict(splits) if isinstance(splits, Mapping) else None,
        "leakage": dict(leakage) if isinstance(leakage, Mapping) else None,
        "claim": claim if _safe_text(claim, identifier=True) else None,
    }
    if data_state == "unavailable":
        value["error"] = "unavailable"
        value["reason"] = reason if reason in _UNAVAILABLE_REASONS else "unavailable"
    return value


def _dataset_card(card: Mapping[str, Any]) -> dict[str, Any]:
    """Project only bounded, non-sensitive card metadata to the UI."""

    result: dict[str, Any] = {"schema_version": 1}
    for name in ("smoke_only",):
        if isinstance(card.get(name), bool):
            result[name] = card[name]
    for name in ("eligible_records", "total_records", "duplicate_count", "excluded_records"):
        value = card.get(name)
        if type(value) is int and 0 <= value <= MAX_VERIFY_ROWS:
            result[name] = value
    quality = card.get("quality_counts")
    if isinstance(quality, Mapping):
        result["quality_counts"] = {
            str(key): int(value)
            for key, value in quality.items()
            if isinstance(key, str) and type(value) is int and 0 <= value <= MAX_VERIFY_ROWS
        }
    for name in ("leakage", "missingness", "censoring", "coverage"):
        nested = card.get(name)
        if not isinstance(nested, Mapping):
            continue
        safe: dict[str, Any] = {}
        for key, value in nested.items():
            if not isinstance(key, str) or _key(key) in _SENSITIVE_WORDS:
                continue
            scalar = _scalar(value)
            if scalar is not None:
                safe[key[:64]] = scalar
        result[name] = safe
    return result


class DashboardService:
    """Read-only service bound to immutable paths/providers at construction."""

    def __init__(
        self,
        database_path: str | os.PathLike[str],
        profile_path: str | os.PathLike[str] | None = None,
        dataset_path: str | os.PathLike[str] | None = None,
        *,
        profiles_path: str | os.PathLike[str] | None = None,
        dataset_provider: DatasetProvider | DatasetSnapshotLike | None = None,
        shadow_provider: ShadowProvider | ShadowDecisionLike | None = None,
        dataset: DatasetProvider | DatasetSnapshotLike | None = None,
        dataset_snapshot: DatasetProvider | DatasetSnapshotLike | None = None,
        shadow: ShadowProvider | ShadowDecisionLike | None = None,
        portfolio_path: str | os.PathLike[str] | None = None,
        portfolio_provider: PortfolioProvider | Any | None = None,
        portfolio_snapshot: PortfolioProvider | Any | None = None,
        portfolio: PortfolioProvider | Any | None = None,
        expected_dataset_hash: str | None = None,
        dataset_sha256: str | None = None,
    ) -> None:
        if profile_path is not None and profiles_path is not None and Path(profile_path) != Path(profiles_path):
            raise DashboardError("profile paths conflict")
        if dataset_provider is not None and dataset is not None and dataset_provider is not dataset:
            raise DashboardError("dataset providers conflict")
        if dataset_snapshot is not None and (dataset_provider is not None or dataset is not None) and dataset_snapshot is not dataset_provider and dataset_snapshot is not dataset:
            raise DashboardError("dataset providers conflict")
        if shadow_provider is not None and shadow is not None and shadow_provider is not shadow:
            raise DashboardError("shadow providers conflict")
        if portfolio_provider is not None and portfolio_snapshot is not None and portfolio_provider is not portfolio_snapshot:
            raise DashboardError("portfolio providers conflict")
        if portfolio_snapshot is not None and portfolio is not None and portfolio_snapshot is not portfolio:
            raise DashboardError("portfolio providers conflict")
        if expected_dataset_hash is not None and dataset_sha256 is not None and expected_dataset_hash != dataset_sha256:
            raise DashboardError("dataset hashes conflict")
        self.database_path = Path(database_path)
        self.profile_path = Path(profile_path if profile_path is not None else profiles_path) if (profile_path is not None or profiles_path is not None) else None
        self.dataset_path = Path(dataset_path) if dataset_path is not None else None
        self.dataset_provider = dataset_provider if dataset_provider is not None else dataset_snapshot if dataset_snapshot is not None else dataset
        self.shadow_provider = shadow_provider if shadow_provider is not None else shadow
        self.portfolio_path = Path(portfolio_path) if portfolio_path is not None else None
        self.portfolio_provider = portfolio_provider if portfolio_provider is not None else portfolio_snapshot if portfolio_snapshot is not None else portfolio
        configured_hash = expected_dataset_hash if expected_dataset_hash is not None else dataset_sha256
        if configured_hash is not None and not _safe_hash(configured_hash):
            raise DashboardError("dataset hash is invalid")
        provider_hash = getattr(self.dataset_provider, "dataset_hash", None)
        self._dataset_expected_hash = configured_hash if configured_hash is not None else provider_hash if _safe_hash(provider_hash) else None
        self._dataset_hash_lock = threading.RLock()

    def _memory(self) -> tuple[list[sqlite3.Row], str]:
        view: ReadOnlyMemoryView | None = None
        try:
            path = _safe_path(self.database_path)
            before = _path_signature(path, sidecars=True)
            total_bytes = sum(item[3] for item in before)
            if total_bytes > MAX_DATABASE_BYTES:
                raise DashboardUnavailable("bounded")
            view = OptimizationMemoryV2.open_read_only(path, max_records=MAX_VERIFY_ROWS, max_payload_bytes=MAX_PAYLOAD_BYTES)
            if not view.integrity().ok:
                raise DashboardUnavailable("integrity")
            deadline = time.monotonic() + READ_TIMEOUT_SECONDS
            with view.read_connection() as connection:
                if _path_signature(path, sidecars=True) != before:
                    raise DashboardUnavailable("identity")
                connection.set_progress_handler(lambda: 1 if time.monotonic() >= deadline else 0, 1000)
                if connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
                    raise DashboardUnavailable("invalid")
                if connection.execute("PRAGMA user_version").fetchone()[0] != USER_VERSION:
                    raise DashboardUnavailable("invalid")
                count = int(connection.execute("SELECT COUNT(*) FROM optimization_records").fetchone()[0])
                if count > MAX_VERIFY_ROWS:
                    raise DashboardUnavailable("bounded")
                rows = connection.execute("SELECT * FROM optimization_records ORDER BY seq ASC LIMIT ?", (MAX_VERIFY_ROWS + 1,)).fetchall()
                if len(rows) > MAX_VERIFY_ROWS:
                    raise DashboardUnavailable("bounded")
                material = [[row["seq"], row["record_id"], row["record_hash"]] for row in rows]
                revision = _canonical_hash(material)
                if _path_signature(path, sidecars=True) != before:
                    raise DashboardUnavailable("identity")
                return rows, revision
        except DashboardUnavailable:
            raise
        except (sqlite3.Error, OSError, ValueError, TypeError, UnicodeError) as exc:
            message = str(exc).lower()
            reason = "timeout" if "interrupt" in message or "busy" in message or "locked" in message else "invalid"
            raise DashboardUnavailable(reason) from exc
        except Exception as exc:
            raise DashboardUnavailable("unavailable") from exc
        finally:
            if view is not None:
                view.close()

    @staticmethod
    def _payload(row: sqlite3.Row) -> Mapping[str, Any]:
        try:
            parsed = loads_strict(bytes(row["payload"]), max_bytes=MAX_PAYLOAD_BYTES, max_depth=MAX_JSON_DEPTH)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise DashboardUnavailable("memory payload is invalid") from exc
        if not isinstance(parsed, Mapping):
            raise DashboardUnavailable("memory payload is not an object")
        return parsed

    def history(self, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= MAX_HISTORY_ROWS:
            raise DashboardError("limit is outside the registered range")
        try:
            rows, revision = self._memory()
        except DashboardUnavailable as exc:
            value = _error_payload(exc.reason)
            value.update({"revision": None, "total": 0, "returned": 0, "truncated": False, "history": []})
            return value
        recent: list[dict[str, Any]] = []
        for row in reversed(rows[-limit:]):
            recent.append(_summary_from_payload(self._payload(row), row=row))
        return {
            "schema_version": 1,
            "read_only": True,
            "data_state": "available" if rows else "empty",
            "revision": revision,
            "total": len(rows),
            "returned": len(recent),
            "truncated": len(rows) > limit,
            "history": recent,
        }

    def _profile_data(self) -> dict[str, Any]:
        if self.profile_path is None:
            return {"schema_version": 1, "read_only": True, "data_state": "not_configured", "profiles": []}
        try:
            before = _path_signature(self.profile_path)
            if before[-1][3] > MAX_PROFILE_BYTES:
                raise DashboardUnavailable("bounded")
            value = AtomicProfileStore(self.profile_path).load()
            if _path_signature(self.profile_path) != before:
                raise DashboardUnavailable("identity")
            if not isinstance(value, Mapping):
                raise DashboardUnavailable("profile is malformed")
            profiles_value = value.get("profiles")
            if not isinstance(profiles_value, Mapping):
                raise DashboardUnavailable("profile index is malformed")
            if len(profiles_value) > MAX_HISTORY_ROWS:
                raise DashboardUnavailable("profile index exceeds dashboard bound")
            profiles: list[dict[str, Any]] = []
            for item in profiles_value.values():
                if not isinstance(item, Mapping):
                    raise DashboardUnavailable("profile entry is malformed")
                entry: dict[str, Any] = {}
                for name in ("profile_id", "fingerprint", "candidate", "qualified", "version", "profile_hash"):
                    raw = item.get(name)
                    if name == "profile_hash":
                        clean = _safe_hash(raw)
                    elif name in {"qualified"}:
                        clean = raw if isinstance(raw, bool) else None
                    elif name in {"version"}:
                        clean = raw if type(raw) is int and 0 < raw <= 10**12 else None
                    else:
                        clean = _safe_text(raw, identifier=True)
                    if clean is not None:
                        entry[name] = clean
                if set(("profile_id", "fingerprint", "candidate", "qualified", "version", "profile_hash")) - set(entry):
                    raise DashboardUnavailable("profile entry is invalid")
                profiles.append(entry)
            profiles.sort(key=lambda item: item["profile_id"])
            if _path_signature(self.profile_path) != before:
                raise DashboardUnavailable("identity")
            return {
                "schema_version": 1,
                "read_only": True,
                "data_state": "available",
                "version": value.get("version") if type(value.get("version")) is int else 0,
                "mode": _safe_text(value.get("mode"), identifier=True) or "unknown",
                "baseline": _safe_text(value.get("baseline"), identifier=True),
                "active": _safe_text(value.get("active"), identifier=True),
                "pinned": _safe_text(value.get("pinned"), identifier=True),
                "rollback_latched": value.get("rollback_latched") if isinstance(value.get("rollback_latched"), bool) else False,
                "rollback_reason": _safe_text(value.get("rollback_reason")),
                "profiles": profiles,
            }
        except DashboardUnavailable:
            raise
        except (ProfileError, OSError, TypeError, ValueError, UnicodeError) as exc:
            return _error_payload("invalid")
        except Exception:
            return _error_payload("unavailable")

    def profiles(self) -> dict[str, Any]:
        try:
            return self._profile_data()
        except DashboardUnavailable as exc:
            value = _error_payload(exc.reason)
            value["profiles"] = []
            return value

    @staticmethod
    def _provider_value(provider: Any) -> Mapping[str, Any]:
        if provider is None:
            raise DashboardUnavailable("provider is not configured")
        value = provider
        if hasattr(provider, "as_dict") and callable(provider.as_dict):
            value = provider.as_dict()
        elif hasattr(provider, "snapshot") and callable(provider.snapshot):
            value = provider.snapshot()
        if not isinstance(value, Mapping):
            raise DashboardUnavailable("provider returned invalid data")
        return _jsonable(value)

    def dataset(self) -> dict[str, Any]:
        if self.dataset_provider is None and self.dataset_path is None:
            return _dataset_payload("not_configured")
        try:
            before: tuple[Any, ...] | None = None
            materialized_bytes: bytes | None = None
            if self.dataset_provider is not None:
                value = dict(self._provider_value(self.dataset_provider))
            else:
                assert self.dataset_path is not None
                before = _path_signature(self.dataset_path)
                raw = self.dataset_path.read_bytes()
                if _path_signature(self.dataset_path) != before:
                    raise DashboardUnavailable("identity")
                if len(raw) > MAX_DATASET_BYTES:
                    raise DashboardUnavailable("bounded")
                value = loads_strict(raw, max_bytes=MAX_DATASET_BYTES, max_depth=MAX_JSON_DEPTH, max_items=100_000)
                if not isinstance(value, Mapping):
                    raise DashboardUnavailable("invalid")
                value = dict(value)
                # The materialized v1 artifact is the canonical payload itself;
                # its identity is the external SHA-256 of these exact bytes.
                # Re-serializing catches whitespace/key-order changes before
                # any metadata is shown.  An embedded hash would be a second,
                # ambiguous identity contract and is rejected for this path.
                if "sha256" in value or "dataset_hash" in value:
                    raise DashboardUnavailable("invalid")
                try:
                    materialized_bytes = canonical_bytes(
                        value,
                        max_bytes=MAX_DATASET_BYTES,
                        max_depth=MAX_JSON_DEPTH,
                        max_items=100_000,
                    )
                except (TypeError, ValueError) as exc:
                    raise DashboardUnavailable("invalid") from exc
                if raw != materialized_bytes:
                    raise DashboardUnavailable("invalid")
            # Dataset snapshots are versioned immutable artifacts.  Missing or
            # malformed identity is unavailable; accepting a card-only object
            # would turn unverified metadata into a data source.
            if value.get("schema_version") != 1:
                raise DashboardUnavailable("invalid")
            if materialized_bytes is not None:
                advertised = hashlib.sha256(materialized_bytes).hexdigest()
            else:
                # Immutable providers may expose a DatasetSnapshot-style
                # self-hash.  Validate it only against the explicitly
                # non-self projection, then use the canonical provider bytes
                # as the dashboard identity.
                embedded = value.get("sha256") or value.get("dataset_hash")
                if "sha256" in value or "dataset_hash" in value:
                    body = {key: child for key, child in value.items() if key not in {"sha256", "dataset_hash"}}
                    if not _safe_hash(embedded) or _canonical_hash(body) != embedded:
                        raise DashboardUnavailable("integrity")
                    value = body
                try:
                    materialized_bytes = canonical_bytes(value, max_bytes=MAX_DATASET_BYTES, max_depth=MAX_JSON_DEPTH, max_items=100_000)
                except (TypeError, ValueError) as exc:
                    raise DashboardUnavailable("invalid") from exc
                advertised = hashlib.sha256(materialized_bytes).hexdigest()
            with self._dataset_hash_lock:
                if self._dataset_expected_hash is not None and advertised != self._dataset_expected_hash:
                    raise DashboardUnavailable("integrity")
                # Pin the first verified snapshot when no external expected
                # hash was supplied; a later same-path rewrite cannot become a
                # new dataset merely because it is still canonical JSON.
                if self._dataset_expected_hash is None:
                    self._dataset_expected_hash = advertised
            card = value.get("card")
            records = value.get("records")
            splits = value.get("splits")
            if not isinstance(card, Mapping) or not isinstance(records, (list, tuple)) or not isinstance(splits, Mapping):
                raise DashboardUnavailable("invalid")
            if set(splits) != {"train", "validation", "holdout"}:
                raise DashboardUnavailable("invalid")
            if card.get("schema_version", 1) != 1:
                raise DashboardUnavailable("invalid")
            split_counts: dict[str, int] = {}
            for name, ids in splits.items():
                if not isinstance(name, str) or not isinstance(ids, (list, tuple)) or len(ids) > MAX_VERIFY_ROWS:
                    raise DashboardUnavailable("bounded")
                if any(not isinstance(item, str) or not item for item in ids):
                    raise DashboardUnavailable("invalid")
                split_counts[name[:64]] = len(ids)
            if len(records) > MAX_VERIFY_ROWS:
                raise DashboardUnavailable("bounded")
            if any(not isinstance(item, Mapping) for item in records):
                raise DashboardUnavailable("invalid")
            card_view = _dataset_card(card)
            leakage = card_view.get("leakage") if isinstance(card_view.get("leakage"), Mapping) else None
            claim = card.get("claim") if _safe_text(card.get("claim"), identifier=True) else None
            result = _dataset_payload(
                "available",
                dataset_hash=advertised,
                card=card_view,
                record_count=len(records),
                splits=split_counts,
                leakage=leakage,
                claim=claim,
            )
            if before is not None and _path_signature(self.dataset_path) != before:
                raise DashboardUnavailable("identity")
            return result
        except DashboardUnavailable as exc:
            return _dataset_payload("unavailable", reason=exc.reason)
        except (OSError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            return _dataset_payload("unavailable", reason="invalid")
        except Exception:
            return _dataset_payload("unavailable", reason="unavailable")

    def shadow(self) -> dict[str, Any]:
        if self.shadow_provider is None:
            return {"schema_version": 1, "read_only": True, "data_state": "not_configured", "decision": None}
        try:
            value = self._provider_value(self.shadow_provider)
            result: dict[str, Any] = {"schema_version": 1, "read_only": True, "data_state": "available"}
            for name in ("fingerprint_hash", "evidence_hash", "candidate_id", "status", "feasible", "reasons"):
                raw = value.get(name)
                if name.endswith("_hash"):
                    clean = _safe_hash(raw)
                elif name == "feasible":
                    clean = raw if isinstance(raw, bool) else None
                elif name == "reasons":
                    clean = [_safe_text(item) for item in raw[:16] if _safe_text(item)] if isinstance(raw, (list, tuple)) else None
                else:
                    clean = _safe_text(raw, identifier=True)
                if clean is not None:
                    result[name] = clean
            for key in ("ratios", "cis"):
                raw = value.get(key)
                if isinstance(raw, Mapping):
                    result[key] = {str(name): _safe_number(item) for name, item in raw.items() if isinstance(name, str) and _safe_number(item) is not None}
            return result
        except DashboardUnavailable as exc:
            value = _error_payload(exc.reason)
            value["decision"] = None
            return value
        except (TypeError, ValueError, OSError):
            value = _error_payload("invalid")
            value["decision"] = None
            return value
        except Exception:
            value = _error_payload("unavailable")
            value["decision"] = None
            return value

    def portfolio(self) -> dict[str, Any]:
        if self.portfolio_provider is None and self.portfolio_path is None:
            return _portfolio_payload("not_configured")
        try:
            if self.portfolio_provider is not None:
                value = dict(self._provider_value(self.portfolio_provider))
                if value.get("schema") == PORTFOLIO_MANIFEST_SCHEMA:
                    value = build_portfolio(value).as_dict()
            else:
                assert self.portfolio_path is not None
                before = _path_signature(self.portfolio_path)
                raw = self.portfolio_path.read_bytes()
                if len(raw) > MAX_DATASET_BYTES:
                    raise DashboardUnavailable("bounded")
                if _path_signature(self.portfolio_path) != before:
                    raise DashboardUnavailable("identity")
                parsed = loads_strict(raw, max_bytes=MAX_DATASET_BYTES, max_depth=MAX_JSON_DEPTH, max_items=100_000)
                if not isinstance(parsed, Mapping) or canonical_bytes(parsed, max_bytes=MAX_DATASET_BYTES, max_depth=MAX_JSON_DEPTH, max_items=100_000) != raw:
                    raise DashboardUnavailable("invalid")
                if parsed.get("schema") == PORTFOLIO_MANIFEST_SCHEMA:
                    value = build_portfolio(raw).as_dict()
                else:
                    value = dict(parsed)
                if _path_signature(self.portfolio_path) != before:
                    raise DashboardUnavailable("identity")
            if value.get("schema") != PORTFOLIO_SNAPSHOT_SCHEMA:
                raise DashboardUnavailable("invalid")
            return _portfolio_payload("available", value=value)
        except DashboardUnavailable as exc:
            return _portfolio_payload("unavailable", reason=exc.reason)
        except (PortfolioError, OSError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
            return _portfolio_payload("unavailable", reason="invalid")
        except Exception:
            return _portfolio_payload("unavailable", reason="unavailable")

    def decisions(self, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
        """Read-only view of the RL-ready decision log and its estimate.

        The panel exists so a logged decision is visible the same day it is
        made.  It reports estimates together with their status; an estimate
        below the sample floor is shown as ``insufficient_data`` and is not a
        result.
        """

        if type(limit) is not int or not 1 <= limit <= MAX_HISTORY_ROWS:
            raise DashboardError("limit is outside the registered range")
        empty = {"revision": None, "total": 0, "labelled": 0, "observed": 0, "censoring": {},
                 "policies": {}, "actions": {}, "decisions": [], "estimates": {},
                 "learning_claim": False, "no_activation": True}
        try:
            rows, revision = self._memory()
        except DashboardUnavailable as exc:
            value = _error_payload(exc.reason)
            value.update(empty)
            return value
        logged: list[Mapping[str, Any]] = []
        outcomes: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if str(row["kind"]) != "system":
                continue
            try:
                payload = self._payload(row)
            except DashboardUnavailable:
                continue
            schema = payload.get("schema")
            if schema == DECISION_SCHEMA and isinstance(payload.get("decision_id"), str):
                logged.append(payload)
            elif schema == OUTCOME_SCHEMA and isinstance(payload.get("decision_id"), str):
                outcomes[payload["decision_id"]] = payload
        known = {payload["decision_id"] for payload in logged}
        paired = {key: value for key, value in outcomes.items() if key in known}
        table = []
        for payload in reversed(logged[-limit:]):
            outcome = paired.get(payload["decision_id"], {})
            table.append({
                "decision_id": _safe_text(payload.get("decision_id"), identifier=True),
                "policy_id": _safe_text(payload.get("policy_id"), identifier=True),
                "rule": _safe_text(payload.get("selection_rule"), identifier=True),
                "chosen": _safe_text(payload.get("chosen"), identifier=True),
                "propensity": _safe_number(payload.get("propensity")),
                "actions": len(payload.get("candidate_set") or ()),
                "censoring": _safe_text(outcome.get("censoring"), identifier=True),
                "reward": _safe_number(outcome.get("reward")),
            })
        estimates: dict[str, Any] = {}
        try:
            steps = load_steps(list(logged) + list(paired.values()))
            environment = ReplayEnv(steps)
            estimates = {
                name: estimate.as_dict()
                for name, estimate in evaluate_offline(
                    environment, SelectionPolicy("deterministic_order_v1"),
                    min_samples=DEFAULT_MIN_SAMPLES,
                ).items()
            }
        except (ReplayError, ValueError, TypeError):
            estimates = {}
        observed = sum(1 for value in paired.values() if value.get("censoring") == "observed")
        return {
            "schema_version": 1, "read_only": True,
            "data_state": "available" if logged else "empty",
            "revision": revision, "total": len(logged), "labelled": len(paired), "observed": observed,
            "censoring": dict(sorted(Counter(str(value.get("censoring")) for value in paired.values()).items())),
            "policies": dict(sorted(Counter(str(value.get("policy_id")) for value in logged).items())),
            "actions": dict(sorted(Counter(str(value.get("chosen")) for value in logged).items())),
            "decisions": table, "estimates": estimates,
            "min_samples": DEFAULT_MIN_SAMPLES,
            "learning_claim": False, "no_activation": True,
        }

    def status(self) -> dict[str, Any]:
        status: dict[str, Any] = {"schema_version": 1, "read_only": True, "service": "friday_optimizer", "data_state": "available"}
        try:
            rows, revision = self._memory()
            status.update({
                "data_state": "available" if rows else "empty",
                "memory_revision": revision,
                "history_count": len(rows),
                "kinds": dict(sorted(Counter(str(row["kind"]) for row in rows).items())),
            })
            if rows:
                latest = rows[-1]
                status["latest"] = _summary_from_payload(self._payload(latest), row=latest)
        except DashboardUnavailable:
            status.update(_error_payload("unavailable"))
        try:
            status["profiles"] = self._profile_data()
        except DashboardUnavailable:
            status["profiles"] = _error_payload("unavailable")
        try:
            status["dataset"] = self.dataset()
        except DashboardUnavailable:
            status["dataset"] = _error_payload("unavailable")
        try:
            status["shadow"] = self.shadow()
        except DashboardUnavailable:
            status["shadow"] = _error_payload("unavailable")
        status["portfolio"] = self.portfolio()
        required_components = ((self.profile_path is not None, status.get("profiles")), (self.dataset_path is not None, status.get("dataset")), (self.dataset_provider is not None, status.get("dataset")), (self.portfolio_path is not None or self.portfolio_provider is not None, status.get("portfolio")))
        if any(configured and isinstance(component, Mapping) and component.get("data_state") == "unavailable" for configured, component in required_components):
            status["data_state"] = "unavailable"
            status["error"] = "unavailable"
        latest = status.get("latest")
        if isinstance(latest, Mapping):
            for key in ("state", "wait_reason", "fingerprint", "ood", "ood_reason"):
                if key in latest:
                    status[key] = latest[key]
        return status

    # Compatibility name used by earlier local dashboards.
    snapshot = status


class DatasetSnapshotLike(Protocol):
    def as_dict(self) -> Mapping[str, Any]: ...


class ShadowDecisionLike(Protocol):
    def as_dict(self) -> Mapping[str, Any]: ...


HTML = """<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#070b12"><title>Friday Optimizer · Local History</title>
<link rel="stylesheet" href="/assets/app.css"></head><body>
<main><header><p class="eyebrow">PROJECT_FRIDAY / LOCAL CONTROL PLANE</p><h1>Optimizer<br><span>History</span></h1>
<p class="lede">Read-only Hardware- und Gemma-Optimierung. Nur lokale Evidenz, keine Prompts, keine Schreib-API.</p></header>
<section class="state" aria-live="polite"><div><small>system state</small><strong id="state">loading…</strong></div><div><small>wait reason</small><strong id="wait">—</strong></div><div><small>data state</small><strong id="data">loading…</strong></div><div><small>fingerprint / OOD</small><strong id="fingerprint">—</strong></div></section>
<section class="grid"><article><h2>Runtime</h2><dl id="runtime"><div><dt>TTFT</dt><dd>—</dd></div><div><dt>Decode tokens/s</dt><dd>—</dd></div><div><dt>CI / MDE</dt><dd>—</dd></div><div><dt>Correctness</dt><dd>—</dd></div><div><dt>Peak / RSS</dt><dd>—</dd></div><div><dt>Swap before / after</dt><dd>—</dd></div><div><dt>Lease / PID / mode</dt><dd>—</dd></div></dl></article>
<article><h2>Bindings</h2><dl id="bindings"><div><dt>Dataset</dt><dd>—</dd></div><div><dt>Candidate</dt><dd>—</dd></div><div><dt>Code</dt><dd>—</dd></div><div><dt>Profile</dt><dd>—</dd></div><div><dt>Rollback</dt><dd>—</dd></div></dl></article></section>
<section class="panel"><div class="panel-head"><h2>Chronological history</h2><span id="count">—</span></div><div class="table-wrap"><table><thead><tr><th>Time</th><th>Kind</th><th>Status</th><th>TTFT</th><th>Decode/s</th><th>Correctness</th><th>Memory</th></tr></thead><tbody id="history"><tr><td colspan="7">Loading…</td></tr></tbody></table></div></section>
<section class="panel"><div class="panel-head"><h2>Decision log (RL-ready)</h2><span id="dcount">—</span></div><div class="table-wrap"><table><thead><tr><th>Policy</th><th>Rule</th><th>Action</th><th>Propensity</th><th>Actions</th><th>Censoring</th><th>Reward</th></tr></thead><tbody id="decisions"><tr><td colspan="7">Loading…</td></tr></tbody></table></div></section>
<footer>friday@local · read-only · loopback-only · <span id="revision">revision pending</span></footer></main><script src="/assets/app.js" defer></script></body></html>"""

CSS = r""":root{color-scheme:dark;--bg:#070b12;--panel:#0d1420;--line:#253247;--ink:#e7eef8;--muted:#8ea0b8;--accent:#67e8f9;--ok:#7cf29a;--warn:#f6cf70;--bad:#ff8095;font:15px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}*{box-sizing:border-box}html,body{margin:0;min-width:320px;background:var(--bg);color:var(--ink)}body{background:radial-gradient(circle at 90% 0,#112438 0,transparent 42rem),var(--bg)}main{width:min(1180px,calc(100% - 32px));margin:auto;padding:28px 0 54px}header{border-top:1px solid var(--accent);padding-top:14px}.eyebrow,small,dt,.panel-head span{color:var(--muted);font-size:.68rem;letter-spacing:.12em;text-transform:uppercase}.eyebrow{color:var(--accent)}h1{font-size:clamp(3.5rem,11vw,8.5rem);font-weight:500;line-height:.82;letter-spacing:-.09em;margin:50px 0 26px}h1 span{color:transparent;-webkit-text-stroke:1px var(--accent)}.lede{max-width:55ch;color:var(--muted);line-height:1.6}.state,.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);margin:24px 0}.state>div,.grid article{background:var(--panel);padding:16px}.state strong{display:block;margin-top:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--accent);font-weight:500}.grid{grid-template-columns:1fr 1fr;background:none;gap:16px}.grid article,.panel{border:1px solid var(--line);border-radius:6px}.grid h2,.panel h2{font-size:.9rem;font-weight:500;letter-spacing:.08em;text-transform:uppercase;margin:0 0 14px}.grid dl{display:grid;grid-template-columns:1fr 1fr;gap:1px;margin:0;background:var(--line)}dl div{padding:10px;background:var(--panel)}dt{font-size:.6rem}dd{margin:6px 0 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.panel{overflow:hidden;margin-top:16px;background:var(--panel)}.panel-head{display:flex;justify-content:space-between;gap:16px;padding:16px;border-bottom:1px solid var(--line)}.panel-head h2{margin:0}.table-wrap{overflow:auto;max-height:620px}table{width:100%;border-collapse:collapse;font-size:.78rem}th,td{padding:11px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{position:sticky;top:0;background:var(--panel);color:var(--muted);font-size:.62rem;font-weight:400;text-transform:uppercase}footer{display:flex;justify-content:space-between;gap:16px;margin-top:18px;padding-top:12px;border-top:1px solid var(--line);color:var(--muted);font-size:.7rem}@media(max-width:760px){main{width:min(100% - 20px,1180px);padding-top:18px}.state{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}h1{margin-top:38px}.grid dl{grid-template-columns:1fr 1fr}footer{display:block}footer span{display:block;margin-top:6px}}@media(max-width:430px){.state{grid-template-columns:1fr}.grid dl{grid-template-columns:1fr}h1{font-size:3.35rem}}
"""

JS = r"""(()=>{const esc=v=>String(v??"—").replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const text=v=>v===null||v===undefined?'—':esc(v);const q=id=>document.getElementById(id);const put=(id,v)=>{const e=q(id);if(e)e.textContent=v??'—'};const run=async()=>{try{const s=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json());put('state',s.state||'not recorded');put('wait',s.wait_reason||'—');put('data',s.data_state||'unavailable');put('fingerprint',(s.fingerprint||'—')+(s.ood===true?' · OOD':'')+(s.ood_reason?' · '+s.ood_reason:''));put('revision',s.memory_revision||'—');const latest=s.latest||{};const d=s.dataset||{};const p=s.profiles||{};const row=document.querySelectorAll('#runtime dd');[latest.ttft_ms===undefined?'—':latest.ttft_ms+' ms',latest.decode_tps===undefined?'—':latest.decode_tps,latest.ci?latest.ci.low+'…'+latest.ci.high+' / '+(latest.mde??'—'):(latest.mde??'—'),latest.correctness,((latest.peak_memory_mb??'—')+' / '+(latest.peak_rss_mb??'—')),((latest.swap_before_mb??'—')+' / '+(latest.swap_after_mb??'—')),((latest.lease??'—')+' / '+(latest.pid??'—')+' / '+(latest.fork??'—'))].forEach((v,i)=>{if(row[i])row[i].textContent=v??'—'}));const b=document.querySelectorAll('#bindings dd');[d.dataset_hash||'—',latest.candidate_hash||latest.candidate||'—',latest.code_hash||'—',p.active||p.mode||'—',p.rollback_latched?'latched':'—'].forEach((v,i)=>{if(b[i])b[i].textContent=v});const h=await fetch('/api/history?limit=100',{cache:'no-store'}).then(r=>r.json());put('count',(h.returned??0)+' / '+(h.total??0));const body=q('history');body.innerHTML='';(h.history||[]).forEach(x=>{const tr=document.createElement('tr');[x.created_at,x.kind,x.status,x.ttft_ms===undefined?'—':x.ttft_ms,x.decode_tps===undefined?'—':x.decode_tps,x.correctness,x.peak_memory_mb===undefined?'—':x.peak_memory_mb].forEach(v=>{const td=document.createElement('td');td.textContent=v??'—';tr.append(td)});body.append(tr)});if(!body.children.length)body.innerHTML='<tr><td colspan="7">Keine Messdaten verfügbar.</td></tr>';const dec=await fetch('/api/decisions?limit=100',{cache:'no-store'}).then(r=>r.json());const est=dec.estimates&&dec.estimates.snips;put('dcount',(dec.observed??0)+' beobachtet / '+(dec.total??0)+' · '+(est?est.status:'keine Schätzung'));const dbody=q('decisions');dbody.innerHTML='';(dec.decisions||[]).forEach(x=>{const tr=document.createElement('tr');[x.policy_id,x.rule,x.chosen,x.propensity,x.actions,x.censoring,x.reward].forEach(v=>{const td=document.createElement('td');td.textContent=v??'—';tr.append(td)});dbody.append(tr)});if(!dbody.children.length)dbody.innerHTML='<tr><td colspan="7">Noch keine Entscheidung protokolliert.</td></tr>'}catch(e){put('data','unavailable');put('state','waiting')}};run()})()"""


def _target(value: str) -> tuple[str, dict[str, list[str]]]:
    try:
        if len(value.encode("ascii", errors="strict")) > MAX_TARGET_BYTES:
            raise DashboardError("request target is too large")
    except UnicodeEncodeError as exc:
        raise DashboardError("request target is not ASCII") from exc
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or parsed.fragment:
        raise DashboardError("absolute targets are forbidden")
    try:
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=2)
    except ValueError as exc:
        raise DashboardError("malformed query") from exc
    return parsed.path, query


def _payload_bytes(value: Any) -> bytes:
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise DashboardUnavailable("response is not serializable") from exc
    if len(payload) > MAX_RESPONSE_BYTES:
        raise DashboardError("response exceeds its byte limit")
    return payload


class DashboardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FridayOptimizer/1"
    sys_version = ""

    @property
    def service(self) -> DashboardService:
        service = getattr(self.server, "dashboard_service", None)
        if not isinstance(service, DashboardService):
            raise RuntimeError("dashboard service missing")
        return service

    def setup(self) -> None:
        super().setup()
        self.request.settimeout(READ_TIMEOUT_SECONDS)

    def _send(self, status: int, content_type: str, payload: bytes, *, head: bool, allow: str | None = None) -> None:
        if len(payload) > MAX_RESPONSE_BYTES:
            payload = _payload_bytes(_error_payload())
            status = int(HTTPStatus.SERVICE_UNAVAILABLE)
        self.send_response(status)
        for name, value in _HEADERS:
            self.send_header(name, value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        if allow is not None:
            self.send_header("Allow", allow)
        self.end_headers()
        if not head:
            self.wfile.write(payload)

    def _json(self, status: int, value: Mapping[str, Any], *, head: bool, allow: str | None = None) -> None:
        try:
            payload = _payload_bytes(value)
        except (DashboardError, DashboardUnavailable):
            status, payload = int(HTTPStatus.SERVICE_UNAVAILABLE), _payload_bytes(_error_payload())
        self._send(status, "application/json; charset=utf-8", payload, head=head, allow=allow)

    def _host_allowed(self) -> bool:
        try:
            peer = ipaddress.ip_address(str(self.client_address[0]))
            if not peer.is_loopback:
                return False
        except ValueError:
            return False
        host = self.headers.get("Host", "")
        if not host or len(host) > 256 or any(char in host for char in "/\\@"):
            return False
        if host.startswith("["):
            end = host.find("]")
            name, port = host[1:end] if end >= 0 else "", host[end + 1:] if end >= 0 else "x"
            if name != "::1" or (port and (not port.startswith(":") or not port[1:].isdigit())):
                return False
            return True
        name, separator, port = host.partition(":")
        if name not in {"127.0.0.1", "localhost"}:
            return False
        return not separator or (port.isdigit() and 0 < int(port) <= 65535)

    def _method_not_allowed(self) -> None:
        self._json(int(HTTPStatus.METHOD_NOT_ALLOWED), {"error": "method_not_allowed", "allow": ["GET", "HEAD"]}, head=False, allow="GET, HEAD")

    def _dispatch(self, *, head: bool) -> None:
        if not self._host_allowed():
            self._json(421, {"error": "misdirected_request"}, head=head)
            return
        if self.command not in {"GET", "HEAD"}:
            self._method_not_allowed()
            return
        try:
            path, query = _target(self.path)
            if self.headers.get("Content-Length", "0") not in {"", "0"}:
                raise DashboardError("request body is not allowed")
            if path == "/" and not query:
                self._send(200, "text/html; charset=utf-8", HTML.encode(), head=head)
                return
            if path == "/assets/app.css" and not query:
                payload = CSS.encode()
                if len(payload) > MAX_ASSET_BYTES:
                    raise DashboardError("asset exceeds its byte limit")
                self._send(200, "text/css; charset=utf-8", payload, head=head)
                return
            if path == "/assets/app.js" and not query:
                payload = JS.encode()
                if len(payload) > MAX_ASSET_BYTES:
                    raise DashboardError("asset exceeds its byte limit")
                self._send(200, "application/javascript; charset=utf-8", payload, head=head)
                return
            if path == "/api/status" and not query:
                value = self.service.status()
                self._json(503 if value.get("data_state") == "unavailable" else 200, value, head=head)
                return
            if path == "/api/history":
                if set(query) - {"limit"} or len(query.get("limit", [])) > 1:
                    raise DashboardError("history query is invalid")
                raw = query.get("limit", [str(DEFAULT_LIMIT)])[0]
                if not raw.isdigit() or len(raw) > 4:
                    raise DashboardError("history limit is invalid")
                value = self.service.history(int(raw))
                self._json(200, value, head=head)
                return
            if path == "/api/decisions":
                if set(query) - {"limit"} or len(query.get("limit", [])) > 1:
                    raise DashboardError("decisions query is invalid")
                raw = query.get("limit", [str(DEFAULT_LIMIT)])[0]
                if not raw.isdigit() or len(raw) > 4:
                    raise DashboardError("decisions limit is invalid")
                value = self.service.decisions(int(raw))
                self._json(503 if value.get("data_state") == "unavailable" else 200, value, head=head)
                return
            endpoint = {"/api/dataset": self.service.dataset, "/api/profiles": self.service.profiles, "/api/shadow": self.service.shadow, "/api/portfolio": self.service.portfolio}.get(path)
            if endpoint is not None and not query:
                value = endpoint()
                self._json(503 if value.get("data_state") == "unavailable" else 200, value, head=head)
                return
            self._json(404, {"error": "not_found"}, head=head)
        except DashboardError:
            self._json(400, {"error": "invalid_request"}, head=head)
        except DashboardUnavailable as exc:
            reason = "timeout" if "timeout" in str(exc).lower() else "unavailable"
            self._json(503, _error_payload(reason), head=head)
        except (OSError, sqlite3.Error, ValueError, TypeError, UnicodeError):
            self._json(503, _error_payload(), head=head)
        except Exception:
            # No exception text is sent to a client.  The local dashboard is a
            # diagnostics surface, not a traceback oracle.
            self._json(503, _error_payload(), head=head)

    def do_GET(self) -> None:
        self._dispatch(head=False)

    def do_HEAD(self) -> None:
        self._dispatch(head=True)

    do_POST = _method_not_allowed
    do_PUT = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_OPTIONS = _method_not_allowed

    def log_message(self, *_args: object) -> None:
        return None


# Compatibility aliases for callers that used the naming in the earlier
# runtime dashboards.  They point at the same read-only implementation.
DashboardRequestHandler = DashboardHandler
DatasetSnapshotProvider = DatasetProvider
ShadowDecisionProvider = ShadowProvider


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, database_path: str | os.PathLike[str], port: int = 0, **kwargs: Any) -> None:
        if type(port) is not int or not 0 <= port <= 65535:
            raise DashboardError("port is outside the registered range")
        self.dashboard_service = DashboardService(database_path, **kwargs)
        super().__init__((LOOPBACK_HOST, port), DashboardHandler)


def serve(database_path: str | os.PathLike[str], port: int = 0, **kwargs: Any) -> DashboardServer:
    """Create a loopback server; callers own its lifecycle."""

    return DashboardServer(database_path, port, **kwargs)


__all__ = [
    "CSP", "CSS", "DashboardError", "DashboardHandler", "DashboardRequestHandler", "DashboardServer",
    "DashboardService", "DatasetProvider", "PortfolioProvider", "HTML", "HOST", "JS",
    "LOOPBACK_HOST", "MAX_ASSET_BYTES", "MAX_HISTORY_ROWS", "MAX_RESPONSE_BYTES",
    "MAX_TARGET_BYTES", "ShadowDecisionProvider", "ShadowProvider", "DatasetSnapshotProvider", "serve",
]
