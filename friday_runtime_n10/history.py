"""Private append-only SQLite history with full hash-chain replay."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import quote

from friday_n10_v2.canonical import canonical_json_bytes, canonical_sha256, strict_json_loads

from .constants import (
    HISTORY_KINDS,
    MAX_CANONICAL_BYTES,
    MAX_HISTORY_ROWS,
    N10_DECISION_RECORD_ID,
    RUNTIME_ID,
    SCHEMA_VERSION,
    SQLITE_APPLICATION_ID,
)

_MIGRATION = Path(__file__).with_name("migrations") / "0001_initial.sql"
_SHA256_CHARS = frozenset("0123456789abcdef")


class HistoryError(RuntimeError):
    """Runtime evidence cannot be stored or replayed without weakening history."""


class HistoryConflict(HistoryError):
    """A runtime run identifier is already bound to different bytes."""


@dataclass(frozen=True)
class PersistenceOutcome:
    record_id: str
    entity_key: str
    state: str


def _migration_bytes() -> bytes:
    try:
        info = _MIGRATION.lstat()
        payload = _MIGRATION.read_bytes()
    except OSError as exc:
        raise HistoryError("runtime migration is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or _MIGRATION.is_symlink() or len(payload) != info.st_size:
        raise HistoryError("runtime migration is not a stable regular file")
    return payload


def _migration_sha256() -> str:
    return hashlib.sha256(_migration_bytes()).hexdigest()


def _checked_path(source: os.PathLike[str] | str, *, create_parent: bool) -> Path:
    path = Path(os.path.abspath(Path(source).expanduser()))
    if create_parent:
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise HistoryError("database parent cannot be created") from exc
    try:
        parent_info = path.parent.lstat()
    except OSError as exc:
        raise HistoryError("database parent is unavailable") from exc
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise HistoryError("database parent must be a real directory")
    if parent_info.st_uid != os.geteuid() or parent_info.st_mode & 0o077:
        raise HistoryError("database parent ownership or permissions are unsafe")
    resolved = path.parent.resolve(strict=True) / path.name
    if resolved.exists():
        _verify_file(resolved)
    return resolved


def _secure_create(path: Path) -> None:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise HistoryError("database file cannot be securely created") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
            raise HistoryError("created database is not a private regular file")
    finally:
        os.close(descriptor)


def _verify_file(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise HistoryError("database file is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_uid != os.geteuid():
        raise HistoryError("database must be a regular non-symlink file")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise HistoryError("database ownership or permissions must remain private")


def _configure(connection: sqlite3.Connection, *, read_only: bool) -> None:
    connection.row_factory = sqlite3.Row
    required = (
        "SQLITE_DBCONFIG_DEFENSIVE",
        "SQLITE_DBCONFIG_TRUSTED_SCHEMA",
        "SQLITE_DBCONFIG_DQS_DDL",
        "SQLITE_DBCONFIG_DQS_DML",
        "SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION",
    )
    if not hasattr(connection, "setconfig") or not hasattr(connection, "getconfig") or any(
        not hasattr(sqlite3, name) for name in required
    ):
        raise HistoryError("required SQLite defensive controls are unavailable")
    try:
        connection.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
        connection.setconfig(sqlite3.SQLITE_DBCONFIG_TRUSTED_SCHEMA, False)
        connection.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DDL, False)
        connection.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
        connection.setconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, False)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("PRAGMA busy_timeout = 5000")
        if read_only:
            connection.execute("PRAGMA query_only = ON")
        expected = (
            (sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True),
            (sqlite3.SQLITE_DBCONFIG_TRUSTED_SCHEMA, False),
            (sqlite3.SQLITE_DBCONFIG_DQS_DDL, False),
            (sqlite3.SQLITE_DBCONFIG_DQS_DML, False),
            (sqlite3.SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, False),
        )
        if any(connection.getconfig(option) != value for option, value in expected):
            raise HistoryError("SQLite defensive controls did not take effect")
        if connection.execute("PRAGMA trusted_schema").fetchone()[0] != 0:
            raise HistoryError("SQLite trusted schema remained enabled")
        if read_only and connection.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise HistoryError("read-only database did not enter query_only mode")
    except (sqlite3.Error, ValueError) as exc:
        raise HistoryError("cannot enable required SQLite defensive controls") from exc


def _master_snapshot(connection: sqlite3.Connection) -> list[tuple[str, str, str, str]]:
    rows = connection.execute(
        "SELECT type,name,tbl_name,coalesce(sql,'') FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name,tbl_name"
    ).fetchall()
    return [tuple(str(value) for value in row) for row in rows]


def _expected_snapshot() -> list[tuple[str, str, str, str]]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(_migration_bytes().decode("utf-8", errors="strict"))
        connection.execute(
            "INSERT INTO metadata(singleton,schema_version,runtime_id,migration_sha256) "
            "VALUES(1,?,?,?)",
            (SCHEMA_VERSION, RUNTIME_ID, _migration_sha256()),
        )
        return _master_snapshot(connection)
    finally:
        connection.close()


def _verify_schema(connection: sqlite3.Connection) -> None:
    if (
        connection.execute("PRAGMA application_id").fetchone()[0] != SQLITE_APPLICATION_ID
        or connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION
    ):
        raise HistoryError("database identity or schema version is not Friday runtime")
    if _master_snapshot(connection) != _expected_snapshot():
        raise HistoryError("database schema differs from the registered migration")
    integrity = connection.execute("PRAGMA integrity_check(1)").fetchall()
    if len(integrity) != 1 or tuple(integrity[0]) != ("ok",):
        raise HistoryError("database integrity check failed")
    rows = connection.execute(
        "SELECT singleton,schema_version,runtime_id,migration_sha256 FROM metadata"
    ).fetchall()
    if len(rows) != 1 or tuple(rows[0]) != (1, SCHEMA_VERSION, RUNTIME_ID, _migration_sha256()):
        raise HistoryError("database metadata does not replay")


def _canonical_object(value: Mapping[str, Any], name: str) -> tuple[dict[str, Any], bytes, str]:
    if not isinstance(value, Mapping):
        raise HistoryError(f"{name} must be an object")
    payload = canonical_json_bytes(value)
    if len(payload) > MAX_CANONICAL_BYTES:
        raise HistoryError(f"{name} exceeds its byte limit")
    checked = strict_json_loads(payload)
    if not isinstance(checked, dict):
        raise HistoryError(f"{name} must be an object")
    return checked, payload, hashlib.sha256(payload).hexdigest()


def _validated_provenance(value: Mapping[str, Any]) -> tuple[dict[str, Any], bytes, str]:
    checked, payload, _ = _canonical_object(value, "provenance")
    digest = checked.get("provenance_sha256")
    body = dict(checked)
    body.pop("provenance_sha256", None)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in _SHA256_CHARS for character in digest)
        or canonical_sha256(body) != digest
        or checked.get("runtime_id") != RUNTIME_ID
        or checked.get("schema_version") != SCHEMA_VERSION
    ):
        raise HistoryError("runtime provenance does not replay")
    for field, value_field in (
        ("code_sha256", "code_files"),
        ("spec_sha256", "spec_files"),
        ("environment_sha256", "environment"),
        ("hardware_sha256", "hardware"),
    ):
        if not isinstance(checked.get(value_field), Mapping) or canonical_sha256(
            checked[value_field]
        ) != checked.get(field):
            raise HistoryError(f"runtime provenance projection differs: {field}")
    return checked, payload, digest


def _validated_report(value: Mapping[str, Any]) -> tuple[dict[str, Any], bytes, str, str]:
    checked, payload, digest = _canonical_object(value, "report")
    kind = checked.get("kind")
    run_id = checked.get("run_id")
    if (
        kind not in HISTORY_KINDS
        or checked.get("runtime_id") != RUNTIME_ID
        or checked.get("schema_version") != SCHEMA_VERSION
        or checked.get("formal_claim") is not False
        or checked.get("n10_decision_record_id") != N10_DECISION_RECORD_ID
        or not isinstance(checked.get("status"), str)
        or not 1 <= len(checked["status"]) <= 96
        or not isinstance(run_id, str)
        or not 1 <= len(run_id) <= 96
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
            for character in run_id
        )
    ):
        raise HistoryError("runtime report is outside the closed schema")
    return checked, payload, digest, f"{kind}:{run_id}"


class History:
    def __init__(self, path: Path, connection: sqlite3.Connection, *, read_only: bool) -> None:
        self.path = path
        self.connection = connection
        self.read_only = read_only

    @classmethod
    def open(
        cls,
        source: os.PathLike[str] | str,
        *,
        read_only: bool = False,
        initialize: bool = False,
    ) -> "History":
        if read_only and initialize:
            raise HistoryError("read-only initialization is forbidden")
        path = _checked_path(source, create_parent=initialize)
        existed = path.exists()
        if initialize and not existed:
            _secure_create(path)
        _verify_file(path)
        try:
            if read_only:
                uri = f"file:{quote(str(path))}?mode=ro"
                connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5.0)
            else:
                connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
            _configure(connection, read_only=read_only)
            if initialize and not existed:
                connection.executescript(_migration_bytes().decode("utf-8", errors="strict"))
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO metadata(singleton,schema_version,runtime_id,migration_sha256) "
                    "VALUES(1,?,?,?)",
                    (SCHEMA_VERSION, RUNTIME_ID, _migration_sha256()),
                )
                connection.execute("COMMIT")
            _verify_schema(connection)
        except (sqlite3.Error, UnicodeError, HistoryError) as exc:
            if "connection" in locals():
                connection.close()
            if isinstance(exc, HistoryError):
                raise
            raise HistoryError("database initialization or verification failed") from exc
        return cls(path, connection, read_only=read_only)

    def __enter__(self) -> "History":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def read_transaction(self) -> Iterator[None]:
        self.connection.execute("BEGIN")
        try:
            _verify_file(self.path)
            _verify_schema(self.connection)
            yield
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def verified_records(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT rowid,* FROM records ORDER BY rowid").fetchall()
        if len(rows) > MAX_HISTORY_ROWS:
            raise HistoryError("runtime history exceeds its registered row bound")
        result: list[dict[str, Any]] = []
        previous: str | None = None
        for row in rows:
            report_bytes = row["report_json"].encode("utf-8", errors="strict")
            provenance_bytes = row["provenance_json"].encode("utf-8", errors="strict")
            report = strict_json_loads(report_bytes)
            provenance = strict_json_loads(provenance_bytes)
            if not isinstance(report, dict) or not isinstance(provenance, dict):
                raise HistoryError("stored runtime JSON is not an object")
            checked_report, replay_report, report_digest, entity_key = _validated_report(report)
            checked_provenance, replay_provenance, provenance_digest = (
                _validated_provenance(provenance)
            )
            if replay_report != report_bytes or replay_provenance != provenance_bytes:
                raise HistoryError("stored runtime JSON is not canonical")
            if (
                row["previous_record_id"] != previous
                or row["runtime_id"] != RUNTIME_ID
                or row["entity_key"] != entity_key
                or row["record_kind"] != checked_report["kind"]
                or row["status"] != checked_report["status"]
                or row["report_sha256"] != report_digest
                or row["provenance_sha256"] != provenance_digest
            ):
                raise HistoryError("stored runtime projections or chain do not replay")
            body = {
                "previous_record_id": previous,
                "runtime_id": RUNTIME_ID,
                "entity_key": entity_key,
                "record_kind": checked_report["kind"],
                "status": checked_report["status"],
                "created_at_unix_ns": row["created_at_unix_ns"],
                "report_sha256": report_digest,
                "provenance_sha256": provenance_digest,
            }
            if canonical_sha256(body) != row["record_id"]:
                raise HistoryError("stored runtime record identity does not replay")
            previous = row["record_id"]
            result.append(
                {
                    "rowid": row["rowid"],
                    "record_id": row["record_id"],
                    **body,
                    "report": checked_report,
                    "provenance": checked_provenance,
                }
            )
        return result

    def persist(
        self,
        report: Mapping[str, Any],
        provenance: Mapping[str, Any],
        *,
        created_at_unix_ns: int | None = None,
    ) -> PersistenceOutcome:
        if self.read_only:
            raise HistoryError("read-only runtime history cannot persist")
        checked_report, report_bytes, report_digest, entity_key = _validated_report(report)
        _, provenance_bytes, provenance_digest = _validated_provenance(provenance)
        now = time.time_ns() if created_at_unix_ns is None else created_at_unix_ns
        if isinstance(now, bool) or not isinstance(now, int) or not 0 <= now < 1 << 63:
            raise HistoryError("created_at_unix_ns is invalid")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            _verify_file(self.path)
            _verify_schema(self.connection)
            current = self.verified_records()
            if len(current) >= MAX_HISTORY_ROWS:
                raise HistoryError("runtime history reached its registered row bound")
            existing = next((row for row in current if row["entity_key"] == entity_key), None)
            if existing is not None:
                raise HistoryConflict(f"runtime entity already exists: {entity_key}")
            previous = current[-1]["record_id"] if current else None
            body = {
                "previous_record_id": previous,
                "runtime_id": RUNTIME_ID,
                "entity_key": entity_key,
                "record_kind": checked_report["kind"],
                "status": checked_report["status"],
                "created_at_unix_ns": now,
                "report_sha256": report_digest,
                "provenance_sha256": provenance_digest,
            }
            record_id = canonical_sha256(body)
            self.connection.execute(
                "INSERT INTO records(record_id,previous_record_id,runtime_id,entity_key,"
                "record_kind,status,created_at_unix_ns,report_json,report_sha256,"
                "provenance_json,provenance_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record_id,
                    previous,
                    RUNTIME_ID,
                    entity_key,
                    checked_report["kind"],
                    checked_report["status"],
                    now,
                    report_bytes.decode("utf-8"),
                    report_digest,
                    provenance_bytes.decode("utf-8"),
                    provenance_digest,
                ),
            )
            self.connection.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise HistoryConflict(f"runtime entity already exists: {entity_key}") from exc
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise
        return PersistenceOutcome(record_id, entity_key, "persisted")


__all__ = [
    "History",
    "HistoryConflict",
    "HistoryError",
    "PersistenceOutcome",
]
