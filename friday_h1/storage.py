"""Append-only SQLite storage with full H1-v2 history replay."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import quote

from .canonical import canonical_json_bytes, canonical_sha256, strict_json_loads
from .constants import (
    MAX_HISTORY_ROWS,
    RECORD_KINDS,
    SCHEMA_VERSION,
    SQLITE_APPLICATION_ID,
    STUDY_ID,
)
from .protocol import ProtocolError, validate_history

_MIGRATION = Path(__file__).with_name("migrations") / "0001_initial.sql"
_SHA256_CHARS = frozenset("0123456789abcdef")


class StorageError(RuntimeError):
    """Formal evidence cannot be stored or verified without weakening the contract."""


class StorageConflict(StorageError):
    """A sealed entity already exists and can never be retried or replaced."""


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
        raise StorageError("H1-v2 migration is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or _MIGRATION.is_symlink() or len(payload) != info.st_size:
        raise StorageError("H1-v2 migration is not a stable regular file")
    return payload


def _migration_sha256() -> str:
    return hashlib.sha256(_migration_bytes()).hexdigest()


def _checked_path(source: os.PathLike[str] | str, *, create_parent: bool) -> Path:
    path = Path(os.path.abspath(Path(source).expanduser()))
    if path.exists() and path.is_symlink():
        raise StorageError("database path must not be a symlink")
    if create_parent:
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError("database parent cannot be created") from exc
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise StorageError("database parent is unavailable") from exc
    if not parent.is_dir():
        raise StorageError("database parent is not a directory")
    return parent / path.name


def _secure_create(path: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        return
    except OSError as exc:
        raise StorageError("database file cannot be securely created") from exc
    else:
        os.close(descriptor)


def _verify_file(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise StorageError("database file is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise StorageError("database must be a regular non-symlink file")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise StorageError("database permissions must remain private")


def _configure(connection: sqlite3.Connection, *, read_only: bool) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA trusted_schema = OFF")
    connection.execute("PRAGMA busy_timeout = 5000")
    if read_only:
        connection.execute("PRAGMA query_only = ON")
        if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise StorageError("read-only database did not enter query_only mode")


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
            "INSERT INTO metadata(singleton,schema_version,study_id,migration_sha256) "
            "VALUES(1,?,?,?)",
            (SCHEMA_VERSION, STUDY_ID, _migration_sha256()),
        )
        return _master_snapshot(connection)
    finally:
        connection.close()


def _verify_schema(connection: sqlite3.Connection) -> None:
    application_id = connection.execute("PRAGMA application_id").fetchone()[0]
    user_version = connection.execute("PRAGMA user_version").fetchone()[0]
    if application_id != SQLITE_APPLICATION_ID or user_version != SCHEMA_VERSION:
        raise StorageError("database identity or schema version is not H1-v2")
    if _master_snapshot(connection) != _expected_snapshot():
        raise StorageError("database schema differs from the registered migration")
    row = connection.execute(
        "SELECT singleton,schema_version,study_id,migration_sha256 FROM metadata"
    ).fetchall()
    if len(row) != 1 or tuple(row[0]) != (1, SCHEMA_VERSION, STUDY_ID, _migration_sha256()):
        raise StorageError("database metadata does not replay")


def _entity_key(payload: Mapping[str, Any]) -> tuple[str, str, str]:
    kind = payload.get("kind")
    if kind not in RECORD_KINDS:
        raise StorageError("unknown formal record kind")
    stage = payload.get("stage", "preregistration")
    session_id = payload.get("session_id", "")
    if kind == "preregistration":
        return "preregistration", "preregistration", ""
    if kind in {"calibration_session", "confirmation_session", "session_failure"}:
        return f"session:{stage}:{session_id}", str(stage), str(session_id)
    return str(kind).replace("_", "-"), str(stage), ""


def _validated_provenance(value: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise StorageError("provenance must be an object")
    checked = strict_json_loads(canonical_json_bytes(value))
    if not isinstance(checked, dict):
        raise StorageError("provenance must be an object")
    digest = checked.get("provenance_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in _SHA256_CHARS for character in digest)
    ):
        raise StorageError("provenance digest is invalid")
    body = dict(checked)
    body.pop("provenance_sha256")
    if canonical_sha256(body) != digest:
        raise StorageError("provenance digest does not replay")
    if checked.get("study_id") != STUDY_ID or checked.get("schema_version") != SCHEMA_VERSION:
        raise StorageError("provenance belongs to another study")
    return checked, digest


class Storage:
    def __init__(
        self,
        path: Path,
        connection: sqlite3.Connection,
        *,
        read_only: bool,
    ) -> None:
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
    ) -> "Storage":
        if read_only and initialize:
            raise StorageError("read-only initialization is forbidden")
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
        except sqlite3.Error as exc:
            raise StorageError("database cannot be opened") from exc
        try:
            _configure(connection, read_only=read_only)
            if initialize and not existed:
                connection.executescript(_migration_bytes().decode("utf-8", errors="strict"))
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO metadata(singleton,schema_version,study_id,migration_sha256) "
                    "VALUES(1,?,?,?)",
                    (SCHEMA_VERSION, STUDY_ID, _migration_sha256()),
                )
                connection.execute("COMMIT")
            _verify_schema(connection)
        except (sqlite3.Error, UnicodeError, StorageError) as exc:
            connection.close()
            if isinstance(exc, StorageError):
                raise
            raise StorageError("database initialization or verification failed") from exc
        return cls(path, connection, read_only=read_only)

    def __enter__(self) -> "Storage":
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

    def _decoded_rows(self, *, replay_history: bool = True) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT rowid,* FROM records ORDER BY rowid"
        ).fetchall()
        if len(rows) > MAX_HISTORY_ROWS:
            raise StorageError("formal history exceeds its registered row bound")
        result: list[dict[str, Any]] = []
        for row in rows:
            payload_bytes = row["payload_json"].encode("utf-8", errors="strict")
            provenance_bytes = row["provenance_json"].encode("utf-8", errors="strict")
            payload = strict_json_loads(payload_bytes)
            provenance = strict_json_loads(provenance_bytes)
            if not isinstance(payload, dict) or not isinstance(provenance, dict):
                raise StorageError("stored record JSON is not an object")
            if hashlib.sha256(payload_bytes).hexdigest() != row["payload_sha256"]:
                raise StorageError("stored payload digest does not replay")
            checked_provenance, provenance_digest = _validated_provenance(provenance)
            if provenance_digest != row["provenance_sha256"]:
                raise StorageError("stored provenance projection does not replay")
            entity_key, stage, session_id = _entity_key(payload)
            if (
                row["study_id"] != STUDY_ID
                or row["entity_key"] != entity_key
                or row["record_kind"] != payload["kind"]
                or row["stage"] != stage
                or row["session_id"] != session_id
                or row["status"] != payload["status"]
                or bool(row["formal_claim"]) is not payload["formal_claim"]
                or payload.get("provenance_sha256") != provenance_digest
            ):
                raise StorageError("stored record projections do not replay")
            body = {
                "study_id": STUDY_ID,
                "entity_key": entity_key,
                "record_kind": payload["kind"],
                "stage": stage,
                "session_id": session_id,
                "status": payload["status"],
                "formal_claim": payload["formal_claim"],
                "created_at_unix_ns": row["created_at_unix_ns"],
                "payload_sha256": row["payload_sha256"],
                "provenance_sha256": provenance_digest,
            }
            if canonical_sha256(body) != row["record_id"]:
                raise StorageError("stored record identity does not replay")
            result.append(
                {
                    "rowid": row["rowid"],
                    "record_id": row["record_id"],
                    **body,
                    "payload": payload,
                    "provenance": checked_provenance,
                }
            )
        if replay_history:
            try:
                validate_history([row["payload"] for row in result])
            except ProtocolError as exc:
                raise StorageError("stored formal history does not replay") from exc
        return result

    def verified_records(self) -> list[dict[str, Any]]:
        return self._decoded_rows()

    def persist(
        self,
        payload: Mapping[str, Any],
        provenance: Mapping[str, Any],
        *,
        created_at_unix_ns: int | None = None,
    ) -> PersistenceOutcome:
        if self.read_only:
            raise StorageError("read-only storage cannot persist")
        checked_provenance, provenance_digest = _validated_provenance(provenance)
        payload_value = strict_json_loads(canonical_json_bytes(payload))
        if not isinstance(payload_value, dict):
            raise StorageError("formal payload must be an object")
        if payload_value.get("provenance_sha256") != provenance_digest:
            raise StorageError("payload and provenance are not bound")
        entity_key, stage, session_id = _entity_key(payload_value)
        now = time.time_ns() if created_at_unix_ns is None else created_at_unix_ns
        if isinstance(now, bool) or not isinstance(now, int) or not 0 <= now < 1 << 63:
            raise StorageError("created_at_unix_ns is invalid")
        payload_bytes = canonical_json_bytes(payload_value)
        provenance_bytes = canonical_json_bytes(checked_provenance)
        payload_digest = hashlib.sha256(payload_bytes).hexdigest()
        body = {
            "study_id": STUDY_ID,
            "entity_key": entity_key,
            "record_kind": payload_value["kind"],
            "stage": stage,
            "session_id": session_id,
            "status": payload_value["status"],
            "formal_claim": payload_value["formal_claim"],
            "created_at_unix_ns": now,
            "payload_sha256": payload_digest,
            "provenance_sha256": provenance_digest,
        }
        record_id = canonical_sha256(body)
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            _verify_file(self.path)
            _verify_schema(self.connection)
            # Each stored row and its database projections are verified first;
            # the single combined replay below then validates the entire old
            # history plus the candidate. Replaying the same bootstrap before,
            # during, and after one atomic INSERT adds no evidence.
            current = self._decoded_rows(replay_history=False)
            if any(row["entity_key"] == entity_key for row in current):
                raise StorageConflict(f"sealed entity already exists: {entity_key}")
            validate_history([row["payload"] for row in current] + [payload_value])
            self.connection.execute(
                "INSERT INTO records("
                "record_id,study_id,entity_key,record_kind,stage,session_id,status,"
                "formal_claim,created_at_unix_ns,payload_json,payload_sha256,"
                "provenance_json,provenance_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record_id,
                    STUDY_ID,
                    entity_key,
                    payload_value["kind"],
                    stage,
                    session_id,
                    payload_value["status"],
                    int(payload_value["formal_claim"]),
                    now,
                    payload_bytes.decode("utf-8"),
                    payload_digest,
                    provenance_bytes.decode("utf-8"),
                    provenance_digest,
                ),
            )
            self.connection.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            self.connection.execute("ROLLBACK")
            raise StorageConflict(f"sealed entity already exists: {entity_key}") from exc
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise
        return PersistenceOutcome(record_id=record_id, entity_key=entity_key, state="persisted")

    def by_kind(self, kind: str) -> list[dict[str, Any]]:
        return [row for row in self._decoded_rows() if row["record_kind"] == kind]

    def session(self, stage: str, session_id: str) -> dict[str, Any] | None:
        key = f"session:{stage}:{session_id}"
        return next((row for row in self._decoded_rows() if row["entity_key"] == key), None)


__all__ = [
    "PersistenceOutcome",
    "Storage",
    "StorageConflict",
    "StorageError",
]
