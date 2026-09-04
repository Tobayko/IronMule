"""Append-only, hash-chained runtime history — one implementation, many runtimes.

Structurally this is ``friday_head_skip_runtime/history.py`` with the parts that
differ between the three sealed copies lifted into ``HistorySpec``: the runtime
identifier, the SQLite application id, the admissible record kinds, and the
report contract. Everything the copies share — the defensive SQLite
configuration, the schema replay, the record-identity chain, the byte-exact
canonical round trip — is written once.

Every read verifies the whole chain. A record whose stored JSON is not the
canonical encoding of itself, or whose ``previous_record_id`` does not follow
the row before it, ends the read; there is no partial trust.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import quote

from friday_n10_v2.canonical import canonical_json_bytes, canonical_sha256, strict_json_loads

from .files import UnsafeFile, regular_bytes

_MIGRATION = Path(__file__).with_name("migrations") / "0001_runtime.sql"
APPLICATION_ID = 0x46524354  # ASCII "FRCT"
_SHA256_CHARS = frozenset("0123456789abcdef")
_RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_KIND_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

MAX_CANONICAL_BYTES = 4 * 1024 * 1024
MAX_HISTORY_ROWS = 256


class HistoryError(RuntimeError):
    """Runtime evidence cannot be stored or replayed without weakening history."""


class HistoryConflict(HistoryError):
    """A runtime entity key is already bound to different bytes."""


@dataclass(frozen=True)
class PersistenceOutcome:
    record_id: str
    entity_key: str
    state: str


@dataclass(frozen=True)
class HistorySpec:
    """Everything that distinguishes one runtime's history from another's.

    The schema itself is shared and fixed. Which record kinds a runtime admits
    is a property of the runtime, so it is enforced in Python — on write *and*
    on every read — rather than baked into a CHECK constraint that would force a
    separate migration, and a separate migration hash, per runtime.
    """

    runtime_id: str
    kinds: frozenset[str]
    schema_version: int = 1
    max_rows: int = MAX_HISTORY_ROWS
    max_canonical_bytes: int = MAX_CANONICAL_BYTES
    #: Extra, package-specific checks on an otherwise well-formed report.
    report_validator: Callable[[Mapping[str, Any]], None] | None = field(
        default=None, compare=False
    )

    def __post_init__(self) -> None:
        if not _RUNTIME_ID_RE.fullmatch(self.runtime_id or ""):
            raise HistoryError("runtime_id is not a bounded safe identifier")
        if not self.kinds or any(not _KIND_RE.fullmatch(kind) for kind in self.kinds):
            raise HistoryError("record kinds must be non-empty lower-case identifiers")
        if self.schema_version != 1:
            raise HistoryError("only schema version 1 is registered")
        if not 1 <= self.max_rows <= 4096:
            raise HistoryError("max_rows is outside the registered range")

    @property
    def application_id(self) -> int:
        return APPLICATION_ID

    def migration_sql(self) -> str:
        try:
            return regular_bytes(_MIGRATION).decode("utf-8", errors="strict")
        except (UnsafeFile, UnicodeError) as exc:
            raise HistoryError("runtime migration is unavailable") from exc

    def migration_bytes(self) -> bytes:
        return self.migration_sql().encode("utf-8")

    def migration_sha256(self) -> str:
        return hashlib.sha256(self.migration_bytes()).hexdigest()


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
    if (
        not hasattr(connection, "setconfig")
        or not hasattr(connection, "getconfig")
        or any(not hasattr(sqlite3, name) for name in required)
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


class RuntimeHistory:
    """A verified handle on one runtime's append-only history."""

    def __init__(
        self,
        spec: HistorySpec,
        path: Path,
        connection: sqlite3.Connection,
        *,
        read_only: bool,
    ) -> None:
        self.spec = spec
        self.path = path
        self.connection = connection
        self.read_only = read_only

    # -- schema ---------------------------------------------------------------
    def _expected_snapshot(self) -> list[tuple[str, str, str, str]]:
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(self.spec.migration_sql())
            connection.execute(
                "INSERT INTO metadata(singleton,schema_version,runtime_id,migration_sha256) "
                "VALUES(1,?,?,?)",
                (self.spec.schema_version, self.spec.runtime_id, self.spec.migration_sha256()),
            )
            return _master_snapshot(connection)
        finally:
            connection.close()

    def verify_schema(self) -> None:
        connection = self.connection
        if (
            connection.execute("PRAGMA application_id").fetchone()[0] != self.spec.application_id
            or connection.execute("PRAGMA user_version").fetchone()[0] != self.spec.schema_version
        ):
            raise HistoryError("database identity or schema version is not this runtime")
        if _master_snapshot(connection) != self._expected_snapshot():
            raise HistoryError("database schema differs from the registered migration")
        integrity = connection.execute("PRAGMA integrity_check(1)").fetchall()
        if len(integrity) != 1 or tuple(integrity[0]) != ("ok",):
            raise HistoryError("database integrity check failed")
        rows = connection.execute(
            "SELECT singleton,schema_version,runtime_id,migration_sha256 FROM metadata"
        ).fetchall()
        expected = (
            1,
            self.spec.schema_version,
            self.spec.runtime_id,
            self.spec.migration_sha256(),
        )
        if len(rows) != 1 or tuple(rows[0]) != expected:
            raise HistoryError("database metadata does not replay")

    @classmethod
    def open(
        cls,
        spec: HistorySpec,
        source: os.PathLike[str] | str,
        *,
        read_only: bool = False,
        initialize: bool = False,
    ) -> "RuntimeHistory":
        if read_only and initialize:
            raise HistoryError("read-only initialization is forbidden")
        path = _checked_path(source, create_parent=initialize)
        existed = path.exists()
        if initialize and not existed:
            _secure_create(path)
        _verify_file(path)
        connection: sqlite3.Connection | None = None
        try:
            if read_only:
                uri = f"file:{quote(str(path))}?mode=ro"
                connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5.0)
            else:
                connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
            _configure(connection, read_only=read_only)
            history = cls(spec, path, connection, read_only=read_only)
            if initialize and not existed:
                connection.executescript(spec.migration_sql())
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO metadata(singleton,schema_version,runtime_id,migration_sha256) "
                    "VALUES(1,?,?,?)",
                    (spec.schema_version, spec.runtime_id, spec.migration_sha256()),
                )
                connection.execute("COMMIT")
            history.verify_schema()
        except (sqlite3.Error, UnicodeError, HistoryError) as exc:
            if connection is not None:
                connection.close()
            if isinstance(exc, HistoryError):
                raise
            raise HistoryError("database initialization or verification failed") from exc
        return history

    def __enter__(self) -> "RuntimeHistory":
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
            self.verify_schema()
            yield
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    # -- validation -----------------------------------------------------------
    def _canonical_object(self, value: Mapping[str, Any], name: str):
        if not isinstance(value, Mapping):
            raise HistoryError(f"{name} must be an object")
        payload = canonical_json_bytes(value)
        if len(payload) > self.spec.max_canonical_bytes:
            raise HistoryError(f"{name} exceeds its byte limit")
        checked = strict_json_loads(payload)
        if not isinstance(checked, dict):
            raise HistoryError(f"{name} must be an object")
        return checked, payload, hashlib.sha256(payload).hexdigest()

    def _validated_provenance(self, value: Mapping[str, Any]):
        checked, payload, _ = self._canonical_object(value, "provenance")
        digest = checked.get("provenance_sha256")
        body = dict(checked)
        body.pop("provenance_sha256", None)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _SHA256_CHARS for character in digest)
            or canonical_sha256(body) != digest
            or checked.get("runtime_id") != self.spec.runtime_id
            or checked.get("schema_version") != self.spec.schema_version
        ):
            raise HistoryError("runtime provenance does not replay")
        for digest_field, value_field in (
            ("code_sha256", "code_files"),
            ("spec_sha256", "spec_files"),
            ("environment_sha256", "environment"),
            ("hardware_sha256", "hardware"),
        ):
            if not isinstance(checked.get(value_field), Mapping) or canonical_sha256(
                checked[value_field]
            ) != checked.get(digest_field):
                raise HistoryError(f"runtime provenance projection differs: {digest_field}")
        return checked, payload, digest

    def _validated_report(self, value: Mapping[str, Any]):
        checked, payload, digest = self._canonical_object(value, "report")
        kind = checked.get("kind")
        run_id = checked.get("run_id")
        status = checked.get("status")
        if (
            kind not in self.spec.kinds
            or checked.get("runtime_id") != self.spec.runtime_id
            or checked.get("schema_version") != self.spec.schema_version
            or not isinstance(run_id, str)
            or not 1 <= len(run_id) <= 96
            or not isinstance(status, str)
            or not 1 <= len(status) <= 96
        ):
            raise HistoryError("runtime report is outside the closed schema")
        if self.spec.report_validator is not None:
            self.spec.report_validator(checked)
        entity_key = f"{kind}:{run_id}"
        if len(entity_key) > 160:
            raise HistoryError("runtime entity key exceeds its bound")
        return checked, payload, digest, entity_key

    def _body(
        self,
        *,
        previous: str | None,
        entity_key: str,
        kind: str,
        status: str,
        created_at_unix_ns: int,
        report_digest: str,
        provenance_digest: str,
    ) -> dict[str, Any]:
        return {
            "previous_record_id": previous,
            "runtime_id": self.spec.runtime_id,
            "entity_key": entity_key,
            "record_kind": kind,
            "status": status,
            "created_at_unix_ns": created_at_unix_ns,
            "report_sha256": report_digest,
            "provenance_sha256": provenance_digest,
        }

    # -- reads and writes -----------------------------------------------------
    def verified_records(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT rowid,* FROM records ORDER BY rowid").fetchall()
        if len(rows) > self.spec.max_rows:
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
            checked_report, replay_report, report_digest, entity_key = self._validated_report(
                report
            )
            checked_provenance, replay_provenance, provenance_digest = (
                self._validated_provenance(provenance)
            )
            if replay_report != report_bytes or replay_provenance != provenance_bytes:
                raise HistoryError("stored runtime JSON is not canonical")
            if (
                row["previous_record_id"] != previous
                or row["runtime_id"] != self.spec.runtime_id
                or row["entity_key"] != entity_key
                or row["record_kind"] != checked_report["kind"]
                or row["status"] != checked_report["status"]
                or row["report_sha256"] != report_digest
                or row["provenance_sha256"] != provenance_digest
            ):
                raise HistoryError("stored runtime projections or chain do not replay")
            body = self._body(
                previous=previous,
                entity_key=entity_key,
                kind=checked_report["kind"],
                status=checked_report["status"],
                created_at_unix_ns=row["created_at_unix_ns"],
                report_digest=report_digest,
                provenance_digest=provenance_digest,
            )
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
        checked_report, report_bytes, report_digest, entity_key = self._validated_report(report)
        _, provenance_bytes, provenance_digest = self._validated_provenance(provenance)
        now = time.time_ns() if created_at_unix_ns is None else created_at_unix_ns
        if isinstance(now, bool) or not isinstance(now, int) or not 0 <= now < 1 << 63:
            raise HistoryError("created_at_unix_ns is invalid")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            _verify_file(self.path)
            self.verify_schema()
            current = self.verified_records()
            if len(current) >= self.spec.max_rows:
                raise HistoryError("runtime history reached its registered row bound")
            if any(row["entity_key"] == entity_key for row in current):
                raise HistoryConflict(f"runtime entity already exists: {entity_key}")
            previous = current[-1]["record_id"] if current else None
            body = self._body(
                previous=previous,
                entity_key=entity_key,
                kind=checked_report["kind"],
                status=checked_report["status"],
                created_at_unix_ns=now,
                report_digest=report_digest,
                provenance_digest=provenance_digest,
            )
            record_id = canonical_sha256(body)
            self.connection.execute(
                "INSERT INTO records(record_id,previous_record_id,runtime_id,entity_key,"
                "record_kind,status,created_at_unix_ns,report_json,report_sha256,"
                "provenance_json,provenance_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record_id,
                    previous,
                    self.spec.runtime_id,
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
    "HistoryConflict",
    "HistoryError",
    "HistorySpec",
    "MAX_CANONICAL_BYTES",
    "MAX_HISTORY_ROWS",
    "PersistenceOutcome",
    "RuntimeHistory",
]
