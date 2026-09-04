"""Append-only SQLite Optimization Memory v2.

The store is intentionally boring: one immutable record table, a hash chain,
strict transaction boundaries, and read-only query connections.  It is suitable
for offline evidence import; it does not execute models, inspect hardware, or
rewrite any existing evidence source.  It forces SQLite's DELETE journal mode;
the main database plus ``-journal``, ``-wal``, and ``-shm`` sidecars count toward
the configured byte bound.  The writer owns one connection until ``close()``;
all public operations are serialized by an ``RLock`` and the connection is not
usable after close.
"""

from __future__ import annotations

import contextlib
from enum import Enum
import os
import sqlite3
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from typing import Any, Iterable, Iterator, Mapping

from .canonical import CanonicalJSONError, canonical_bytes, loads_strict, sha256_hex
from .records import DataPhase, OptimizationRecord, QualityClass, RecordKind

APPLICATION_ID = 0x46524D32  # ASCII-ish "FRM2", stable and unique to Friday memory.
USER_VERSION = 2
GENESIS_HASH = sha256_hex(b"project-friday:optimization-memory-v2:genesis")
MAX_RECORDS = 100_000
MAX_PAYLOAD_BYTES = 1_048_576
MAX_SOURCE_HASH_BYTES = 128
MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
MAX_DATABASE_BYTES = 64 * 1024 * 1024
_EXPECTED_COLUMNS = {
    "seq", "record_id", "kind", "quality", "phase", "payload", "payload_hash",
    "source_hash", "created_at", "prev_record_hash", "record_hash",
}


class OptimizationMemoryError(Exception):
    """Base class for memory-store errors."""


class UnsafeDatabaseError(OptimizationMemoryError):
    """Raised for symlinks, non-regular files, or unsafe database state."""


class MemoryConflictError(OptimizationMemoryError):
    """Raised when an existing ID is paired with different evidence."""


class MemoryLimitError(OptimizationMemoryError):
    """Raised when a configured record/payload/snapshot bound is exceeded."""


class IntegrityError(OptimizationMemoryError):
    """Raised when a chain or database integrity check fails."""


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    ok: bool
    sqlite_ok: bool
    chain_ok: bool
    rows: int
    error: str | None = None


# Compatibility for early private callers; it is deliberately omitted from
# ``__all__`` and from the package root so builtins.MemoryError is never
# shadowed at the public API boundary.
MemoryError = OptimizationMemoryError


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS optimization_records (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    quality TEXT NOT NULL,
    phase TEXT NOT NULL,
    payload BLOB NOT NULL,
    payload_hash TEXT NOT NULL,
    source_hash TEXT,
    created_at TEXT NOT NULL,
    prev_record_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL UNIQUE,
    CHECK (length(record_id) BETWEEN 1 AND 256),
    CHECK (kind IN ('environment','workload','candidate','compile','correctness',
                   'benchmark','profile','promotion','system','import','dataset')),
    CHECK (quality IN ('formal','engineering','exploratory','legacy_summary',
                       'invalid','quarantined')),
    CHECK (phase IN ('feature','label')),
    CHECK (length(payload_hash) = 64),
    CHECK (length(prev_record_hash) = 64),
    CHECK (length(record_hash) = 64),
    CHECK (source_hash IS NULL OR length(source_hash) = 64)
);
"""
_CREATE_INDEXES = (
    """CREATE INDEX IF NOT EXISTS optimization_records_kind_seq
       ON optimization_records(kind, seq)""",
    """CREATE INDEX IF NOT EXISTS optimization_records_quality_seq
       ON optimization_records(quality, seq)""",
)
_CREATE_TRIGGERS = (
    """CREATE TRIGGER IF NOT EXISTS optimization_records_no_update
       BEFORE UPDATE ON optimization_records
       BEGIN
           SELECT RAISE(ABORT, 'optimization memory is append-only');
       END""",
    """CREATE TRIGGER IF NOT EXISTS optimization_records_no_delete
       BEFORE DELETE ON optimization_records
       BEGIN
           SELECT RAISE(ABORT, 'optimization memory is append-only');
       END""",
)


def _safe_path(path: os.PathLike[str] | str) -> Path:
    try:
        candidate = Path(os.path.abspath(os.fspath(path)))
    except (TypeError, ValueError) as exc:
        raise UnsafeDatabaseError("database path must be a filesystem path") from exc
    if str(candidate) in {":memory:", ""}:
        raise UnsafeDatabaseError("a named regular database file is required")
    # Check all existing ancestor components without resolving through them.
    current = Path(candidate.anchor or os.sep)
    for component in candidate.parts[1:-1]:
        current /= component
        try:
            component_info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise UnsafeDatabaseError(f"cannot inspect database ancestor: {exc}") from exc
        if stat.S_ISLNK(component_info.st_mode):
            raise UnsafeDatabaseError("database ancestor may not be a symlink")
        if not stat.S_ISDIR(component_info.st_mode):
            raise UnsafeDatabaseError("database ancestor must be a directory")
    try:
        info = candidate.lstat()
    except FileNotFoundError:
        parent = candidate.parent
        if not parent.exists() or not stat.S_ISDIR(parent.stat().st_mode):
            raise UnsafeDatabaseError("database parent directory must already exist")
        if parent.is_symlink():
            raise UnsafeDatabaseError("database parent may not be a symlink")
        return candidate
    except OSError as exc:
        raise UnsafeDatabaseError(f"cannot inspect database path: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise UnsafeDatabaseError("database symlinks are refused")
    if not stat.S_ISREG(info.st_mode):
        raise UnsafeDatabaseError("database path must be a regular file")
    return candidate


def _identity(path: Path) -> tuple[int, int]:
    """Return a regular leaf's filesystem identity without following links."""

    try:
        info = path.lstat()
    except OSError as exc:
        raise UnsafeDatabaseError(f"cannot inspect database identity: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise UnsafeDatabaseError("database leaf changed to a non-regular file")
    return (info.st_dev, info.st_ino)


def _normalized_sql(value: str | None) -> str:
    return " ".join((value or "").lower().split()).rstrip(";")


def _validate_schema(connection: sqlite3.Connection) -> None:
    """Fail closed unless every v2 table/constraint/index/trigger is exact."""

    table_row = connection.execute(
        "SELECT type, sql FROM sqlite_master WHERE name='optimization_records'"
    ).fetchone()
    if table_row is None or table_row[0] != "table":
        raise UnsafeDatabaseError("optimization_records table is missing")
    expected_table_sql = _CREATE_TABLE.replace("CREATE TABLE IF NOT EXISTS", "CREATE TABLE")
    if _normalized_sql(table_row[1]) != _normalized_sql(expected_table_sql):
        raise UnsafeDatabaseError("optimization_records SQL schema is not exact")
    expected_columns = [
        ("seq", "INTEGER", 0, 1),
        ("record_id", "TEXT", 1, 0),
        ("kind", "TEXT", 1, 0),
        ("quality", "TEXT", 1, 0),
        ("phase", "TEXT", 1, 0),
        ("payload", "BLOB", 1, 0),
        ("payload_hash", "TEXT", 1, 0),
        ("source_hash", "TEXT", 0, 0),
        ("created_at", "TEXT", 1, 0),
        ("prev_record_hash", "TEXT", 1, 0),
        ("record_hash", "TEXT", 1, 0),
    ]
    columns = connection.execute("PRAGMA table_info(optimization_records)").fetchall()
    actual_columns = [(row[1], row[2], row[3], row[5]) for row in columns]
    if actual_columns != expected_columns:
        raise UnsafeDatabaseError("optimization_records columns/types are not exact")

    expected_index_sql = {
        "optimization_records_kind_seq": _CREATE_INDEXES[0].replace(
            "IF NOT EXISTS ", ""
        ),
        "optimization_records_quality_seq": _CREATE_INDEXES[1].replace(
            "IF NOT EXISTS ", ""
        ),
    }
    expected_trigger_sql = {
        "optimization_records_no_update": _CREATE_TRIGGERS[0].replace(
            "IF NOT EXISTS ", ""
        ),
        "optimization_records_no_delete": _CREATE_TRIGGERS[1].replace(
            "IF NOT EXISTS ", ""
        ),
    }
    objects = connection.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE name='optimization_records' "
        "OR (tbl_name='optimization_records' AND type IN ('index','trigger'))"
    ).fetchall()
    object_names = {row[1] for row in objects}
    index_rows = connection.execute("PRAGMA index_list(optimization_records)").fetchall()
    autoindexes = {row[1] for row in index_rows if row[1].startswith("sqlite_autoindex_optimization_records_")}
    if autoindexes != {
        "sqlite_autoindex_optimization_records_1",
        "sqlite_autoindex_optimization_records_2",
    }:
        raise UnsafeDatabaseError("unique record indexes are not exact")
    expected_names = {
        "optimization_records",
        *expected_index_sql,
        *expected_trigger_sql,
        *autoindexes,
    }
    if object_names != expected_names:
        raise UnsafeDatabaseError("unexpected optimization memory schema objects")
    by_name = {row[1]: row for row in objects}
    for name, sql in {**expected_index_sql, **expected_trigger_sql}.items():
        if _normalized_sql(by_name[name][2]) != _normalized_sql(sql):
            raise UnsafeDatabaseError(f"schema object SQL is not exact: {name}")
    index_flags = {row[1]: row[2] for row in index_rows}
    if set(index_flags) != {
        "optimization_records_kind_seq",
        "optimization_records_quality_seq",
        *autoindexes,
    }:
        raise UnsafeDatabaseError("optimization memory indexes are not exact")
    if index_flags["optimization_records_kind_seq"] != 0 or index_flags["optimization_records_quality_seq"] != 0:
        raise UnsafeDatabaseError("filter indexes must not be unique")
    if any(index_flags[name] != 1 for name in autoindexes):
        raise UnsafeDatabaseError("record identity indexes must be unique")
    for name, expected_columns_for_index in (
        ("optimization_records_kind_seq", ("kind", "seq")),
        ("optimization_records_quality_seq", ("quality", "seq")),
    ):
        actual = tuple(row[2] for row in connection.execute(f"PRAGMA index_info({name})"))
        if actual != expected_columns_for_index:
            raise UnsafeDatabaseError(f"index columns are not exact: {name}")


class OptimizationMemoryV2:
    """Secure append-only SQLite memory for typed optimization evidence.

    Parameters are bounded by default and may be lowered for tests or a small
    deployment.  Existing files are never migrated destructively: application
    identity and user version must already match, while missing v2 tables are
    created transactionally.
    """

    application_id = APPLICATION_ID
    user_version = USER_VERSION

    def __init__(
        self,
        path: os.PathLike[str] | str,
        *,
        max_records: int = MAX_RECORDS,
        max_payload_bytes: int = MAX_PAYLOAD_BYTES,
        max_snapshot_bytes: int = MAX_SNAPSHOT_BYTES,
        max_database_bytes: int = MAX_DATABASE_BYTES,
    ) -> None:
        if not isinstance(max_records, int) or isinstance(max_records, bool) or max_records <= 0:
            raise ValueError("max_records must be a positive integer")
        if not isinstance(max_payload_bytes, int) or isinstance(max_payload_bytes, bool) or max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be a positive integer")
        if not isinstance(max_snapshot_bytes, int) or isinstance(max_snapshot_bytes, bool) or max_snapshot_bytes <= 0:
            raise ValueError("max_snapshot_bytes must be a positive integer")
        if not isinstance(max_database_bytes, int) or isinstance(max_database_bytes, bool) or max_database_bytes <= 0:
            raise ValueError("max_database_bytes must be a positive integer")
        self.path = _safe_path(path)
        self.max_records = max_records
        self.max_payload_bytes = max_payload_bytes
        self.max_snapshot_bytes = max_snapshot_bytes
        self.max_database_bytes = max_database_bytes
        self._lock = threading.RLock()
        self._ensure_file()
        self._connection = self._open(read_only=False)
        self._database_identity = _identity(self.path)
        try:
            self._initialize()
        except Exception:
            self._connection.close()
            raise

    def _ensure_file(self) -> None:
        if self.path.exists():
            before = _identity(self.path)
            os.chmod(self.path, 0o600, follow_symlinks=False)
            if _identity(self.path) != before:
                raise UnsafeDatabaseError("database leaf changed while securing permissions")
            return
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        mode = stat.S_IRUSR | stat.S_IWUSR
        try:
            descriptor = os.open(self.path, flags, mode)
        except FileExistsError:
            # A concurrent creator is okay only if it is a regular non-symlink;
            # revalidate rather than following a race-created link.
            _safe_path(self.path)
        except OSError as exc:
            raise UnsafeDatabaseError(f"cannot create database: {exc}") from exc
        else:
            os.close(descriptor)
        _safe_path(self.path)
        before = _identity(self.path)
        os.chmod(self.path, 0o600, follow_symlinks=False)
        if _identity(self.path) != before:
            raise UnsafeDatabaseError("database leaf changed while securing permissions")

    def _open(self, *, read_only: bool) -> sqlite3.Connection:
        _safe_path(self.path)
        before = _identity(self.path)
        if read_only:
            uri = f"file:{quote(str(self.path), safe='/')}?mode=ro"
            connection = sqlite3.connect(
                uri, uri=True, isolation_level=None, check_same_thread=False
            )
        else:
            connection = sqlite3.connect(
                self.path, isolation_level=None, check_same_thread=False
            )
        try:
            _safe_path(self.path)
            if _identity(self.path) != before:
                raise UnsafeDatabaseError("database leaf changed during connection")
        except Exception:
            connection.close()
            raise
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        if read_only:
            connection.execute("PRAGMA query_only=ON")
        else:
            connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize(self) -> None:
        conn = self._connection
        application_id = conn.execute("PRAGMA application_id").fetchone()[0]
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
        if application_id not in (0, APPLICATION_ID):
            raise UnsafeDatabaseError(f"unexpected SQLite application_id: {application_id}")
        if user_version not in (0, USER_VERSION):
            raise UnsafeDatabaseError(f"unexpected SQLite user_version: {user_version}")
        # A v2 user_version without our application identity is not silently
        # claimed.  This prevents opening an unrelated SQLite file by accident.
        if user_version == USER_VERSION and application_id != APPLICATION_ID:
            raise UnsafeDatabaseError("v2 database has the wrong application_id")
        self._check_storage_size()
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        if journal_mode.lower() != "delete":
            raise UnsafeDatabaseError("SQLite delete journal mode is unavailable")
        existing = conn.execute(
            "SELECT name, type FROM sqlite_master WHERE name='optimization_records'"
        ).fetchone()
        if existing is None and application_id == APPLICATION_ID and user_version == USER_VERSION:
            raise UnsafeDatabaseError("v2 database is missing its records table")
        if existing is None and application_id == 0 and user_version == 0:
            foreign_objects = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table','index','trigger') AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if foreign_objects:
                raise UnsafeDatabaseError("database contains unrelated SQLite objects")
        if existing is not None and existing["type"] != "table":
            raise UnsafeDatabaseError("optimization_records is not a table")
        if existing is not None:
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(optimization_records)")
            }
            if not _EXPECTED_COLUMNS.issubset(columns):
                raise UnsafeDatabaseError("existing optimization memory schema is incompatible")
            # Never repair a live v2 schema in place: a missing/tampered index
            # or trigger is evidence corruption and must fail closed.
            _validate_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(f"PRAGMA application_id={APPLICATION_ID}")
            conn.execute(f"PRAGMA user_version={USER_VERSION}")
            conn.execute(_CREATE_TABLE)
            for statement in (*_CREATE_INDEXES, *_CREATE_TRIGGERS):
                conn.execute(statement)
            _validate_schema(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        os.chmod(self.path, 0o600, follow_symlinks=False)
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        max_pages = max(1, self.max_database_bytes // page_size)
        conn.execute(f"PRAGMA max_page_count={max_pages}")
        self._check_storage_size()
        if conn.execute("SELECT COUNT(*) FROM optimization_records").fetchone()[0] > self.max_records:
            raise MemoryLimitError("database contains more records than configured bound")

    def close(self) -> None:
        """Close the writer connection."""

        with self._lock:
            self._connection.close()

    def _assert_live_database(self) -> None:
        _safe_path(self.path)
        if _identity(self.path) != self._database_identity:
            raise UnsafeDatabaseError("database leaf changed after opening")

    def _storage_paths(self) -> tuple[Path, ...]:
        """Return the main database and SQLite sidecars relevant to size limits."""

        return (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
            Path(f"{self.path}-journal"),
        )

    def _check_storage_size(self) -> None:
        total = 0
        for candidate in self._storage_paths():
            try:
                info = candidate.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise UnsafeDatabaseError(f"cannot inspect database sidecar: {exc}") from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise UnsafeDatabaseError("database sidecars must be regular non-symlink files")
            total += info.st_size
        if total > self.max_database_bytes:
            raise MemoryLimitError("database and SQLite sidecars exceed configured byte bound")

    def _verify_chain_on_connection(self, connection: sqlite3.Connection) -> bool:
        rows = connection.execute(
            "SELECT * FROM optimization_records ORDER BY seq ASC LIMIT ?", (self.max_records + 1,)
        ).fetchall()
        if len(rows) > self.max_records:
            return False
        previous = GENESIS_HASH
        for row in rows:
            try:
                payload = bytes(row["payload"])
                parsed = loads_strict(payload, max_bytes=self.max_payload_bytes)
                if canonical_bytes(parsed, max_bytes=self.max_payload_bytes) != payload:
                    return False
                if sha256_hex(payload) != row["payload_hash"]:
                    return False
                if row["prev_record_hash"] != previous:
                    return False
                record = OptimizationRecord(
                    record_id=row["record_id"], kind=row["kind"], quality=row["quality"],
                    phase=row["phase"], payload=parsed, source_hash=row["source_hash"],
                    created_at=row["created_at"],
                )
                expected = self._record_hash(
                    record, created_at=row["created_at"],
                    prev_record_hash=row["prev_record_hash"],
                )
                if expected != row["record_hash"]:
                    return False
                previous = row["record_hash"]
            except (ValueError, TypeError, CanonicalJSONError, UnicodeError):
                return False
        return True

    def _verify_preappend(self, connection: sqlite3.Connection) -> None:
        """Verify every integrity layer while the caller owns ``BEGIN IMMEDIATE``."""

        sqlite_result = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if sqlite_result != "ok":
            raise IntegrityError(f"SQLite integrity_check failed: {sqlite_result}")
        if connection.execute("PRAGMA journal_mode").fetchone()[0].lower() != "delete":
            raise UnsafeDatabaseError("WAL or another non-delete journal mode is refused")
        _validate_schema(connection)
        if not self._verify_chain_on_connection(connection):
            raise IntegrityError("optimization memory hash chain is invalid")
        self._check_storage_size()

    def __enter__(self) -> "OptimizationMemoryV2":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextlib.contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a read-only, foreign-key-enforcing SQLite connection."""

        with self._lock:
            connection = self._open(read_only=True)
            try:
                yield connection
            finally:
                connection.close()

    def _record_hash(
        self,
        record: OptimizationRecord,
        *,
        created_at: str,
        prev_record_hash: str,
    ) -> str:
        envelope = {
            "created_at": created_at,
            "kind": record.kind.value,
            "payload_hash": record.payload_hash,
            "phase": record.phase.value,
            "prev_record_hash": prev_record_hash,
            "quality": record.quality.value,
            "record_id": record.record_id,
            "source_hash": record.source_hash,
        }
        return sha256_hex(canonical_bytes(envelope, max_bytes=16_384))

    @staticmethod
    def _same_record_input(first: OptimizationRecord, second: OptimizationRecord) -> bool:
        return (
            first.record_id == second.record_id
            and first.kind == second.kind
            and first.quality == second.quality
            and first.phase == second.phase
            and first.payload_bytes == second.payload_bytes
            and first.source_hash == second.source_hash
            and first.created_at == second.created_at
        )

    @staticmethod
    def _same_input(row: sqlite3.Row, record: OptimizationRecord) -> bool:
        return (
            row["record_id"] == record.record_id
            and row["kind"] == record.kind.value
            and row["quality"] == record.quality.value
            and row["phase"] == record.phase.value
            and row["payload_hash"] == record.payload_hash
            and bytes(row["payload"]) == record.payload_bytes
            and row["source_hash"] == record.source_hash
            and row["created_at"] == (record.created_at or "")
        )

    def append(self, record: OptimizationRecord) -> sqlite3.Row:
        """Atomically append *record* or return the exact existing row.

        A repeated ID with identical payload is idempotent.  Any other payload
        for that ID raises :class:`MemoryConflictError`; failures roll back all
        writes made by this call.
        """

        with self._lock:
            if not isinstance(record, OptimizationRecord):
                raise TypeError("append expects OptimizationRecord")
            payload = record.payload_bytes
            if len(payload) > self.max_payload_bytes:
                raise MemoryLimitError("record payload exceeds configured bound")
            self._assert_live_database()
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._verify_preappend(conn)
                existing = conn.execute(
                    "SELECT * FROM optimization_records WHERE record_id=?", (record.record_id,)
                ).fetchone()
                if existing is not None:
                    if self._same_input(existing, record):
                        conn.commit()
                        return existing
                    raise MemoryConflictError(
                        f"record_id already contains different input: {record.record_id}"
                    )
                count = conn.execute("SELECT COUNT(*) FROM optimization_records").fetchone()[0]
                if count >= self.max_records:
                    raise MemoryLimitError("maximum record count exceeded")
                previous = conn.execute(
                    "SELECT record_hash FROM optimization_records ORDER BY seq DESC LIMIT 1"
                ).fetchone()
                previous_hash = previous[0] if previous is not None else GENESIS_HASH
                # Timestamps are caller-supplied evidence.  Omitting one is kept
                # deterministic so identical input in two stores has identical
                # snapshot/hash output; no wall-clock value is injected implicitly.
                created_at = record.created_at or ""
                record_hash = self._record_hash(
                    record, created_at=created_at, prev_record_hash=previous_hash
                )
                conn.execute(
                    """INSERT INTO optimization_records
                    (record_id, kind, quality, phase, payload, payload_hash, source_hash,
                     created_at, prev_record_hash, record_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (record.record_id, record.kind.value, record.quality.value,
                     record.phase.value, payload, record.payload_hash, record.source_hash,
                     created_at, previous_hash, record_hash),
                )
                self._check_storage_size()
                row = conn.execute(
                    "SELECT * FROM optimization_records WHERE record_id=?", (record.record_id,)
                ).fetchone()
                conn.commit()
                assert row is not None
                return row
            except Exception:
                conn.rollback()
                raise

    def append_many(self, records: Iterable[OptimizationRecord]) -> list[sqlite3.Row]:
        """Append a batch atomically, preserving input order and chain order."""

        with self._lock:
            # Untrusted generators are bounded before any write transaction starts.
            materialized: list[OptimizationRecord] = []
            seen: dict[str, OptimizationRecord] = {}
            for index, record in enumerate(records):
                if index >= self.max_records + 1:
                    raise MemoryLimitError("append_many input exceeds configured row bound")
                if not isinstance(record, OptimizationRecord):
                    raise TypeError("append_many expects OptimizationRecord values")
                payload = record.payload_bytes
                if len(payload) > self.max_payload_bytes:
                    raise MemoryLimitError("record payload exceeds configured bound")
                prior = seen.get(record.record_id)
                if prior is not None and not self._same_record_input(prior, record):
                    raise MemoryConflictError(f"record_id conflict in batch: {record.record_id}")
                seen.setdefault(record.record_id, record)
                materialized.append(record)
            self._assert_live_database()
            conn = self._connection
            existing_ids = {
                record_id
                for record_id in seen
                if conn.execute(
                    "SELECT 1 FROM optimization_records WHERE record_id=?", (record_id,)
                ).fetchone()
            }
            current_count = conn.execute("SELECT COUNT(*) FROM optimization_records").fetchone()[0]
            if current_count + len(seen) - len(existing_ids) > self.max_records:
                raise MemoryLimitError("maximum record count exceeded")
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._verify_preappend(conn)
                result: list[sqlite3.Row] = []
                for record in materialized:
                    payload = record.payload_bytes
                    existing = conn.execute(
                        "SELECT * FROM optimization_records WHERE record_id=?", (record.record_id,)
                    ).fetchone()
                    if existing is not None:
                        if not self._same_input(existing, record):
                            raise MemoryConflictError(f"record_id conflict: {record.record_id}")
                        result.append(existing)
                        continue
                    count = conn.execute("SELECT COUNT(*) FROM optimization_records").fetchone()[0]
                    if count >= self.max_records:
                        raise MemoryLimitError("maximum record count exceeded")
                    previous = conn.execute(
                        "SELECT record_hash FROM optimization_records ORDER BY seq DESC LIMIT 1"
                    ).fetchone()
                    previous_hash = previous[0] if previous is not None else GENESIS_HASH
                    created_at = record.created_at or ""
                    record_hash = self._record_hash(
                        record, created_at=created_at, prev_record_hash=previous_hash
                    )
                    conn.execute(
                        """INSERT INTO optimization_records
                        (record_id, kind, quality, phase, payload, payload_hash, source_hash,
                         created_at, prev_record_hash, record_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (record.record_id, record.kind.value, record.quality.value,
                         record.phase.value, payload, record.payload_hash, record.source_hash,
                         created_at, previous_hash, record_hash),
                    )
                    result.append(conn.execute(
                        "SELECT * FROM optimization_records WHERE record_id=?", (record.record_id,)
                    ).fetchone())
                self._check_storage_size()
                conn.commit()
                return result
            except Exception:
                conn.rollback()
                raise

    def get(self, record_id: str) -> sqlite3.Row | None:
        """Return one immutable row by ID, or ``None``."""

        with self._lock:
            with self.read_connection() as conn:
                return conn.execute(
                    "SELECT * FROM optimization_records WHERE record_id=?", (record_id,)
                ).fetchone()

    def list(
        self,
        *,
        kind: RecordKind | str | None = None,
        quality: QualityClass | str | None = None,
        phase: DataPhase | str | None = None,
        limit: int = 1_000,
        offset: int = 0,
    ) -> list[sqlite3.Row]:
        """Return deterministic sequence-ordered rows under a bounded page."""

        if not isinstance(limit, int) or isinstance(limit, bool) or not 0 <= limit <= 10_000:
            raise ValueError("limit must be an integer between 0 and 10000")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        clauses: list[str] = []
        values: list[str] = []
        for name, value, enum_type in (
            ("kind", kind, RecordKind),
            ("quality", quality, QualityClass),
            ("phase", phase, DataPhase),
        ):
            if value is not None:
                if isinstance(value, Enum):
                    value = value.value
                if not isinstance(value, str) or value not in {item.value for item in enum_type}:
                    raise ValueError(f"invalid {name}")
                clauses.append(f"{name}=?")
                values.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._lock:
            with self.read_connection() as conn:
                return conn.execute(
                    f"SELECT * FROM optimization_records{where} ORDER BY seq ASC LIMIT ? OFFSET ?",
                    (*values, limit, offset),
                ).fetchall()

    def verify_chain(self) -> bool:
        """Verify payload hashes, links, record hashes, and typed columns."""

        with self._lock:
            try:
                with self.read_connection() as conn:
                    _validate_schema(conn)
                    return self._verify_chain_on_connection(conn)
            except (sqlite3.DatabaseError, OSError, UnsafeDatabaseError):
                return False

    def integrity(self) -> IntegrityReport:
        """Return SQLite and hash-chain integrity without mutating the DB."""

        with self._lock:
            try:
                with self.read_connection() as conn:
                    sqlite_result = conn.execute("PRAGMA integrity_check").fetchone()[0]
                    rows = conn.execute("SELECT COUNT(*) FROM optimization_records").fetchone()[0]
                    _validate_schema(conn)
                    chain_ok = self._verify_chain_on_connection(conn)
                sqlite_ok = sqlite_result == "ok"
                return IntegrityReport(sqlite_ok and chain_ok, sqlite_ok, chain_ok, rows)
            except (sqlite3.DatabaseError, OSError, UnsafeDatabaseError) as exc:
                return IntegrityReport(False, False, False, 0, str(exc))

    verify_integrity = verify_chain

    def snapshot(self) -> bytes:
        """Return a deterministic canonical JSON snapshot of all records."""

        prefix = (
            b'{"application_id":' + canonical_bytes(APPLICATION_ID)
            + b',"genesis_hash":' + canonical_bytes(GENESIS_HASH)
            + b',"records":['
        )
        suffix = b'],"user_version":' + canonical_bytes(USER_VERSION) + b"}"
        if len(prefix) + len(suffix) > self.max_snapshot_bytes:
            raise MemoryLimitError("snapshot exceeds configured byte bound")
        output = bytearray(prefix)
        first = True
        with self.read_connection() as conn:
            cursor = conn.execute("SELECT * FROM optimization_records ORDER BY seq ASC")
            seen_rows = 0
            while True:
                rows = cursor.fetchmany(64)
                if not rows:
                    break
                for row in rows:
                    seen_rows += 1
                    if seen_rows > self.max_records:
                        raise MemoryLimitError("snapshot row bound exceeded")
                    payload = loads_strict(bytes(row["payload"]), max_bytes=self.max_payload_bytes)
                    row_value = {
                        "seq": row["seq"], "record_id": row["record_id"],
                        "kind": row["kind"], "quality": row["quality"],
                        "phase": row["phase"], "payload": payload,
                        "payload_hash": row["payload_hash"], "source_hash": row["source_hash"],
                        "created_at": row["created_at"],
                        "prev_record_hash": row["prev_record_hash"],
                        "record_hash": row["record_hash"],
                    }
                    try:
                        encoded_row = canonical_bytes(row_value, max_bytes=self.max_snapshot_bytes)
                    except CanonicalJSONError as exc:
                        raise MemoryLimitError("snapshot exceeds configured byte bound") from exc
                    extra = len(encoded_row) + (0 if first else 1)
                    if len(output) + extra + len(suffix) > self.max_snapshot_bytes:
                        raise MemoryLimitError("snapshot exceeds configured byte bound")
                    if not first:
                        output.extend(b",")
                    output.extend(encoded_row)
                    first = False
        output.extend(suffix)
        return bytes(output)

    def snapshot_hash(self) -> str:
        """Return the SHA-256 identity of :meth:`snapshot`."""

        return sha256_hex(self.snapshot())

    # Explicit aliases make the public boundary discoverable without exposing
    # the private SQLite connection as an application API.
    append_record = append
    append_records = append_many
    get_record = get
    list_records = list
    integrity_check = integrity

    @classmethod
    def open_read_only(
        cls,
        path: os.PathLike[str] | str,
        *,
        max_records: int = MAX_RECORDS,
        max_payload_bytes: int = MAX_PAYLOAD_BYTES,
        max_snapshot_bytes: int = MAX_SNAPSHOT_BYTES,
    ) -> "ReadOnlyMemoryView":
        """Open an existing v2 database without any filesystem/database writes.

        Unlike the normal writer constructor this method never creates a leaf,
        changes permissions, migrates a schema, changes journal mode, or sets a
        persistent SQLite pragma.  A missing, symlinked, malformed, or changed
        database fails closed.
        """

        return ReadOnlyMemoryView(path, max_records=max_records, max_payload_bytes=max_payload_bytes, max_snapshot_bytes=max_snapshot_bytes)


class ReadOnlyMemoryView:
    """Stable, read-only view over an existing :class:`OptimizationMemoryV2`.

    Every operation opens ``mode=ro`` and enables SQLite's connection-local
    ``query_only`` flag.  The path's device/inode identity is checked before
    and after each read so a replacement cannot be mistaken for the original
    evidence store.
    """

    __slots__ = (
        "path", "identity", "max_records", "max_payload_bytes", "max_snapshot_bytes",
        "_closed",
    )

    def __init__(
        self,
        path: os.PathLike[str] | str,
        *,
        max_records: int = MAX_RECORDS,
        max_payload_bytes: int = MAX_PAYLOAD_BYTES,
        max_snapshot_bytes: int = MAX_SNAPSHOT_BYTES,
    ) -> None:
        if not isinstance(max_records, int) or isinstance(max_records, bool) or max_records <= 0:
            raise ValueError("max_records must be a positive integer")
        if not isinstance(max_payload_bytes, int) or isinstance(max_payload_bytes, bool) or max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be a positive integer")
        if not isinstance(max_snapshot_bytes, int) or isinstance(max_snapshot_bytes, bool) or max_snapshot_bytes <= 0:
            raise ValueError("max_snapshot_bytes must be a positive integer")
        candidate = _safe_path(path)
        if not candidate.exists():
            raise UnsafeDatabaseError("read-only memory database does not exist")
        self.path = candidate
        self.identity = _identity(candidate)
        self.max_records = max_records
        self.max_payload_bytes = max_payload_bytes
        self.max_snapshot_bytes = max_snapshot_bytes
        self._closed = False
        # Open and validate once, then close.  This performs no writes and
        # ensures constructor success means the schema itself is trustworthy.
        with self.read_connection() as connection:
            _validate_schema(connection)

    def _record_hash(
        self,
        record: OptimizationRecord,
        *,
        created_at: str,
        prev_record_hash: str,
    ) -> str:
        return OptimizationMemoryV2._record_hash(
            self, record, created_at=created_at, prev_record_hash=prev_record_hash
        )

    def _assert_stable(self) -> None:
        if self._closed:
            raise UnsafeDatabaseError("read-only memory view is closed")
        if _identity(self.path) != self.identity:
            raise UnsafeDatabaseError("database path identity changed")

    @contextlib.contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        self._assert_stable()
        uri = f"file:{quote(str(self.path), safe='/')}?mode=ro"
        try:
            connection = sqlite3.connect(
                uri, uri=True, isolation_level=None, check_same_thread=False
            )
        except sqlite3.Error as exc:
            raise UnsafeDatabaseError("cannot open memory database read-only") from exc
        connection.row_factory = sqlite3.Row
        try:
            # query_only is connection-local; it cannot alter the database in
            # a mode=ro connection and is independently checked below.
            connection.execute("PRAGMA query_only=ON")
            if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
                raise UnsafeDatabaseError("SQLite query_only was not enabled")
            _validate_schema(connection)
            yield connection
            self._assert_stable()
        finally:
            connection.close()

    @property
    def application_id(self) -> int:
        with self.read_connection() as connection:
            return int(connection.execute("PRAGMA application_id").fetchone()[0])

    @property
    def user_version(self) -> int:
        with self.read_connection() as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    @property
    def read_only(self) -> bool:
        return True

    @property
    def schema_version(self) -> int:
        return self.user_version

    @property
    def schema_ok(self) -> bool:
        try:
            with self.read_connection():
                return True
        except (OSError, sqlite3.Error, OptimizationMemoryError):
            return False

    @property
    def schema(self) -> Mapping[str, Any]:
        with self.read_connection() as connection:
            columns = tuple(tuple(row) for row in connection.execute("PRAGMA table_info(optimization_records)"))
            return {"application_id": self.application_id, "user_version": self.user_version, "table": "optimization_records", "columns": columns}

    def _verify_chain(self, connection: sqlite3.Connection) -> bool:
        return OptimizationMemoryV2._verify_chain_on_connection(self, connection)

    def verify_chain(self) -> bool:
        try:
            with self.read_connection() as connection:
                return self._verify_chain(connection)
        except (sqlite3.Error, OSError, OptimizationMemoryError):
            return False

    def integrity(self) -> IntegrityReport:
        try:
            with self.read_connection() as connection:
                sqlite_result = connection.execute("PRAGMA integrity_check").fetchone()[0]
                rows = int(connection.execute("SELECT COUNT(*) FROM optimization_records").fetchone()[0])
                chain_ok = self._verify_chain(connection)
                sqlite_ok = sqlite_result == "ok"
                return IntegrityReport(sqlite_ok and chain_ok, sqlite_ok, chain_ok, rows)
        except (sqlite3.Error, OSError, OptimizationMemoryError) as exc:
            return IntegrityReport(False, False, False, 0, str(exc))

    def get(self, record_id: str) -> sqlite3.Row | None:
        with self.read_connection() as connection:
            return connection.execute("SELECT * FROM optimization_records WHERE record_id=?", (record_id,)).fetchone()

    def list(
        self,
        *,
        kind: RecordKind | str | None = None,
        quality: QualityClass | str | None = None,
        phase: DataPhase | str | None = None,
        limit: int = 1_000,
        offset: int = 0,
    ) -> list[sqlite3.Row]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 0 <= limit <= 10_000:
            raise ValueError("limit must be an integer between 0 and 10000")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        clauses: list[str] = []
        values: list[str] = []
        for name, value, enum_type in (("kind", kind, RecordKind), ("quality", quality, QualityClass), ("phase", phase, DataPhase)):
            if value is not None:
                if isinstance(value, Enum):
                    value = value.value
                if not isinstance(value, str) or value not in {item.value for item in enum_type}:
                    raise ValueError(f"invalid {name}")
                clauses.append(name + "=?")
                values.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.read_connection() as connection:
            return connection.execute(
                f"SELECT * FROM optimization_records{where} ORDER BY seq ASC LIMIT ? OFFSET ?",
                (*values, limit, offset),
            ).fetchall()

    def snapshot(self) -> bytes:
        # Reuse the canonical writer implementation without exposing a writer
        # connection; all nested reads still go through this view.
        prefix = b'{"application_id":' + canonical_bytes(APPLICATION_ID) + b',"genesis_hash":' + canonical_bytes(GENESIS_HASH) + b',"records":['
        suffix = b'],"user_version":' + canonical_bytes(USER_VERSION) + b"}"
        output = bytearray(prefix)
        first = True
        with self.read_connection() as connection:
            seen_rows = 0
            for row in connection.execute("SELECT * FROM optimization_records ORDER BY seq ASC"):
                seen_rows += 1
                if seen_rows > self.max_records:
                    raise MemoryLimitError("snapshot row bound exceeded")
                payload = loads_strict(bytes(row["payload"]), max_bytes=self.max_payload_bytes)
                encoded = canonical_bytes({"seq": row["seq"], "record_id": row["record_id"], "kind": row["kind"], "quality": row["quality"], "phase": row["phase"], "payload": payload, "payload_hash": row["payload_hash"], "source_hash": row["source_hash"], "created_at": row["created_at"], "prev_record_hash": row["prev_record_hash"], "record_hash": row["record_hash"]}, max_bytes=self.max_snapshot_bytes)
                extra = len(encoded) + (0 if first else 1)
                if len(output) + extra + len(suffix) > self.max_snapshot_bytes:
                    raise MemoryLimitError("snapshot exceeds configured byte bound")
                if not first:
                    output.extend(b",")
                output.extend(encoded)
                first = False
        output.extend(suffix)
        return bytes(output)

    def snapshot_hash(self) -> str:
        return sha256_hex(self.snapshot())

    def append(self, *_: Any, **__: Any) -> None:
        raise UnsafeDatabaseError("read-only memory view cannot append")

    append_many = append
    append_record = append
    append_records = append_many
    get_record = get
    list_records = list
    integrity_check = integrity
    verify_integrity = verify_chain

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "ReadOnlyMemoryView":
        self._assert_stable()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "APPLICATION_ID",
    "GENESIS_HASH",
    "IntegrityError",
    "IntegrityReport",
    "MAX_PAYLOAD_BYTES",
    "MAX_DATABASE_BYTES",
    "MAX_RECORDS",
    "MemoryConflictError",
    "MemoryLimitError",
    "OptimizationMemoryError",
    "OptimizationMemoryV2",
    "ReadOnlyMemoryView",
    "UnsafeDatabaseError",
]
