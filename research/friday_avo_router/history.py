"""Private append-only SQLite history for shadow-router engineering evidence."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import quote

from friday_n10_v2.canonical import canonical_json_bytes, canonical_sha256, strict_json_loads

from .constants import (
    HISTORY_KINDS,
    MAX_CANONICAL_BYTES,
    MAX_DATABASE_BYTES,
    MAX_HISTORY_ROWS,
    ROUTER_ID,
    SCHEMA_VERSION,
    SQLITE_APPLICATION_ID,
)


_MIGRATION = Path(__file__).with_name("migrations") / "001_init.sql"
_HEX = frozenset("0123456789abcdef")
_PROVENANCE_KEYS = {
    "router_id",
    "schema_version",
    "git_revision",
    "git_dirty",
    "git_diff_sha256",
    "code_files",
    "code_sha256",
    "spec_files",
    "spec_sha256",
    "environment",
    "environment_sha256",
    "hardware",
    "hardware_sha256",
    "provenance_sha256",
}


class HistoryError(RuntimeError):
    """Router evidence cannot be stored or replayed safely."""


class HistoryConflict(HistoryError):
    """A run identifier is already bound to different canonical bytes."""


@dataclass(frozen=True)
class PersistenceOutcome:
    state: str
    record_id: str


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _migration_bytes() -> bytes:
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(_MIGRATION, flags)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_CANONICAL_BYTES:
            raise HistoryError("router migration is unsafe or oversized")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            payload = handle.read(MAX_CANONICAL_BYTES + 1)
            after = os.fstat(handle.fileno())
        path_after = _MIGRATION.lstat()
    except OSError as exc:
        raise HistoryError("router migration is unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        len(payload) > MAX_CANONICAL_BYTES
        or len(payload) != info.st_size
        or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        != (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
        )
    ):
        raise HistoryError("router migration is not a stable regular file")
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
        parent = path.parent.lstat()
    except OSError as exc:
        raise HistoryError("database parent is unavailable") from exc
    if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
        raise HistoryError("database parent must be a real directory")
    return path


def _verify_file(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise HistoryError("router database is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise HistoryError("router database must be a regular non-symlink file")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise HistoryError("router database must have mode 0600")
    if info.st_size > MAX_DATABASE_BYTES:
        raise HistoryError("router database exceeds the fixed size budget")
    return info


def _secure_create(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError as exc:
        raise HistoryError("router database cannot be created safely") from exc
    os.close(descriptor)


def _connect(path: Path, *, read_only: bool) -> sqlite3.Connection:
    if read_only:
        uri = f"file:{quote(str(path), safe='/')}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5.0)
    else:
        connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA trusted_schema=OFF")
    connection.execute("PRAGMA busy_timeout=5000")
    if read_only:
        connection.execute("PRAGMA query_only=ON")
    return connection


def _master_snapshot(connection: sqlite3.Connection) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        tuple(row)
        for row in connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
        )
    )


@lru_cache(maxsize=1)
def _expected_snapshot() -> tuple[tuple[str, str, str, str], ...]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(_migration_bytes().decode("utf-8", errors="strict"))
        return _master_snapshot(connection)
    finally:
        connection.close()


def _verify_schema(connection: sqlite3.Connection) -> None:
    app_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if app_id != SQLITE_APPLICATION_ID or user_version != SCHEMA_VERSION:
        raise HistoryError("router database identity differs")
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise HistoryError("router database integrity check failed")
    if _master_snapshot(connection) != _expected_snapshot():
        raise HistoryError("router database schema differs")
    row = connection.execute(
        "SELECT schema_version,router_id,migration_sha256 FROM metadata WHERE singleton=1"
    ).fetchone()
    if row is None or tuple(row) != (SCHEMA_VERSION, ROUTER_ID, _migration_sha256()):
        raise HistoryError("router metadata differs")


def _canonical_object(
    value: Mapping[str, Any], name: str
) -> tuple[dict[str, Any], bytes, str]:
    if not isinstance(value, Mapping):
        raise HistoryError(f"{name} must be an object")
    plain = dict(value)
    try:
        payload = canonical_json_bytes(plain)
    except (TypeError, ValueError) as exc:
        raise HistoryError(f"{name} is not canonical JSON") from exc
    if len(payload) > MAX_CANONICAL_BYTES:
        raise HistoryError(f"{name} exceeds the canonical size budget")
    return plain, payload, hashlib.sha256(payload).hexdigest()


def _validated_report(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes, str, str]:
    report, payload, digest = _canonical_object(value, "report")
    required = {
        "schema_version",
        "router_id",
        "run_id",
        "kind",
        "status",
        "formal_claim",
        "decision_record_ids",
        "router",
        "metrics",
    }
    if set(report) != required:
        raise HistoryError("router report keys differ")
    if report["schema_version"] != SCHEMA_VERSION or report["router_id"] != ROUTER_ID:
        raise HistoryError("router report identity differs")
    if report["kind"] not in HISTORY_KINDS:
        raise HistoryError("router report kind is not registered")
    if not isinstance(report["run_id"], str) or not 1 <= len(report["run_id"]) <= 120:
        raise HistoryError("router run id is invalid")
    if not isinstance(report["status"], str) or not 1 <= len(report["status"]) <= 96:
        raise HistoryError("router status is invalid")
    if report["formal_claim"] is not False:
        raise HistoryError("router records cannot carry formal claims")
    for key in ("decision_record_ids", "router", "metrics"):
        if not isinstance(report[key], Mapping):
            raise HistoryError(f"router report {key} must be an object")
    decision_ids = report["decision_record_ids"]
    if set(decision_ids) != {"n8", "n10"} or not all(
        _is_sha256(value) for value in decision_ids.values()
    ):
        raise HistoryError("router decision record ids differ")
    router = report["router"]
    if router.get("enforced_plan") != "serial_shadow_only" or not isinstance(
        router.get("ready"), bool
    ):
        raise HistoryError("router report state differs")
    return report, payload, digest, f"{report['kind']}:{report['run_id']}"


def _validated_provenance(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes, str]:
    provenance, payload, digest = _canonical_object(value, "provenance")
    if set(provenance) != _PROVENANCE_KEYS:
        raise HistoryError("router provenance keys differ")
    if provenance.get("router_id") != ROUTER_ID or provenance.get("schema_version") != SCHEMA_VERSION:
        raise HistoryError("router provenance identity differs")
    if provenance.get("git_dirty") is not False:
        raise HistoryError("router provenance must be clean")
    revision = provenance.get("git_revision")
    if not isinstance(revision, str) or len(revision) != 40 or not set(revision) <= _HEX:
        raise HistoryError("router Git revision is invalid")
    for key in (
        "git_diff_sha256",
        "code_sha256",
        "spec_sha256",
        "environment_sha256",
        "hardware_sha256",
        "provenance_sha256",
    ):
        if not _is_sha256(provenance.get(key)):
            raise HistoryError(f"router provenance {key} is invalid")
    for value_key, digest_key in (
        ("code_files", "code_sha256"),
        ("spec_files", "spec_sha256"),
        ("environment", "environment_sha256"),
        ("hardware", "hardware_sha256"),
    ):
        component = provenance.get(value_key)
        if not isinstance(component, Mapping) or not component:
            raise HistoryError(f"router provenance {value_key} is invalid")
        if value_key in {"code_files", "spec_files"} and not all(
            isinstance(path, str)
            and 1 <= len(path) <= 512
            and "\0" not in path
            and _is_sha256(file_digest)
            for path, file_digest in component.items()
        ):
            raise HistoryError(f"router provenance {value_key} is invalid")
        if canonical_sha256(component) != provenance[digest_key]:
            raise HistoryError(f"router provenance {digest_key} differs")
    expected = dict(provenance)
    claimed = expected.pop("provenance_sha256")
    if canonical_sha256(expected) != claimed:
        raise HistoryError("router provenance digest differs")
    return provenance, payload, digest


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
        path = _checked_path(source, create_parent=initialize and not read_only)
        created = False
        if not os.path.lexists(path):
            if read_only or not initialize:
                raise HistoryError("router database does not exist")
            _secure_create(path)
            created = True
        before = _verify_file(path)
        connection = _connect(path, read_only=read_only)
        try:
            after_open = _verify_file(path)
            if (before.st_dev, before.st_ino) != (after_open.st_dev, after_open.st_ino):
                raise HistoryError("router database identity changed while opening")
            if created:
                migration = _migration_bytes().decode("utf-8", errors="strict")
                connection.executescript(migration)
                connection.execute(f"PRAGMA application_id={SQLITE_APPLICATION_ID}")
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                connection.execute(
                    "INSERT INTO metadata(singleton,schema_version,router_id,migration_sha256) "
                    "VALUES(1,?,?,?)",
                    (SCHEMA_VERSION, ROUTER_ID, _migration_sha256()),
                )
                os.chmod(path, 0o600)
            _verify_schema(connection)
        except Exception:
            connection.close()
            if created:
                try:
                    path.unlink()
                except OSError:
                    pass
            raise
        return cls(path, connection, read_only=read_only)

    def __enter__(self) -> "History":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def read_transaction(self) -> Iterator[None]:
        _verify_file(self.path)
        _verify_schema(self.connection)
        self.connection.execute("BEGIN")
        try:
            yield
        finally:
            self.connection.execute("ROLLBACK")

    def verified_records(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT rowid,* FROM records ORDER BY rowid").fetchall()
        if len(rows) > MAX_HISTORY_ROWS:
            raise HistoryError("router history exceeds the row budget")
        previous: str | None = None
        verified: list[dict[str, Any]] = []
        for row in rows:
            try:
                report = strict_json_loads(
                    row["report_json"].encode("utf-8"), maximum=MAX_CANONICAL_BYTES
                )
                provenance = strict_json_loads(
                    row["provenance_json"].encode("utf-8"), maximum=MAX_CANONICAL_BYTES
                )
            except (TypeError, ValueError) as exc:
                raise HistoryError("router record JSON is invalid") from exc
            report_value, report_bytes, report_sha, entity_key = _validated_report(report)
            provenance_value, provenance_bytes, provenance_sha = _validated_provenance(provenance)
            created = row["created_at_unix_ns"]
            if not isinstance(created, int) or not 0 <= created <= 2**63 - 1:
                raise HistoryError("router record timestamp is invalid")
            if (
                row["previous_record_id"] != previous
                or row["router_id"] != ROUTER_ID
                or row["entity_key"] != entity_key
                or row["record_kind"] != report_value["kind"]
                or row["status"] != report_value["status"]
                or row["report_json"].encode("utf-8") != report_bytes
                or row["report_sha256"] != report_sha
                or row["provenance_json"].encode("utf-8") != provenance_bytes
                or row["provenance_sha256"] != provenance_sha
            ):
                raise HistoryError("router record columns differ from canonical content")
            material = {
                "domain": "friday-avo-router-record-v1",
                "previous_record_id": previous,
                "created_at_unix_ns": created,
                "report_sha256": report_sha,
                "provenance_sha256": provenance_sha,
            }
            record_id = canonical_sha256(material)
            if row["record_id"] != record_id:
                raise HistoryError("router record hash chain differs")
            verified.append(
                {
                    "record_id": record_id,
                    "previous_record_id": previous,
                    "created_at_unix_ns": created,
                    "report": report_value,
                    "provenance": provenance_value,
                }
            )
            previous = record_id
        return verified

    def persist(
        self,
        report: Mapping[str, Any],
        provenance: Mapping[str, Any],
        *,
        created_at_unix_ns: int | None = None,
    ) -> PersistenceOutcome:
        if self.read_only:
            raise HistoryError("read-only router history cannot persist")
        report_value, report_bytes, report_sha, entity_key = _validated_report(report)
        _, provenance_bytes, provenance_sha = _validated_provenance(provenance)
        created = time.time_ns() if created_at_unix_ns is None else created_at_unix_ns
        if isinstance(created, bool) or not isinstance(created, int) or not 0 <= created <= 2**63 - 1:
            raise HistoryError("router record timestamp is invalid")
        _verify_file(self.path)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            _verify_schema(self.connection)
            existing_rows = self.verified_records()
            for existing in existing_rows:
                existing_report = existing["report"]
                if f"{existing_report['kind']}:{existing_report['run_id']}" == entity_key:
                    same = (
                        canonical_json_bytes(existing_report) == report_bytes
                        and canonical_json_bytes(existing["provenance"]) == provenance_bytes
                    )
                    if same:
                        self.connection.execute("COMMIT")
                        return PersistenceOutcome("existing", existing["record_id"])
                    raise HistoryConflict("router run id is already bound to different bytes")
            if len(existing_rows) >= MAX_HISTORY_ROWS:
                raise HistoryError("router history row budget is exhausted")
            previous = existing_rows[-1]["record_id"] if existing_rows else None
            record_id = canonical_sha256(
                {
                    "domain": "friday-avo-router-record-v1",
                    "previous_record_id": previous,
                    "created_at_unix_ns": created,
                    "report_sha256": report_sha,
                    "provenance_sha256": provenance_sha,
                }
            )
            self.connection.execute(
                "INSERT INTO records(record_id,previous_record_id,router_id,entity_key,"
                "record_kind,status,created_at_unix_ns,report_json,report_sha256,"
                "provenance_json,provenance_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record_id,
                    previous,
                    ROUTER_ID,
                    entity_key,
                    report_value["kind"],
                    report_value["status"],
                    created,
                    report_bytes.decode("utf-8"),
                    report_sha,
                    provenance_bytes.decode("utf-8"),
                    provenance_sha,
                ),
            )
            self.connection.execute("COMMIT")
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise
        return PersistenceOutcome("inserted", record_id)


def snapshot_revision(records: list[dict[str, Any]]) -> str:
    return canonical_sha256(
        [
            [row["record_id"], row["report"]["kind"], row["report"]["status"]]
            for row in records
        ]
    )
