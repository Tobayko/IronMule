"""Private append-only SQLite history for Phase-1B evidence."""

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

from .canonical import CanonicalError, canonical_json_bytes, canonical_sha256, strict_json_loads
from .constants import (
    BENCHMARK_RUN_ID,
    CONTRACT_ID,
    EXPERIMENT_ID,
    HISTORY_KINDS,
    MAX_CANONICAL_BYTES,
    MAX_DATABASE_BYTES,
    MAX_HISTORY_ROWS,
    QUALIFICATION_RUN_ID,
    SCHEMA_VERSION,
    SQLITE_APPLICATION_ID,
    WORKLOAD_ID,
)
from .kernel_source import HIDDEN_SIZE, KERNEL_NAME, KERNEL_SOURCE_SHA256, ROWS


_MIGRATION = Path(__file__).with_name("migrations") / "001_init.sql"
_HEX = frozenset("0123456789abcdef")
_PROVENANCE_KEYS = {
    "experiment_id",
    "contract_id",
    "schema_version",
    "git_revision",
    "git_dirty",
    "git_status_sha256",
    "code_files",
    "code_sha256",
    "spec_files",
    "spec_sha256",
    "source",
    "source_binding_sha256",
    "environment",
    "environment_sha256",
    "hardware",
    "hardware_sha256",
    "provenance_sha256",
}


class HistoryError(RuntimeError):
    """Phase-1B evidence cannot be stored or replayed safely."""


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
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_CANONICAL_BYTES:
            raise HistoryError("Phase-1B migration is unsafe or oversized")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            payload = handle.read(MAX_CANONICAL_BYTES + 1)
            after = os.fstat(handle.fileno())
        path_after = _MIGRATION.lstat()
    except OSError as exc:
        raise HistoryError("Phase-1B migration is unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if (
        len(payload) > MAX_CANONICAL_BYTES
        or len(payload) != before.st_size
        or identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or identity
        != (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
        )
    ):
        raise HistoryError("Phase-1B migration is not a stable regular file")
    return payload


def _migration_sha256() -> str:
    return hashlib.sha256(_migration_bytes()).hexdigest()


def _checked_path(source: os.PathLike[str] | str, *, create_parent: bool) -> Path:
    path = Path(os.path.abspath(Path(source).expanduser()))
    if create_parent:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        parent = path.parent.lstat()
    except OSError as exc:
        raise HistoryError("database parent is unavailable") from exc
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid not in {0, os.getuid()}
        or stat.S_IMODE(parent.st_mode) & 0o022
    ):
        raise HistoryError("database parent must be a trusted non-writable real directory")
    return path


def _verify_file(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise HistoryError("Phase-1B database is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise HistoryError("Phase-1B database must be a regular non-symlink file")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise HistoryError("Phase-1B database must have mode 0600")
    if info.st_size > MAX_DATABASE_BYTES:
        raise HistoryError("Phase-1B database exceeds its fixed size budget")
    return info


def _secure_create(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
    except OSError as exc:
        raise HistoryError("Phase-1B database cannot be created safely") from exc
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
    else:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
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
    if int(connection.execute("PRAGMA application_id").fetchone()[0]) != SQLITE_APPLICATION_ID:
        raise HistoryError("Phase-1B application id differs")
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) != SCHEMA_VERSION:
        raise HistoryError("Phase-1B schema version differs")
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise HistoryError("Phase-1B database integrity check failed")
    if _master_snapshot(connection) != _expected_snapshot():
        raise HistoryError("Phase-1B database schema differs")
    row = connection.execute(
        "SELECT schema_version,experiment_id,migration_sha256 FROM metadata WHERE singleton=1"
    ).fetchone()
    if row is None or tuple(row) != (SCHEMA_VERSION, EXPERIMENT_ID, _migration_sha256()):
        raise HistoryError("Phase-1B metadata differs")


def _canonical_object(
    value: Mapping[str, Any], name: str
) -> tuple[dict[str, Any], bytes, str]:
    if not isinstance(value, Mapping):
        raise HistoryError(f"{name} must be an object")
    plain = dict(value)
    try:
        payload = canonical_json_bytes(plain)
    except CanonicalError as exc:
        raise HistoryError(f"{name} is not bounded canonical JSON") from exc
    return plain, payload, hashlib.sha256(payload).hexdigest()


def _validated_report(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes, str, str]:
    report, payload, digest = _canonical_object(value, "report")
    required = {
        "schema_version",
        "experiment_id",
        "run_id",
        "kind",
        "status",
        "formal_claim",
        "action",
        "scope",
        "metrics",
    }
    if set(report) != required:
        raise HistoryError("Phase-1B report keys differ")
    if report["schema_version"] != SCHEMA_VERSION or report["experiment_id"] != EXPERIMENT_ID:
        raise HistoryError("Phase-1B report identity differs")
    kind = report["kind"]
    run_id = report["run_id"]
    if kind not in HISTORY_KINDS or not isinstance(run_id, str):
        raise HistoryError("Phase-1B report kind or run id differs")
    if kind == "qualification" and run_id != QUALIFICATION_RUN_ID:
        raise HistoryError("qualification run id differs")
    if kind == "benchmark" and run_id != BENCHMARK_RUN_ID:
        raise HistoryError("benchmark run id differs")
    if kind == "failure" and run_id not in {QUALIFICATION_RUN_ID, BENCHMARK_RUN_ID}:
        raise HistoryError("failure run id differs")
    if not isinstance(report["status"], str) or not 1 <= len(report["status"]) <= 96:
        raise HistoryError("Phase-1B status is invalid")
    if report["formal_claim"] is not False:
        raise HistoryError("Phase-1B records cannot carry formal claims")
    if report["action"] not in {
        "baseline_fallback",
        "qualification_only",
        "candidate_scope_eligible",
    }:
        raise HistoryError("Phase-1B action differs")
    scope = report["scope"]
    expected_scope = {
        "contract_id": CONTRACT_ID,
        "workload_id": WORKLOAD_ID,
        "shape": [ROWS, HIDDEN_SIZE],
        "dtype": "float16",
        "source_sha256": KERNEL_SOURCE_SHA256,
        "kernel_name": KERNEL_NAME,
        "runtime_activation": False,
    }
    if scope != expected_scope or not isinstance(report["metrics"], Mapping):
        raise HistoryError("Phase-1B report scope or metrics differ")
    return report, payload, digest, f"{kind}:{run_id}"


def _validated_provenance(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes, str]:
    provenance, payload, digest = _canonical_object(value, "provenance")
    if set(provenance) != _PROVENANCE_KEYS:
        raise HistoryError("Phase-1B provenance keys differ")
    if (
        provenance["experiment_id"] != EXPERIMENT_ID
        or provenance["contract_id"] != CONTRACT_ID
        or provenance["schema_version"] != SCHEMA_VERSION
        or provenance["git_dirty"] is not False
    ):
        raise HistoryError("Phase-1B provenance identity differs")
    revision = provenance.get("git_revision")
    if not isinstance(revision, str) or len(revision) != 40 or not set(revision) <= _HEX:
        raise HistoryError("Phase-1B Git revision is invalid")
    for key in (
        "git_status_sha256",
        "code_sha256",
        "spec_sha256",
        "source_binding_sha256",
        "environment_sha256",
        "hardware_sha256",
        "provenance_sha256",
    ):
        if not _is_sha256(provenance.get(key)):
            raise HistoryError(f"Phase-1B provenance {key} is invalid")
    for value_key, digest_key in (
        ("code_files", "code_sha256"),
        ("spec_files", "spec_sha256"),
        ("source", "source_binding_sha256"),
        ("environment", "environment_sha256"),
        ("hardware", "hardware_sha256"),
    ):
        component = provenance.get(value_key)
        if not isinstance(component, Mapping) or not component:
            raise HistoryError(f"Phase-1B provenance {value_key} is invalid")
        if value_key in {"code_files", "spec_files"} and not all(
            isinstance(path, str)
            and 1 <= len(path) <= 512
            and "\0" not in path
            and _is_sha256(file_digest)
            for path, file_digest in component.items()
        ):
            raise HistoryError(f"Phase-1B provenance {value_key} is invalid")
        if canonical_sha256(component) != provenance[digest_key]:
            raise HistoryError(f"Phase-1B provenance {digest_key} differs")
    if provenance["source"] != {
        "kernel_name": KERNEL_NAME,
        "source_sha256": KERNEL_SOURCE_SHA256,
    }:
        raise HistoryError("Phase-1B source binding differs")
    expected = dict(provenance)
    claimed = expected.pop("provenance_sha256")
    if canonical_sha256(expected) != claimed:
        raise HistoryError("Phase-1B provenance digest differs")
    return provenance, payload, digest


class History:
    def __init__(
        self,
        path: Path,
        connection: sqlite3.Connection,
        *,
        read_only: bool,
        identity: tuple[int, int],
    ) -> None:
        self.path = path
        self.connection = connection
        self.read_only = read_only
        self.identity = identity

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
                raise HistoryError("Phase-1B database does not exist")
            _secure_create(path)
            created = True
        before = _verify_file(path)
        connection = _connect(path, read_only=read_only)
        try:
            after = _verify_file(path)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise HistoryError("Phase-1B database identity changed while opening")
            if created:
                connection.executescript(_migration_bytes().decode("utf-8", errors="strict"))
                connection.execute(f"PRAGMA application_id={SQLITE_APPLICATION_ID}")
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                connection.execute(
                    "INSERT INTO metadata(singleton,schema_version,experiment_id,migration_sha256) "
                    "VALUES(1,?,?,?)",
                    (SCHEMA_VERSION, EXPERIMENT_ID, _migration_sha256()),
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
        return cls(
            path,
            connection,
            read_only=read_only,
            identity=(after.st_dev, after.st_ino),
        )

    def __enter__(self) -> "History":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _verify_identity(self) -> None:
        info = _verify_file(self.path)
        if (info.st_dev, info.st_ino) != self.identity:
            raise HistoryError("Phase-1B database identity changed after opening")

    @contextmanager
    def read_transaction(self) -> Iterator[None]:
        self._verify_identity()
        _verify_schema(self.connection)
        self.connection.execute("BEGIN")
        try:
            yield
        finally:
            self.connection.execute("ROLLBACK")

    def verified_records(self) -> list[dict[str, Any]]:
        self._verify_identity()
        rows = self.connection.execute("SELECT rowid,* FROM records ORDER BY rowid").fetchall()
        if len(rows) > MAX_HISTORY_ROWS:
            raise HistoryError("Phase-1B history exceeds the row budget")
        previous: str | None = None
        seen_run_ids: set[str] = set()
        verified: list[dict[str, Any]] = []
        for row in rows:
            try:
                report = strict_json_loads(
                    row["report_json"].encode("utf-8"), maximum=MAX_CANONICAL_BYTES
                )
                provenance = strict_json_loads(
                    row["provenance_json"].encode("utf-8"), maximum=MAX_CANONICAL_BYTES
                )
            except CanonicalError as exc:
                raise HistoryError("Phase-1B record JSON is invalid") from exc
            report_value, report_bytes, report_sha, entity_key = _validated_report(report)
            if report_value["run_id"] in seen_run_ids:
                raise HistoryError("Phase-1B history repeats a once-only run id")
            seen_run_ids.add(report_value["run_id"])
            provenance_value, provenance_bytes, provenance_sha = _validated_provenance(provenance)
            created = row["created_at_unix_ns"]
            if isinstance(created, bool) or not isinstance(created, int) or not 0 <= created <= 2**63 - 1:
                raise HistoryError("Phase-1B timestamp is invalid")
            if (
                row["previous_record_id"] != previous
                or row["experiment_id"] != EXPERIMENT_ID
                or row["entity_key"] != entity_key
                or row["record_kind"] != report_value["kind"]
                or row["status"] != report_value["status"]
                or row["report_json"].encode("utf-8") != report_bytes
                or row["report_sha256"] != report_sha
                or row["provenance_json"].encode("utf-8") != provenance_bytes
                or row["provenance_sha256"] != provenance_sha
            ):
                raise HistoryError("Phase-1B record columns differ from canonical content")
            record_id = canonical_sha256(
                {
                    "domain": "friday-phase1b-record-v1",
                    "previous_record_id": previous,
                    "created_at_unix_ns": created,
                    "report_sha256": report_sha,
                    "provenance_sha256": provenance_sha,
                }
            )
            if row["record_id"] != record_id:
                raise HistoryError("Phase-1B record hash chain differs")
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
            raise HistoryError("read-only Phase-1B history cannot persist")
        report_value, report_bytes, report_sha, entity_key = _validated_report(report)
        _, provenance_bytes, provenance_sha = _validated_provenance(provenance)
        created = time.time_ns() if created_at_unix_ns is None else created_at_unix_ns
        if isinstance(created, bool) or not isinstance(created, int) or not 0 <= created <= 2**63 - 1:
            raise HistoryError("Phase-1B timestamp is invalid")
        self._verify_identity()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            _verify_schema(self.connection)
            existing = self.verified_records()
            for row in existing:
                old = row["report"]
                if old["run_id"] == report_value["run_id"]:
                    same = (
                        f"{old['kind']}:{old['run_id']}" == entity_key
                        and
                        canonical_json_bytes(old) == report_bytes
                        and canonical_json_bytes(row["provenance"]) == provenance_bytes
                    )
                    if same:
                        self.connection.execute("COMMIT")
                        return PersistenceOutcome("existing", row["record_id"])
                    raise HistoryConflict("Phase-1B once-only run id is already terminal")
            if len(existing) >= MAX_HISTORY_ROWS:
                raise HistoryError("Phase-1B history row budget is exhausted")
            previous = existing[-1]["record_id"] if existing else None
            record_id = canonical_sha256(
                {
                    "domain": "friday-phase1b-record-v1",
                    "previous_record_id": previous,
                    "created_at_unix_ns": created,
                    "report_sha256": report_sha,
                    "provenance_sha256": provenance_sha,
                }
            )
            self.connection.execute(
                "INSERT INTO records(record_id,previous_record_id,experiment_id,entity_key,"
                "record_kind,status,created_at_unix_ns,report_json,report_sha256,"
                "provenance_json,provenance_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record_id,
                    previous,
                    EXPERIMENT_ID,
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
