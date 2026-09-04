"""Append-only SQLite storage for provenance-bound H1/H2 evidence."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import quote

from .canonical import CanonicalError, canonical_json, canonical_json_bytes, canonical_sha256
from .registry import (
    MAX_PROVENANCE_BYTES,
    MAX_REPORT_BYTES,
    REGISTERED_TOOLS,
    SCHEMA_VERSION,
    SQLITE_APPLICATION_ID,
)

_MIGRATION = Path(__file__).with_name("migrations") / "0001_initial.sql"
_TABLES = {"evidence_metadata", "evidence_records"}
_SCHEMA_OBJECTS = {
    ("table", "evidence_metadata"),
    ("table", "evidence_records"),
    ("index", "idx_evidence_recorded"),
    ("index", "idx_evidence_tool_recorded"),
    ("index", "idx_evidence_status_recorded"),
    ("index", "sqlite_autoindex_evidence_records_1"),
    ("index", "sqlite_autoindex_evidence_records_2"),
    ("trigger", "evidence_records_no_update"),
    ("trigger", "evidence_records_no_delete"),
    ("trigger", "evidence_metadata_no_update"),
    ("trigger", "evidence_metadata_no_delete"),
}
_RECORD_COLUMNS = (
    "record_id", "schema_version", "evidence_kind", "source_key", "tool",
    "workload_key", "result_status", "raw_measurements_available",
    "observed_at_unix_ns", "recorded_at_unix_ns", "report_json", "report_sha256",
    "provenance_json", "provenance_sha256", "git_revision", "git_dirty",
    "code_sha256", "spec_sha256", "environment_sha256", "hardware_key",
)
_REQUIRED_PROVENANCE = {
    "schema_version",
    "tool",
    "workload_key",
    "git_revision",
    "git_dirty",
    "code_sha256",
    "spec_sha256",
    "environment_sha256",
    "hardware_key",
    "provenance_sha256",
}


class StorageError(RuntimeError):
    """Evidence cannot be stored or verified without weakening the contract."""


@dataclass(frozen=True)
class PersistenceOutcome:
    record_id: str
    state: str


def _strict_object(payload: str, name: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise StorageError(f"duplicate JSON key in {name}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=pairs)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StorageError(f"invalid {name} JSON") from exc
    if not isinstance(value, dict):
        raise StorageError(f"{name} must be a JSON object")
    try:
        if canonical_json(value) != payload:
            raise StorageError(f"{name} is not canonical JSON")
    except CanonicalError as exc:
        raise StorageError(f"invalid {name}: {exc}") from exc
    return value


def _digest_text(value: Mapping[str, Any], name: str, maximum: int) -> tuple[str, str]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise StorageError(f"{name} must be an object with text keys")
    try:
        payload = canonical_json_bytes(value)
    except CanonicalError as exc:
        raise StorageError(f"invalid {name}: {exc}") from exc
    if len(payload) > maximum:
        raise StorageError(f"{name} exceeds its byte limit")
    return payload.decode("utf-8"), hashlib.sha256(payload).hexdigest()


def _checked_path(path: str | Path, *, create_parent: bool) -> Path:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    if create_parent:
        try:
            candidate.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError("cannot create database parent") from exc
    try:
        parent_status = candidate.parent.lstat()
    except OSError as exc:
        raise StorageError("database parent is unavailable") from exc
    if stat.S_ISLNK(parent_status.st_mode) or not stat.S_ISDIR(parent_status.st_mode):
        raise StorageError("database parent must be a real directory")
    if parent_status.st_uid != os.geteuid() or parent_status.st_mode & 0o077:
        raise StorageError("database parent ownership or permissions are unsafe")
    resolved = candidate.parent.resolve(strict=True) / candidate.name
    if resolved.exists():
        try:
            file_status = resolved.lstat()
        except OSError as exc:
            raise StorageError("database path is unavailable") from exc
        if stat.S_ISLNK(file_status.st_mode) or not stat.S_ISREG(file_status.st_mode):
            raise StorageError("database path must be a regular file")
        if file_status.st_uid != os.geteuid() or stat.S_IMODE(file_status.st_mode) != 0o600:
            raise StorageError("database ownership or permissions are unsafe")
    return resolved


def _secure_create(path: Path) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise StorageError("cannot securely create evidence database") from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_uid != os.geteuid():
            raise StorageError("created evidence database is not a private regular file")
    finally:
        os.close(descriptor)


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
        raise StorageError("required SQLite defensive controls are unavailable")
    try:
        connection.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
        connection.setconfig(sqlite3.SQLITE_DBCONFIG_TRUSTED_SCHEMA, False)
        connection.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DDL, False)
        connection.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
        connection.setconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, False)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA busy_timeout=5000")
        if read_only:
            connection.execute("PRAGMA query_only=ON")
        expected = (
            (sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True),
            (sqlite3.SQLITE_DBCONFIG_TRUSTED_SCHEMA, False),
            (sqlite3.SQLITE_DBCONFIG_DQS_DDL, False),
            (sqlite3.SQLITE_DBCONFIG_DQS_DML, False),
            (sqlite3.SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, False),
        )
        if any(connection.getconfig(option) != value for option, value in expected):
            raise StorageError("SQLite defensive controls did not take effect")
        if connection.execute("PRAGMA trusted_schema").fetchone()[0] != 0:
            raise StorageError("SQLite trusted schema remained enabled")
        if read_only and connection.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise StorageError("SQLite query-only defense did not take effect")
    except (sqlite3.Error, ValueError) as exc:
        raise StorageError("cannot enable required SQLite defensive controls") from exc


class EvidenceStorage:
    """Verified storage handle; read-only handles cannot mutate by construction."""

    def __init__(self, path: Path, connection: sqlite3.Connection, *, read_only: bool) -> None:
        self.path = path
        self._connection = connection
        self.read_only = read_only

    @classmethod
    def open(
        cls, path: str | Path, *, read_only: bool = False, initialize: bool = False
    ) -> "EvidenceStorage":
        if read_only and initialize:
            raise StorageError("read-only storage cannot initialize a database")
        checked = _checked_path(path, create_parent=initialize)
        existed = checked.exists()
        created = False
        if not existed and initialize:
            _secure_create(checked)
            created = True
        try:
            if read_only:
                if not existed:
                    raise StorageError("evidence database does not exist")
                uri = f"file:{quote(str(checked), safe='/')}?mode=ro&immutable=0"
                connection = sqlite3.connect(uri, uri=True, timeout=5.0)
            else:
                if not existed and not initialize:
                    raise StorageError("evidence database does not exist")
                connection = sqlite3.connect(str(checked), timeout=5.0)
        except sqlite3.Error as exc:
            if created:
                try:
                    checked.unlink(missing_ok=True)
                except OSError:
                    pass
            raise StorageError("cannot open evidence database") from exc
        try:
            _configure(connection, read_only=read_only)
            storage = cls(checked, connection, read_only=read_only)
            if created:
                storage._initialize()
            storage.verify_schema()
            return storage
        except Exception:
            connection.close()
            if created:
                try:
                    checked.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def __enter__(self) -> "EvidenceStorage":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _initialize(self) -> None:
        migration = _MIGRATION.read_bytes()
        migration_sha = hashlib.sha256(migration).hexdigest()
        try:
            self._connection.executescript(
                "BEGIN EXCLUSIVE;\n" + migration.decode("utf-8")
            )
            self._connection.execute(f"PRAGMA application_id={SQLITE_APPLICATION_ID}")
            self._connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._connection.execute(
                "INSERT INTO evidence_metadata(singleton,schema_version,migration_sha256) "
                "VALUES(1,?,?)",
                (SCHEMA_VERSION, migration_sha),
            )
            self._connection.commit()
        except (OSError, UnicodeError, sqlite3.Error) as exc:
            self._connection.rollback()
            raise StorageError("cannot initialize evidence schema") from exc

    def verify_schema(self) -> None:
        try:
            application_id = self._connection.execute("PRAGMA application_id").fetchone()[0]
            user_version = self._connection.execute("PRAGMA user_version").fetchone()[0]
            integrity = self._connection.execute("PRAGMA integrity_check(1)").fetchone()[0]
            tables = {
                row[0]
                for row in self._connection.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            metadata = self._connection.execute(
                "SELECT schema_version,migration_sha256 FROM evidence_metadata WHERE singleton=1"
            ).fetchall()
            objects = {
                (row[0], row[1])
                for row in self._connection.execute(
                    "SELECT type,name FROM sqlite_schema "
                    "WHERE name NOT LIKE 'sqlite_%' OR name LIKE 'sqlite_autoindex_evidence_records_%'"
                )
            }
            record_columns = tuple(
                row[1] for row in self._connection.execute("PRAGMA table_info(evidence_records)")
            )
        except sqlite3.Error as exc:
            raise StorageError("cannot verify evidence schema") from exc
        expected_migration = hashlib.sha256(_MIGRATION.read_bytes()).hexdigest()
        if (
            application_id != SQLITE_APPLICATION_ID
            or user_version != SCHEMA_VERSION
            or integrity != "ok"
            or tables != _TABLES
            or objects != _SCHEMA_OBJECTS
            or record_columns != _RECORD_COLUMNS
            or len(metadata) != 1
            or metadata[0][0] != SCHEMA_VERSION
            or metadata[0][1] != expected_migration
        ):
            raise StorageError("evidence database does not match registered schema v1")

    @contextmanager
    def read_transaction(self) -> Iterator[None]:
        try:
            self._connection.execute("BEGIN")
            yield
            self._connection.rollback()
        except sqlite3.Error as exc:
            self._connection.rollback()
            raise StorageError("evidence read transaction failed") from exc

    @staticmethod
    def _validated_provenance(
        provenance: Mapping[str, Any], tool: str, evidence_kind: str
    ) -> tuple[str, str, dict[str, Any]]:
        text, digest = _digest_text(provenance, "provenance", MAX_PROVENANCE_BYTES)
        copied = _strict_object(text, "provenance")
        if not _REQUIRED_PROVENANCE.issubset(copied):
            raise StorageError("provenance is incomplete")
        if copied["schema_version"] != SCHEMA_VERSION:
            raise StorageError("provenance schema version is inconsistent")
        if copied["tool"] != tool or copied["workload_key"] != REGISTERED_TOOLS[tool]:
            raise StorageError("provenance tool binding is inconsistent")
        if copied.get("provenance_kind") != evidence_kind:
            raise StorageError("provenance evidence kind is inconsistent")
        embedded = copied.pop("provenance_sha256")
        if embedded != canonical_sha256(copied):
            raise StorageError("provenance digest is invalid")
        copied["provenance_sha256"] = embedded
        for field in (
            "code_sha256", "spec_sha256", "environment_sha256", "hardware_key"
        ):
            if (
                not isinstance(copied[field], str)
                or len(copied[field]) != 64
                or any(character not in "0123456789abcdef" for character in copied[field])
            ):
                raise StorageError(f"invalid provenance field: {field}")
        revision = copied["git_revision"]
        if (
            not isinstance(revision, str)
            or not 40 <= len(revision) <= 64
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise StorageError("invalid Git revision in provenance")
        if type(copied["git_dirty"]) is not bool:
            raise StorageError("invalid Git dirty flag in provenance")
        if evidence_kind == "native":
            projection_fields = {
                "code_sha256": "code_files",
                "spec_sha256": "spec_files",
                "environment_sha256": "environment",
                "hardware_key": "hardware",
            }
            if copied["git_dirty"]:
                raise StorageError("native provenance must bind a clean Git worktree")
            if set(revision) == {"0"}:
                raise StorageError("native provenance cannot use a missing Git revision")
            if copied.get("git_diff_sha256") != hashlib.sha256(b"").hexdigest():
                raise StorageError("native provenance contains a non-empty Git diff")
        else:
            projection_fields = {
                "code_sha256": "code",
                "spec_sha256": "spec",
                "environment_sha256": "environment",
                "hardware_key": "hardware",
            }
        for digest_field, value_field in projection_fields.items():
            value = copied.get(value_field)
            if not isinstance(value, Mapping) or canonical_sha256(value) != copied[digest_field]:
                raise StorageError(f"provenance projection mismatch: {digest_field}")
        return text, digest, copied

    @staticmethod
    def _record_body(
        *,
        evidence_kind: str,
        source_key: str,
        tool: str,
        result_status: str,
        raw_measurements_available: bool,
        observed_at_unix_ns: int | None,
        recorded_at_unix_ns: int,
        report_sha256: str,
        provenance_sha256: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "evidence_kind": evidence_kind,
            "source_key": source_key,
            "tool": tool,
            "workload_key": REGISTERED_TOOLS[tool],
            "result_status": result_status,
            "raw_measurements_available": raw_measurements_available,
            "observed_at_unix_ns": observed_at_unix_ns,
            "recorded_at_unix_ns": recorded_at_unix_ns,
            "report_sha256": report_sha256,
            "provenance_sha256": provenance_sha256,
        }

    def persist(
        self,
        *,
        evidence_kind: str,
        source_key: str,
        tool: str,
        report: Mapping[str, Any],
        provenance: Mapping[str, Any],
        result_status: str,
        raw_measurements_available: bool,
        observed_at_unix_ns: int | None,
        recorded_at_unix_ns: int | None = None,
    ) -> PersistenceOutcome:
        if self.read_only:
            raise StorageError("read-only evidence storage cannot persist")
        if evidence_kind not in {"native", "legacy_summary"}:
            raise StorageError("unregistered evidence kind")
        if tool not in REGISTERED_TOOLS:
            raise StorageError("unregistered evidence tool")
        if not isinstance(source_key, str) or not 1 <= len(source_key) <= 200:
            raise StorageError("invalid evidence source key")
        if not isinstance(result_status, str) or not 1 <= len(result_status) <= 128:
            raise StorageError("invalid result status")
        if type(raw_measurements_available) is not bool:
            raise StorageError("raw measurement availability must be boolean")
        if report.get("formal_claim") is not False:
            raise StorageError("schema v1 evidence cannot claim formal H1/H2 status")
        if evidence_kind == "legacy_summary" and raw_measurements_available:
            raise StorageError("legacy summaries cannot claim raw measurements")
        if (
            evidence_kind == "native"
            and result_status != "measurement_failed"
            and not raw_measurements_available
        ):
            raise StorageError("completed native evidence must contain raw measurements")
        if observed_at_unix_ns is not None and (
            type(observed_at_unix_ns) is not int or observed_at_unix_ns < 0
        ):
            raise StorageError("invalid observation timestamp")
        recorded = time.time_ns() if recorded_at_unix_ns is None else recorded_at_unix_ns
        if type(recorded) is not int or recorded < 0:
            raise StorageError("invalid recording timestamp")

        report_json, report_sha = _digest_text(report, "report", MAX_REPORT_BYTES)
        provenance_json, provenance_sha, checked_provenance = self._validated_provenance(
            provenance, tool, evidence_kind
        )
        body = self._record_body(
            evidence_kind=evidence_kind,
            source_key=source_key,
            tool=tool,
            result_status=result_status,
            raw_measurements_available=raw_measurements_available,
            observed_at_unix_ns=observed_at_unix_ns,
            recorded_at_unix_ns=recorded,
            report_sha256=report_sha,
            provenance_sha256=provenance_sha,
        )
        record_id = canonical_sha256(body)
        values = (
            record_id,
            SCHEMA_VERSION,
            evidence_kind,
            source_key,
            tool,
            REGISTERED_TOOLS[tool],
            result_status,
            int(raw_measurements_available),
            observed_at_unix_ns,
            recorded,
            report_json,
            report_sha,
            provenance_json,
            provenance_sha,
            checked_provenance["git_revision"],
            int(bool(checked_provenance["git_dirty"])),
            checked_provenance["code_sha256"],
            checked_provenance["spec_sha256"],
            checked_provenance["environment_sha256"],
            checked_provenance["hardware_key"],
        )
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(
                "INSERT INTO evidence_records(" + ",".join(_RECORD_COLUMNS) + ") "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                values,
            )
            self._connection.commit()
            return PersistenceOutcome(record_id=record_id, state="inserted")
        except sqlite3.IntegrityError:
            self._connection.rollback()
            existing = self._connection.execute(
                "SELECT record_id,result_status,raw_measurements_available,"
                "observed_at_unix_ns,report_sha256,provenance_sha256,workload_key "
                "FROM evidence_records "
                "WHERE evidence_kind=? AND tool=? AND source_key=?",
                (evidence_kind, tool, source_key),
            ).fetchone()
            if existing and (
                existing[1] == result_status
                and bool(existing[2]) == raw_measurements_available
                and existing[3] == observed_at_unix_ns
                and existing[4] == report_sha
                and existing[5] == provenance_sha
                and existing[6] == REGISTERED_TOOLS[tool]
            ):
                return PersistenceOutcome(record_id=existing[0], state="already_present")
            raise StorageError("evidence source key is already bound to different bytes")
        except sqlite3.Error as exc:
            self._connection.rollback()
            raise StorageError("evidence persistence failed") from exc

    def verified_rows(self) -> list[dict[str, Any]]:
        try:
            rows = self._connection.execute(
                "SELECT * FROM evidence_records ORDER BY recorded_at_unix_ns DESC, record_id"
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError("cannot read evidence records") from exc
        verified: list[dict[str, Any]] = []
        for row in rows:
            report = _strict_object(row["report_json"], "report")
            provenance = _strict_object(row["provenance_json"], "provenance")
            if canonical_sha256(report) != row["report_sha256"]:
                raise StorageError("stored report digest mismatch")
            _, provenance_sha, checked = self._validated_provenance(
                provenance, row["tool"], row["evidence_kind"]
            )
            if provenance_sha != row["provenance_sha256"]:
                raise StorageError("stored provenance digest mismatch")
            for field in (
                "git_revision", "code_sha256", "spec_sha256", "environment_sha256", "hardware_key"
            ):
                if row[field] != checked[field]:
                    raise StorageError(f"stored provenance projection mismatch: {field}")
            if bool(row["git_dirty"]) != checked["git_dirty"]:
                raise StorageError("stored provenance projection mismatch: git_dirty")
            if row["workload_key"] != REGISTERED_TOOLS[row["tool"]]:
                raise StorageError("stored workload key is inconsistent")
            body = self._record_body(
                evidence_kind=row["evidence_kind"],
                source_key=row["source_key"],
                tool=row["tool"],
                result_status=row["result_status"],
                raw_measurements_available=bool(row["raw_measurements_available"]),
                observed_at_unix_ns=row["observed_at_unix_ns"],
                recorded_at_unix_ns=row["recorded_at_unix_ns"],
                report_sha256=row["report_sha256"],
                provenance_sha256=row["provenance_sha256"],
            )
            if canonical_sha256(body) != row["record_id"]:
                raise StorageError("stored record identity mismatch")
            item = dict(row)
            item["report"] = report
            item["provenance"] = provenance
            verified.append(item)
        return verified

    def get_verified(self, record_id: str) -> dict[str, Any] | None:
        if (
            not isinstance(record_id, str)
            or len(record_id) != 64
            or any(character not in "0123456789abcdef" for character in record_id)
        ):
            raise StorageError("invalid evidence record id")
        return next((row for row in self.verified_rows() if row["record_id"] == record_id), None)


__all__ = ["EvidenceStorage", "PersistenceOutcome", "StorageError"]
