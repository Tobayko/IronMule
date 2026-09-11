"""Transactional, append-only SQLite v1 storage for H0 evidence."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from .canonical import canonical_json, canonical_json_bytes, canonical_sha256
from .constants import SQLITE_APPLICATION_ID, STORAGE_SCHEMA_VERSION
from .manifest import manifest_hash, validate_manifest
from .protocol import PRODUCTION_JSON_DEPTH, ClosedManifest, validate_result


class StorageError(RuntimeError):
    """Raised for storage identity, mode, or append-only contract failures."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

PERSISTENCE_SCHEMA_VERSION = 1
PERSISTENCE_MAX_RESULT_BYTES = 1 * 1024 * 1024
PERSISTENCE_MAX_CHILD_BYTES = 64 * 1024
PERSISTENCE_MAX_EVENT_BYTES = 64 * 1024
PERSISTENCE_MAX_BUNDLE_BYTES = 4 * 1024 * 1024
PERSISTENCE_MAX_CHILD_ROWS = 10_000
PERSISTENCE_JSON_DEPTH = PRODUCTION_JSON_DEPTH
_SQLITE_INT_MAX = (1 << 63) - 1

_REQUIRED_TABLE_COLUMNS = {
    "schema_migrations": {"version", "name"},
    "db_identity": {"key", "value"},
    "runs": {
        "run_id", "phase", "mode", "manifest_json", "manifest_hash",
        "code_sha256", "spec_sha256", "environment_sha256", "revision",
        "revision_missing_reason", "created_at_unix_ns",
    },
    "status_events": {
        "event_id", "run_id", "event_kind", "status", "payload_json",
        "payload_hash", "recorded_at_ns",
    },
    "raw_samples": {
        "sample_id", "run_id", "session_id", "sample_kind", "sample_index",
        "block_index", "arm", "value", "unit", "observed_at_ns",
    },
    "scalar_metrics": {
        "metric_id", "run_id", "metric_name", "scope", "value",
        "missing_reason", "unit", "recorded_at_ns",
    },
    "correctness_metrics": {
        "metric_id", "run_id", "case_name", "metric_name", "value", "unit",
        "passed", "detail_json",
    },
    "artifacts": {
        "artifact_id", "run_id", "artifact_name", "artifact_kind", "sha256",
        "metadata_json",
    },
}
_REQUIRED_PRIMARY_KEYS = {
    "schema_migrations": "version",
    "db_identity": "key",
    "runs": "run_id",
    "status_events": "event_id",
    "raw_samples": "sample_id",
    "scalar_metrics": "metric_id",
    "correctness_metrics": "metric_id",
    "artifacts": "artifact_id",
}
_REQUIRED_APPEND_ONLY_TRIGGERS = {
    **{
        f"{table}_append_only_{operation.lower()}": (table, operation)
        for table in _REQUIRED_TABLE_COLUMNS
        for operation in ("UPDATE", "DELETE")
    },
}


def _normalized_schema_sql(sql: Any) -> str:
    """Normalize only insignificant whitespace in one SQLite DDL string."""

    if not isinstance(sql, str) or not sql.strip():
        raise StorageError("schema object has no SQL definition")
    return " ".join(sql.split())


@dataclass(frozen=True)
class PersistenceOutcome:
    """Outcome of one complete Common-Result persistence attempt.

    ``state`` is intentionally closed: ``inserted`` means this call committed
    the bundle, while ``idempotent`` means an identical complete bundle was
    already committed. Artifact paths are never accepted or opened here;
    artifact hashes and metadata are declarative evidence only.
    """

    state: Literal["inserted", "idempotent"]
    run_id: str
    bundle_sha256: str


class Storage:
    """One SQLite v1 connection; writes are inserts only on evidence tables."""

    IDENTITY = "friday_h0.sqlite.v1"

    def __init__(self, connection: sqlite3.Connection, *, read_only: bool = False) -> None:
        self._connection = connection
        self._connection.row_factory = sqlite3.Row
        self._read_only = read_only
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=250")
        if read_only:
            self._connection.execute("PRAGMA query_only=ON")

    @classmethod
    def open(cls, path: str | Path, *, read_only: bool = False) -> "Storage":
        """Open a database, migrate writable instances idempotently, or use SQLite mode=ro."""

        database = Path(path)
        if "\x00" in str(database):
            raise StorageError("database path contains NUL")
        if read_only:
            uri = f"file:{quote(str(database.resolve()), safe='/')}?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
            try:
                storage = cls(connection, read_only=True)
                storage._verify_identity()
                return storage
            except Exception:
                connection.close()
                raise
        database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(database))
        try:
            storage = cls(connection)
            storage._migrate()
            storage._verify_identity()
            return storage
        except Exception:
            connection.close()
            raise

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the connection for read-only dashboard queries and integrity checks."""

        return self._connection

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _require_writable(self) -> None:
        if self._read_only:
            raise StorageError("database opened in mode=ro")

    def _migrate(self) -> None:
        self._require_writable()
        migration_path = Path(__file__).with_name("migrations") / "0001_initial.sql"
        try:
            sql = migration_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StorageError(f"cannot read migration: {exc}") from exc
        try:
            user_version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            if user_version > STORAGE_SCHEMA_VERSION:
                raise StorageError(f"unsupported schema version {user_version}")
            if user_version == STORAGE_SCHEMA_VERSION:
                self._verify_identity()
                return
            existing_objects = self._connection.execute(
                """SELECT 1 FROM sqlite_master
                   WHERE type IN ('table', 'index', 'trigger')
                     AND name NOT LIKE 'sqlite_%' LIMIT 1"""
            ).fetchone()
            if existing_objects is not None:
                raise StorageError("refusing silent upgrade of an unversioned non-empty database")
            self._connection.executescript(sql)
            self._connection.commit()
        except sqlite3.Error as exc:
            self._connection.rollback()
            raise StorageError(f"migration failed: {exc}") from exc

    def _verify_identity(self) -> None:
        try:
            application_id = int(self._connection.execute("PRAGMA application_id").fetchone()[0])
            user_version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            row = self._connection.execute(
                "SELECT value FROM db_identity WHERE key='identity'"
            ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"not a friday_h0 database: {exc}") from exc
        if application_id != SQLITE_APPLICATION_ID:
            raise StorageError("database application_id mismatch")
        if user_version != STORAGE_SCHEMA_VERSION:
            raise StorageError("database user_version mismatch")
        if row is None or row[0] != self.IDENTITY:
            raise StorageError("database identity mismatch")
        self._verify_schema_integrity()

    @staticmethod
    def _schema_snapshot(connection: sqlite3.Connection) -> dict[str, Any]:
        """Capture the complete user schema and SQLite index contracts."""

        objects: set[tuple[str, str, str, str]] = set()
        for row in connection.execute(
            """SELECT type, name, tbl_name, sql FROM sqlite_master
               WHERE type IN ('table', 'index', 'trigger', 'view')
                 AND name NOT LIKE 'sqlite_%'"""
        ):
            object_type, name, table_name, sql = tuple(row)
            objects.add(
                (
                    str(object_type),
                    str(name),
                    str(table_name),
                    _normalized_schema_sql(sql),
                )
            )

        tables = {
            table_name: tuple(
                tuple(row)
                for row in connection.execute(
                    f'PRAGMA table_xinfo("{table_name.replace(chr(34), chr(34) * 2)}")'
                )
            )
            for table_name in sorted(
                name
                for object_type, name, _table_name, _sql in objects
                if object_type == "table"
            )
        }
        indexes: dict[str, tuple[tuple[Any, ...], ...]] = {}
        index_columns: dict[tuple[str, str], tuple[tuple[Any, ...], ...]] = {}
        for table_name in tables:
            listed = []
            for row in connection.execute(
                f'PRAGMA index_list("{table_name.replace(chr(34), chr(34) * 2)}")'
            ):
                values = tuple(row)
                # seq is an implementation ordering, while name, uniqueness,
                # origin, and partial are the durable index contract.
                index_name = str(values[1])
                listed.append((index_name, int(values[2]), str(values[3]), int(values[4])))
                index_columns[(table_name, index_name)] = tuple(
                    tuple(index_row)
                    for index_row in connection.execute(
                        f'PRAGMA index_xinfo("{index_name.replace(chr(34), chr(34) * 2)}")'
                    )
                )
            indexes[table_name] = tuple(sorted(listed))
        return {
            "objects": frozenset(objects),
            "tables": tables,
            "indexes": indexes,
            "index_columns": index_columns,
        }

    @classmethod
    def _expected_schema_snapshot(cls) -> dict[str, Any]:
        """Materialize the checked-in v1 DDL in memory as the schema authority."""

        migration_path = Path(__file__).with_name("migrations") / "0001_initial.sql"
        try:
            sql = migration_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StorageError(f"cannot read migration: {exc}") from exc
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(sql)
            return cls._schema_snapshot(connection)
        except sqlite3.Error as exc:
            raise StorageError(f"cannot materialize v1 schema authority: {exc}") from exc
        finally:
            connection.close()

    def _verify_schema_integrity(self) -> None:
        """Fail closed unless the database exactly matches the immutable v1 DDL."""

        try:
            expected = self._expected_schema_snapshot()
            actual = self._schema_snapshot(self._connection)
            if actual["objects"] != expected["objects"]:
                missing = sorted(expected["objects"] - actual["objects"])
                extra = sorted(actual["objects"] - expected["objects"])
                raise StorageError(
                    f"database schema objects differ from v1 DDL; missing={missing[:8]} extra={extra[:8]}"
                )
            if actual["tables"] != expected["tables"]:
                raise StorageError("database table_xinfo contract differs from v1 DDL")
            if actual["indexes"] != expected["indexes"]:
                raise StorageError("database index_list contract differs from v1 DDL")
            if actual["index_columns"] != expected["index_columns"]:
                raise StorageError("database index_xinfo contract differs from v1 DDL")
            migration = self._connection.execute(
                "SELECT name FROM schema_migrations WHERE version=?", (STORAGE_SCHEMA_VERSION,)
            ).fetchone()
            if migration is None or migration["name"] != "0001_initial":
                raise StorageError("schema migration 0001_initial is missing or altered")
        except StorageError:
            raise
        except sqlite3.Error as exc:
            raise StorageError(f"database schema integrity check failed: {exc}") from exc

    @staticmethod
    def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StorageError(f"{name} must be numeric and not bool")
        try:
            result = float(value)
        except (OverflowError, ValueError, TypeError) as exc:
            raise StorageError(f"{name} must be finite numeric data") from exc
        if not math.isfinite(result) or (nonnegative and result < 0):
            raise StorageError(f"{name} must be finite{ ' and nonnegative' if nonnegative else ''}")
        return result

    @staticmethod
    def _text(value: Any, name: str, *, maximum: int = 128) -> str:
        if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
            raise StorageError(f"{name} must be a bounded non-empty string")
        return value

    @staticmethod
    def _integer(value: Any, name: str, *, nonnegative: bool = True) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise StorageError(f"{name} must be an integer and not bool")
        if nonnegative and value < 0:
            raise StorageError(f"{name} must be nonnegative")
        if value < -_SQLITE_INT_MAX - 1 or value > _SQLITE_INT_MAX:
            raise StorageError(f"{name} is outside SQLite INTEGER range")
        return value

    @staticmethod
    def _bounded_json(value: Any, name: str, *, limit: int) -> str:
        stack: list[tuple[Any, int, bool]] = [(value, 0, False)]
        active: set[int] = set()
        try:
            while stack:
                node, depth, leaving = stack.pop()
                if leaving:
                    active.discard(id(node))
                    continue
                if depth > PERSISTENCE_JSON_DEPTH:
                    raise StorageError(f"{name} JSON depth exceeds {PERSISTENCE_JSON_DEPTH}")
                if isinstance(node, Mapping):
                    children = list(node.values())
                elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
                    children = list(node)
                else:
                    continue
                node_id = id(node)
                if node_id in active:
                    raise StorageError(f"{name} contains a JSON cycle")
                active.add(node_id)
                stack.append((node, depth, True))
                stack.extend((child, depth + 1, False) for child in reversed(children))
            encoded = canonical_json_bytes(value)
        except StorageError:
            raise
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise StorageError(f"{name} is not finite canonical JSON: {exc}") from exc
        if len(encoded) > limit:
            raise StorageError(f"{name} exceeds {limit} bytes")
        return encoded.decode("utf-8")

    @staticmethod
    def _sequence(value: Any, name: str) -> list[Mapping[str, Any]]:
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
            raise StorageError(f"{name} must be a list or tuple of objects")
        if len(value) > PERSISTENCE_MAX_CHILD_ROWS:
            raise StorageError(f"{name} exceeds {PERSISTENCE_MAX_CHILD_ROWS} rows")
        rows: list[Mapping[str, Any]] = []
        for index, row in enumerate(value):
            if not isinstance(row, Mapping):
                raise StorageError(f"{name}[{index}] must be an object")
            rows.append(row)
        return rows

    @staticmethod
    def _optional_timestamp(value: Any, name: str) -> int | None:
        if value is None:
            return None
        return Storage._integer(value, name)

    @classmethod
    def _normalize_raw_samples(cls, value: Any) -> list[dict[str, Any]]:
        rows = cls._sequence(value, "raw_samples")
        allowed = {
            "session_id", "sample_kind", "sample_index", "block_index", "arm",
            "value", "unit", "observed_at_ns",
        }
        normalized: list[dict[str, Any]] = []
        identities: set[tuple[str, int]] = set()
        for index, row in enumerate(rows):
            required = {"session_id", "sample_kind", "value", "unit"}
            if set(row) - allowed or not required.issubset(row):
                raise StorageError(f"raw_samples[{index}] has unknown or missing fields")
            sample_kind = cls._text(row["sample_kind"], f"raw_samples[{index}].sample_kind")
            sample_index = cls._integer(row.get("sample_index", 0), f"raw_samples[{index}].sample_index")
            identity = (sample_kind, sample_index)
            if identity in identities:
                raise StorageError("raw_samples contains duplicate (sample_kind, sample_index)")
            identities.add(identity)
            normalized.append({
                "session_id": cls._text(row["session_id"], f"raw_samples[{index}].session_id"),
                "sample_kind": sample_kind,
                "sample_index": sample_index,
                "block_index": cls._integer(row.get("block_index", 0), f"raw_samples[{index}].block_index"),
                "arm": cls._text(row.get("arm", "unknown"), f"raw_samples[{index}].arm"),
                "value": cls._finite(row["value"], f"raw_samples[{index}].value", nonnegative=True),
                "unit": cls._text(row["unit"], f"raw_samples[{index}].unit"),
                "observed_at_ns": cls._optional_timestamp(row.get("observed_at_ns"), f"raw_samples[{index}].observed_at_ns"),
            })
        normalized.sort(key=lambda item: (item["sample_kind"], item["sample_index"]))
        return normalized

    @classmethod
    def _normalize_scalar_metrics(cls, value: Any) -> list[dict[str, Any]]:
        rows = cls._sequence(value, "scalar_metrics")
        allowed = {"metric_name", "value", "unit", "scope", "missing_reason", "recorded_at_ns"}
        normalized: list[dict[str, Any]] = []
        identities: set[str] = set()
        for index, row in enumerate(rows):
            if set(row) - allowed or "metric_name" not in row or "value" not in row or "unit" not in row:
                raise StorageError(f"scalar_metrics[{index}] has unknown or missing fields")
            name = cls._text(row["metric_name"], f"scalar_metrics[{index}].metric_name")
            if name in identities:
                raise StorageError("scalar_metrics contains duplicate metric_name")
            identities.add(name)
            metric_value = row["value"]
            missing_reason = row.get("missing_reason")
            if (metric_value is None) == (missing_reason is None):
                raise StorageError("scalar metric requires exactly one of value or missing_reason")
            if metric_value is not None:
                metric_value = cls._finite(metric_value, f"scalar_metrics[{index}].value")
            else:
                missing_reason = cls._text(missing_reason, f"scalar_metrics[{index}].missing_reason", maximum=256)
            normalized.append({
                "metric_name": name,
                "value": metric_value,
                "missing_reason": missing_reason,
                "unit": cls._text(row["unit"], f"scalar_metrics[{index}].unit"),
                "scope": cls._text(row.get("scope", "run"), f"scalar_metrics[{index}].scope"),
                "recorded_at_ns": cls._optional_timestamp(row.get("recorded_at_ns"), f"scalar_metrics[{index}].recorded_at_ns"),
            })
        normalized.sort(key=lambda item: item["metric_name"])
        return normalized

    @classmethod
    def _normalize_correctness_metrics(cls, value: Any) -> list[dict[str, Any]]:
        rows = cls._sequence(value, "correctness_metrics")
        allowed = {"case_name", "metric_name", "value", "unit", "passed", "detail"}
        normalized: list[dict[str, Any]] = []
        identities: set[tuple[str, str]] = set()
        for index, row in enumerate(rows):
            required = {"case_name", "metric_name", "value", "unit", "passed"}
            if set(row) - allowed or not required.issubset(row):
                raise StorageError(f"correctness_metrics[{index}] has unknown or missing fields")
            case_name = cls._text(row["case_name"], f"correctness_metrics[{index}].case_name")
            metric_name = cls._text(row["metric_name"], f"correctness_metrics[{index}].metric_name")
            identity = (case_name, metric_name)
            if identity in identities:
                raise StorageError("correctness_metrics contains duplicate case/metric identity")
            identities.add(identity)
            if not isinstance(row["passed"], bool):
                raise StorageError("correctness_metrics.passed must be bool")
            detail = row.get("detail", {})
            if not isinstance(detail, Mapping):
                raise StorageError("correctness metric detail must be an object")
            detail_json = cls._bounded_json(detail, f"correctness_metrics[{index}].detail", limit=PERSISTENCE_MAX_CHILD_BYTES)
            normalized.append({
                "case_name": case_name,
                "metric_name": metric_name,
                "value": cls._finite(row["value"], f"correctness_metrics[{index}].value"),
                "unit": cls._text(row["unit"], f"correctness_metrics[{index}].unit"),
                "passed": row["passed"],
                "detail": json.loads(detail_json),
            })
        normalized.sort(key=lambda item: (item["case_name"], item["metric_name"]))
        return normalized

    @classmethod
    def _normalize_artifacts(cls, value: Any) -> list[dict[str, Any]]:
        rows = cls._sequence(value, "artifacts")
        allowed = {"artifact_name", "artifact_kind", "sha256", "metadata"}
        normalized: list[dict[str, Any]] = []
        identities: set[tuple[str, str]] = set()
        for index, row in enumerate(rows):
            if set(row) != allowed:
                raise StorageError(f"artifacts[{index}] has unknown or missing fields")
            name = cls._text(row["artifact_name"], f"artifacts[{index}].artifact_name")
            kind = cls._text(row["artifact_kind"], f"artifacts[{index}].artifact_kind")
            identity = (name, kind)
            if identity in identities:
                raise StorageError("artifacts contains duplicate artifact identity")
            identities.add(identity)
            sha256 = row["sha256"]
            if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
                raise StorageError("artifact sha256 must be 64 lowercase hexadecimal characters")
            metadata = row["metadata"]
            if not isinstance(metadata, Mapping):
                raise StorageError("artifact metadata must be an object")
            metadata_json = cls._bounded_json(metadata, f"artifacts[{index}].metadata", limit=PERSISTENCE_MAX_CHILD_BYTES)
            normalized.append({
                "artifact_name": name,
                "artifact_kind": kind,
                "sha256": sha256,
                "metadata": json.loads(metadata_json),
            })
        normalized.sort(key=lambda item: (item["artifact_name"], item["artifact_kind"]))
        return normalized

    @classmethod
    def _bundle_from_rows(
        cls,
        *,
        manifest: ClosedManifest,
        result: Mapping[str, Any],
        raw_samples: Any,
        scalar_metrics: Any,
        correctness_metrics: Any,
        artifacts: Any,
    ) -> tuple[dict[str, Any], str, str]:
        validated_result = validate_result(result, manifest=manifest)
        result_json = cls._bounded_json(validated_result, "result", limit=PERSISTENCE_MAX_RESULT_BYTES)
        result_value = json.loads(result_json)
        result_sha256 = canonical_sha256(result_value)
        bundle = {
            "schema_version": PERSISTENCE_SCHEMA_VERSION,
            "run_id": manifest.run_id,
            "manifest_sha256": manifest.sha256,
            "result_sha256": result_sha256,
            "result": result_value,
            "raw_samples": cls._normalize_raw_samples(raw_samples),
            "scalar_metrics": cls._normalize_scalar_metrics(scalar_metrics),
            "correctness_metrics": cls._normalize_correctness_metrics(correctness_metrics),
            "artifacts": cls._normalize_artifacts(artifacts),
        }
        bundle_bytes = canonical_json_bytes(bundle)
        if len(bundle_bytes) > PERSISTENCE_MAX_BUNDLE_BYTES:
            raise StorageError(f"persistence bundle exceeds {PERSISTENCE_MAX_BUNDLE_BYTES} bytes")
        return bundle, result_sha256, canonical_sha256(bundle)

    @staticmethod
    def _payload_for_bundle(bundle: Mapping[str, Any], result_sha256: str, bundle_sha256: str) -> dict[str, Any]:
        result = bundle["result"]
        return {
            "schema_version": PERSISTENCE_SCHEMA_VERSION,
            "status": result["status"],
            "classification": result["classification"],
            "action": result["action"],
            "result_sha256": result_sha256,
            "bundle_sha256": bundle_sha256,
            "result": result,
            "bundle": bundle,
        }

    @classmethod
    def _rows_bundle_for_existing(cls, connection: sqlite3.Connection, run_id: str) -> dict[str, Any]:
        raw = [
            {
                "session_id": row["session_id"], "sample_kind": row["sample_kind"],
                "sample_index": row["sample_index"], "block_index": row["block_index"],
                "arm": row["arm"], "value": row["value"], "unit": row["unit"],
                "observed_at_ns": row["observed_at_ns"],
            }
            for row in connection.execute(
                "SELECT session_id,sample_kind,sample_index,block_index,arm,value,unit,observed_at_ns "
                "FROM raw_samples WHERE run_id=? ORDER BY sample_kind,sample_index", (run_id,)
            )
        ]
        scalar = [
            {
                "metric_name": row["metric_name"], "value": row["value"],
                "missing_reason": row["missing_reason"], "unit": row["unit"],
                "scope": row["scope"], "recorded_at_ns": row["recorded_at_ns"],
            }
            for row in connection.execute(
                "SELECT metric_name,value,missing_reason,unit,scope,recorded_at_ns "
                "FROM scalar_metrics WHERE run_id=? ORDER BY metric_name", (run_id,)
            )
        ]
        correctness = [
            {
                "case_name": row["case_name"], "metric_name": row["metric_name"],
                "value": row["value"], "unit": row["unit"], "passed": bool(row["passed"]),
                "detail": json.loads(row["detail_json"]),
            }
            for row in connection.execute(
                "SELECT case_name,metric_name,value,unit,passed,detail_json "
                "FROM correctness_metrics WHERE run_id=? ORDER BY case_name,metric_name", (run_id,)
            )
        ]
        artifacts = [
            {
                "artifact_name": row["artifact_name"], "artifact_kind": row["artifact_kind"],
                "sha256": row["sha256"], "metadata": json.loads(row["metadata_json"]),
            }
            for row in connection.execute(
                "SELECT artifact_name,artifact_kind,sha256,metadata_json "
                "FROM artifacts WHERE run_id=? ORDER BY artifact_name,artifact_kind", (run_id,)
            )
        ]
        return {
            "raw_samples": raw,
            "scalar_metrics": scalar,
            "correctness_metrics": correctness,
            "artifacts": artifacts,
        }

    @classmethod
    def _check_existing_bundle(
        cls,
        connection: sqlite3.Connection,
        manifest: ClosedManifest,
        bundle: Mapping[str, Any],
        result_sha256: str,
        bundle_sha256: str,
    ) -> None:
        run = connection.execute(
            """SELECT phase,mode,manifest_hash,manifest_json,
                      code_sha256,spec_sha256,environment_sha256,
                      revision,revision_missing_reason
                 FROM runs WHERE run_id=?""",
            (manifest.run_id,),
        ).fetchone()
        if run is None:
            return
        provenance = manifest.value["provenance"]
        revision = provenance["revision"]
        expected_mirrors = {
            "phase": manifest.value["phase"],
            "mode": manifest.value["mode"],
            "manifest_hash": manifest.sha256,
            "manifest_json": canonical_json(manifest.value),
            "code_sha256": provenance["code_sha256"],
            "spec_sha256": provenance["spec_sha256"],
            "environment_sha256": provenance["environment_sha256"],
            "revision": revision["value"],
            "revision_missing_reason": revision["missing_reason"],
        }
        for field, expected in expected_mirrors.items():
            if run[field] != expected:
                raise StorageError(f"run_id already exists with a different manifest mirror: {field}")
        # created_at_unix_ns is provenance for the first insert, not an identity
        # field; replay intentionally remains independent of the caller's clock.
        events = connection.execute(
            "SELECT status,payload_json,payload_hash FROM status_events "
            "WHERE run_id=? AND event_kind='common_result' ORDER BY event_id", (manifest.run_id,)
        ).fetchall()
        if len(events) != 1:
            raise StorageError("run_id has an incomplete or ambiguous common_result bundle")
        payload_text = events[0]["payload_json"]
        if not isinstance(payload_text, str):
            raise StorageError("existing common_result payload is not text")
        try:
            payload_bytes = payload_text.encode("utf-8")
        except UnicodeError as exc:
            raise StorageError("existing common_result payload is not UTF-8") from exc
        if len(payload_bytes) > PERSISTENCE_MAX_BUNDLE_BYTES:
            raise StorageError("existing common_result payload exceeds the bundle limit")
        try:
            payload = json.loads(payload_text)
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise StorageError("existing common_result payload is invalid") from exc
        if not isinstance(payload, Mapping):
            raise StorageError("existing common_result payload must be an object")
        expected_payload = cls._payload_for_bundle(bundle, result_sha256, bundle_sha256)
        try:
            canonical_payload = cls._bounded_json(
                payload, "existing common_result payload", limit=PERSISTENCE_MAX_BUNDLE_BYTES
            )
            payload_bundle = payload["bundle"]
            payload_hash = canonical_sha256(payload)
            payload_bundle_hash = canonical_sha256(payload_bundle)
        except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise StorageError("existing common_result payload is invalid") from exc
        if (
            events[0]["status"] != bundle["result"]["status"]
            or events[0]["payload_json"] != canonical_payload
            or events[0]["payload_hash"] != payload_hash
            or payload != expected_payload
            or payload_bundle_hash != bundle_sha256
        ):
            raise StorageError("run_id exists with a different common_result bundle")
        try:
            existing_rows_raw = cls._rows_bundle_for_existing(connection, manifest.run_id)
            existing_rows = {
                "raw_samples": cls._normalize_raw_samples(existing_rows_raw["raw_samples"]),
                "scalar_metrics": cls._normalize_scalar_metrics(existing_rows_raw["scalar_metrics"]),
                "correctness_metrics": cls._normalize_correctness_metrics(existing_rows_raw["correctness_metrics"]),
                "artifacts": cls._normalize_artifacts(existing_rows_raw["artifacts"]),
            }
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise StorageError("existing common_result child rows are invalid") from exc
        expected_rows = {key: bundle[key] for key in existing_rows}
        if existing_rows != expected_rows:
            raise StorageError("run_id exists with different child rows")

    def persist_common_result(
        self,
        manifest: ClosedManifest,
        result: Mapping[str, Any],
        *,
        created_at_unix_ns: int,
        raw_samples: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
        scalar_metrics: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
        correctness_metrics: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
        artifacts: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
        recorded_at_ns: int | None = None,
    ) -> PersistenceOutcome:
        """Atomically persist one closed Common-Result bundle.

        The canonical persistence envelope is schema 1 with ``result``, sorted
        child arrays, the manifest hash, a result SHA-256, and a bundle SHA-256
        over the complete envelope. The database receives one ``common_result``
        event whose status is exactly ``result['status']``. All validation and
        canonicalization precede ``BEGIN IMMEDIATE``; SQL failures roll back
        every row. Replaying an identical complete bundle is idempotent.

        Artifact entries are declarative hashes and bounded metadata. No path is
        accepted, opened, or treated as content that this method verified.
        """

        self._require_writable()
        if not isinstance(manifest, ClosedManifest):
            raise TypeError("persist_common_result requires a ClosedManifest")
        self._integer(created_at_unix_ns, "created_at_unix_ns")
        recorded_at_ns = self._optional_timestamp(recorded_at_ns, "recorded_at_ns")
        if self._connection.in_transaction:
            raise StorageError("persist_common_result requires an idle connection")
        bundle, result_sha256, bundle_sha256 = self._bundle_from_rows(
            manifest=manifest,
            result=result,
            raw_samples=raw_samples,
            scalar_metrics=scalar_metrics,
            correctness_metrics=correctness_metrics,
            artifacts=artifacts,
        )
        payload = self._payload_for_bundle(bundle, result_sha256, bundle_sha256)
        manifest_json = canonical_json(manifest.value)
        provenance = manifest.value["provenance"]
        self._bounded_json(payload, "common_result payload", limit=PERSISTENCE_MAX_BUNDLE_BYTES)
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            existing = self._connection.execute(
                "SELECT 1 FROM runs WHERE run_id=?", (manifest.run_id,)
            ).fetchone()
            if existing is not None:
                self._check_existing_bundle(
                    self._connection, manifest, bundle, result_sha256, bundle_sha256
                )
                self._connection.commit()
                return PersistenceOutcome("idempotent", manifest.run_id, bundle_sha256)
            revision = provenance["revision"]
            self._connection.execute(
                "INSERT INTO runs(run_id,phase,mode,manifest_json,manifest_hash,"
                "code_sha256,spec_sha256,environment_sha256,revision,"
                "revision_missing_reason,created_at_unix_ns) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    manifest.run_id, manifest.value["phase"], manifest.value["mode"], manifest_json,
                    manifest.sha256, provenance["code_sha256"], provenance["spec_sha256"],
                    provenance["environment_sha256"], revision["value"], revision["missing_reason"],
                    created_at_unix_ns,
                ),
            )
            payload_json = canonical_json(payload)
            self._connection.execute(
                "INSERT INTO status_events(run_id,event_kind,status,payload_json,payload_hash,recorded_at_ns) "
                "VALUES (?,?,?,?,?,?)",
                (manifest.run_id, "common_result", bundle["result"]["status"], payload_json,
                 canonical_sha256(payload), recorded_at_ns),
            )
            for row in bundle["raw_samples"]:
                self._connection.execute(
                    "INSERT INTO raw_samples(run_id,session_id,sample_kind,sample_index,block_index,arm,value,unit,observed_at_ns) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (manifest.run_id, row["session_id"], row["sample_kind"], row["sample_index"],
                     row["block_index"], row["arm"], row["value"], row["unit"], row["observed_at_ns"]),
                )
            for row in bundle["scalar_metrics"]:
                self._connection.execute(
                    "INSERT INTO scalar_metrics(run_id,metric_name,scope,value,missing_reason,unit,recorded_at_ns) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (manifest.run_id, row["metric_name"], row["scope"], row["value"], row["missing_reason"],
                     row["unit"], row["recorded_at_ns"]),
                )
            for row in bundle["correctness_metrics"]:
                self._connection.execute(
                    "INSERT INTO correctness_metrics(run_id,case_name,metric_name,value,unit,passed,detail_json) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (manifest.run_id, row["case_name"], row["metric_name"], row["value"], row["unit"],
                     int(row["passed"]), canonical_json(row["detail"])),
                )
            for row in bundle["artifacts"]:
                self._connection.execute(
                    "INSERT INTO artifacts(run_id,artifact_name,artifact_kind,sha256,metadata_json) "
                    "VALUES (?,?,?,?,?)",
                    (manifest.run_id, row["artifact_name"], row["artifact_kind"], row["sha256"],
                     canonical_json(row["metadata"])),
                )
            self._connection.commit()
            return PersistenceOutcome("inserted", manifest.run_id, bundle_sha256)
        except StorageError:
            self._connection.rollback()
            raise
        except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
            self._connection.rollback()
            raise StorageError(f"common_result persistence rolled back: {exc}") from exc

    def verify_common_result_bundle(
        self,
        manifest: ClosedManifest,
        result: Mapping[str, Any],
        *,
        raw_samples: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
        scalar_metrics: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
        correctness_metrics: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
        artifacts: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
    ) -> str:
        """Verify an already persisted Common-Result bundle without writing.

        The input is normalized through the same path as
        :meth:`persist_common_result`; the existing replay checker then verifies
        the immutable run mirrors, one common-result wrapper, all hashes, and
        every normalized child row.  This method never starts a transaction or
        changes SQLite state and is suitable for mode=ro loader verification.
        """

        if not isinstance(manifest, ClosedManifest):
            raise TypeError("verify_common_result_bundle requires a ClosedManifest")
        bundle, result_sha256, bundle_sha256 = self._bundle_from_rows(
            manifest=manifest,
            result=result,
            raw_samples=raw_samples,
            scalar_metrics=scalar_metrics,
            correctness_metrics=correctness_metrics,
            artifacts=artifacts,
        )
        payload = self._payload_for_bundle(bundle, result_sha256, bundle_sha256)
        self._bounded_json(payload, "common_result payload", limit=PERSISTENCE_MAX_BUNDLE_BYTES)
        exists = self._connection.execute(
            "SELECT 1 FROM runs WHERE run_id=?", (manifest.run_id,)
        ).fetchone()
        if exists is None:
            raise StorageError("common_result bundle is missing")
        try:
            self._check_existing_bundle(
                self._connection, manifest, bundle, result_sha256, bundle_sha256
            )
        except StorageError:
            raise
        except (sqlite3.Error, OSError, TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise StorageError("common_result bundle verification failed") from exc
        return "verified"

    def create_run(self, manifest: Mapping[str, Any], *, created_at_unix_ns: int) -> str:
        """Insert one immutable run and return its manifest hash."""

        self._require_writable()
        validated = validate_manifest(manifest)
        run_id = validated["run_id"]
        manifest_json = canonical_json(validated)
        digest = manifest_hash(validated)
        created_at_unix_ns = self._integer(created_at_unix_ns, "created_at_unix_ns")
        provenance = validated["provenance"]
        revision = provenance["revision"]
        try:
            with self._connection:
                self._connection.execute(
                    """INSERT INTO runs(
                           run_id, phase, mode, manifest_json, manifest_hash,
                           code_sha256, spec_sha256, environment_sha256,
                           revision, revision_missing_reason, created_at_unix_ns
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id, validated["phase"], validated["mode"], manifest_json, digest,
                        provenance["code_sha256"], provenance["spec_sha256"],
                        provenance["environment_sha256"], revision["value"],
                        revision["missing_reason"], created_at_unix_ns,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise StorageError(f"run insert rejected: {exc}") from exc
        return digest

    record_run = create_run

    def append_status_event(
        self,
        run_id: str,
        event_kind: str,
        status: str,
        payload: Mapping[str, Any],
        *,
        recorded_at_ns: int | None = None,
    ) -> int:
        self._require_writable()
        run_id = self._text(run_id, "run_id")
        event_kind = self._text(event_kind, "event_kind")
        status = self._text(status, "status")
        if not isinstance(payload, Mapping):
            raise StorageError("payload must be an object")
        payload_json = self._bounded_json(payload, "status event payload", limit=PERSISTENCE_MAX_EVENT_BYTES)
        payload_hash = canonical_sha256(payload)
        recorded_at_ns = self._optional_timestamp(recorded_at_ns, "recorded_at_ns")
        try:
            with self._connection:
                cursor = self._connection.execute(
                    """INSERT INTO status_events
                       (run_id, event_kind, status, payload_json, payload_hash, recorded_at_ns)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (run_id, event_kind, status, payload_json, payload_hash, recorded_at_ns),
                )
        except sqlite3.IntegrityError as exc:
            raise StorageError(f"status event rejected: {exc}") from exc
        return int(cursor.lastrowid)

    def append_raw_sample(
        self,
        run_id: str,
        session_id: str,
        block_index: int,
        arm: str,
        value: float,
        unit: str,
        *,
        sample_kind: str = "timing",
        sample_index: int = 0,
        observed_at_ns: int | None = None,
    ) -> int:
        self._require_writable()
        run_id = self._text(run_id, "run_id")
        session_id = self._text(session_id, "session_id")
        sample_kind = self._text(sample_kind, "sample_kind")
        arm = self._text(arm, "arm")
        unit = self._text(unit, "unit")
        block_index = self._integer(block_index, "block_index")
        sample_index = self._integer(sample_index, "sample_index")
        value = self._finite(value, "value", nonnegative=True)
        observed_at_ns = self._optional_timestamp(observed_at_ns, "observed_at_ns")
        try:
            with self._connection:
                cursor = self._connection.execute(
                    """INSERT INTO raw_samples
                       (run_id, session_id, sample_kind, sample_index, block_index, arm, value, unit, observed_at_ns)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (run_id, session_id, sample_kind, sample_index, block_index, arm, value, unit, observed_at_ns),
                )
        except sqlite3.IntegrityError as exc:
            raise StorageError(f"raw sample rejected: {exc}") from exc
        return int(cursor.lastrowid)

    def append_scalar_metric(
        self,
        run_id: str,
        metric_name: str,
        value: float | None,
        unit: str,
        *,
        scope: str = "run",
        missing_reason: str | None = None,
        recorded_at_ns: int | None = None,
    ) -> int:
        self._require_writable()
        run_id = self._text(run_id, "run_id")
        metric_name = self._text(metric_name, "metric_name")
        unit = self._text(unit, "unit")
        scope = self._text(scope, "scope")
        if (value is None) == (missing_reason is None):
            raise StorageError("exactly one of value or missing_reason is required")
        if value is not None:
            value = self._finite(value, "value")
        else:
            missing_reason = self._text(missing_reason, "missing_reason", maximum=256)
        recorded_at_ns = self._optional_timestamp(recorded_at_ns, "recorded_at_ns")
        try:
            with self._connection:
                cursor = self._connection.execute(
                    """INSERT INTO scalar_metrics
                       (run_id, metric_name, scope, value, missing_reason, unit, recorded_at_ns)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (run_id, metric_name, scope, value, missing_reason, unit, recorded_at_ns),
                )
        except sqlite3.IntegrityError as exc:
            raise StorageError(f"scalar metric rejected: {exc}") from exc
        return int(cursor.lastrowid)

    def append_correctness_metric(
        self,
        run_id: str,
        case_name: str,
        metric_name: str,
        value: float,
        unit: str,
        passed: bool,
        *,
        detail: Mapping[str, Any] | None = None,
    ) -> int:
        self._require_writable()
        run_id = self._text(run_id, "run_id")
        case_name = self._text(case_name, "case_name")
        metric_name = self._text(metric_name, "metric_name")
        unit = self._text(unit, "unit")
        if not isinstance(passed, bool):
            raise StorageError("passed must be bool")
        value = self._finite(value, "value")
        detail_json = self._bounded_json(
            detail or {}, "correctness metric detail", limit=PERSISTENCE_MAX_CHILD_BYTES
        )
        try:
            with self._connection:
                cursor = self._connection.execute(
                    """INSERT INTO correctness_metrics
                       (run_id, case_name, metric_name, value, unit, passed, detail_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (run_id, case_name, metric_name, value, unit, int(passed), detail_json),
                )
        except sqlite3.IntegrityError as exc:
            raise StorageError(f"correctness metric rejected: {exc}") from exc
        return int(cursor.lastrowid)

    def append_artifact(
        self,
        run_id: str,
        artifact_name: str,
        artifact_kind: str,
        sha256: str,
        metadata: Mapping[str, Any],
    ) -> int:
        self._require_writable()
        run_id = self._text(run_id, "run_id")
        artifact_name = self._text(artifact_name, "artifact_name")
        artifact_kind = self._text(artifact_kind, "artifact_kind")
        if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
            raise StorageError("sha256 must be 64 lowercase hexadecimal characters")
        if not isinstance(metadata, Mapping):
            raise StorageError("metadata must be an object")
        metadata_json = self._bounded_json(
            metadata, "artifact metadata", limit=PERSISTENCE_MAX_CHILD_BYTES
        )
        try:
            with self._connection:
                cursor = self._connection.execute(
                    """INSERT INTO artifacts
                       (run_id, artifact_name, artifact_kind, sha256, metadata_json)
                       VALUES (?, ?, ?, ?, ?)""",
                    (run_id, artifact_name, artifact_kind, sha256, metadata_json),
                )
        except sqlite3.IntegrityError as exc:
            raise StorageError(f"artifact rejected: {exc}") from exc
        return int(cursor.lastrowid)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["manifest"] = json.loads(result.pop("manifest_json"))
        return result

    def rows(self, table: str, run_id: str) -> list[dict[str, Any]]:
        """Read a fixed table for a future mode=ro dashboard; no SQL interpolation is accepted."""

        allowed = {"status_events", "raw_samples", "scalar_metrics", "correctness_metrics", "artifacts"}
        if table not in allowed:
            raise StorageError("unsupported dashboard table")
        return [
            dict(row)
            for row in self._connection.execute(
                f"SELECT * FROM {table} WHERE run_id=? ORDER BY rowid", (run_id,)
            )
        ]
