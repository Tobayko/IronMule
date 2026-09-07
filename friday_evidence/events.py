"""Small append-only metadata journal for product calibration events.

The journal deliberately has no model or prompt semantics.  Payloads are
canonical JSON metadata, and the recursive key deny-list prevents accidental
capture of user content at this boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
import json
import os
from pathlib import Path
import sqlite3
import stat
import time
from typing import Any
from urllib.parse import quote

from .canonical import CanonicalError, canonical_json_bytes, canonical_sha256
from .storage import StorageError, _checked_path, _secure_create


class EventJournalError(RuntimeError):
    """The event journal cannot uphold its storage or metadata contract."""


def _deny_extension_load(action: int, _arg1: Any, arg2: Any, _database: Any, _source: Any) -> int:
    """Deny SQL extension loading even when Python lacks the disable method."""
    if action == getattr(sqlite3, "SQLITE_FUNCTION", 31) and arg2 == "load_extension":
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _secure_extension_controls(connection: Any) -> None:
    """Keep extension loading disabled on both full and reduced Python builds."""
    try:
        disable = connection.enable_load_extension
    except AttributeError:
        # macOS Python builds may omit the optional API; SQLite defaults to
        # disabled loading, and the authorizer closes the SQL-function route.
        pass
    else:
        try:
            disable(False)
        except (sqlite3.Error, OSError) as exc:
            raise EventJournalError("cannot disable SQLite extension loading") from exc
    try:
        connection.set_authorizer(_deny_extension_load)
    except (AttributeError, sqlite3.Error) as exc:
        raise EventJournalError("cannot install SQLite extension-loading guard") from exc


_SCHEMA_VERSION = 1
_APPLICATION_ID = 0x494D4556  # ``IMEV``
_MAX_DB_BYTES = 64 * 1024 * 1024
_MAX_PAYLOAD_BYTES = 256 * 1024
_MAX_QUERY_LIMIT = 1000
_MAX_DEPTH = 32
_MAX_NODES = 100_000
_BUSY_TIMEOUT_MS = 2_000
_ZERO_SHA256 = "0" * 64
_KINDS = frozenset({
    "run_started", "readiness", "deferred", "worker_started", "sample",
    "validation", "run_finished", "activation", "rollback", "pause",
})
_FORBIDDEN_KEYS = frozenset({
    "prompt", "prompts", "messages", "content", "text", "tokens",
    "token_ids", "answers",
})
_EVENT_COLUMNS = (
    "seq", "recorded_unix_ns", "run_id", "kind", "payload_json",
    "prev_sha256", "sha256",
)
_SCHEMA_OBJECTS = {
    ("table", "event_journal"),
    ("trigger", "event_journal_no_update"),
    ("trigger", "event_journal_no_delete"),
}


def _normalized_sql(value: str) -> str:
    return " ".join(value.split()).lower()


_EXPECTED_SQL = {
    ("table", "event_journal"): _normalized_sql(
        "CREATE TABLE event_journal (seq INTEGER PRIMARY KEY,"
        "recorded_unix_ns INTEGER NOT NULL,run_id TEXT NOT NULL,"
        "kind TEXT NOT NULL,payload_json TEXT NOT NULL,"
        "prev_sha256 TEXT NOT NULL,sha256 TEXT NOT NULL)"
    ),
    ("trigger", "event_journal_no_update"): _normalized_sql(
        "CREATE TRIGGER event_journal_no_update BEFORE UPDATE ON event_journal "
        "BEGIN SELECT RAISE(ABORT, 'event journal is append-only'); END"
    ),
    ("trigger", "event_journal_no_delete"): _normalized_sql(
        "CREATE TRIGGER event_journal_no_delete BEFORE DELETE ON event_journal "
        "BEGIN SELECT RAISE(ABORT, 'event journal is append-only'); END"
    ),
}


def _checked_file(path: Path) -> tuple[Path, bool]:
    """Validate a journal path without creating it."""
    try:
        checked = _checked_path(path, create_parent=False)
    except StorageError as exc:
        raise EventJournalError(str(exc)) from exc
    try:
        info = os.lstat(checked)
    except FileNotFoundError:
        return checked, False
    except OSError as exc:
        raise EventJournalError("event journal path is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise EventJournalError("event journal must be a regular file, not a symlink")
    if info.st_uid != os.geteuid() or info.st_nlink != 1:
        raise EventJournalError("event journal ownership or link count is unsafe")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise EventJournalError("event journal file must be private")
    if info.st_size > _MAX_DB_BYTES:
        raise EventJournalError("event journal exceeds the maximum database size")
    return checked, True


def _validate_run_id(run_id: Any) -> str:
    if not isinstance(run_id, str) or len(run_id) != 32:
        raise EventJournalError("run_id must be a 32-character hexadecimal UUID")
    if any(character not in "0123456789abcdefABCDEF" for character in run_id):
        raise EventJournalError("run_id must be a 32-character hexadecimal UUID")
    return run_id


def _validate_payload(payload: Any) -> str:
    nodes = 0
    active: set[int] = set()

    def visit(value: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_NODES:
            raise EventJournalError("event payload contains too many values")
        if depth > _MAX_DEPTH:
            raise EventJournalError("event payload is too deeply nested")
        if isinstance(value, Mapping):
            marker = id(value)
            if marker in active:
                raise EventJournalError("event payload contains a cycle")
            active.add(marker)
            try:
                for key, child in value.items():
                    if not isinstance(key, str):
                        raise EventJournalError("event payload object keys must be strings")
                    if key in _FORBIDDEN_KEYS:
                        raise EventJournalError(f"event payload contains forbidden key: {key}")
                    visit(child, depth + 1)
            finally:
                active.remove(marker)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            marker = id(value)
            if marker in active:
                raise EventJournalError("event payload contains a cycle")
            active.add(marker)
            try:
                for child in value:
                    visit(child, depth + 1)
            finally:
                active.remove(marker)

    visit(payload, 0)
    try:
        encoded = canonical_json_bytes(payload)
    except (CanonicalError, UnicodeError, TypeError, ValueError) as exc:
        raise EventJournalError("event payload is not supported canonical JSON") from exc
    if len(encoded) > _MAX_PAYLOAD_BYTES:
        raise EventJournalError("event payload exceeds 256 KiB")
    return encoded.decode("utf-8")


def _validate_kind(kind: Any) -> str:
    if not isinstance(kind, str) or kind not in _KINDS:
        raise EventJournalError("event kind is not registered")
    return kind


def _event_body(
    seq: int, recorded_unix_ns: int, run_id: str, kind: str,
    payload: Any, prev_sha256: str,
) -> dict[str, Any]:
    return {
        "seq": seq,
        "recorded_unix_ns": recorded_unix_ns,
        "run_id": run_id,
        "kind": kind,
        "payload": payload,
        "prev_sha256": prev_sha256,
    }


class EventJournal(AbstractContextManager["EventJournal"]):
    """A private, append-only event journal backed by one SQLite file."""

    def __init__(self, path: str | Path, read_only: bool = False):
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        checked, existed = _checked_file(candidate)
        if read_only and not existed:
            raise EventJournalError("event journal does not exist")
        created = False
        if not existed:
            try:
                _secure_create(checked)
            except StorageError as exc:
                raise EventJournalError("cannot securely create event journal") from exc
            created = True
        try:
            if read_only:
                uri = f"file:{quote(str(checked), safe='/')}?mode=ro&immutable=0"
                connection = sqlite3.connect(uri, uri=True, timeout=_BUSY_TIMEOUT_MS / 1000)
            else:
                connection = sqlite3.connect(str(checked), timeout=_BUSY_TIMEOUT_MS / 1000)
        except sqlite3.Error as exc:
            if created:
                try:
                    checked.unlink()
                except OSError:
                    pass
            raise EventJournalError("cannot open event journal") from exc
        self.path = checked
        self.read_only = read_only
        self._connection: sqlite3.Connection | None = connection
        try:
            self._configure()
            if created:
                self._initialize()
            self._verify_schema()
            if not self.read_only:
                self._set_page_limit()
            self._check_size()
            self._verify_chain()
        except Exception as exc:
            connection.close()
            self._connection = None
            if created:
                try:
                    checked.unlink()
                except OSError:
                    pass
            if isinstance(exc, EventJournalError):
                raise
            raise EventJournalError("cannot validate event journal") from exc

    def _connection_or_error(self) -> sqlite3.Connection:
        if self._connection is None:
            raise EventJournalError("event journal is closed")
        return self._connection

    def _configure(self) -> None:
        connection = self._connection_or_error()
        try:
            _secure_extension_controls(connection)
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            mode = connection.execute(
                "PRAGMA journal_mode" if self.read_only else "PRAGMA journal_mode=DELETE"
            ).fetchone()[0]
            if str(mode).lower() != "delete":
                raise EventJournalError("event journal must use DELETE journaling")
            if self.read_only:
                connection.execute("PRAGMA query_only=ON")
                if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
                    raise EventJournalError("read-only event journal is not query-only")
        except EventJournalError:
            raise
        except sqlite3.Error as exc:
            raise EventJournalError("cannot configure event journal SQLite controls") from exc

    def _initialize(self) -> None:
        connection = self._connection_or_error()
        try:
            connection.execute(f"PRAGMA application_id={_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            connection.executescript(
                "CREATE TABLE event_journal ("
                "seq INTEGER PRIMARY KEY,"
                "recorded_unix_ns INTEGER NOT NULL,"
                "run_id TEXT NOT NULL,"
                "kind TEXT NOT NULL,"
                "payload_json TEXT NOT NULL,"
                "prev_sha256 TEXT NOT NULL,"
                "sha256 TEXT NOT NULL"
                ");"
                "CREATE TRIGGER event_journal_no_update BEFORE UPDATE ON event_journal "
                "BEGIN SELECT RAISE(ABORT, 'event journal is append-only'); END;"
                "CREATE TRIGGER event_journal_no_delete BEFORE DELETE ON event_journal "
                "BEGIN SELECT RAISE(ABORT, 'event journal is append-only'); END;"
            )
            connection.commit()
            self._set_page_limit()
        except (sqlite3.Error, ValueError) as exc:
            connection.rollback()
            raise EventJournalError("cannot initialize event journal schema") from exc

    def _set_page_limit(self) -> None:
        connection = self._connection_or_error()
        try:
            page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
            if page_size <= 0:
                raise EventJournalError("event journal has an invalid page size")
            maximum = _MAX_DB_BYTES // page_size
            current = int(connection.execute(f"PRAGMA max_page_count={maximum}").fetchone()[0])
            if current > maximum:
                raise EventJournalError("event journal exceeds the maximum database size")
        except EventJournalError:
            raise
        except (sqlite3.Error, TypeError, ValueError) as exc:
            raise EventJournalError("cannot constrain event journal size") from exc

    def _check_size(self) -> None:
        try:
            info = self.path.lstat()
        except OSError as exc:
            raise EventJournalError("event journal path is unavailable") from exc
        if info.st_uid != os.geteuid() or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) & 0o077:
            raise EventJournalError("event journal ownership or permissions are unsafe")
        if info.st_size > _MAX_DB_BYTES:
            raise EventJournalError("event journal exceeds the maximum database size")

    def _verify_schema(self) -> None:
        connection = self._connection_or_error()
        try:
            application_id = connection.execute("PRAGMA application_id").fetchone()[0]
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            integrity = connection.execute("PRAGMA integrity_check(1)").fetchone()[0]
            objects = {
                (row[0], row[1]) for row in connection.execute(
                    "SELECT type,name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
                )
            }
            definitions = {
                (row[0], row[1]): _normalized_sql(row[2] or "")
                for row in connection.execute(
                    "SELECT type,name,sql FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
                )
            }
            columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(event_journal)"))
        except sqlite3.Error as exc:
            raise EventJournalError("cannot verify event journal schema") from exc
        if (
            application_id != _APPLICATION_ID
            or version != _SCHEMA_VERSION
            or integrity != "ok"
            or objects != _SCHEMA_OBJECTS
            or definitions != _EXPECTED_SQL
            or columns != _EVENT_COLUMNS
        ):
            raise EventJournalError("event journal schema is invalid")

    @staticmethod
    def _row_event(row: sqlite3.Row | tuple[Any, ...]) -> dict[str, Any]:
        seq, recorded, run_id, kind, payload_json, previous, digest = row
        if type(seq) is not int or type(recorded) is not int:
            raise EventJournalError("event journal contains invalid integer fields")
        if not isinstance(run_id, str) or not isinstance(kind, str):
            raise EventJournalError("event journal contains invalid identity fields")
        try:
            payload = json.loads(payload_json)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EventJournalError("event journal contains invalid payload JSON") from exc
        if _validate_run_id(run_id) != run_id or _validate_kind(kind) != kind:
            raise EventJournalError("event journal contains invalid event identity")
        if not isinstance(previous, str) or len(previous) != 64 or not all(c in "0123456789abcdef" for c in previous):
            raise EventJournalError("event journal contains invalid chain predecessor")
        if not isinstance(digest, str) or len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
            raise EventJournalError("event journal contains invalid event digest")
        payload_text = _validate_payload(payload)
        if payload_text != payload_json:
            raise EventJournalError("event journal payload is not canonical")
        event = {
            "seq": seq,
            "recorded_unix_ns": recorded,
            "run_id": run_id,
            "kind": kind,
            "payload": payload,
            "prev_sha256": previous,
            "sha256": digest,
        }
        body = {key: event[key] for key in (
            "seq", "recorded_unix_ns", "run_id", "kind", "payload", "prev_sha256"
        )}
        if canonical_sha256(body) != digest:
            raise EventJournalError("event journal event digest is invalid")
        return event

    def _verify_chain(self) -> int:
        connection = self._connection_or_error()
        count = 0
        previous = _ZERO_SHA256
        expected_seq = 1
        try:
            rows = connection.execute(
                "SELECT seq,recorded_unix_ns,run_id,kind,payload_json,prev_sha256,sha256 "
                "FROM event_journal ORDER BY seq ASC"
            )
            for row in rows:
                event = self._row_event(row)
                if event["seq"] != expected_seq or event["prev_sha256"] != previous:
                    raise EventJournalError("event journal hash chain is broken")
                previous = event["sha256"]
                expected_seq += 1
                count += 1
            return count
        except EventJournalError:
            raise
        except sqlite3.Error as exc:
            raise EventJournalError("cannot verify event journal") from exc

    def _begin_read(self) -> None:
        try:
            self._connection_or_error().execute("BEGIN")
        except sqlite3.Error as exc:
            raise EventJournalError("cannot begin event journal read") from exc

    def _rollback_read(self) -> None:
        try:
            self._connection_or_error().rollback()
        except sqlite3.Error:
            pass

    def append(self, run_id: str, kind: str, payload: Any) -> dict[str, Any]:
        if self.read_only:
            raise EventJournalError("read-only event journal cannot append")
        run_id = _validate_run_id(run_id)
        kind = _validate_kind(kind)
        payload_json = _validate_payload(payload)
        connection = self._connection_or_error()
        try:
            connection.execute("BEGIN IMMEDIATE")
            # Never extend a chain that was already corrupted between opens.
            self._verify_chain()
            row = connection.execute(
                "SELECT seq,sha256 FROM event_journal ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            seq = 1 if row is None else int(row[0]) + 1
            previous = _ZERO_SHA256 if row is None else row[1]
            recorded = time.time_ns()
            decoded = json.loads(payload_json)
            body = _event_body(seq, recorded, run_id, kind, decoded, previous)
            digest = canonical_sha256(body)
            connection.execute(
                "INSERT INTO event_journal "
                "(seq,recorded_unix_ns,run_id,kind,payload_json,prev_sha256,sha256) "
                "VALUES(?,?,?,?,?,?,?)",
                (seq, recorded, run_id, kind, payload_json, previous, digest),
            )
            connection.commit()
            self._check_size()
            return {**body, "sha256": digest}
        except EventJournalError:
            try:
                connection.rollback()
            except sqlite3.Error:
                pass
            raise
        except (sqlite3.Error, TypeError, ValueError, OverflowError) as exc:
            try:
                connection.rollback()
            except sqlite3.Error:
                pass
            raise EventJournalError("cannot append event journal record") from exc

    def events(self, run_id: str | None = None, limit: int = 100, after_seq: int = 0) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= _MAX_QUERY_LIMIT:
            raise EventJournalError("event history limit must be an integer from 1 through 1000")
        if type(after_seq) is not int or after_seq < 0:
            raise EventJournalError("after_seq must be a non-negative integer")
        if run_id is not None:
            run_id = _validate_run_id(run_id)
        connection = self._connection_or_error()
        try:
            self._begin_read()
            self._verify_chain()
            if run_id is None:
                rows = connection.execute(
                    "SELECT seq,recorded_unix_ns,run_id,kind,payload_json,prev_sha256,sha256 "
                    "FROM event_journal WHERE seq>? ORDER BY seq ASC LIMIT ?",
                    (after_seq, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT seq,recorded_unix_ns,run_id,kind,payload_json,prev_sha256,sha256 "
                    "FROM event_journal WHERE run_id=? AND seq>? ORDER BY seq ASC LIMIT ?",
                    (run_id, after_seq, limit),
                ).fetchall()
            result = [self._row_event(row) for row in rows]
            self._rollback_read()
            return result
        except EventJournalError:
            self._rollback_read()
            raise
        except sqlite3.Error as exc:
            self._rollback_read()
            raise EventJournalError("cannot read event journal history") from exc

    def latest(self, run_id: str | None = None) -> dict[str, Any] | None:
        if run_id is not None:
            run_id = _validate_run_id(run_id)
        connection = self._connection_or_error()
        try:
            self._begin_read()
            self._verify_chain()
            if run_id is None:
                row = connection.execute(
                    "SELECT seq,recorded_unix_ns,run_id,kind,payload_json,prev_sha256,sha256 "
                    "FROM event_journal ORDER BY seq DESC LIMIT 1"
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT seq,recorded_unix_ns,run_id,kind,payload_json,prev_sha256,sha256 "
                    "FROM event_journal WHERE run_id=? ORDER BY seq DESC LIMIT 1",
                    (run_id,),
                ).fetchone()
            result = None if row is None else self._row_event(row)
            self._rollback_read()
            return result
        except EventJournalError:
            self._rollback_read()
            raise
        except sqlite3.Error as exc:
            self._rollback_read()
            raise EventJournalError("cannot read latest event journal record") from exc

    def verify(self) -> int:
        try:
            self._begin_read()
            result = self._verify_chain()
            self._rollback_read()
            return result
        except EventJournalError:
            self._rollback_read()
            raise

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "EventJournal":
        self._connection_or_error()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["EventJournal", "EventJournalError"]
