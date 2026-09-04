"""Bounded, read-only adapters used by the L1.1 corpus auditor.

The adapters in this module deliberately have a small surface.  They never
follow links, never execute data, and never expose a general SQL query method.
This is important because the corpus contains historical evidence produced by
several unrelated runners and must be treated as untrusted input.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from urllib.parse import quote
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_bytes, loads_strict


DEFAULT_DISCOVERY_ROOTS: tuple[str, ...] = (
    ".friday-data",
    "experiments",
    "profiles",
    "tests/fixtures",
)
KNOWN_NON_EVIDENCE_PREFIXES: tuple[str, ...] = (".friday-data/models",)
KNOWN_NON_EVIDENCE_SUFFIXES: tuple[str, ...] = (".pyc",)
KNOWN_NON_EVIDENCE_PATH_PREFIXES: tuple[str, ...] = (".friday-data/optimizer-",)


class AdapterError(ValueError):
    """A source could not be safely inspected."""


class BoundsExceeded(AdapterError):
    """A source exceeded an explicit reader limit."""

    def __init__(self, message: str, *, partial_files: Sequence["DiscoveredFile"] = ()):
        super().__init__(message)
        self.partial_files = tuple(partial_files)


class InvalidJSON(AdapterError):
    """JSON is malformed, non-UTF-8, duplicated, or otherwise unsafe."""


class DuplicateJSONKey(InvalidJSON):
    """A JSON object contains a duplicate key."""


@dataclass(frozen=True)
class DiscoveryLimits:
    """Hard limits for discovery and source readers.

    Limits are intentionally conservative but large enough for the archived
    evidence files in the repository.  Callers can lower them for tests or
    hostile inputs; zero and negative values are rejected.
    """

    max_files: int = 10_000
    max_file_bytes: int = 32 * 1024 * 1024
    max_total_bytes: int = 512 * 1024 * 1024
    max_depth: int = 8
    max_json_depth: int = 32
    max_json_nodes: int = 1_000_000
    max_sqlite_rows: int = 100_000

    def __post_init__(self) -> None:
        for name in (
            "max_files",
            "max_file_bytes",
            "max_total_bytes",
            "max_depth",
            "max_json_depth",
            "max_json_nodes",
            "max_sqlite_rows",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class DiscoveredFile:
    """A stable, bounded file inventory row."""

    path: Path
    relative_path: str
    root_name: str
    kind: str
    size_bytes: int
    sha256: str
    st_dev: int | None = None
    st_ino: int | None = None
    read_error: str | None = None
    excluded_reason: str | None = None


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise AdapterError("path escaped selected root") from exc


def _check_regular(path: Path) -> os.stat_result:
    """Stat without following a symlink and reject non-regular files."""

    _reject_symlink_ancestors(path)
    st = path.lstat()
    if os.path.islink(path):
        raise AdapterError("symlink is not an admissible corpus source")
    if not os.path.isfile(path):
        raise AdapterError("corpus source is not a regular file")
    return st


def _reject_symlink_ancestors(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor or os.sep)
    for component in absolute.parts[1:-1]:
        current /= component
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise AdapterError("source ancestor may not be a symlink")
        except FileNotFoundError:
            raise AdapterError("source ancestor does not exist")


def _identity(stat_result: os.stat_result) -> tuple[int, int, int]:
    return (int(stat_result.st_dev), int(stat_result.st_ino), int(stat_result.st_size))


def read_stable_bytes(
    path: str | os.PathLike[str],
    *,
    limits: DiscoveryLimits | None = None,
    expected_identity: tuple[int, int, int] | None = None,
    expected_sha256: str | None = None,
) -> tuple[bytes, tuple[int, int, int]]:
    """Read one regular file through one descriptor and verify TOCTOU identity."""

    limits = limits or DiscoveryLimits()
    source = Path(path)
    _reject_symlink_ancestors(source)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(source, flags)
    except OSError as exc:
        raise AdapterError(f"cannot safely open source: {source}") from exc
    try:
        before_stat = os.fstat(fd)
        if not stat.S_ISREG(before_stat.st_mode):
            raise AdapterError("corpus source is not a regular file")
        before = _identity(before_stat)
        if expected_identity is not None and before != expected_identity:
            raise AdapterError(f"source changed before read: {source}")
        if before[2] > limits.max_file_bytes:
            raise BoundsExceeded(f"file exceeds {limits.max_file_bytes} bytes: {source}")
        chunks: list[bytes] = []
        remaining = before[2]
        while remaining:
            block = os.read(fd, min(1024 * 1024, remaining))
            if not block:
                raise AdapterError(f"file truncated while being read: {source}")
            chunks.append(block)
            remaining -= len(block)
        after = _identity(os.fstat(fd))
        if after != before:
            raise AdapterError(f"source changed while being read: {source}")
        if expected_identity is not None and after != expected_identity:
            raise AdapterError(f"source changed after read: {source}")
        raw = b"".join(chunks)
        if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise AdapterError(f"source content changed while being read: {source}")
        return raw, after
    finally:
        os.close(fd)


def _sha256_file(
    path: Path,
    *,
    size: int,
    limits: DiscoveryLimits,
    expected_identity: tuple[int, int, int] | None = None,
) -> tuple[str, tuple[int, int, int]]:
    raw, identity = read_stable_bytes(
        path, limits=limits, expected_identity=expected_identity or None
    )
    if len(raw) != size:
        raise AdapterError(f"file size changed while being read: {path}")
    return hashlib.sha256(raw).hexdigest(), identity


def discover_files(
    project_root: str | os.PathLike[str],
    *,
    roots: Sequence[str] = DEFAULT_DISCOVERY_ROOTS,
    limits: DiscoveryLimits | None = None,
) -> tuple[DiscoveredFile, ...]:
    """Discover only explicitly allowed roots in stable sorted order.

    ``os.scandir`` is used with ``follow_symlinks=False`` and each candidate is
    checked again with ``lstat``.  A symlink is skipped and never descended
    into.  Missing roots are represented by no rows, not by a broad fallback
    scan of the project.
    """

    limits = limits or DiscoveryLimits()
    base = Path(project_root).expanduser().resolve()
    if not base.is_dir():
        raise AdapterError(f"project root is not a directory: {base}")
    normalized_roots: list[tuple[str, Path]] = []
    for root_name in roots:
        if not isinstance(root_name, str) or not root_name or Path(root_name).is_absolute():
            raise AdapterError("discovery roots must be non-empty relative paths")
        raw_candidate = base / root_name
        # Check the non-resolved path first.  Resolving a symlink before this
        # check would make the later ``is_symlink`` guard ineffective.
        if raw_candidate.is_symlink():
            continue
        candidate = raw_candidate.resolve()
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise AdapterError("discovery root escapes project root") from exc
        normalized_roots.append((Path(root_name).as_posix(), candidate))

    found: list[DiscoveredFile] = []
    total_bytes = 0

    def walk(root_name: str, directory: Path, depth: int) -> None:
        nonlocal total_bytes
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return
        try:
            for entry in entries:
                candidate = Path(entry.path)
                try:
                    st = candidate.lstat()
                except (FileNotFoundError, PermissionError):
                    continue
                if os.path.islink(candidate):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    relative_directory = candidate.relative_to(base).as_posix()
                    if relative_directory in KNOWN_NON_EVIDENCE_PREFIXES or any(relative_directory.startswith(prefix) for prefix in KNOWN_NON_EVIDENCE_PATH_PREFIXES):
                        found.append(
                            DiscoveredFile(
                                path=candidate,
                                relative_path=relative_directory,
                                root_name=root_name,
                                kind="excluded",
                                size_bytes=0,
                                sha256="",
                                excluded_reason=("optimizer_control_plane_state" if any(relative_directory.startswith(prefix) for prefix in KNOWN_NON_EVIDENCE_PATH_PREFIXES) else "known_local_model_cache"),
                            )
                        )
                        continue
                    if depth >= limits.max_depth:
                        raise BoundsExceeded(
                            f"discovery exceeds depth {limits.max_depth}: {candidate}",
                            partial_files=found,
                        )
                    walk(root_name, candidate, depth + 1)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                relative_file = candidate.relative_to(base).as_posix()
                control_state = any(relative_file.startswith(prefix) for prefix in KNOWN_NON_EVIDENCE_PATH_PREFIXES)
                if control_state or candidate.suffix.lower() in KNOWN_NON_EVIDENCE_SUFFIXES or "__pycache__" in candidate.parts:
                    found.append(
                        DiscoveredFile(
                            path=candidate,
                            relative_path=relative_file,
                            root_name=root_name,
                            kind="excluded",
                            size_bytes=0,
                            sha256="",
                            excluded_reason=("optimizer_control_plane_state" if control_state else "python_bytecode_cache"),
                        )
                    )
                    continue
                if len(found) >= limits.max_files:
                    # A bounded discovery is a partial, safe inventory.  The
                    # caller can still audit all admissible evidence instead
                    # of losing the complete result because a model blob is
                    # present below an explicitly allowed root.
                    raise BoundsExceeded(f"discovery exceeds {limits.max_files} files", partial_files=found)
                size = int(st.st_size)
                if size > limits.max_file_bytes:
                    raise BoundsExceeded(f"file exceeds {limits.max_file_bytes} bytes: {candidate}", partial_files=found)
                if total_bytes + size > limits.max_total_bytes:
                    raise BoundsExceeded(f"discovery exceeds {limits.max_total_bytes} bytes", partial_files=found)
                try:
                    digest, stable_identity = _sha256_file(
                        candidate,
                        size=size,
                        limits=limits,
                        expected_identity=_identity(st),
                    )
                    read_error = None
                except AdapterError as exc:
                    # Preserve the discovered path and identity so the corpus
                    # audit can produce a terminal quarantine report instead
                    # of silently treating a race as missing data.
                    digest = ""
                    stable_identity = _identity(st)
                    read_error = str(exc)
                rel = candidate.relative_to(base).as_posix()
                suffix = candidate.suffix.lower()
                kind = "sqlite" if suffix in {".sqlite", ".sqlite3", ".db"} else suffix.lstrip(".") or "file"
                found.append(
                    DiscoveredFile(
                        path=candidate,
                        relative_path=rel,
                        root_name=root_name,
                        kind=kind,
                        size_bytes=size,
                        sha256=digest,
                        st_dev=stable_identity[0],
                        st_ino=stable_identity[1],
                        read_error=read_error,
                    )
                )
                total_bytes += size
        finally:
            # ``os.scandir`` returns DirEntry objects, not open handles; the
            # iterator itself is already exhausted and owns no per-entry
            # ``close`` method.
            pass

    for root_name, directory in normalized_roots:
        if directory.is_dir() and not os.path.islink(directory):
            walk(root_name, directory, 0)
    return tuple(sorted(found, key=lambda row: row.relative_path))


def _bounded_json(value: Any, *, depth: int, nodes: list[int], limits: DiscoveryLimits) -> Any:
    nodes[0] += 1
    if nodes[0] > limits.max_json_nodes:
        raise BoundsExceeded("JSON node limit exceeded")
    if depth > limits.max_json_depth:
        raise BoundsExceeded("JSON nesting limit exceeded")
    if isinstance(value, dict):
        return {
            key: _bounded_json(item, depth=depth + 1, nodes=nodes, limits=limits)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _bounded_json(item, depth=depth + 1, nodes=nodes, limits=limits)
            for item in value
        ]
    return value


def read_bounded_json(
    path: str | os.PathLike[str],
    *,
    limits: DiscoveryLimits | None = None,
    expected_identity: tuple[int, int, int] | None = None,
    expected_sha256: str | None = None,
) -> Any:
    """Read strict UTF-8 JSON from the same stable bytes used for hashing."""

    limits = limits or DiscoveryLimits()
    source = Path(path)
    try:
        raw, _ = read_stable_bytes(
            source,
            limits=limits,
            expected_identity=expected_identity,
            expected_sha256=expected_sha256,
        )
        value = loads_strict(
            raw,
            max_depth=limits.max_json_depth,
            max_items=limits.max_json_nodes,
            max_bytes=limits.max_file_bytes,
        )
    except BoundsExceeded:
        raise
    except (AdapterError, ValueError) as exc:
        if isinstance(exc, AdapterError):
            raise
        raise InvalidJSON(f"invalid JSON: {source}") from exc
    # ``loads_strict`` bounds each container; this second pass bounds total
    # nodes as well, which matters for many shallow samples (E15/B39d).
    return _bounded_json(value, depth=0, nodes=[0], limits=limits)


@dataclass(frozen=True)
class SQLiteTableInventory:
    name: str
    columns: tuple[str, ...]
    row_count: int | None
    count_error: str | None = None


@dataclass(frozen=True)
class SQLiteSchemaInventory:
    path: Path
    tables: tuple[SQLiteTableInventory, ...]
    identity: Mapping[str, str]


def _quote_identifier(identifier: str) -> str:
    if not isinstance(identifier, str) or not identifier or "\x00" in identifier:
        raise AdapterError("invalid SQLite identifier")
    return '"' + identifier.replace('"', '""') + '"'


class SQLiteReadOnlyAdapter:
    """Inventory and read known record tables without mutation or free SQL."""

    def __init__(self, path: str | os.PathLike[str], *, limits: DiscoveryLimits | None = None):
        self.path = Path(path)
        self.limits = limits or DiscoveryLimits()
        self._connection: sqlite3.Connection | None = None
        self._initial_identity: tuple[int, int, int] | None = None
        self._sidecar_identities: dict[str, tuple[int, int, int] | None] = {}
        self._transaction_open = False

    def __enter__(self) -> "SQLiteReadOnlyAdapter":
        st = _check_regular(self.path)
        self._initial_identity = _identity(st)
        if st.st_size > self.limits.max_file_bytes:
            raise BoundsExceeded(f"SQLite source exceeds {self.limits.max_file_bytes} bytes")
        uri = f"file:{quote(self.path.resolve().as_posix(), safe='/')}?mode=ro"
        self._connection = sqlite3.connect(uri, uri=True)
        self._connection.execute("PRAGMA query_only=ON")
        # query_only is verified rather than merely requested.
        if self._connection.execute("PRAGMA query_only").fetchone()[0] != 1:
            self.close()
            raise AdapterError("SQLite query_only could not be enabled")
        self._sidecar_identities = {
            suffix: self._optional_identity(Path(str(self.path) + suffix))
            for suffix in ("-wal", "-shm", "-journal")
        }
        self._connection.execute("BEGIN")
        self._transaction_open = True
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise AdapterError("SQLite adapter must be used as a context manager")
        return self._connection

    def close(self) -> None:
        changed: Exception | None = None
        if self._initial_identity is not None:
            try:
                current = _identity(self.path.lstat())
                if current != self._initial_identity:
                    changed = AdapterError("SQLite source changed during read")
            except OSError as exc:
                changed = AdapterError("SQLite source disappeared during read")
                changed.__cause__ = exc
            for suffix, expected in self._sidecar_identities.items():
                actual = self._optional_identity(Path(str(self.path) + suffix))
                if actual != expected:
                    changed = AdapterError(f"SQLite sidecar changed during read: {suffix}")
        if self._connection is not None:
            if self._transaction_open:
                try:
                    self._connection.rollback()
                except sqlite3.DatabaseError:
                    pass
            self._connection.close()
            self._connection = None
        self._initial_identity = None
        self._sidecar_identities = {}
        self._transaction_open = False
        if changed is not None:
            raise changed

    @staticmethod
    def _optional_identity(path: Path) -> tuple[int, int, int] | None:
        try:
            return _identity(path.lstat())
        except FileNotFoundError:
            return None

    def schema_inventory(self) -> SQLiteSchemaInventory:
        conn = self.connection
        names = [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        tables: list[SQLiteTableInventory] = []
        for name in names:
            columns = tuple(
                str(row[1])
                for row in conn.execute(f"PRAGMA table_info({_quote_identifier(name)})")
            )
            try:
                count = int(conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(name)}").fetchone()[0])
                count_error = None
            except sqlite3.DatabaseError as exc:
                count, count_error = None, type(exc).__name__
            tables.append(SQLiteTableInventory(name, columns, count, count_error))
        identity: dict[str, str] = {}
        for table in ("db_identity", "metadata", "evidence_metadata"):
            if table not in names:
                continue
            try:
                if table == "db_identity":
                    rows = self._read_metadata_rows(table, limit=16)
                else:
                    rows = self._read_metadata_rows(table, limit=16)
            except (AdapterError, sqlite3.DatabaseError):
                continue
            for row in rows:
                if "key" in row and "value" in row:
                    identity[str(row["key"])] = str(row["value"])
                elif "schema_version" in row:
                    identity["schema_version"] = str(row["schema_version"])
        return SQLiteSchemaInventory(self.path, tuple(tables), dict(sorted(identity.items())))

    # ``inventory`` is the concise public spelling used by audit callers.
    inventory = schema_inventory

    def _read_metadata_rows(self, table: str, *, limit: int) -> tuple[dict[str, Any], ...]:
        """Read metadata for inventory only; metadata is never an import row."""

        cursor = self.connection.execute(f"SELECT * FROM {_quote_identifier(table)} LIMIT ?", (limit,))
        columns = tuple(item[0] for item in cursor.description or ())
        return tuple(dict(zip(columns, row)) for row in cursor.fetchall())

    def known_rows(self, table: str, *, limit: int | None = None) -> tuple[dict[str, Any], ...]:
        """Read rows only from a fixed, recognized evidence table."""

        allowed = {
            "records",
            "evidence_records",
            "runs",
            "bundles",
        }
        if table not in allowed:
            raise AdapterError(f"SQLite table is not a known safe adapter: {table}")
        max_rows = min(limit or self.limits.max_sqlite_rows, self.limits.max_sqlite_rows)
        if max_rows <= 0:
            raise ValueError("limit must be positive")
        names = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
            )
        }
        if not names:
            return ()
        # The table name is allowlisted above; no caller-provided WHERE/SQL is
        # accepted.  LIMIT is bound as a parameter.
        cursor = self.connection.execute(f"SELECT * FROM {_quote_identifier(table)} LIMIT ?", (max_rows,))
        columns = tuple(item[0] for item in cursor.description or ())
        return tuple(dict(zip(columns, row)) for row in cursor.fetchall())

    def records(self, *, limit: int | None = None) -> tuple[dict[str, Any], ...]:
        """Read the one known record family present in this database."""

        for table in ("evidence_records", "records", "runs", "bundles"):
            rows = self.known_rows(table, limit=limit)
            if rows:
                return tuple(self._decode_known_row(table, row) for row in rows)
        return ()

    def _decode_known_row(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        """Decode only schema-known JSON payload columns strictly."""

        payload_columns = {
            "evidence_records": ("report_json",),
            "records": ("payload_json", "report_json"),
            "runs": ("manifest_json",),
            "bundles": ("result_json", "manifest_json"),
        }[table]
        result = dict(row)
        errors: list[str] = []
        for column in payload_columns:
            raw = row.get(column)
            if raw is None:
                continue
            if not isinstance(raw, str):
                errors.append(f"{column}:not_text")
                continue
            try:
                result["payload"] = loads_strict(
                    raw,
                    max_bytes=self.limits.max_file_bytes,
                    max_depth=self.limits.max_json_depth,
                    max_items=self.limits.max_json_nodes,
                )
                break
            except ValueError as exc:
                errors.append(f"{column}:{type(exc).__name__}")
        if errors:
            result["_payload_decode_error"] = ";".join(errors)
        return result


# Compatibility spellings kept intentionally thin; all routes retain the same
# read-only allowlist and bounds.
SQLiteAdapter = SQLiteReadOnlyAdapter
read_only_discover = discover_files
canonical_json_bytes = canonical_bytes
strict_json_loads = loads_strict


__all__ = [
    "AdapterError",
    "BoundsExceeded",
    "DEFAULT_DISCOVERY_ROOTS",
    "KNOWN_NON_EVIDENCE_PREFIXES",
    "KNOWN_NON_EVIDENCE_SUFFIXES",
    "KNOWN_NON_EVIDENCE_PATH_PREFIXES",
    "DiscoveredFile",
    "DiscoveryLimits",
    "DuplicateJSONKey",
    "InvalidJSON",
    "SQLiteReadOnlyAdapter",
    "SQLiteAdapter",
    "SQLiteSchemaInventory",
    "SQLiteTableInventory",
    "canonical_json_bytes",
    "discover_files",
    "read_bounded_json",
    "read_stable_bytes",
    "read_only_discover",
    "strict_json_loads",
]
