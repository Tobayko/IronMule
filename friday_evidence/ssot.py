"""Lossless evidence corpus with canonical records and explicit source occurrences.

Original files, trace-buffer aliases and reachable historical Git data are preserved
by content hash. Repeated exports share verified record identity; independent runs
are never collapsed merely because their values agree. This is a local, regenerable
corpus, not authority to qualify a runtime path or rewrite sealed source evidence.

Attribution retains its evidence label. An unmarked commit establishes no AI author.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import math
import os
import re
import sqlite3
import subprocess
import statistics
import tempfile
import time
import zlib
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SSOT_PATH = PROJECT_ROOT / ".friday-data" / "ssot.sqlite3"
SCHEMA_VERSION = 2
SSOT_APPLICATION_ID = 0x46535354  # ASCII "FSST"

SCAN_ROOTS = (".friday-data", "research", "experiments", "profiles")
EXTRA_FILES = ("docs/EXPERIMENT_MATRIX.json",)
SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        ".worktrees",
        ".pytest_cache",
        ".ruff_cache",
        ".projectatlas",
        "__pycache__",
        "node_modules",
        "site-packages",
        ".ipynb_checkpoints",
        "ProjectAtlas",
    }
)
SKIP_RELATIVE_DIRS = frozenset({".friday-data/models", ".friday-data/tooling"})
DATA_SUFFIXES = frozenset({".json", ".jsonl", ".sqlite3", ".sqlite", ".db",
                           ".partial", ".csv", ".tsv", ".npy", ".npz", ".bin"})
SQLITE_SUFFIXES = frozenset({".sqlite3", ".sqlite", ".db"})
OBJECT_CHUNK_BYTES = 1 << 20

INLINE_PAYLOAD_LIMIT = 1 << 20  # payloads above this are addressed by source path + hash
ARRAY_SUMMARY_THRESHOLD = 32
MAX_METRICS_PER_RUN = 20_000
TIME_KEYS = (
    "observed_at",
    "exported_at",
    "generated_at",
    "created_utc",
    "created_at",
    "recorded_at",
    "started_at",
    "timestamp",
)
STATUS_KEYS = ("status", "result_status", "state", "decision", "classification", "formal_claim")
KIND_KEYS = ("schema", "experiment_id", "kind", "record_kind", "phase", "mode", "tool")
AGENT_KEYS = ("agent", "produced_by", "measured_by", "author_agent", "agent_id")
KNOWN_AGENTS = ("claude", "codex", "gemini")
COMMIT_PATTERN = re.compile(r"\b[0-9a-f]{40}\b")
PAYLOAD_SCAN_LIMIT = 1 << 16
ARCHIVE_MANIFEST = Path(".friday-data/ironmule-evidence-archive/MANIFEST.json")
JOURNAL_EVENT_FIELDS = frozenset({"seq", "kind", "payload", "sha256", "prev_sha256"})

SCHEMA = """
CREATE TABLE ssot_meta (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    schema_version INTEGER NOT NULL,
    built_at_unix_ns INTEGER NOT NULL,
    project_root TEXT NOT NULL,
    builder_sha256 TEXT NOT NULL
);
CREATE TABLE source_objects (
    sha256 TEXT PRIMARY KEY,
    bytes INTEGER NOT NULL,
    chunks INTEGER NOT NULL
);
CREATE TABLE source_chunks (
    sha256 TEXT NOT NULL REFERENCES source_objects(sha256),
    ordinal INTEGER NOT NULL,
    compressed BLOB NOT NULL,
    PRIMARY KEY (sha256, ordinal)
) WITHOUT ROWID;
CREATE TABLE sources (
    source_id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    shape TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL REFERENCES source_objects(sha256),
    mtime_unix_ns INTEGER NOT NULL,
    run_count INTEGER NOT NULL,
    agent TEXT NOT NULL,
    agent_source TEXT NOT NULL,
    error TEXT,
    link_target TEXT,
    origin TEXT NOT NULL DEFAULT 'filesystem',
    git_blob TEXT
);
CREATE TABLE payloads (
    sha256 TEXT PRIMARY KEY,
    json TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    format TEXT NOT NULL CHECK(format IN ('json', 'legacy_nonfinite_json'))
);
CREATE TABLE canonical_records (
    run_id INTEGER PRIMARY KEY,
    canonical_key TEXT NOT NULL UNIQUE,
    source_id INTEGER NOT NULL REFERENCES sources(source_id),
    study TEXT NOT NULL,
    native_id TEXT NOT NULL,
    kind TEXT,
    status TEXT,
    observed_at_unix_ns INTEGER,
    payload_sha256 TEXT NOT NULL REFERENCES payloads(sha256),
    payload_bytes INTEGER NOT NULL,
    provenance_json TEXT,
    agent TEXT NOT NULL,
    agent_detail TEXT,
    agent_source TEXT NOT NULL,
    metrics_truncated INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_runs_study ON canonical_records(study, observed_at_unix_ns DESC);
CREATE INDEX idx_runs_status ON canonical_records(status);
CREATE INDEX idx_runs_agent ON canonical_records(agent, study);
CREATE TABLE record_occurrences (
    source_id INTEGER NOT NULL REFERENCES sources(source_id),
    native_id TEXT NOT NULL,
    run_id INTEGER NOT NULL REFERENCES canonical_records(run_id),
    provenance_json TEXT,
    agent TEXT NOT NULL,
    agent_detail TEXT,
    agent_source TEXT NOT NULL,
    PRIMARY KEY (source_id, native_id)
) WITHOUT ROWID;
CREATE VIEW runs AS
SELECT r.*, CASE WHEN p.bytes <= 1048576 THEN p.json ELSE NULL END AS payload_json,
       p.format AS payload_format
FROM canonical_records r JOIN payloads p ON p.sha256 = r.payload_sha256;
CREATE VIEW origins AS
SELECT o.run_id, s.path AS source_path, o.native_id, o.provenance_json,
       o.agent, o.agent_detail, o.agent_source
FROM record_occurrences o JOIN sources s ON s.source_id = o.source_id;
CREATE TABLE metrics (
    run_id INTEGER NOT NULL REFERENCES canonical_records(run_id),
    path TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (run_id, path)
) WITHOUT ROWID;
CREATE INDEX idx_metrics_path ON metrics(path, value);
CREATE VIEW measurement AS
SELECT r.run_id, r.study, r.native_id, r.kind, r.status, r.agent, r.agent_source,
       r.observed_at_unix_ns, s.path AS source_path, m.path AS metric, m.value
FROM runs r
JOIN sources s ON s.source_id = r.source_id
JOIN metrics m ON m.run_id = r.run_id;
"""


class SsotError(RuntimeError):
    """The index could not be built or read."""


@dataclass
class Run:
    native_id: str
    payload: object
    kind: str | None = None
    status: str | None = None
    observed_at_unix_ns: int | None = None
    provenance: object | None = None
    extra_metrics: dict[str, float] = field(default_factory=dict)
    identity: str | None = None


@dataclass(frozen=True)
class Attribution:
    """Who produced a measurement, and what evidence says so."""

    agent: str
    detail: str | None
    source: str


UNATTRIBUTED = Attribution("unattributed", None, "no_evidence")


# --------------------------------------------------------------------------- helpers


def _read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path), safe='/')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA trusted_schema=OFF")
    connection.row_factory = sqlite3.Row
    return connection


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _payload_format(value: object) -> str:
    """Preserve historical infinities/test vectors while labeling their encoding."""

    try:
        json.dumps(value, allow_nan=False)
        return "json"
    except ValueError:
        return "legacy_nonfinite_json"


def _parse_time(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        # Heuristic: nanoseconds since epoch stay far above any plausible second count.
        return value if value > 10**14 else value * 1_000_000_000
    if isinstance(value, float) and math.isfinite(value):
        return int(value * 1_000_000_000)
    if isinstance(value, str):
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1e9)
        except ValueError:
            return None
    return None


def _first(document: object, keys: Sequence[str]) -> str | None:
    if not isinstance(document, dict):
        return None
    for key in keys:
        value = document.get(key)
        if isinstance(value, str) and value:
            return value[:200]
    return None


def _document_time(document: object) -> int | None:
    if not isinstance(document, dict):
        return None
    for key in TIME_KEYS:
        for candidate in (key, f"{key}_unix_ns", f"{key}_ns", f"{key}_utc"):
            if candidate in document:
                parsed = _parse_time(document[candidate])
                if parsed is not None:
                    return parsed
    return None


def flatten_metrics(
    value: object, prefix: str = "", out: dict[str, float] | None = None
) -> dict[str, float]:
    """Every finite numeric leaf of a document, addressed by its dotted path."""

    out = {} if out is None else out
    if len(out) >= MAX_METRICS_PER_RUN:
        return out
    if isinstance(value, bool):
        out[prefix or "value"] = float(value)
    elif isinstance(value, (int, float)) and math.isfinite(value):
        out[prefix or "value"] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            flatten_metrics(child, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(value, (list, tuple)):
        numbers = [
            item for item in value if isinstance(item, (int, float)) and not isinstance(item, bool)
        ]
        if len(value) > ARRAY_SUMMARY_THRESHOLD:
            out[f"{prefix}.count"] = float(len(value))
            if len(numbers) == len(value) and numbers:
                out.update(summarise(prefix, numbers))
                return out
            # ponytail: long non-numeric arrays are indexed only to the threshold;
            # the full payload stays reachable through runs.payload_json / source_path.
            value = list(value)[:ARRAY_SUMMARY_THRESHOLD]
        for index, item in enumerate(value):
            flatten_metrics(item, f"{prefix}[{index}]" if prefix else f"[{index}]", out)
    return out


def summarise(prefix: str, numbers: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(item) for item in numbers if math.isfinite(item))
    if not ordered:
        return {}
    return {
        f"{prefix}.count": float(len(ordered)),
        f"{prefix}.min": ordered[0],
        f"{prefix}.max": ordered[-1],
        f"{prefix}.mean": statistics.fmean(ordered),
        f"{prefix}.median": statistics.median(ordered),
    }


def _load_json(text: str) -> object:
    return json.loads(text)


# --------------------------------------------------------------------------- adapters


def _column(row: sqlite3.Row, *names: str) -> object | None:
    """The first present, non-null column; study tables renamed these over time."""

    keys = set(row.keys())
    for name in names:
        if name in keys and row[name] is not None:
            return row[name]
    return None


def _records_runs(connection: sqlite3.Connection) -> Iterator[Run]:
    for row in connection.execute("SELECT * FROM records"):
        provenance = _column(row, "provenance_json")
        identifier = _column(row, "record_id", "entity_key", "seq")
        yield Run(
            native_id=str(identifier),
            payload=_load_json(str(_column(row, "report_json", "payload_json") or "")),
            kind=_column(row, "record_kind", "kind"),
            status=_column(row, "status", "formal_claim"),
            observed_at_unix_ns=_column(row, "created_at_unix_ns", "recorded_at_unix_ns"),
            provenance=_load_json(str(provenance)) if provenance is not None else None,
        )


def _evidence_runs(connection: sqlite3.Connection) -> Iterator[Run]:
    for row in connection.execute("SELECT * FROM evidence_records"):
        yield Run(
            native_id=row["record_id"],
            payload=_load_json(row["report_json"]),
            kind=row["tool"],
            status=row["result_status"],
            observed_at_unix_ns=row["observed_at_unix_ns"] or row["recorded_at_unix_ns"],
            provenance={
                "record": _load_json(row["provenance_json"]),
                "git_revision": row["git_revision"],
                "git_dirty": row["git_dirty"],
                "code_sha256": row["code_sha256"],
                "environment_sha256": row["environment_sha256"],
            },
        )


def _bundle_runs(connection: sqlite3.Connection) -> Iterator[Run]:
    for row in connection.execute("SELECT * FROM bundles"):
        yield Run(
            native_id=row["entity_id"],
            payload=_load_json(row["bundle_json"]),
            kind=row["entity_kind"],
            status=row["status"],
            observed_at_unix_ns=row["created_at_unix_ns"],
            provenance=_load_json(row["lineage_json"]),
        )


def _optimization_runs(connection: sqlite3.Connection) -> Iterator[Run]:
    for row in connection.execute("SELECT * FROM optimization_records"):
        payload = row["payload"]
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        yield Run(
            native_id=row["record_id"],
            payload=_load_json(payload),
            kind=row["kind"],
            status=row["quality"],
            observed_at_unix_ns=_parse_time(row["created_at"]),
            provenance={"phase": row["phase"], "record_hash": row["record_hash"]},
        )


def _journal_runs(connection: sqlite3.Connection) -> Iterator[Run]:
    for row in connection.execute("SELECT * FROM event_journal"):
        payload = _load_json(row["payload_json"])
        yield Run(
            native_id=f"{row['run_id']}:{row['seq']}",
            payload=payload,
            kind=row["kind"],
            status=_first(payload, STATUS_KEYS),
            observed_at_unix_ns=row["recorded_unix_ns"],
            provenance={"sha256": row["sha256"], "prev_sha256": row["prev_sha256"]},
            identity=_journal_identity({**dict(row), "payload": payload}),
        )


def _journal_identity(event: dict) -> str:
    """Deduplicate an event only after recomputing its actual content identity."""

    from .canonical import canonical_sha256

    body = {key: event[key] for key in
            ("seq", "recorded_unix_ns", "run_id", "kind", "payload", "prev_sha256")}
    if canonical_sha256(body) != event["sha256"]:
        raise SsotError("journal event digest does not match its contents")
    return "journal:" + event["sha256"]


def _journal_export_runs(document: object) -> list[Run] | None:
    """A hash-chained journal exported to JSON, read at the same grain as its table."""

    events = document.get("events") if isinstance(document, dict) else None
    if not isinstance(events, list) or not events:
        return None
    if not all(isinstance(event, dict) and JOURNAL_EVENT_FIELDS <= set(event) for event in events):
        return None
    return [
        Run(
            native_id=f"{event.get('run_id') or 'run'}:{event['seq']}",
            payload=event["payload"],
            kind=event["kind"],
            status=_first(event["payload"], STATUS_KEYS),
            observed_at_unix_ns=_parse_time(event.get("recorded_unix_ns")),
            provenance={
                "sha256": event["sha256"],
                "prev_sha256": event["prev_sha256"],
                "journal": event.get("source_id"),
            },
            identity=_journal_identity(event),
        )
        for event in events
    ]


def _h0_runs(connection: sqlite3.Connection) -> Iterator[Run]:
    statuses = {
        row["run_id"]: row["status"]
        for row in connection.execute(
            "SELECT run_id, status FROM status_events ORDER BY event_id"
        )
    }
    metrics: dict[str, dict[str, float]] = {}
    for row in connection.execute(
        "SELECT run_id, scope, metric_name, value FROM scalar_metrics WHERE value IS NOT NULL"
    ):
        metrics.setdefault(row["run_id"], {})[f"{row['scope']}.{row['metric_name']}"] = float(
            row["value"]
        )
    for row in connection.execute(
        "SELECT run_id, case_name, metric_name, value, passed FROM correctness_metrics"
    ):
        bucket = metrics.setdefault(row["run_id"], {})
        base = f"correctness.{row['case_name']}.{row['metric_name']}"
        bucket[base] = float(row["value"])
        bucket[f"{base}.passed"] = float(row["passed"])
    samples: dict[str, dict[str, list[float]]] = {}
    for row in connection.execute(
        "SELECT run_id, sample_kind, arm, value FROM raw_samples ORDER BY sample_id"
    ):
        samples.setdefault(row["run_id"], {}).setdefault(
            f"samples.{row['sample_kind']}.{row['arm']}", []
        ).append(float(row["value"]))
    for run_id, groups in samples.items():
        bucket = metrics.setdefault(run_id, {})
        for prefix, values in groups.items():
            bucket.update(summarise(prefix, values))
    for row in connection.execute("SELECT * FROM runs"):
        yield Run(
            native_id=row["run_id"],
            payload=_load_json(row["manifest_json"]),
            kind=row["mode"],
            status=statuses.get(row["run_id"]),
            observed_at_unix_ns=row["created_at_unix_ns"],
            provenance={
                "phase": row["phase"],
                "manifest_hash": row["manifest_hash"],
                "code_sha256": row["code_sha256"],
                "spec_sha256": row["spec_sha256"],
                "environment_sha256": row["environment_sha256"],
                "revision": row["revision"],
            },
            extra_metrics=metrics.get(row["run_id"], {}),
        )


SQLITE_ADAPTERS = (
    ("h0_study", {"runs", "scalar_metrics"}, _h0_runs),
    ("evidence_records", {"evidence_records"}, _evidence_runs),
    ("bundles", {"bundles"}, _bundle_runs),
    ("optimization_records", {"optimization_records"}, _optimization_runs),
    ("event_journal", {"event_journal"}, _journal_runs),
    ("records", {"records"}, _records_runs),
)


def _sqlite_source(path: Path) -> tuple[str, list[Run]]:
    connection = _read_only(path)
    try:
        tables = {
            row["name"]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for shape, required, adapter in SQLITE_ADAPTERS:
            if required <= tables:
                return shape, list(adapter(connection))
    finally:
        connection.close()
    raise SsotError("unsupported evidence database schema")


def _json_source(path: Path) -> tuple[str, list[Run]]:
    if path.suffix == ".jsonl":
        runs = []
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if line:
                    runs.append(_json_run(f"{path.stem}:{number}", _load_json(line)))
        return "jsonl", runs
    document = _load_json(path.read_text(encoding="utf-8"))
    events = _journal_export_runs(document)
    if events is None:
        return "json", [_json_run(path.stem, document)]
    # The source object retains the complete envelope. Indexing it as another
    # measurement would count every embedded event's metrics a second time.
    return "journal_export", events


def _json_run(native_id: str, document: object) -> Run:
    return Run(
        native_id=native_id,
        payload=document,
        kind=_first(document, KIND_KEYS),
        status=_first(document, STATUS_KEYS),
        observed_at_unix_ns=_document_time(document),
    )


# --------------------------------------------------------------------------- agents


def _git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def _attribute_commit(sha: str, subject: str, body: str) -> Attribution:
    authors = {}
    for line in body.splitlines():
        if not line.lower().startswith("co-authored-by:"):
            continue
        name = line.split(":", 1)[1].split("<")[0].strip()
        for agent in KNOWN_AGENTS:
            if re.search(r"\b" + agent + r"\b", name.lower()):
                authors[agent] = name
    if len(authors) == 1:
        agent, name = next(iter(authors.items()))
        return Attribution(agent, name, "commit_trailer")
    if authors:
        return Attribution("multiple", ", ".join(sorted(authors)), "commit_trailer")
    head = subject[:60]
    if "(" in head and ")" in head:
        scope = head[head.index("(") + 1 : head.index(")")].strip().lower()
        if scope in KNOWN_AGENTS:
            return Attribution(scope, subject[:160], "commit_scope")
    return UNATTRIBUTED


def commit_log(root: Path = PROJECT_ROOT) -> list[tuple[str, int, Attribution]]:
    """Every commit as (hash, author time in ns, attribution), oldest first."""

    text = _git(root, "log", "--all", "--format=%H%x1f%at%x1f%s%x1f%b%x1e")
    entries: list[tuple[str, int, Attribution]] = []
    for entry in text.split("\x1e"):
        entry = entry.strip("\n")
        if not entry:
            continue
        parts = entry.split("\x1f")
        if len(parts) != 4:
            continue
        sha, timestamp, subject, body = parts
        sha = sha.strip()
        try:
            authored = int(timestamp) * 1_000_000_000
        except ValueError:
            continue
        entries.append((sha, authored, _attribute_commit(sha, subject, body)))
    entries.sort(key=lambda item: item[1])
    return entries


def commit_attributions(root: Path = PROJECT_ROOT) -> dict[str, Attribution]:
    """Attribution for every commit in the repository, keyed by full hash."""

    return {sha: attribution for sha, _, attribution in commit_log(root)}


def tracked_commits(root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Newest commit that touched each tracked path."""

    text = _git(root, "log", "--all", "--date-order", "--name-only", "--format=%x1ecommit %H")
    newest: dict[str, str] = {}
    for entry in text.split("\x1e"):
        lines = [line for line in entry.splitlines() if line.strip()]
        if not lines or not lines[0].startswith("commit "):
            continue
        sha = lines[0].split(" ", 1)[1].strip()
        for path in lines[1:]:
            newest.setdefault(path, sha)
    return newest


def _archive_heads(root: Path) -> dict[str, str]:
    manifest = root / ARCHIVE_MANIFEST
    if not manifest.is_file():
        return {}
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    base = ARCHIVE_MANIFEST.parent
    heads: dict[str, str] = {}
    for entry in document.get("entries", []):
        stored = entry.get("stored_as")
        head = entry.get("source_head")
        if isinstance(stored, str) and isinstance(head, str):
            heads[str(base / stored)] = head
    return heads


class Attributor:
    """Resolves the producing agent for a source file and for individual runs."""

    def __init__(self, root: Path = PROJECT_ROOT) -> None:
        self.root = root
        self.log = commit_log(root)
        self.commits = {sha: attribution for sha, _, attribution in self.log}
        self.timeline = [timestamp for _, timestamp, _ in self.log]
        self.tracked = tracked_commits(root)
        self.archive = _archive_heads(root)

    def at_time(self, observed_at_unix_ns: int | None) -> Attribution:
        """The agent whose commit was newest when the run was recorded.

        Weaker than a referenced commit: it says who was working, not who measured.
        """

        if observed_at_unix_ns is None or not self.timeline:
            return UNATTRIBUTED
        index = bisect.bisect_right(self.timeline, observed_at_unix_ns) - 1
        if index < 0:
            return UNATTRIBUTED
        found = self.log[index][2]
        return Attribution(found.agent, found.detail, "commit_at_time")

    def for_source(self, relative_path: str) -> Attribution:
        head = self.archive.get(relative_path)
        if head and head in self.commits:
            found = self.commits[head]
            return Attribution(found.agent, found.detail, "archive_manifest")
        sha = self.tracked.get(relative_path)
        if sha and sha in self.commits:
            found = self.commits[sha]
            return Attribution(found.agent, found.detail, "git_history")
        return UNATTRIBUTED

    def for_run(
        self, run: Run, fallback: Attribution, written_at_unix_ns: int | None = None
    ) -> Attribution:
        for document in (run.provenance, run.payload):
            declared = _first(document, AGENT_KEYS)
            if declared:
                return Attribution(declared.lower(), declared, "payload_field")
        for text in (
            _canonical(run.provenance) if run.provenance is not None else "",
            _canonical(run.payload)[:PAYLOAD_SCAN_LIMIT],
        ):
            for sha in COMMIT_PATTERN.findall(text):
                found = self.commits.get(sha)
                if found:
                    return Attribution(found.agent, found.detail, "provenance_commit")
        if fallback.agent != UNATTRIBUTED.agent:
            return fallback
        timed = self.at_time(run.observed_at_unix_ns)
        if timed.agent != UNATTRIBUTED.agent:
            return timed
        written = self.at_time(written_at_unix_ns)
        if written.agent == UNATTRIBUTED.agent:
            return written
        return Attribution(written.agent, written.detail, "file_mtime")


# --------------------------------------------------------------------------- build


def discover(root: Path) -> list[Path]:
    """Inventory data and archived inputs without descending into installed tools."""

    root = root.resolve()
    found: set[Path] = set()
    skip_dirs = {root / item for item in SKIP_RELATIVE_DIRS}
    for name in SCAN_ROOTS:
        base = root / name
        if not base.is_dir():
            continue
        for directory, dirs, names in os.walk(base):
            parent = Path(directory)
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIR_NAMES
                             and parent / d not in skip_dirs
                             and not (parent / d).is_symlink())
            for filename in names:
                path = parent / filename
                if filename == ".DS_Store" or path.suffix in {".lock", ".building"}:
                    continue
                if filename.endswith(("-wal", "-shm", "-journal")):
                    continue
                if name != ".friday-data" and path.suffix not in DATA_SUFFIXES:
                    continue
                if path.is_symlink():
                    target = path.resolve()
                    if not target.is_relative_to(root) or any(d in target.parents for d in skip_dirs):
                        raise SsotError(f"evidence alias leaves the evidence roots: {path.relative_to(root)}")
                    if not path.exists():
                        raise SsotError(f"evidence alias is broken: {path.relative_to(root)}")
                if not path.is_file():
                    continue
                if path.suffix in SQLITE_SUFFIXES and _is_ssot(path):
                    continue
                found.add(path)
    for name in EXTRA_FILES:
        candidate = root / name
        if candidate.is_file():
            found.add(candidate)
    return sorted(found)


def _source_fingerprints(paths: Iterable[Path]) -> dict[Path, tuple[str, int, str | None]]:
    """Hash shared trace-buffer targets once while retaining every original alias."""

    cache = {}
    result = {}
    for path in paths:
        info = path.stat()
        identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if identity not in cache:
            cache[identity] = _sha256_file(path)
        result[path] = (*cache[identity], os.readlink(path) if path.is_symlink() else None)
    return result


def _is_ssot(path: Path) -> bool:
    connection = None
    try:
        connection = _read_only(path)
        return connection.execute("PRAGMA application_id").fetchone()[0] == SSOT_APPLICATION_ID
    except sqlite3.Error:
        return False
    finally:
        if connection is not None:
            connection.close()


def _store_blocks(connection: sqlite3.Connection, blocks: Iterable[bytes], digest: str, size: int) -> None:
    if connection.execute("SELECT 1 FROM source_objects WHERE sha256=?", (digest,)).fetchone():
        return
    connection.execute("INSERT INTO source_objects VALUES(?,?,0)", (digest, size))
    observed = hashlib.sha256()
    observed_bytes = 0
    chunks = 0
    for block in blocks:
        observed.update(block)
        observed_bytes += len(block)
        connection.execute("INSERT INTO source_chunks VALUES(?,?,?)",
                           (digest, chunks, zlib.compress(block)))
        chunks += 1
    if observed.hexdigest() != digest or observed_bytes != size:
        raise SsotError("source changed while preserving its bytes")
    connection.execute("UPDATE source_objects SET chunks=? WHERE sha256=?", (chunks, digest))


def _store_object(connection: sqlite3.Connection, path: Path, digest: str, size: int,
                  reuse_from: Path | None = None) -> None:
    """Retain original bytes once; reuse only independently verified compressed objects."""

    if connection.execute("SELECT 1 FROM source_objects WHERE sha256=?", (digest,)).fetchone():
        return
    if reuse_from is not None:
        row = connection.execute("SELECT bytes FROM previous.source_objects WHERE sha256=?", (digest,)).fetchone()
        if row is not None and row[0] == size:
            for _ in source_content(digest, reuse_from):
                pass
            connection.execute("INSERT INTO source_objects SELECT * FROM previous.source_objects WHERE sha256=?", (digest,))
            connection.execute("INSERT INTO source_chunks SELECT * FROM previous.source_chunks WHERE sha256=?", (digest,))
            return
    with path.open("rb") as handle:
        _store_blocks(connection, iter(lambda: handle.read(OBJECT_CHUNK_BYTES), b""), digest, size)


def _git_data_objects(root: Path) -> list[tuple[str, str]]:
    """Reachable historical data blobs, excluding tools, credentials and model weights."""

    if not (root / ".git").exists():
        return []
    try:
        listing = subprocess.check_output(
            ["git", "-C", str(root), "rev-list", "--objects", "--all"],
            text=True, stderr=subprocess.PIPE, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SsotError("cannot inventory historical Git data") from exc
    rows = []
    for line in listing.splitlines():
        oid, separator, relative = line.partition(" ")
        if not separator or not re.fullmatch(r"[0-9a-f]{40,64}", oid):
            continue
        if not (relative.startswith(("research/", "experiments/", "profiles/")) or relative in EXTRA_FILES):
            continue
        if Path(relative).suffix not in DATA_SUFFIXES:
            continue
        rows.append((oid, relative))
    return sorted(set(rows))


def _archive_scopes(root: Path) -> dict[str, set[str]]:
    """The existing archive manifest explicitly identifies copies of one artifact."""

    manifest = root / ARCHIVE_MANIFEST
    if not manifest.is_file():
        return {}
    scopes: dict[str, set[str]] = {}
    for entry in json.loads(manifest.read_text(encoding="utf-8"))["entries"]:
        names = scopes.setdefault(entry["sha256"], set())
        if "file" in entry:
            names.add(Path(entry["file"]).name)
        stored = entry.get("stored_as") or entry.get("duplicate_of")
        if stored:
            names.add(Path(stored).name)
    return scopes


def _ingest_git_history(connection: sqlite3.Connection, root: Path,
                        objects: Sequence[tuple[str, str]], attributor: Attributor) -> int:
    """Retain old versions as source occurrences, never as new repeated experiments."""

    for oid, relative in objects:
        content = subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", oid])
        digest = hashlib.sha256(content).hexdigest()
        _store_blocks(connection, iter(lambda h=io.BytesIO(content): h.read(OBJECT_CHUNK_BYTES), b""),
                      digest, len(content))
        original = connection.execute("SELECT source_id,run_count FROM sources WHERE path=? AND sha256=?",
                                      (relative, digest)).fetchone()
        # An unchanged Git blob is the very same current source, not another run.
        if original is not None:
            run_count = original["run_count"] if isinstance(original, sqlite3.Row) else original[1]
            runs = []
            shape = "git_alias"
        elif Path(relative).suffix in {".json", ".jsonl", ".partial"}:
            decoded = content.decode("utf-8")
            if Path(relative).suffix == ".jsonl":
                runs = [_json_run(f"{Path(relative).stem}:{number}", _load_json(line))
                        for number, line in enumerate(decoded.splitlines(), 1) if line.strip()]
            else:
                document = _load_json(decoded)
                runs = _journal_export_runs(document)
                if runs is None:
                    runs = [_json_run(Path(relative).stem, document)]
            run_count, shape = len(runs), "git_history"
        else:
            runs, run_count, shape = [], 0, "git_artifact"
        cursor = connection.execute(
            "INSERT INTO sources(path,shape,bytes,sha256,mtime_unix_ns,run_count,agent,agent_source,origin,git_blob)"
            " VALUES(?,?,?,?,0,?,'unattributed','git_blob','git',?)",
            (f"git:{oid}:{relative}", shape, len(content), digest, run_count, oid),
        )
        source_id = int(cursor.lastrowid)
        if original is not None:
            connection.execute(
                "INSERT INTO record_occurrences SELECT ?,native_id,run_id,provenance_json,agent,agent_detail,agent_source"
                " FROM record_occurrences WHERE source_id=?", (source_id, original[0]),
            )
        else:
            _insert_runs(connection, source_id, f"history/{Path(relative).parent}", runs,
                         attributor, UNATTRIBUTED, 0)
    return len(objects)


def _check_archive(connection: sqlite3.Connection, root: Path) -> int:
    manifest = root / ARCHIVE_MANIFEST
    if not manifest.is_file():
        return 0
    document = json.loads(manifest.read_text(encoding="utf-8"))
    entries = document["entries"]
    if len(entries) != document["total_entries"]:
        raise SsotError("historical archive entry count does not match its manifest")
    digests = set()
    for entry in entries:
        stored = entry.get("stored_as") or entry.get("duplicate_of")
        if not isinstance(stored, str):
            raise SsotError("historical archive entry has no stored source")
        path = (manifest.parent / stored).resolve()
        if not path.is_relative_to(manifest.parent.resolve()):
            raise SsotError("historical archive source escapes its directory")
        row = connection.execute("SELECT sha256, bytes FROM sources WHERE path=?",
                                 (path.relative_to(root).as_posix(),)).fetchone()
        if row is None or tuple(row) != (entry["sha256"], entry["bytes"]):
            raise SsotError(f"historical archive source is missing or changed: {stored}")
        digests.add(entry["sha256"])
    if len(digests) != document["unique_files"]:
        raise SsotError("historical archive unique-object count does not match its manifest")
    return len(entries)


def _study(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if relative.parts[0] == ".friday-data":
        return path.stem if len(relative.parts) == 2 else "/".join(relative.parts[1:-1])
    return "/".join(relative.parts[:-1]) or relative.parts[0]


def build(ssot_path: Path = DEFAULT_SSOT_PATH, root: Path = PROJECT_ROOT,
          *, reuse_from: Path | None = None) -> dict[str, object]:
    """Build a complete, lossless corpus and publish it only after validation."""

    root, ssot_path = root.resolve(), ssot_path.absolute()
    if ssot_path.is_symlink() or (ssot_path.exists() and not _is_ssot(ssot_path)):
        raise SsotError("output must be absent or an existing derived SSOT database")
    ssot_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    sources = discover(root)
    history = _git_data_objects(root)
    fingerprints = _source_fingerprints(sources)
    archive_scopes = _archive_scopes(root)
    attributor = Attributor(root)
    if reuse_from is not None:
        reuse_from = reuse_from.resolve()
        if not _is_ssot(reuse_from):
            raise SsotError("reuse input must be a derived SSOT corpus")
    fd, name = tempfile.mkstemp(prefix=ssot_path.name + ".", suffix=".building",
                                dir=ssot_path.parent)
    os.close(fd)
    temporary = Path(name)
    started = time.time_ns()
    builder_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    totals = {"sources": 0, "runs": 0, "occurrences": 0, "metrics": 0, "failed_sources": 0}
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(SCHEMA)
        connection.execute(f"PRAGMA application_id={SSOT_APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        if reuse_from is not None:
            connection.execute("ATTACH DATABASE ? AS previous",
                               (f"file:{quote(str(reuse_from), safe='/')}?mode=ro",))
        for source_path in sources:
            if source_path == ssot_path:
                continue
            for suffix in ("-wal", "-journal"):
                sidecar = Path(str(source_path) + suffix)
                if source_path.suffix in SQLITE_SUFFIXES and sidecar.exists() and sidecar.stat().st_size:
                    raise SsotError(f"source has an active SQLite journal: {source_path.name}")
            digest, size, link_target = fingerprints[source_path]
            _store_object(connection, source_path, digest, size, reuse_from)
            if source_path.suffix in SQLITE_SUFFIXES:
                shape, runs = _sqlite_source(source_path)
            elif source_path.suffix in {".json", ".jsonl"}:
                shape, runs = _json_source(source_path)
            elif source_path.suffix == ".partial":
                try:
                    shape, runs = _json_source(source_path)
                    shape = "partial_" + shape
                except (ValueError, UnicodeError):
                    shape, runs = "partial_artifact", []
            else:
                shape, runs = "artifact", []
            relative = source_path.relative_to(root).as_posix()
            source_agent = attributor.for_source(relative)
            cursor = connection.execute(
                "INSERT INTO sources(path, shape, bytes, sha256, mtime_unix_ns, run_count, agent,"
                " agent_source, error, link_target) VALUES(?,?,?,?,?,?,?,?,NULL,?)",
                (relative, shape, size, digest, source_path.stat().st_mtime_ns, len(runs),
                 source_agent.agent, source_agent.source, link_target),
            )
            source_id = int(cursor.lastrowid)
            totals["sources"] += 1
            totals["occurrences"] += len(runs)
            scope = "archive:" + digest if source_path.name in archive_scopes.get(digest, set()) else None
            _insert_runs(connection, source_id, _study(source_path, root), runs, attributor,
                         source_agent, source_path.stat().st_mtime_ns, source_scope=scope)
        totals["archive_entries"] = _check_archive(connection, root)
        totals["git_history_objects"] = _ingest_git_history(connection, root, history, attributor)
        if _git_data_objects(root) != history:
            raise SsotError("Git data history changed during the build")
        if set(discover(root)) != set(sources):
            raise SsotError("source inventory changed during the build")
        after = _source_fingerprints(sources)
        for path, expected in fingerprints.items():
            if after[path] != expected:
                raise SsotError(f"source changed during the build: {path.relative_to(root)}")
        totals["runs"] = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        totals["sources"] = connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        totals["occurrences"] = connection.execute("SELECT COUNT(*) FROM record_occurrences").fetchone()[0]
        totals["metrics"] = connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        totals["objects"] = connection.execute("SELECT COUNT(*) FROM source_objects").fetchone()[0]
        totals["duplicate_occurrences"] = totals["occurrences"] - totals["runs"]
        connection.execute(
            "INSERT INTO ssot_meta(singleton, schema_version, built_at_unix_ns, project_root,"
            " builder_sha256) VALUES(1,?,?,?,?)", (SCHEMA_VERSION, started, str(root), builder_sha),
        )
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise SsotError("the corpus has unresolved references")
        connection.commit()
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise SsotError("the corpus failed SQLite integrity verification")
        connection.close()
        os.replace(temporary, ssot_path)
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)
    totals["database"] = str(ssot_path)
    totals["build_seconds"] = round((time.time_ns() - started) / 1e9, 3)
    return totals


def _insert_runs(
    connection: sqlite3.Connection,
    source_id: int,
    study: str,
    runs: Iterable[Run],
    attributor: "Attributor",
    source_agent: Attribution,
    written_at_unix_ns: int,
    *, source_scope: str | None = None,
) -> None:
    seen: set[str] = set()
    for ordinal, run in enumerate(runs):
        native_id = run.native_id
        if native_id in seen:
            native_id = f"{native_id}#{len(seen)}"
        seen.add(native_id)
        observed_at = run.observed_at_unix_ns or _document_time(run.payload)
        payload = _canonical(run.payload)
        payload_bytes = len(payload.encode("utf-8"))
        payload_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        metrics = flatten_metrics(run.payload)
        metrics.update(run.extra_metrics)
        truncated = int(len(metrics) >= MAX_METRICS_PER_RUN)
        attribution = attributor.for_run(run, source_agent, written_at_unix_ns)
        provenance = _canonical(run.provenance) if run.provenance is not None else None
        identifier = (_first(run.payload, ("run_id", "record_id", "experiment_id"))
                      or (native_id if re.fullmatch(r"[0-9a-f]{64}", native_id) else None))
        if run.identity:
            key = run.identity
        elif identifier:
            # Equal values do not establish identity. Bind the original identifier
            # together with all measurement/provenance context, not its filename.
            identity = [identifier, run.kind, run.status, observed_at, payload_sha,
                        provenance, run.extra_metrics]
            key = "record:" + hashlib.sha256(_canonical(identity).encode()).hexdigest()
        elif source_scope:
            key = f"{source_scope}:{ordinal}"
        else:
            # Unknown identity remains a distinct observation; only its stored
            # content is deduplicated. Never erase a repetition based on its value.
            key = f"location:{source_id}:{native_id}"
        connection.execute("INSERT OR IGNORE INTO payloads VALUES(?,?,?,?)",
                           (payload_sha, payload, payload_bytes, _payload_format(run.payload)))
        cursor = connection.execute(
            "INSERT OR IGNORE INTO canonical_records(canonical_key, source_id, study, native_id,"
            " kind, status, observed_at_unix_ns, payload_sha256, payload_bytes, provenance_json,"
            " agent, agent_detail, agent_source, metrics_truncated) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (key, source_id, study, native_id, run.kind, run.status, observed_at,
             payload_sha, payload_bytes, provenance, attribution.agent, attribution.detail,
             attribution.source, truncated),
        )
        row = connection.execute("SELECT run_id,payload_sha256 FROM canonical_records WHERE canonical_key=?",
                                 (key,)).fetchone()
        run_id = row[0]
        if row[1] != payload_sha:
            raise SsotError("one canonical identity has conflicting payloads")
        connection.execute("INSERT INTO record_occurrences VALUES(?,?,?,?,?,?,?)",
                           (source_id, native_id, run_id, provenance, attribution.agent,
                            attribution.detail, attribution.source))
        if cursor.rowcount:
            connection.executemany("INSERT INTO metrics(run_id,path,value) VALUES(?,?,?)",
                                   ((run_id, path, value) for path, value in metrics.items()))



# --------------------------------------------------------------------------- read


def source_content(digest: str, ssot_path: Path = DEFAULT_SSOT_PATH) -> Iterator[bytes]:
    """Recover exact original bytes from the corpus, without requiring the old path."""

    connection = _read_only(ssot_path)
    try:
        expected = connection.execute("SELECT bytes,chunks FROM source_objects WHERE sha256=?",
                                      (digest,)).fetchone()
        if expected is None:
            raise SsotError("source object does not exist")
        observed, size, count = hashlib.sha256(), 0, 0
        for row in connection.execute("SELECT ordinal,compressed FROM source_chunks WHERE sha256=? ORDER BY ordinal",
                                      (digest,)):
            if row["ordinal"] != count:
                raise SsotError("source object has missing chunks")
            decoder = zlib.decompressobj()
            block = decoder.decompress(row["compressed"], OBJECT_CHUNK_BYTES + 1)
            if len(block) > OBJECT_CHUNK_BYTES or not decoder.eof or decoder.unused_data:
                raise SsotError("source object has an invalid compressed chunk")
            size += len(block)
            count += 1
            observed.update(block)
            yield block
        if (size, count, observed.hexdigest()) != (expected["bytes"], expected["chunks"], digest):
            raise SsotError("source object content does not match its identity")
    finally:
        connection.close()


def verify(ssot_path: Path = DEFAULT_SSOT_PATH, *, check_sources: bool = False) -> dict[str, object]:
    """Independently read every stored object and check references and optional originals."""

    connection = _read_only(ssot_path)
    try:
        meta = connection.execute("SELECT * FROM ssot_meta").fetchone()
        if meta is None or meta["schema_version"] != SCHEMA_VERSION:
            raise SsotError("SSOT schema is outdated; rebuild the derived corpus")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise SsotError("SQLite integrity check failed")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise SsotError("corpus contains unresolved references")
        objects = connection.execute("SELECT sha256 FROM source_objects ORDER BY sha256").fetchall()
        byte_count = 0
        for row in objects:
            for block in source_content(row["sha256"], ssot_path):
                byte_count += len(block)
        for row in connection.execute("SELECT sha256,json,bytes,format FROM payloads"):
            content = row["json"].encode("utf-8")
            if hashlib.sha256(content).hexdigest() != row["sha256"] or len(content) != row["bytes"]:
                raise SsotError("a canonical payload does not match its identity")
            if _payload_format(json.loads(row["json"])) != row["format"]:
                raise SsotError("a payload encoding label does not match its contents")
        missing = connection.execute(
            "SELECT s.path FROM sources s LEFT JOIN record_occurrences o USING(source_id)"
            " GROUP BY s.source_id HAVING COUNT(o.run_id) != s.run_count"
        ).fetchall()
        if missing:
            raise SsotError("source record coverage is incomplete")
        root = Path(meta["project_root"])
        checked = 0
        if check_sources:
            indexed = set()
            rows = connection.execute("SELECT path,sha256,bytes,link_target FROM sources WHERE origin='filesystem'").fetchall()
            current = _source_fingerprints(root / row["path"] for row in rows)
            for row in rows:
                path = root / row["path"]
                if current[path] != (row["sha256"], row["bytes"], row["link_target"]):
                    raise SsotError(f"original source changed: {row['path']}")
                indexed.add(path)
                checked += 1
            if indexed != set(discover(root)):
                raise SsotError("current source inventory differs from the corpus")
            _check_archive(connection, root)
            history = connection.execute("SELECT git_blob,sha256,bytes FROM sources WHERE origin='git'").fetchall()
            if {row['git_blob'] for row in history} != {oid for oid, _ in _git_data_objects(root)}:
                raise SsotError("Git data history differs from the corpus")
            for row in history:
                content = subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", row['git_blob']])
                if (hashlib.sha256(content).hexdigest(), len(content)) != (row['sha256'], row['bytes']):
                    raise SsotError("historical Git object differs from the corpus")
                checked += 1
        return {"state": "verified", "objects": len(objects), "original_bytes": byte_count,
                "original_sources_checked": checked,
                "runs": connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
                "occurrences": connection.execute("SELECT COUNT(*) FROM record_occurrences").fetchone()[0]}
    finally:
        connection.close()


def status(ssot_path: Path = DEFAULT_SSOT_PATH) -> dict[str, object]:
    with closing(_read_only(ssot_path)) as connection:
        meta = connection.execute("SELECT * FROM ssot_meta").fetchone()
        if meta is None or meta["schema_version"] != SCHEMA_VERSION:
            raise SsotError("SSOT schema is outdated; rebuild the derived corpus")
        recent = [dict(row) for row in connection.execute(
            "SELECT study,native_id,status,observed_at_unix_ns FROM runs "
            "ORDER BY observed_at_unix_ns DESC,run_id DESC LIMIT 8"
        )]
        studies = [
            dict(row)
            for row in connection.execute(
                "SELECT study, count(*) AS runs FROM runs GROUP BY study ORDER BY runs DESC"
            )
        ]
        shapes = [
            dict(row)
            for row in connection.execute(
                "SELECT shape, count(*) AS sources, sum(run_count) AS runs, sum(bytes) AS bytes"
                " FROM sources GROUP BY shape ORDER BY sources DESC"
            )
        ]
        failed = [
            dict(row)
            for row in connection.execute("SELECT path, error FROM sources WHERE error IS NOT NULL")
        ]
        agents = [
            dict(row)
            for row in connection.execute(
                "SELECT agent, agent_source, count(*) AS runs FROM runs"
                " GROUP BY agent, agent_source ORDER BY runs DESC"
            )
        ]
        return {
            "database": str(ssot_path),
            "schema_version": meta["schema_version"] if meta else None,
            "built_at_unix_ns": meta["built_at_unix_ns"] if meta else None,
            "sources": connection.execute("SELECT count(*) FROM sources").fetchone()[0],
            "runs": connection.execute("SELECT count(*) FROM runs").fetchone()[0],
            "objects": connection.execute("SELECT count(*) FROM source_objects").fetchone()[0],
            "original_bytes": connection.execute("SELECT sum(bytes) FROM source_objects").fetchone()[0],
            "occurrences": connection.execute("SELECT count(*) FROM record_occurrences").fetchone()[0],
            "metrics": connection.execute("SELECT count(*) FROM metrics").fetchone()[0],
            "distinct_metric_paths": connection.execute(
                "SELECT count(DISTINCT path) FROM metrics"
            ).fetchone()[0],
            "shapes": shapes,
            "agents": agents,
            "studies": studies,
            "recent": recent,
            "nonstandard_payloads": connection.execute(
                "SELECT COUNT(*) FROM payloads WHERE format != 'json'"
            ).fetchone()[0],
            "unreadable_sources": failed,
        }


def query(sql: str, ssot_path: Path = DEFAULT_SSOT_PATH, limit: int = 200) -> list[dict]:
    with closing(_read_only(ssot_path)) as connection:
        rows = connection.execute(sql).fetchmany(limit)
        return [dict(row) for row in rows]


def metric_paths(
    pattern: str | None = None, ssot_path: Path = DEFAULT_SSOT_PATH, limit: int = 50
) -> list[dict]:
    sql = (
        "SELECT m.path, count(*) AS runs, count(DISTINCT r.study) AS studies,"
        " min(m.value) AS min, max(m.value) AS max"
        " FROM metrics m JOIN runs r ON r.run_id = m.run_id"
    )
    parameters: tuple = ()
    if pattern:
        sql += " WHERE m.path LIKE ?"
        parameters = (f"%{pattern}%",)
    sql += " GROUP BY m.path ORDER BY runs DESC, m.path LIMIT ?"
    with closing(_read_only(ssot_path)) as connection:
        return [dict(row) for row in connection.execute(sql, parameters + (limit,))]


# --------------------------------------------------------------------------- CLI


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="friday ssot", allow_abbrev=False)
    parser.add_argument("--database", type=Path, default=DEFAULT_SSOT_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    building = commands.add_parser("build", allow_abbrev=False)
    building.add_argument("--reuse", type=Path, help="reuse verified source objects from an existing corpus")
    commands.add_parser("status", allow_abbrev=False)
    verification = commands.add_parser("verify", allow_abbrev=False)
    verification.add_argument("--check-sources", action="store_true")
    export = commands.add_parser("export-source", allow_abbrev=False)
    export.add_argument("sha256")
    export.add_argument("output", type=Path)
    sql = commands.add_parser("query", allow_abbrev=False)
    sql.add_argument("sql")
    sql.add_argument("--limit", type=int, default=200)
    names = commands.add_parser("metrics", allow_abbrev=False)
    names.add_argument("pattern", nargs="?")
    names.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "build":
            _print(build(args.database, reuse_from=args.reuse))
        elif args.command == "status":
            _print(status(args.database))
        elif args.command == "verify":
            _print(verify(args.database, check_sources=args.check_sources))
        elif args.command == "export-source":
            created = False
            try:
                fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                created = True
                with os.fdopen(fd, "wb") as handle:
                    for block in source_content(args.sha256, args.database):
                        handle.write(block)
            except Exception:
                if created:
                    args.output.unlink(missing_ok=True)
                raise
            _print({"state": "exported", "sha256": args.sha256, "output": str(args.output)})
        elif args.command == "query":
            _print(query(args.sql, args.database, args.limit))
        elif args.command == "metrics":
            _print(metric_paths(args.pattern, args.database, args.limit))
    except (SsotError, ValueError, UnicodeError, KeyError, zlib.error) as exc:
        _print({"state": "integrity_error", "detail": str(exc)})
        return 65
    except sqlite3.Error as exc:
        _print({"state": "sqlite_error", "detail": str(exc)})
        return 65
    except OSError as exc:
        _print({"state": "io_error", "detail": str(exc)})
        return 66
    return 0


__all__ = [
    "Attribution",
    "Attributor",
    "build",
    "commit_attributions",
    "commit_log",
    "discover",
    "flatten_metrics",
    "main",
    "metric_paths",
    "query",
    "status",
    "source_content",
    "verify",
]


if __name__ == "__main__":
    raise SystemExit(main())
