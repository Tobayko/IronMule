"""Single source of truth: one read-only SQLite index over every measurement source.

Sources stay untouched. Sealed study databases and raw JSON are opened read-only and
never rewritten; this module only builds a derived index next to them. Rebuild is a
full, atomic replace, so the index can always be thrown away and regenerated.

Every run carries the agent that produced it in `runs.agent`, with `runs.agent_source`
naming the evidence used, strongest first:

* `payload_field` — the measurement itself names its agent.
* `provenance_commit` — provenance references a commit; that commit's attribution wins.
* `archive_manifest` — the content-addressed archive records the producing commit.
* `git_history` — newest commit that touched the file.
* `commit_trailer` / `commit_scope` — `Co-Authored-By: Claude ...` / `feat(gemini): ...`.
* `repo_default_codex` — a commit with neither marker; this tree is Codex-driven
  (`AGENTS.md`), so it is attributed to Codex. That is an inference, not a receipt.
* `no_evidence` — nothing attributable; the run stays `unattributed`.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import statistics
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SSOT_PATH = PROJECT_ROOT / ".friday-data" / "ssot.sqlite3"
SCHEMA_VERSION = 1
SSOT_APPLICATION_ID = 0x46535354  # ASCII "FSST"

SCAN_ROOTS = (".friday-data", "research", "experiments")
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
SKIP_RELATIVE_DIRS = frozenset({".friday-data/models"})

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
CREATE TABLE sources (
    source_id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    shape TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    mtime_unix_ns INTEGER NOT NULL,
    run_count INTEGER NOT NULL,
    agent TEXT NOT NULL,
    agent_source TEXT NOT NULL,
    error TEXT
);
CREATE TABLE runs (
    run_id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(source_id),
    study TEXT NOT NULL,
    native_id TEXT NOT NULL,
    kind TEXT,
    status TEXT,
    observed_at_unix_ns INTEGER,
    payload_json TEXT,
    payload_sha256 TEXT NOT NULL,
    payload_bytes INTEGER NOT NULL,
    provenance_json TEXT,
    agent TEXT NOT NULL,
    agent_detail TEXT,
    agent_source TEXT NOT NULL,
    metrics_truncated INTEGER NOT NULL DEFAULT 0,
    UNIQUE(source_id, native_id)
);
CREATE INDEX idx_runs_study ON runs(study, observed_at_unix_ns DESC);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_agent ON runs(agent, study);
CREATE TABLE metrics (
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
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
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"unparsed_text": text[:4096]}


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
            payload = payload.decode("utf-8", "replace")
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
        )


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
    with _read_only(path) as connection:
        tables = {
            row["name"]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for shape, required, adapter in SQLITE_ADAPTERS:
            if required <= tables:
                return shape, list(adapter(connection))
    return "sqlite_unknown", []


def _json_source(path: Path) -> tuple[str, list[Run]]:
    if path.suffix == ".jsonl":
        runs = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if line:
                    runs.append(_json_run(f"{path.stem}:{number}", _load_json(line)))
        return "jsonl", runs
    document = _load_json(path.read_text(encoding="utf-8", errors="replace"))
    events = _journal_export_runs(document)
    if events is None:
        return "json", [_json_run(path.stem, document)]
    return "journal_export", [_json_run(path.stem, document), *events]


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
    for line in body.splitlines():
        lowered = line.lower()
        claimed = "claude" in lowered or "anthropic" in lowered
        if lowered.startswith("co-authored-by:") and claimed:
            name = line.split(":", 1)[1].split("<")[0].strip()
            return Attribution("claude", name or None, "commit_trailer")
    head = subject[:60]
    if "(" in head and ")" in head:
        scope = head[head.index("(") + 1 : head.index(")")].strip().lower()
        if scope in KNOWN_AGENTS:
            return Attribution(scope, subject[:160], "commit_scope")
    # ponytail: this tree is Codex-driven (AGENTS.md); an unmarked commit is Codex work.
    return Attribution("codex", sha[:12], "repo_default_codex")


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
    """Every measurement file the index covers, in stable order."""

    found: list[Path] = []
    skip_dirs = {root / item for item in SKIP_RELATIVE_DIRS}
    for name in SCAN_ROOTS:
        base = root / name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in {".json", ".jsonl", ".sqlite3"}:
                continue
            if SKIP_DIR_NAMES & set(path.parts):
                continue
            if any(parent in skip_dirs for parent in path.parents):
                continue
            found.append(path)
    for name in EXTRA_FILES:
        candidate = root / name
        if candidate.is_file():
            found.append(candidate)
    return sorted(set(found))


def _study(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if relative.parts[0] == ".friday-data":
        return path.stem if len(relative.parts) == 2 else "/".join(relative.parts[1:-1])
    return "/".join(relative.parts[:-1]) or relative.parts[0]


def build(ssot_path: Path = DEFAULT_SSOT_PATH, root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Rebuild the index from scratch and atomically replace the previous one."""

    ssot_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ssot_path.with_name(ssot_path.name + ".building")
    temporary.unlink(missing_ok=True)
    started = time.time_ns()
    builder_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    attributor = Attributor(root)
    totals = {"sources": 0, "runs": 0, "metrics": 0, "failed_sources": 0}

    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(SCHEMA)
        connection.execute(f"PRAGMA application_id={SSOT_APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        for source_path in discover(root):
            if source_path.resolve() == ssot_path.resolve():
                continue
            digest, size = _sha256_file(source_path)
            error: str | None = None
            try:
                shape, runs = (
                    _sqlite_source(source_path)
                    if source_path.suffix == ".sqlite3"
                    else _json_source(source_path)
                )
            except (sqlite3.Error, OSError, ValueError, RecursionError) as exc:
                shape, runs, error = "unreadable", [], f"{type(exc).__name__}: {exc}"[:400]
                totals["failed_sources"] += 1
            relative = str(source_path.relative_to(root))
            source_agent = attributor.for_source(relative)
            cursor = connection.execute(
                "INSERT INTO sources(path, shape, bytes, sha256, mtime_unix_ns, run_count, agent,"
                " agent_source, error) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    relative,
                    shape,
                    size,
                    digest,
                    source_path.stat().st_mtime_ns,
                    len(runs),
                    source_agent.agent,
                    source_agent.source,
                    error,
                ),
            )
            source_id = int(cursor.lastrowid)
            totals["sources"] += 1
            totals["runs"] += len(runs)
            totals["metrics"] += _insert_runs(
                connection,
                source_id,
                _study(source_path, root),
                runs,
                attributor,
                source_agent,
                source_path.stat().st_mtime_ns,
            )
        connection.execute(
            "INSERT INTO ssot_meta(singleton, schema_version, built_at_unix_ns, project_root,"
            " builder_sha256) VALUES(1,?,?,?,?)",
            (SCHEMA_VERSION, started, str(root), builder_sha),
        )
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, ssot_path)
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
) -> int:
    written = 0
    seen: set[str] = set()
    for run in runs:
        native_id = run.native_id
        if native_id in seen:
            native_id = f"{native_id}#{len(seen)}"
        seen.add(native_id)
        observed_at = run.observed_at_unix_ns or _document_time(run.payload)
        payload = _canonical(run.payload)
        payload_bytes = len(payload.encode("utf-8"))
        metrics = flatten_metrics(run.payload)
        metrics.update(run.extra_metrics)
        truncated = int(len(metrics) >= MAX_METRICS_PER_RUN)
        attribution = attributor.for_run(run, source_agent, written_at_unix_ns)
        cursor = connection.execute(
            "INSERT INTO runs(source_id, study, native_id, kind, status, observed_at_unix_ns,"
            " payload_json, payload_sha256, payload_bytes, provenance_json, agent, agent_detail,"
            " agent_source, metrics_truncated) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                source_id,
                study,
                native_id,
                run.kind,
                run.status,
                observed_at,
                payload if payload_bytes <= INLINE_PAYLOAD_LIMIT else None,
                hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                payload_bytes,
                _canonical(run.provenance) if run.provenance is not None else None,
                attribution.agent,
                attribution.detail,
                attribution.source,
                truncated,
            ),
        )
        run_id = int(cursor.lastrowid)
        connection.executemany(
            "INSERT OR IGNORE INTO metrics(run_id, path, value) VALUES(?,?,?)",
            ((run_id, path, value) for path, value in metrics.items()),
        )
        written += len(metrics)
    return written


# --------------------------------------------------------------------------- read


def status(ssot_path: Path = DEFAULT_SSOT_PATH) -> dict[str, object]:
    with _read_only(ssot_path) as connection:
        meta = connection.execute("SELECT * FROM ssot_meta").fetchone()
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
            "metrics": connection.execute("SELECT count(*) FROM metrics").fetchone()[0],
            "distinct_metric_paths": connection.execute(
                "SELECT count(DISTINCT path) FROM metrics"
            ).fetchone()[0],
            "shapes": shapes,
            "agents": agents,
            "studies": studies,
            "unreadable_sources": failed,
        }


def query(sql: str, ssot_path: Path = DEFAULT_SSOT_PATH, limit: int = 200) -> list[dict]:
    with _read_only(ssot_path) as connection:
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
    with _read_only(ssot_path) as connection:
        return [dict(row) for row in connection.execute(sql, parameters + (limit,))]


# --------------------------------------------------------------------------- CLI


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="friday ssot", allow_abbrev=False)
    parser.add_argument("--database", type=Path, default=DEFAULT_SSOT_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", allow_abbrev=False)
    commands.add_parser("status", allow_abbrev=False)
    sql = commands.add_parser("query", allow_abbrev=False)
    sql.add_argument("sql")
    sql.add_argument("--limit", type=int, default=200)
    names = commands.add_parser("metrics", allow_abbrev=False)
    names.add_argument("pattern", nargs="?")
    names.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "build":
            _print(build(args.database))
        elif args.command == "status":
            _print(status(args.database))
        elif args.command == "query":
            _print(query(args.sql, args.database, args.limit))
        elif args.command == "metrics":
            _print(metric_paths(args.pattern, args.database, args.limit))
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
]


if __name__ == "__main__":
    raise SystemExit(main())
