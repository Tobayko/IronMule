"""Immutable DATA1 report import and leakage-safe dataset projection.

The older :mod:`friday_optimizer` corpus remains a separate diagnostic
inventory.  This module accepts only the explicit ``ironmule.*.v1`` contracts
produced by the portable runners and never manufactures a performance label for
an incorrect, unsupported, censored, or failed trial.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

from friday_evidence.canonical import canonical_json_bytes, canonical_sha256
from friday_evidence.events import EventJournal, EventJournalError
from friday_evidence.statistics import paired_ratio, summarise

from .contracts import (
    ContractError,
    MAX_JSON_BYTES,
    file_sha256,
    load_json,
    sha256,
    spec_digest,
    validate_spec,
    write_json_new,
)


REPORT_SCHEMA = "ironmule.collection.v1"
DATASET_SCHEMA = "ironmule.dataset.v1"
SPLIT_PROTOCOL = "within_backend_new_workload"
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
_TRIAL_STATUSES = frozenset({"valid", "incorrect", "unsupported", "censored", "failed"})
_ORDERS = frozenset({"AB", "BA"})
_MAX_REPORTS = 10_000
_MAX_TRIALS = 1_000
_MAX_COST_NODES = 10_000
_GROUP_FIELDS = (
    "lineage_id",
    "prompt_family",
    "shape_family",
    "capture_session",
    "a_sha256",
    "b_sha256",
    "weight_sha256",
)


class CorpusError(ContractError):
    """The immutable corpus contract could not be upheld."""


def _state_root(state_dir: str | os.PathLike[str]) -> Path:
    root = Path(state_dir).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        raise CorpusError("state_directory_symlink")
    if root.exists():
        info = root.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise CorpusError("state_directory_not_private")
    return root


def _require_private_file(path: Path) -> None:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise CorpusError("corpus_artifact_not_private")


def _require_private_directory(path: Path) -> None:
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise CorpusError("corpus_directory_not_private")


def _finite_positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CorpusError(f"invalid_{field}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise CorpusError(f"invalid_{field}")
    return number


def _validate_costs(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CorpusError("costs_object_required")
    nodes = 0

    def visit(item: Any) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_COST_NODES:
            raise CorpusError("costs_too_large")
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str) or not key:
                    raise CorpusError("invalid_cost_key")
                visit(child)
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child)
            return
        if item is None or isinstance(item, (str, bool)):
            return
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            if not math.isfinite(float(item)) or float(item) < 0:
                raise CorpusError("invalid_cost_value")
            return
        raise CorpusError("invalid_cost_value")

    visit(value)
    canonical_json_bytes(value)
    return value


def _validate_samples(
    value: Any, *, status: str, pairs: int, smoke_only: bool
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > pairs:
        raise CorpusError("invalid_samples")
    if status == "valid" and smoke_only and value:
        raise CorpusError("smoke_trial_cannot_supply_performance_samples")
    if status == "valid" and not smoke_only and len(value) != pairs:
        raise CorpusError("valid_trial_requires_all_pairs")
    result: list[dict[str, Any]] = []
    for sample in value:
        if not isinstance(sample, dict) or sample.get("order") not in _ORDERS:
            raise CorpusError("invalid_sample")
        exact = sample.get("exact")
        if type(exact) is not bool:
            raise CorpusError("invalid_sample_exactness")
        baseline = _finite_positive(sample.get("baseline_seconds"), "baseline_seconds")
        candidate = _finite_positive(sample.get("candidate_seconds"), "candidate_seconds")
        if status == "valid" and not exact:
            raise CorpusError("valid_trial_must_be_exact")
        # Preserve the submitted canonical value (including integer-vs-float
        # spelling) in the immutable raw report after validating its numeric
        # interpretation.
        result.append(dict(sample))
    return result


def _same_json(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _validate_report(report: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise CorpusError("invalid_collection_schema")
    raw = canonical_json_bytes(report)
    if len(raw) > MAX_JSON_BYTES:
        raise CorpusError("report_too_large")
    run_id = report.get("run_id")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise CorpusError("invalid_run_id")
    if run_id != spec["run_id"]:
        raise CorpusError("report_spec_run_mismatch")
    expected_digest = spec_digest(spec)
    if report.get("spec_sha256") != expected_digest:
        raise CorpusError("report_spec_digest_mismatch")
    if report.get("backend") != spec["backend"] or report.get("partition") != spec["partition"]:
        raise CorpusError("report_job_scope_mismatch")
    if report.get("performance_claim") is not False:
        raise CorpusError("collection_cannot_claim_performance")
    sha256(report.get("code_sha256"), "code_sha256")
    if spec.get("code_sha256") != report["code_sha256"]:
        raise CorpusError("report_code_digest_mismatch")
    for field in ("hardware", "environment"):
        value = report.get(field)
        if not isinstance(value, dict):
            raise CorpusError(f"invalid_{field}")
        if not value and report.get("status") not in {"failed", "censored"}:
            raise CorpusError(f"invalid_{field}")
        canonical_json_bytes(value)
    status = report.get("status")
    if not isinstance(status, str) or not status or len(status) > 64:
        raise CorpusError("invalid_report_status")
    trials = report.get("trials")
    if not isinstance(trials, list) or len(trials) > _MAX_TRIALS:
        raise CorpusError("invalid_trials")
    cases = {case["case_id"]: case for case in spec["cases"]}
    seen: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for trial in trials:
        if not isinstance(trial, dict):
            raise CorpusError("trial_object_required")
        case_id = trial.get("case_id")
        candidate = trial.get("candidate")
        if case_id not in cases or candidate not in spec["candidates"]:
            raise CorpusError("trial_outside_spec")
        key = (case_id, candidate)
        if key in seen:
            raise CorpusError("duplicate_trial")
        seen.add(key)
        if trial.get("run_id") != run_id or trial.get("backend") != report["backend"]:
            raise CorpusError("trial_run_or_backend_mismatch")
        if trial.get("partition") != report["partition"]:
            raise CorpusError("trial_partition_mismatch")
        if not _same_json(trial.get("case"), cases[case_id]):
            raise CorpusError("trial_case_differs_from_preregistered_case")
        for field in ("hardware", "environment"):
            if not isinstance(trial.get(field), dict):
                raise CorpusError(f"invalid_trial_{field}")
            if report[field] and not _same_json(trial.get(field), report[field]):
                raise CorpusError(f"trial_{field}_mismatch")
        if trial.get("code_sha256") != report["code_sha256"]:
            raise CorpusError("trial_code_digest_mismatch")
        if trial.get("spec_sha256") != expected_digest:
            raise CorpusError("trial_spec_digest_mismatch")
        trial_status = trial.get("status")
        if trial_status not in _TRIAL_STATUSES:
            raise CorpusError("invalid_trial_status")
        smoke_only = trial.get("smoke_only") is True
        if smoke_only != (spec["mode"] == "smoke" and candidate == "native"):
            raise CorpusError("invalid_smoke_trial_marker")
        samples = _validate_samples(
            trial.get("samples"), status=trial_status, pairs=spec["pairs"], smoke_only=smoke_only
        )
        costs = _validate_costs(trial.get("costs"))
        normalized.append({**trial, "samples": samples, "costs": costs})
    # A completed/passed job must make missing arms explicit.  Failed jobs may
    # legitimately terminate before the full panel and remain useful history.
    if spec["mode"] == "measure" and status in {"complete", "completed", "passed", "valid"}:
        expected = {(case_id, candidate) for case_id in cases for candidate in spec["candidates"]}
        if seen != expected:
            raise CorpusError("completed_report_missing_trial")
    return {**report, "trials": normalized}


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _case_rows(entries: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec, report in entries:
        for case in spec["cases"]:
            rows.append({
                "run_id": report["run_id"],
                "backend": report["backend"],
                "partition": report["partition"],
                "hardware_sha256": canonical_sha256(report["hardware"]),
                "case": case,
            })
    return rows


def _groups(
    entries: Sequence[tuple[dict[str, Any], dict[str, Any]]]
) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    rows = _case_rows(entries)
    union = _UnionFind(len(rows))
    first: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        case = row["case"]
        for field in _GROUP_FIELDS:
            marker_name = "input_sha256" if field in {"a_sha256", "b_sha256"} else field
            marker = (marker_name, case[field])
            previous = first.setdefault(marker, index)
            union.union(index, previous)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[union.find(index)].append(index)
    assignment: dict[tuple[str, str], str] = {}
    partitions: dict[str, str] = {}
    for indexes in components.values():
        values = {rows[index]["partition"] for index in indexes}
        if len(values) != 1:
            raise CorpusError("connected_capture_group_crosses_partitions")
        tokens = []
        for index in indexes:
            for field in _GROUP_FIELDS:
                marker_name = "input_sha256" if field in {"a_sha256", "b_sha256"} else field
                tokens.append(f"{marker_name}:{rows[index]['case'][field]}")
        group_sha = canonical_sha256(sorted(set(tokens)))
        partition = next(iter(values))
        partitions[group_sha] = partition
        for index in indexes:
            assignment[(rows[index]["run_id"], rows[index]["case"]["case_id"])] = group_sha
    return assignment, partitions


def _sealed_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        try:
            current = load_json(path)
        except (OSError, ValueError) as exc:
            raise CorpusError("sealed_artifact_invalid") from exc
        if not _same_json(current, value):
            raise CorpusError("sealed_artifact_tampered_or_conflicting")
        return
    try:
        write_json_new(path, value)
    except (OSError, ValueError) as exc:
        raise CorpusError("cannot_seal_artifact") from exc


class _ImportLock:
    def __init__(self, root: Path):
        self.path = root / ".import.lock"
        self.descriptor: int | None = None

    def __enter__(self) -> "_ImportLock":
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) & 0o077:
            os.close(descriptor)
            raise CorpusError("unsafe_import_lock")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        self.descriptor = descriptor
        return self

    def __exit__(self, *_args: object) -> None:
        if self.descriptor is not None:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = None


def _load_entries(root: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    index = root / "imports"
    if not index.exists():
        return []
    if index.is_symlink() or not index.is_dir():
        raise CorpusError("invalid_import_index")
    _require_private_directory(index)
    paths = sorted(index.glob("*.json"))
    if len(paths) > _MAX_REPORTS:
        raise CorpusError("too_many_imported_reports")
    entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for path in paths:
        _require_private_file(path)
        manifest = load_json(path)
        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or path.name != f"{run_id}.json" or not _RUN_ID.fullmatch(run_id):
            raise CorpusError("invalid_import_manifest")
        spec_sha = sha256(manifest.get("spec_sha256"), "spec_sha256")
        report_sha = sha256(manifest.get("report_sha256"), "report_sha256")
        spec_path = root / "specs" / f"{spec_sha}.json"
        report_path = root / "reports" / f"{report_sha}.json"
        try:
            _require_private_directory(spec_path.parent)
            _require_private_directory(report_path.parent)
            _require_private_file(spec_path)
            _require_private_file(report_path)
            if file_sha256(spec_path) != spec_sha or file_sha256(report_path) != report_sha:
                raise CorpusError("sealed_artifact_hash_mismatch")
            spec = validate_spec(load_json(spec_path))
            report = _validate_report(load_json(report_path), spec)
        except (OSError, ValueError) as exc:
            if isinstance(exc, CorpusError):
                raise
            raise CorpusError("sealed_artifact_invalid") from exc
        if report["run_id"] != run_id:
            raise CorpusError("manifest_run_mismatch")
        expected_manifest = {
            "schema": "ironmule.import-manifest.v1",
            "run_id": run_id,
            "spec_sha256": spec_sha,
            "report_sha256": report_sha,
            "backend": report["backend"],
            "partition": report["partition"],
            "trial_count": len(report["trials"]),
        }
        if not _same_json(manifest, expected_manifest):
            raise CorpusError("import_manifest_tampered_or_conflicting")
        entries.append((spec, report))
    _groups(entries)
    if entries:
        try:
            with EventJournal(root / "events.sqlite3", read_only=True) as journal:
                journal.verify()
        except (OSError, EventJournalError) as exc:
            raise CorpusError("corpus_event_journal_invalid") from exc
    return entries


def import_report(
    state_dir: str | os.PathLike[str], report: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    """Validate and immutably import one preregistered collection report.

    Re-importing the same canonical report is idempotent.  Reusing its run ID
    with different bytes, mutating a sealed artifact, or connecting a capture
    group to another partition fails closed.
    """
    root = _state_root(state_dir)
    try:
        # Canonical round-tripping takes an owned snapshot before any path is
        # opened, so a caller cannot mutate the accepted object during import.
        checked_spec = validate_spec(json.loads(canonical_json_bytes(spec)))
        checked_report = _validate_report(json.loads(canonical_json_bytes(report)), checked_spec)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, CorpusError):
            raise
        raise CorpusError(str(exc)) from exc
    spec_sha = spec_digest(checked_spec)
    report_sha = canonical_sha256(checked_report)
    run_id = checked_report["run_id"]
    execution_verified = _execution_verified(root, checked_report)
    with _ImportLock(root):
        entries = _load_entries(root)
        existing = next(((s, r) for s, r in entries if r["run_id"] == run_id), None)
        if existing is not None:
            existing_spec, existing_report = existing
            if spec_digest(existing_spec) != spec_sha or canonical_sha256(existing_report) != report_sha:
                raise CorpusError("run_id_already_bound_to_different_artifact")
            return {
                "schema": "ironmule.import.v1",
                "status": "already_imported",
                "run_id": run_id,
                "spec_sha256": spec_sha,
                "report_sha256": report_sha,
                "partition": checked_report["partition"],
                "backend": checked_report["backend"],
                "trial_count": len(checked_report["trials"]),
                "execution_verified": execution_verified,
                "performance_claim": False,
            }
        _groups([*entries, (checked_spec, checked_report)])
        _sealed_json(root / "specs" / f"{spec_sha}.json", checked_spec)
        _sealed_json(root / "reports" / f"{report_sha}.json", checked_report)
        manifest = {
            "schema": "ironmule.import-manifest.v1",
            "run_id": run_id,
            "spec_sha256": spec_sha,
            "report_sha256": report_sha,
            "backend": checked_report["backend"],
            "partition": checked_report["partition"],
            "trial_count": len(checked_report["trials"]),
        }
        _sealed_json(root / "imports" / f"{run_id}.json", manifest)
        with EventJournal(root / "events.sqlite3") as journal:
            journal.append(run_id, "validation", {
                "phase": "portable_corpus_import",
                "spec_sha256": spec_sha,
                "report_sha256": report_sha,
                "backend": checked_report["backend"],
                "partition": checked_report["partition"],
                "trial_count": len(checked_report["trials"]),
            })
    return {
        "schema": "ironmule.import.v1",
        "status": "imported",
        "run_id": run_id,
        "spec_sha256": spec_sha,
        "report_sha256": report_sha,
        "partition": checked_report["partition"],
        "backend": checked_report["backend"],
        "trial_count": len(checked_report["trials"]),
        "execution_verified": execution_verified,
        "performance_claim": False,
    }


def _ratio_summary(samples: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    baseline = [sample["baseline_seconds"] for sample in samples]
    candidate = [sample["candidate_seconds"] for sample in samples]
    paired = paired_ratio(candidate, baseline, resamples=1_000, seed=seed)
    return {
        "pair_count": len(samples),
        "baseline": summarise(baseline),
        "candidate": summarise(candidate),
        "median_ratio": paired["median_ratio"],
        "ci_low": paired["ci_low"],
        "ci_high": paired["ci_high"],
    }


def _execution_verified(root: Path, report: dict[str, Any]) -> bool:
    """Trust only the local append-only control receipt, never report input."""
    expected = {
        "verified_execution": True,
        "backend": report["backend"],
        "spec_sha256": report["spec_sha256"],
        "report_sha256": canonical_sha256(report),
        "terminal_state": "complete",
    }
    try:
        with EventJournal(root / "control.sqlite3", read_only=True) as journal:
            events = journal.events(run_id=report["run_id"], limit=1_000)
    except (OSError, EventJournalError):
        return False
    for event in events:
        payload = event.get("payload")
        if event.get("kind") != "validation" or not isinstance(payload, Mapping):
            continue
        if all(payload.get(key) == value for key, value in expected.items()):
            return True
    return False


def _dataset(root: Path, *, include_holdout: bool) -> dict[str, Any]:
    entries = _load_entries(root)
    assignments, group_partitions = _groups(entries)
    records: list[dict[str, Any]] = []
    all_record_digests: list[str] = []
    status_counts: Counter[str] = Counter()
    partition_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    backend_sample_counts: Counter[str] = Counter()
    report_status_counts: Counter[str] = Counter()
    grouped_counts: Counter[str] = Counter(group_partitions.values())
    holdout_digests: list[str] = []
    report_rows: list[dict[str, Any]] = []
    unique_cases: set[str] = set()
    for spec, report in entries:
        execution_verified = _execution_verified(root, report)
        report_status_counts[report["status"]] += 1
        report_rows.append({
            "run_id": report["run_id"],
            "backend": report["backend"],
            "partition": report["partition"],
            "status": report["status"],
            "spec_sha256": report["spec_sha256"],
            "report_sha256": canonical_sha256(report),
            "trial_count": len(report["trials"]),
            "execution_verified": execution_verified,
        })
        hardware_sha = canonical_sha256(report["hardware"])
        environment_sha = canonical_sha256(report["environment"])
        for trial in report["trials"]:
            case = trial["case"]
            unique_cases.add(canonical_sha256({
                "lineage_id": case["lineage_id"],
                "a_sha256": case["a_sha256"],
                "b_sha256": case["b_sha256"],
                "weight_sha256": case["weight_sha256"],
                "model_sha256": case["model_sha256"],
                "shape": case["shape"],
            }))
            status_counts[trial["status"]] += 1
            partition_counts[trial["partition"]] += 1
            backend_counts[trial["backend"]] += 1
            backend_sample_counts[trial["backend"]] += len(trial["samples"])
            group_sha = assignments[(trial["run_id"], trial["case_id"])]
            record: dict[str, Any] = {
                "run_id": trial["run_id"],
                "case_id": trial["case_id"],
                "backend": trial["backend"],
                "candidate": trial["candidate"],
                "partition": trial["partition"],
                "status": trial["status"],
                "group_sha256": group_sha,
                "stratum_sha256": canonical_sha256({
                    "backend": trial["backend"],
                    "hardware_sha256": hardware_sha,
                    "model_sha256": case["model_sha256"],
                }),
                "hardware_sha256": hardware_sha,
                "environment_sha256": environment_sha,
                "code_sha256": trial["code_sha256"],
                "spec_sha256": trial["spec_sha256"],
                "shape": list(case["shape"]),
                "dtype": case["dtype"],
                "model_id": case["model_id"],
                "model_sha256": case["model_sha256"],
                "lineage_id": case["lineage_id"],
                "weight_sha256": case["weight_sha256"],
                "prompt_family": case["prompt_family"],
                "shape_family": case["shape_family"],
                "capture_session": case["capture_session"],
                "input_sha256": canonical_sha256([case["a_sha256"], case["b_sha256"]]),
                "costs": trial["costs"],
                "sample_count": len(trial["samples"]),
                "execution_verified": execution_verified,
                "label": None,
            }
            if (
                execution_verified
                and trial["status"] == "valid"
                and trial["samples"]
                and not trial.get("smoke_only", False)
            ):
                seed = int(canonical_sha256([trial["run_id"], trial["case_id"], trial["candidate"]])[:8], 16)
                record["label"] = _ratio_summary(trial["samples"], seed)
            digest = canonical_sha256(record)
            all_record_digests.append(digest)
            if trial["partition"] == "holdout":
                holdout_digests.append(digest)
            if include_holdout or trial["partition"] != "holdout":
                records.append(record)
    records.sort(
        key=lambda item: (
            item["partition"], item["backend"], item["run_id"], item["case_id"], item["candidate"]
        )
    )
    body: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "split_protocol": SPLIT_PROTOCOL,
        "reports": sorted(report_rows, key=lambda item: item["run_id"]),
        "records": records,
        "counts": {
            "reports": len(entries),
            "records": len(all_record_digests),
            "unique_cases": len(unique_cases),
            "raw_samples": sum(backend_sample_counts.values()),
            "visible_records": len(records),
            "by_status": dict(sorted(status_counts.items())),
            "by_partition": dict(sorted(partition_counts.items())),
            "by_backend": dict(sorted(backend_counts.items())),
            "raw_samples_by_backend": dict(sorted(backend_sample_counts.items())),
            "by_report_status": dict(sorted(report_status_counts.items())),
            "execution_verified_reports": sum(row["execution_verified"] for row in report_rows),
            "groups_by_partition": dict(sorted(grouped_counts.items())),
        },
        "holdout": {
            "hidden": not include_holdout,
            "record_count": partition_counts.get("holdout", 0),
            "group_count": grouped_counts.get("holdout", 0),
            "projection_sha256": canonical_sha256(sorted(holdout_digests)),
        },
        "performance_claim": False,
    }
    body["dataset_sha256"] = canonical_sha256({
        "split_protocol": SPLIT_PROTOCOL,
        "record_digests": sorted(all_record_digests),
        "report_digests": sorted(canonical_sha256(report) for _, report in entries),
        "group_partitions": dict(sorted(group_partitions.items())),
    })
    return body


def build_dataset(state_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a deterministic read-only dataset with hidden holdout labels."""
    return _dataset(_state_root(state_dir), include_holdout=False)


def _build_training_dataset(state_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Internal evaluator view; callers must never expose its holdout rows."""
    return _dataset(_state_root(state_dir), include_holdout=True)


__all__ = ["CorpusError", "build_dataset", "import_report"]
