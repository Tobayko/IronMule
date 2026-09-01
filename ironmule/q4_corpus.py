"""Read-only historical corpus primitives for Q4.

The corpus is deliberately not a replay dataset.  Historical files are useful
as provenance, priors, censored failures, or Q3 validation/holdout evidence,
but they do not become Q4 training transitions.  This module only deals with
content metadata and conservative eligibility gates; it never executes a
record, imports a model, or rewrites an input file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .q4_contracts import (
    ArtifactRecord,
    Dataset,
    HistoricalRole,
    SplitManifest,
    _default_action_pools,
    canonical_sha256,
)


CORPUS_SCHEMA = "ironmule.q4_corpus.v1"
IMPORT_REPORT_SCHEMA = "ironmule.q4_import_report.v1"
_SOURCE_RE = re.compile(r"(?<![A-Za-z0-9])(Q2|B35|B36|B27|E14b|E16|X1|Q3[c-f]|E11)(?![A-Za-z0-9])", re.IGNORECASE)
_Q4_SIZES = ("1B", "4B", "12B")
_QUALITY = {"RAW_SAMPLES", "SUMMARY_ONLY", "PARTIAL"}
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


def _is_derived_report(logical_name: str) -> bool:
    name = Path(logical_name).name.lower()
    return name.startswith("q4_implementation_report") and name.endswith((".md", ".sha256"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(_safe_read_file(path))


def _safe_read_file(path: Path) -> bytes:
    """Read one regular file through a no-follow, bounded, stable fd."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat_is_regular(before.st_mode) or before.st_size > MAX_ARTIFACT_BYTES:
            raise OSError("artifact is not a bounded regular file")
        data = bytearray()
        while len(data) <= MAX_ARTIFACT_BYTES:
            chunk = os.read(fd, min(1024 * 1024, MAX_ARTIFACT_BYTES + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(fd)
        if len(data) > MAX_ARTIFACT_BYTES or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise OSError("artifact changed during read")
        return bytes(data)
    finally:
        os.close(fd)


def stat_is_regular(mode: int) -> bool:
    # Avoid importing the large pathlib/shutil surface; S_IFMT/S_IFREG are
    # stable stdlib constants on all supported hosts.
    import stat
    return stat.S_ISREG(mode)


def _json(path: Path) -> Any | None:
    try:
        return json.loads(_safe_read_file(path).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _walk(root: Path) -> tuple[tuple[str, Path, str], ...]:
    """Return (path-free logical name, file path, source alias) rows."""
    if not root.exists() or not root.is_dir():
        return ()
    result: list[tuple[str, Path, str]] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file() and not item.is_symlink()), key=lambda item: item.as_posix()):
        try:
            if path.stat().st_size > MAX_ARTIFACT_BYTES:
                continue
        except OSError:
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        result.append((relative, path, "raw"))
    return tuple(result)


def _source_name(logical_name: str, payload: Any | None = None) -> str:
    match = _SOURCE_RE.search(logical_name)
    if match:
        value = match.group(1)
        if value.lower().startswith("q3"):
            return "Q3" + value[-1].lower()
        return "E14b" if value.lower() == "e14b" else value.upper() if value.lower().startswith(("q", "b", "e", "x")) else value
    if isinstance(payload, Mapping):
        for key in ("experiment", "experiment_id", "study", "id"):
            value = payload.get(key)
            if isinstance(value, str) and _SOURCE_RE.fullmatch(value):
                if value.lower().startswith("q3"):
                    return "Q3" + value[-1].lower()
                return "E14b" if value.lower() == "e14b" else value.upper()
    return "UNCLASSIFIED"


def role_for_source(source: str) -> HistoricalRole | None:
    source = source.strip()
    return {
        "Q2": HistoricalRole.Q3_VALIDATION,
        "B35": HistoricalRole.Q3_VALIDATION,
        "B36": HistoricalRole.Q3_SEALED_HOLDOUT,
        "B27": HistoricalRole.PRIOR_ONLY,
        "E14b": HistoricalRole.PRIOR_ONLY,
        "E16": HistoricalRole.PRIOR_ONLY,
        "X1": HistoricalRole.PRIOR_ONLY,
        "E11": HistoricalRole.LEDGER_ONLY,
        "Q3c": HistoricalRole.CENSORED_FAILURE,
        "Q3d": HistoricalRole.CENSORED_FAILURE,
        "Q3e": HistoricalRole.CENSORED_FAILURE,
        "Q3f": HistoricalRole.CENSORED_FAILURE,
    }.get(source)


def quality_for_file(path: Path, payload: Any | None = None) -> str:
    name = path.name.lower()
    if name.endswith(".partial") or name.startswith(".") and ".partial" in name:
        return "PARTIAL"
    if ("summary" in name or "review" in name or "preregistration" in name
            or name.endswith(".sha256") or name.endswith(".md")):
        return "SUMMARY_ONLY"
    if path.suffix.lower() != ".json" or not isinstance(payload, (Mapping, list)):
        return "SUMMARY_ONLY"
    # A JSON result is raw only if it contains repeat-level observations.  A
    # top-level aggregate such as X1's ``cells`` remains summary-only unless
    # the file explicitly carries samples/runs/measured children.
    if isinstance(payload, Mapping) and _has_repeat_level(payload):
        return "RAW_SAMPLES"
    return "SUMMARY_ONLY"


def _find_values(value: Any, names: set[str], *, limit: int = 1000) -> list[Any]:
    found: list[Any] = []
    def visit(item: Any) -> None:
        if len(found) >= limit:
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                if isinstance(key, str) and key.lower() in names:
                    found.append(child)
                visit(child)
        elif isinstance(item, list):
            for child in item[:128]:
                visit(child)
    visit(value)
    return found


def _all_true(value: Any, names: set[str]) -> bool:
    values = _find_values(value, names)
    return bool(values) and all(item is True for item in values)


def _has_repeat_level(payload: Any) -> bool:
    values = _find_values(payload, {"samples", "measured", "children", "repeat_samples", "checkpoint", "checkpoints"})
    return any(isinstance(item, (list, tuple, Mapping)) and bool(item) for item in values)


def _has_identity(payload: Any) -> bool:
    # Identity is accepted only if the source explicitly reports every gate
    # it claims to measure.  Absence is never treated as success.
    values = _find_values(payload, {"token_identity", "identity_gate", "canonical_correctness_gate", "stop_reason_identity", "token_count_identity", "state_identity", "deterministic"})
    return bool(values) and all(item is True for item in values if isinstance(item, bool)) and any(item is True for item in values)


def _identity_gate(source: str, payload: Any | None) -> bool:
    """Require an explicit source-specific correctness gate.

    A generic ``token_identity=true`` buried in a summary is not enough.  Only
    B36's fully repeated pair evidence can be a historical safe performance
    label; every other historical source remains prior/validation metadata.
    """
    if source != "B36":
        return False
    required = {
        "identity_gate", "canonical_correctness_gate", "post_evidence_complete",
        "no_crash", "token_identity",
    }
    values = {name: _find_values(payload, {name}) for name in required}
    return all(values[name] and all(item is True for item in values[name]) for name in required)


def _has_resources(payload: Any) -> bool:
    keys = {"mlx_active_memory_bytes", "mlx_peak_memory_bytes", "rss_peak_bytes", "swap_delta_bytes", "swap_before_bytes", "swap_after_bytes", "active", "peak", "rss_bytes", "swap"}
    values = [item for item in _find_values(payload, keys) if not isinstance(item, (Mapping, list))]
    # A resource-history container alone is not a resource gate.  We require
    # concrete, non-null numeric observations and at least one memory/swap
    # value; absence remains unknown rather than a successful zero.
    numeric = [item for item in values if isinstance(item, (int, float)) and not isinstance(item, bool)]
    if numeric:
        return True
    # B36 records the evaluator-owned resource result in pair hard-gates while
    # process-start checkpoints legitimately contain null memory fields.
    gates = _find_values(payload, {"hard_gates"})
    for gate in gates:
        if isinstance(gate, Mapping) and all(gate.get(name) is True for name in ("peak_memory", "swap", "no_crash")):
            return True
    return False


def _resource_gate(source: str, payload: Any | None) -> bool:
    """Require all evaluator-owned resource gates for the one usable source."""
    if source != "B36" or not isinstance(payload, Mapping):
        return False
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 16:
        return False
    for pair in pairs:
        if not isinstance(pair, Mapping):
            return False
        result = pair.get("pair_result")
        # Every pair must carry the complete evaluator-owned gate set.  A
        # single numeric peak value is never sufficient.
        if not isinstance(result, Mapping) or result.get("status") != "ok":
            return False
        gates = result.get("hard_gates")
        required = {"complete", "identity", "no_crash", "peak_memory", "swap", "timings", "token_identity"}
        if not isinstance(gates, Mapping) or set(gates) != required or not all(gates.get(name) is True for name in required):
            return False
    return True


def _b36_gate(payload: Any | None) -> bool:
    """Exact full B36 qualification gate used for the historical label."""
    if not isinstance(payload, Mapping) or payload.get("schema") != "ironmule.b36.v1" or str(payload.get("status", "")).lower() != "complete":
        return False
    constants = payload.get("constants")
    if constants != {"max_tokens": 32, "no_retry": True, "pairs": 16, "repeats": 5, "warmups": 2}:
        return False
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 16:
        return False
    required_hard_gates = {"complete", "identity", "no_crash", "peak_memory", "swap", "timings", "token_identity"}
    for pair in pairs:
        if not isinstance(pair, Mapping):
            return False
        result = pair.get("pair_result", {})
        if not isinstance(result, Mapping) or result.get("status") != "ok":
            return False
        gates = result.get("hard_gates")
        if not isinstance(gates, Mapping) or set(gates) != required_hard_gates or not all(gates.get(name) is True for name in required_hard_gates):
            return False
        children = pair.get("children")
        if not isinstance(children, list) or len(children) != 2:
            return False
        for child in children:
            if not isinstance(child, Mapping) or child.get("schema") != "ironmule.b36.child.v1":
                return False
            if child.get("returncode") != 0 or child.get("crashed") is not False or child.get("no_crash") is not True or child.get("identity_gate") is not True or child.get("canonical_correctness_gate") is not True or child.get("post_evidence_complete") is not True:
                return False
            if not isinstance(child.get("warmups"), list) or len(child["warmups"]) != 2 or not isinstance(child.get("measured"), list) or len(child["measured"]) != 5:
                return False
    return True


def _status(payload: Any | None) -> str:
    if not isinstance(payload, Mapping):
        return "UNREADABLE"
    for key in ("status", "verdict", "result_type", "classification"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "UNSTATED"


def _model_sizes(payload: Any | None) -> tuple[str, ...]:
    """Only accept an explicit model_size field; never parse model IDs/names."""
    values = _find_values(payload, {"model_size"})
    return tuple(sorted({value for value in values if value in _Q4_SIZES}))


def _invalid_cells(source: str, payload: Any | None) -> tuple[str, ...]:
    if source != "E14b" or not isinstance(payload, Mapping):
        return ()
    invalid: set[str] = set()
    correctness = payload.get("correctness")
    if isinstance(correctness, Mapping) and correctness.get("generated_tokens_equal") is False:
        divergent = correctness.get("divergent")
        batches = []
        if isinstance(divergent, list):
            batches = [item.get("batch") for item in divergent if isinstance(item, Mapping)]
        for batch in batches:
            invalid.add(f"batch={batch}")
    return tuple(sorted(invalid))


@dataclass(frozen=True, slots=True)
class CorpusArtifact:
    """One content-addressed input; ``path`` is intentionally not serialized."""

    artifact_id: str
    sha256: str
    logical_name: str
    source_alias: str
    source_name: str
    quality: str
    role: str | None
    status: str
    raw_sample_gate: bool
    identity_gate: bool
    resource_gate: bool
    eligible_for_performance: bool
    excluded_reason: str | None = None
    model_sizes: tuple[str, ...] = ()
    invalid_cells: tuple[str, ...] = ()

    def to_artifact_record(self) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=self.artifact_id, sha256=self.sha256,
            quality=self.quality, source_name=self.source_name,
            role=self.role or "UNCLASSIFIED", status=self.status,
            source_alias=self.source_alias,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ironmule.q4_corpus_artifact.v1",
            "artifact_id": self.artifact_id, "sha256": self.sha256,
            "logical_name": self.logical_name, "source_alias": self.source_alias,
            "source_name": self.source_name, "quality": self.quality,
            "role": self.role, "status": self.status,
            "raw_sample_gate": self.raw_sample_gate,
            "identity_gate": self.identity_gate,
            "resource_gate": self.resource_gate,
            "eligible_for_performance": self.eligible_for_performance,
            "excluded_reason": self.excluded_reason,
            "model_sizes": list(self.model_sizes),
            "invalid_cells": list(self.invalid_cells),
        }


@dataclass(frozen=True, slots=True)
class Corpus:
    artifacts: tuple[CorpusArtifact, ...]
    duplicate_groups: tuple[tuple[str, tuple[str, ...]], ...]
    conflicts: tuple[tuple[str, tuple[str, ...]], ...]
    skipped_derived_reports: int = 0

    @property
    def unique_artifacts(self) -> tuple[CorpusArtifact, ...]:
        return self.artifacts

    @property
    def source_artifact_records(self) -> tuple[ArtifactRecord, ...]:
        return tuple(item.to_artifact_record() for item in self.artifacts)

    def report(self) -> dict[str, Any]:
        role_counts: dict[str, int] = {}
        quality_counts: dict[str, int] = {}
        for item in self.artifacts:
            role_counts[item.role or "UNCLASSIFIED"] = role_counts.get(item.role or "UNCLASSIFIED", 0) + 1
            quality_counts[item.quality] = quality_counts.get(item.quality, 0) + 1
        q4_splits = {"Q4_TRAIN": {size: 0 for size in _Q4_SIZES}, "Q4_VALIDATION": {size: 0 for size in _Q4_SIZES}, "Q4_SEALED_HOLDOUT": {size: 0 for size in _Q4_SIZES}}
        observed_sizes = {size for item in self.artifacts for size in item.model_sizes if size in _Q4_SIZES}
        return {
            "schema": IMPORT_REPORT_SCHEMA,
            "corpus_schema": CORPUS_SCHEMA,
            "artifact_count": len(self.artifacts) + sum(max(0, len(paths) - 1) for _, paths in self.duplicate_groups),
            "unique_artifact_count": len(self.artifacts),
            "duplicate_groups": [{"sha256": digest, "artifacts": list(paths)} for digest, paths in self.duplicate_groups],
            "conflicts": [{"logical_name": name, "sha256": list(digests)} for name, digests in self.conflicts],
            "skipped_derived_reports": self.skipped_derived_reports,
            "roles": dict(sorted(role_counts.items())),
            "qualities": dict(sorted(quality_counts.items())),
            "eligible_performance_artifacts": sum(item.eligible_for_performance for item in self.artifacts),
            "excluded_artifacts": sum(not item.eligible_for_performance for item in self.artifacts),
            "historical_q3": {"Q3_VALIDATION": sum(item.role == HistoricalRole.Q3_VALIDATION.value for item in self.artifacts), "Q3_SEALED_HOLDOUT": sum(item.role == HistoricalRole.Q3_SEALED_HOLDOUT.value for item in self.artifacts), "LEDGER_ONLY": sum(item.role == HistoricalRole.LEDGER_ONLY.value for item in self.artifacts)},
            "observed_explicit_model_sizes": sorted(observed_sizes),
            "q4_coverage": {"Q4_TRAIN": {size: 0 for size in _Q4_SIZES}, "Q4_VALIDATION": {size: 0 for size in _Q4_SIZES}, "Q4_SEALED_HOLDOUT": {size: 0 for size in _Q4_SIZES}},
            "missing_required_model_cells": {split: list(_Q4_SIZES) for split in q4_splits},
            "no_27b_q4_cell": True,
            "sequential_horizon": {"horizon": 17, "transitions": 0, "complete_trajectories": 0, "eligible": False, "reason": "historical corpus has no Q4 sequential horizon"},
            "foreign_evidence": "MISSING",
            "no_invented_performance": True,
        }

    def to_dataset(self, preregistration_sha256: str) -> Dataset:
        return Dataset(
            preregistration_sha256=preregistration_sha256,
            source_artifacts=self.source_artifact_records,
            action_pools=_default_action_pools(), contexts=(), states=(), trajectories=(), transitions=(), outcomes=(),
            split_manifest=SplitManifest.empty(), seed_manifest={"source": "historical-import", "transitions": "none"},
            no_invented_performance=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"schema": CORPUS_SCHEMA, "artifacts": [item.to_dict() for item in self.artifacts], "duplicate_groups": [{"sha256": digest, "artifacts": list(paths)} for digest, paths in self.duplicate_groups], "conflicts": [{"logical_name": name, "sha256": list(digests)} for name, digests in self.conflicts], "skipped_derived_reports": self.skipped_derived_reports}


def inspect_sources(sources: Iterable[tuple[str, Path]]) -> Corpus:
    """Inventory raw roots read-only and deduplicate only on content hash."""
    rows: list[tuple[str, Path, str]] = []
    skipped_derived_reports = 0
    for alias, root in sources:
        root = Path(root)
        for logical_name, path, _ in _walk(root):
            if _is_derived_report(logical_name):
                skipped_derived_reports += 1
                continue
            rows.append((alias, path, logical_name))
    rows.sort(key=lambda row: (row[0], row[2]))
    by_hash: dict[str, list[str]] = {}
    by_logical: dict[str, set[str]] = {}
    unique: dict[str, CorpusArtifact] = {}
    for alias, path, logical_name in rows:
        try:
            raw_bytes = _safe_read_file(path)
        except OSError:
            continue
        digest = sha256_bytes(raw_bytes)
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            payload = None
        source = _source_name(logical_name, payload)
        role = role_for_source(source)
        quality = quality_for_file(path, payload)
        status = _status(payload)
        raw_gate = quality == "RAW_SAMPLES" and _has_repeat_level(payload)
        identity_gate = _identity_gate(source, payload)
        resource_gate = _resource_gate(source, payload)
        invalid_cells = _invalid_cells(source, payload)
        excluded_reason: str | None = None
        if source == "B35" and isinstance(payload, Mapping) and (status.lower() in {"invalid", "failed", "rejected"} or payload.get("valid_for_performance") is False or payload.get("valid_for_metrics") is False):
            excluded_reason = "B35 invalid attempt retained but excluded"
        elif source in {"Q3c", "Q3d", "Q3e", "Q3f"}:
            excluded_reason = "Q3 failure/censored evidence; never a reward label"
        elif source in {"E14b", "E16", "X1", "B27"}:
            excluded_reason = "historical prior-only evidence; no Q4 action conversion"
        elif quality != "RAW_SAMPLES":
            excluded_reason = "summary-only or non-raw artifact"
        elif not identity_gate:
            excluded_reason = "identity/correctness gate absent or failed"
        elif not resource_gate:
            excluded_reason = "resource gate absent or failed"
        model_sizes = _model_sizes(payload)
        eligible = bool(source == "B36" and _b36_gate(payload) and raw_gate and identity_gate and resource_gate and role == HistoricalRole.Q3_SEALED_HOLDOUT and not excluded_reason)
        artifact = unique.get(digest)
        if artifact is None:
            artifact = CorpusArtifact(
                artifact_id=f"artifact-{digest}", sha256=digest, logical_name=logical_name,
                source_alias=alias, source_name=source, quality=quality, role=role.value if role else None,
                status=status, raw_sample_gate=raw_gate, identity_gate=identity_gate,
                resource_gate=resource_gate, eligible_for_performance=eligible,
                excluded_reason=excluded_reason, model_sizes=model_sizes,
                invalid_cells=invalid_cells,
            )
            unique[digest] = artifact
        by_hash.setdefault(digest, []).append(f"{alias}:{logical_name}")
        by_logical.setdefault(logical_name, set()).add(digest)
    duplicates = tuple((digest, tuple(sorted(paths))) for digest, paths in sorted(by_hash.items()) if len(paths) > 1)
    conflicts = tuple((name, tuple(sorted(digests))) for name, digests in sorted(by_logical.items()) if len(digests) > 1)
    return Corpus(artifacts=tuple(unique[digest] for digest in sorted(unique)), duplicate_groups=duplicates, conflicts=conflicts, skipped_derived_reports=skipped_derived_reports)


def historical_adapter(source: str, payload: Any | None, *, digest: str, alias: str = "raw", logical_name: str | None = None, quality: str | None = None) -> CorpusArtifact:
    """Adapt one already-read file; useful for synthetic contract tests."""
    source = source.strip()
    role = role_for_source(source)
    quality = quality or ("RAW_SAMPLES" if _has_repeat_level(payload) else "SUMMARY_ONLY")
    status = _status(payload)
    identity = _identity_gate(source, payload)
    resources = _resource_gate(source, payload)
    raw = quality == "RAW_SAMPLES" and _has_repeat_level(payload)
    excluded = None
    if source.startswith("Q3"):
        excluded = "Q3 failure/censored evidence; never a reward label"
    elif source in {"B27", "E14b", "E16", "X1"}:
        excluded = "historical prior-only evidence; no Q4 action conversion"
    elif quality != "RAW_SAMPLES":
        excluded = "summary-only or non-raw artifact"
    elif not identity:
        excluded = "identity/correctness gate absent or failed"
    elif not resources:
        excluded = "resource gate absent or failed"
    return CorpusArtifact(
        artifact_id=f"artifact-{digest}", sha256=digest,
        logical_name=logical_name or source, source_alias=alias, source_name=source,
        quality=quality, role=role.value if role else None, status=status,
        raw_sample_gate=raw, identity_gate=identity, resource_gate=resources,
        eligible_for_performance=bool(source == "B36" and _b36_gate(payload) and raw and identity and resources and role == HistoricalRole.Q3_SEALED_HOLDOUT and excluded is None),
        excluded_reason=excluded, model_sizes=_model_sizes(payload), invalid_cells=_invalid_cells(source, payload),
    )


def adapt_source(source: str, payload: Any | None, *, digest: str,
                 alias: str = "raw", logical_name: str | None = None,
                 quality: str | None = None) -> CorpusArtifact:
    """Named adapter entry point used by migration tests and tooling."""
    return historical_adapter(source, payload, digest=digest, alias=alias,
                              logical_name=logical_name, quality=quality)


# Keep the source-role table visible to reports without making each adapter a
# second, subtly different implementation.  These wrappers intentionally
# accept only already-read payloads and a caller-supplied content hash.
def adapt_q2(payload: Any | None, *, digest: str, **kwargs: Any) -> CorpusArtifact:
    return historical_adapter("Q2", payload, digest=digest, **kwargs)


def adapt_b35(payload: Any | None, *, digest: str, **kwargs: Any) -> CorpusArtifact:
    return historical_adapter("B35", payload, digest=digest, **kwargs)


def adapt_b36(payload: Any | None, *, digest: str, **kwargs: Any) -> CorpusArtifact:
    return historical_adapter("B36", payload, digest=digest, **kwargs)


def adapt_b27(payload: Any | None, *, digest: str, **kwargs: Any) -> CorpusArtifact:
    return historical_adapter("B27", payload, digest=digest, **kwargs)


def adapt_e14b(payload: Any | None, *, digest: str, **kwargs: Any) -> CorpusArtifact:
    return historical_adapter("E14b", payload, digest=digest, **kwargs)


def adapt_e16(payload: Any | None, *, digest: str, **kwargs: Any) -> CorpusArtifact:
    return historical_adapter("E16", payload, digest=digest, **kwargs)


def adapt_x1(payload: Any | None, *, digest: str, **kwargs: Any) -> CorpusArtifact:
    return historical_adapter("X1", payload, digest=digest, **kwargs)


def adapt_q3_failure(source: str, payload: Any | None, *, digest: str, **kwargs: Any) -> CorpusArtifact:
    if source not in {"Q3c", "Q3d", "Q3e", "Q3f"}:
        raise ValueError("Q3 failure adapter accepts Q3c/Q3d/Q3e/Q3f only")
    return historical_adapter(source, payload, digest=digest, **kwargs)


__all__ = [
    "CORPUS_SCHEMA", "IMPORT_REPORT_SCHEMA", "CorpusArtifact", "Corpus",
    "sha256_bytes", "sha256_file", "role_for_source", "quality_for_file",
    "inspect_sources", "historical_adapter",
    "adapt_source", "adapt_q2", "adapt_b35", "adapt_b36", "adapt_b27",
    "adapt_e14b", "adapt_e16", "adapt_x1", "adapt_q3_failure",
]
