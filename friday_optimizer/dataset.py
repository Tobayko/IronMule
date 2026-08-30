"""Deterministic, leakage-aware dataset snapshots for L1.1.

The builder is deliberately a snapshot builder, not a trainer.  It excludes
invalid/quarantined/summary-only evidence from labels while retaining their
counts and provenance in the dataset card.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_bytes
from .corpus import CorpusInventory, NormalizedRecord, QualityClass


SPLITS: tuple[str, ...] = ("train", "validation", "holdout")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


def _hash(value: Any) -> str:
    return hashlib.sha256(
        canonical_bytes(value, max_bytes=64 * 1024 * 1024, max_items=2_000_000)
    ).hexdigest()


def _canonical(value: Any) -> bytes:
    return canonical_bytes(value, max_bytes=64 * 1024 * 1024, max_items=2_000_000)


def _snapshot_value(value: Any) -> Any:
    """Keep scalar labels/features, hashing oversized raw sample arrays."""

    thawed = _thaw(value)
    try:
        encoded = canonical_bytes(thawed, max_bytes=64 * 1024, max_items=100_000)
    except ValueError:
        encoded = canonical_bytes(thawed, max_bytes=64 * 1024 * 1024, max_items=2_000_000)
        return {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}
    return thawed


def _snapshot_record(record: NormalizedRecord) -> dict[str, Any]:
    # Raw measurement arrays remain in the source corpus and are addressed by
    # content hash; snapshots carry only the fields needed for replay/training.
    flattened = dict(_flatten_mapping(record.data))
    selected = set(record.feature_fields) | set(record.label_fields)
    projection = {field: _snapshot_value(flattened[field]) for field in sorted(selected) if field in flattened}
    return {
        "record_id": record.record_id,
        "source_path": record.source_path,
        "source_kind": record.source_kind,
        "quality": record.quality.value,
        "source_sha256": record.source_sha256,
        "content_fingerprint": record.content_fingerprint,
        "feature_fields": list(record.feature_fields),
        "label_fields": list(record.label_fields),
        "missing_fields": list(record.missing_fields),
        "censored_fields": list(record.censored_fields),
        "identity": {
            "study_id": record.study_id,
            "run_id": record.run_id,
            "observed_time": record.observed_time,
            "hardware_fingerprint": record.hardware_fingerprint,
            "model_fingerprint": record.model_fingerprint,
            "workload_fingerprint": record.workload_fingerprint,
            "prompt_family": record.prompt_family,
        },
        "dirty": record.dirty,
        "duplicate_of": record.duplicate_of,
        "source_verified": record.source_verified,
        "manifest_verified": record.manifest_verified,
        "contract_verified": record.contract_verified,
        "selected_values": projection,
    }


def _flatten_mapping(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        result: list[tuple[str, Any]] = []
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else key
            result.extend(_flatten_mapping(value[key], child))
        return result
    return [(prefix, value)]


def _source_identity(record: NormalizedRecord) -> str:
    # Item/row suffixes belong to one source file for split purposes.
    return record.source_path.split("#", 1)[0]


def _prompt_fingerprint(record: NormalizedRecord) -> str:
    if record.prompt_family != "unknown":
        return _hash(record.prompt_family)
    data = record.data
    for key in ("prompt", "prompt_text", "prompt_id", "prompt_hash"):
        if isinstance(data, Mapping) and key in data:
            return _hash(data[key])
    return "unknown"


def _model_workload_fingerprint(record: NormalizedRecord) -> str:
    if record.model_fingerprint == "unknown" or record.workload_fingerprint == "unknown":
        return "unknown"
    return _hash((record.model_fingerprint, record.workload_fingerprint))


@dataclass(frozen=True)
class LeakageReport:
    """All detected cross-split identity collisions."""

    content_collisions: tuple[tuple[str, tuple[str, ...]], ...] = ()
    source_collisions: tuple[tuple[str, tuple[str, ...]], ...] = ()
    prompt_collisions: tuple[tuple[str, tuple[str, ...]], ...] = ()
    model_workload_collisions: tuple[tuple[str, tuple[str, ...]], ...] = ()
    dirty_clean_mismatches: tuple[tuple[str, tuple[str, ...]], ...] = ()
    summary_only_labels: tuple[str, ...] = ()
    unknown_identity_records: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not any(
            (
                self.content_collisions,
                self.source_collisions,
                self.prompt_collisions,
                self.model_workload_collisions,
                self.dirty_clean_mismatches,
                self.summary_only_labels,
                self.unknown_identity_records,
            )
        )

    @property
    def has_leakage(self) -> bool:
        return not self.clean

    def as_dict(self) -> dict[str, Any]:
        return {
            "content_collisions": [[key, list(splits)] for key, splits in self.content_collisions],
            "source_collisions": [[key, list(splits)] for key, splits in self.source_collisions],
            "prompt_collisions": [[key, list(splits)] for key, splits in self.prompt_collisions],
            "model_workload_collisions": [[key, list(splits)] for key, splits in self.model_workload_collisions],
            "dirty_clean_mismatches": [[key, list(splits)] for key, splits in self.dirty_clean_mismatches],
            "summary_only_labels": list(self.summary_only_labels),
            "unknown_identity_records": list(self.unknown_identity_records),
            "clean": self.clean,
        }


def _cross_split(records: Sequence[NormalizedRecord], assignments: Mapping[str, str], key_fn) -> tuple[tuple[str, tuple[str, ...]], ...]:
    buckets: dict[str, set[str]] = {}
    for record in records:
        split = assignments.get(record.record_id)
        if split not in SPLITS:
            continue
        key = key_fn(record)
        if key == "unknown":
            continue
        buckets.setdefault(key, set()).add(split)
    return tuple(sorted((key, tuple(sorted(splits, key=SPLITS.index))) for key, splits in buckets.items() if len(splits) > 1))


def detect_leakage(records: Sequence[NormalizedRecord], assignments: Mapping[str, str]) -> LeakageReport:
    """Detect identity reuse and summary labels across explicit splits."""

    content = _cross_split(records, assignments, lambda r: r.content_fingerprint)
    source = _cross_split(records, assignments, _source_identity)
    prompt = _cross_split(records, assignments, _prompt_fingerprint)
    model_workload = _cross_split(
        records,
        assignments,
        _model_workload_fingerprint,
    )
    dirty_buckets: dict[str, dict[bool | None, set[str]]] = {}
    for record in records:
        split = assignments.get(record.record_id)
        if split not in SPLITS or record.dirty is None:
            continue
        # Dirty/clean is a provenance state for a source.  Do not include
        # content here: changed payloads are the very situation this mismatch
        # detector is meant to expose.
        key = _hash(_source_identity(record))
        dirty_buckets.setdefault(key, {}).setdefault(record.dirty, set()).add(split)
    dirty = tuple(
        sorted(
            (key, tuple(sorted({split for splits in values.values() for split in splits}, key=SPLITS.index)))
            for key, values in dirty_buckets.items()
            if True in values and False in values and len({split for splits in values.values() for split in splits}) > 1
        )
    )
    summary = tuple(
        sorted(record.record_id for record in records if record.quality == QualityClass.LEGACY_SUMMARY and record.label_fields)
    )
    unknown = tuple(
        sorted(
            record.record_id
            for record in records
            if assignments.get(record.record_id) in SPLITS and not record.identity_complete
        )
    )
    return LeakageReport(content, source, prompt, model_workload, dirty, summary, unknown)


@dataclass(frozen=True)
class DatasetCard:
    """Stable, JSON-compatible dataset metadata."""

    schema_version: int
    coverage: Mapping[str, Any]
    quality_counts: Mapping[str, int]
    missingness: Mapping[str, int]
    censoring: Mapping[str, int]
    duplicates: int
    quarantined: int
    split: Mapping[str, Any]
    known_limits: tuple[str, ...]
    claim: str
    smoke_only: bool
    leakage: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("coverage", "quality_counts", "missingness", "censoring", "split", "leakage"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        object.__setattr__(self, "known_limits", tuple(self.known_limits))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "coverage": _thaw(self.coverage),
            "quality_counts": _thaw(self.quality_counts),
            "missingness": _thaw(self.missingness),
            "censoring": _thaw(self.censoring),
            "duplicates": self.duplicates,
            "quarantined": self.quarantined,
            "split": _thaw(self.split),
            "known_limits": list(self.known_limits),
            "claim": self.claim,
            "smoke_only": self.smoke_only,
            "leakage": _thaw(self.leakage),
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)


@dataclass(frozen=True)
class DatasetSnapshot:
    """Immutable canonical dataset plus split/card metadata."""

    records: tuple[NormalizedRecord, ...]
    splits: Mapping[str, tuple[str, ...]]
    card: DatasetCard
    canonical_bytes: bytes
    sha256: str
    assignments: Mapping[str, str]
    leakage: LeakageReport

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_bytes, bytes):
            raise ValueError("snapshot canonical_bytes must be bytes")
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "splits", _freeze(self.splits))
        object.__setattr__(self, "assignments", _freeze(self.assignments))
        expected_ids = {record.record_id for record in self.records}
        if len(expected_ids) != len(self.records):
            raise ValueError("snapshot records must have unique record IDs")
        if tuple(self.records) != tuple(sorted(self.records, key=lambda r: (r.record_id, r.content_fingerprint, r.source_path))):
            raise ValueError("snapshot records are not in canonical order")
        split_keys = set(self.splits)
        if split_keys != set(SPLITS):
            raise ValueError("snapshot must contain exactly train/validation/holdout splits")
        flattened = [record_id for name in SPLITS for record_id in self.splits[name]]
        if len(flattened) != len(set(flattened)) or not set(flattened).issubset(expected_ids):
            raise ValueError("snapshot split IDs are not unique known records")
        if set(self.assignments) != set(flattened):
            raise ValueError("snapshot assignments do not cover split IDs exactly")
        if self.card.get("leakage") != self.leakage.as_dict():
            raise ValueError("snapshot card leakage does not match leakage report")
        for name in SPLITS:
            if any(self.assignments[record_id] != name for record_id in self.splits[name]):
                raise ValueError("snapshot assignment disagrees with split membership")
        body = {
            "schema_version": 1,
            "records": [_snapshot_record(record) for record in self.records],
            "splits": {name: list(self.splits[name]) for name in SPLITS},
            "card": self.card.as_dict(),
        }
        expected_bytes = _canonical(body)
        if self.canonical_bytes != expected_bytes:
            raise ValueError("snapshot canonical bytes do not match records/card/splits")
        if self.sha256 != hashlib.sha256(expected_bytes).hexdigest():
            raise ValueError("snapshot hash does not match canonical bytes")

    @property
    def dataset_hash(self) -> str:
        return self.sha256

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "records": [_snapshot_record(record) for record in self.records],
            "splits": {name: list(self.splits.get(name, ())) for name in SPLITS},
            "card": self.card.as_dict(),
            "sha256": self.sha256,
        }


def _group_key(record: NormalizedRecord) -> tuple[str, ...]:
    """Return the mandatory anti-leakage group dimensions."""

    return (
        _source_identity(record),
        record.study_id,
        record.run_id,
        record.observed_time,
        record.hardware_fingerprint,
        record.model_fingerprint,
        record.workload_fingerprint,
        record.prompt_family,
    )


def _group_components(records: Sequence[NormalizedRecord]) -> tuple[tuple[tuple[str, ...], ...], ...]:
    """Build connected leakage groups across each required dimension.

    A source can contain several runs and a model/workload can recur in several
    source files.  Treating the eight dimensions as one tuple would allow the
    same source or model/workload pair to land in different splits.  Connected
    components make the anti-leakage rule explicit while still ignoring the
    uninformative ``unknown`` sentinel.
    """

    keys = [_group_key(record) for record in records]
    parent = list(range(len(keys)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    owners: dict[tuple[int, str], int] = {}
    for index, key in enumerate(keys):
        for dimension, value in enumerate(key):
            if value == "unknown":
                continue
            owner = owners.get((dimension, value))
            if owner is None:
                owners[(dimension, value)] = index
            else:
                union(index, owner)
    components: dict[int, list[tuple[str, ...]]] = {}
    for index, key in enumerate(keys):
        components.setdefault(find(index), []).append(key)
    return tuple(
        tuple(sorted(component))
        for component in sorted(components.values(), key=lambda values: _hash(values))
    )


class DatasetBuilder:
    """Build byte-identical snapshots without random row splitting."""

    def __init__(self, records: CorpusInventory | Iterable[NormalizedRecord] | None = None):
        if isinstance(records, CorpusInventory):
            records = records.records
        self.records = tuple(records or ())

    @staticmethod
    def group_key(record: NormalizedRecord) -> tuple[str, ...]:
        return _group_key(record)

    def _assign_groups(self, eligible: Sequence[NormalizedRecord]) -> dict[str, str]:
        groups: dict[tuple[tuple[str, ...], ...], list[NormalizedRecord]] = {}
        key_to_component: dict[tuple[str, ...], tuple[tuple[str, ...], ...]] = {}
        for component in _group_components(eligible):
            for key in component:
                key_to_component[key] = component
        for record in eligible:
            component = key_to_component[_group_key(record)]
            groups.setdefault(component, []).append(record)
        ordered = sorted(groups, key=lambda key: _hash(key))
        count = len(ordered)
        if count <= 1:
            allocation = (count, 0, 0)
        elif count == 2:
            allocation = (1, 0, 1)
        else:
            holdout = max(1, round(count * 0.2))
            validation = max(1, round(count * 0.1))
            if holdout + validation >= count:
                validation = 1
                holdout = 1
            allocation = (count - validation - holdout, validation, holdout)
        assignment: dict[str, str] = {}
        cursor = 0
        for split, size in zip(SPLITS, allocation):
            for group in ordered[cursor : cursor + size]:
                for record in groups[group]:
                    assignment[record.record_id] = split
            cursor += size
        return assignment

    def build(
        self,
        records: CorpusInventory | Iterable[NormalizedRecord] | None = None,
        *,
        assignments: Mapping[str, str] | None = None,
    ) -> DatasetSnapshot:
        source_records = records
        if isinstance(source_records, CorpusInventory):
            source_records = source_records.records
        all_records = tuple(
            sorted(
                source_records if source_records is not None else self.records,
                key=lambda r: (r.record_id, r.content_fingerprint, r.source_path),
            )
        )
        eligible = tuple(
            record
            for record in all_records
            if record.training_eligible
            and record.quality not in {QualityClass.INVALID, QualityClass.QUARANTINED, QualityClass.LEGACY_SUMMARY}
        )
        # Exact content-addressed dedupe is performed only for learning rows;
        # excluded evidence stays visible in the snapshot/card.
        unique: list[NormalizedRecord] = []
        seen: set[str] = set()
        duplicate_count = 0
        for record in eligible:
            if record.content_fingerprint in seen:
                duplicate_count += 1
                continue
            seen.add(record.content_fingerprint)
            unique.append(record)
        eligible = tuple(unique)
        if assignments is None:
            assignment = self._assign_groups(eligible)
        else:
            eligible_ids = {record.record_id for record in eligible}
            unexpected = set(assignments) - eligible_ids
            if unexpected:
                raise ValueError("explicit assignments include an ineligible or unknown-identity record")
            if any(not isinstance(key, str) or not isinstance(value, str) for key, value in assignments.items()):
                raise ValueError("explicit assignment keys and split names must be strings")
            assignment = {key: value for key, value in assignments.items() if key in {r.record_id for r in eligible}}
            invalid = set(assignment.values()) - set(SPLITS)
            if invalid:
                raise ValueError(f"unknown dataset split(s): {sorted(invalid)}")
            missing = {record.record_id for record in eligible} - set(assignment)
            if missing:
                raise ValueError("explicit assignments must cover every eligible record")
            # Explicit assignments are still group constrained; silently
            # accepting a row split would make the card misleading.
            groups: dict[tuple[tuple[str, ...], ...], set[str]] = {}
            component_by_key: dict[tuple[str, ...], tuple[tuple[str, ...], ...]] = {}
            for component in _group_components(eligible):
                for key in component:
                    component_by_key[key] = component
            for record in eligible:
                component = component_by_key[_group_key(record)]
                groups.setdefault(component, set()).add(assignment[record.record_id])
            if any(len(splits) > 1 for splits in groups.values()):
                raise ValueError("a leakage group cannot be assigned to multiple splits")
        split_ids = {name: tuple(sorted((rid for rid, split in assignment.items() if split == name))) for name in SPLITS}
        leakage = detect_leakage(eligible, assignment)
        quality_counts = {quality.value: 0 for quality in QualityClass}
        for record in all_records:
            quality_counts[record.quality.value] += 1
        missingness: dict[str, int] = {}
        censoring: dict[str, int] = {}
        for record in all_records:
            for field in record.missing_fields:
                missingness[field] = missingness.get(field, 0) + 1
            for field in record.censored_fields:
                censoring[field] = censoring.get(field, 0) + 1
        missingness = dict(sorted(missingness.items()))
        censoring = dict(sorted(censoring.items()))
        has_all_splits = all(split_ids[name] for name in SPLITS)
        smoke_only = len(eligible) < 3 or not has_all_splits or leakage.has_leakage
        known_limits = {
            "groups, not rows, determine train/validation/holdout membership",
            "invalid, quarantined and summary-only evidence is excluded from labels",
            "no performance or generalisation claim is made from a smoke-only snapshot",
        }
        if not leakage.clean:
            known_limits.add("leakage detector reported one or more cross-split collisions")
        card = DatasetCard(
            schema_version=1,
            coverage={
                "records_seen": len(all_records),
                "eligible_records": len(eligible),
                "groups": len(_group_components(eligible)),
                "sources": len({_source_identity(record) for record in all_records}),
                "models": len({record.model_fingerprint for record in eligible}),
                "hardware": len({record.hardware_fingerprint for record in eligible}),
            },
            quality_counts=dict(sorted(quality_counts.items())),
            missingness=missingness,
            censoring=censoring,
            duplicates=duplicate_count,
            quarantined=quality_counts[QualityClass.QUARANTINED.value],
            split={"groups": {name: sum(1 for component in _group_components(eligible) if any(assignment.get(r.record_id) == name for r in eligible if _group_key(r) in component)) for name in SPLITS}, "records": {name: len(split_ids[name]) for name in SPLITS}},
            known_limits=tuple(sorted(known_limits)),
            claim="no_learning_claim" if smoke_only else "offline_snapshot_only",
            smoke_only=smoke_only,
            leakage=leakage.as_dict(),
        )
        body = {
            "schema_version": 1,
            "records": [_snapshot_record(record) for record in all_records],
            "splits": {name: list(split_ids[name]) for name in SPLITS},
            "card": card.as_dict(),
        }
        encoded = _canonical(body)
        digest = hashlib.sha256(encoded).hexdigest()
        return DatasetSnapshot(all_records, split_ids, card, encoded, digest, dict(sorted(assignment.items())), leakage)

    build_snapshot = build


__all__ = [
    "DatasetBuilder",
    "DatasetCard",
    "DatasetSnapshot",
    "LeakageReport",
    "SPLITS",
    "detect_leakage",
]
