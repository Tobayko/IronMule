from __future__ import annotations

from dataclasses import replace

from friday_optimizer.corpus import NormalizedRecord, QualityClass
from friday_optimizer.dataset import DatasetBuilder, detect_leakage


def _record(index: int, *, source: str = "source.json", dirty: bool | None = None, quality: QualityClass = QualityClass.ENGINEERING) -> NormalizedRecord:
    return NormalizedRecord(
        record_id=f"r{index}",
        source_path=source,
        source_kind="json",
        quality=quality,
        data={"study": f"s{index}", "run": f"run{index}", "model": "gemma-1b", "workload": f"w{index}", "timing": {"p50": index + 1}},
        feature_fields=("model", "workload"),
        label_fields=("timing.p50",),
        study_id=f"s{index}",
        run_id=f"run{index}",
        observed_time=f"t{index}",
        hardware_fingerprint="m1-max",
        model_fingerprint="gemma-1b-q4",
        workload_fingerprint=f"w{index}",
        prompt_family=f"p{index}",
        dirty=dirty,
        source_sha256="0" * 64,
        manifest_verified=True,
        contract_verified=True,
        identity_contract_valid=True,
        contract_id="fixture.contract.v1",
        contract_version=1,
        contract_hash="f" * 64,
        logical_source_file=source,
    )


def test_snapshot_is_deterministic_and_canonical() -> None:
    records = [_record(index) for index in range(8)]
    one = DatasetBuilder(records).build()
    two = DatasetBuilder(list(reversed(records))).build()
    assert one.canonical_bytes == two.canonical_bytes
    assert one.sha256 == two.sha256
    assert one.card["schema_version"] == 1


def test_group_split_has_no_leakage() -> None:
    records = [_record(index, source=f"source-{index // 2}.json") for index in range(9)]
    snapshot = DatasetBuilder(records).build()
    assert set(snapshot.splits) == {"train", "validation", "holdout"}
    assert snapshot.leakage.clean
    group_splits = {}
    for record in records:
        group = DatasetBuilder.group_key(record)
        split = snapshot.assignments.get(record.record_id)
        if split:
            group_splits.setdefault(group, set()).add(split)
    assert all(len(splits) == 1 for splits in group_splits.values())


def test_small_evidence_has_no_learning_claim() -> None:
    snapshot = DatasetBuilder([_record(1)]).build()
    assert snapshot.card["smoke_only"]
    assert snapshot.card["claim"] == "no_learning_claim"


def test_unknown_identity_cannot_enter_explicit_split() -> None:
    unknown = replace(_record(1), model_fingerprint="unknown")
    try:
        DatasetBuilder([unknown]).build(assignments={unknown.record_id: "train"})
    except ValueError as exc:
        assert "ineligible" in str(exc)
    else:
        raise AssertionError("unknown identity must be rejected")


def test_forged_snapshot_bytes_or_hash_are_rejected() -> None:
    snapshot = DatasetBuilder([_record(index) for index in range(4)]).build()
    from dataclasses import replace as dataclass_replace
    try:
        dataclass_replace(snapshot, sha256="0" * 64)
    except ValueError:
        pass
    else:
        raise AssertionError("forged snapshot hash must be rejected")


def test_duplicate_is_content_addressed_and_summary_not_label() -> None:
    duplicate = _record(1, source="a.json")
    same = replace(_record(1, source="b.json"), record_id="r3")
    summary = _record(2, quality=QualityClass.LEGACY_SUMMARY)
    snapshot = DatasetBuilder([duplicate, same, summary]).build()
    assert snapshot.card["duplicates"] == 1
    assert snapshot.card["quality_counts"]["legacy_summary"] == 1
    assert not any(summary.record_id in ids for ids in snapshot.splits.values())


def test_leakage_detector_catches_dirty_clean_and_cross_split_fingerprints() -> None:
    first = _record(1, source="same.json", dirty=True)
    second = _record(2, source="same.json", dirty=False)
    assignments = {first.record_id: "train", second.record_id: "holdout"}
    report = detect_leakage([first, second], assignments)
    assert report.source_collisions
    assert report.dirty_clean_mismatches
