from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from friday_evidence.canonical import canonical_sha256
from friday_evidence.events import EventJournal
from friday_evidence.portable.contracts import spec_digest, write_json_new
from friday_evidence.portable.corpus import CorpusError, build_dataset, import_report
from friday_evidence.portable.learning import _holdout_state, train_and_evaluate


_CODE_SHA = "c" * 64
_MODEL_SHA = "d" * 64
_HARDWARE = {"accelerator": "test-control-plane", "device_id": "unit"}
_ENVIRONMENT = {"python": "test", "runtime": "none"}


def _case(partition: str, suffix: str, *, a_sha: str | None = None) -> dict:
    return {
        "case_id": f"case-{partition}-{suffix}",
        "a_file": f"a-{partition}-{suffix}.npy",
        "b_file": f"b-{partition}-{suffix}.npy",
        "a_sha256": a_sha or (suffix * 64)[:64],
        "b_sha256": (suffix * 64)[:64].lower(),
        "shape": [8, 16, 12],
        "dtype": "float32",
        "lineage_id": f"lineage-{partition}-{suffix}",
        "weight_sha256": ("e" + suffix * 63)[:64],
        "prompt_family": f"prompt-{partition}-{suffix}",
        "shape_family": f"shape-{partition}-{suffix}",
        "capture_session": f"session-{partition}-{suffix}",
        "model_id": "owner/model",
        "model_sha256": _MODEL_SHA,
        "source_kind": "real_model_capture",
        "partition": partition,
    }


def _spec(partition: str, run_id: str, case: dict) -> dict:
    return {
        "schema": "ironmule.experiment.v1",
        "run_id": run_id,
        "backend": "mlx",
        "mode": "measure",
        "partition": partition,
        "seed": 7,
        "warmup": 2 + 3,
        "pairs": 12,
        "server_seconds": 30,
        "work_seconds": 20,
        "candidates": ["native", "compiled"],
        "cases": [case],
        "code_sha256": _CODE_SHA,
    }


def _samples(ratio: float) -> list[dict]:
    return [
        {
            "order": "AB" if index % 2 == 0 else "BA",
            "baseline_seconds": 1.0 + index / 100,
            "candidate_seconds": (1.0 + index / 100) * ratio,
            "exact": True,
        }
        for index in range(12)
    ]


def _report(spec: dict, *, unsupported: bool = False) -> dict:
    digest = spec_digest(spec)
    trials = []
    candidates = ["native"] if unsupported else spec["candidates"]
    for candidate in candidates:
        status = "unsupported" if unsupported else "valid"
        trials.append({
            "run_id": spec["run_id"],
            "case_id": spec["cases"][0]["case_id"],
            "backend": spec["backend"],
            "candidate": candidate,
            "partition": spec["partition"],
            "case": deepcopy(spec["cases"][0]),
            "hardware": deepcopy(_HARDWARE),
            "environment": deepcopy(_ENVIRONMENT),
            "code_sha256": _CODE_SHA,
            "spec_sha256": digest,
            "status": status,
            "samples": [] if unsupported else _samples(1.0 if candidate == "native" else 0.9),
            "costs": {"discovery_wall_seconds": 0.0 if candidate == "native" else 0.2},
        })
    return {
        "schema": "ironmule.collection.v1",
        "run_id": spec["run_id"],
        "spec_sha256": digest,
        "backend": spec["backend"],
        "partition": spec["partition"],
        "hardware": deepcopy(_HARDWARE),
        "environment": deepcopy(_ENVIRONMENT),
        "code_sha256": _CODE_SHA,
        "status": "failed" if unsupported else "complete",
        "trials": trials,
        "performance_claim": False,
    }


def test_import_is_idempotent_and_sealed_tamper_is_rejected(tmp_path: Path) -> None:
    spec = _spec("train", "1" * 32, _case("train", "1"))
    report = _report(spec)
    first = import_report(tmp_path, report, spec)
    second = import_report(tmp_path, report, spec)
    assert first["status"] == "imported"
    assert second["status"] == "already_imported"
    assert first["report_sha256"] == second["report_sha256"]
    assert first["report_sha256"] == canonical_sha256(report)
    assert first["execution_verified"] is False
    assert not any("file" in key or "path" in key for key in first)

    changed = deepcopy(report)
    changed["status"] = "failed"
    with pytest.raises(CorpusError, match="run_id_already_bound"):
        import_report(tmp_path, changed, spec)

    sealed_report = tmp_path / "reports" / f"{first['report_sha256']}.json"
    sealed_report.write_text("{}", encoding="utf-8")
    with pytest.raises(CorpusError, match="hash_mismatch|artifact_invalid"):
        build_dataset(tmp_path)


def test_holdout_is_hidden_and_nonvalid_outcomes_never_get_labels(tmp_path: Path) -> None:
    train_spec = _spec("train", "2" * 32, _case("train", "2"))
    holdout_spec = _spec("holdout", "3" * 32, _case("holdout", "3"))
    unsupported_spec = _spec("validation", "4" * 32, _case("validation", "4"))
    import_report(tmp_path, _report(train_spec), train_spec)
    import_report(tmp_path, _report(holdout_spec), holdout_spec)
    import_report(tmp_path, _report(unsupported_spec, unsupported=True), unsupported_spec)

    before = sorted(
        (path.relative_to(tmp_path).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*") if path.is_file()
    )
    dataset = build_dataset(tmp_path)
    after = sorted(
        (path.relative_to(tmp_path).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*") if path.is_file()
    )
    assert after == before
    assert dataset["holdout"]["hidden"] is True
    assert dataset["holdout"]["record_count"] == 2
    assert all(record["partition"] != "holdout" for record in dataset["records"])
    unsupported = [record for record in dataset["records"] if record["status"] == "unsupported"]
    assert len(unsupported) == 1
    assert unsupported[0]["label"] is None
    assert all(record["label"] is None for record in dataset["records"])
    assert dataset["counts"]["execution_verified_reports"] == 0
    assert dataset["counts"]["by_status"] == {"unsupported": 1, "valid": 4}
    assert dataset["counts"]["raw_samples"] == 48
    assert dataset["counts"]["raw_samples_by_backend"] == {"mlx": 48}
    assert dataset["counts"]["unique_cases"] == 3
    assert all("sample_count" in record for record in dataset["records"])
    assert dataset["performance_claim"] is False


def test_connected_tensor_or_capture_metadata_cannot_cross_partitions(tmp_path: Path) -> None:
    shared_a = "a" * 64
    train_spec = _spec("train", "5" * 32, _case("train", "5", a_sha=shared_a))
    validation_case = _case("validation", "6")
    # Role changes do not make identical tensor bytes independent.
    validation_case["b_sha256"] = shared_a
    validation_spec = _spec("validation", "6" * 32, validation_case)
    import_report(tmp_path, _report(train_spec), train_spec)
    with pytest.raises(CorpusError, match="connected_capture_group_crosses_partitions"):
        import_report(tmp_path, _report(validation_spec), validation_spec)


def test_model_hash_is_a_stratum_not_a_group_and_pilot_has_no_claim(tmp_path: Path) -> None:
    train_spec = _spec("train", "7" * 32, _case("train", "7"))
    validation_spec = _spec("validation", "8" * 32, _case("validation", "8"))
    import_report(tmp_path, _report(train_spec), train_spec)
    # Same model hash is intentionally legal across partitions; grouping it
    # would collapse the complete within-backend split.
    import_report(tmp_path, _report(validation_spec), validation_spec)
    result = train_and_evaluate(tmp_path)
    assert result["status"] == "no_learning_claim"
    assert result["performance_claim"] is False
    assert result["replay_gate_passed"] is False
    assert result["model"] is None
    assert result["ood_policy"] == "no_recommendation"
    assert not (tmp_path / "models").exists()


def test_smoke_and_preworker_failure_are_history_not_performance_labels(tmp_path: Path) -> None:
    smoke_spec = _spec("train", "9" * 32, _case("train", "9"))
    smoke_spec["mode"] = "smoke"
    smoke_report = _report(smoke_spec)
    smoke_report["status"] = "smoke"
    smoke_report["trials"] = [smoke_report["trials"][0]]
    smoke_report["trials"][0]["samples"] = []
    smoke_report["trials"][0]["smoke_only"] = True
    import_report(tmp_path, smoke_report, smoke_spec)

    failed_spec = _spec("validation", "a" * 32, _case("validation", "a"))
    failed_report = {
        "schema": "ironmule.collection.v1",
        "run_id": failed_spec["run_id"],
        "spec_sha256": spec_digest(failed_spec),
        "backend": "mlx",
        "partition": "validation",
        "hardware": {},
        "environment": {},
        "code_sha256": _CODE_SHA,
        "status": "failed",
        "trials": [],
        "performance_claim": False,
    }
    import_report(tmp_path, failed_report, failed_spec)
    dataset = build_dataset(tmp_path)
    assert dataset["counts"]["reports"] == 2
    assert dataset["counts"]["records"] == 1
    assert dataset["counts"]["raw_samples"] == 0
    assert dataset["counts"]["unique_cases"] == 1
    assert dataset["records"][0]["label"] is None


def test_only_local_control_receipt_can_authorize_labels(tmp_path: Path) -> None:
    spec = _spec("train", "b" * 32, _case("train", "b"))
    report = _report(spec)
    report["verified_execution"] = True  # Untrusted incoming assertion.
    imported = import_report(tmp_path, report, spec)
    assert imported["execution_verified"] is False
    assert all(record["label"] is None for record in build_dataset(tmp_path)["records"])

    with EventJournal(tmp_path / "control.sqlite3") as journal:
        journal.append(spec["run_id"], "validation", {
            "verified_execution": True,
            "backend": "mlx",
            "spec_sha256": spec_digest(spec),
            "report_sha256": canonical_sha256(report),
            "terminal_state": "complete",
            "supervisor_sha256": "f" * 64,
        })
    verified = build_dataset(tmp_path)
    assert verified["counts"]["execution_verified_reports"] == 1
    assert all(record["execution_verified"] for record in verified["records"])
    assert all(record["label"] is not None for record in verified["records"])
    assert verified["performance_claim"] is False


def test_holdout_claim_is_one_shot_and_exactly_cacheable(tmp_path: Path) -> None:
    claim = {
        "schema": "ironmule.holdout-claim.v1",
        "holdout_identity_sha256": "1" * 64,
        "dataset_sha256": "2" * 64,
        "frozen_policy_sha256": "3" * 64,
        "policy_code_sha256": "4" * 64,
        "holdout_group_sha256s": ["6" * 64],
    }
    status, cached, result_path = _holdout_state(tmp_path, claim)
    assert (status, cached) == ("new", None)
    result = {
        "schema": "ironmule.learning.v1",
        "holdout_claim_sha256": canonical_sha256(claim),
        "replay_gate_passed": False,
        "performance_claim": False,
    }
    write_json_new(result_path, result)
    status, cached, _ = _holdout_state(tmp_path, claim)
    assert status == "cached"
    assert cached == result
    changed = {**claim, "frozen_policy_sha256": "5" * 64}
    status, cached, _ = _holdout_state(tmp_path, changed)
    assert (status, cached) == ("consumed_by_different_policy", None)
