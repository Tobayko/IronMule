"""Offline portfolio contract tests; no cache or model resolver is used."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from friday_optimizer.candidates import CandidateRegistry
from friday_optimizer.ironmule_adapter import EXECUTION_FILE_REGISTRY_HASH
from friday_optimizer.portfolio import MANIFEST_SCHEMA, MODEL_IDS, STATUSES, PortfolioError, build_portfolio


def digest(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def identity(size: str) -> dict:
    return {"revision": f"rev-{size}", "manifest_sha256": digest("manifest-" + size), "tokenizer_sha256": digest("tokenizer-" + size), "architecture": "gemma3_text" if size == "1b" else "gemma3", "quant_bits": 4, "quant_group_size": 64, "identity_sha256": digest("identity-" + size), "identity_document_sha256": digest("identity-document-" + size)}


def prereg(size: str, ident: dict, workload: str) -> dict:
    return {"schema": "friday.optimizer.preregistration.v1", "status": "SEALED", "fingerprint_hash": digest("fingerprint-" + size), "model_identity_sha256": ident["identity_sha256"], "model_manifest_sha256": ident["manifest_sha256"], "tokenizer_sha256": ident["tokenizer_sha256"], "workload_contract_sha256": workload, "candidate_id": "combined_core_profile", "optimizer_head": digest("head-" + size)[:40], "code_manifest_sha256": digest("code-" + size), "registry_sha256": EXECUTION_FILE_REGISTRY_HASH, "result_schema": "friday.optimizer.session-result.v1", "start_authorization_schema": "friday.optimizer.start-authorization.v1", "duration_minutes": 5, "stages": ["calibrate", "test"], "preregistration_sha256": digest("prereg-" + size)}


def evidence(size: str, ident: dict, workload: str, *, quality: str = "engineering") -> dict:
    return {"sha256": digest("evidence-" + size), "quality": quality, "model_identity_hash": ident["identity_sha256"], "hardware_hash": digest("hardware-" + size), "workload_hash": workload, "correctness": True, "resources": True, "raw_pairs": [{"pair": 1}, {"pair": 2}, {"pair": 3}]}


def manifest(*, readiness: str = "ready", include_prereg: bool = True) -> dict:
    rows = []
    for size in ("1b", "4b", "12b"):
        ident = identity(size)
        workload = digest("workload-" + size)
        hardware = digest("hardware-" + size)
        rows.append({"size": size, "model_id": MODEL_IDS[size], "cache_status": "verified", "identity": ident, "evidence": [evidence(size, {**ident, "hardware_hash": hardware, "workload_hash": workload}, workload)], "preregistration": prereg(size, ident, workload) if include_prereg else None, "preregistration_sha256": digest("prereg-" + size), "readiness": readiness, "hardware_hash": hardware, "workload_hash": workload})
    legacy_identity = identity("27b")
    rows.append({"size": "27b", "model_id": MODEL_IDS["27b"], "cache_status": "missing", "identity": None, "evidence": [evidence("27b", {**legacy_identity, "hardware_hash": digest("hardware-27b"), "workload_hash": digest("workload-27b")}, digest("workload-27b"), quality="legacy_summary")], "preregistration": None, "readiness": "unknown"})
    return {"schema": MANIFEST_SCHEMA, "version": 1, "models": rows, "candidate_id": "combined_core_profile", "registry_hash": CandidateRegistry().registry_hash, "cache_inventory_sha256": digest("cache-inventory"), "evidence_inventory_sha256": digest("evidence-inventory"), "readiness_evidence_sha256": digest("readiness-inventory")}


def test_all_four_cells_are_separate_and_next_point_is_deterministic() -> None:
    first = build_portfolio(manifest())
    second = build_portfolio(manifest())
    assert first.canonical_bytes == second.canonical_bytes
    assert [entry.size for entry in first.entries] == ["1b", "4b", "12b", "27b"]
    assert {entry.status for entry in first.entries} == {"ready_for_experiment", "missing_local_model"}
    assert first.next_safe_measurement == {"action": "aa_calibration", "size": "1b", "requires_user_start": True}
    assert first.entries[3].evidence_counts["legacy_summary"] == 1
    assert first.entries[3].usable_records == 0
    assert first.as_dict()["no_model_load"] is True
    assert first.as_dict()["no_download"] is True
    assert first.as_dict()["no_activation"] is True


@pytest.mark.parametrize("readiness,expected", [("blocked", "waiting_readiness"), ("unknown", "waiting_readiness")])
def test_readiness_blocks_without_reclassifying_static_identity(readiness: str, expected: str) -> None:
    result = build_portfolio(manifest(readiness=readiness))
    assert all(entry.status == expected or entry.status == "missing_local_model" for entry in result.entries)
    assert result.next_safe_measurement == {"action": "recheck_readiness", "size": "1b", "requires_user_start": False}


def test_verified_without_sealed_prereg_is_insufficient() -> None:
    result = build_portfolio(manifest(include_prereg=False))
    assert all(entry.status == "insufficient_evidence" or entry.status == "missing_local_model" for entry in result.entries)


def test_quality_and_cross_identity_mismatches_are_not_usable() -> None:
    value = manifest()
    value["models"][0]["evidence"] = [evidence("1b", identity("1b"), digest("other"), quality="exploratory")]
    result = build_portfolio(value)
    one = result.entries[0]
    assert one.status == "ready_for_experiment"
    assert one.usable_records == 0


def test_blocked_preregistered_cell_waits_even_without_usable_records() -> None:
    value = manifest(readiness="blocked")
    value["models"][1]["evidence"] = []
    result = build_portfolio(value)
    assert result.entries[1].status == "waiting_readiness"
    assert result.entries[1].usable_records == 0


def test_ready_preregistered_cell_does_not_require_historical_records() -> None:
    value = manifest()
    value["models"][1]["evidence"] = []
    result = build_portfolio(value)
    assert result.entries[1].status == "ready_for_experiment"
    assert result.entries[1].usable_records == 0


def test_provenance_hashes_are_required_and_bound_to_snapshot() -> None:
    value = manifest()
    first = build_portfolio(value)
    value["cache_inventory_sha256"] = digest("different-cache")
    second = build_portfolio(value)
    assert first.snapshot_sha256 != second.snapshot_sha256
    assert first.as_dict()["cache_inventory_sha256"] == digest("cache-inventory")
    assert first.as_dict()["evidence_inventory_sha256"] == digest("evidence-inventory")
    assert first.entries[0].hardware_hash == digest("hardware-1b")
    assert first.entries[0].workload_hash == digest("workload-1b")
    value = manifest()
    del value["evidence_inventory_sha256"]
    with pytest.raises(PortfolioError, match="evidence_inventory"):
        build_portfolio(value)
    value = manifest()
    value["readiness_evidence_sha256"] = digest("different-readiness")
    assert first.snapshot_sha256 != build_portfolio(value).snapshot_sha256


def test_tampered_registry_is_rejected() -> None:
    value = manifest()
    value["registry_hash"] = digest("wrong-registry")
    with pytest.raises(PortfolioError, match="registry"):
        build_portfolio(value)


@pytest.mark.parametrize("field", ["candidate_id", "registry_hash"])
def test_manifest_requires_explicit_candidate_and_registry(field: str) -> None:
    value = manifest()
    del value[field]
    with pytest.raises(PortfolioError, match="candidate|registry"):
        build_portfolio(value)


def test_mismatched_preregistration_document_hash_is_insufficient() -> None:
    value = manifest()
    value["models"][0]["preregistration_sha256"] = digest("different-prereg")
    assert build_portfolio(value).entries[0].status == "insufficient_evidence"


def test_preregistration_execution_registry_is_separate_from_manifest_registry() -> None:
    value = manifest()
    assert build_portfolio(value).entries[0].status == "ready_for_experiment"
    value = manifest()
    value["models"][0]["preregistration"].pop("registry_sha256")
    assert build_portfolio(value).entries[0].status == "insufficient_evidence"
    value = manifest()
    value["models"][0]["preregistration"]["registry_sha256"] = digest("wrong-execution-registry")
    assert build_portfolio(value).entries[0].status == "insufficient_evidence"


def test_malformed_identity_becomes_unsupported_and_private_fields_rejected() -> None:
    value = manifest()
    value["models"][0]["identity"] = {"revision": "rev", "manifest_sha256": "bad", "tokenizer_sha256": digest("tok"), "architecture": "gemma3", "identity_sha256": digest("id")}
    assert build_portfolio(value).entries[0].status == "unsupported"
    value = manifest()
    value["models"][0]["evidence"][0]["prompt"] = "secret"
    with pytest.raises(PortfolioError, match="unknown_field|private"):
        build_portfolio(value)


def test_json_source_requires_canonical_strict_bytes(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.json"
    path.write_text(json.dumps(manifest(), sort_keys=True, separators=(",", ":")))
    assert build_portfolio(path).entries[0].status == "ready_for_experiment"
    path.write_text(json.dumps(manifest()) + "\n")
    with pytest.raises(PortfolioError, match="canonical"):
        build_portfolio(path)


def test_json_source_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "real.json"
    target.write_text(json.dumps(manifest(), sort_keys=True, separators=(",", ":")))
    link = tmp_path / "portfolio.json"
    link.symlink_to(target)
    with pytest.raises(PortfolioError, match="source_invalid"):
        build_portfolio(link)


def test_model_ids_and_sizes_are_closed() -> None:
    value = manifest()
    value["models"][0]["model_id"] = "mlx-community/other"
    with pytest.raises(PortfolioError, match="model_id"):
        build_portfolio(value)
    value = manifest()
    value["models"][0]["size"] = "7b"
    with pytest.raises(PortfolioError, match="size"):
        build_portfolio(value)


def test_gemma3_text_architecture_is_only_allowed_for_1b() -> None:
    value = manifest()
    value["models"][1]["identity"]["architecture"] = "gemma3_text"
    assert build_portfolio(value).entries[1].status == "unsupported"
