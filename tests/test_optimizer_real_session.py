"""Offline contract tests for the real-session boundary."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from friday_optimizer.collector import CollectorReport, CurrentSnapshot, WorkloadContract
from friday_optimizer.fingerprint import EnvironmentFingerprint, ExactFingerprint, ModelFingerprint, WorkloadFingerprint
from friday_optimizer.real_session import EXECUTION_FILE_REGISTRY_HASH, FingerprintReport, Preregistration, RealSessionError, SessionExecutionOutcome, SessionPlan, SessionResult, RealSessionController, _atomic_new, _evaluate_worker, _result_event, _worker_hash, start_authorization_hash
from friday_optimizer.readiness import ProbeSnapshot, ReadinessPolicy
from friday_optimizer.session import AdapterResult
from friday_optimizer.ironmule_adapter import ParsedIronMuleResult


def _report(model_id: str = "google/gemma-3-4b-it") -> CollectorReport:
    exact = ExactFingerprint(
        EnvironmentFingerprint("Apple M1 Max", "Apple GPU", 32 * 1024**3, 10, "26.5.2", "0.32.0", "0.31.3", "3.12", "a" * 40),
        ModelFingerprint(model_id, "rev", "b" * 64, "gemma", 4, 64, "c" * 64),
        WorkloadFingerprint("default", "c" * 64, "generator", "short", 1, 1, 32, True, False, "ac", "interactive"),
    )
    return CollectorReport(
        current_snapshot=None, fingerprint=exact, model_source_sha256="d" * 64,
        workload_contract_sha256="e" * 64, profile_contract_sha256=None,
    )


def test_fingerprint_report_is_canonical_and_path_free() -> None:
    report = FingerprintReport.from_collector(_report())
    encoded = report.canonical_bytes
    assert hashlib.sha256(encoded).hexdigest() == hashlib.sha256(encoded).hexdigest()
    assert b"/Users/" not in encoded
    assert report.fingerprint_hash == _report().fingerprint_hash
    assert json.loads(encoded)["schema"] == "friday.optimizer.fingerprint-report.v1"


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("/Users/alice/models/gemma-3-4b-it", "<local-model>"),
        ("~/models/gemma-3-4b-it", "<local-model>"),
        ("../models/gemma-3-4b-it", "<local-model>"),
        (r"C:\\Users\\alice\\models\\gemma-3-4b-it", "<local-model>"),
        ("mlx-community/gemma-3-4b-it-4bit", "mlx-community/gemma-3-4b-it-4bit"),
    ],
)
def test_public_dict_redacts_local_model_id_without_changing_exact_report(model_id: str, expected: str) -> None:
    report = FingerprintReport.from_collector(_report(model_id))
    exact_before = report.as_dict()
    report_hash_before = report.report_hash
    fingerprint_hash_before = report.fingerprint_hash
    public = report.public_dict()
    assert public["report"]["fingerprint"]["model"]["model_id"] == expected
    assert report.as_dict() == exact_before
    assert report.report_hash == report_hash_before
    assert report.fingerprint_hash == fingerprint_hash_before


def test_atomic_new_is_0600_and_never_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "fingerprint.json"
    _atomic_new(target, b"{}")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    with pytest.raises(RealSessionError, match="output_exists"):
        _atomic_new(target, b"changed")
    assert target.read_bytes() == b"{}"


def test_preregistration_requires_sealed_and_exact_hashes() -> None:
    digest = lambda seed: hashlib.sha256(seed.encode()).hexdigest()
    values = {
        "schema": "friday.optimizer.preregistration.v1", "status": "SEALED",
        "experiment": "T1", "fingerprint_hash": digest("fp"), "code_hash": digest("code"),
        "dataset_hash": digest("dataset"), "checkout_head": "0123456789abcdef0123456789abcdef01234567",
        "optimizer_head": "0123456789abcdef0123456789abcdef01234567",
        "optimizer_tree_sha256": digest("tree"), "code_manifest_sha256": digest("manifest"),
        "adapter_sha256": digest("adapter"), "worker_sha256": digest("worker"),
        "registry_sha256": digest("registry"), "source_digest": digest("source"),
        "workload_file_sha256": digest("workload-file"), "workload_contract_sha256": digest("workload"),
        "model_identity_sha256": digest("model"), "model_manifest_sha256": digest("model-manifest"),
        "tokenizer_sha256": digest("tokenizer"), "candidate": "combined_core_profile", "duration_minutes": 5,
        "stages": ["calibrate", "test"], "result_schema": "friday.optimizer.session-result.v1",
        "start_authorization_schema": "friday.optimizer.start-authorization.v1",
    }
    parsed = Preregistration.load(values)
    assert parsed.sha256
    with pytest.raises(RealSessionError, match="not_sealed"):
        Preregistration.load(dict(values, status="OPEN"))


def test_preregistration_rejects_markdown_and_placeholders(tmp_path: Path) -> None:
    markdown = tmp_path / "PREREGISTRATION.md"
    markdown.write_text("# not machine readable")
    with pytest.raises(RealSessionError, match="json_required"):
        Preregistration.load(markdown)
    with pytest.raises(RealSessionError, match="field_missing"):
        Preregistration.load({"schema": "friday.optimizer.preregistration.v1", "status": "SEALED", "experiment": "x", "optimizer_head": "a" * 40})


def test_worker_qualified_without_raw_evidence_is_not_a_recommendation() -> None:
    class EvaluatorSpy:
        called = False
        def evaluate(self, *args, **kwargs):
            self.called = True
            class Decision:
                status = "qualified"
                qualified = True
                reasons = ()
                evidence_hash = "a" * 64
                baseline_ratios = {}
                confidence_intervals = {}
            return Decision()
    exact = _report().fingerprint
    assert exact is not None
    spy = EvaluatorSpy()
    decision, summary = _evaluate_worker(AdapterResult("qualified", {"status": "confirmed"}), exact=exact, candidate_id="combined_core_profile", evaluator=spy)  # type: ignore[arg-type]
    assert decision is not None and spy.called
    assert summary["qualified"] is False
    assert "raw_aa_evidence_missing" in summary["reasons"]
    assert "raw_ab_evidence_missing" in summary["reasons"]


def test_session_result_and_start_authorization_are_content_bound() -> None:
    body = {
        "schema": "friday.optimizer.session-result.v1", "experiment": "T1", "classification": "inconclusive",
        "run_ok": True, "recommendation_available": False, "no_activation": True,
        "session_id_hash": hashlib.sha256(b"s1").hexdigest(), "start_authorization_hash": "c" * 64, "created_at_utc": "2026-08-30T00:00:00Z", "transitions": [], "fingerprint_hash": "a" * 64,
        "preregistration_hash": "b" * 64, "candidate_id": "combined_core_profile",
    }
    measurement = {"schema": "friday.optimizer.measurement-evidence.v1", "status": "unavailable", "reason": "missing"}
    measurement["evidence_sha256"] = hashlib.sha256(json.dumps(measurement, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    body["measurement_evidence"] = measurement
    body["result_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    parsed = SessionResult.from_mapping(body)
    assert parsed.result_hash == body["result_hash"]
    assert start_authorization_hash("s1", 5, "combined_core_profile", True) != start_authorization_hash("s2", 5, "combined_core_profile", True)


def test_history_event_binds_persisted_result_hash_and_timestamp() -> None:
    result = {"result_hash": "d" * 64, "created_at_utc": "2026-08-30T00:00:00Z", "state": "baseline", "status": "inconclusive", "recommendation_available": False}
    event = _result_event(result, "e" * 64, code_hash="a" * 64, dataset_hash="b" * 64, prereg="c" * 64, fingerprint="f" * 64, candidate="combined_core_profile")
    assert event.evidence_hash == result["result_hash"]
    assert event.created_at == result["created_at_utc"]


def test_execution_outcome_is_separate_from_persisted_result() -> None:
    body = {"schema": "friday.optimizer.session-result.v1", "experiment": "T1", "classification": "inconclusive", "run_ok": True, "recommendation_available": False, "no_activation": True, "session_id_hash": hashlib.sha256(b"s1").hexdigest(), "start_authorization_hash": "c" * 64, "created_at_utc": "2026-08-30T00:00:00Z", "transitions": [], "fingerprint_hash": "a" * 64, "preregistration_hash": "b" * 64, "candidate_id": "combined_core_profile"}
    measurement = {"schema": "friday.optimizer.measurement-evidence.v1", "status": "unavailable", "reason": "missing"}
    measurement["evidence_sha256"] = hashlib.sha256(json.dumps(measurement, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    body["measurement_evidence"] = measurement
    body["result_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    outcome = SessionExecutionOutcome(body, False, "HistoryError", False)
    assert outcome.result["result_hash"] == body["result_hash"]
    assert outcome.persistence_ok is False


def test_authorized_fake_controller_evaluator_result_and_history(tmp_path: Path) -> None:
    """The complete path stays fake while exercising every authorization edge."""
    runtime_commit = "0123456789abcdef0123456789abcdef01234567"
    exact = ExactFingerprint(
        EnvironmentFingerprint("Apple M1", "Apple GPU", 32 * 1024**3, 10, "26.5.2", "0.32.0", "0.31.3", "3.12", runtime_commit),
        ModelFingerprint("google/gemma-3-4b-it", "rev", "b" * 64, "gemma", 4, 64, "c" * 64),
        WorkloadFingerprint("default", "c" * 64, "generator", "short", 1, 1, 32, True, False, "ac", "interactive"),
    )
    workload_path = tmp_path / "workload.json"
    workload_path.write_text(json.dumps({"schema": "friday.workload_contract.v1", "prompt_family": "default", "tokenizer": "c" * 64, "generator": "generator", "context_bucket": "short", "batch": 1, "concurrency": 1, "max_tokens": 32, "greedy": True, "prompt_logprobs": False, "power_mode": "ac", "mode": "interactive"}, sort_keys=True, separators=(",", ":")))
    model_path = tmp_path / "model.json"
    model_path.write_text("{}")
    workload_source = hashlib.sha256(workload_path.read_bytes()).hexdigest()
    contract_hash = WorkloadContract.from_json(workload_path).contract_hash
    fp_report = FingerprintReport.from_collector(CollectorReport(CurrentSnapshot(exact.environment, "ac", 32), exact, hashlib.sha256(model_path.read_bytes()).hexdigest(), workload_source, None))
    digest = lambda seed: hashlib.sha256(seed.encode()).hexdigest()
    prereg_map = {"schema": "friday.optimizer.preregistration.v1", "status": "SEALED", "experiment": "E2E", "fingerprint_hash": exact.fingerprint_hash, "code_hash": digest("code"), "dataset_hash": digest("dataset"), "checkout_head": runtime_commit, "optimizer_head": runtime_commit, "optimizer_tree_sha256": digest("tree"), "code_manifest_sha256": digest("manifest"), "adapter_sha256": digest("adapter"), "worker_sha256": _worker_hash(), "registry_sha256": EXECUTION_FILE_REGISTRY_HASH, "source_digest": digest("source"), "workload_file_sha256": workload_source, "workload_contract_sha256": contract_hash, "model_identity_sha256": digest("model"), "model_manifest_sha256": digest("manifest-model"), "tokenizer_sha256": digest("tokenizer"), "candidate": "combined_core_profile", "duration_minutes": 5, "stages": ["calibrate", "test"], "result_schema": "friday.optimizer.session-result.v1", "start_authorization_schema": "friday.optimizer.start-authorization.v1"}
    prereg = Preregistration.load(prereg_map)
    class Validation:
        head = runtime_commit
        source_digest = digest("source")
        interpreter_sha256 = digest("interpreter")
    class Stage:
        def cleanup(self): pass
    class Adapter:
        def __init__(self, _binding): self.authorized = []; self.calls = []
        def validate_checkout(self): return Validation()
        def plan_stage(self, stage, **_): self.calls.append("plan:" + stage); return Stage()
        def authorize_stage(self, _spec, sid): self.authorized.append(sid); return Stage()
        def run_stage(self, spec, **kw):
            if len(self.authorized) == 1:
                aa_left = [{"session_id": "aa", "pair_id": f"a{i}", "arm": "aa_left", "order": order, "fingerprint": exact.fingerprint_hash, "workload": "w", "ttft_seconds": 1, "decode_tps": 10, "tokens": 2, "total_ns": [100, 101], "prefill_ns": [50, 51], "decode_ns": [50, 50], "decode_steps": 2, "deterministic": True, "mlx_peak_bytes": 100, "token_hash": digest("tokens"), "count_hash": digest("count"), "text_equivalence_hash": digest("text")} for i, order in enumerate(("AB", "BA", "AB", "BA"))]
                aa_right = [{**row, "arm": "aa_right", "ttft_seconds": 1.0, "decode_tps": 10.0} for row in aa_left]
                raw_pairs = [{"pair_id": row["pair_id"], "order": row["order"], "left": row, "right": aa_right[i]} for i, row in enumerate(aa_left)]
                parsed = ParsedIronMuleResult("calibrate", runtime_commit, exact.fingerprint_hash, "combined_core_profile", True, 2, "eos", digest("response"), {"ttft_ms": 1.0, "decode_tokens_per_second": 10.0, "peak_memory_bytes": 100, "peak_rss_bytes": 100, "swap_delta_bytes": 0, "resource_gate_passed": True}, None, None, False, evidence={"aa_baseline_samples": aa_left, "aa_control_samples": aa_right, "raw_pairs": raw_pairs, "calibration": {"complete": True, "trial_count": 1, "trials": [{}], "baseline": {}, "candidate": {}, "evidence_sha256": digest("calibration")}})
                return AdapterResult("ok", parsed.as_dict())
            baseline = [{"session_id": "x", "pair_id": f"p{i}", "arm": "A", "order": order, "fingerprint": exact.fingerprint_hash, "workload": "w", "ttft_seconds": 1, "decode_tps": 10, "tokens": 2, "total_ns": [100, 101], "prefill_ns": [50, 51], "decode_ns": [50, 50], "decode_steps": 2, "deterministic": True, "mlx_peak_bytes": 100, "token_hash": digest("tokens"), "count_hash": digest("count"), "text_equivalence_hash": digest("text")} for i, order in enumerate(("AB", "BA", "AB", "BA"))]
            candidate = [{**row, "arm": "B", "ttft_seconds": .9, "decode_tps": 11} for row in baseline]
            raw_pairs = [{"pair_id": row["pair_id"], "order": row["order"], "left": row, "right": candidate[i]} for i, row in enumerate(baseline)]
            parsed = ParsedIronMuleResult("test", runtime_commit, exact.fingerprint_hash, "combined_core_profile", True, 2, "eos", digest("response"), {"ttft_ms": 1.0, "decode_tokens_per_second": 10.0, "peak_memory_bytes": 100, "peak_rss_bytes": 100, "swap_delta_bytes": 0, "resource_gate_passed": True}, None, None, True, evidence={"baseline_samples": baseline, "candidate_samples": candidate, "raw_pairs": raw_pairs, "calibration": {"diagnostic": True}})
            return AdapterResult("qualified", parsed.as_dict())
    class Evaluator:
        def evaluate(self, *args, **kwargs):
            self.kwargs = kwargs
            class Decision:
                status = "qualified"; qualified = True; reasons = (); evidence_hash = digest("eval"); baseline_ratios = {"ttft": .9}; confidence_intervals = {"ttft": (.8, .95)}
            return Decision()
    probe = lambda: ProbeSnapshot(ac_connected=True, low_power=False, swap_used_bytes=0, memory_available_bytes=100, memory_total_bytes=200, load_1m=.1, cpu_percent=1, workload_active=False, process_tree_readable=True)
    from friday_optimizer.memory import OptimizationMemoryV2
    memory = tmp_path / "memory.sqlite3"
    with OptimizationMemoryV2(memory): pass
    plan = SessionPlan(fp_report, prereg, "combined_core_profile", 5, runtime_commit, digest("source"), digest("interpreter"))
    identity = {"head": runtime_commit, "optimizer_tree_sha256": digest("tree"), "code_manifest_sha256": digest("manifest"), "adapter_sha256": digest("adapter"), "worker_sha256": _worker_hash(), "registry_sha256": EXECUTION_FILE_REGISTRY_HASH}
    controller = RealSessionController(plan, tmp_path, runtime_commit, "/bin/sh", model_path, workload_path, runtime_commit, "sid", memory, tmp_path / "result.json", True, collector=type("C", (), {"collect": lambda self, **_: CollectorReport(CurrentSnapshot(exact.environment, "ac", 32), exact, hashlib.sha256(model_path.read_bytes()).hexdigest(), workload_source, None)})(), probe=probe, adapter_factory=Adapter, binding_factory=lambda **_: object(), readiness_policy=ReadinessPolicy(sample_interval_seconds=0), identity_provider=lambda: identity, evaluator=Evaluator())
    outcome = controller.run()
    assert isinstance(outcome, SessionExecutionOutcome)
    persisted = json.loads((tmp_path / "result.json").read_text())
    assert dict(outcome.result) == persisted
    assert outcome.result["result_hash"] == persisted["result_hash"]
    assert outcome.result["recommendation_available"] is True
    assert outcome.result["measurement_evidence"]["status"] == "complete"
    assert outcome.result["measurement_evidence"]["calibration"]["pairs"][0]["arms"]["aa_baseline_samples"]["total_ns"] == [100, 101]
    serialized_measurement = json.dumps(outcome.result["measurement_evidence"])
    assert all(secret not in serialized_measurement for secret in ("logical_tokens", "token_ids", "pid", "stdout", "stderr", "prompt", "path"))
    assert outcome.history_written is True
    assert all(call not in {"activate", "canary"} for call in [])
