"""Closed multi-process Phase-1B qualification and benchmark controller."""

from __future__ import annotations

import time
from typing import Any

from .constants import (
    AA_BOOTSTRAP_SEED,
    AA_MODE,
    AB_BOOTSTRAP_SEED,
    AB_MODE,
    CHARACTERIZE_MODE,
    CONTROLLER_TIMEOUT_SECONDS,
    CONTRACT_ID,
    EXPERIMENT_ID,
    MEMORY_REGRESSION_ALLOWANCE_BYTES,
    MINIMUM_IMPROVEMENT_FRACTION,
    MLX_MEMORY_LIMIT_BYTES,
    QUALIFICATION_MODE,
    SCHEMA_VERSION,
    WORKER_RSS_LIMIT_BYTES,
    WORKLOAD_ID,
)
from .kernel_source import (
    HIDDEN_SIZE,
    KERNEL_NAME,
    KERNEL_SOURCE_SHA256,
    ROWS,
)
from .statistics import hierarchical_ratio, select_baseline, timing_summary
from .supervisor import run_worker


def scope() -> dict[str, Any]:
    return {
        "contract_id": CONTRACT_ID,
        "workload_id": WORKLOAD_ID,
        "shape": [ROWS, HIDDEN_SIZE],
        "dtype": "float16",
        "source_sha256": KERNEL_SOURCE_SHA256,
        "kernel_name": KERNEL_NAME,
        "runtime_activation": False,
    }


def _base_report(run_id: str, kind: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "kind": kind,
        "status": "running",
        "formal_claim": False,
        "action": "baseline_fallback",
        "scope": scope(),
        "metrics": {},
    }


def _envelope_resource_ok(envelope: dict[str, Any]) -> bool:
    if not envelope.get("ok") or not isinstance(envelope.get("result"), dict):
        return False
    worker_rss = envelope["result"].get("process", {}).get("rss_peak_bytes")
    supervisor_rss = envelope.get("supervisor", {}).get("rss_peak_bytes")
    values = [value for value in (worker_rss, supervisor_rss) if isinstance(value, int)]
    return bool(values) and max(values) < WORKER_RSS_LIMIT_BYTES


def qualification_report(run_id: str, provenance: dict[str, Any]) -> dict[str, Any]:
    report = _base_report(run_id, "qualification")
    started = time.monotonic_ns()
    deadline = started + int(CONTROLLER_TIMEOUT_SECONDS * 1e9)
    envelope = run_worker(
        QUALIFICATION_MODE,
        0,
        controller_deadline_ns=deadline,
        expected_provenance=provenance,
    )
    passed = bool(
        _envelope_resource_ok(envelope)
        and isinstance(envelope.get("result"), dict)
        and envelope["result"].get("status") == "passed"
        and envelope["result"].get("evidence", {}).get("passed") is True
    )
    report["status"] = "qualification_passed" if passed else "qualification_failed"
    report["action"] = "qualification_only" if passed else "baseline_fallback"
    report["metrics"] = {
        "gate_passed": passed,
        "worker": envelope,
        "controller_wall_ns": time.monotonic_ns() - started,
    }
    return report


def _stage_failure(
    report: dict[str, Any],
    *,
    stage: str,
    sessions: list[dict[str, Any]],
    started_ns: int,
) -> dict[str, Any]:
    report["status"] = "benchmark_failed_terminal"
    report["action"] = "baseline_fallback"
    report["metrics"].update(
        {
            "gate_passed": False,
            "failed_stage": stage,
            "failed_sessions": sessions,
            "controller_wall_ns": time.monotonic_ns() - started_ns,
        }
    )
    return report


def benchmark_report(run_id: str, provenance: dict[str, Any]) -> dict[str, Any]:
    report = _base_report(run_id, "benchmark")
    started = time.monotonic_ns()
    deadline = started + int(CONTROLLER_TIMEOUT_SECONDS * 1e9)
    metrics: dict[str, Any] = {
        "direction": "candidate_over_baseline",
        "minimum_improvement_fraction": MINIMUM_IMPROVEMENT_FRACTION,
    }
    report["metrics"] = metrics

    characterization = [
        run_worker(
            CHARACTERIZE_MODE,
            index,
            controller_deadline_ns=deadline,
            expected_provenance=provenance,
        )
        for index in range(3)
    ]
    metrics["characterization_sessions"] = characterization
    if not all(_envelope_resource_ok(session) for session in characterization):
        return _stage_failure(
            report,
            stage="baseline_characterization",
            sessions=characterization,
            started_ns=started,
        )
    characterization_samples = [
        session["result"]["evidence"]["timing"]["samples_ns"]
        for session in characterization
    ]
    selection = select_baseline(characterization_samples)
    selected = selection["selected"]
    metrics["baseline_selection"] = selection
    metrics["baseline_summaries"] = {
        name: timing_summary(
            [
                value
                for session in characterization_samples
                for value in session[name]
            ]
        )
        for name in characterization_samples[0]
    }

    aa_sessions = [
        run_worker(
            AA_MODE,
            index,
            selected,
            controller_deadline_ns=deadline,
            expected_provenance=provenance,
        )
        for index in range(3)
    ]
    metrics["aa_sessions"] = aa_sessions
    if not all(_envelope_resource_ok(session) for session in aa_sessions):
        return _stage_failure(
            report, stage="aa_worker", sessions=aa_sessions, started_ns=started
        )
    aa_pairs = [
        (
            session["result"]["evidence"]["timing"]["samples_ns"]["b"],
            session["result"]["evidence"]["timing"]["samples_ns"]["a"],
        )
        for session in aa_sessions
    ]
    aa = hierarchical_ratio(aa_pairs, seed=AA_BOOTSTRAP_SEED)
    aa_gates = {
        "point_in_envelope": 0.98 <= aa["ratio"] <= 1.02,
        "ci_in_envelope": 0.98 <= aa["ci95_low"] and aa["ci95_high"] <= 1.02,
        "ci_contains_one": aa["ci95_low"] <= 1.0 <= aa["ci95_high"],
        "sessions_in_envelope": all(
            0.95 <= value <= 1.05 for value in aa["session_ratios"]
        ),
    }
    aa["gates"] = aa_gates
    aa["passed"] = all(aa_gates.values())
    metrics["aa"] = aa
    if not aa["passed"]:
        report["status"] = "measurement_system_no_go"
        report["action"] = "baseline_fallback"
        metrics["gate_passed"] = False
        metrics["failed_stage"] = "aa_gate"
        metrics["ab_sessions"] = []
        metrics["controller_wall_ns"] = time.monotonic_ns() - started
        return report

    ab_sessions = [
        run_worker(
            AB_MODE,
            index,
            selected,
            controller_deadline_ns=deadline,
            expected_provenance=provenance,
        )
        for index in range(3)
    ]
    metrics["ab_sessions"] = ab_sessions
    if not all(_envelope_resource_ok(session) for session in ab_sessions):
        return _stage_failure(
            report, stage="ab_worker", sessions=ab_sessions, started_ns=started
        )
    ab_pairs = [
        (
            session["result"]["evidence"]["timing"]["samples_ns"]["candidate"],
            session["result"]["evidence"]["timing"]["samples_ns"]["baseline"],
        )
        for session in ab_sessions
    ]
    ab = hierarchical_ratio(ab_pairs, seed=AB_BOOTSTRAP_SEED)
    memory_sessions: list[dict[str, Any]] = []
    for session in ab_sessions:
        memory = session["result"]["evidence"]["memory"]
        candidate_peak = int(memory["candidate"]["peak_bytes"])
        baseline_peak = int(memory["baseline"]["peak_bytes"])
        worker_rss = session["result"]["process"]["rss_peak_bytes"]
        supervisor_rss = session["supervisor"]["rss_peak_bytes"]
        rss_peak = max(
            value for value in (worker_rss, supervisor_rss) if isinstance(value, int)
        )
        gates = {
            "candidate_mlx_peak": candidate_peak <= MLX_MEMORY_LIMIT_BYTES,
            "candidate_vs_baseline": candidate_peak
            <= baseline_peak + MEMORY_REGRESSION_ALLOWANCE_BYTES,
            "worker_rss": isinstance(rss_peak, int)
            and rss_peak < WORKER_RSS_LIMIT_BYTES,
        }
        memory_sessions.append(
            {
                "candidate_peak_bytes": candidate_peak,
                "baseline_peak_bytes": baseline_peak,
                "worker_rss_peak_bytes": rss_peak,
                "worker_reported_rss_peak_bytes": worker_rss,
                "supervisor_sampled_rss_peak_bytes": supervisor_rss,
                "gates": gates,
                "passed": all(gates.values()),
            }
        )
    memory_passed = all(value["passed"] for value in memory_sessions)
    ab_gates = {
        "minimum_improvement": ab["ratio"] <= 1.0 - MINIMUM_IMPROVEMENT_FRACTION,
        "ci_excludes_one": ab["ci95_high"] < 1.0,
        "no_bad_session": all(value <= 1.05 for value in ab["session_ratios"]),
        "memory": memory_passed,
    }
    ab["gates"] = ab_gates
    ab["passed"] = all(ab_gates.values())
    metrics["ab"] = ab
    metrics["memory_sessions"] = memory_sessions
    metrics["gate_passed"] = ab["passed"]
    metrics["controller_wall_ns"] = time.monotonic_ns() - started
    if ab["passed"]:
        report["status"] = "candidate_promoted_scope_only"
        report["action"] = "candidate_scope_eligible"
    else:
        report["status"] = (
            "candidate_regression" if ab["ratio"] > 1.0 else "candidate_inconclusive"
        )
        report["action"] = "baseline_fallback"
    return report
