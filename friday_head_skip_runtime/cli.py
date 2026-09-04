"""Closed command-line boundary for the bounded head-skip runtime."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .constants import (
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_FORMAL_DATABASE_PATH,
    DEFAULT_RUNTIME_DATABASE_PATH,
    FORMAL_DECISION_SHA256,
    GPU_RUN_ID,
    POLICY_MAX_LOAD_NS,
    POLICY_RUN_ID,
    QUALIFICATION_ID,
    RUNTIME_ID,
    SCHEMA_VERSION,
)
from .dashboard import DashboardService, serve
from .executor import RuntimeController
from .history import History
from .provenance import collect_provenance


class _UsageError(Exception):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise _UsageError


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="run_head_skip_runtime", allow_abbrev=False)
    parser.add_argument("--formal-database", default=str(DEFAULT_FORMAL_DATABASE_PATH))
    parser.add_argument("--database", default=str(DEFAULT_RUNTIME_DATABASE_PATH))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("policy", allow_abbrev=False)
    cpu = commands.add_parser("benchmark-policy", allow_abbrev=False)
    cpu.add_argument("--run-id")
    cpu.add_argument("--execute", action="store_true")
    gpu = commands.add_parser("validate-gpu", allow_abbrev=False)
    gpu.add_argument("--run-id")
    gpu.add_argument("--execute", action="store_true")
    snapshot = commands.add_parser("snapshot", allow_abbrev=False)
    snapshot.add_argument("--limit", type=int, default=64)
    dashboard = commands.add_parser("dashboard", allow_abbrev=False)
    dashboard.add_argument("--port", type=int, default=DEFAULT_DASHBOARD_PORT)
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _run_id(kind: str, supplied: str | None) -> str:
    expected = {
        "policy_overhead": POLICY_RUN_ID,
        "runtime_validation": GPU_RUN_ID,
    }.get(kind)
    if expected is None or (supplied is not None and supplied != expected):
        raise ValueError("run ID differs from the frozen qualification")
    return expected


def _released(args: argparse.Namespace) -> bool:
    if getattr(args, "execute", False):
        return True
    _print({"state": "not_released", "hint": "pass --execute"})
    return False


def _report(
    *, kind: str, run_id: str, status: str, measurement: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "runtime_id": RUNTIME_ID,
        "qualification_id": QUALIFICATION_ID,
        "formal_claim": False,
        "formal_decision_sha256": FORMAL_DECISION_SHA256,
        "kind": kind,
        "run_id": run_id,
        "status": status,
        **measurement,
    }


def _persist(
    database: Path, report: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, str]:
    with History.open(database, initialize=True) as history:
        outcome = history.persist(report, provenance)
    return {
        "record_id": outcome.record_id,
        "entity_key": outcome.entity_key,
        "persistence": outcome.state,
    }


def _stable_provenance(before: dict[str, Any]) -> None:
    after = collect_provenance(require_clean=True)
    if after["provenance_sha256"] != before["provenance_sha256"]:
        raise RuntimeError("runtime source or environment changed during measurement")


def _registered_path(value: str, expected: Path, name: str) -> Path:
    observed = Path(os.path.abspath(Path(value).expanduser()))
    if observed != expected.absolute():
        raise RuntimeError(f"{name} path differs from the frozen qualification")
    return observed


def _require_fresh_cpu_history(database: Path) -> None:
    if not database.exists():
        return
    with History.open(database, read_only=True) as history:
        with history.read_transaction():
            rows = history.verified_records()
    if rows:
        raise RuntimeError("CPU qualification has already been attempted")


def _attach_load_gate(measurement: dict[str, Any], policy_load_ns: int) -> None:
    if (
        isinstance(policy_load_ns, bool)
        or not isinstance(policy_load_ns, int)
        or policy_load_ns <= 0
    ):
        raise RuntimeError("policy load duration is invalid")
    metrics = measurement.get("metrics")
    thresholds = measurement.get("thresholds")
    if not isinstance(metrics, dict) or not isinstance(thresholds, dict):
        raise RuntimeError("runtime measurement lacks gate projections")
    load_gate = policy_load_ns <= POLICY_MAX_LOAD_NS
    measurement["policy_load_ns"] = policy_load_ns
    thresholds["policy_max_load_ns"] = POLICY_MAX_LOAD_NS
    metrics["policy_load_gate_passed"] = load_gate
    metrics["gate_passed"] = metrics.get("gate_passed") is True and load_gate


def _failure_report(
    *, run_id: str, command: str, failure: BaseException, controller: RuntimeController | None
) -> dict[str, Any]:
    return _report(
        kind="runtime_failure",
        run_id=run_id,
        status="measurement_failed",
        measurement={
            "failed_command": command,
            "failure_type": type(failure).__name__,
            "policy": asdict(controller.evidence) if controller is not None else None,
            "metrics": {"gate_passed": False},
        },
    )


def _measure(args: argparse.Namespace, *, kind: str, function: Any) -> int:
    run_id = _run_id(kind, args.run_id)
    before: dict[str, Any] | None = None
    controller: RuntimeController | None = None
    measurement_started = False
    try:
        formal_database = _registered_path(
            args.formal_database, DEFAULT_FORMAL_DATABASE_PATH, "formal database"
        )
        runtime_database = _registered_path(
            args.database, DEFAULT_RUNTIME_DATABASE_PATH, "runtime database"
        )
        before = collect_provenance(require_clean=True)
        started = time.perf_counter_ns()
        if kind == "policy_overhead":
            _require_fresh_cpu_history(runtime_database)
            controller = RuntimeController.for_cpu_qualification(formal_database)
        elif kind == "runtime_validation":
            controller = RuntimeController.for_gpu_qualification(
                formal_database,
                runtime_database,
                runtime_identity_provider=lambda: before,
            )
        else:
            raise RuntimeError("unknown qualification stage")
        policy_load_ns = time.perf_counter_ns() - started
        if not controller.evidence.authorized:
            raise RuntimeError(f"runtime policy unavailable: {controller.evidence.reason}")
        if kind == "runtime_validation":
            attempt = _report(
                kind="runtime_validation_attempt",
                run_id=GPU_RUN_ID,
                status="runtime_validation_started",
                measurement={
                    "policy": asdict(controller.evidence),
                    "metrics": {"gate_passed": False},
                },
            )
            _persist(runtime_database, attempt, before)
        measurement_started = True
        measurement = function(controller)
        if not isinstance(measurement, dict):
            raise RuntimeError("runtime measurement is not an object")
        _attach_load_gate(measurement, policy_load_ns)
        _stable_provenance(before)
        gate = measurement.get("metrics", {}).get("gate_passed") is True
        status = f"{kind}_passed" if gate else f"{kind}_failed_gate"
        report = _report(kind=kind, run_id=run_id, status=status, measurement=measurement)
        outcome = _persist(runtime_database, report, before)
        _print({"state": "measured", **outcome, "report": report})
        return 0 if gate else 2
    except Exception as exc:
        if before is not None and measurement_started:
            try:
                _stable_provenance(before)
                failure = _failure_report(
                    run_id=f"{run_id}-failure",
                    command=args.command,
                    failure=exc,
                    controller=controller,
                )
                _persist(runtime_database, failure, before)
            except Exception:
                pass
        _print({"state": "failed", "failure_type": type(exc).__name__})
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except _UsageError:
        _print({"state": "invalid_arguments"})
        return 64
    try:
        if args.command == "policy":
            formal_database = _registered_path(
                args.formal_database, DEFAULT_FORMAL_DATABASE_PATH, "formal database"
            )
            runtime_database = _registered_path(
                args.database, DEFAULT_RUNTIME_DATABASE_PATH, "runtime database"
            )
            controller = RuntimeController.from_evidence(
                formal_database, runtime_database
            )
            _print({"state": "loaded", "evidence": asdict(controller.evidence)})
            return 0 if controller.evidence.authorized else 2
        if args.command == "benchmark-policy":
            if not _released(args):
                return 78
            from .benchmark import benchmark_policy_overhead

            return _measure(args, kind="policy_overhead", function=benchmark_policy_overhead)
        if args.command == "validate-gpu":
            if not _released(args):
                return 78
            from .benchmark import run_mlx_validation

            return _measure(args, kind="runtime_validation", function=run_mlx_validation)
        if args.command == "snapshot":
            _print(DashboardService(Path(args.database)).snapshot(args.limit))
            return 0
        if args.command == "dashboard":
            serve(Path(args.database), port=args.port)
            return 0
    except Exception as exc:
        _print({"state": "failed", "failure_type": type(exc).__name__})
        return 1
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
