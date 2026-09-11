"""Closed command line boundary for the AVO-lite shadow router."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from friday_runtime.constants import DEFAULT_H1_DATABASE_PATH
from friday_runtime_n10.constants import DEFAULT_N10_DATABASE_PATH

from .benchmark import benchmark_policy_overhead, validate_real_tensor_shadow
from .constants import (
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_DATABASE_PATH,
    ENFORCED_PLAN,
    POLICY_RUN_ID,
    ROUTER_ID,
    SCHEMA_VERSION,
    SHADOW_RUN_ID,
)
from .dashboard import DashboardService, serve
from .history import History, HistoryError
from .provenance import ProvenanceError, collect_provenance
from .router import ShadowRouter


class RouterCliError(RuntimeError):
    """The requested router command cannot proceed without weakening its contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--database", default=str(DEFAULT_DATABASE_PATH))
    parser.add_argument("--h1-database", default=str(DEFAULT_H1_DATABASE_PATH))
    parser.add_argument("--n10-database", default=str(DEFAULT_N10_DATABASE_PATH))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("policy", allow_abbrev=False)
    policy = commands.add_parser("benchmark-policy", allow_abbrev=False)
    policy.add_argument("--run-id", default=POLICY_RUN_ID)
    policy.add_argument("--execute", action="store_true")
    shadow = commands.add_parser("validate-shadow", allow_abbrev=False)
    shadow.add_argument("--run-id", default=SHADOW_RUN_ID)
    shadow.add_argument("--execute", action="store_true")
    snapshot = commands.add_parser("snapshot", allow_abbrev=False)
    snapshot.add_argument("--limit", type=int, default=100)
    dashboard = commands.add_parser("dashboard", allow_abbrev=False)
    dashboard.add_argument("--port", type=int, default=DEFAULT_DASHBOARD_PORT)
    return parser


def _print(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), sort_keys=True, separators=(",", ":")))


def _power_source() -> str | None:
    completed = subprocess.run(
        ["/usr/bin/pmset", "-g", "batt"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=3.0,
        check=False,
    )
    if completed.returncode != 0:
        return None
    text = completed.stdout.decode("utf-8", errors="replace")
    if "AC Power" in text:
        return "ac"
    if "Battery Power" in text:
        return "battery"
    return "unknown"


def _load_router(h1_database: str | Path, n10_database: str | Path) -> ShadowRouter:
    return ShadowRouter.from_evidence(
        n8_kwargs={"evidence_path": Path(h1_database)},
        n10_kwargs={"evidence_path": Path(n10_database)},
    )


def _records(database: Path) -> list[dict[str, Any]]:
    if not database.exists():
        return []
    with History.open(database, read_only=True) as history:
        with history.read_transaction():
            return history.verified_records()


def _require_fresh_kind(database: Path, kind: str) -> None:
    if any(row["report"]["kind"] == kind for row in _records(database)):
        raise RouterCliError(f"{kind} is already terminal and cannot be repeated")
    if any(row["report"]["kind"] == "router_failure" for row in _records(database)):
        raise RouterCliError("router history already contains a terminal failure")


def _require_run_id(actual: str, expected: str) -> None:
    if actual != expected:
        raise RouterCliError("run id differs from the frozen preregistration")


def _require_policy_gate(database: Path) -> None:
    records = _records(database)
    policy = [row for row in records if row["report"]["kind"] == "policy_overhead"]
    if len(policy) != 1 or policy[0]["report"]["status"] != "policy_overhead_passed":
        raise RouterCliError("shadow validation requires exactly one passed policy gate")


def _resource_metrics(started_wall_ns: int, started_cpu_ns: int) -> dict[str, object]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "process_wall_ns": time.perf_counter_ns() - started_wall_ns,
        "process_cpu_ns": time.process_time_ns() - started_cpu_ns,
        "rss_peak_bytes": int(usage.ru_maxrss),
        "power_source": _power_source(),
        "pid": os.getpid(),
    }


def _report(
    *,
    run_id: str,
    kind: str,
    status: str,
    router: ShadowRouter,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "router_id": ROUTER_ID,
        "run_id": run_id,
        "kind": kind,
        "status": status,
        "formal_claim": False,
        "decision_record_ids": router.decision_record_ids,
        "router": router.evidence_summary(),
        "metrics": dict(metrics),
    }


def _persist(database: Path, report: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    with History.open(database, initialize=True) as history:
        outcome = history.persist(report, provenance)
    return outcome.record_id


def _persist_terminal_failure(
    *,
    database: Path,
    run_id: str,
    stage: str,
    failure: Exception,
    provenance: Mapping[str, Any],
    router: ShadowRouter | None,
    started_wall: int,
    started_cpu: int,
) -> int:
    ids = router.decision_record_ids if router is not None else {"n8": "0" * 64, "n10": "0" * 64}
    state = (
        router.evidence_summary()
        if router is not None
        else {"ready": False, "enforced_plan": ENFORCED_PLAN, "stage": stage}
    )
    metrics = {
        "stage": stage,
        "error_type": type(failure).__name__,
        "message": str(failure)[:512],
        "gate_passed": False,
    }
    metrics.update(_resource_metrics(started_wall, started_cpu))
    report = {
        "schema_version": SCHEMA_VERSION,
        "router_id": ROUTER_ID,
        "run_id": run_id,
        "kind": "router_failure",
        "status": "router_failed_terminal",
        "formal_claim": False,
        "decision_record_ids": ids,
        "router": state,
        "metrics": metrics,
    }
    record_id = _persist(database, report, provenance)
    _print({"record_id": record_id, "report": report})
    return 1


def _not_released() -> int:
    _print({"state": "not_released", "hint": "pass --execute"})
    return 78


def _benchmark_policy(args: argparse.Namespace) -> int:
    if not args.execute:
        return _not_released()
    database = Path(args.database)
    _require_run_id(args.run_id, POLICY_RUN_ID)
    _require_fresh_kind(database, "policy_overhead")
    provenance = collect_provenance(require_clean=True)
    started_wall = time.perf_counter_ns()
    started_cpu = time.process_time_ns()
    router: ShadowRouter | None = None
    try:
        load_started = time.perf_counter_ns()
        router = _load_router(args.h1_database, args.n10_database)
        cold_load_ns = time.perf_counter_ns() - load_started
        metrics = benchmark_policy_overhead(router, cold_load_ns=cold_load_ns)
        metrics.update(_resource_metrics(started_wall, started_cpu))
    except Exception as exc:
        return _persist_terminal_failure(
            database=database,
            run_id=args.run_id,
            stage="policy_overhead",
            failure=exc,
            provenance=provenance,
            router=router,
            started_wall=started_wall,
            started_cpu=started_cpu,
        )
    status = "policy_overhead_passed" if metrics["gate_passed"] else "policy_overhead_failed"
    report = _report(
        run_id=args.run_id,
        kind="policy_overhead",
        status=status,
        router=router,
        metrics=metrics,
    )
    record_id = _persist(database, report, provenance)
    _print({"record_id": record_id, "report": report})
    return 0 if metrics["gate_passed"] else 2


def _validate_shadow(args: argparse.Namespace) -> int:
    if not args.execute:
        return _not_released()
    database = Path(args.database)
    _require_run_id(args.run_id, SHADOW_RUN_ID)
    _require_policy_gate(database)
    _require_fresh_kind(database, "shadow_validation")
    provenance = collect_provenance(require_clean=True)
    started_wall = time.perf_counter_ns()
    started_cpu = time.process_time_ns()
    router: ShadowRouter | None = None
    try:
        load_started = time.perf_counter_ns()
        router = _load_router(args.h1_database, args.n10_database)
        metrics = validate_real_tensor_shadow(router)
        metrics["cold_load_ns"] = time.perf_counter_ns() - load_started
        metrics.update(_resource_metrics(started_wall, started_cpu))
    except Exception as exc:
        return _persist_terminal_failure(
            database=database,
            run_id=args.run_id,
            stage="shadow_validation",
            failure=exc,
            provenance=provenance,
            router=router,
            started_wall=started_wall,
            started_cpu=started_cpu,
        )
    status = "shadow_router_validated" if metrics["gate_passed"] else "shadow_validation_failed"
    report = _report(
        run_id=args.run_id,
        kind="shadow_validation",
        status=status,
        router=router,
        metrics=metrics,
    )
    record_id = _persist(database, report, provenance)
    _print({"record_id": record_id, "report": report})
    return 0 if metrics["gate_passed"] else 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "policy":
            _print(_load_router(args.h1_database, args.n10_database).evidence_summary())
            return 0
        if args.command == "benchmark-policy":
            return _benchmark_policy(args)
        if args.command == "validate-shadow":
            return _validate_shadow(args)
        if args.command == "snapshot":
            _print(DashboardService(Path(args.database)).snapshot(args.limit))
            return 0
        if args.command == "dashboard":
            serve(Path(args.database), port=args.port)
            return 0
        raise RouterCliError("unknown router command")
    except (HistoryError, ProvenanceError, RouterCliError, RuntimeError, ValueError) as exc:
        _print({"state": "failed_closed", "error": type(exc).__name__, "message": str(exc)})
        return 1


__all__ = ["main"]
