"""Closed command line for Phase-1B qualification, benchmark, and history."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .constants import (
    BENCHMARK_RUN_ID,
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_DATABASE_PATH,
    EXPERIMENT_ID,
    QUALIFICATION_RUN_ID,
    SCHEMA_VERSION,
)
from .dashboard import serve
from .experiment import benchmark_report, qualification_report, scope
from .history import History
from .provenance import collect_provenance


class CliError(RuntimeError):
    """A Phase-1B CLI transition is not authorized by the frozen contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", allow_abbrev=False)
    qualify = commands.add_parser("qualify", allow_abbrev=False)
    qualify.add_argument("--run-id", default=QUALIFICATION_RUN_ID)
    qualify.add_argument("--execute", action="store_true")
    benchmark = commands.add_parser("benchmark", allow_abbrev=False)
    benchmark.add_argument("--run-id", default=BENCHMARK_RUN_ID)
    benchmark.add_argument("--execute", action="store_true")
    dashboard = commands.add_parser("dashboard", allow_abbrev=False)
    dashboard.add_argument("--port", type=int, default=DEFAULT_DASHBOARD_PORT)
    return parser


def _print(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), sort_keys=True, separators=(",", ":")))


def _records(database: Path) -> list[dict[str, Any]]:
    if not os.path.lexists(database):
        return []
    with History.open(database, read_only=True) as history:
        with history.read_transaction():
            return history.verified_records()


def _require_fresh(database: Path, run_id: str) -> None:
    if any(row["report"]["run_id"] == run_id for row in _records(database)):
        raise CliError("run id is already terminal and cannot be repeated")


def _require_run_id(actual: str, expected: str) -> None:
    if actual != expected:
        raise CliError("run id differs from the frozen preregistration")


def _qualification_record(database: Path) -> dict[str, Any]:
    matches = [
        row
        for row in _records(database)
        if row["report"]["run_id"] == QUALIFICATION_RUN_ID
        and row["report"]["kind"] == "qualification"
        and row["report"]["status"] == "qualification_passed"
    ]
    if len(matches) != 1:
        raise CliError("benchmark requires exactly one passed qualification")
    return matches[0]


def _persist(database: Path, report: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    with History.open(database, initialize=True) as history:
        return history.persist(report, provenance).record_id


def _failure_report(run_id: str, stage: str, failure: BaseException) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "kind": "failure",
        "status": "controller_failed_terminal",
        "formal_claim": False,
        "action": "baseline_fallback",
        "scope": scope(),
        "metrics": {
            "stage": stage,
            "error_type": type(failure).__name__,
            "message": str(failure)[:512],
            "gate_passed": False,
        },
    }


def _not_released() -> int:
    _print({"state": "not_released", "hint": "pass --execute"})
    return 78


def _qualify(args: argparse.Namespace) -> int:
    if not args.execute:
        return _not_released()
    database = Path(args.database)
    _require_run_id(args.run_id, QUALIFICATION_RUN_ID)
    _require_fresh(database, args.run_id)
    provenance = collect_provenance(require_clean=True)
    try:
        report = qualification_report(args.run_id, provenance)
    except Exception as exc:
        report = _failure_report(args.run_id, "qualification", exc)
    record_id = _persist(database, report, provenance)
    _print({"record_id": record_id, "report": report})
    return 0 if report["status"] == "qualification_passed" else 2


def _benchmark(args: argparse.Namespace) -> int:
    if not args.execute:
        return _not_released()
    database = Path(args.database)
    _require_run_id(args.run_id, BENCHMARK_RUN_ID)
    _require_fresh(database, args.run_id)
    qualification = _qualification_record(database)
    provenance = collect_provenance(require_clean=True)
    if qualification["provenance"]["provenance_sha256"] != provenance["provenance_sha256"]:
        raise CliError("qualification and benchmark provenance differ")
    try:
        report = benchmark_report(args.run_id, provenance)
    except Exception as exc:
        report = _failure_report(args.run_id, "benchmark", exc)
    record_id = _persist(database, report, provenance)
    _print({"record_id": record_id, "report": report})
    return 0 if report["status"] == "candidate_promoted_scope_only" else 2


def _status(database: Path) -> int:
    records = _records(database)
    _print(
        {
            "experiment_id": EXPERIMENT_ID,
            "record_count": len(records),
            "records": [
                {
                    "record_id": row["record_id"],
                    "run_id": row["report"]["run_id"],
                    "kind": row["report"]["kind"],
                    "status": row["report"]["status"],
                    "action": row["report"]["action"],
                }
                for row in records
            ],
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    database = DEFAULT_DATABASE_PATH
    try:
        if args.command == "status":
            return _status(database)
        if args.command == "dashboard":
            serve(database, port=args.port)
            return 0
        if args.command == "qualify":
            args.database = str(database)
            return _qualify(args)
        args.database = str(database)
        return _benchmark(args)
    except (CliError, RuntimeError, OSError, ValueError) as exc:
        _print({"error": type(exc).__name__, "message": str(exc)[:512]})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
