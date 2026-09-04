"""Closed command-line boundary for formal H1-v2 sealing and execution."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .constants import (
    CALIBRATION,
    CONFIRMATION,
    DEFAULT_DATABASE_PATH,
    INTER_SESSION_COOLDOWN_SECONDS,
    SESSION_ORDER,
)
from .dashboard import DashboardService, serve
from .protocol import build_preregistration, study_specification, validate_preregistration
from .runner import (
    current_record,
    decide_study,
    execute_session,
    preflight_session,
    seal_confirmation,
    seal_preregistration,
    summarize_calibration,
)


class _UsageError(Exception):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise _UsageError


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="run_h1_v2", allow_abbrev=False)
    parser.add_argument("--database", default=str(DEFAULT_DATABASE_PATH))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("spec", allow_abbrev=False)
    commands.add_parser("self-check", allow_abbrev=False)
    seal = commands.add_parser("seal", allow_abbrev=False)
    seal.add_argument("--execute", action="store_true")
    preflight = commands.add_parser("preflight", allow_abbrev=False)
    preflight.add_argument("--stage", required=True, choices=(CALIBRATION, CONFIRMATION))
    preflight.add_argument("--id", required=True, choices=SESSION_ORDER)
    session = commands.add_parser("session", allow_abbrev=False)
    session.add_argument("--stage", required=True, choices=(CALIBRATION, CONFIRMATION))
    session.add_argument("--id", required=True, choices=SESSION_ORDER)
    session.add_argument("--execute", action="store_true")
    calibration = commands.add_parser("summarize-calibration", allow_abbrev=False)
    calibration.add_argument("--execute", action="store_true")
    confirmation = commands.add_parser("seal-confirmation", allow_abbrev=False)
    confirmation.add_argument("--execute", action="store_true")
    decision = commands.add_parser("decide", allow_abbrev=False)
    decision.add_argument("--execute", action="store_true")
    stage = commands.add_parser("run-stage", allow_abbrev=False)
    stage.add_argument("--stage", required=True, choices=(CALIBRATION, CONFIRMATION))
    stage.add_argument("--execute", action="store_true")
    snapshot = commands.add_parser("snapshot", allow_abbrev=False)
    snapshot.add_argument("--limit", type=int, default=50)
    dashboard = commands.add_parser("dashboard", allow_abbrev=False)
    dashboard.add_argument("--port", type=int, default=8768)
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _released(args: argparse.Namespace) -> bool:
    if getattr(args, "execute", False):
        return True
    _print({"state": "not_released", "hint": "pass --execute"})
    return False


def _outcome(value: Any) -> dict[str, Any]:
    return {
        "state": value.state,
        "record_id": value.record_id,
        "entity_key": value.entity_key,
    }


def _verified_cooldown() -> None:
    started = time.monotonic()
    time.sleep(INTER_SESSION_COOLDOWN_SECONDS)
    if time.monotonic() - started + 1e-9 < INTER_SESSION_COOLDOWN_SECONDS:
        raise RuntimeError("inter-session cooldown did not elapse")


def _run_stage(stage: str, database: Path) -> int:
    for index, session_id in enumerate(SESSION_ORDER):
        if index:
            _verified_cooldown()
        argv = [
            sys.executable,
            "-m",
            "friday_h1.cli",
            "--database",
            str(database),
            "session",
            "--stage",
            stage,
            "--id",
            session_id,
            "--execute",
        ]
        completed = subprocess.run(argv, cwd=Path(__file__).resolve().parents[1], check=False)
        if completed.returncode != 0:
            _print({"state": "session_failed", "stage": stage, "session_id": session_id})
            return completed.returncode
    outcome = summarize_calibration(database) if stage == CALIBRATION else decide_study(database)
    _print(_outcome(outcome))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except _UsageError:
        _print({"state": "invalid_arguments"})
        return 64
    database = Path(args.database)
    try:
        if args.command == "spec":
            spec = study_specification()
            _print({"study_specification": spec})
            return 0
        if args.command == "self-check":
            digest = "0" * 64
            prereg = build_preregistration(digest)
            validate_preregistration(prereg)
            _print(
                {
                    "self_check": "pass",
                    "study_id": prereg["study_id"],
                    "study_spec_sha256": prereg["study_spec_sha256"],
                }
            )
            return 0
        if args.command == "seal":
            if not _released(args):
                return 78
            _print(_outcome(seal_preregistration(database)))
            return 0
        if args.command == "preflight":
            _print({"state": "preflight_ok", **preflight_session(args.stage, args.id, database_path=database)})
            return 0
        if args.command == "session":
            if not _released(args):
                return 78
            _print(_outcome(execute_session(args.stage, args.id, database_path=database)))
            return 0
        if args.command == "summarize-calibration":
            if not _released(args):
                return 78
            _print(_outcome(summarize_calibration(database)))
            return 0
        if args.command == "seal-confirmation":
            if not _released(args):
                return 78
            _print(_outcome(seal_confirmation(database)))
            return 0
        if args.command == "decide":
            if not _released(args):
                return 78
            _print(_outcome(decide_study(database)))
            return 0
        if args.command == "run-stage":
            if not _released(args):
                return 78
            return _run_stage(args.stage, database)
        if args.command == "snapshot":
            _print(DashboardService(database).snapshot(args.limit))
            return 0
        if args.command == "dashboard":
            serve(database, port=args.port)
            return 0
    except Exception as exc:
        _print({"state": "failed", "failure_type": type(exc).__name__})
        return 1
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
