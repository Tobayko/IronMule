"""Closed command-line boundary for the controlled H0.1 execution path.

Every command that would touch the GPU sits behind the same explicit
``--execute`` release gate H0 uses.  Without it the process exits before any
runner, backend, MLX or NumPy import happens at all.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any, Sequence

from .constants import SESSION_ORDER

EXIT_USAGE = 64
EXIT_PROVENANCE = 65
EXIT_STORAGE = 66
EXIT_PARENT = 70
EXIT_MLX_LOCKED = 78


class _UsageError(Exception):
    """Private parser signal; user input is never reflected in output."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise _UsageError


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="friday_h01", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", allow_abbrev=False)
    preflight.add_argument("--execute", action="store_true")
    preflight.add_argument("--parent-run-id", default=None)

    session = commands.add_parser("session", allow_abbrev=False)
    session.add_argument("--id", required=True, choices=SESSION_ORDER)
    session.add_argument("--execute", action="store_true")
    session.add_argument("--parent-run-id", default=None)

    study = commands.add_parser("study", allow_abbrev=False)
    study.add_argument("--execute", action="store_true")

    run_all = commands.add_parser("run-all", allow_abbrev=False)
    run_all.add_argument("--execute", action="store_true")
    run_all.add_argument("--parent-run-id", default=None)
    return parser


def _print_fields(fields: dict[str, Any]) -> None:
    print(json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _run_preflight(parent_run_id: str | None) -> int:
    from .runner import MlxBackend, preflight

    report = preflight(parent_run_id=parent_run_id, backend_factory=MlxBackend)
    _print_fields({"state": "preflight_ok", **report})
    return 0


def _run_session(session_id: str, parent_run_id: str | None) -> int:
    from .runner import run_session

    outcome = run_session(session_id, parent_run_id=parent_run_id)
    _print_fields(
        {
            "state": outcome.persistence_state,
            "session_id": outcome.session_id,
            "run_id": outcome.run_id,
            "status": outcome.status,
            "failed_gates": outcome.failed_gates,
            "bundle_sha256": outcome.bundle_sha256,
            "wall_seconds": outcome.wall_seconds,
        }
    )
    return 0 if outcome.status == "h01_session_complete" else EXIT_PARENT


def _run_study() -> int:
    from .runner import run_study

    outcome = run_study()
    _print_fields(
        {
            "state": outcome.persistence_state,
            "study_id": outcome.study_id,
            "status": outcome.status,
            "failed_gate_count": outcome.failed_gate_count,
            "bundle_sha256": outcome.bundle_sha256,
        }
    )
    return 0


def _run_all(parent_run_id: str | None) -> int:
    """Run the six registered sessions as six separate processes, then the study.

    Separate processes are a contract requirement, not an implementation
    preference: the preregistered design specifies exactly six independent
    session processes in the order C0,V0,C1,V1,C2,V2.
    """

    for session_id in SESSION_ORDER:
        argv = [
            sys.executable,
            "-m",
            "friday_h01.cli",
            "session",
            "--id",
            session_id,
            "--execute",
        ]
        if parent_run_id is not None:
            argv += ["--parent-run-id", parent_run_id]
        completed = subprocess.run(argv, check=False)
        if completed.returncode != 0:
            _print_fields(
                {
                    "state": "session_failed",
                    "session_id": session_id,
                    "code": completed.returncode,
                }
            )
            return completed.returncode
    return _run_study()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(list(argv) if argv is not None else None)
    except _UsageError:
        _print_fields({"state": "usage_error", "code": EXIT_USAGE})
        return EXIT_USAGE
    except SystemExit as exc:
        if exc.code == 0:
            raise
        _print_fields({"state": "usage_error", "code": EXIT_USAGE})
        return EXIT_USAGE

    # This return intentionally precedes every runner, storage, NumPy and MLX
    # import.  Live H0.1 execution remains a separately approved feature.
    if not args.execute:
        _print_fields({"state": "not_released", "command": args.command})
        return EXIT_MLX_LOCKED

    try:
        if args.command == "preflight":
            return _run_preflight(args.parent_run_id)
        if args.command == "session":
            return _run_session(args.id, args.parent_run_id)
        if args.command == "study":
            return _run_study()
        if args.command == "run-all":
            return _run_all(args.parent_run_id)
    except RuntimeError as exc:
        from .storage import StorageError

        _print_fields({"state": "failed", "error": type(exc).__name__})
        return EXIT_STORAGE if isinstance(exc, StorageError) else EXIT_PARENT
    except (ImportError, OSError):
        # Do not expose paths, tracebacks, or environment details on stdout.
        _print_fields({"state": "failed", "error": "environment"})
        return EXIT_PARENT
    except ValueError:
        _print_fields({"state": "failed", "error": "provenance"})
        return EXIT_PROVENANCE
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
