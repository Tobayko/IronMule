"""Closed command-line boundary for offline H0 runs and the local dashboard."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from typing import Any, Sequence

from .constants import MLX_MODES


EXIT_USAGE = 64
EXIT_PROVENANCE = 65
EXIT_STORAGE = 66
EXIT_PARENT = 70
EXIT_DASHBOARD = 75
EXIT_MLX_LOCKED = 78

_OFFLINE_CHOICES = (
    "analysis_slow",
    "analysis_known_win",
    "analysis_wrong_fixture",
    "analysis_missing_data",
    "control_timeout",
    "control_exit_70",
)
_PROCESS_SETS = ("characterization", "confirmation")


class _UsageError(Exception):
    """Private parser signal; user input is never reflected in output."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise _UsageError


def _port(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer in 0..65535") from exc
    if not 0 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("port must be an integer in 0..65535")
    return parsed


def _process_index(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("process-index must be an integer in 0..2") from exc
    if not 0 <= parsed <= 2:
        raise argparse.ArgumentTypeError("process-index must be an integer in 0..2")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="python -m friday_h0.cli")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_ArgumentParser)
    commands.add_parser("db-init", allow_abbrev=False)
    offline = commands.add_parser("offline", allow_abbrev=False)
    offline.add_argument("--mode", choices=_OFFLINE_CHOICES, required=True)
    dashboard = commands.add_parser("dashboard", allow_abbrev=False)
    dashboard.add_argument("--port", type=_port, default=0)
    mlx = commands.add_parser("mlx-run", allow_abbrev=False)
    mlx.add_argument("--mode", choices=tuple(sorted(MLX_MODES)), required=True)
    mlx.add_argument("--process-set", choices=_PROCESS_SETS, required=True)
    mlx.add_argument("--process-index", type=_process_index, required=True)
    mlx.add_argument("--execute", action="store_true")
    return parser


def _print_fields(fields: dict[str, Any]) -> None:
    # Values are intentionally restricted to bounded scalar protocol fields.
    print(" ".join(f"{key}={fields[key]}" for key in fields))


def _run_offline(mode: str) -> int:
    from .runner import result_exit_code, run_offline

    outcome = run_offline(mode)
    _print_fields({
        "state": outcome.persistence.state,
        "run_id": outcome.manifest.run_id,
        "status": outcome.result["status"],
        "classification": outcome.result["classification"],
        "action": outcome.result["action"],
        "bundle_sha256": outcome.persistence.bundle_sha256,
    })
    return result_exit_code(outcome.result)


def _run_dashboard(port: int) -> int:
    from .dashboard import serve
    from .runner import database_path

    server = serve(database_path(), port)
    stop_lock = threading.Lock()
    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        with stop_lock:
            if stopping:
                return
            stopping = True
        # shutdown() must not be called by the serve_forever thread itself.
        threading.Thread(target=server.shutdown, name="friday-h0-dashboard-shutdown", daemon=True).start()

    previous = {value: signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)}
    try:
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        _print_fields({"state": "serving", "port": int(server.server_address[1])})
        server.serve_forever(poll_interval=0.1)
        return 0
    finally:
        for value, handler in previous.items():
            signal.signal(value, handler)
        server.server_close()


def _run_mlx(mode: str, process_set: str, process_index: int) -> int:
    from .runner import result_exit_code, run_mlx

    outcome = run_mlx(mode, process_set, process_index)
    _print_fields({
        "state": outcome.persistence.state,
        "run_id": outcome.manifest.run_id,
        "status": outcome.result["status"],
        "classification": outcome.result["classification"],
        "action": outcome.result["action"],
        "bundle_sha256": outcome.persistence.bundle_sha256,
    })
    return result_exit_code(outcome.result)


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
    # This return intentionally precedes runner/dashboard imports and all
    # manifest/worker activity.  MLX remains a separately approved feature.
    if args.command == "mlx-run" and not args.execute:
        _print_fields({"state": "not_released", "command": "mlx-run"})
        return EXIT_MLX_LOCKED
    if args.command == "mlx-run":
        try:
            return _run_mlx(args.mode, args.process_set, args.process_index)
        except RuntimeError as exc:
            from .storage import StorageError

            if isinstance(exc, StorageError):
                return EXIT_STORAGE
            return EXIT_PARENT
        except (ImportError, OSError):
            return EXIT_PARENT
        except ValueError:
            return EXIT_PROVENANCE
    try:
        if args.command == "db-init":
            from .runner import initialize_database

            initialize_database()
            _print_fields({"state": "initialized", "identity": "friday_h0.sqlite.v1"})
            return 0
        if args.command == "offline":
            return _run_offline(args.mode)
        if args.command == "dashboard":
            return _run_dashboard(args.port)
    except RuntimeError as exc:
        from .storage import StorageError

        if isinstance(exc, StorageError):
            return EXIT_DASHBOARD if args.command == "dashboard" else EXIT_STORAGE
        if args.command == "dashboard":
            return EXIT_DASHBOARD
        return EXIT_PARENT
    except (ImportError, OSError):
        # Do not expose paths, tracebacks, or environment details on stdout.
        if args.command == "dashboard":
            return EXIT_DASHBOARD
        return EXIT_PARENT
    except ValueError:
        return EXIT_PROVENANCE if args.command == "offline" else EXIT_USAGE
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
