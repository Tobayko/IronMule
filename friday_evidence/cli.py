"""Closed CLI for evidence verification, import, history, and loopback UI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .dashboard import DEFAULT_PORT, DashboardError, DashboardService, serve
from .legacy import import_legacy_summaries
from .registry import DEFAULT_DATABASE_PATH, LEGACY_SOURCE_PATH
from .storage import EvidenceStorage, StorageError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="friday evidence", allow_abbrev=False)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot", allow_abbrev=False)
    snapshot.add_argument("--limit", type=int, default=50)
    detail = commands.add_parser("detail", allow_abbrev=False)
    detail.add_argument("--id", required=True)
    commands.add_parser("verify", allow_abbrev=False)
    initialize = commands.add_parser("init", allow_abbrev=False)
    initialize.add_argument("--apply", action="store_true")
    legacy = commands.add_parser("import-legacy", allow_abbrev=False)
    legacy.add_argument("--source", type=Path, default=LEGACY_SOURCE_PATH)
    legacy.add_argument("--apply", action="store_true")
    server = commands.add_parser("serve", allow_abbrev=False)
    server.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "init":
            if not args.apply:
                _print({"state": "not_applied", "hint": "pass --apply"})
                return 78
            with EvidenceStorage.open(args.database, initialize=True):
                pass
            _print({"state": "initialized", "database": str(args.database)})
            return 0
        if args.command == "import-legacy":
            if not args.apply:
                _print({"state": "not_applied", "hint": "pass --apply"})
                return 78
            with EvidenceStorage.open(args.database, initialize=True) as storage:
                outcomes = import_legacy_summaries(storage, args.source)
            _print(
                {
                    "state": "imported",
                    "inserted": sum(item.state == "inserted" for item in outcomes),
                    "already_present": sum(item.state == "already_present" for item in outcomes),
                    "total": len(outcomes),
                }
            )
            return 0
        if args.command == "verify":
            with EvidenceStorage.open(args.database, read_only=True) as storage:
                rows = storage.verified_rows()
            _print({"state": "verified", "records": len(rows), "read_only": True})
            return 0
        if args.command == "snapshot":
            _print(DashboardService(args.database).snapshot(args.limit))
            return 0
        if args.command == "detail":
            value = DashboardService(args.database).detail(args.id)
            _print(value or {"state": "not_found"})
            return 0 if value else 1
        if args.command == "serve":
            print(f"Friday evidence UI: http://127.0.0.1:{args.port}")
            serve(args.database, port=args.port)
            return 0
    except StorageError:
        _print({"state": "storage_error"})
        return 66
    except (DashboardError, UnicodeError, ValueError):
        _print({"state": "invalid_request"})
        return 64
    return 64


__all__ = ["main"]
