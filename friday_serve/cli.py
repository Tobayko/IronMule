"""Serve entry point. Reads the device profile, refuses to invent one."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from friday_calibrate.profile import HISTORY, newest_profile
from friday_runtime_core.history import HistoryError, RuntimeHistory

from .server import Server

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / ".friday-data" / "device-profile.sqlite3"
DEFAULT_MODEL = "mlx-community/gemma-3-4b-it-4bit"


def _print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


def load_profile(database: str | Path):
    """The newest profile in a verified chain, or ``None`` — never a guess."""

    try:
        with RuntimeHistory.open(HISTORY, database, read_only=True) as history:
            with history.read_transaction():
                rows = history.verified_records()
    except (HistoryError, OSError):
        return None
    return newest_profile(rows)


def _latch(database: str | Path):
    """Persist the breaker into the same hash chain the profile lives in."""

    from friday_calibrate.profile import FAILURE_KIND
    from friday_runtime_core.breaker import PersistentLatch
    from friday_runtime_core.provenance import ProvenanceSpec, collect_provenance

    spec = ProvenanceSpec(
        runtime_id=HISTORY.runtime_id,
        code_directories=("friday_serve", "friday_calibrate", "friday_runtime_core"),
        spec_files=("requirements-apple-silicon.txt",),
    )

    def load():
        with RuntimeHistory.open(HISTORY, database, read_only=True) as history:
            with history.read_transaction():
                rows = history.verified_records()
        failures = [row for row in rows if row["record_kind"] == FAILURE_KIND]
        return failures[-1]["report"].get("reason") if failures else None

    def append(reason):
        with RuntimeHistory.open(HISTORY, database) as history:
            history.persist(
                {
                    "schema_version": HISTORY.schema_version,
                    "runtime_id": HISTORY.runtime_id,
                    "kind": FAILURE_KIND,
                    "run_id": f"serve-failure-{reason}",
                    "status": "measurement_failed",
                    "reason": reason,
                    "formal_claim": False,
                },
                collect_provenance(spec, require_clean=False),
            )

    return PersistentLatch(load=load, append=append)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--database", default=str(DEFAULT_DATABASE))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status", allow_abbrev=False)

    generate = commands.add_parser("generate", allow_abbrev=False)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-tokens", type=int, default=32)
    generate.add_argument("--execute", action="store_true")

    args = parser.parse_args(argv)
    profile = load_profile(args.database)

    if args.command == "status":
        from .dispatch import explain

        described = explain(profile)
        described["database"] = args.database
        described["serves"] = "baseline" if profile is None else "device_profile_dispatch"
        _print(described)
        return 0

    if not args.execute:
        _print(
            {
                "state": "not_released",
                "hint": "pass --execute; generation loads the local model",
                "profile": None if profile is None else profile.profile_id,
                "would_use": [] if profile is None else list(profile.verified_knobs()),
            }
        )
        return 78

    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from .ironmule_backend import IronMuleBackend
    from .rl_controller import AdaptiveRLController

    rl_path = PROJECT_ROOT / ".friday-data" / "rl-controller.json"
    rl_ctrl = AdaptiveRLController.load(rl_path) if rl_path.exists() else None

    backend = IronMuleBackend.load(args.model)
    server = Server(backend, profile, latch=_latch(args.database), rl_controller=rl_ctrl)
    result = server.generate(args.prompt, args.max_tokens)
    _print(
        {
            "state": "generated",
            "plan": result.plan,
            "reason": result.reason,
            "knobs": dict(result.knobs),
            "tokens": len(result.tokens),
            "token_sha256": result.token_sha256,
            "prefill_ms": result.prefill_ns / 1e6,
            "decode_ms": result.decode_ns / 1e6,
            "text": result.text,
            "formal_claim": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
