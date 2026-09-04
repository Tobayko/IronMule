"""Calibration entry point. Nothing here touches the GPU without ``--execute``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from friday_runtime_core.history import HistoryError, RuntimeHistory
from friday_runtime_core.provenance import ProvenanceSpec, collect_provenance, machine_sha256

from . import plan
from .profile import HISTORY, profile_for
from .runner import MODELS, CalibrationError, build_runner, calibrate


def _resolve_model(value: str | None) -> tuple[str, int]:
    """Model id plus the prompt length to assert, ``0`` meaning "record it".

    The 897-token count is a property of the sealed 4B tokenisation, not of the
    prompt, so asserting it for another model would fail on a correct run.
    """

    model_id = MODELS.get(value or "", value) or plan.MODEL_ID
    return model_id, plan.PROMPT_TOKENS if model_id == plan.MODEL_ID else 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / ".friday-data" / "device-profile.sqlite3"

PROVENANCE = ProvenanceSpec(
    runtime_id=HISTORY.runtime_id,
    code_directories=("friday_calibrate", "friday_runtime_core"),
    spec_files=("docs/DEVICE_PROFILE_SPEC.md", "requirements-apple-silicon.txt"),
)


def _print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


def _self_check() -> int:
    _print({"state": "self_check", **plan.as_dict()})
    return 0


MODEL_HELP = f"Model id, or a shorthand from {sorted(MODELS)} (default: the sealed 4B)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    described = commands.add_parser("plan", allow_abbrev=False)
    described.add_argument("--pairs", type=int, default=plan.DEFAULT_PAIRS)
    described.add_argument("--model", default=None, help=MODEL_HELP)

    run = commands.add_parser("run", allow_abbrev=False)
    run.add_argument("--pairs", type=int, default=plan.DEFAULT_PAIRS)
    run.add_argument("--database", default=str(DEFAULT_DATABASE))
    run.add_argument("--model", default=None, help=MODEL_HELP)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--self-check", action="store_true", dest="self_check")

    show = commands.add_parser("show", allow_abbrev=False)
    show.add_argument("--database", default=str(DEFAULT_DATABASE))
    show.add_argument("--model", default=None, help="Model ID; default: any model calibrated here")

    args = parser.parse_args(argv)

    if args.command == "plan":
        model_id, prompt_tokens = _resolve_model(args.model)
        _print(plan.as_dict(args.pairs, model_id=model_id, prompt_tokens=prompt_tokens))
        return 0

    if args.command == "show":
        try:
            with RuntimeHistory.open(HISTORY, args.database, read_only=True) as history:
                with history.read_transaction():
                    rows = history.verified_records()
        except (HistoryError, OSError) as exc:
            _print({"state": "no_profile", "reason": str(exc)})
            return 1
        machine = machine_sha256()
        profile = profile_for(rows, machine_sha256=machine, model_id=args.model)
        if profile is None:
            _print(
                {
                    "state": "no_profile",
                    "records": len(rows),
                    "machine_sha256": machine,
                    "model_id": args.model,
                }
            )
            return 1
        _print(
            {
                "state": "profile",
                "machine_sha256": machine,
                "profile_id": profile.profile_id,
                "model_id": profile.model_id,
                "model_revision": profile.model_revision,
                "mde": profile.mde,
                "verified": list(profile.verified_knobs()),
                "unverified": list(profile.unverified()),
                "knobs": [verdict.as_dict() for verdict in profile.knobs],
                "width_curve": {str(k): v for k, v in profile.width_curve.items()},
            }
        )
        return 0

    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from _bench import release_gate  # noqa: E402

    gate = release_gate(args, _self_check)
    if gate is not None:
        return gate
    if not 1 <= args.pairs <= 32:
        raise SystemExit("pairs must be between 1 and 32")

    model_id, prompt_tokens = _resolve_model(args.model)
    provenance = collect_provenance(PROVENANCE)
    try:
        runner, identity, guard = build_runner(
            args.pairs, model_id=model_id, prompt_tokens=prompt_tokens
        )
        profile = calibrate(
            runner,
            identity,
            hardware_sha256=provenance["hardware_sha256"],
            environment_sha256=provenance["environment_sha256"],
            pairs=args.pairs,
        )
    except CalibrationError as exc:
        _print({"state": "calibration_failed", "reason": str(exc)})
        return 1

    report = profile.as_report(f"calibration-{profile.profile_id}")
    with RuntimeHistory.open(HISTORY, args.database, initialize=True) as history:
        outcome = history.persist(report, provenance)
    _print(
        {
            "state": "calibrated",
            "record_id": outcome.record_id,
            "model_id": profile.model_id,
            "prompt_tokens": identity.get("prompt_tokens"),
            "verified": list(profile.verified_knobs()),
            "unverified": list(profile.unverified()),
            "mde": profile.mde,
            "budget": guard.summary(),
            "formal_claim": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
