"""Shared preconditions for every measuring tool.

Kept in one place because a precondition that drifts between tools is worse than
no precondition: two runs would then be gated differently while both claim to
follow the same rules.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from friday_evidence.budget import BudgetError, BudgetGuard  # noqa: E402
from friday_evidence.run import run_persisted  # noqa: E402


class PowerError(SystemExit):
    """Raised when the machine is not in a state that allows measuring."""


def read_power_source() -> str:
    """Return 'ac_power', 'battery_power' or 'unknown' without ever raising."""

    try:
        completed = subprocess.run(
            ["/usr/bin/pmset", "-g", "ps"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    text = completed.stdout.decode("utf-8", errors="replace")
    if "AC Power" in text:
        return "ac_power"
    if "Battery Power" in text:
        return "battery_power"
    return "unknown"


def require_ac_power() -> str:
    """Refuse to measure unless the machine is on mains power.

    This is a measurement requirement, not a courtesy: on battery macOS caps the
    GPU power budget, so a run is neither comparable to a mains run nor gentle on
    the hardware.  Failing closed is the point -- a silently degraded measurement
    is worse than no measurement.
    """

    source = read_power_source()
    if source != "ac_power":
        raise PowerError(
            f"refused: mains power is a measurement requirement (found: {source})"
        )
    return source


def release_gate(args, self_check) -> int | None:
    """Return an exit code when the tool must not proceed, else None.

    Centralised on purpose: this exact five-line pattern was copied into each
    tool, and the one tool that lacked it (`run_h0_aa`) really did start a
    six-process GPU run from a stray invocation.  One implementation cannot
    diverge; five copies already had.
    """

    if getattr(args, "self_check", False):
        return self_check()
    if not getattr(args, "execute", False):
        print(json.dumps({"state": "not_released", "hint": "pass --execute"}))
        return 78
    return None


__all__ = [
    "BudgetError",
    "BudgetGuard",
    "PowerError",
    "read_power_source",
    "release_gate",
    "require_ac_power",
    "run_persisted",
]
