"""Where the terminal status reads from. Read-only, no GPU, no network, fast.

Nothing here opens a model or a socket. Every source is either a file on disk or
one of the existing ``DashboardService`` objects, which already return plain
dicts — those classes are *used*, not modified: they live in packages whose
``code_sha256`` is part of a sealed identity.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKLOG = PROJECT_ROOT / "BACKLOG.md"

#: The knobs the project has measured, with the study that measured them. Shown
#: when no device profile exists yet, so the screen is never empty — but marked
#: as "not calibrated here", because that is what they are.
KNOWN_KNOBS = (
    {"knob": "head_skip", "ratio": 0.846385, "ci_low": None, "ci_high": None},
    {"knob": "fixed_compiled", "ratio": 0.9296, "ci_low": None, "ci_high": None},
    {"knob": "bundled_readback", "ratio": 0.9581, "ci_low": 0.953471, "ci_high": 0.959885},
    {"knob": "prefill_step_size", "ratio": 0.9288, "ci_low": None, "ci_high": None},
)

#: The runtime histories that exist. Each is (label, module path, database attr).
RUNTIME_HISTORIES = (
    ("n10", "friday_runtime_n10"),
    ("head-skip", "friday_head_skip_runtime"),
    ("h1", "friday_runtime"),
)


def open_backlog_entries(limit: int = 8) -> list[str]:
    """The open backlog headings, minus the ones already marked answered."""

    try:
        text = BACKLOG.read_text()
    except OSError:
        return []
    entries = []
    for line in text.splitlines():
        match = re.match(r"^## (\S+) — (.+)$", line)
        if not match:
            continue
        identifier, title = match.group(1), match.group(2)
        # An entry whose title *begins* with a closing word is done. "F1 -
        # warmer Arm beantwortet, kalter Arm offen" was open while the second
        # clause was; the word has to lead, not merely appear.
        if title.split()[0] in {"beantwortet", "geschlossen", "abgeschlossen", "erledigt"}:
            continue
        entries.append(f"{identifier}  " + re.sub(r"\s*\(neu [0-9-]+\)$", "", title))
    return entries[:limit]


def recent_runs(limit: int = 6) -> list[dict[str, Any]]:
    """Newest records across the existing runtime histories, newest first."""

    import importlib

    collected: list[dict[str, Any]] = []
    for label, module_name in RUNTIME_HISTORIES:
        try:
            module = importlib.import_module(f"{module_name}.dashboard")
            constants = importlib.import_module(f"{module_name}.constants")
        except Exception:
            continue
        path = getattr(constants, "DEFAULT_RUNTIME_DATABASE_PATH", None)
        if path is None:
            continue
        try:
            snapshot = module.DashboardService(path).snapshot()
        except Exception:
            continue
        for row in snapshot.get("recent", []):
            metrics = row.get("metrics") or {}
            collected.append(
                {
                    "source": label,
                    "run_id": row.get("run_id"),
                    "kind": row.get("kind"),
                    "status": row.get("status"),
                    "ratio": metrics.get("ratio"),
                    "created_at_unix_ns": row.get("created_at_unix_ns") or 0,
                }
            )
    collected.sort(key=lambda row: row["created_at_unix_ns"], reverse=True)
    return collected[:limit]


def device_profile(database: str | Path | None = None):
    """The newest verified device profile, or ``None``. Never a guess."""

    try:
        from friday_calibrate.profile import HISTORY, newest_profile
        from friday_runtime_core.history import RuntimeHistory
    except Exception:
        return None
    path = database or (PROJECT_ROOT / ".friday-data" / "device-profile.sqlite3")
    try:
        with RuntimeHistory.open(HISTORY, path, read_only=True) as history:
            with history.read_transaction():
                rows = history.verified_records()
    except Exception:
        return None
    try:
        return newest_profile(rows)
    except Exception:
        return None


def circuit_reason(database: str | Path | None = None) -> str | None:
    try:
        from friday_calibrate.profile import FAILURE_KIND, HISTORY
        from friday_runtime_core.history import RuntimeHistory
    except Exception:
        return None
    path = database or (PROJECT_ROOT / ".friday-data" / "device-profile.sqlite3")
    try:
        with RuntimeHistory.open(HISTORY, path, read_only=True) as history:
            with history.read_transaction():
                rows = history.verified_records()
    except Exception:
        return None
    failures = [row for row in rows if row["record_kind"] == FAILURE_KIND]
    return failures[-1]["report"].get("reason") if failures else None


__all__ = [
    "KNOWN_KNOBS",
    "PROJECT_ROOT",
    "circuit_reason",
    "device_profile",
    "open_backlog_entries",
    "recent_runs",
]


#: The two dashboards that carry content the terminal status has to absorb
#: before their HTTP servers can be deleted (BACKLOG U1): the H0 signal board
#: over 28 runs, and the optimizer's decision stream.
H0_DATABASE = PROJECT_ROOT / ".friday-data" / "h0.sqlite3"
OPTIMIZER_DATABASE = PROJECT_ROOT / ".friday-data" / "optimizer-v2.sqlite3"


def h0_board(limit: int = 6) -> dict[str, Any]:
    """The H0 signal board, read-only. Same numbers, no server."""

    try:
        from friday_h0.dashboard import DashboardService

        snapshot = DashboardService(H0_DATABASE).snapshot()
    except Exception as exc:
        return {"available": False, "reason": type(exc).__name__}
    runs = snapshot.get("runs") or snapshot.get("recent") or []
    return {
        "available": True,
        "total": snapshot.get("total", len(runs)),
        "by_status": snapshot.get("by_status") or {},
        "recent": list(runs)[:limit],
    }


def optimizer_decisions(limit: int = 6) -> dict[str, Any]:
    """The decision stream and its off-policy estimate, read-only."""

    try:
        from friday_optimizer.dashboard import DashboardService

        payload = DashboardService(OPTIMIZER_DATABASE).decisions(limit=100)
    except Exception as exc:
        return {"available": False, "reason": type(exc).__name__}
    estimates = payload.get("estimates") or {}
    snips = estimates.get("snips") or {}
    return {
        "available": True,
        "total": payload.get("total", 0),
        "observed": payload.get("observed", 0),
        # An empty corpus must read as "insufficient_data", never as a number.
        "estimate_status": snips.get("status", "keine Schaetzung"),
        "recent": list(payload.get("decisions") or [])[:limit],
    }
