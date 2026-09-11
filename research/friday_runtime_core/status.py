"""One terminal status for the whole project. No server, no browser, no colour needed.

Twelve `dashboard.py` files carry 4431 lines between them, each a loopback
`ThreadingHTTPServer` with its own HTML renderer. The data layer underneath was
always clean — `DashboardService.snapshot()` returns plain dicts and only
`_html()` makes a web page out of them — so what gets replaced here is the
presentation, not the evaluation.

**Why line-oriented output rather than curses.** Only `curses` is available
(`rich`, `textual`, `blessed` are not installed and installing is blocked), and
it is still the wrong tool: a full-screen TUI with an alternate screen and
cursor addressing is markedly worse for a screen reader than plain lines in the
normal scrollback. Line-oriented is simultaneously the leaner and the more
accessible choice — the two goals do not conflict here.

**Accessibility is a build rule, not a coat of paint.** Colour never carries
meaning on its own: every state has a word (`[ok]`, `[FAIL]`, `[--]`, `[!]`).
`NO_COLOR` is honoured, a non-TTY stdout drops every escape, there is no
animation, no spinner and no cursor addressing, the output stays in the
scrollback and pipes cleanly, columns keep a fixed order under real headings,
and everything is plain ASCII. Below 60 columns the table stacks into labelled
lines instead of truncating.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: Every state is a word first. Colour, where it is used at all, only tints one.
MARKS = {
    "ok": "[ok]  ",
    "fail": "[FAIL]",
    "off": "[--]  ",
    "warn": "[!]   ",
}
_COLOURS = {"ok": "\033[32m", "fail": "\033[31m", "warn": "\033[33m", "off": ""}
_RESET = "\033[0m"

MIN_WIDTH = 60
DEFAULT_WIDTH = 92


def use_colour(stream=None) -> bool:
    """Colour only when a terminal asked for it and nothing forbade it."""

    stream = stream if stream is not None else sys.stdout
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def terminal_width() -> int:
    raw = os.environ.get("COLUMNS")
    if raw and raw.strip().isdigit():
        return max(MIN_WIDTH, int(raw.strip()))
    try:
        return max(MIN_WIDTH, os.get_terminal_size().columns)
    except OSError:
        return DEFAULT_WIDTH


def mark(state: str, *, colour: bool = False) -> str:
    word = MARKS.get(state, MARKS["off"])
    if not colour or not _COLOURS.get(state):
        return word
    return f"{_COLOURS[state]}{word}{_RESET}"


def _number(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "---"
    return f"{float(value):.{digits}f}"


def _interval(low: Any, high: Any) -> str:
    if low is None or high is None:
        return "---"
    return f"{_number(low)} - {_number(high)}"


@dataclass(frozen=True)
class Row:
    """One line of a section: a name, some columns, a state and a note."""

    name: str
    columns: tuple[str, ...] = ()
    state: str = "off"
    note: str = ""


@dataclass(frozen=True)
class Section:
    title: str
    headings: tuple[str, ...] = ()
    rows: tuple[Row, ...] = ()
    lines: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "headings": list(self.headings),
            "lines": list(self.lines),
            "rows": [
                {
                    "name": row.name,
                    "columns": list(row.columns),
                    "state": row.state,
                    "note": row.note,
                }
                for row in self.rows
            ],
        }


@dataclass(frozen=True)
class Status:
    generated_at: str
    sections: tuple[Section, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "friday.status.v1",
            "generated_at": self.generated_at,
            "sections": [section.as_dict() for section in self.sections],
        }


def render(status: Status, *, width: int | None = None, colour: bool = False) -> str:
    """Render the status as plain lines. Same data as ``--json``, by construction."""

    width = width or terminal_width()
    stacked = width < 80
    out: list[str] = []
    header = "Project Friday - Status"
    out.append(f"{header}{status.generated_at.rjust(max(1, width - len(header)))}".rstrip())
    for section in status.sections:
        out.append("")
        out.append(section.title.upper())
        for line in section.lines:
            out.append(f"  {line}")
        if not section.rows:
            continue
        name_width = max([len(row.name) for row in section.rows] + [10])
        column_widths = [
            max([len(row.columns[index]) if index < len(row.columns) else 0 for row in section.rows])
            for index in range(max((len(row.columns) for row in section.rows), default=0))
        ]
        if section.headings and not stacked:
            heading = "  " + section.headings[0].ljust(name_width + 2)
            for index, title in enumerate(section.headings[1:]):
                pad = column_widths[index] if index < len(column_widths) else len(title)
                heading += title.ljust(max(pad, len(title)) + 2)
            out.append(heading.rstrip())
        for row in section.rows:
            if stacked:
                # Narrow terminals stack into labelled lines rather than truncate:
                # a cut-off number is worse than a longer, complete one.
                out.append(f"  {row.name}")
                for index, value in enumerate(row.columns):
                    label = (
                        section.headings[index + 1]
                        if len(section.headings) > index + 1
                        else f"col{index + 1}"
                    )
                    out.append(f"      {label}: {value}")
                out.append(f"      Status: {mark(row.state, colour=colour).strip()} {row.note}".rstrip())
                continue
            line = "  " + row.name.ljust(name_width + 2)
            for index, value in enumerate(row.columns):
                pad = column_widths[index] if index < len(column_widths) else len(value)
                line += value.ljust(pad + 2)
            line += mark(row.state, colour=colour)
            if row.note:
                line += f" {row.note}"
            out.append(line.rstrip())
    return "\n".join(out) + "\n"


# -- section builders ---------------------------------------------------------
def device_section(hardware: Mapping[str, Any], profile) -> Section:
    memory = hardware.get("memory_bytes")
    try:
        gigabytes = f"{int(memory) / 1e9:.0f} GB"
    except (TypeError, ValueError):
        gigabytes = "? GB"
    chip = hardware.get("cpu_brand") or hardware.get("machine") or "unknown"
    lines = [f"{chip}  {gigabytes}  macOS {hardware.get('macos') or '?'}"]
    if profile is None:
        lines.append("Profil  keines - es gilt die Baseline (siehe Backlog D1)")
    else:
        lines.append(f"Profil  {profile.profile_id}   MDE {_number(profile.mde, 4)}")
        lines.append(f"Modell  {profile.model_id} @ {profile.model_revision[:8]}")
    return Section(title="Geraet", lines=tuple(lines))


def knob_section(profile, known: Sequence[Mapping[str, Any]] = ()) -> Section:
    """What each knob is doing, and why. Never a bare colour."""

    rows: list[Row] = []
    if profile is not None:
        for verdict in profile.knobs:
            state = {"verified": "ok", "failed": "off", "not_applicable": "off"}[verdict.verdict]
            note = {
                "verified": "aktiv",
                "failed": "aus",
                "not_applicable": "nicht anwendbar",
            }[verdict.verdict]
            rows.append(
                Row(
                    name=verdict.knob,
                    columns=(_number(verdict.ratio), _interval(verdict.ci_low, verdict.ci_high)),
                    state=state,
                    note=note if not verdict.reason else f"{note} - {verdict.reason[:40]}",
                )
            )
    else:
        for entry in known:
            rows.append(
                Row(
                    name=str(entry.get("knob", "?")),
                    columns=(
                        _number(entry.get("ratio")),
                        _interval(entry.get("ci_low"), entry.get("ci_high")),
                    ),
                    state="off",
                    note="nicht auf diesem Geraet kalibriert",
                )
            )
    return Section(
        title="Knoepfe",
        headings=("Knopf", "Ratio", "95%-KI", "Status"),
        rows=tuple(rows),
    )


def runtime_section(profile, circuit_reason: str | None) -> Section:
    active = profile is not None and bool(profile.verified_knobs())
    lines = [
        f"Pfad      {'optimiert' if active else 'baseline'}",
        f"Breaker   {'ausgeloest: ' + circuit_reason if circuit_reason else 'offen'}",
    ]
    if not active:
        lines.append("Grund     " + ("kein Geraeteprofil" if profile is None else "kein Knopf verifiziert"))
    return Section(title="Runtime", lines=tuple(lines))


#: Record kinds shortened for the eye. A truncated identifier that cuts through
#: a date is worse than a short label plus the full id in ``--json``.
_KIND_LABELS = {
    "runtime_validation": "Validierung",
    "runtime_validation_attempt": "Validierung (Start)",
    "policy_overhead": "Policy-Overhead",
    "runtime_failure": "Fehler",
    "device_profile": "Geraeteprofil",
}


def runs_section(runs: Sequence[Mapping[str, Any]]) -> Section:
    rows = []
    for run in runs:
        status = str(run.get("status", ""))
        if "fail" in status:
            state = "warn"
        elif status.endswith("passed"):
            state = "ok"
        else:
            state = "off"
        kind = str(run.get("kind", ""))
        rows.append(
            Row(
                name=str(run.get("source", "?")),
                columns=(_KIND_LABELS.get(kind, kind), _number(run.get("ratio"))),
                state=state,
                note=status,
            )
        )
    return Section(
        title="Letzte Laeufe",
        headings=("Quelle", "Art", "Ratio", "Status"),
        rows=tuple(rows),
    )


def open_section(entries: Sequence[str]) -> Section:
    return Section(title="Offen", lines=tuple(entries) or ("nichts offen",))


__all__ = [
    "DEFAULT_WIDTH",
    "MARKS",
    "MIN_WIDTH",
    "Row",
    "Section",
    "Status",
    "decisions_section",
    "device_section",
    "knob_section",
    "mark",
    "open_section",
    "render",
    "runs_section",
    "runtime_section",
    "signal_section",
    "terminal_width",
    "use_colour",
]


def _plain(value: Any) -> str:
    """Unwrap the ``{"value": ...}`` envelope some evidence rows carry."""

    if isinstance(value, Mapping) and "value" in value:
        value = value["value"]
    return "" if value is None else str(value)


def signal_section(board: Mapping[str, Any], limit: int = 6) -> Section:
    """The H0 signal board over its runs — the content that kept a server alive."""

    if not board.get("available"):
        return Section(title="H0-Signale", lines=(f"nicht lesbar ({board.get('reason', '?')})",))
    rows = []
    for run in board.get("recent", [])[:limit]:
        if not isinstance(run, Mapping):
            continue
        status = _plain(run.get("status"))
        if "fail" in status or status == "invalid":
            state = "warn"
        elif "pass" in status or status == "completed":
            state = "ok"
        else:
            state = "off"
        rows.append(
            Row(
                name=_plain(run.get("run_id") or run.get("id") or "?")[:30],
                columns=(_plain(run.get("kind"))[:20], _number(run.get("ratio"))),
                state=state,
                note=status,
            )
        )
    return Section(
        title="H0-Signale",
        headings=("Lauf", "Art", "Ratio", "Status"),
        lines=(f"{board.get('total', 0)} Laeufe im Signal-Board",),
        rows=tuple(rows),
    )


def decisions_section(decisions: Mapping[str, Any], limit: int = 6) -> Section:
    """The decision stream. An empty corpus reads as empty, never as a number."""

    if not decisions.get("available"):
        return Section(
            title="Entscheidungen", lines=(f"nicht lesbar ({decisions.get('reason', '?')})",)
        )
    lines = [
        f"{decisions.get('observed', 0)} beobachtet von {decisions.get('total', 0)}",
        f"OPE-Schaetzung  {decisions.get('estimate_status', '?')}",
    ]
    rows = []
    for entry in decisions.get("recent", [])[:limit]:
        if not isinstance(entry, Mapping):
            continue
        censoring = _plain(entry.get("censoring"))
        rows.append(
            Row(
                name=_plain(entry.get("policy_id") or "?")[:24],
                columns=(
                    _plain(entry.get("rule"))[:18],
                    _plain(entry.get("chosen"))[:22],
                    _number(entry.get("propensity"), 3),
                ),
                state="ok" if censoring == "observed" else "off",
                note=censoring,
            )
        )
    if not rows:
        lines.append("keine Entscheidung protokolliert - R2s Kampagne bleibt der Korpusweg")
    return Section(
        title="Entscheidungen",
        headings=("Policy", "Regel", "Aktion", "Propensity", "Status"),
        lines=tuple(lines),
        rows=tuple(rows),
    )
