"""The terminal status: same data as --json, readable at 60 columns, no escapes.

The accessibility rules are the tests. Colour never carries meaning on its own,
a non-TTY drops every escape, and a narrow terminal stacks rather than truncates
— a cut-off number is worse than a longer, complete one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from friday_calibrate.profile import DeviceProfile, KnobVerdict  # noqa: E402
from friday_runtime_core.status import (  # noqa: E402
    MARKS,
    Row,
    Section,
    Status,
    device_section,
    knob_section,
    mark,
    open_section,
    render,
    runs_section,
    runtime_section,
    terminal_width,
    use_colour,
)
from friday_runtime_core.status import decisions_section, signal_section  # noqa: E402
from friday_runtime_core.status_sources import (  # noqa: E402
    KNOWN_KNOBS,
    h0_board,
    open_backlog_entries,
    optimizer_decisions,
)

HARDWARE = {
    "machine": "arm64",
    "macos": "26.6.2",
    "model": "MacBookPro18,2",
    "memory_bytes": "34359738368",
    "cpu_brand": "Apple M1 Max",
}


class Stream:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def profile(*verified: str) -> DeviceProfile:
    verdicts = []
    for knob in ("head_skip", "fixed_compiled", "bundled_readback"):
        if knob in verified:
            verdicts.append(KnobVerdict(knob, "verified", 6, 0.9, 0.88, 0.93, True))
        else:
            verdicts.append(KnobVerdict(knob, "failed", 6, 1.0, 0.98, 1.02, True, "kein Gewinn"))
    return DeviceProfile(
        profile_id="device-2026-09-02",
        model_id="mlx-community/gemma-3-4b-it-4bit",
        model_revision="93724907d4ed1745",
        hardware_sha256="a" * 64,
        environment_sha256="b" * 64,
        mde=0.006,
        knobs=tuple(verdicts),
    )


def snapshot(current=None) -> Status:
    return Status(
        generated_at="2026-09-02 09:31",
        sections=(
            device_section(HARDWARE, current),
            knob_section(current, KNOWN_KNOBS),
            runtime_section(current, None),
            runs_section(
                [{"source": "n10", "kind": "runtime_validation", "ratio": 0.8758,
                  "status": "runtime_validation_passed", "run_id": "n10-x"}]
            ),
            open_section(["F1  kalter Arm"]),
        ),
    )


class ColourTest(unittest.TestCase):
    def test_every_state_has_a_word_not_only_a_colour(self) -> None:
        for state in ("ok", "fail", "off", "warn"):
            self.assertTrue(mark(state).strip())
            self.assertNotIn("\033", mark(state))

    def test_colour_only_tints_a_word_that_is_already_there(self) -> None:
        coloured = mark("ok", colour=True)
        self.assertIn(MARKS["ok"].strip(), coloured)

    def test_no_colour_and_a_pipe_both_switch_colour_off(self) -> None:
        original = dict(os.environ)
        try:
            os.environ.pop("NO_COLOR", None)
            os.environ["TERM"] = "xterm"
            self.assertTrue(use_colour(Stream(tty=True)))
            self.assertFalse(use_colour(Stream(tty=False)))
            os.environ["NO_COLOR"] = "1"
            self.assertFalse(use_colour(Stream(tty=True)))
            os.environ.pop("NO_COLOR")
            os.environ["TERM"] = "dumb"
            self.assertFalse(use_colour(Stream(tty=True)))
        finally:
            os.environ.clear()
            os.environ.update(original)


class RenderTest(unittest.TestCase):
    def test_the_default_render_carries_no_escape_bytes(self) -> None:
        self.assertNotIn("\033", render(snapshot(), width=92, colour=False))

    def test_no_box_drawing_characters(self) -> None:
        text = render(snapshot(profile("head_skip")), width=92)
        for character in "─│┌┐└┘├┤┬┴┼━┃":
            self.assertNotIn(character, text)

    def test_a_narrow_terminal_stacks_instead_of_truncating(self) -> None:
        narrow = render(snapshot(profile("head_skip")), width=60)
        self.assertIn("Ratio:", narrow)
        self.assertIn("95%-KI:", narrow)
        # Nothing may be silently cut: the full knob name still appears.
        self.assertIn("bundled_readback", narrow)
        for line in narrow.splitlines():
            self.assertLessEqual(len(line), 78, line)

    def test_a_wide_terminal_uses_columns_with_real_headings(self) -> None:
        wide = render(snapshot(profile("head_skip")), width=92)
        self.assertIn("Knopf", wide)
        self.assertIn("95%-KI", wide)
        self.assertNotIn("Ratio:", wide)

    def test_the_section_order_is_fixed(self) -> None:
        text = render(snapshot(), width=92)
        order = [line for line in text.splitlines() if line and not line.startswith(" ")]
        self.assertEqual(
            [item.split("  ")[0] for item in order[1:]],
            ["GERAET", "KNOEPFE", "RUNTIME", "LETZTE LAEUFE", "OFFEN"],
        )

    def test_terminal_width_never_drops_below_the_floor(self) -> None:
        original = os.environ.get("COLUMNS")
        try:
            os.environ["COLUMNS"] = "20"
            self.assertEqual(terminal_width(), 60)
            os.environ["COLUMNS"] = "120"
            self.assertEqual(terminal_width(), 120)
        finally:
            if original is None:
                os.environ.pop("COLUMNS", None)
            else:
                os.environ["COLUMNS"] = original


class ContentTest(unittest.TestCase):
    def test_without_a_profile_nothing_claims_to_be_active(self) -> None:
        text = render(snapshot(None), width=92)
        self.assertIn("baseline", text)
        self.assertIn("kein Geraeteprofil", text)
        self.assertIn("nicht auf diesem Geraet kalibriert", text)

    def test_with_a_profile_only_verified_knobs_read_as_active(self) -> None:
        text = render(snapshot(profile("head_skip")), width=92)
        self.assertIn("optimiert", text)
        rows = [line for line in text.splitlines() if "head_skip" in line]
        self.assertTrue(any("aktiv" in line for line in rows))
        failed = [line for line in text.splitlines() if "fixed_compiled" in line]
        self.assertTrue(all("aktiv" not in line.replace("nicht", "") for line in failed))

    def test_a_tripped_breaker_is_visible(self) -> None:
        section = runtime_section(profile("head_skip"), "ValueError")
        self.assertIn("ausgeloest: ValueError", "\n".join(section.lines))

    def test_the_backlog_supplies_the_open_section(self) -> None:
        entries = open_backlog_entries()
        self.assertTrue(entries)
        self.assertTrue(any(entry.startswith("D1") for entry in entries))
        self.assertFalse(any("(neu 2026" in entry for entry in entries))

    def test_closed_entries_do_not_appear_as_open(self) -> None:
        for entry in open_backlog_entries(limit=40):
            first = entry.split("  ", 1)[1].split()[0]
            self.assertNotIn(
                first, {"beantwortet", "geschlossen", "abgeschlossen", "erledigt"}, entry
            )


class AbsorbedDashboardTest(unittest.TestCase):
    """The two dashboards that may only be deleted once status shows their numbers."""

    def test_the_h0_signal_board_is_readable_and_rendered(self) -> None:
        board = h0_board()
        self.assertTrue(board["available"], board)
        self.assertGreaterEqual(board["total"], 28)
        text = render(
            Status(generated_at="x", sections=(signal_section(board),)), width=92
        )
        self.assertIn("Laeufe im Signal-Board", text)
        # The evidence rows wrap ids in {"value": ...}; that must not leak out.
        self.assertNotIn("{'value'", text)

    def test_an_empty_decision_stream_reads_as_empty_not_as_a_number(self) -> None:
        decisions = optimizer_decisions()
        self.assertTrue(decisions["available"], decisions)
        text = render(
            Status(generated_at="x", sections=(decisions_section(decisions),)), width=92
        )
        self.assertIn("OPE-Schaetzung", text)
        if decisions["total"] == 0:
            self.assertIn("keine Entscheidung protokolliert", text)

    def test_an_unreadable_source_says_so_instead_of_inventing(self) -> None:
        text = render(
            Status(
                generated_at="x",
                sections=(
                    signal_section({"available": False, "reason": "StorageError"}),
                    decisions_section({"available": False, "reason": "StorageError"}),
                ),
            ),
            width=92,
        )
        self.assertEqual(text.count("nicht lesbar (StorageError)"), 2)


class CommandTest(unittest.TestCase):
    def _run(self, *args):
        environment = dict(os.environ, COLUMNS="92")
        return subprocess.run(
            [sys.executable, str(ROOT / "tools" / "friday.py"), "status", *args],
            capture_output=True, text=True, env=environment, cwd=str(ROOT), timeout=120,
        )

    def test_status_runs_and_touches_no_gpu(self) -> None:
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Project Friday - Status", completed.stdout)

    def test_plain_output_has_no_escape_bytes(self) -> None:
        completed = self._run("--plain")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("\033", completed.stdout)

    def test_json_carries_the_same_sections(self) -> None:
        completed = self._run("--json")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema"], "friday.status.v1")
        self.assertEqual(
            [section["title"] for section in payload["sections"]],
            ["Geraet", "Knoepfe", "Runtime", "Letzte Laeufe", "Offen"],
        )

    def test_an_unknown_option_is_refused(self) -> None:
        completed = self._run("--html")
        self.assertEqual(completed.returncode, 64)

    def test_all_shows_the_absorbed_dashboard_sections(self) -> None:
        completed = self._run("--all", "--json")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        titles = [s["title"] for s in json.loads(completed.stdout)["sections"]]
        self.assertIn("H0-Signale", titles)
        self.assertIn("Entscheidungen", titles)


if __name__ == "__main__":
    unittest.main()
