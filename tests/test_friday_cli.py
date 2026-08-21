"""Checks for the unified entry point.

These matter for anyone but the author: the CLI is the first thing a new user
touches, and a broken dispatch or a missing release gate would be discovered at
the worst possible moment -- while running a measurement.
"""

from __future__ import annotations

import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "friday_cli", PROJECT_ROOT / "tools" / "friday.py"
)
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


class ToolRegistryTest(unittest.TestCase):
    def test_every_registered_script_exists(self) -> None:
        for name, (script, _description) in cli.TOOLS.items():
            with self.subTest(tool=name):
                self.assertTrue((cli.TOOLS_DIR / script).is_file(), script)

    def test_every_tool_has_a_description(self) -> None:
        for name, (_script, description) in cli.TOOLS.items():
            with self.subTest(tool=name):
                self.assertTrue(description.strip())

    def test_every_measuring_tool_exposes_a_main(self) -> None:
        for name, (script, _d) in cli.TOOLS.items():
            with self.subTest(tool=name):
                module = cli._load(cli.TOOLS_DIR / script)
                self.assertTrue(callable(module.main))


class DispatchTest(unittest.TestCase):
    def test_unknown_tool_is_refused_with_a_usage_code(self) -> None:
        with redirect_stdout(io.StringIO()) as out:
            code = cli.main(["does-not-exist"])
        self.assertEqual(code, 64)
        self.assertEqual(json.loads(out.getvalue())["error"], "unknown tool")

    def test_list_succeeds(self) -> None:
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["list"]), 0)
        self.assertIn("loop", out.getvalue())

    def test_help_succeeds(self) -> None:
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main([]), 0)

    def test_parameterless_tool_rejects_arguments(self) -> None:
        # The guard takes no arguments; silently ignoring them would hide a typo.
        with redirect_stdout(io.StringIO()) as out:
            code = cli.main(["guard", "--nonsense"])
        self.assertEqual(code, 64)
        self.assertEqual(json.loads(out.getvalue())["error"], "tool takes no arguments")


class ReleaseGateTest(unittest.TestCase):
    """Nothing may measure without an explicit --execute.

    ``aa`` shipped without this gate and a stray invocation really did start a
    six-process GPU run; only the resume check kept it from recording anything.
    The membership test below is what makes that class of miss impossible: a new
    measuring tool has to be listed here or the suite fails.
    """

    # Every tool that can touch the GPU.  ``guard`` runs tests only.
    MEASURING_TOOLS = ("loop", "dispatch", "cooldown", "aa", "model-loop", "codegen", "roofline", "fusion")
    NON_MEASURING_TOOLS = ("guard", "evidence")

    def test_the_two_groups_cover_every_registered_tool(self) -> None:
        self.assertEqual(
            sorted(self.MEASURING_TOOLS + self.NON_MEASURING_TOOLS), sorted(cli.TOOLS)
        )

    def test_measuring_tools_are_locked_by_default(self) -> None:
        for tool in self.MEASURING_TOOLS:
            with self.subTest(tool=tool):
                with redirect_stdout(io.StringIO()) as out:
                    code = cli.main([tool])
                self.assertEqual(code, 78)
                self.assertEqual(json.loads(out.getvalue())["state"], "not_released")

    def test_self_check_runs_without_gpu_for_every_measuring_tool(self) -> None:
        for tool in self.MEASURING_TOOLS:
            with self.subTest(tool=tool):
                with redirect_stdout(io.StringIO()) as out:
                    code = cli.main([tool, "--self-check"])
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(out.getvalue())["self_check"], "pass")


class DocumentationTest(unittest.TestCase):
    def test_readme_links_resolve(self) -> None:
        import re

        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)h][^)]*)\)", readme):
            path = target.split("#")[0]
            if not path:
                continue
            with self.subTest(link=path):
                self.assertTrue((PROJECT_ROOT / path).exists(), path)

    def test_results_document_links_resolve(self) -> None:
        import re

        results = (PROJECT_ROOT / "docs" / "ERGEBNISSE.md").read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)h][^)]*)\)", results):
            path = target.split("#")[0]
            if not path:
                continue
            with self.subTest(link=path):
                self.assertTrue((PROJECT_ROOT / "docs" / path).exists(), path)

    def test_readme_documents_every_registered_tool(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        for name in cli.TOOLS:
            with self.subTest(tool=name):
                self.assertIn(f"`{name}`", readme)


class SharedPreconditionTest(unittest.TestCase):
    """The power gate lives in one place.

    A gate that drifts between tools is worse than none: two runs would then be
    gated differently while both claim to follow the same rules.
    """

    def setUp(self) -> None:
        self.shared = cli._shared()

    def test_power_source_is_one_of_three_known_states(self) -> None:
        self.assertIn(
            self.shared.read_power_source(), {"ac_power", "battery_power", "unknown"}
        )

    def test_reading_power_never_raises(self) -> None:
        # doctor must be able to report a problem, not crash on one.
        for _ in range(3):
            self.shared.read_power_source()

    def test_every_measuring_tool_uses_the_shared_gate(self) -> None:
        for tool in ("dispatch", "cooldown", "aa"):
            with self.subTest(tool=tool):
                module = cli._load(cli.TOOLS_DIR / cli.TOOLS[tool][0])
                gate = getattr(module, "require_ac_power", None) or getattr(
                    module, "_require_ac_power", None
                )
                self.assertIsNotNone(gate)
                # Identity cannot be compared here: every load creates a fresh
                # module object.  Where the function came from is what matters.
                self.assertEqual(gate.__module__, "_bench")

    def test_refusal_is_a_systemexit_not_a_silent_pass(self) -> None:
        self.assertTrue(issubclass(self.shared.PowerError, SystemExit))


class ResearchEvidenceContractTest(unittest.TestCase):
    """Every H1/H2 tool shares persistence and the same hardware-budget guard."""

    EVIDENCE_TOOLS = {
        "dispatch", "cooldown", "loop", "model-loop", "codegen", "roofline", "fusion"
    }

    def test_registry_matches_every_h1_h2_measuring_tool(self) -> None:
        from friday_evidence.registry import RAW_REPORT_FIELDS, REGISTERED_TOOLS

        self.assertEqual(set(REGISTERED_TOOLS), self.EVIDENCE_TOOLS)
        self.assertEqual(set(RAW_REPORT_FIELDS), self.EVIDENCE_TOOLS)

    def test_every_h1_h2_tool_uses_shared_budget_and_persistence(self) -> None:
        for tool in sorted(self.EVIDENCE_TOOLS):
            with self.subTest(tool=tool):
                module = cli._load(cli.TOOLS_DIR / cli.TOOLS[tool][0])
                self.assertEqual(module.BudgetGuard.__module__, "friday_evidence.budget")
                self.assertEqual(module.run_persisted.__module__, "friday_evidence.run")


if __name__ == "__main__":
    unittest.main()


class SequencerLogicTest(unittest.TestCase):
    """The A/A sequencer's success rule, which was wrong once already.

    ``friday_h0.cli`` returns 0 only for ``action=promoted``.  A single A/A
    process cannot be promoted -- that needs the aggregate -- so it exits 10 while
    being perfectly valid.  Judging by exit code alone aborted a healthy run after
    its first process.
    """

    def setUp(self) -> None:
        self.aa = cli._load(cli.TOOLS_DIR / "run_h0_aa.py")

    def test_exit_ten_with_a_completed_status_counts_as_success(self) -> None:
        self.assertTrue(
            self.aa._process_succeeded(
                10, "status=completed classification=measurement_complete"
            )
        )

    def test_exit_zero_also_counts(self) -> None:
        self.assertTrue(
            self.aa._process_succeeded(
                0, "status=completed classification=measurement_complete"
            )
        )

    def test_a_real_failure_code_is_never_success(self) -> None:
        for code in (64, 65, 66, 70, 78):
            with self.subTest(code=code):
                self.assertFalse(
                    self.aa._process_succeeded(
                        code, "status=completed classification=measurement_complete"
                    )
                )

    def test_invalid_status_is_not_success_despite_a_tolerated_code(self) -> None:
        # warmup_unstable really produced this: exit 10, but status=invalid.
        self.assertFalse(
            self.aa._process_succeeded(10, "status=invalid classification=invalid")
        )

    def test_empty_output_is_not_success(self) -> None:
        self.assertFalse(self.aa._process_succeeded(10, ""))

    def test_the_frozen_process_plan_is_exactly_three_plus_three(self) -> None:
        self.assertEqual(self.aa.PROCESS_SETS, ("characterization", "confirmation"))
        self.assertEqual(self.aa.PROCESS_INDICES, (0, 1, 2))


class DocumentationCoverageTest(unittest.TestCase):
    """Every tool and every finding must be written down somewhere.

    Added after a review found the roofline result, the disturbance timescale and
    the retracted fusion claim documented in the journal but missing from the
    files a reader actually starts with.
    """

    DOCS = ("PROJECT_STATUS.md", "docs/ERGEBNISSE.md", "README.md",
            "docs/ARBEITSJOURNAL.md")

    def setUp(self) -> None:
        self.texts = {
            name: (PROJECT_ROOT / name).read_text(encoding="utf-8") for name in self.DOCS
        }

    def test_every_tool_script_is_documented_somewhere(self) -> None:
        scripts = {script for script, _ in cli.TOOLS.values()}
        for script in sorted(scripts):
            with self.subTest(script=script):
                self.assertTrue(
                    any(script in text for text in self.texts.values()),
                    f"{script} is not mentioned in any document",
                )

    def test_headline_findings_reach_the_entry_documents(self) -> None:
        # These are the results a reader must not have to dig for.
        entry = self.texts["README.md"] + self.texts["docs/ERGEBNISSE.md"]
        for finding, needle in (
            ("disturbance timescale", "340 ms"),
            ("memory-bound roofline", "Bandbreite genutzt"),
            ("retracted fusion claim", "verworfen"),
            ("paired-vs-unpaired core result", "gepaart"),
        ):
            with self.subTest(finding=finding):
                self.assertIn(needle, entry)

    def test_no_document_still_sells_the_retracted_fusion_gain(self) -> None:
        # The 12-15% figure may appear, but never as a headline claim without its
        # retraction nearby.
        for name in ("README.md", "docs/ERGEBNISSE.md"):
            text = self.texts[name]
            if "12,4" in text or "15,0" in text:
                with self.subTest(document=name):
                    self.assertIn("wertlos", text, f"{name} cites the figure without the retraction")
