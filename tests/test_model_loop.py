"""Offline checks for H2: parsing what a language model proposes.

No GPU, no model, no network.  This is the trust boundary of the whole tool: the
model is an untrusted source of integers, and everything downstream assumes the
parser already threw away anything unusable.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "model_loop", PROJECT_ROOT / "tools" / "model_loop.py"
)
h2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(h2)


class ProposalParsingTest(unittest.TestCase):
    TRIED = {2, 4}

    def parse(self, text: str) -> list[int]:
        return h2.parse_proposals(text, already_tried=self.TRIED)

    def test_plain_json_array(self) -> None:
        self.assertEqual(self.parse("[3, 5, 7]"), [3, 5, 7])

    def test_fenced_json_is_accepted(self) -> None:
        # Gemma answers with a ```json fence in practice.
        self.assertEqual(self.parse("```json\n[3, 5]\n```"), [3, 5])

    def test_already_measured_values_are_dropped(self) -> None:
        self.assertEqual(self.parse("[2, 4, 6]"), [6])

    def test_out_of_range_values_are_dropped(self) -> None:
        self.assertEqual(self.parse("[900, 0, -3, 17, 1]"), [])

    def test_floats_are_not_silently_truncated(self) -> None:
        self.assertEqual(self.parse("[3.5, 5]"), [5])

    def test_booleans_are_not_treated_as_integers(self) -> None:
        # bool subclasses int; True must never become batch size 1.
        self.assertEqual(self.parse("[true, false, 5]"), [5])

    def test_numeric_strings_are_rejected(self) -> None:
        self.assertEqual(self.parse('["7", 5]'), [5])

    def test_nested_structures_discard_the_whole_answer(self) -> None:
        # The extractor stops at the first "]", so a nested answer yields invalid
        # JSON and nothing survives -- not even the usable 5.  Failing closed is
        # the right call here: nesting means the answer is malformed, and
        # salvaging fragments from a malformed answer is how parsers get abused.
        self.assertEqual(self.parse("[[3], {}, 5]"), [])


class UntrustedInputTest(unittest.TestCase):
    """A bad answer must cost a round and nothing else."""

    def parse(self, text: str) -> list[int]:
        return h2.parse_proposals(text, already_tried=set())

    def test_prose_executes_nothing(self) -> None:
        self.assertEqual(self.parse("I suggest trying 5 and 7, they look promising."), [])

    def test_shell_injection_executes_nothing(self) -> None:
        for hostile in ("[$(rm -rf /)]", "[`whoami`]", "['; DROP TABLE runs; --']"):
            with self.subTest(payload=hostile):
                self.assertEqual(self.parse(hostile), [])

    def test_empty_and_malformed_answers_yield_nothing(self) -> None:
        for text in ("", "[", "[1,2", "null", "{}", "[]"):
            with self.subTest(text=text):
                self.assertEqual(self.parse(text), [])

    def test_a_huge_list_cannot_flood_the_measurement_budget(self) -> None:
        flood = "[" + ",".join(str(n) for n in range(2, 17)) + "]"
        self.assertLessEqual(len(self.parse(flood)), h2.MAX_PROPOSALS)

    def test_duplicates_collapse(self) -> None:
        self.assertEqual(self.parse("[5, 5, 5, 6]"), [5, 6])

    def test_every_accepted_value_is_inside_the_registered_range(self) -> None:
        for value in self.parse("[2, 9, 16]"):
            self.assertGreaterEqual(value, h2.MIN_BATCH)
            self.assertLessEqual(value, h2.MAX_BATCH)


class PromptTest(unittest.TestCase):
    def test_prompt_shows_measured_evidence(self) -> None:
        prompt = h2.build_prompt(
            [{"batch_size": 4, "ratio": 0.83, "clears_mde": True}], {4}
        )
        self.assertIn("N=4", prompt)
        self.assertIn("0.830", prompt)
        self.assertIn("beats threshold", prompt)

    def test_prompt_marks_candidates_that_failed(self) -> None:
        prompt = h2.build_prompt(
            [{"batch_size": 3, "ratio": 0.97, "clears_mde": False}], {3}
        )
        self.assertIn("not beyond noise", prompt)

    def test_prompt_handles_an_empty_first_round(self) -> None:
        prompt = h2.build_prompt([], set())
        self.assertIn("nothing measured yet", prompt)

    def test_prompt_carries_the_measured_device_facts(self) -> None:
        # The model is supposed to reason from real numbers, not guesses.
        prompt = h2.build_prompt([], set())
        self.assertIn("340 ms", prompt)


class ReleaseGateTest(unittest.TestCase):
    def test_locked_without_execute(self) -> None:
        self.assertEqual(h2.main([]), 78)

    def test_self_check_passes(self) -> None:
        self.assertEqual(h2.main(["--self-check"]), 0)

    def test_thresholds_are_shared_with_the_harness(self) -> None:
        # H2 must not quietly use a friendlier threshold than the other tools.
        self.assertEqual(h2.MDE, h2._LOOP.MDE)


if __name__ == "__main__":
    unittest.main()
