"""The S3 replayer: reconstruct iteration structure from the emitted tokens."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "identity_break_measure", ROOT / "experiments" / "identity_break" / "measure.py"
)
measure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measure)


def test_a_sequence_without_repeats_accepts_nothing():
    """No prompt n-gram repeats -> the draft is empty -> one token per iteration."""

    prompt = list(range(100, 140))
    emitted = [200, 201, 202, 203, 204]
    entries = measure.replay(prompt, emitted, ngram=3, k=2)
    assert measure.well_formed(entries, k=2)
    assert [entry["j"] for entry in entries[1:]] == [0, 0, 0, 0]
    assert [entry["iteration"] for entry in entries[1:]] == [0, 1, 2, 3]


def test_a_repeated_prompt_ngram_is_accepted_and_groups_tokens():
    """The prompt ends with `7 8 9 10`; after emitting `7 8 9` the lookup drafts
    `10`, and because the run really emitted `10` there, the gate accepted it —
    so both land in one iteration."""

    prompt = [1, 2, 3, 7, 8, 9, 10, 11, 4, 5, 6]
    emitted = [7, 8, 9, 10, 11]
    entries = measure.replay(prompt, emitted, ngram=3, k=2)
    assert measure.well_formed(entries, k=2)
    grouped = {}
    for entry in entries[1:]:
        grouped.setdefault(entry["iteration"], []).append(entry["j"])
    assert any(len(js) > 1 for js in grouped.values()), grouped
    assert all(js == list(range(len(js))) for js in grouped.values())


def test_first_divergence_finds_the_index():
    assert measure.first_divergence([1, 2, 3], [1, 2, 4]) == 2
    assert measure.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert measure.first_divergence([1, 2, 3], [1, 2]) == 2


def test_well_formed_rejects_an_oversized_iteration():
    broken = [{"index": 0, "iteration": None, "j": None, "draft": []}] + [
        {"index": i, "iteration": 0, "j": i - 1, "draft": []} for i in range(1, 6)
    ]
    assert not measure.well_formed(broken, k=2)
