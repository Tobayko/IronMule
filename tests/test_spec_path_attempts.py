"""The amendment's retry counter — the one branch that can declare a cell dead."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "spec_path_measure", ROOT / "experiments" / "spec_path" / "measure.py"
)
measure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measure)

PATCHED = measure.PATCHED_SHA256
OTHER = "f" * 64


def test_first_attempt_of_an_empty_log():
    assert measure.attempt_number([], PATCHED, 1) == 1


def test_attempts_do_not_leak_across_widths():
    log = [{"runtime_py_sha256": PATCHED, "width": 2},
           {"runtime_py_sha256": PATCHED, "width": 2}]
    assert measure.attempt_number(log, PATCHED, 1) == 1
    assert measure.attempt_number(log, PATCHED, 2) == 3


def test_attempts_do_not_leak_across_code_versions():
    """A reworked patch starts at one; otherwise a cell dies before it ran."""

    log = [{"runtime_py_sha256": OTHER, "width": 1},
           {"runtime_py_sha256": OTHER, "width": 1},
           {"runtime_py_sha256": OTHER, "width": 1}]
    assert measure.attempt_number(log, PATCHED, 1) == 1
    assert measure.attempt_number(log, OTHER, 1) == 4 > measure.MAX_WARMUP_ATTEMPTS


def test_entries_without_the_hash_never_count():
    log = [{"width": 1}, {"runtime_py_sha256": None, "width": 1}]
    assert measure.attempt_number(log, PATCHED, 1) == 1
