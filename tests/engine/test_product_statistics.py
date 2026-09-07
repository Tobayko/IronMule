"""Pure statistics portability and validation contracts."""

from __future__ import annotations

import math
from pathlib import Path
import os
import subprocess
import sys

import pytest

from friday_evidence.statistics import paired_ratio, summarise


ROOT = Path(__file__).resolve().parents[2]


def test_import_from_outside_project_does_not_load_model_stack(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from friday_evidence.statistics import summarise; print('mlx' in sys.modules, 'mlx_lm' in sys.modules, 'numpy' in sys.modules, summarise([0, 1])['median'])"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False False 0.5"


def test_summarise_preserves_historical_numerics_and_allows_zero() -> None:
    assert summarise([3.0, 1.0, 2.0, 5.0, 4.0]) == {
        "n": 5, "median": 3.0, "min": 1.0, "max": 5.0,
        "p95": 5.0, "stdev": math.sqrt(2.5),
    }
    assert summarise([0.0]) == {"n": 1, "median": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0, "stdev": 0.0}


def test_paired_ratio_preserves_default_seed_and_resampling() -> None:
    result = paired_ratio([9.0, 8.0, 10.0], [10.0, 10.0, 10.0], resamples=200)
    assert result["median_ratio"] == 0.9
    assert result["ci_low"] <= result["median_ratio"] <= result["ci_high"]
    assert result["pairs"] == [0.9, 0.8, 1.0]


@pytest.mark.parametrize("samples", [[], [float("nan")], [float("inf")], [True], [-1.0], [10**1000]])
def test_summarise_rejects_invalid_samples(samples: list[object]) -> None:
    with pytest.raises(ValueError):
        summarise(samples)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("candidate", "baseline", "resamples"),
    [([1.0], [], 10), ([0.0], [1.0], 10), ([1.0], [0.0], 10), ([math.nan], [1.0], 10), ([True], [1.0], 10), ([1.0], [1.0], 0), ([1.0], [1.0], True)],
)
def test_paired_ratio_rejects_invalid_or_truncated_inputs(candidate: list[object], baseline: list[object], resamples: object) -> None:
    with pytest.raises(ValueError):
        paired_ratio(candidate, baseline, resamples=resamples)  # type: ignore[arg-type]


def test_paired_ratio_rejects_nonfinite_and_negative_timings() -> None:
    with pytest.raises(ValueError):
        paired_ratio([-1.0], [1.0])
    with pytest.raises(ValueError):
        paired_ratio([1.0], [float("inf")])
