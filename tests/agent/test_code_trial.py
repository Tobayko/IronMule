"""Independent crossover gates and process isolation for code variants."""
from copy import deepcopy

import pytest

from training.code_trial import PAIRS, judge


def pairs():
    rows = []
    for i in range(PAIRS):
        ref = dict(peak_rss_kib=4000, tokens=48, completed=16, correct=16, wall_s=1, ttft_s=.3,
                   output_sha256="aa", workload={"seed": i}, binding={"machine": "fixture"},
                   evaluator_sha256="bb", profiled=False)
        cand = {**ref, "wall_s": .94, "ttft_s": .29}
        rows.append(deepcopy((ref, cand)))
    return rows


def test_gates_require_all_complete_bound_pairs():
    rows = pairs()
    assert judge(rows)["passed"]
    with pytest.raises(ValueError):
        judge(rows[:-1])
    for field, value in (("correct", 15), ("completed", 17), ("output_sha256", "different"),
                         ("wall_s", 1.01), ("peak_rss_kib", 4201), ("profiled", True),
                         ("tokens", 46), ("evaluator_sha256", "different")):
        rows = pairs()
        rows[1][1][field] = value
        assert not judge(rows)["passed"], field
    rows = pairs()
    for _, cand in rows:
        cand["ttft_s"] = .301
    assert not judge(rows)["passed"]
    for row in rows[3]:
        row["binding"] = {"machine": "new"}
    assert not judge(rows)["passed"]
    rows[0][1]["wall_s"] = float("nan")
    with pytest.raises(ValueError):
        judge(rows)
