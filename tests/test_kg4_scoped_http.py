import json

import pytest

from lab.kg4 import scoped_http as study


def hit(bucket=None, value=11):
    counts = [0] * 64
    if bucket is not None:
        counts[bucket - 1] = value
    return {"kg4": counts, "cublas": [0]}


def state(low, wide, switches):
    return {
        "mode": "low-width8",
        "q4_buckets": [6, 7, 8],
        "low_runs": low,
        "wide_runs": wide,
        "switches": switches,
        "parked": True,
    }


def test_real_low_then_wide_transition_in_http32_is_allowed():
    scope = study.scope_delta(state(2, 3, 4), state(3, 4, 6), True)
    assert study.dispatch(hit(6), scope, candidate=True, width=32, performance=True)
    assert not study.dispatch(
        hit(32), scope, candidate=True, width=32, performance=True
    )
    assert not study.dispatch(
        hit(38), scope, candidate=True, width=32, performance=True
    )
    assert not study.dispatch(hit(6), scope, candidate=True, width=8, performance=True)
    assert not study.dispatch(hit(6), {}, candidate=False, width=8, performance=True)


def test_demand_coverage_and_low_hits_are_required_for_timing():
    low = study.scope_delta(state(0, 0, 0), state(1, 0, 2), True)
    wide = study.scope_delta(state(0, 0, 0), state(0, 1, 0), True)
    assert study.dispatch(hit(8), low, candidate=True, width=8, performance=True)
    assert not study.dispatch(hit(), low, candidate=True, width=8, performance=True)
    assert not study.dispatch(hit(8), low, candidate=True, width=32, performance=True)
    assert not study.dispatch(hit(8), wide, candidate=True, width=32, performance=True)
    assert study.dispatch(hit(), wide, candidate=True, width=32, performance=True)
    with pytest.raises(ValueError, match="restarted"):
        study.scope_delta(state(4, 0, 4), state(3, 0, 4), True)


def test_disjoint_performance_and_factual_inputs():
    seen = set()
    for rep in range(3):
        for width in study.WIDTHS:
            prompts, qs = study.inputs(1178421 + 200003, rep, width)
            group = prompts + [q.prompt for q in qs]
            assert len(prompts) == width and len(qs) == 18
            assert set(group).isdisjoint(seen)
            assert {q.category for q in qs} == {"extraction", "facts", "arithmetic"}
            seen.update(group)


def test_unqualified_receipts_require_no_capsule_gpu_or_model(tmp_path):
    receipt = tmp_path / "native.json"
    receipt.write_text("{}")
    out = tmp_path / "report.json"
    assert (
        study.main(
            [
                "--gguf",
                str(tmp_path / "unused"),
                "--tokenizer",
                str(tmp_path / "unused"),
                "--receipt-a",
                str(receipt),
                "--receipt-b",
                str(receipt),
                "--receipt-c",
                str(receipt),
                "--state-dir",
                str(tmp_path / "never-opened"),
                "--output",
                str(out),
                "--budget-seconds",
                "1",
            ]
        )
        == 0
    )
    result = json.loads(out.read_bytes())
    assert (
        result["status"] == "skipped"
        and result["eligible"] is False
        and result["rows"] == []
    )
    assert not (tmp_path / "never-opened").exists()
