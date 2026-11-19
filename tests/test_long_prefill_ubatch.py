import copy
import json
from types import SimpleNamespace

import pytest

from lab.portable_gpu import long_prefill_ubatch as study


def test_rejected_parent_does_not_launch_or_touch_model(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("rejected receipt launched the expensive child")

    monkeypatch.setattr(study.subprocess, "run", forbidden)
    record = tmp_path / "rejected.json"
    record.write_text('{"status":"rejected"}')
    out = tmp_path / "out"
    assert (
        study.main(
            [
                "--gguf",
                str(tmp_path / "missing"),
                "--receipt",
                str(record),
                "--out-dir",
                str(out),
                "--budget-seconds",
                "1",
            ]
        )
        == 0
    )
    assert (
        json.loads((out / "ubatch-study.json").read_bytes())["reason"]
        == "unqualified_512"
    )
    assert list(out.iterdir()) == [out / "ubatch-study.json"]


def test_holdout_rejects_different_tokens_or_stops_even_if_scores_match():
    case = {"reference": {"tokens": [42], "stop": "length", "correct": True}}
    prior = {
        "matches": True,
        "shapes": {"2x444": {"matches": True, "cases": [case], "witnesses": {}}},
    }
    assert study.same_holdout(prior, prior)
    for field, value in (("tokens", [99]), ("stop", "eos"), ("correct", False)):
        different = copy.deepcopy(prior)
        different["shapes"]["2x444"]["cases"][0]["reference"][field] = value
        assert not study.same_holdout(prior, different)


@pytest.mark.parametrize("available", [None, (100, 1000)])
def test_memory_fails_closed_when_unknown_or_too_small(monkeypatch, available):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(MemoryError):
        study.memory(SimpleNamespace(gpu_memory_bytes=lambda _: available), 101)
