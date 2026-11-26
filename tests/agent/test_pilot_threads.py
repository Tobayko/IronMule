"""Grouped Pilot replay does not silently turn partial trials into timing labels."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from training.pilot_threads import examples, fit, save


def record():
    groups = []
    for band in range(1, 8):
        group = {"group": f"ctx-bin-{band}", "context": band * 64, "rows": 1, "features": [0.] * 20,
                 "base": 1, "observations": [
                     {"threads": n, "repetition": r, "seconds": (1 / n if band < 4 else n), "tokens_digest": "a"}
                     for r in (0, 1) for n in (1, 2)]}
        groups.append(group)
    return {"binding": "test", "profile": {"cores": 2}, "groups": groups}


@pytest.mark.parametrize("field,value", [("seconds", None), ("tokens_digest", "changed"), ("repetition", 1)])
def test_complete_counterbalanced_pairs(field, value):
    r = record()
    r["groups"][0]["observations"][0][field] = value
    assert len(list(examples(r))) == 6


def test_grouped_fit_updates_scores_and_splits_context_families(tmp_path):
    r = record()
    r["groups"].extend(deepcopy(r["groups"]))
    args = SimpleNamespace(trials=tmp_path / "trials.json", weights=tmp_path / "weights.npz",
                           report=tmp_path / "report.json", seed=73)
    save(args.trials, r)
    fit(args)
    import json
    report = json.loads(args.report.read_text())
    assert not set(report["train_groups"]).intersection(g["group"] for g in report["heldout"])
    assert all(g["scores_before"] != g["scores_after"] for g in report["heldout"])
    with np.load(args.weights) as weights:
        assert weights["binding"] == "test"
