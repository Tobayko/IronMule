"""Grouped Pilot replay does not silently turn partial trials into timing labels."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training.pilot_threads import examples, fit, holdout_uncertainty, save


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


def test_uncertainty_audit_keeps_incomplete_cells_in_coverage():
    root = Path(__file__).resolve().parents[2] / "evidence/runtime"
    record = json.loads((root / "cpu-pilot-four-slot-trials.json").read_text())
    with np.load(root / "cpu-pilot-four-slot-candidate.npz", allow_pickle=False) as model:
        weights = [model[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")]
    audited = holdout_uncertainty(record, weights)
    assert audited["observed_cells"] == 6 and audited["complete_cells"] == 2
    assert audited["correct_among_resolved"] == 0 and audited["resolved_decisions"] == 1
    assert audited["probability_calibration"] == "unknown"
