import json
from pathlib import Path

import pytest

from training import sampling_evidence as evidence

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "lab/autonomous/results"
ORDER = "4e87c90dad8d9f21db28a69501613f44a98b9399b1b5838316f276978de288e7"
SAMPLE = DATA / "a100-calibration-sampling" / ("trial-" + ORDER)


def test_sampling_proposal_is_reproducible_without_online_services(tmp_path):
    index = evidence.manifest(SAMPLE)
    inputs = {name: (SAMPLE / name).read_bytes() for name in ("order.json", "decision.json", "request.json", "response.json", "candidate.diff")}
    ref, actual, qualified, rationale = evidence.verify(json.loads(inputs["order.json"]), inputs, DATA, tmp_path)
    assert evidence.sha(actual) == index["candidate_sha256"] and actual != ref
    assert qualified is None and rationale
    rows = list(evidence.experiences(ROOT, str((SAMPLE / "manifest.json").relative_to(ROOT))))
    assert {row["kind"] for row in rows} >= {"calibration_sampling_order", "calibration_sampling_candidate"}
    assert not any(row["label_eligible"] for row in rows)
    assert next(row for row in rows if row["kind"] == "calibration_sampling_order")["provenance"] == "simulated"


@pytest.mark.parametrize("bad_name,value", [
    ("candidate.diff", b"edited proposal"),
    ("outcome.json", json.dumps({"stage": "active", "order": ORDER}).encode()),
    ("result.json", json.dumps({"schema": "ironmule.followup-trial.v1", "passed": False, "artifacts": {}}).encode()),
])
def test_proposal_cannot_stand_in_for_bound_measurements(tmp_path, bad_name, value):
    inputs = {name: (SAMPLE / name).read_bytes() for name in ("order.json", "decision.json", "request.json", "response.json", "candidate.diff")}
    inputs[bad_name] = value
    with pytest.raises(ValueError):
        evidence.verify(json.loads(inputs["order.json"]), inputs, DATA, tmp_path)
