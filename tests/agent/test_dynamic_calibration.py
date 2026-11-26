import hashlib
from pathlib import Path

import pytest

from lab.pk6 import calibration_grade, startup_search, warm_result
from training.code_trial import read


ROOT=Path(__file__).resolve().parents[2]/"lab/autonomous/results"
TRIAL=ROOT/"a100-dynamic-calibration"


def test_real_source_rebuild_and_simulation_with_distinct_measured_inputs():
    source=startup_search.materialize(TRIAL)
    assert hashlib.sha256(source).hexdigest()==calibration_grade.CANDIDATE
    forecast=startup_search.rank(ROOT/"a100-pr57-warm-repeat",TRIAL,startup_search.baseline_source())
    assert forecast==read(TRIAL/"forecast.json")
    assert forecast["kind"]=="simulated"
    assert len(set(forecast["inputs"].values()))==2


def test_the_unmodified_http_control_prevents_publication():
    directory=TRIAL/"http"
    first,second=(read(directory/f"aa-{i}.json") for i in (0,1))
    with pytest.raises(ValueError,match="selected profile or delivered work changed"):
        calibration_grade.pair(first,second,98600,False,same=True)
    diagnostic=read(directory/"diagnostic-candidate.json")
    calibration_grade.validated(diagnostic,"candidate",98600,False)
    assert diagnostic["pk2_calls"]==15
    assert read(directory/"result.json")["status"]=="rejected"


def test_prospective_http_score_replays_all_288_width_32_requests():
    folder=ROOT/"a100-pr57-warm-repeat"
    assert warm_result.result(folder)==read(folder/"result.json")
