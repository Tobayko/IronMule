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


def test_authored_calibrator_preserves_live_nll_and_full_sampling_on_uncertainty(monkeypatch):
    import ast
    import math
    import os
    from collections import Counter
    from types import SimpleNamespace

    original=startup_search.baseline_source()
    generated=startup_search.materialize(TRIAL).decode()
    observed=read(TRIAL/"init-K1.json")["phases"]["measure_pk2"]["calls"]
    thresholds=(0,128,256,512,1024)

    def evaluate(source, enabled, *, drift=False, likelihood=1.00073):
        clock=0.
        chosen=0
        counts=Counter()
        numeric=[]
        def select(t):
            nonlocal chosen
            chosen=t
            return 0
        def burst(group):
            nonlocal clock
            position=counts[chosen]
            duration=observed[5*position+thresholds.index(chosen)][0]
            if drift and chosen==128 and position==2:
                duration*=1.5
            counts[chosen]+=1
            clock+=duration
        def nll(chunk,bos):
            numeric.append((chosen,tuple(chunk),tuple(bos)))
            return math.log(likelihood) if chosen else 0.
        fake=SimpleNamespace(gpu_layers=999,pack=True,L=SimpleNamespace(llama_vocab_bos=lambda v:0),
                             vocab=None,seqs=32,seq_ctx=2048,tokenize=lambda _:list(range(5000)),
                             _clear=lambda:None,_pack_prefill=burst,chunk_nll=nll)
        monkeypatch.setenv("IRONMULE_TRIAL_LOCAL_CALIBRATION",str(int(enabled)))
        env={"os":os,"math":math,"time":SimpleNamespace(perf_counter=lambda:clock),
             "pk2_set_min_rows":select,"Job":lambda ids,limit:SimpleNamespace(ids=ids),
             "OPTION_BOUND":1.005,"PK2_CALIBRATION":"mix"}
        node=startup_search.source_methods(source)["measure_pk2"]
        import textwrap
        code=textwrap.dedent(ast.get_source_segment(source,node))
        exec("from __future__ import annotations\n"+code,env)
        result=env["measure_pk2"](fake)
        return result,counts,numeric,chosen

    ref,original_bursts,reference_nll,chosen=evaluate(original,False)
    assert sum(original_bursts.values())==20 and chosen==128
    assert evaluate(generated,False)==(ref,original_bursts,reference_nll,chosen)
    candidate,shortened,checks,selected=evaluate(generated,True)
    assert sum(shortened.values())==15 and set(candidate)==set(ref)
    assert checks==reference_nll and selected==chosen
    assert evaluate(generated,True,drift=True)==evaluate(original,False,drift=True)
    _,_,checks,selected=evaluate(generated,True,likelihood=1.02)
    assert len(checks)==2 and selected==0


def test_resume_training_reuses_the_measured_report_and_exact_weights(tmp_path):
    from lab.pk6 import ordered_release as deploy

    first = deploy.fit(tmp_path, TRIAL / "ordered")
    assert deploy.fit(tmp_path, TRIAL / "ordered") == first
    assert (tmp_path / "ordered-head.npz").read_bytes() == (TRIAL / "ordered-head.npz").read_bytes()
    assert first["metrics"]["holdout_correct_profile"]


def test_gate_binds_the_actual_settings_board_facts_and_reference_likelihood():
    from copy import deepcopy
    from lab.pk6.ordered_quality import validate

    observed = read(TRIAL / "numerical/observed.json")
    settings = read(TRIAL / "ordered/aa-0.json")["binding"]["settings"]
    assert validate(observed, settings)["chunks"] == 16
    damaged = deepcopy(observed)
    for row in [*damaged["gate"]["tried"], damaged["gate"]["streams"]]:
        row["factual"]["candidate_correct"] = 17
    with pytest.raises(ValueError, match="quality receipt"):
        validate(damaged, settings)
    with pytest.raises(ValueError, match="different device"):
        validate({**observed, "device_sha256":"0"*64}, settings)
