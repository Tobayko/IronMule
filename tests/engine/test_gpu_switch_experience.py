"""The learning CLI records ordered controller experiments without qualifying the challenger."""
import hashlib
import json

import pytest

from ironmule import dream, gpu_experience


MODEL = "6" * 64
GPU = [{"provider": "cuda", "description": "fixture GPU"}]


def report(*, faster=False, correct=True, repeats=1):
    result = {"schema": "ironmule.ggml-switch-probe.v1", "status": "complete", "provider": "cuda", "devices": GPU,
              "model_sha256": MODEL, "machine": "a box", "cost_aware_gpu_branch_active": True,
              "serving_offloaded_layers": [[4, 4]], "long_prompt": 256, "threads": 2, "max_output_tokens": 3,
              "seq_ctx": 512, "source_sha256": {"lab/portable_gpu/switch_probe.py": "7" * 64}, "repeats": repeats,
              "replays": [], "all_output_ids_and_stops_match": correct, "all_copy_correct": correct}
    for seed in range(repeats):
        order = ("unpriced", "cost_aware") if seed % 2 == 0 else ("cost_aware", "unpriced")
        for arm in order:
            challenger = arm == "cost_aware"
            duration = (0.5 if faster else 2) if challenger else 1
            ok = correct or not challenger
            request = {"expected": "silver", "correct": ok, "ttft_ms": 10 * duration,
                       "tokens": [11 if ok else 99], "stop": "length"}
            wave = {"wave": "short", "elapsed_s": duration, "physical_switch_s": None, "requests": [request]}
            result["replays"].append({"policy": arm, "seed": seed, "waves": [wave]})
    return result


@pytest.mark.parametrize("faster,correct,repeats,outcome",
                         [(False, True, 1, "regression"), (True, True, 2, "improvement"), (True, False, 1, "quality_rejected")])
def test_cli_keeps_observed_switch_verdict_separate_from_width_training(tmp_path, capsys, faster, correct, repeats, outcome):
    state, path = tmp_path / "state", tmp_path / "report.json"
    path.write_text(json.dumps(report(faster=faster, correct=correct, repeats=repeats)))
    assert dream.main(["--import-gpu-report", str(path), "--state-dir", str(state)]) == 0
    result = json.loads(capsys.readouterr().out)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = state / "gpu-switch-experience" / MODEL / f"{digest}.json"
    assert json.loads(record.read_text()) == result == gpu_experience.import_report(path, state)
    assert result["outcome"] == outcome and result["qualification"] == "observation_only"
    assert result["repeats"] == repeats and result["both_orders_measured"] == (repeats == 2)
    assert result["cost_aware_throughput_change_pct"] == (100 if faster else -50)
    assert result["cost_aware_median_ttft_change_pct"] == (-50 if faster else 100)
    assert result["all_copy_correct"] == correct and result["machine"] == "a box"
    assert not (state / "gpu-width-experience").exists()
    assert gpu_experience.suggest(state, MODEL, GPU, "a box", 512) is None


def test_an_unpaired_run_and_false_output_match_cannot_be_recorded(tmp_path):
    path = tmp_path / "bad.json"
    observation = report(correct=False)
    observation["all_output_ids_and_stops_match"] = True
    path.write_text(json.dumps(observation))
    with pytest.raises(ValueError, match="quality summary"):
        gpu_experience.import_report(path, tmp_path)
    observation["replays"][1]["policy"] = "unpriced"
    path.write_text(json.dumps(observation))
    with pytest.raises(ValueError, match="one arm"):
        gpu_experience.import_report(path, tmp_path)
    assert not (tmp_path / "gpu-switch-experience").exists()
