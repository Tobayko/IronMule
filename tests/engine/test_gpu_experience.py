"""Imported GPU measurements can teach search priorities, never qualify serving on their own."""
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ironmule import gpu_experience as exp

GPU = [{"description": "fixture card", "provider": "cuda", "name": "CUDA0"}]
MODEL = hashlib.sha256(b"model").hexdigest()


def pair(role, narrow, wide, matched=True, ctx=32):
    return {"role": role, "prompt_tokens": ctx, "concurrency": 4,
            "quality_matched": matched, "exact_output_match": matched, "all_copy_tasks_correct": True,
            "arms": [{"seqs": width, "tokens_per_second": [rate, rate], "ttft_ms": [100 / rate] * 2}
                     for width, rate in ((1, narrow), (4, wide))]}


def put(tmp_path, rows, filename="result", **changes):
    data = {"schema": exp.SCHEMA, "status": "complete", "stage": "compare", "offloaded_layers": [[9, 9]],
            "provider": "cuda", "devices": GPU, "machine": "box", "model_sha256": MODEL, "runs": rows, **changes}
    path = tmp_path / (filename + ".json")
    path.write_text(json.dumps(data))
    return exp.import_report(path, tmp_path)


def test_fitted_width_hints_combine_losses_after_restart_and_deduplicate_overlap(tmp_path):
    win = pair("calibration", 10, 20)
    first = put(tmp_path, [win, pair("holdout", 11, 22, ctx=128)])
    assert first["fit"]["training_pairs"] == first["fit"]["holdout_pairs"] == 1
    assert len(first["fit"]["weights"]) == 5
    assert first["fit"]["holdout_mae"][0] < first["fit"]["equal_cost_mae"][0]
    assert exp.suggest(tmp_path, MODEL, GPU, "box", 512)["widths"] == [4, 1]
    put(tmp_path, [win, pair("calibration", 40, 10)], filename="repeat")
    learned = exp.suggest(tmp_path, MODEL, GPU, "box", 512)
    assert learned["widths"] == [1, 4] and learned["training_pairs"] == 2
    put(tmp_path, [win], filename="duplicate", extra_field="different serialization")
    assert exp.suggest(tmp_path, MODEL, GPU, "box", 512) == learned
    assert exp.suggest(tmp_path, "0"*64, GPU, "box", 512) is None
    assert exp.suggest(tmp_path, MODEL, [{"description": "other", "provider": "rocm"}], "box", 512) is None
    put(tmp_path, [pair("calibration", 20, 11, matched=False)], filename="failed")
    assert exp.suggest(tmp_path, MODEL, GPU, "box", 512) is None


def test_relabelled_holdout_never_enters_a_pooled_fit(tmp_path):
    winning = pair("calibration", 10, 30)
    put(tmp_path, [winning, pair("calibration", 25, 12)])
    put(tmp_path, [pair("calibration", 25, 12), {**winning, "role": "holdout"}], filename="other")
    learned = exp.suggest(tmp_path, MODEL, GPU, "box", 512)
    assert learned["training_pairs"] == 1 and learned["widths"] == [1, 4]


def test_import_retains_slow_and_failed_transition_observations_outside_fit(tmp_path):
    training = [pair("calibration", 10, 20)]
    original = put(tmp_path, training)
    control = {"prompt_tokens": 32, "concurrency": 1, "seqs": 1, "reconfigured": False,
               "elapsed_s": 1.0, "ttft_ms": 4, "correct": [True], "output_tokens": [[42]]}
    slower = {**control, "seqs": 4, "reconfigured": True, "elapsed_s": 3.0, "ttft_ms": 15}
    failed = {**slower, "correct": [False], "output_tokens": [[13]]}
    replay = {"arms": {"fixed": [[control], [control]], "selected": [[slower], [failed]]},
              "quality_matched": False, "switches": 2}
    imported = put(tmp_path, training, filename="replay", transition=replay)
    assert imported["fit"] == imported["pooled_fit"] == original["fit"]
    assert imported["transition"] == {"quality_matched": False, "switches": 2,
                                      "selected_over_fixed_elapsed": [3.0, 3.0]}
    record = tmp_path / "gpu-width-experience" / MODEL / (imported["source_sha256"] + ".json")
    waves = json.loads(record.read_text())["transition"]["waves"]
    assert waves[0][0]["matched"] and waves[0][0]["selected"]["ttft_ms"] == 15
    assert not waves[1][0]["matched"] and waves[1][0]["selected"]["reconfigured"]
    with pytest.raises(ValueError, match="disagrees"):
        put(tmp_path, training, filename="invalid", transition={**replay, "quality_matched": True})


@pytest.mark.parametrize("kwargs", [{"offloaded_layers": []}, {"status": "incomplete"}, {"provider": "cpu"},
                                   {"model_sha256": "x"}])
def test_import_refuses_incomplete_non_gpu_or_unbound_records(tmp_path, kwargs):
    with pytest.raises(ValueError):
        put(tmp_path, [pair("calibration", 1, 2)], **kwargs)
    assert not (tmp_path / "gpu-width-experience").exists()


def test_serving_remeasures_cost_and_filters_unmatched_new_widths(tmp_path, monkeypatch):
    from ironmule import ggml_agent, ggml_engine as ge
    put(tmp_path, [pair("calibration", 10, 20)])
    model = tmp_path / "test.gguf"
    model.write_bytes(b"model")
    monkeypatch.setattr(ge, "ggml_devices", lambda: GPU)
    monkeypatch.setattr(ge, "gpu_layers", lambda: 99)
    monkeypatch.setattr(ge, "gpu_memory_bytes", lambda device: (10*2**30, 10*2**30))
    monkeypatch.setattr(ge, "machine_key", lambda: "box")

    def measure(command, **kwargs):
        payload = json.loads(kwargs["input"])
        if command[2] == ge.CELL_CHILD:
            answer = {"cell_bytes": 100}
        else:
            widths = payload["widths"]
            assert payload["experience"]["concurrency"] == 4 and 1 in widths and 4 in widths
            answer = {"rates": {str(w): w*5 for w in widths}, "step_s": {str(w): 0.12 for w in widths},
                      "settings": {}, "output_digests": {str(w): ("bad" if w == 1 else "ok") for w in widths},
                      "workload_steps": {str(w): {"rows": w, "kv": w*32, "seconds": 0.07} for w in (1, 4)}}
        return SimpleNamespace(returncode=0, stdout=json.dumps(answer))
    monkeypatch.setattr(subprocess, "run", measure)
    result = ge.choose_sequences(str(model), cache=tmp_path / "cache", experience_root=tmp_path)
    assert result["sequences"] == 64  # the standard fitted widths and their measured rates choose this
    configs = ggml_agent.configs_from_measurement(result["measured"], result["seq_ctx"])
    assert "w1" not in configs and "w4" in configs
    assert (4, 0, 128, 0.07) in ggml_agent.anchors_of(configs["w4"])


def test_import_cli_defaults_to_private_product_state_read_by_serve(tmp_path, monkeypatch):
    from ironmule import dream
    from ironmule_product.state import ProductStore

    monkeypatch.delenv("IRONMULE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    fixture = tmp_path / "source"
    fixture.mkdir(mode=0o700)
    put(fixture, [pair("calibration", 10, 15)])
    assert dream.main(["--import-gpu-report", str(fixture / "result.json")]) == 0
    state = ProductStore().root
    assert state == tmp_path / ".ironmule" / "product"
    assert state.stat().st_mode & 0o777 == 0o700
    ProductStore().setup("server")
    assert exp.suggest(state, MODEL, GPU, "box", 512)["widths"] == [4, 1]


def test_leave_one_gpu_model_out_separates_devices_quality_and_duplicate_trials(tmp_path):
    from lab.portable_gpu.transfer import evaluate

    put(tmp_path, [pair("calibration", 10, 40), pair("holdout", 12, 36)])
    assert not evaluate(tmp_path)["folds"]
    put(tmp_path, [pair("calibration", 10, 40)], filename="overlap", note="same physical source trial")
    put(tmp_path, [pair("calibration", 40, 10), pair("holdout", 33, 11),
                   pair("holdout", 15, 25, matched=False)], filename="second",
        machine="other box", devices=[{"description": "second card", "name": "CUDA0", "provider": "cuda"}])
    report = evaluate(tmp_path)
    assert len(report["folds"]) == 2 and not report["skipped"]
    folds = {f["target_gpu"]: f for f in report["folds"]}
    assert folds["fixture card"]["metrics"]["single_slot"] == 1  # the other card's calibration prefers one
    blind = folds["second card"]
    assert blind["training_pairs"] == 1 and blind["metrics"]["wider"] == 2
    assert blind["metrics"]["unsafe_wider_quality"] == 1
    assert blind["metrics"]["matched_wider_lost_throughput"] == blind["metrics"]["matched_wider_later_ttft"] == 1
    assert report["corpus"]["scored_pairs"] == 2
    assert report["corpus"]["throughput"]["model_mae"] > report["corpus"]["throughput"]["equal_cost_mae"]
