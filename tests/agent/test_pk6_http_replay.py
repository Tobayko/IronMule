import json
import tarfile

import pytest

from lab.pk6 import replay_http as replay


def test_complete_pr57_comparison_reproduces_original_verdict_and_delivery_totals():
    observed = replay.historical()
    assert observed["fixed_token_work"] and observed["facts_all_18"]
    scores = observed["original_scoring"]
    assert (scores["K"]["tok_s_32_median"], scores["vllm"]["tok_s_32_median"]) == (304.61, 276.88)
    assert (scores["K"]["ttft_p50_32_median"], scores["vllm"]["ttft_p50_32_median"]) == (2321.2, 1822.8)
    assert scores["verdict"]["ratio_tok_s"] == 1.1002
    assert observed["waves"]["K2-32"]["completion_tokens"] == [1024] * 3
    assert observed["waves"]["vllm-8"]["completion_tokens"] == [256] * 2


def test_inconsistent_rate_or_missing_measured_wave_is_rejected(tmp_path):
    name = "http_K1_32.jsonl"
    with tarfile.open(replay.POD37) as tar:
        rows = [json.loads(line) for line in tar.extractfile("./" + name)]
    rows[1]["tokens"] -= 128
    path = tmp_path / name
    path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="token rate inconsistent"):
        replay.waves(tmp_path, "K1", "ironmule", 32)
    path.write_text("\n".join(map(json.dumps, rows[:-1])))
    with pytest.raises(ValueError, match="missing warm-up"):
        replay.waves(tmp_path, "K1", "ironmule", 32)


def test_fresh_vllm_alone_reports_its_own_waves_and_no_cross_system_ratio(tmp_path, monkeypatch):
    doc = {"historical_H_artifacts_reused": False, "distinct_from_prior_H": True, "models": {
        "quantization_match": False, "vllm_reference": {"repo": "fixture-awq"}, "ironmule_product": {"sha256": "fixture"}},
        "http_protocol": {"fixture": {"client_widths": [32, 8], "max_tokens": 32, "seed": 20261119},
                          "order": ["product-K1", "vllm-AWQ", "product-K2"]},
        "immutable_source_manifest": {"files": {}}, "experiment_id": "fixture", "target": {"gpu": "fixture"}}
    registration = tmp_path / "preregistration.json"
    registration.write_text(json.dumps(doc))
    monkeypatch.setattr(replay, "FRESH_REGISTRATION_SHA", replay.checksum(registration))
    with tarfile.open(replay.POD37) as tar:
        for name in replay.filenames(("vllm",)):
            rows = [json.loads(line) for line in tar.extractfile("./" + name)]
            rows[0]["warmup_s"] += .3
            (tmp_path / name).write_text("\n".join(map(json.dumps, rows)))
    stats = replay.fresh(tmp_path)
    assert stats["status"] == "incomplete" and stats["throughput_ratio"] is None
    assert stats["recorded_starts"] == ["vllm"] and set(stats["per_engine"]) == {"vllm"}
    assert stats["missing"] == replay.filenames(("K1", "K2"))
    assert stats["facts_all_18_in_recorded_starts"]
    registration.write_text(registration.read_text() + " ")
    with pytest.raises(ValueError, match="immutable agreement"):
        replay.fresh(tmp_path)
