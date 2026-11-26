"""Historical decisions must retain their hardware and comparison boundaries."""

import json
from pathlib import Path

import pytest

from training import experience_archive as archive


def test_report_provenance_and_exclusions():
    rows = list(archive.build())
    assert len({row["id"] for row in rows}) == len(rows)
    placement = [r for r in rows if r["kind"] == "placement_pair"]
    good = [r for r in placement if r["label_eligible"]]
    assert len(good) == 80
    assert all(r["quality"]["response_and_stop_digests"][0] == r["quality"]["response_and_stop_digests"][1]
               for r in good)
    assert {r["reason"] for r in placement if not r["label_eligible"]} == {
        "response_digests_unavailable", "single_worker"}
    assert len({r["group"] for r in good}) == 2

    k1 = next(r for r in rows if r["locator"] == "./http_K1_32.jsonl:2")
    assert k1["metrics"][0]["tok_s"] == 297.06
    assert k1["base_commit"] == "02b285b"
    assert k1["label_eligible"] is False
    assert any(r["reason"] == "different_quantization" for r in rows)
    forecast = [r for r in rows if r["kind"] == "traffic_forecast"]
    assert forecast and all(r["provenance"] == "simulated" and not r["label_eligible"] for r in forecast)
    assert any(r["reason"] == "cross_pod_reference_changed" for r in rows)
    scratch = next(r for r in rows if r["source"] == "evidence/runtime/cpu-pilot-four-slot-discovery.json")
    assert scratch["decision"] is False and scratch["label_eligible"] is False
    assert scratch["quality"]["correct"] == scratch["quality"]["completed"] == 16
    [metric] = scratch["metrics"]
    assert metric["head_proposals"] == 5 and metric["scratch_measured_decisions"] == 0
    assert metric["search_and_warmup_s"] == pytest.approx(83.03, abs=.001)


def test_failed_rebuild_keeps_previous_index(tmp_path):
    out = tmp_path / "index.jsonl"
    archive.write_archive(out, iter([{"kind": "test", "name": "previous"}]))

    def interrupted():
        yield {"kind": "test", "name": "replacement"}
        raise ValueError("bad row")

    with pytest.raises(ValueError, match="bad row"):
        archive.write_archive(out, interrupted())
    assert list(archive.search(out, "previous", 1)) == [{"kind": "test", "name": "previous"}]
    assert list(tmp_path.iterdir()) == [out]


def test_mismatched_output_is_preserved_without_a_reward():
    path = Path("lab/placement/results/smollm2-cpu-reward-burst-v3.json")
    trace = json.loads(path.read_text())
    trace["trials"][0]["outputs"][1] = "0" * 64
    row = next(archive.placement(str(path), "test-only", trace))
    assert row["label_eligible"] is False
    assert row["reason"] == "different_completions"
    assert len(row["metrics"]) == 2


def test_published_code_change_rejects_and_retains_codex_rationale(tmp_path):
    root = Path(__file__).resolve().parents[2]
    raw = "lab/autonomous/results/cpu-tokenizer-01/result.json"
    index = tmp_path / "index.jsonl"
    rows = list(archive.code_experiment(root, raw, archive.canonical_sha256((root / raw).read_bytes())))
    archive.write_archive(index, rows)
    [rejected] = list(archive.search(index, "rejected GgufTokenizer.encode ctypes", 1))
    assert rejected["decision"] is False and rejected["label_eligible"] is False
    assert "4/6 pairs faster" in rejected["note"]
    assert "Codex" in rejected["authorship"] and "IronMule" in rejected["authorship"]
    assert len({r["group"] for r in rows}) == 1
    assert len([r for r in rows if r["kind"] == "code_trial_pair"]) == 6
