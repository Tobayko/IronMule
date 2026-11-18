"""Verified rival gaps can direct a native holdout. Only measured local wins change a bound budget."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ironmule import compare_learning as learn


def comparison():
    b = {"gguf_sha256": "1" * 64, "machine": "fixture", "seq_ctx": 512, "prompt_budget": 0}
    report = {"schema": "ironmule.compare.v1", "device": "cpu:host", "protocol": {"max_tokens": 6},
              "arms": [], "runs": []}
    for engine, rate, ttft in (("ironmule", 10, 40), ("vllm", 15, 37), ("llama_cpp", 9, 43)):
        identity = {"gguf_sha256": b["gguf_sha256"], "tokenizer_sha256": "3" * 64, "device": report["device"],
                    "ggml": b, "autonomous": engine == "ironmule", "placement_verified": True, "version": "test",
                    "baseline_stock": engine == "llama_cpp"}
        report["arms"].append({"engine": engine, "repeat": 0, "identity": identity,
                               "tokenization": {"verified": True, "digest": "a" * 64}})
        report["runs"].append({"engine": engine, "repeat": 0, "concurrency": 4, "quality": {"passed": 4},
                               "performance_prompt_tokens": [32, 32, 64, 64], "generation": [[6, "length"]] * 4,
                               "tokens_per_second": rate, "ttft_p95_ms": ttft})
    return report


def test_only_verified_paired_competitor_gaps_select_work():
    r = comparison()
    choice = learn.select(r)
    assert choice["opponent"] == "vllm" and choice["focus"] == "throughput"
    r["arms"][1]["tokenization"]["digest"] = "f" * 64
    assert learn.select(r) is None
    r["arms"][1]["tokenization"]["digest"] = "a" * 64
    r["runs"][1]["performance_prompt_tokens"] = [63] * 4
    assert learn.select(r) is None
    r = comparison()
    r["arms"][1]["identity"]["placement_verified"] = False
    assert learn.select(r) is None
    r = comparison()
    r["runs"][1]["generation"] = [[2, "stop"]] * 4
    assert learn.select(r) is None
    r["runs"][2]["tokens_per_second"] = 30
    r["arms"][2]["identity"]["baseline_stock"] = False
    assert learn.select(r) is None


@pytest.mark.parametrize("outcome", ["win", "quality_loss", "throughput_loss", "tail_loss", "unchanged_execution"])
def test_saves_trials_but_only_an_exact_local_win_serves(tmp_path, monkeypatch, outcome):
    report = comparison()
    def child(cmd, *, input, **kw):
        trial = json.loads(input)
        a = {"output_digest": "same", "execution_shape": "original", "correct": 4, "requests": 4,
             "tokens_per_second": 10, "median_ttft_ms": 10, "p95_ttft_ms": 12, "budget_applied": 0}
        b = {**a, "tokens_per_second": 11, "execution_shape": "changed", "budget_applied": 1}
        if outcome == "quality_loss":
            b["output_digest"] = "different"
        if outcome == "throughput_loss":
            b["tokens_per_second"] = 9
        if outcome == "tail_loss":
            b["p95_ttft_ms"] = 13
        if outcome == "unchanged_execution":
            b["execution_shape"] = a["execution_shape"]
        assert trial["candidate"] == 64 and trial["incumbent"] == 0
        orders = [["incumbent", "candidate"], ["candidate", "incumbent"]] * 2
        return SimpleNamespace(returncode=0, stdout=json.dumps([{"candidate": b, "incumbent": a, "order": o} for o in orders]))
    monkeypatch.setattr(learn.subprocess, "run", child)
    args = dict(gguf=Path("fixture.gguf"), state=tmp_path, provider="cpu", deadline=time.monotonic() + 10)
    result = learn.respond(report, **args)
    binding = result["selection"]["binding"]
    assert len(result["holdout"]) == 4
    rules = learn.read_rules(tmp_path, binding)
    assert rules == ({result["workload_key"]: 64} if outcome == "win" else {})
    assert result["status"] == ("qualified" if outcome == "win" else "retained_incumbent")
    assert learn.read_rules(tmp_path, {**binding, "machine": "other"}) == {}
    assert learn.respond(report, **args) == result  # the same evidence is never counted twice


def test_timeout_keeps_a_plan_without_qualifying(tmp_path):
    out = learn.respond(comparison(), gguf="none", state=tmp_path, provider="cpu", deadline=time.monotonic()-1)
    assert out["status"] == "deferred" and "budget exhausted" in out["reason"]
    assert not list(tmp_path.rglob("qualified.json"))


def test_next_report_uses_different_candidate_after_a_loss(tmp_path):
    r = comparison()
    b = learn.select(r)["binding"]
    directory = learn.rule_directory(tmp_path, b)
    directory.mkdir(parents=True)
    key = learn.shape([32, 32, 64, 64], 6)
    previous = {"workload_key": key, "incumbent_budget": 0, "candidate_budget": 64, "status": "retained_incumbent"}
    (directory / "previous.json").write_text(json.dumps(previous))
    result = learn.respond(r, gguf="none", state=tmp_path, provider="cpu", deadline=time.monotonic() - 1)
    assert result["candidate_budget"] == 32
