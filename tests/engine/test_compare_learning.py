"""Verified rival gaps can direct a native holdout. Only measured local wins change a bound budget."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ironmule import compare_learning as learn


def comparison():
    b = {"gguf_sha256": "1" * 64, "machine": "fixture", "host_sha256": "2" * 64, "seq_ctx": 512, "prompt_budget": 0,
         "implementation": learn.implementation()}
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


def test_current_gap_selected_even_when_old_engine_lost_by_more(tmp_path):
    from copy import deepcopy

    report = comparison()
    new = deepcopy(report["arms"] + report["runs"])
    for row in new:
        row["repeat"] = 1
    report["arms"].extend(new[:3])
    report["runs"].extend(new[3:])
    report["arms"][0]["identity"]["ggml"]["implementation"] = "old"
    report["runs"][0]["tokens_per_second"] = 3
    assert learn.select(report)["binding"]["implementation"] == "old"
    out = learn.respond(report, gguf="none", state=tmp_path, provider="cpu", deadline=time.monotonic() - 1)
    assert out["selection"]["binding"]["implementation"] == learn.implementation()
    assert out["selection"]["throughput_ratio"] == 10/15


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


def test_shared_pool_probe_recreates_derived_prefix_setting(monkeypatch):
    from ironmule import ggml_engine

    def open_instrumented(model, *, threads, seqs, seq_ctx, speculate, shared, cap, max_rows, kv_unified):
        assert shared and kv_unified and cap == 8
        raise RuntimeError("reached native constructor")

    monkeypatch.setattr(ggml_engine, "file_sha256", lambda path: "model")
    monkeypatch.setattr(ggml_engine, "GgmlEngine", open_instrumented)
    for key in ("IRONMULE_GGML_SEQS", "IRONMULE_GGML_SEQ_CTX", "IRONMULE_KG_LIB"):
        monkeypatch.setenv(key, "")
    binding = {"gguf_sha256": "model", "threads": 2, "seqs": 4, "seq_ctx": 512, "speculate": True, "shared": True,
               "cap": 8, "max_rows": 16, "settings": {"shared_prefix": True, "kv_unified": True}}
    with pytest.raises(RuntimeError, match="reached native constructor"):
        learn.trial({"binding": binding, "gguf": "test.gguf", "provider": "cpu", "device": "cpu:fixture"})


def test_cuda_holdout_reconstructs_physical_mask_before_native_initialization(monkeypatch):
    from ironmule_product import compare_holdout as launcher, compare_servers

    uid = "GPU-e66e3a63-7a14-4ff0-5f1c-3d6ee532c695"
    binding = {"seq_ctx": 2048, "seqs": 64, "kg_sha256": None}
    arm = {"device": uid, "cuda_visible_devices": uid, "provider": "cuda", "placement_verified": True, "ggml": binding}
    report = {"device": uid, "arms": [{"engine": "ironmule", "identity": arm}]}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("IRONMULE_KG_LIB", "other.so")
    selected = launcher.verified_device(report, binding, "cuda")
    def physical_device(env, engine):
        assert engine == "ironmule" and env["CUDA_VISIBLE_DEVICES"] == "0,1"
        env["CUDA_VISIBLE_DEVICES"] = uid
        return uid, "NVIDIA L40S"
    monkeypatch.setattr(compare_servers, "_device", physical_device)
    env = launcher.worker_environment(selected, binding, "cuda")
    assert env["CUDA_VISIBLE_DEVICES"] == uid and env["IRONMULE_KG_LIB"] == ""
    assert (env["IRONMULE_GGML_SEQS"], env["IRONMULE_GGML_SEQ_CTX"]) == ("64", "2048")
    monkeypatch.setattr(compare_servers, "_device", lambda env, engine: ("GPU-other", "L40S"))
    with pytest.raises(ValueError, match="device differs"):
        launcher.worker_environment(selected, binding, "cuda")
    arm["cuda_visible_devices"] = "0"
    with pytest.raises(ValueError, match="verified CUDA selector"):
        launcher.verified_device(report, binding, "cuda")


def test_binding_difference_lists_only_known_field_names_and_never_promotes(monkeypatch, tmp_path):
    actual = {"machine": "current", "pilot": "present", "settings": {}}
    old = {"machine": "other", "pilot": None, "settings": {}, "/private/path": "extra"}
    difference = learn.BindingMismatch(actual, old)
    assert difference.fields == ["machine", "pilot", "unexpected_field"]
    monkeypatch.setattr(learn.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=3, stdout=json.dumps({"binding_mismatch": difference.fields})))
    result = learn.respond(comparison(), gguf="model", state=tmp_path, provider="cpu", deadline=time.monotonic() + 10)
    assert result["status"] == "deferred" and result["binding_mismatch"] == difference.fields
    assert "/private/path" not in json.dumps(result) and not list(tmp_path.rglob("qualified.json"))


def test_retry_uses_separate_output_and_refuses_different_model_or_service_state(monkeypatch, tmp_path):
    from ironmule_product.compare_holdout import replay

    report = comparison()
    report["status"] = "complete"
    model, state = tmp_path / "m.gguf", tmp_path / "worker"
    arm = report["arms"][0]["identity"]
    arm.update(provider="cpu", argv=["serve", "--state-dir", str(state), "--gguf", str(model)])
    source, output = tmp_path / "raw.json", tmp_path / "holdout.json"
    original = json.dumps(report)
    source.write_text(original)
    base = ["--report", str(source), "--gguf", str(model), "--output", str(output), "--state"]
    with pytest.raises(SystemExit):
        replay([*base, str(tmp_path / "other-state")])
    assert not output.exists()
    monkeypatch.setattr(learn, "respond", lambda *args, **kw: {"status": "retained_incumbent"})
    for key in ("IRONMULE_GGML_SEQ_CTX", "IRONMULE_KG_LIB"):
        monkeypatch.setenv(key, "")
    replay([*base, str(state)])
    assert json.loads(output.read_text())["status"] == "retained_incumbent"
    assert source.read_text() == original
    with pytest.raises(SystemExit):
        replay([*base, str(state)])
