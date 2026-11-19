"""Offline leave-one-repeat-out native prefill selection. Timings below are synthetic fixtures."""
import pytest

from training.prefill_selection import evaluate
from lab.portable_gpu.prefill_http import _scope
from ironmule_product.compare import Unverified


def fixture():
    report = {"schema": "ironmule.prefill-http.v1", "status": "measured", "stage": "finished",
              "protocol": {"repeats": 3, "arms": ["stock", "batch"], "concurrency": [8], "requests": 8,
                           "max_tokens": 32, "context": 2048, "seed": 912},
              "deployments": [], "measurements": []}
    for rep in range(3):
        seed = 912 + rep * 1009 + 8
        for name in (("stock", "batch") if rep % 2 == 0 else ("batch", "stock")):
            identity = {"version": "test", "provider": "cuda", "device_name": "test device",
                        "gguf_sha256": "1" * 64, "tokenizer_sha256": "2" * 64, "implementation_sha256": "3" * 64,
                        "placement_verified": True, "autonomous": True}
            report["deployments"].append({"deployment": name, "repeat": rep, "identity": identity,
                                          "prompt_ids_verified": True, "prompt_token_digest": "5" * 64})
            baseline = name == "stock"
            duration = 1.25 if baseline else 1.125
            quality = {f"{seed}:{cat}-{i}": True for cat in ("arithmetic", "extraction", "facts") for i in range(6)}
            report["measurements"].append({
                "deployment": name, "repeat": rep, "concurrency": 8, "requests": 8,
                "request_id_prefix": f"qwen3-912-{rep}-8", "performance_prompt_tokens": [32, 350] * 4,
                "prompt_ids_sha256": [f"{rep * 30 + i:064x}" for i in range(26)],
                "generation": [[10, "length"]] * 8, "tokens_per_second": 80 / duration, "wall_s": duration,
                "completion_tokens": 80, "ttft_p95_ms": 10 if baseline else 9,
                "quality": {"answers": quality, "total": 18, "passed": 18},
                "profile": {"outputs_sha256": "a" * 64, "samples": 8, "context": 2048, "shared": False,
                            "sequences": 8, "schedule": "fixed",
                            "scope": {"arm_sha256": ("a" if baseline else "b") * 64,
                                      "settings_device_sha256": "c" * 64, "native_runtime_sha256": "d" * 64, "implementation_sha256": "3" * 64},
                            "batch_delta": {"calls": 0 if baseline else 1, "rows": 0 if baseline else 64,
                                            "prompts": 0 if baseline else 2}}})
    return report


def test_holdout_never_selects_itself_and_losing_round_prevents_recommendation():
    report = fixture()
    group = evaluate(report)["groups"][0]
    assert group["decision"] == "candidate_for_new_local_holdout"
    assert [x["choice"] for x in group["folds"]] == ["grouped", "split", "grouped"]
    trial = report["measurements"][-1]
    trial["ttft_p95_ms"] = 12
    trial["profile"]["outputs_sha256"] = "7" * 64
    key = next(iter(trial["quality"]["answers"]))
    trial["quality"]["answers"][key] = False
    trial["quality"]["passed"] -= 1
    rejected = evaluate(report)["groups"][0]
    assert rejected["decision"] == "retain_split"
    assert rejected["folds"][2]["choice"] == "grouped"  # Retained its blind prediction.
    assert set(rejected["folds"][2]["test_rejections"]) == {"p95_regression", "lost_correct_answer", "different_output"}
    assert [f["choice"] for f in rejected["folds"][:2]] == ["split", "split"]


def test_dispatch_model_scope_and_disjoint_traces_are_required():
    report = fixture()
    for row in report["measurements"]:
        del row["profile"]["scope"]
    assert evaluate(report)["groups"][0]["decision"] == "candidate_requires_bound_remeasurement"
    report["deployments"][-1]["identity"]["implementation_sha256"] = "6" * 64
    assert evaluate(report)["excluded"][0]["reason"] == "different_deployment_or_prompt_proof"
    report["deployments"][-1]["identity"]["implementation_sha256"] = "3" * 64
    for row in report["measurements"]:
        if row["repeat"] == 2:
            row["prompt_ids_sha256"] = report["measurements"][0]["prompt_ids_sha256"]
    assert evaluate(report)["groups"][0]["decision"] == "insufficient_independent_rounds"
    report = fixture()
    report["measurements"][-1]["profile"]["scope"]["settings_device_sha256"] = "4" * 64
    assert evaluate(report)["excluded"][0]["reason"] == "different_worker_scope"
    report = fixture()
    report["measurements"][-1]["profile"]["batch_delta"] = {"calls": 0, "rows": 0, "prompts": 0}
    assert "no_verified_grouped_work" in evaluate(report)["groups"][0]["folds"][2]["test_rejections"]
    report["schema"] = "ironmule.compare.v1"
    with pytest.raises(ValueError, match="requires_native"):
        evaluate(report)


def test_producer_records_effective_scope_and_refuses_mismatched_gpu():
    binding = {"provider": "cuda", "host_sha256": "h", "implementation": "i", "machine": "m",
               "cuda_uuid": "af0102030405060708090a0b0c0d0e0f", "gguf_sha256": "model", "seqs": 2, "study_native_runtime_sha256": "d" * 64}
    server = {"device": "GPU-af010203-0405-0607-0809-0a0b0c0d0e0f", "gguf_sha256": "model"}
    scope = _scope(binding, server)
    other = _scope({**binding, "machine": "with grouped toggle"}, server)
    assert other["arm_sha256"] != scope["arm_sha256"]
    assert other["settings_device_sha256"] == scope["settings_device_sha256"]
    assert _scope({**binding, "seqs": 8}, server)["settings_device_sha256"] != scope["settings_device_sha256"]
    assert _scope({**binding, "study_native_runtime_sha256": "7" * 64}, server)["settings_device_sha256"] != scope["settings_device_sha256"]
    with pytest.raises(Unverified):
        _scope(binding, {**server, "device": "GPU-bf010203-0405-0607-0809-0a0b0c0d0e0f"})


def test_measurement_only_binding_reads_the_loaded_worker_objects(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from ironmule import compare_learning, ggml_binary, ggml_engine
    from ironmule_product import admission_learning
    from ironmule_product.ggml_bridge import GgmlBridge
    import ironmule_devices

    bridge = object.__new__(GgmlBridge)
    bridge._gate = {"admission_state_root": tmp_path}
    bridge._engine = SimpleNamespace(L=object())
    bridge._gguf, bridge._gguf_sha256 = "study.gguf", "b" * 64
    seen, loaded = [], []
    monkeypatch.setattr(compare_learning, "engine_binding", lambda *_: {"gguf_sha256": "b" * 64})
    monkeypatch.setattr(admission_learning, "AdmissionLearning", lambda *args, **kw: seen.append(args[3]))
    monkeypatch.setattr(ggml_binary, "identity", lambda: loaded.append(1) or {"sha256": {"lib": "a" * 64}})
    monkeypatch.setattr(ggml_engine, "ggml_devices", lambda L: ("gpu",))
    monkeypatch.setattr(ironmule_devices, "runtime_binding", lambda devices: {"devices": devices})
    monkeypatch.delenv("IRONMULE_PREFILL_STUDY", raising=False)
    bridge._bind_admission_learning()
    assert not loaded and bridge._experiment_binding == seen[-1]
    monkeypatch.setenv("IRONMULE_PREFILL_STUDY", "1")
    bridge._bind_admission_learning()
    assert loaded == [1]
    assert "study_native_runtime_sha256" not in seen[-1]
    assert len(bridge._experiment_binding["study_native_runtime_sha256"]) == 64



def test_unavailable_grouped_arm_keeps_incumbent_measurements_and_layout_reason():
    report = fixture()
    report["status"] = "partial"
    report["errors"] = [{"deployment": "batch", "repeat": 0,
                         "reason": "actual shared-pool placement differs from requested arm"}]
    report["measurements"] = [row for row in report["measurements"] if row["deployment"] == "stock"]
    report["deployments"] = [row for row in report["deployments"] if row["deployment"] == "stock"]
    out = evaluate(report)
    assert out["status"] == "insufficient_evidence" and not out["groups"]
    assert out["reason"] == "no_verified_split_grouped_pairs"
    assert out["native_arms"]["stock"] == {"timed_cells": 3, "verified_repeats": [0, 1, 2]}
    assert out["native_arms"]["batch"] == {"timed_cells": 0, "verified_repeats": []}
    assert out["arm_failures"][0]["reason"] == "requested_kv_layout_unavailable"
    report["status"] = "incomplete"
    incomplete = evaluate(report)
    assert not incomplete["groups"] and incomplete["status"] == "insufficient_evidence"
    assert [r["correct"] for r in incomplete["verified_cells"]] == [18, 18, 18]
