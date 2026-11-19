"""Synthetic native records test the scoped KG4 judge, without executing CUDA."""
import hashlib

import pytest

from ironmule.factual_gate import cases
from lab.kg4 import scoped_native as study


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def hits(*buckets):
    counts = [0] * 64
    for bucket in buckets:
        counts[bucket - 1] += 1
    return {"kg4": counts, "cublas": [0]}


def reports():
    output = []
    for name in ("stock", "scoped", "stock-closing"):
        scoped = name == "scoped"
        seed = 17
        facts = {c.key: True for c in cases(seed + 65537)}
        record = {"schema": study.SCHEMA, "arm": name, "status": "measured", "seed": seed,
                  "facts_seed": seed + 65537, "facts": facts.copy(), "facts_low": facts.copy(),
                  "facts_repeat": facts.copy(), "facts_low_repeat": facts.copy(),
                  "gguf_sha256": study.MODEL, "graph_capture": False, "context": 768, "sequences": 32,
                  "native": {**{n: sha(n) for n in ("llama_backend_init", "ggml_init", "ggml_backend_reg_count",
                                                    "backend/CPU")}, "backend/CUDA": study.BACKEND},
                  "gpu": {"name": "NVIDIA L40S", "driver": "test", "cc": "89", "uuid_sha256": sha("device")},
                  "engine_sha256": sha("engine"), "harness_sha256": sha("harness"), "offloaded_layers": 65,
                  "cuda_model_mib": 18000, "plugin_sha256": study.PLUGIN if scoped else None,
                  "scope_source_sha256": sha("scope implementation"),
                  "nll_low": {str(b): [1.0] * b for b in (6, 7, 8)},
                  "nll_low_hits": {str(b): hits(b) if scoped else hits() for b in (6, 7, 8)},
                  "nll": [1.0] * 16, "nll_b32": [1.0] * 16, "facts_low_hits": hits(8) if scoped else hits(),
                  "disable_hits": hits(), "masks_after": [0, 0],
                  "scope": {"mode": "low-width8", "parked": True, "low_runs": 2, "wide_runs": 2,
                            "q4_buckets": [6, 7, 8]}, "waves": []}
        for rep in range(2):
            for width in (8, 32):
                elapsed = .9 if scoped and width == 8 else 1
                record["waves"].append({
                    "repeat": rep, "width": width, "work": [[16, "length"] for _ in range(width)],
                    "prompt_ids_sha256": sha(f"input {rep} {width}"), "output_ids_sha256": sha(f"output {rep} {width}"),
                    "output_tokens": width * 16, "elapsed_s": elapsed, "output_tok_s": width * 16 / elapsed,
                    "ttft_ms": [20.] * width, "ttft_p95_ms": 20., "hits": hits(6, 7, 8) if scoped and width == 8 else hits()})
        output.append(record)
    return output


def rejected(records, reason):
    result = study.qualify(*records)
    assert not result["eligible"]
    assert reason in result["reasons"]


def test_scoped_native_qualifies_only_the_narrow_action():
    stock, scoped, close = reports()
    # The numerical path can change low-width tokens if its independent quality and NLL gates hold.
    scoped["waves"][0]["output_ids_sha256"] = sha("another narrow output")
    result = study.qualify(stock, scoped, close)
    assert result["eligible"], result["reasons"]
    assert result["metrics"]["8"]["throughput_ratio"] == pytest.approx(1 / .9)
    assert result["metrics"]["32"]["throughput_ratio"] == 1


@pytest.mark.parametrize(("arm", "wave", "buckets", "reason"), [
    (1, 0, (), "dispatch_w8"), (1, 0, (5, 8), "dispatch_w8"), (1, 0, (8, 40), "dispatch_w8"),
    (1, 1, (8,), "dispatch_w32"), (2, 2, (7,), "dispatch_w8")])
def test_real_dispatch_must_be_exclusively_q4_b6_to_b8(arm, wave, buckets, reason):
    records = reports()
    records[arm]["waves"][wave]["hits"] = hits(*buckets)
    rejected(records, reason)


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("facts_low_hits", hits(40), "factual_dispatch"), ("disable_hits", hits(8), "scope_not_parked"),
    ("masks_after", [1, 0], "scope_not_parked"), ("graph_capture", True, "different_runtime"),
    ("offloaded_layers", 64, "different_device_or_offload"), ("plugin_sha256", sha("other"), "wrong_plugin"),
    ("gguf_sha256", sha("other"), "different_native_or_model"), ("harness_sha256", sha("other"), "different_runtime"),
    ("nll_b32", [1.1] * 16, "nll_b32"), ("nll", [float("nan")] * 16, "invalid_nll"),
])
def test_rejects_invalid_proof_and_regressions(field, value, reason):
    records = reports()
    records[1][field] = value
    rejected(records, reason)


def test_a_fresh_timed_prompt_set_is_required_for_each_repeat_and_width():
    records = reports()
    for record in records:
        record["waves"][2]["prompt_ids_sha256"] = record["waves"][0]["prompt_ids_sha256"]
    rejected(records, "repeated_timed_inputs")
    records = reports()
    records[2]["waves"][0]["prompt_ids_sha256"] = sha("different closing workload")
    rejected(records, "different_inputs")


def test_a_win_over_opening_stock_is_insufficient_when_closing_stock_is_faster():
    records = reports()
    closing = records[2]["waves"][0]
    closing.update(elapsed_s=.85, output_tok_s=closing["output_tokens"] / .85)
    rejected(records, "throughput_w8")
    records = reports()
    records[2]["waves"][1].update(ttft_ms=[18.] * 32, ttft_p95_ms=18.)
    rejected(records, "p95_w32")


def test_quality_checks_both_real_widths_and_the_exact_wide_fallback():
    for field in ("facts", "facts_low"):
        records = reports()
        scores = records[1][field]
        scores[next(iter(scores))] = False
        rejected(records, field)
    records = reports()
    records[1]["waves"][1]["output_ids_sha256"] = sha("unexpected wide output")
    rejected(records, "wide_output_changed")


def test_low_width_nll_must_use_the_named_bucket_and_preserve_scores():
    records = reports()
    records[1]["nll_low"]["7"] = [1.1] * 7
    assert not study.qualify(*records)["eligible"]
    records = reports()
    records[1]["nll_low_hits"]["7"] = hits(8)
    assert not study.qualify(*records)["eligible"]
    records = reports()
    records[2]["scope_source_sha256"] = sha("changed scope")
    assert not study.qualify(*records)["eligible"]
