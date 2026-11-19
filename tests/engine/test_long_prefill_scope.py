"""Regressions for the strictly receipt-scoped long prefill arm."""

from copy import deepcopy
from types import SimpleNamespace as NS

from ironmule import long_prefill_study as study
from ironmule.ggml_engine import GgmlEngine, Job


def test_only_locally_recorded_cold_dense_rectangles_are_eligible():
    e = object.__new__(GgmlEngine)
    e.seq_ctx = 2048
    e.prefill_batch_long = True

    def groups(lengths, kept=None, slots=None):
        rows = [
            (s, Job([17] * n, 1), k)
            for s, n, k in zip(
                slots or range(len(lengths)), lengths, kept or [0] * len(lengths)
            )
        ]
        return [len(g) for g in e._prefill_groups(rows)]

    assert groups([444] * 4) == [1] * 4
    e.prefill_long_shapes = {444: (2, 4)}
    assert groups([444] * 7) == [4, 2, 1]
    assert groups([33] * 8) == [1] * 8
    assert groups([445] * 2, [1, 1]) == [1, 1]
    assert groups([444] * 4, slots=[0, 2, 3, 4]) == [1, 2, 1]
    e.prefill_long_shapes = {1024: (2,)}
    assert groups([444] * 4) == [1] * 4
    assert groups([1024] * 4) == [2, 2]


def example():
    gate = {"protocol": "ironmule.long-prefill.v1", "matches": True, "shapes": {}}
    for w in (2, 4):
        stock = [{"calls": 0, "rows": 0, "prompts": 0}] * (8 // w)
        cases = []
        for i in range(8):
            output = {"tokens": [1], "stop": "length", "correct": True}
            cases.append(
                {
                    "category": ("arithmetic", "facts", "extraction")[i % 3],
                    "reference": output,
                    "repeat": output,
                    "candidate": output,
                    "prompt_sha256": f"{i:064x}",
                    "prompt_tokens": 444,
                }
            )
        gate["shapes"][f"{w}x444"] = {
            "matches": True,
            "reasons": [],
            "cases": cases,
            "calls": 8 // w,
            "witnesses": {
                "reference": stock,
                "repeat": stock,
                "candidate": [{"calls": 1, "prompts": w, "rows": 444 * w}] * (8 // w),
            },
        }
    waves = []
    for rep in range(3):
        for width in (8, 32):
            for mode, sec in (("off", 2), ("long-444", 1)):
                waves.append(
                    dict(
                        repeat=rep,
                        width=width,
                        mode=mode,
                        seconds=sec,
                        output_tokens=12,
                        prompts_sha256="4" * 64,
                        outputs_sha256="5" * 64,
                        hits={"2": 0, "4": int(mode != "off")},
                    )
                )
    return {
        "schema": study.SCHEMA,
        "status": "measured",
        "provider": "cuda",
        "seqs": 32,
        "seq_ctx": 2048,
        "ubatch": 512,
        "settings": {"flash_attn": True, "kv_unified": False},
        "offloaded_layers": [65, 65],
        "protocol_sha256": study.protocol_hash(),
        "gguf_sha256": "0" * 64,
        "gpu_uuid_sha256": study.uuid_hash("a" * 32),
        "native_runtime_sha256": "2" * 64,
        "implementation_sha256": "3" * 64,
        "waves": waves,
        "long_gate": gate,
        "nll": {"chunks": 16, "interval": [1.0, 1.002]},
        "factual": {"passes": True},
    }


def test_receipt_requires_measured_shape_hits_width_quality_speed_and_device():
    r = example()
    assert study.qualified_receipt(r, cuda_uuid="a" * 32)
    assert not study.qualified_receipt(r, cuda_uuid="b" * 32)
    assert not study.qualified_receipt(dict(r, provider="cpu"))
    assert not study.qualified_receipt(dict(r, ubatch=2048))
    assert not study.qualified_receipt(dict(r, factual={"passes": False}))
    for change in (
        lambda x: x["waves"][1].update(hits={"2": 0, "4": 0}),
        lambda x: x["long_gate"]["shapes"].update({"8x33": {}}),
        lambda x: x["long_gate"]["shapes"]["2x444"]["cases"][1].update(
            candidate={"tokens": [3]}
        ),
    ):
        x = deepcopy(r)
        change(x)
        assert not study.qualified_receipt(x)
    x = deepcopy(r)
    for s in x["waves"]:
        s["seconds"] = 2
    assert not study.qualified_receipt(x)


def test_no_worker_receipt_keeps_incumbent(monkeypatch):
    monkeypatch.delenv("IRONMULE_GGML_PREFILL_RECEIPT", raising=False)
    engine = NS(prefill_batch=False, prefill_batch_long=True)
    assert not study.enable_worker(
        engine, None, "a" * 64, {"passes": True, "streams_serving": {"settings": {}}}
    )
    assert not engine.prefill_batch


def test_only_time_witnessed_receipt_shapes_become_eligible():
    r = example()
    assert study.measured_shapes(r["long_gate"], r["waves"]) == {444: (4,)}
    r["waves"][1]["hits"]["2"] = 1
    assert study.measured_shapes(r["long_gate"], r["waves"]) == {444: (2, 4)}


def test_all_and_long_and_default_results_have_distinct_cache_keys(
    tmp_path, monkeypatch
):
    from ironmule import ggml_engine as ge

    model = tmp_path / "sample.gguf"
    model.write_bytes(b"test")
    monkeypatch.delenv("IRONMULE_KG5", raising=False)
    monkeypatch.setattr(ge, "machine_key", lambda: "this machine")
    monkeypatch.setattr(ge, "_kernel_digest", lambda: "")
    keys = set()
    for mode in ("0", "1", "long-444"):
        monkeypatch.setenv("IRONMULE_GGML_PREFILL_BATCH", mode)
        keys.add(ge.gate_key(model, model))
    assert len(keys) == 3
