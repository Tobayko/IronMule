import importlib
from types import SimpleNamespace

import pytest

from ironmule import numeric_choice
from ironmule.numeric_plans import CUDA_PRE_AMPERE


def test_candidates_follow_the_measured_table():
    qwen = [c["plan"] for c in numeric_choice.candidates("mlx_lm.models.qwen3", CUDA_PRE_AMPERE)]
    assert qwen == ["native", "float16", "float32"], "recommended plans, fastest first"
    gemma = numeric_choice.candidates("mlx_lm.models.gemma3_text", CUDA_PRE_AMPERE)
    assert [c["plan"] for c in gemma] == ["native", "float32"], "float16 is refused for Gemma 3"
    assert {c["verdict"] for c in gemma} == {"recommended"}, "NEXT1-C measured both on a T4"
    gpt = numeric_choice.candidates("mlx_lm.models.gpt_oss", CUDA_PRE_AMPERE)
    assert [(c["plan"], c["verdict"]) for c in gpt] == [
        ("float16", "unqualified"), ("float32", "unqualified"), ("native", "unmeasured")]
    llama = [c["plan"] for c in numeric_choice.candidates("mlx_lm.models.llama", CUDA_PRE_AMPERE)]
    assert "float32" not in llama, "measured slower for Llama"
    assert numeric_choice.candidates("mlx_lm.models.qwen3", None) == [], "no plans on an unknown device"
    metal = numeric_choice.candidates("mlx_lm.models.gemma3_text", "metal:applegpu_g13s")
    assert [(c["plan"], c["verdict"]) for c in metal] == [("float16", "unmeasured"), ("float32", "unmeasured")], (
        "NUM2: no table on Apple Silicon, so every plan but the CUDA-only native one is gated")


def test_gate_arithmetic_passes_equal_and_refuses_worse_text_likelihood():
    reference = [3.0, 3.1, 2.9, 3.05, 3.2, 2.95, 3.0, 3.1]
    same = numeric_choice.ratio_interval(reference, [x + 0.0001 for x in reference])
    assert same["passes"] and same["ratio"] == pytest.approx(1.0001, abs=1e-6)
    worse = numeric_choice.ratio_interval(reference, [x + 0.05 for x in reference])
    assert not worse["passes"] and worse["interval"][0] > 1.04
    with pytest.raises(ValueError):
        numeric_choice.ratio_interval([1.0], [1.0])


@pytest.fixture
def fake_device(monkeypatch, tmp_path):
    monkeypatch.setattr(numeric_choice, "_store", lambda: tmp_path / "numeric.json")
    monkeypatch.setattr(numeric_choice, "device_class", lambda info: CUDA_PRE_AMPERE)
    import mlx.core as mx
    monkeypatch.setattr(mx.cuda, "is_available", lambda: True)
    tune = importlib.import_module("ironmule.tune")  # `ironmule.tune` the attribute is the function
    monkeypatch.setattr(tune, "_close_engine", lambda engine: None)
    monkeypatch.setattr(tune, "_release_device_memory", lambda: None)
    loads = []

    def load(plan):
        loads.append(plan)
        return SimpleNamespace(model=SimpleNamespace(plan=plan)), SimpleNamespace(encode=lambda text: [1] * 9000)
    return loads, load


def test_recommended_plan_is_taken_without_a_gate_and_remembered(fake_device, monkeypatch):
    loads, load = fake_device
    monkeypatch.setattr(numeric_choice, "architecture_of", lambda model: "mlx_lm.models.qwen3")
    decision = numeric_choice.choose("m", "f", "a" * 64, load=load, log=lambda _: None)
    assert decision["plan"] == "native" and "recommended" in decision["reason"]
    assert loads == [None], "only the stock model is loaded to read its architecture"
    again = numeric_choice.choose("m", "f", "a" * 64, load=load, log=lambda _: None)
    assert again == decision and loads == [None], "decided once per machine and model"


@pytest.mark.parametrize("plan_nll_shift,plan_s,expected", [
    (0.0001, 0.5, "float16"),   # inside the bound and faster: the first such plan is taken
    (0.05, 0.5, None),          # faster but worse text likelihood: refused here
    (0.0001, 0.99, None),       # inside the bound but not faster: not worth it
])
def test_unqualified_plans_are_gated_on_the_device(fake_device, monkeypatch, plan_nll_shift, plan_s, expected):
    loads, load = fake_device
    # gpt-oss has no recommended plan in the table, so every candidate is gated on the device.
    monkeypatch.setattr(numeric_choice, "architecture_of", lambda model: "mlx_lm.models.gpt_oss")
    reference = [3.0, 3.1, 2.9, 3.05, 3.2, 2.95, 3.0, 3.1]
    monkeypatch.setattr(numeric_choice, "nll_per_chunk", lambda engine, ids: (
        reference if engine.model.plan is None else [x + plan_nll_shift for x in reference]))
    monkeypatch.setattr(numeric_choice, "_speed", lambda engine, tokenizer: 1.0 if engine.model.plan is None else plan_s)
    decision = numeric_choice.choose("m", "f", "b" * 64, load=load, log=lambda _: None)
    assert decision["plan"] == expected


def test_the_gate_runs_in_a_child_and_a_failed_child_chooses_nothing(monkeypatch, tmp_path):
    import subprocess
    monkeypatch.setattr(numeric_choice, "_store", lambda: tmp_path / "numeric.json")
    calls = []

    def child(argv, **_):
        calls.append(argv)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(-6, argv)
        numeric_choice._save("f-" + "c" * 16, {"plan": "float16", "reason": "gate passed"})

    monkeypatch.setattr(subprocess, "run", child)
    failed = numeric_choice.choose_out_of_process("m", "f", "c" * 64, log=lambda _: None)
    assert failed["plan"] is None and "CalledProcessError" in failed["reason"]
    assert numeric_choice.choose_out_of_process("m", "f", "c" * 64)["plan"] == "float16"
    assert numeric_choice.choose_out_of_process("m", "f", "c" * 64)["plan"] == "float16"
    assert len(calls) == 2, "a stored decision is read without starting a child"


def test_plans_that_cannot_change_the_checkpoint_are_not_even_loaded(fake_device, monkeypatch):
    import mlx.core as mx
    loads, load = fake_device

    def float16_checkpoint(plan):
        engine, tokenizer = load(plan)
        engine.model.parameters = lambda: {"scales": mx.zeros((2,), dtype=mx.float16)}
        return engine, tokenizer

    monkeypatch.setattr(numeric_choice, "architecture_of", lambda model: "mlx_lm.models.llama")
    decision = numeric_choice.choose("m", "f", "d" * 64, load=float16_checkpoint, log=lambda _: None)
    assert decision["plan"] is None and decision["checkpoint_dtype"] == "float16"
    assert loads == [None], "float16 is this checkpoint's own type and native needs bfloat16"


def test_the_fastest_passing_plan_wins_not_the_first(fake_device, monkeypatch):
    loads, load = fake_device
    # gpt-oss has no recommended plan in the table, so every candidate is gated on the device.
    monkeypatch.setattr(numeric_choice, "architecture_of", lambda model: "mlx_lm.models.gpt_oss")
    reference = [3.0, 3.1, 2.9, 3.05, 3.2, 2.95, 3.0, 3.1]
    monkeypatch.setattr(numeric_choice, "nll_per_chunk", lambda engine, ids: reference)
    monkeypatch.setattr(numeric_choice, "_speed", lambda engine, tokenizer: {None: 1.0, "float32": 0.5,
                                                                            "native": 0.1}[engine.model.plan])
    decision = numeric_choice.choose("m", "f", "g" * 64, load=load, log=lambda _: None)
    assert decision["plan"] == "native", "float32 is gated first but native is five times faster"


def test_the_speed_check_serves_through_the_runtime_on_the_serving_knobs(monkeypatch):
    import importlib
    from ironmule.runtime import BASELINE
    service = importlib.import_module("ironmule.service")
    seen = []

    class FakeRuntime:
        def __init__(self, engine, tokenizer):
            self.engine = engine

        def encode(self, text):
            return [1, 2]

        def serve(self, requests):
            seen.append((self.engine.knobs, requests[0].max_tokens))
            return [SimpleNamespace(tokens=[7] * 64)]  # an early stop: time is scaled to 128 tokens

    monkeypatch.setattr(service, "Runtime", FakeRuntime)
    engine = SimpleNamespace(knobs=BASELINE)
    assert set(numeric_choice._speed(engine, None)) == {"short", "long"}
    knobs, max_tokens = seen[-1]
    assert len(seen) == 8 and max_tokens == 128
    assert knobs.compiled_fixed_cache and knobs.head_skip_prefill and knobs.readback_every == 4


@pytest.mark.parametrize("check,expected", [
    ({"quantized_modules": 9, "logit_rel_error": 0.0145, "top1_equal": True, "stock_s": 60.0, "dequantize_s": 0.5},
     "dequantize"),
    ({"quantized_modules": 9, "logit_rel_error": 0.9, "top1_equal": True, "stock_s": 60.0, "dequantize_s": 0.5},
     None),  # a wrong expansion
    ({"quantized_modules": 9, "logit_rel_error": 0.01, "top1_equal": False, "stock_s": 60.0, "dequantize_s": 0.5},
     None),  # another first token
    ({"quantized_modules": 9, "logit_rel_error": 0.01, "top1_equal": True, "stock_s": 1.0, "dequantize_s": 1.0},
     None),  # not faster
    ({"quantized_modules": 0}, None),                                                                  # nothing to do
    ({"quantized_modules": 9, "dense_bytes": 2**36, "memory_bytes": 2**35}, None),                    # does not fit
])
def test_a_cpu_takes_dequantize_only_after_its_bounded_check(monkeypatch, tmp_path, check, expected):
    import platform

    import mlx.core as mx
    monkeypatch.setattr(numeric_choice, "_store", lambda: tmp_path / "numeric.json")
    monkeypatch.setattr(mx.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(numeric_choice, "cpu_check", lambda model_id, load: check)
    decision = numeric_choice.choose("m", "f", "h" * 64, load=lambda plan: None, log=lambda _: None)
    assert decision["device"] == "cpu:x86_64" and decision["plan"] == expected


def test_dequantize_model_keeps_the_checkpoints_own_values():
    import mlx.core as mx
    import mlx.nn as nn
    from ironmule.numeric_plans import dequantize_model

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 64)
            self.layers = [nn.Linear(64, 64)]
            self.head = nn.Linear(64, 32, bias=False)

    model = Tiny()
    nn.quantize(model, group_size=32, bits=4)
    want = mx.dequantize(model.head.weight, model.head.scales, model.head.biases, 32, 4).astype(mx.float32)
    x = mx.random.normal((1, 64), key=mx.random.key(5)).astype(mx.bfloat16)
    before = model.head(model.layers[0](x)).astype(mx.float32)
    assert dequantize_model(model) == 3
    assert type(model.head) is nn.Linear and type(model.embed) is nn.Embedding and type(model.layers[0]) is nn.Linear
    assert mx.array_equal(model.head.weight, want), "the values are the checkpoint's own"
    after = model.head(model.layers[0](x.astype(mx.float32)))
    assert float(mx.max(mx.abs(after - before)) / mx.max(mx.abs(before))) < 2e-2


def test_each_plan_is_measured_in_its_own_child(monkeypatch, tmp_path):
    import json
    import subprocess

    import mlx.core as mx
    monkeypatch.setattr(numeric_choice, "_store", lambda: tmp_path / "numeric.json")
    monkeypatch.setattr(mx.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    reference = [3.0, 3.1, 2.9, 3.05, 3.2, 2.95]
    children = []

    def run(argv, **_):
        plan = argv[-2] or None
        children.append(plan)
        if plan == "float32":
            return SimpleNamespace(returncode=-9, stdout="")  # killed: that plan is simply not taken
        row = {"nll": reference, "speed_s": 1.0 if plan is None else 0.8}
        if plan is None:
            row.update(architecture="mlx_lm.models.qwen3", dtype="bfloat16")
        return SimpleNamespace(returncode=0, stdout="noise\n@@PLAN " + json.dumps(row) + "\n")

    monkeypatch.setattr(subprocess, "run", run)
    decision = numeric_choice.choose("m", "f", "k" * 64, log=lambda _: None)
    assert children == [None, "float16", "float32"], "one process per load"
    assert decision["plan"] == "float16"
    assert decision["tried"][1]["outcome"] == "unavailable: RuntimeError"


@pytest.mark.parametrize("short,long,expected", [
    (0.90, 0.85, True),    # faster on both
    (0.99, 0.80, True),    # not slower on short, clearly faster on long
    (1.01, 0.80, False),   # slower on short answers: refused (Gemma 3 4B float32, NUM6)
    (0.97, 0.96, False),   # nowhere clearly faster
])
def test_a_plan_is_faster_only_if_slower_nowhere(short, long, expected):
    ratios = numeric_choice._speed_ratios({"short": 1.0, "long": 2.0}, {"short": short, "long": 2 * long})
    assert numeric_choice._faster(ratios) is expected


def test_newer_cuda_gpus_are_gated_without_a_table():
    plans = numeric_choice.candidates("mlx_lm.models.gemma3_text", "cuda:NVIDIA L4")
    assert [(c["plan"], c["verdict"]) for c in plans] == [("float16", "unmeasured"), ("float32", "unmeasured")], (
        "float16 is not refused off the measured device class, and native stays below compute capability 8")
