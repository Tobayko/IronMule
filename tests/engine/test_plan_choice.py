"""AG4: the agent's own choice of plan and engine (ironmule/plan_choice.py), by the rule BACKLOG AG4 fixed."""

from __future__ import annotations

import pytest

from ironmule import plan_choice as pc


def _speed(base_step=0.052, plan_steps=None, base_first=0.5, plan_first=0.5):
    plan_steps = plan_steps or {"1": 0.053, "8": 0.17}
    row = {"base_first_s": base_first, "base_step_s": base_step, "plan_first_s": plan_first, "plan_step_s": plan_steps}
    return {"short": row, "long": dict(row)}


def _measured(passes=True, **speed):
    return {"eligible": True, "install": {}, "speed": _speed(**speed),
            "gate": {"passes": passes, "ratio": 1.001, "interval": [0.998, 1.004]}}


def test_the_stacked_engine_serves_when_it_passes_every_part_of_the_rule():
    decision = pc.decide(_measured())
    assert decision["stacked"] and decision["tokens_per_second"]["stacked_8"] == pytest.approx(8 / 0.17)
    assert "2.45x" in decision["reason"]


@pytest.mark.parametrize("measured, why", [
    ({"eligible": False, "reason": "Qwen 3 only"}, "no stacked engine"),
    (_measured(passes=False), "quality gate"),
    (_measured(plan_steps={"1": 0.053, "8": 0.5}), "no more tokens"),
    (_measured(plan_steps={"1": 0.06, "8": 0.17}), "one request alone"),  # 0.87x of the base path
    (_measured(plan_first=0.6), "first token"),                          # 1.2x later
])
def test_the_base_path_serves_otherwise(measured, why):
    decision = pc.decide(measured)
    assert not decision["stacked"] and why in decision["reason"]


def test_the_choice_is_made_once_and_shown_with_its_reasons(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "_store", lambda: tmp_path / "plan_choice.json")
    calls = []

    def numeric(model_id, fingerprint, identity, log):
        calls.append("numeric")
        return {"plan": "native", "reason": "gate passed"}

    def measure(model_id, compute_dtype, base_only=False):
        calls.append(("measure", compute_dtype))
        return _measured()

    setup = pc.choose_setup("m", "fp", "a" * 64, log=lambda _: None, numeric=numeric, measure=measure)
    assert setup["engine"] == "dense" and setup["compute_dtype"] == "native" and "tokens per second" in setup["reason"]
    assert setup["stacked"]["gate"]["passes"] and setup["numeric"]["reason"] == "gate passed"
    again = pc.choose_setup("m", "fp", "a" * 64, log=lambda _: None, numeric=numeric, measure=measure)
    assert again["engine"] == "dense" and calls == ["numeric", ("measure", "native")]  # stored, not re-measured


def test_a_float_plan_is_compared_against_not_stacked_on(tmp_path, monkeypatch):
    """AG4-Q run 1: NUM1 chose float16 for Qwen 3 bf16, and the dense plan refused float16 weights. The stacked
    engine runs on the checkpoint's bf16 instead and must beat the float16 path's own speed."""
    monkeypatch.setattr(pc, "_store", lambda: tmp_path / "plan_choice.json")
    calls = []

    def measure(model_id, compute_dtype, base_only=False):
        calls.append((compute_dtype, base_only))
        if base_only:  # float16 alone: 3.2x faster than bf16 emulation
            row = {"base_first_s": 0.2, "base_step_s": 0.016}
            return {"short": row, "long": dict(row)}
        return _measured(base_step=0.052, plan_steps={"1": 0.05, "8": 0.04}, base_first=0.5, plan_first=0.21)

    setup = pc.choose_setup("m", "fp", "b" * 64, log=lambda _: None,
                            numeric=lambda *a, **k: {"plan": "float16", "reason": "table"}, measure=measure)
    assert calls == [(None, False), ("float16", True)]
    row = setup["stacked"]["speed"]["short"]
    assert row["base_step_s"] == 0.016 and row["stacked_base_step_s"] == 0.052
    # 8 / 0.04 = 200 tokens/s against float16's 62.5, but alone 20 against 62.5: quiet traffic would regress
    assert setup["engine"] == "reference" and setup["compute_dtype"] == "float16"
    assert "one request alone" in setup["reason"]


mx = pytest.importorskip("mlx.core")


def test_measure_stacked_runs_on_a_cpu_stand_in():
    """Every number the rule reads, on a tiny random 4-bit Gemma 3 with FA2's semantics in plain MLX."""
    from ironmule import dense_gemma

    from tests.engine.test_dense_gemma import _Tok, tiny_model

    model = tiny_model()

    def make_plan(model_, info):
        plan = dense_gemma.GemmaDensePlan(model_, slots=4, capacity=96, attention=dense_gemma.reference_attention,
                                          native=False)
        return plan, dense_gemma.probe(plan)

    lines = []
    measured = pc.measure_stacked(model, _Tok(), None, chunks=4, size=16, steps=3, long_prompt="x" * 40,
                                  make_plan=make_plan, log=lines.append)
    assert measured["eligible"] and set(measured["gate"]) >= {"ratio", "interval", "passes"}
    assert [line.split(": ")[2].split(" (")[0] for line in lines] == [
        "base path timed", "gate reference on the base path, 4 chunks", "stacked plan installed",
        "gate on the stacked plan", "stacked steps timed at every width, short prompt",
        "stacked steps timed at every width, long prompt"]
    assert set(measured["speed"]) == {"short", "long"}
    assert set(measured["speed"]["short"]["plan_step_s"]) == {"1", "2", "3", "4"}, "AG5: every width is timed"
    assert "stacked" in pc.decide(measured)


def test_a_model_without_a_stacked_engine_is_measured_as_ineligible(monkeypatch):
    monkeypatch.setattr(pc, "_base_speed", lambda model, ids, steps, samples=None: (0.1, 0.05))

    def refuse(model, info):
        raise ValueError("the dense plan serves Qwen 3 checkpoints only")

    tok = type("T", (), {"encode": lambda self, text: [1, 2, 3]})()
    assert pc.measure_stacked(object(), tok, None, make_plan=refuse) == {
        "eligible": False, "reason": "the dense plan serves Qwen 3 checkpoints only"}


def test_a_model_no_stacked_plan_serves_is_refused_before_anything_is_timed(monkeypatch):
    """For a model no plan here serves, the agent refuses the stacked engine without timing the base path: on a
    slow CPU the base path's 32 timed steps alone can take minutes."""
    timed = []
    monkeypatch.setattr(pc, "_base_speed", lambda model, ids, steps, samples=None: timed.append(1) or (0.1, 0.05))
    monkeypatch.setattr(pc, "on_cpu", lambda: True)  # CPU6: on a CPU the CPU plan's own refusal, nothing timed
    out = pc.measure_stacked(object(), None, None)
    assert out["eligible"] is False and "CPU stacked plan serves llama" in out["reason"] and timed == []
    monkeypatch.setattr(pc, "on_cpu", lambda: False)  # elsewhere without a CUDA GPU below 8, as before
    out = pc.measure_stacked(object(), None, None)
    assert out["eligible"] is False and "CUDA GPUs below compute capability 8" in out["reason"] and timed == []
    t4 = {"compute_capability_major": 7, "device_name": "Tesla T4"}
    reason = pc.stacked_refusal(type("M", (), {"args": None})(), t4)
    assert reason == "the dense plan serves Qwen 3 checkpoints only" and timed == []


def test_a_stacked_engine_that_wins_only_under_load_serves_beside_the_single_path_where_both_fit():
    """AG4 step 2: AG4-Q run 2's numbers (dense 5x at width 8, 0.82x alone); with room for two copies the agent
    serves both and routes each request, without room it keeps the single path."""
    measured = _measured(base_step=0.0173, plan_steps={"1": 0.0212, "8": 0.0275})
    fits = pc.decide({**measured, "memory": {"two_paths_fit": True}})
    assert not fits["stacked"] and fits["route"] and "routes each request" in fits["reason"]
    tight = pc.decide({**measured, "memory": {"two_paths_fit": False}})
    assert not tight["stacked"] and not tight["route"]


def test_the_product_serves_the_float16_plan_nums1_can_choose():
    """AG4-R's smoke on a T4: NUM1 chose `float16`, which the product's worker did not accept. It does now, in the
    client, the serve flag and the worker's own arguments."""
    import inspect

    from ironmule_product import worker
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.types import ModelSpec

    client = MLXWorkerClient(ModelSpec("m", "r", "/tmp/m", 1), compute_dtype="float16")
    assert client.compute_dtype == "float16"
    source = inspect.getsource(worker.main)
    assert '"float16"' in source and "mx.float16" in source
    with pytest.raises(ValueError):
        MLXWorkerClient(ModelSpec("m", "r", "/tmp/m", 1), compute_dtype="int8")


def test_the_setup_lists_every_lever_the_agent_knows_built_or_not(tmp_path, monkeypatch):
    """HQ1: the levers read from HyperQwen are known to the agent and shown as not built yet; the built ones say
    what happened to them on this machine."""
    monkeypatch.setattr(pc, "_store", lambda: tmp_path / "plan_choice.json")
    measured = _measured(base_step=0.0173, plan_steps={"1": 0.0212, "8": 0.0275})
    measured["memory"] = {"two_paths_fit": True}
    setup = pc.choose_setup("m", "fp", "c" * 64, log=lambda _: None,
                            numeric=lambda *a, **k: {"plan": None, "reason": "none"},
                            measure=lambda *a, **k: measured)
    by_name = {lever["name"]: lever for lever in setup["levers"]}
    assert set(by_name) == {lever["name"] for lever in pc.LEVERS}
    assert by_name["routed_paths"]["here"] == "used" and by_name["stacked_engine"]["here"].startswith("used")
    assert by_name["numeric_plan"]["here"].startswith("measured, none chosen")
    sources = {"tensor_core_rows": "HyperQwen", "context_speculation": "Strata", "chunked_prefill": "vLLM",
               "paged_kv": "vLLM", "kv_cache_8bit": "HyperQwen", "expert_cache": "Strata", "tpu_path": "TPU"}
    for planned, source in sources.items():
        assert by_name[planned]["status"] == "planned" and by_name[planned]["here"] == "not built yet"
        assert source in by_name[planned]["what"]
    assert "int8_head" not in by_name  # it re-quantised weights: the owner's boundary (HQ1)


def test_the_agent_gate_scores_bos_led_chunks_like_kb2s_own_teacher_forcing():
    """With the gate's BOS lead, the agent's two gate paths (the base decode and the stacked plan, rows with and
    without a lead in one group) equal KB2's independent implementation: same tokens, same positions."""
    from ironmule import dense_gemma

    from tests.engine.test_dense_gemma import tiny_model
    from tools import cuda_probe as probe

    model = tiny_model()
    lead, piece, other = [2], list(range(10, 27)), list(range(40, 57))
    assert pc._decode_nll(model, piece, lead) == pytest.approx(probe.kb2_native_nll(model, lead, piece), abs=1e-6)
    assert pc._decode_nll(model, other) == pytest.approx(probe.kb2_native_nll(model, [], other), abs=1e-6)
    plan = dense_gemma.GemmaDensePlan(model, slots=4, capacity=32, attention=dense_gemma.reference_attention,
                                      native=False)
    agent = pc._plan_nll(plan, [piece, other], [lead, []])
    assert agent == pytest.approx(probe.kb2_plan_nll(plan, [(lead, piece), ([], other)]), abs=1e-6)
    assert agent[0] != pytest.approx(pc._plan_nll(plan, [piece])[0], abs=1e-4), "the lead is really used"


def test_measure_stacked_runs_with_a_bos_tokenizer():
    from ironmule import dense_gemma

    from tests.engine.test_dense_gemma import _Tok, tiny_model

    class BosTok(_Tok):
        bos_token_id = 2

        def encode(self, text):
            return [2] + [ord(c) % 200 + 3 for c in text]

    model = tiny_model()

    def make_plan(model_, info):
        plan = dense_gemma.GemmaDensePlan(model_, slots=4, capacity=96, attention=dense_gemma.reference_attention,
                                          native=False)
        return plan, dense_gemma.probe(plan)

    measured = pc.measure_stacked(model, BosTok(), None, chunks=4, size=16, steps=3, long_prompt="x" * 40,
                                  make_plan=make_plan)
    assert measured["eligible"] and measured["gate"]["chunks"] == 4


def test_a_setup_stored_under_the_old_gate_is_measured_again(tmp_path, monkeypatch):
    import json

    from ironmule.numeric_choice import GATE_VERSION

    store = tmp_path / "plan_choice.json"
    monkeypatch.setattr(pc, "_store", lambda: store)
    key = "fp-" + "a" * 16
    store.write_text(json.dumps({key: {"engine": "dense", "compute_dtype": "native"}}))  # made under gate 1
    calls = []

    def numeric(model_id, fingerprint, identity, log):
        calls.append("numeric")
        return {"plan": "native", "reason": "gate passed"}

    def measure(model_id, compute_dtype, base_only=False):
        calls.append(("measure", compute_dtype))
        return _measured(passes=False)

    setup = pc.choose_setup("m", "fp", "a" * 64, log=lambda _: None, numeric=numeric, measure=measure)
    assert calls == ["numeric", ("measure", "native")] and setup["engine"] != "dense"
    assert json.loads(store.read_text())[key]["gate_version"] == GATE_VERSION


KB2_RUN2_STEPS = {"1": 0.04884, "2": 0.06781, "3": 0.08511, "4": 0.11297, "5": 0.12212, "6": 0.14133,
                  "7": 0.19027, "8": 0.39412}  # KB2-0 run 2's Gemma plan on a T4; today's native step 0.04726


def test_the_stacked_engine_is_judged_and_served_at_its_best_width():
    """AG5: on KB2-0 run 2's curve the width-8 rule refused the plan (20.3 against 21.2 tokens/s); at its best width,
    6, it makes about 2x, and the setup serves at 6."""
    decision = pc.decide(_measured(base_step=0.04726, plan_steps=KB2_RUN2_STEPS, base_first=0.5, plan_first=0.5))
    assert decision["stacked"] and decision["width"] == 6 and "at width 6" in decision["reason"]
    assert decision["tokens_per_second"]["stacked_6"] == pytest.approx(6 / 0.14133)
    assert set(decision["tokens_per_second"]["by_width"]) == set(KB2_RUN2_STEPS)
    assert decision["tokens_per_second"]["by_width"]["8"] < decision["tokens_per_second"]["base_one"]


def test_a_tie_takes_the_narrower_width_and_no_width_wins_is_refused():
    tie = pc.decide(_measured(plan_steps={"1": 0.053, "4": 0.1, "8": 0.2}))
    assert tie["stacked"] and tie["width"] == 4
    flat = pc.decide(_measured(base_step=0.05, plan_steps={"1": 0.06, "2": 0.12, "8": 0.48}))
    assert not flat["stacked"] and "at any width" in flat["reason"]


def test_the_setup_carries_the_width_the_engine_serves_at(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "_store", lambda: tmp_path / "plan_choice.json")

    def numeric(model_id, fingerprint, identity, log):
        return {"plan": "native", "reason": "recommended"}

    def measure(model_id, compute_dtype, base_only=False):
        return _measured(base_step=0.04726, plan_steps=KB2_RUN2_STEPS, base_first=0.5, plan_first=0.5)

    setup = pc.choose_setup("m", "fp", "b" * 64, log=lambda _: None, numeric=numeric, measure=measure)
    assert setup["engine"] == "dense" and setup["width"] == 6


def test_the_router_prices_the_stacked_path_at_the_agents_width():
    from ironmule_product.path_router import PathRouter

    setup = {"stacked": _measured(base_step=0.04726, plan_steps=KB2_RUN2_STEPS), "width": 6}
    router = PathRouter.from_setup(setup)
    assert router.costs[1].capacity == 6
    assert router.costs[1].theta[1] == pytest.approx((0.14133 - 0.04884) / 5 * 1000)
    assert PathRouter.from_setup({**setup, "width": None}).costs[1].capacity == 8


def test_the_base_path_runs_before_the_stacked_plan_takes_its_pool(monkeypatch):
    """AG5-G run 1: on a T4 the base path's whole-prompt prefill (a 2 GB float16 head copy for Gemma 3 12B) ran
    out of memory beside the stacked plan's pool; run 2, with the gate's reference on the base path beside the
    pool, did not finish in two hours. Its timing and the gate's reference now run first, the plan after."""
    calls = []
    monkeypatch.setattr(pc, "_base_speed",
                        lambda model, ids, steps, samples=None: calls.append("base") or (0.1, 0.05))
    monkeypatch.setattr(pc, "_decode_nll", lambda model, ids, lead=None: calls.append("reference") or 1.0)
    monkeypatch.setattr(pc, "_release_memory", lambda: calls.append("release"))

    def make_plan(model, info):
        calls.append("plan")
        raise ValueError("the dense plan serves Qwen 3 checkpoints only")

    tok = type("T", (), {"encode": lambda self, text: [1, 2, 3]})()
    assert pc.measure_stacked(object(), tok, None, chunks=2, size=1, make_plan=make_plan)["eligible"] is False
    assert calls == ["base", "base", "reference", "reference", "release", "plan"]


def test_a_measurement_past_its_time_is_refused_and_the_base_path_kept(monkeypatch):
    """AG5-G run 2: the child's timeout crashed the agent's setup (and the run) after two hours; it now ends in an
    ineligible stacked engine with the reason, and the child is killed by `subprocess.run`."""
    import subprocess

    def run(*args, timeout=None, **kwargs):
        raise subprocess.TimeoutExpired(args[0], timeout)

    monkeypatch.setattr(subprocess, "run", run)
    out = pc._measure_in_child("m", "native")
    assert out == {"eligible": False, "technical": True,
                   "reason": f"the measurement did not finish within {pc.MEASURE_TIMEOUT_S} s"}
    assert pc.MEASURE_TIMEOUT_S <= 3600


def test_a_stacked_engine_without_a_measured_rival_is_not_eligible():
    """`against` read the base-only measurement's speed rows unchecked, so a failed child raised a KeyError."""
    stacked = {"eligible": True, "speed": {"short": {"base_first_s": 0.1, "base_step_s": 0.05}}}
    failed = {"eligible": False, "reason": "the measurement child exited with status 1"}
    assert pc.against(stacked, failed) == {
        "eligible": False, "technical": True,
        "reason": "the path it competes with was not measured: the measurement child exited with status 1"}


# ---------------------------------------------------------------- stability: the setup never fails the server

def test_a_measurement_or_gate_that_raises_leaves_the_base_path_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "_store", lambda: tmp_path / "plan_choice.json")

    def measure(model_id, compute_dtype, base_only=False):
        raise MemoryError("cudaMalloc failed")

    setup = pc.choose_setup("m", "fp", "d" * 64, log=lambda _: None,
                            numeric=lambda *a, **k: {"plan": "native", "reason": "within the gate"},
                            measure=measure)
    assert setup["engine"] == "reference" and setup["compute_dtype"] == "native"
    assert "measurement failed (MemoryError: cudaMalloc failed)" in setup["reason"]

    def numeric(*args, **kwargs):
        raise OSError("disk")

    setup = pc.choose_setup("m", "fp", "e" * 64, log=lambda _: None, numeric=numeric,
                            measure=lambda *a, **k: {"eligible": False, "reason": "no plan"})
    assert setup["compute_dtype"] is None and setup["numeric"]["reason"].startswith("the numeric gate failed")


def test_a_store_of_another_shape_or_on_a_read_only_disk_never_fails_the_setup(tmp_path, monkeypatch):
    from ironmule import numeric_choice as nc

    odd = tmp_path / "odd.json"
    odd.write_text("[1, 2, 3]")
    monkeypatch.setattr(pc, "_store", lambda: odd)
    monkeypatch.setattr(nc, "_store", lambda: odd)
    assert pc._stored("k") is None and nc._stored("k") is None
    blocked = tmp_path / "file"
    blocked.write_text("")
    monkeypatch.setattr(pc, "_store", lambda: blocked / "plan_choice.json")  # its folder is a file
    monkeypatch.setattr(nc, "_store", lambda: blocked / "numeric.json")
    assert pc._save("k", {"engine": "reference"})["engine"] == "reference"
    assert nc._save("k", {"plan": None})["plan"] is None


def test_a_child_that_prints_no_readable_result_is_refused(monkeypatch):
    import subprocess

    class Done:
        returncode, stdout = 0, "@@STACKED {not json\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done())
    assert pc._measure_in_child("m", None) == {"eligible": False, "technical": True,
                                               "reason": "the measurement child returned no readable result"}


def test_a_step_that_runs_out_of_memory_runs_once_more_after_freeing_and_is_recorded(monkeypatch):
    """AG5-G run 3: after its gate had passed, the long prompt's first prefill on the stacked plan failed in
    `cudaMallocAsync ... out of memory`. A step that runs out of memory now runs once more after every freed buffer
    went back; a second failure, or any other error, is raised as before."""
    import pytest

    released, calls, retried = [], [], []
    monkeypatch.setattr(pc, "_release_memory", lambda: released.append(1))

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("cudaMallocAsync(&data, size, stream) failed: out of memory")
        return "timed"

    assert pc._again_if_out_of_memory(flaky, retried, "speed, long prompt") == "timed"
    assert calls == [1, 1] and released == [1] and retried == ["speed, long prompt"]

    def always():
        raise RuntimeError("cudaMallocAsync failed: out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        pc._again_if_out_of_memory(always, [], "gate")

    def other():
        calls.append("other")
        raise RuntimeError("an illegal memory access was encountered")

    with pytest.raises(RuntimeError, match="illegal memory access"):
        pc._again_if_out_of_memory(other, [], "gate")
    assert calls.count("other") == 1, "only running out of memory is tried again"


def test_the_stage_log_carries_the_allocator_and_the_record_its_retries():
    from ironmule import dense_gemma

    from tests.engine.test_dense_gemma import _Tok, tiny_model

    model = tiny_model()

    def make_plan(model_, info):
        plan = dense_gemma.GemmaDensePlan(model_, slots=2, capacity=96, attention=dense_gemma.reference_attention,
                                          native=False)
        return plan, dense_gemma.probe(plan)

    lines = []
    measured = pc.measure_stacked(model, _Tok(), None, chunks=4, size=16, steps=3, long_prompt="x" * 40,
                                  make_plan=make_plan, log=lines.append)
    assert measured["retried_out_of_memory"] == []
    assert all("active" in line and "peak" in line for line in lines), lines


def test_first_tokens_are_medians_of_several_prefills_and_their_samples_are_kept():
    """AG5-G run 5 (post hoc): the rule's first-token ratio read 1.05 in run 4 and 1.18 in run 5 for the same
    prefill code, each side timed once. Each side is now the median of several prefills, all of them recorded."""
    import statistics

    from ironmule import dense_gemma

    from tests.engine.test_dense_gemma import _Tok, tiny_model

    model = tiny_model()

    def make_plan(model_, info):
        plan = dense_gemma.GemmaDensePlan(model_, slots=3, capacity=96, attention=dense_gemma.reference_attention,
                                          native=False)
        return plan, dense_gemma.probe(plan)

    measured = pc.measure_stacked(model, _Tok(), None, chunks=4, size=16, steps=3, long_prompt="x" * 40,
                                  make_plan=make_plan)
    for row in measured["speed"].values():
        assert len(row["base_first_samples_s"]) == pc.FIRST_SAMPLES and len(row["plan_first_samples_s"]) == 3
        assert row["base_first_s"] == statistics.median(row["base_first_samples_s"])
        assert row["plan_first_s"] == statistics.median(row["plan_first_samples_s"])


def test_a_technical_measurement_failure_is_measured_again_and_a_refusal_is_kept(tmp_path, monkeypatch):
    """Review of 2026-11-08: a measurement that failed for a technical reason was stored, so the base path served
    for good after the cause had gone. It now serves the base path once and is measured again on the next start;
    a refusal by a rule is still stored."""
    monkeypatch.setattr(pc, "_store", lambda: tmp_path / "plan_choice.json")
    calls = []

    def numeric(*args, **kwargs):
        calls.append("numeric")
        return {"plan": "native", "reason": "a passed gate"}

    def failed(*args, **kwargs):
        calls.append("failed")
        raise RuntimeError("out of memory")

    def healthy(*args, **kwargs):
        calls.append("healthy")
        row = {"base_first_s": .5, "base_step_s": .052, "plan_first_s": .5, "plan_step_s": {"1": .053, "8": .17}}
        return {"eligible": True, "install": {}, "speed": {"short": row, "long": dict(row)},
                "gate": {"passes": True, "ratio": 1.001, "interval": [.998, 1.004]}}

    first = pc.choose_setup("m", "hw", "a" * 64, log=lambda _: None, numeric=numeric, measure=failed)
    assert first["engine"] == "reference" and first["measurement_failed"] is True
    second = pc.choose_setup("m", "hw", "a" * 64, log=lambda _: None, numeric=numeric, measure=healthy)
    assert second["engine"] == "dense" and "measurement_failed" not in second
    assert pc.choose_setup("m", "hw", "a" * 64, log=lambda _: None, numeric=numeric, measure=failed)["engine"] == \
        "dense", "a stored success is used"
    assert calls == ["numeric", "failed", "numeric", "healthy"]

    refused = lambda *args, **kwargs: {"eligible": False, "reason": "the dense plan serves Qwen 3 checkpoints only"}  # noqa: E731
    pc.choose_setup("m", "hw", "b" * 64, log=lambda _: None, numeric=numeric, measure=refused)
    again = pc.choose_setup("m", "hw", "b" * 64, log=lambda _: None, numeric=numeric, measure=failed)
    assert again["engine"] == "reference" and "measurement_failed" not in again, "a refusal is kept, not remeasured"

    for technical in (lambda *a, **k: {"eligible": False, "technical": True, "reason": "child exited with status 1"},):
        result = pc.choose_setup("m", "hw", "c" * 64, log=lambda _: None, numeric=numeric, measure=technical)
        assert result["measurement_failed"] is True and pc._stored("hw-" + "c" * 16) is None


# -- CPU6 step 3: on a CPU the agent also chooses the BLAS and measures the CPU stacked plan -------------------------


def _blas(choice="openblas", **extra):
    def run(model_id, compute_dtype):
        out = {"cpu_blas": choice, "env": {"LD_PRELOAD": "/x/shim.so"} if choice else {}, "reason": f"{choice}"}
        return {**out, **extra}
    return run


def test_on_a_cpu_the_agent_chooses_the_blas_and_measures_the_stacked_plan_on_it(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "_store", lambda: tmp_path / "plan_choice.json")
    seen = []

    def measure(model_id, compute_dtype, base_only=False):
        seen.append((compute_dtype, base_only))
        return _measured()

    setup = pc.choose_setup("m", "fp", "d" * 64, log=lambda _: None, cpu=True, blas=_blas(),
                            numeric=lambda *a, **k: {"plan": "dequantize", "reason": "CPU check passed"},
                            measure=measure)
    assert setup["engine"] == "dense" and setup["cpu_blas"] == "openblas" and setup["compute_dtype"] == "dequantize"
    assert seen == [("dequantize", False)], "measured on the plan that serves, not on the checkpoint's dtype"
    assert setup["cpu_setup"] == pc.CPU_SETUP_VERSION and "env" not in setup["blas"]
    assert {l["name"]: l["here"] for l in setup["levers"]}["cpu_blas"] == "used: OpenBLAS"


def test_on_a_cpu_routing_is_not_offered_and_the_stock_blas_keeps_its_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "_store", lambda: tmp_path / "plan_choice.json")
    loaded = _measured(plan_steps={"1": 0.07, "8": 0.17})  # wins under load, not alone
    loaded["memory"] = {"two_paths_fit": True}
    setup = pc.choose_setup("m", "fp", "e" * 64, log=lambda _: None, cpu=True,
                            blas=_blas(None, reason="no C compiler"),
                            numeric=lambda *a, **k: {"plan": "dequantize", "reason": "ok"},
                            measure=lambda *a, **k: loaded)
    assert setup["engine"] == "reference" and setup["cpu_blas"] is None
    assert "routing is not offered on a CPU" in setup["reason"]
    assert {l["name"]: l["here"] for l in setup["levers"]}["cpu_blas"].startswith("measured, not chosen")


def test_a_cpu_setup_from_before_step_3_is_measured_again_once_and_a_gpu_s_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "_store", lambda: tmp_path / "plan_choice.json")
    pc._save("fp-" + "f" * 16, {"engine": "reference", "compute_dtype": "dequantize", "reason": "AG4-C"})
    calls = []
    numeric = lambda *a, **k: calls.append("numeric") or {"plan": "dequantize", "reason": "ok"}  # noqa: E731
    gpu = pc.choose_setup("m", "fp", "f" * 64, log=lambda _: None, cpu=False, numeric=numeric,
                          measure=lambda *a, **k: _measured())
    assert gpu["reason"] == "AG4-C" and calls == []
    cpu = pc.choose_setup("m", "fp", "f" * 64, log=lambda _: None, cpu=True, blas=_blas(), numeric=numeric,
                          measure=lambda *a, **k: _measured())
    again = pc.choose_setup("m", "fp", "f" * 64, log=lambda _: None, cpu=True, blas=_blas(), numeric=numeric,
                            measure=lambda *a, **k: _measured())
    assert cpu["engine"] == again["engine"] == "dense" and calls == ["numeric"]


def test_a_blas_measurement_that_fails_keeps_the_stock_blas_and_is_measured_again(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "_store", lambda: tmp_path / "plan_choice.json")

    def broken(model_id, compute_dtype):
        raise RuntimeError("child died")

    setup = pc.choose_setup("m", "fp", "g" * 64, log=lambda _: None, cpu=True, blas=broken,
                            numeric=lambda *a, **k: {"plan": "dequantize", "reason": "ok"},
                            measure=lambda *a, **k: _measured())
    assert setup["cpu_blas"] is None and setup["measurement_failed"] is True and "child died" in setup["blas"]["reason"]


@pytest.mark.parametrize("stock, fast, chosen, why", [
    ({"logits": [1.0, 3.0], "seconds": 10.0, "serving": "stock"},
     {"logits": [1.0, 3.0001], "seconds": 3.0, "serving": "openblas"}, "openblas", "same top token"),
    ({"logits": [1.0, 3.0], "seconds": 10.0, "serving": "stock"},
     {"logits": [1.0, 3.0], "seconds": 9.8, "serving": "openblas"}, None, "0.980 of the stock"),
    ({"logits": [1.0, 3.0], "seconds": 10.0, "serving": "stock"},
     {"logits": [3.2, 3.0], "seconds": 3.0, "serving": "openblas"}, None, "top token differs"),
    ({"logits": [1.0, 3.0], "seconds": 10.0, "serving": "stock"},
     {"logits": [1.0, 3.0], "seconds": 3.0, "serving": "stock"}, None, "did not serve"),
])
def test_the_blas_rule_fixed_before_the_data(stock, fast, chosen, why):
    env = {"LD_PRELOAD": "/x/shim.so"}
    out = pc.choose_blas("m", "dequantize", environment=lambda: (env, {"available": True}),
                         measure=lambda m, c, e: fast if e else stock)
    assert out["cpu_blas"] == chosen and why in out["reason"] and out["env"] == (env if chosen else {})
    off = pc.choose_blas("m", None, environment=lambda: ({}, {"reason": "no C compiler"}), measure=None)
    assert off["cpu_blas"] is None and "no C compiler" in off["reason"]


def test_measure_stacked_runs_the_cpu_plan_on_a_tiny_llama_and_the_rule_reads_it(monkeypatch):
    """The real measurement on MLX's CPU: the CPU plan installed by `stacked_plan`, its gate against the base
    path's own decode, every width timed."""
    from tests.engine.test_dense_cpu import LLAMA, tiny
    from tests.engine.test_dense_gemma import _Tok

    monkeypatch.setattr(pc, "on_cpu", lambda: True)
    model = tiny(LLAMA)
    measured = pc.measure_stacked(model, _Tok(), None, chunks=8, size=12, steps=3, long_prompt="y" * 30)
    assert measured["eligible"] and measured["install"]["kernels"] == "cpu_float32"
    assert measured["gate"]["passes"], measured["gate"]
    assert set(measured["speed"]["short"]["plan_step_s"]) == {str(w) for w in range(1, 9)}
    assert "stacked" in pc.decide(measured)
