"""HW3 and TB1's probe (tools/cuda_probe.py): every part runs end to end on a tiny random Qwen 3 on the CPU, and
the decisions read what the backlog fixed before data. A CPU run checks the tool, not a hardware claim."""

from __future__ import annotations

import json
import math

import pytest

from tools import cuda_probe as probe

mx = pytest.importorskip("mlx.core")


class _Tokenizer:
    # Outside the tiny model's 256-token vocabulary: no run stops early, so every timed run decodes its steps.
    # With 255 a random model's first token was sometimes EOS, and a zero time per token has no logarithm
    # (CI on 843a865).
    eos_token_id = 999
    eos_token_ids = {999}

    def encode(self, text):
        return [ord(c) % 200 + 1 for c in text]

    def decode(self, ids):
        return "".join(chr(32 + i % 90) for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        return self.encode(messages[-1]["content"])


def _tiny():
    from mlx_lm.models import qwen3

    mx.random.seed(11)  # fixed weights, so a failure reproduces
    args = qwen3.ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
                           num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=256, num_key_value_heads=2,
                           max_position_embeddings=512, rope_theta=10000.0, head_dim=16, tie_word_embeddings=True)
    plain, fused = qwen3.Model(args), qwen3.Model(args)
    plain.set_dtype(mx.bfloat16)
    fused.update(plain.parameters())
    mx.eval(plain.parameters(), fused.parameters())
    return plain, fused


def test_every_part_runs_and_the_engine_knobs_keep_the_tiny_models_tokens(tmp_path):
    plain, fused = _tiny()
    out = tmp_path / "report.json"
    report = probe.probe("tiny", out, smoke=True, loaded=(plain, fused, _Tokenizer()))
    assert json.loads(out.read_text())["finished_unix"] == report["finished_unix"]
    for part in ("linear", "attention", "overlap", "batching", "factorial"):
        assert "error" not in report[part], (part, report[part])
    runs = report["factorial"]["runs"]
    assert {r["combo"] for r in runs} == {probe.combo_name(b) for b in probe.combos()}
    assert not [r for r in runs if "error" in r] and not report["factorial"]["errors"]  # fusion ran too
    baseline = [r for r in runs if r["combo"] == "baseline"]
    assert len({r["digest"] for r in baseline if r["prompt"] == 0}) == 1  # the engine is deterministic
    assert all(o["same_tokens"] for o in report["overlap"])  # pipelining builds the same steps
    assert all(b["equal_tokens"] >= 1 for b in report["batching"])
    analysis = report["analysis"]
    assert set(analysis["decisions"]) >= {"compile (8)", "speculation (10)", "overlap (16)", "true batching (1)"}
    assert analysis["factorial"]["main_effects"] is not None and len(analysis["factorial"]["interactions"]) == 10


def _factorial(cost, exact=lambda name: True, rounds=5, prefill=lambda bits, long: 1.0):
    """A synthetic factorial whose decode cost per combination is `cost(bits)` times the baseline's, and its
    prefill `prefill(bits, long)` times."""
    runs = []
    for round_ in range(-1, rounds):
        order = probe.round_order(max(round_, 0))
        for position, bits in enumerate(order[:-1] if round_ < 0 else order):
            for p in range(6):
                name, jitter, long = probe.combo_name(bits), 1 + 0.001 * ((round_ + p) % 3), p >= 3
                runs.append({"combo": name, "round": round_, "prompt": p, "position": position,
                             "ms_per_token": 20.0 * cost(bits) * jitter,
                             "prefill_ms": (900.0 if long else 60.0) * prefill(bits, long) * jitter,
                             "first_difference": None if exact(name) else 5, "acceptance": 0.0})
    return {"runs": runs, "prompt_tokens": [40, 40, 40, 1100, 1100, 1100]}


def test_a_combination_qualifies_only_when_exact_and_faster_than_the_noise_and_its_interaction_is_measured():
    def cost(bits):
        fp, cc, fa, rb, sk = bits
        value = (0.90 if cc else 1.0) * (0.95 if rb else 1.0) * (2.0 if sk else 1.0)
        return value * (1.04 if cc and rb else 1.0)  # compile and readback overlap: together less than the product

    result = probe.analyse_factorial(_factorial(cost, exact=lambda name: "speculate_k" not in name), resamples=200)
    combos = result["combos"]
    assert combos["compiled_fixed_cache"]["qualifies"] and math.isclose(combos["compiled_fixed_cache"]["ratio"], 0.9,
                                                                        rel_tol=1e-6)
    assert not combos["speculate_k"]["exact"] and not combos["speculate_k"]["qualifies"]
    assert not combos["fused_argmax"]["qualifies"]  # no faster than the baseline
    assert combos[result["best_qualifying"]]["ratio"] == pytest.approx(0.9 * 0.95 * 1.04, rel=1e-6)
    assert result["interactions"]["compiled_fixed_cache*readback_every"] == pytest.approx(math.sqrt(1.04), rel=1e-6)
    assert result["interactions"]["fuse_projections*fused_argmax"] == pytest.approx(1.0, abs=1e-9)
    assert result["noise"] < 0.01


def test_the_decisions_count_every_resolved_gain_and_no_unresolved_one():
    flat = _factorial(lambda bits: 1.0)
    report = {"layers": 28, "weights_bytes": 3_400_000_000, "factorial": flat,
              "overlap": [{"long": False, "sync_seconds_per_step": 0.02, "pipelined_seconds_per_step": 0.0198,
                           "saved": [0.01, 0.012, 0.009, 0.011, 0.01], "same_tokens": True},
                          {"long": True, "sync_seconds_per_step": 0.03, "pipelined_seconds_per_step": 0.03,
                           "saved": [0.01, -0.01, 0.0, 0.002, -0.003], "same_tokens": True}],
              "attention": [{"length": 1024, "rows": 1, "masked_seconds": 1.1e-4, "cut_seconds": 1.0e-4,
                             "saved_seconds": [1e-5] * 20, "identical": False}],
              "batching": [{"length": 64, "batch": 1, "seconds_per_step": 0.02, "equal_tokens": 16,
                            "identical_logits": 16},
                           {"length": 64, "batch": 8, "seconds_per_step": 0.03, "equal_tokens": 8,
                            "identical_logits": 6}]}
    decisions = probe.analyse(report, resamples=200)["decisions"]
    assert decisions["overlap (16)"]["verdict"] == "worth building"  # one percent, resolved at one context
    assert decisions["fused attention (6)"]["verdict"] == "worth building"
    assert decisions["true batching (1)"]["exactness"].startswith("token-exact here but not invariant")
    assert decisions["true batching (1)"]["lever_at_8"]["64"] == pytest.approx(8 * 0.02 / 0.03)
    report["overlap"][0]["saved"][2] = -0.001  # one pair the other way: not resolved
    report["attention"][0]["saved_seconds"] = [1e-5, -1e-5] * 10
    decisions = probe.analyse(report, resamples=200)["decisions"]
    assert decisions["overlap (16)"]["verdict"].startswith("closed")
    assert decisions["fused attention (6)"]["verdict"].startswith("closed")


def test_the_kernel_runs_exactly_the_pushed_commit(tmp_path):
    commit = "a" * 40
    folder = probe.write_kernel(tmp_path / "kernel", "someone", commit)
    source = (folder / "run.py").read_text()
    assert f"git+{probe.REPOSITORY}@{commit}" in source and "cuda_probe_report.json" in source
    assert json.loads((folder / "kernel-metadata.json").read_text())["id"] == "someone/ironmule-cuda-probe"
    compile(source, "run.py", "exec")
    with pytest.raises(ValueError):
        probe.write_kernel(tmp_path / "bad", "someone", "main")


def test_the_phase_table_counts_a_prefill_win_where_prefill_dominates_and_a_decode_win_elsewhere():
    """Post hoc RL7 analysis: fusion halves a long prompt's prefill but slows decode 5 %; compiling speeds decode
    10 %. Short answers after long prompts want both, long answers want compiling alone."""
    def decode(bits):
        return (1.05 if bits[0] else 1.0) * (0.90 if bits[1] else 1.0)

    def prefill(bits, long):
        return 0.5 if bits[0] and long else 1.0

    phases = probe.analyse_phases(_factorial(decode, prefill=prefill), resamples=200)
    assert phases["phases"]["fuse_projections"]["long"]["prefill_ratio"] == pytest.approx(0.5, rel=1e-3)
    assert phases["phases"]["fuse_projections"]["short"]["decode_ratio"] == pytest.approx(1.05, rel=1e-3)
    assert "fuse_projections" in phases["winners"]["long/16"] and "compiled_fixed_cache" in phases["winners"]["long/16"]
    assert "fuse_projections" not in phases["winners"]["short/1024"]
    assert phases["headroom_ci95"][0] > phases["noise"] and phases["verdict"].endswith("RL7 continues")
    flat = probe.analyse_phases(_factorial(lambda bits: 0.9 if bits[1] else 1.0), resamples=200)
    assert flat["verdict"].startswith("one combination for every context")


def test_hw4_runs_every_part_and_judges_each_knob_set_bit_for_bit(tmp_path):
    plain, fused = _tiny()
    out = tmp_path / "hw4.json"
    report = probe.probe_hw4("tiny", out, {"compiled_fixed_cache": True, "readback_every": 4}, smoke=True,
                             loaded=(plain, fused, _Tokenizer()))
    assert json.loads(out.read_text())["finished_unix"] == report["finished_unix"]
    for part in ("prefill_floor", "class_a", "separability", "widths", "compile"):
        assert not (isinstance(report[part], dict) and "error" in report[part]), (part, report[part])
    assert not [w for w in report["widths"] if "error" in w]
    analysis = report["analysis"]
    verdicts = analysis["class_a"]["verdicts"]
    assert verdicts["baseline"]["class_a"] and verdicts["compiled_fixed_cache"]["class_a"]
    assert verdicts["speculate_k"]["verify_windows"] is not None  # the k + 1 verify window was compared
    assert set(report["decode_sets"]) >= {"baseline", "readback_every"}
    assert analysis["widths"]["identical_answers_across_sets"]
    assert "RL7 kill 2 (dominance)" in analysis["decisions"]
    assert set(analysis["compile_seconds"]) == set(report["decode_sets"])


def _capture(steps, picks=None, prefill="p", final="f", windows=None):
    row = {"prefill_logits": prefill, "prefill_state": "s", "steps": steps, "picks": picks or [1, 2, 3],
           "final_state": final}
    if windows is not None:
        row["verify_windows"] = windows
    return {"rows": [row]}


def test_class_a_needs_every_bit_and_a_fused_argmax_inherits_its_twins_verdict():
    capture = {"sets": {
        "baseline": _capture(["a", "b", "c"]),
        "compiled_fixed_cache": _capture(["a", "b", "c"]),
        "fuse_projections": _capture(["a", "x", "c"]),
        "head_skip_prefill": _capture(["a", "b", "c"], prefill="q"),
        "fused_argmax": _capture(["1", "2", "3"]),
        "fuse_projections+fused_argmax": _capture(["1", "2", "3"]),
        "speculate_k": _capture(["a", "b", "c"], windows={"0": ["a", "b", "y"]}),
    }}
    verdicts = probe.class_a_sets(capture)
    assert verdicts["compiled_fixed_cache"]["class_a"] and verdicts["fused_argmax"]["class_a"]
    assert not verdicts["fuse_projections"]["class_a"] and verdicts["fuse_projections"]["first_differing_step"] == 1
    assert not verdicts["fuse_projections+fused_argmax"]["class_a"]  # same picks, but its twin's logits differ
    assert not verdicts["head_skip_prefill"]["class_a"] and not verdicts["speculate_k"]["class_a"]
    sets = probe.decode_sets_from(verdicts)
    assert sets["compiled_fixed_cache+readback_every"] == {"compiled_fixed_cache": True, "readback_every": 4}
    assert "fuse_projections" not in sets and "baseline" in sets


def test_rl7_continues_only_when_a_knob_set_changes_order_with_the_group_width():
    def widths(cost):
        return [{"set": name, "width": w, "round": r, "seconds": cost(name, w) * 64, "tokens": [64] * w,
                 "digests": ["d"] * w} for name in ("baseline", "compiled_fixed_cache") for w in (1, 2, 4)
                for r in range(5)]

    parallel = probe.analyse_hw4({"widths": widths(lambda n, w: (0.9 if n != "baseline" else 1.0) * (20 + 13 * w))})
    assert parallel["decisions"]["RL7 kill 2 (dominance)"].startswith("knob sets and scheduling separate")
    # cheaper alone, dearer per further session: compiled wins at width 1 and loses at width 4
    crossing = probe.analyse_hw4({"widths": widths(lambda n, w: 14 + 16 * w if n != "baseline" else 20 + 13 * w)})
    assert crossing["decisions"]["RL7 kill 2 (dominance)"].endswith("continue")
    assert crossing["widths"]["sets"]["compiled_fixed_cache"]["seconds_per_extra_session"] == pytest.approx(16)


def _row_by_row(x, w):
    """A stand-in for the CUDA row kernel on the CPU: each row on its own, so it is batch-invariant."""
    rows = x.reshape(-1, x.shape[-1])
    out = mx.concatenate([(rows[i:i + 1].astype(mx.float32) @ w.astype(mx.float32).T).astype(mx.bfloat16)
                          for i in range(rows.shape[0])])
    return out.reshape(*x.shape[:-1], w.shape[0])


def test_kb1_compares_real_activations_bit_for_bit_and_applies_its_three_kills(tmp_path):
    plain, _ = _tiny()
    report = probe.probe_kb1_report("tiny", tmp_path / "kb1.json", smoke=True, loaded=(plain, _Tokenizer()),
                                    kernel=_row_by_row)
    layers = report["kb1"]
    assert [e["layer"] for e in layers] == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                           "down_proj", "output"]
    assert all(e["rows_compared"] == 8 and set(e["rows"]) == {"1", "2", "4", "8"} for e in layers)
    analysis = report["analysis"]
    assert analysis["kill (ii) batch-invariant"] is True and analysis["valid"]
    assert analysis["kill (iii) limit"] == pytest.approx(8 * (19.8 + 7 * 13.4) / (8 * 19.8))
    assert "verdict" in analysis and json.loads((tmp_path / "kb1.json").read_text())["schema"].endswith("kb1.v1")


def test_kb1_stops_when_eight_rows_cost_what_grouped_batch_1_already_saves():
    def layer(kernel8, mlx1, invariant=8):
        rows = {m: {"invariant_rows": int(m), "kernel_seconds": [kernel8 if m == "8" else 1.0] * 5,
                    "mlx_seconds": [mlx1] * 5} for m in ("1", "2", "4", "8")}
        rows["8"]["invariant_rows"] = invariant
        return {"layer": "q_proj", "per_step": 28, "weight_bytes": 8_000_000, "identical_to_mlx_at_1": 64,
                "rows_compared": 64, "max_difference_to_mlx": 0.0, "rows": rows}

    assert probe.analyse_kb1([layer(1.5, 1.0)], resamples=100)["verdict"].startswith("class A route")
    assert probe.analyse_kb1([layer(6.0, 1.0)], resamples=100)["verdict"].startswith("no lever")
    assert probe.analyse_kb1([layer(1.5, 1.0, invariant=7)], resamples=100)["verdict"].startswith("bug")


def _attention_by_row(q, k, v, lengths):
    """A CPU stand-in for FA2: each row on its own, reading only its first `lengths[b]` positions."""
    rows, heads, dim = q.shape
    groups = heads // k.shape[1]
    out = []
    for b in range(rows):
        n = int(lengths[b].item())
        keys = mx.repeat(k[b, :, :n, :], groups, axis=0).astype(mx.float32)
        values = mx.repeat(v[b, :, :n, :], groups, axis=0).astype(mx.float32)
        scores = (q[b].astype(mx.float32)[:, None, :] @ keys.transpose(0, 2, 1))[:, 0, :] * dim ** -0.5
        out.append((mx.softmax(scores, axis=-1)[:, None, :] @ values)[:, 0, :])
    return mx.stack(out).astype(mx.bfloat16)


def test_the_dense_plan_decodes_like_the_model_and_stacking_rows_changes_no_bit():
    from mlx_lm.models.cache import make_prompt_cache

    plain, _ = _tiny()
    plan = probe.DensePlan(plain, linear=_row_by_row, attention=_attention_by_row)
    prompts = [[5, 9, 13, 17, 21], [3, 7, 11, 15, 19, 23, 27, 31, 35]]
    singles = []
    for ids in prompts:
        states, first = probe._prefill(plain, ids)
        cache = plan.load_prefill(plan.new_cache(1, 64), 0, states)
        logits, _ = plan.step(first, mx.array([len(ids)]), cache)
        singles.append((first, logits))
        model_cache = make_prompt_cache(plain)
        plain(mx.array(ids)[None], cache=model_cache)
        expected = plain(first[:, None], cache=model_cache)[0, -1].astype(mx.float32)
        assert int(mx.argmax(logits[0]).item()) == int(mx.argmax(expected).item())
        assert float(mx.max(mx.abs(logits[0] - expected)).item()) < 0.05 * float(mx.max(mx.abs(expected)).item())
    cache = plan.new_cache(2, 128)  # a different capacity, both rows at their own positions
    for r, ids in enumerate(prompts):
        cache = plan.load_prefill(cache, r, probe._prefill(plain, ids)[0])
    stacked, _ = plan.step(mx.concatenate([f for f, _ in singles]), mx.array([len(p) for p in prompts]), cache)
    for r, (_, logits) in enumerate(singles):
        assert mx.array_equal(stacked[r], logits[0]).item()


def test_kb1_1_runs_end_to_end_on_cpu_stand_ins(tmp_path):
    plain, _ = _tiny()
    plan = probe.DensePlan(plain, linear=_row_by_row, attention=_attention_by_row)
    report = probe.probe_kb1_report("tiny", tmp_path / "kb1.json", smoke=True, loaded=(plain, _Tokenizer()),
                                    kernel=_row_by_row, model_level=True, plan=plan)
    assert "error" not in report["kb1_model"], report["kb1_model"]
    analysis = report["analysis_model"]
    assert analysis["(a) exact within the plan"] is True
    assert set(analysis["(c) quality gate"]["paths"]) == {"main", "long"}
    assert analysis["(d) one request, plan over today's tuned"] > 0
    assert "(b) stacked step over grouped round" in analysis and "double capacity step ratio" in analysis


def test_kb1_2_measures_each_variant_across_widths_and_judges_the_single_row_regression(tmp_path):
    plain, _ = _tiny()
    base = probe.DensePlan(plain, linear=_row_by_row, attention=_attention_by_row)
    variants = {"kb1_1": (base, base.step), "rows1": (base, base.step)}
    report = probe.probe_kb12_report("tiny", tmp_path / "kb12.json", smoke=True, loaded=(plain, _Tokenizer()),
                                     variants=variants)
    assert "error" not in report["kb12"], report["kb12"]
    analysis = report["analysis"]
    for name in variants:
        v = analysis["variants"][name]
        assert v["exact_across_widths"], name
        assert set(v["ms_per_step_by_width"]) == {str(w) for w in range(1, 9)}
    assert analysis["best"] in variants and analysis["(d) single-row regression"] is not None



def _masked_attention(q, k, v, lengths):
    """A compile-friendly CPU stand-in: masked attention over the whole capacity, row by row."""
    rows, heads, dim = q.shape
    groups = heads // k.shape[1]
    out = []
    for b in range(rows):
        keys = mx.repeat(k[b], groups, axis=0).astype(mx.float32)
        values = mx.repeat(v[b], groups, axis=0).astype(mx.float32)
        scores = (q[b].astype(mx.float32)[:, None, :] @ keys.transpose(0, 2, 1))[:, 0, :] * dim ** -0.5
        scores = mx.where(mx.arange(k.shape[2])[None, :] < lengths[b], scores, -mx.inf)
        out.append((mx.softmax(scores, axis=-1)[:, None, :] @ values)[:, 0, :])
    return mx.stack(out).astype(mx.bfloat16)


def test_the_plan_step_compiles_and_the_server_loop_reads_every_fourth_step():
    plain, _ = _tiny()
    plan = probe.DensePlan(plain, linear=_row_by_row, attention=_masked_attention)
    compiled = mx.compile(plan.step)
    states, first = probe._prefill(plain, [5, 9, 13, 17, 21])
    cache = plan.load_prefill(plan.new_cache(1, 64), 0, states)
    eager, _ = plan.step(first.astype(mx.uint32), mx.array([5]), cache)
    traced, _ = compiled(first.astype(mx.uint32), mx.array([5]), cache)
    assert int(mx.argmax(eager[0]).item()) == int(mx.argmax(traced[0]).item())
    seconds, tokens = probe.plan_decode(compiled, first.astype(mx.uint32), mx.array([5]),
                                        plan.load_prefill(plan.new_cache(1, 64), 0, states), 6)
    assert seconds > 0 and len(tokens) == 1 and len(tokens[0]) == 6


def test_kb2_runs_end_to_end_on_cpu_stand_ins(tmp_path):
    """KB2-0 on a tiny random 4-bit Gemma 3 with FA2's semantics in plain MLX: every part runs and the analysis
    reads each decision. The CPU's quantized product is not batch-invariant, so (a) is not asserted here; the
    kernels' exactness is the hardware run's to show."""
    from ironmule import dense_gemma

    from tests.engine.test_dense_gemma import tiny_model

    model = tiny_model()

    def make_plan(slots, capacity):
        return dense_gemma.GemmaDensePlan(model, slots=slots, capacity=capacity,
                                          attention=dense_gemma.reference_attention, native=False)

    report = probe.probe_kb2_report("tiny", tmp_path / "kb2.json", smoke=True, loaded=(model, _Tokenizer(), {}),
                                    make_plan=make_plan)
    assert "error" not in report["kb2"], report["kb2"]
    kb2, analysis = report["kb2"], report["analysis"]
    assert set(kb2["exact"]["stacks"]) == {str(n) for n in probe.KB2_STACKS}
    assert set(kb2["exact_long"]["stacks"]) == {str(probe.KB2_LONG_ROWS)}
    assert max(kb2["prompt_tokens"]) <= 64 and len(kb2["prompt_tokens"]) == probe.KB2_ROWS
    b = analysis["(b) stacked step over single steps"]
    assert 1 <= b["width"] <= probe.KB2_ROWS and set(b["plan_ms_by_width"]) == {str(w) for w in range(1, 9)}
    assert "(b) recorded: ratio at the widest stack" in analysis
    assert set(analysis["(c) quality gate"]["paths"]) == {"main", "long"}
    assert analysis["(d) one request, plan over native"] > 0
    assert json.loads((tmp_path / "kb2.json").read_text())["schema"] == "ironmule.cuda_probe.kb2.v1"


def test_kb2_runs_native_references_before_any_pool_and_keeps_partial_results(tmp_path):
    """KB2-0 run 1 ran out of memory on the T4 and its report kept nothing. Today's native path (its head over
    every prompt position) must run before a plan's pool exists, and a failure keeps the stage it reached."""
    from tests.engine.test_dense_gemma import tiny_model

    model = tiny_model()

    def make_plan(slots, capacity):
        raise MemoryError("cudaMallocAsync failed: out of memory")

    report = probe.probe_kb2_report("tiny", tmp_path / "kb2.json", smoke=True, loaded=(model, _Tokenizer(), {}),
                                    make_plan=make_plan)
    kb2 = report["kb2"]
    assert "out of memory" in kb2["error"] and kb2["stage"] == "main pool"
    assert list(kb2["memory"]) == ["load", "native tokens", "native gate references", "main pool"]
    assert len(kb2["prompt_tokens"]) == probe.KB2_ROWS and "memory_at_error" in kb2
    assert json.loads((tmp_path / "kb2.json").read_text())["kb2"]["stage"] == "main pool"


def test_kb2d_records_every_path_per_token_on_cpu_stand_ins(tmp_path):
    """KB2-D on a tiny random 4-bit Gemma 3: all seven paths run on the chunks, one NLL per scored token, and the
    decode paths agree with the gate's own means (same code as KB2-0's gate)."""
    from ironmule import dense_gemma

    from tests.engine.test_dense_gemma import tiny_model

    model = tiny_model()

    class Tokenizer(_Tokenizer):
        bos_token_id = 2

    def make_plan(slots, capacity):
        return dense_gemma.GemmaDensePlan(model, slots=slots, capacity=capacity,
                                          attention=dense_gemma.reference_attention, native=False)

    report = probe.probe_kb2d_report("tiny", tmp_path / "kb2d.json", smoke=True, loaded=(model, Tokenizer(), {}),
                                     make_plan=make_plan)
    kb2d = report["kb2d"]
    assert "error" not in kb2d, kb2d
    assert kb2d["stage"] == "done" and kb2d["chunks"] == ["0", "1", "2", "3", "4"]
    assert set(kb2d["paths"]) == {"stock_oneshot", "stock_oneshot_bos", "native_oneshot", "native_decode",
                                  "native_decode_bos", "plan_decode", "plan_decode_bos"}
    items = [([], piece) for _, piece in probe.gate_items(Tokenizer(), 5, 12)["main"]]  # as run 2 scored them
    for name, per in kb2d["paths"].items():
        assert all(len(per[str(i)]) == len(items[i][1]) - 1 for i in range(5)), name
    gate_mean = probe.kb2_native_nll(model, *items[1])
    assert abs(sum(kb2d["paths"]["native_decode"]["1"]) / 12 - gate_mean) < 1e-6
    assert abs(probe.kb2_plan_nll(make_plan(1, 1024), [items[1]])[0]
               - sum(kb2d["paths"]["plan_decode"]["1"]) / 12) < 1e-6
    with_bos = probe.kb2_native_nll(model, [2], items[1][1])
    assert abs(sum(kb2d["paths"]["native_decode_bos"]["1"]) / 12 - with_bos) < 1e-6, "exactly one BOS"
    readings = report["analysis"]["readings"]
    assert set(readings["plan bug suspected"]) == {"1", "2", "4"}
    assert isinstance(readings["missing BOS is the trigger"], bool)
    assert json.loads((tmp_path / "kb2d.json").read_text())["schema"] == "ironmule.cuda_probe.kb2d.v1"


def test_kb2d_readings_follow_the_fixed_rule():
    def per(means):
        return {i: [m] * 4 for i, m in means.items()}

    near = {"0": 3.0, "1": 5.3, "2": 6.7, "3": 4.3, "4": 5.1}
    result = {"chunks": list(near), "paths": {
        "stock_oneshot": per(near), "native_oneshot": per(near), "native_decode": per(near),
        "plan_decode": per({**near, "1": 6.5, "2": 5.7}), "stock_oneshot_bos": per(near),
        "native_decode_bos": per(near), "plan_decode_bos": per(near)}}
    readings = probe.analyse_kb2d(result)["readings"]
    assert readings["plan bug suspected"] == {"1": True, "2": True, "4": False}
    assert readings["starts unstable for every path"] == {"1": False, "2": False, "4": False}
    assert readings["missing BOS is the trigger"] is True
    assert probe.analyse_kb2d(result)["plan leaves native at token"]["1"] == 0


def test_probe_gate_items_lead_with_bos_like_the_product_gate():
    """KB1/KB2's gate items follow `numeric_choice.gate_chunks`: unchanged for a tokenizer without a leading BOS,
    and for one with it every main chunk and long prefix starts with exactly one BOS, the same tokens scored."""
    from ironmule import numeric_choice

    class BosTok(_Tokenizer):
        bos_token_id = 2

        def encode(self, text):
            return [2] + [ord(c) % 200 + 3 for c in text]

    plain = probe.gate_items(_Tokenizer(), 4, 12, (48, 6, 4))
    ids = numeric_choice._text_ids(_Tokenizer())
    assert plain["main"] == [([], ids[i * 12:(i + 1) * 12 + 1]) for i in range(4)]
    assert plain["long"][1] == (ids[55:103], ids[103:110])
    led = probe.gate_items(BosTok(), 4, 12, (48, 6, 4))
    bos_ids = numeric_choice._text_ids(BosTok())
    assert led["main"] == numeric_choice.gate_chunks(BosTok(), 4, 12)
    assert led["long"][0] == (bos_ids[:48], bos_ids[48:55]), "the text's own BOS leads the first prefix"
    assert led["long"][1] == ([2] + bos_ids[55:103], bos_ids[103:110])
    assert all(prefix[0] == 2 and 2 not in prefix[1:] for prefix, _ in led["long"])
    h11 = probe.h11_gate_items(BosTok(), {"g": (20, 6, 3)})["g"]
    assert all(prefix[0] == 2 and 2 not in prefix[1:] for prefix, _ in h11)
    stride = max(6, (len(ids) - 26) // 2)  # the old windows, unchanged without a leading BOS
    assert probe.h11_gate_items(_Tokenizer(), {"g": (20, 6, 3)})["g"] == [
        (ids[i * stride:i * stride + 20], ids[i * stride + 20:i * stride + 26]) for i in range(3)]


def test_kb2g_runs_on_gate_2s_chunks_and_is_invalid_without_run_2s_plan(tmp_path):
    """KB2-G on CPU stand-ins: the gate runs on BOS-led chunks (chunk 0 keeps the text's own BOS), and the
    analysis refuses a plan that is not run 2's (a CPU stand-in has no kernel), whatever the gate reads."""
    from ironmule import dense_gemma
    from ironmule.numeric_choice import GATE_VERSION

    from tests.engine.test_dense_gemma import tiny_model

    model = tiny_model()

    class BosTok(_Tokenizer):
        bos_token_id = 2

        def encode(self, text):
            return [2] + [ord(c) % 200 + 3 for c in text]

    def make_plan(slots, capacity):
        return dense_gemma.GemmaDensePlan(model, slots=slots, capacity=capacity,
                                          attention=dense_gemma.reference_attention, native=False)

    report = probe.probe_kb2_report("tiny", tmp_path / "kb2g.json", smoke=True, loaded=(model, BosTok(), {}),
                                    make_plan=make_plan, schema="ironmule.cuda_probe.kb2g.v1",
                                    analyse=probe.analyse_kb2g)
    kb2, analysis = report["kb2"], report["analysis"]
    assert "error" not in kb2, kb2
    assert report["gate_version"] == GATE_VERSION == kb2["gate_version"] == analysis["gate_version"]
    assert kb2["gate_items"]["main_leads"] == [[], [2], [2], [2]]
    assert analysis["main chunks led by BOS"] == 3 and analysis["main chunks"] == 4
    assert set(analysis["(c) quality gate"]["paths"]) == {"main", "long"}
    assert not analysis["valid"] and analysis["verdict"].startswith("invalid")
    assert json.loads((tmp_path / "kb2g.json").read_text())["schema"] == "ironmule.cuda_probe.kb2g.v1"


def test_kb2g_decisions_read_the_fixed_rules():
    stack = {"identical_steps": 10, "steps": 10}
    rounds = [{"plan": {str(w): 0.05 + 0.004 * w for w in range(1, 9)}, "native_one": 0.047} for _ in range(5)]
    reference = [2.5 + 0.01 * i for i in range(16)]

    def result(plan_shift=0.0, kernel=probe.KB2_RUN2_KERNEL, identical=10, native_one=0.047):
        return {"install": {"kernel_sha256": kernel, "plan_version": 1},
                "gate_items": {"main_leads": [[]] + [[2]] * 15},
                "exact": {"stacks": {"3": stack, "8": {"identical_steps": identical, "steps": 10}}},
                "exact_long": {"stacks": {"4": stack}},
                "speed": {"rounds": [{**r, "native_one": native_one} for r in rounds]},
                "gate": {"main": {"reference": reference, "plan": [x + plan_shift for x in reference]},
                         "long": {"reference": reference[:4], "plan": [x + plan_shift for x in reference[:4]]}}}

    good = probe.analyse_kb2g(result())
    assert good["valid"] and good["(a) exact within the plan"] and good["(c) quality gate"]["passes"]
    assert good["verdict"].startswith("(a), (c) and run 2's (b) pass") and good["(b) limitation"] is False
    assert good["main chunks led by BOS"] == 15 and good["(b) decided by"]["width"] == 6
    assert probe.analyse_kb2g(result(plan_shift=0.02))["verdict"] == "does not pass: KB2 returns to Tier 0"
    assert probe.analyse_kb2g(result(identical=9))["verdict"] == "does not pass: KB2 returns to Tier 0"
    assert probe.analyse_kb2g(result(kernel="0" * 64))["verdict"].startswith("invalid")
    slow = probe.analyse_kb2g(result(native_one=0.01))  # this run's (b) is recorded, run 2's decides
    assert slow["(b) limitation"] is True and slow["verdict"].startswith("(a), (c) and run 2's (b) pass")


def test_kb2l_scores_sixteen_bos_led_windows_past_the_window_on_cpu_stand_ins(tmp_path):
    """KB2-L on CPU stand-ins: 16 windows, each prefix led by exactly one BOS, scored regions disjoint, the plan four
    rows stacked; the analysis reads the fixed rule (a CPU run checks the tool, not the plan)."""
    from ironmule import dense_gemma

    from tests.engine.test_dense_gemma import tiny_model

    model = tiny_model()

    class BosTok(_Tokenizer):
        bos_token_id = 2

        def encode(self, text):
            return [2] + [ord(c) % 200 + 3 for c in text]

    def make_plan(slots, capacity):
        return dense_gemma.GemmaDensePlan(model, slots=slots, capacity=capacity,
                                          attention=dense_gemma.reference_attention, native=False)

    report = probe.probe_kb2l_report("tiny", tmp_path / "kb2l.json", smoke=True, loaded=(model, BosTok(), {}),
                                     make_plan=make_plan)
    kb2l, analysis = report["kb2l"], report["analysis"]
    assert "error" not in kb2l, kb2l
    assert analysis["windows"] == 16 == analysis["windows led by BOS"] and len(analysis["differences"]) == 16
    items = probe.h11_gate_items(BosTok(), {"long": (24, 6, 16)})["long"]
    assert len(items) == 16 and all(prefix.count(2) == 1 for prefix, _ in items)
    assert set(analysis) >= {"(a) long pool exact", "(c) long path", "verdict"}
    assert json.loads((tmp_path / "kb2l.json").read_text())["schema"] == "ironmule.cuda_probe.kb2l.v1"


def test_kb2l_decision_reads_the_fixed_rule():
    reference = [2.4 + 0.01 * i for i in range(16)]
    windows = [{"led_by_bos": True}] * 16

    def result(shift=0.0, identical=10, count=16):
        return {"exact_long": {"stacks": {"4": {"identical_steps": identical, "steps": 10}}}, "windows": windows[:count],
                "gate": {"long": {"reference": reference[:count], "plan": [x + shift for x in reference[:count]]}}}

    assert probe.analyse_kb2l(result())["verdict"].startswith("passes")
    assert probe.analyse_kb2l(result(shift=0.02))["verdict"].startswith("does not pass")
    assert probe.analyse_kb2l(result(identical=9))["verdict"].startswith("does not pass")
    assert probe.analyse_kb2l(result(count=4))["verdict"].startswith("does not pass"), "16 windows, as fixed"


def test_kb2_decisions_read_the_fixed_rules():
    rounds = [{"plan": {str(w): 0.05 + 0.004 * w for w in range(1, 9)}, "native_one": 0.047} for _ in range(5)]
    stack = {"identical_steps": 10, "steps": 10}
    result = {"exact": {"stacks": {"3": stack, "8": stack}}, "exact_long": {"stacks": {"4": stack}},
              "speed": {"rounds": rounds}}
    analysis = probe.analyse_kb2(result)
    assert analysis["(a) exact within the plan"] is True
    b = analysis["(b) stacked step over single steps"]
    assert b["passes"] and abs(b["ratio"] - 0.082 / (8 * 0.047)) < 1e-9
    slow = [{"plan": {w: 0.05 * int(w) for w in r["plan"]}, "native_one": 0.047} for r in rounds]  # no lever
    assert not probe.analyse_kb2({**result, "speed": {"rounds": slow}})["(b) stacked step over single steps"]["passes"]
    cliff = [{"plan": {**r["plan"], "8": 0.354}, "native_one": 0.052} for r in rounds]  # the smoke's 8-row jump
    b = probe.analyse_kb2({**result, "speed": {"rounds": cliff}})["(b) stacked step over single steps"]
    assert b["width"] == 7 and b["passes"]
    broken = {**result, "exact": {"stacks": {"3": stack, "8": {"identical_steps": 9, "steps": 10}}}}
    assert probe.analyse_kb2(broken)["(a) exact within the plan"] is False
    assert probe.analyse_kb2(result)["verdict"] == "does not pass as a whole"  # no gate yet


def test_ag4_runs_the_products_own_choice_and_records_it(tmp_path, monkeypatch):
    import importlib

    import huggingface_hub

    from ironmule import plan_choice

    tune = importlib.import_module("ironmule.tune")  # `ironmule.tune` the package attribute is the function

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path / "abc123"))
    monkeypatch.setattr(tune, "resolve_local_model", lambda model_id, revision=None: type(
        "R", (), {"identity": type("I", (), {"identity_sha256": "f" * 64})()})())
    seen = {}

    def choose(model_id, fingerprint, identity, log):
        seen.update(model=model_id, fingerprint=fingerprint, identity=identity)
        log("agent: stacked engine: 2.1x")
        return {"engine": "dense", "compute_dtype": None, "reason": "2.1x"}

    monkeypatch.setattr(plan_choice, "choose_setup", choose)
    report = probe.probe_ag4_report("m", tmp_path / "ag4.json")
    assert "error" not in report and report["setup"]["engine"] == "dense" and report["revision"] == "abc123"
    assert seen["identity"] == "f" * 64 and seen["fingerprint"].startswith("kaggle-")
    assert report["log"] == ["agent: stacked engine: 2.1x"]
    assert json.loads((tmp_path / "ag4.json").read_text())["schema"] == "ironmule.cuda_probe.ag4.v1"


# ---------------------------------------------------------------- KB3-0

class _ChatTok:
    """The tiny model's stand-in tokenizer, with the chat template `chat_ids` reads."""

    eos_token_id = 255

    def encode(self, text):
        return [ord(c) % 200 + 1 for c in text]

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return self.encode(messages[-1]["content"])


def test_kb3_compares_version_3_with_version_1_on_a_cpu_stand_in(tmp_path):
    """The probe's plumbing on a tiny 4-bit Gemma 3 with FA2's semantics in plain MLX for both versions (the CUDA
    kernels need the card): random pools, decode from the same prefills with each version, timing alternated."""
    from ironmule import dense_gemma

    from tests.engine.test_dense_gemma import tiny_model
    from tools import cuda_probe as cp

    model = tiny_model()
    plan = dense_gemma.GemmaDensePlan(model, slots=4, capacity=96, attention=dense_gemma.reference_attention,
                                      native=False)
    seen = set()

    def fa2(*args, version):
        seen.add(version)
        return dense_gemma.reference_attention(*args)

    cases = [{"G": 1, "D": plan.dim, "B": 2, "W": 0, "lengths": [5, 64]},
             {"G": 1, "D": plan.dim, "B": 3, "W": 16, "lengths": [1, 40, 96]}]
    out = cp.probe_kb3(model, _ChatTok(), None, steps=3, rounds=2, speed_steps=3, widths=(1, 2), cases=cases,
                       make_plan=lambda: plan, fa2=fa2, texts={"short": "hi there", "long": "x" * 30})
    assert seen == {1, 3} and out["candidate"] == 3
    assert [c["identical"] for c in out["random"]] == [True, True]
    assert set(out["decode"]) == {"1", "2"} and set(out["decode"]["2"]) == {"1", "3"}
    assert out["decode"]["2"]["1"] == out["decode"]["2"]["3"] and len(out["decode"]["2"]["1"]["hashes"]) == 3
    assert len(out["base"]["short"]) == 2 and len(out["plan"]["3"]["long"]) == 2
    analysis = cp.analyse_kb3(out)
    assert analysis["(a) random pools identical"] and analysis["(a) decode identical"]
    assert analysis["candidate"] == 3 and set(analysis["(b) one-request ratio"]) == {"1", "3"}
    folder = cp.write_kernel(tmp_path / "k", "owner", "d" * 40, "kb31")
    assert "'--plan', 'kb31'" in (folder / "run.py").read_text()
    assert '"id": "owner/ironmule-cuda-probe-kb31"' in (folder / "kernel-metadata.json").read_text()


def test_kb3_rejects_a_candidate_on_any_differing_bit_and_needs_its_speed_margin():
    from tools import cuda_probe as cp

    step = {"1": 0.05, "6": 0.13}
    same = {"hashes": ["a"], "tokens": [[1]]}
    result = {"random": [{"identical": True}], "decode": {"1": {"1": same, "2": same}},
              "base": {"short": [0.048], "long": [0.048]},
              "plan": {"1": {"short": [step], "long": [step]},
                       "2": {"short": [{"1": 0.0505, "6": 0.13}], "long": [{"1": 0.0505, "6": 0.13}]}}}
    assert cp.analyse_kb3(result)["verdict"].startswith("v2 becomes the plan's kernel")  # 0.95 >= 0.92
    slow = {**result, "plan": {**result["plan"], "2": {"short": [{"1": 0.056, "6": 0.13}],
                                                       "long": [{"1": 0.056, "6": 0.13}]}}}
    assert cp.analyse_kb3(slow)["verdict"].startswith("v2 is bit-identical but not fast enough")  # 0.857
    differs = {**result, "decode": {"1": {"1": same, "2": {"hashes": ["b"], "tokens": [[1]]}}}}
    assert cp.analyse_kb3(differs)["verdict"].startswith("v2 is rejected")
    wide_loss = {**result, "plan": {**result["plan"], "2": {"short": [{"1": 0.0505, "6": 0.14}],
                                                            "long": [{"1": 0.0505, "6": 0.14}]}}}
    assert not cp.analyse_kb3(wide_loss)["passes"], "the best width may not lose more than 2 %"
