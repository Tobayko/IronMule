"""HW3 and TB1's probe (tools/cuda_probe.py): every part runs end to end on a tiny random Qwen 3 on the CPU, and
the decisions read what the backlog fixed before data. A CPU run checks the tool, not a hardware claim."""

from __future__ import annotations

import json
import math

import pytest

from tools import cuda_probe as probe

mx = pytest.importorskip("mlx.core")


class _Tokenizer:
    eos_token_id = 255
    eos_token_ids = {255}

    def encode(self, text):
        return [ord(c) % 200 + 1 for c in text]

    def decode(self, ids):
        return "".join(chr(32 + i % 90) for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        return self.encode(messages[-1]["content"])


def _tiny():
    from mlx_lm.models import qwen3

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
