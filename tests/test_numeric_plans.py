"""Every cell of the numeric-plan table must still be what its run says.

The table is the only thing standing between a caller and a plan that doubles their
model's perplexity, so it is pinned the way the README is: each number is re-derived from
the committed run, in both directions. A re-measurement therefore moves the table or fails
here; it cannot quietly disagree with the evidence it cites.
"""

from __future__ import annotations

import json
import math
import random
import statistics as st
from pathlib import Path

import pytest

from ironmule.numeric_plans import (CUDA_PRE_AMPERE, MEASUREMENTS, QUALITY_BOUND,
                                    PlanRefused, architecture_of, check, device_class,
                                    measurements_for, recommend)

ROOT = Path(__file__).resolve().parents[1]

#: The paired gates were computed across two processes, because the two precisions do not
#: fit on one card together. Same seed and draw count as the analysis that produced them.
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_DRAWS = 10000
#: (architecture, plan) -> (bf16 rows file, other rows file, other key)
PAIRED = {
    ("mlx_lm.models.gemma3_text", "float16"): (
        "port2-run6-59ce8efc/quality16-gemma3-4b-bf16.json",
        "port2-run6-59ce8efc/quality16-gemma3-4b-float16.json", "nll_float16"),
    ("mlx_lm.models.qwen3", "float16"): (
        "port2-run6-59ce8efc/quality16-qwen3-8b-bf16.json",
        "port2-run6-59ce8efc/quality16-qwen3-8b-float16.json", "nll_float16"),
    ("mlx_lm.models.gpt_oss", "float16"): (
        "port2-run7-cddae1f9/quality-gptoss-20b-bf16-16.json",
        "port2-run7-cddae1f9/quality-gptoss-20b-float16-16.json", "nll_float16"),
    ("mlx_lm.models.gpt_oss", "float32"): (
        "port2-run7-cddae1f9/quality-gptoss-20b-bf16-16.json",
        "port2-run7-cddae1f9/quality-gptoss-20b-float32-24.json", "nll_float32"),
}
RESULTS = ROOT / "experiments" / "kaggle_compat" / "results"


def _paired_gate(bf16_file: str, other_file: str, other_key: str):
    bf16 = json.loads((RESULTS / bf16_file).read_text())["rows"]
    other = json.loads((RESULTS / other_file).read_text())["rows"][:len(bf16)]
    rows = [(a["nll_bf16"], b[other_key]) for a, b in zip(bf16, other)]
    rng = random.Random(BOOTSTRAP_SEED)

    def ratio(sample):
        return math.exp(st.mean(b for _, b in sample) - st.mean(a for a, _ in sample))

    draws = sorted(ratio([rng.choice(rows) for _ in rows]) for _ in range(BOOTSTRAP_DRAWS))
    return ratio(rows), (draws[250], draws[9750])


@pytest.mark.parametrize("row", MEASUREMENTS,
                         ids=[f"{row.label}-{row.plan}" for row in MEASUREMENTS])
def test_every_wall_ratio_is_the_one_its_run_recorded(row):
    payload = json.loads((ROOT / row.wall_evidence).read_text())
    assert payload["summary"][row.wall_arm]["median_ratio"] == row.wall_ratio, row.label
    assert payload["summary"][row.wall_arm]["failed_processes"] == 0, "a failed arm is not a ratio"


@pytest.mark.parametrize("row", [row for row in MEASUREMENTS if row.quality_known],
                         ids=[f"{row.label}-{row.plan}" for row in MEASUREMENTS
                              if row.quality_known])
def test_every_quality_interval_is_the_one_its_run_supports(row):
    key = (row.architecture, row.plan)
    if key in PAIRED:
        ratio, interval = _paired_gate(*PAIRED[key])
    else:
        payload = json.loads((ROOT / row.quality_evidence[0]).read_text())
        ratio, interval = payload[row.quality_evidence[1]], tuple(payload["ppl_ratio_ci"])
    assert ratio == row.quality_ratio, row.label
    assert interval == row.quality_interval, row.label


def test_a_measured_ruin_is_refused_and_a_wide_interval_is_not():
    """The distinction the whole table turns on, asserted rather than assumed.

    Gemma 3 in float32 measured 1.017846 with an interval that contains 1 — inconclusive on
    16 chunks of a text whose perplexity is 100, and the plan PORT1 shipped by a different
    route. Gemma 3 in float16 measured an interval starting at 1.87. Only the second is a
    refusal; treating the first as one would withdraw a plan the README documents at +91%.
    """
    gemma = "mlx_lm.models.gemma3_text"
    check(gemma, "float32", CUDA_PRE_AMPERE)
    with pytest.raises(PlanRefused, match="2.043792"):
        check(gemma, "float16", CUDA_PRE_AMPERE)
    # Unmeasured architectures and other device classes are not this guard's business.
    check("mlx_lm.models.gemma4_text", "float16", CUDA_PRE_AMPERE)
    check(gemma, "float16", None)
    check(gemma, None, CUDA_PRE_AMPERE)


def test_only_a_faster_and_qualified_plan_is_ever_recommended():
    for architecture in {row.architecture for row in MEASUREMENTS}:
        plan, reason = recommend(architecture, CUDA_PRE_AMPERE)
        assert reason, architecture
        if plan is None:
            continue
        row = next(r for r in measurements_for(architecture, CUDA_PRE_AMPERE) if r.plan == plan)
        assert row.wall_ratio < 1.0, f"{architecture}: recommended a plan that is slower"
        assert row.quality_interval[1] < QUALITY_BOUND, f"{architecture}: recommended past the bound"
    # The one architecture that earns a recommendation, and the fastest of its two plans.
    assert recommend("mlx_lm.models.qwen3", CUDA_PRE_AMPERE)[0] == "float16"
    assert recommend("mlx_lm.models.llama", CUDA_PRE_AMPERE)[0] is None
    assert "no numeric plan has been measured" in recommend("nobody.measured.this", CUDA_PRE_AMPERE)[1]


def test_device_class_names_only_what_changes_the_answer():
    assert device_class({"compute_capability_major": 7}) == CUDA_PRE_AMPERE
    assert device_class({"compute_capability_major": 8}) is None, "Ampere has native bf16"
    assert device_class({"compute_capability_major": True}) is None, "a bool is not a capability"
    assert device_class({}) is None and device_class(None) is None


def test_architecture_comes_from_the_module_mlx_lm_actually_runs():
    """Not from the config, which is what the publisher wrote rather than what runs."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

    mx.set_default_device(mx.cpu)
    model = Qwen3Model(ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=2,
                                 intermediate_size=128, num_attention_heads=4,
                                 num_key_value_heads=2, head_dim=16, vocab_size=128,
                                 rms_norm_eps=1e-5, max_position_embeddings=256,
                                 rope_theta=10000.0, tie_word_embeddings=True))
    nn.quantize(model, group_size=32, bits=4)
    assert architecture_of(model) == "mlx_lm.models.qwen3"
    assert architecture_of(object()) is None
