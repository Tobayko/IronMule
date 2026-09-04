"""Local-only Qwen hybrid-cache correctness gates.

This file never resolves a Hugging Face repository id.  Set IRONMULE_QWEN_MODEL
to an existing local snapshot and run with ``-m integration`` to opt in.  An
explicit but invalid path, config, or load is a test failure rather than a skip.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("IRONMULE_QWEN_MODEL"),
        reason="IRONMULE_QWEN_MODEL is not set; Qwen integration is local-only",
    ),
]

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

IRONMULE_MAX_TOKENS = 8
# The reference loop yields one prefill choice plus eight decode choices, matching
# IronMule's one-prefill-plus-eight semantics.
REFERENCE_OUTPUTS = IRONMULE_MAX_TOKENS + 1
PROMPTS = [
    "Name one advantage of a unified memory architecture.",
    "What does greedy decoding choose at each step?",
]


@pytest.fixture(scope="module")
def qwen_runtime():
    root = Path(os.environ["IRONMULE_QWEN_MODEL"])
    if not root.is_absolute() or not root.is_dir():
        pytest.fail("IRONMULE_QWEN_MODEL must be an existing absolute snapshot directory")
    if not (root / "config.json").is_file() or not list(root.glob("*.safetensors")):
        pytest.fail("Qwen snapshot needs config.json and local safetensors")
    try:
        config = json.loads((root / "config.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        pytest.fail(f"cannot inspect local Qwen config: {exc}")
    architectures = config.get("architectures", ())
    if not any("qwen3_5" in str(value).lower() for value in architectures):
        pytest.fail("IRONMULE_QWEN_MODEL is not a qwen3_5 snapshot")

    import ironmule

    try:
        runtime = ironmule.Runtime.load(
            model_id=str(root), use_tuned_profile=False,
        )
    except Exception as exc:  # pragma: no cover - depends on local hardware/model
        pytest.fail(f"local Qwen snapshot failed to load: {type(exc).__name__}: {exc}")
    return runtime, ironmule


def _reference_tokens(runtime, prompt_ids):
    """Unmodified one-shot MLX-LM greedy loop, with matching output count."""
    import mlx.core as mx

    model = runtime.engine.model
    cache = model.make_cache()
    logits = model(mx.array(prompt_ids)[None, :], cache=cache)
    token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    tokens = [token]
    for _ in range(REFERENCE_OUTPUTS - 1):
        if token in runtime.backend.eos_ids:
            break
        logits = model(mx.array([[token]]), cache=cache)
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        tokens.append(token)
    return tokens


def _stop_reason(tokens, eos_ids):
    return "eos" if any(token in eos_ids for token in tokens) else "length"


def _check_recurrent_shapes(rt, ironmule, prompt_ids):
    import mlx.core as mx
    from ironmule import runtime as ironmule_runtime

    plan = ironmule.StrictOneShotPlan()
    capacity = rt.backend.capacity_for([len(prompt_ids)], IRONMULE_MAX_TOKENS)
    state, token = rt.backend.prefill(prompt_ids, plan, capacity)
    recurrent = [
        tuple(array.shape)
        for layer in state["layers"] if "arrays" in layer
        for array in layer["arrays"]
    ]
    assert recurrent, "Qwen snapshot must expose recurrent ArraysCache layers"
    body = rt.engine._body(capacity, 1)
    previous_hash = rt.backend.kv_hash(state, len(prompt_ids))
    assert isinstance(previous_hash, str) and previous_hash
    for _ in range(2):
        output = body(mx.array([[token]]), state)
        mx.eval(output[0], *ironmule_runtime._leaves(output[1]))
        token = int(mx.argmax(output[0][:, -1, :], axis=-1).item())
        state = output[1]
        current = [
            tuple(array.shape)
            for layer in state["layers"] if "arrays" in layer
            for array in layer["arrays"]
        ]
        assert current == recurrent
        current_hash = rt.backend.kv_hash(
            state, int(state["position"]["offset"].item()))
        assert current_hash != previous_hash
        previous_hash = current_hash


def test_qwen_strict_matches_unmodified_greedy_and_hybrid_shapes(qwen_runtime):
    rt, ironmule = qwen_runtime
    prompt_ids = [rt.encode(prompt) for prompt in PROMPTS]
    references = [_reference_tokens(rt, ids) for ids in prompt_ids]
    for ids in prompt_ids:
        _check_recurrent_shapes(rt, ironmule, ids)

    rt.mode = ironmule.InteractiveMode()
    requests = [
        ironmule.Request(
            prompt_ids=ids,
            max_tokens=IRONMULE_MAX_TOKENS,
            plan=ironmule.StrictOneShotPlan(),
            rid=f"qwen{i}",
        )
        for i, ids in enumerate(prompt_ids)
    ]
    interactive = rt.serve(requests)
    for result, expected in zip(interactive, references):
        assert result.tokens == expected
        assert result.stop_reason == _stop_reason(expected, rt.backend.eos_ids)

    # Only compare grouped execution after the unmodified reference gate passes.
    rt.mode = ironmule.ThroughputMode()
    grouped = rt.serve(requests)
    for result, expected in zip(grouped, references):
        assert result.tokens == expected
        assert result.stop_reason == _stop_reason(expected, rt.backend.eos_ids)
    assert rt.telemetry.snapshot()["correctness_errors"] == 0
