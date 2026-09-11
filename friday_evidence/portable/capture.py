"""Bounded Gemma-1B tensor captures from public DATA1 workload text.

Run only through the local supervisor. Dense FP32 replay is a new operation:
it does not inherit a quantized-inference correctness or speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import time
import uuid

from ..budget import BudgetGuard
from ..canonical import canonical_sha256
from ..registry import BudgetPolicy
from .contracts import code_digest, file_sha256, load_json, validate_case, write_json_new

MODEL_ID = "mlx-community/gemma-3-1b-it-4bit"
REVISION = "2d44e83dc9e80843d22fb941d3d699a0b1351aa6"
# Repository-authored, public workload descriptions; never live user content.
PUBLIC_WORKLOADS = {
    "train": "Explain how a library lends books, records returns, and sorts new books onto shelves. ",
    "validation": "Describe how a garden changes through spring, summer, autumn, and winter. ",
    "holdout": "Explain the route of a parcel from a small workshop to its destination. ",
}
_LAYER_BASES = {"train": 0, "validation": 4, "holdout": 8}
_PROJECTIONS = ("q_proj", "k_proj", "gate_proj", "q_proj")


def capture(model_spec: dict, partition: str, output_dir: Path, *, limit: int = 4,
            case_index: int = 0) -> dict:
    if model_spec.get("model_id") != MODEL_ID or model_spec.get("revision") != REVISION:
        raise ValueError("capture_requires_frozen_gemma_1b")
    if partition not in PUBLIC_WORKLOADS or type(limit) is not int or not 1 <= limit <= 4:
        raise ValueError("invalid_capture_scope")
    if type(case_index) is not int or case_index < 0 or case_index + limit > 4:
        raise ValueError("invalid_capture_case_index")
    output_dir = Path(output_dir)
    if output_dir.is_symlink() or any(parent.is_symlink() for parent in output_dir.parents):
        raise ValueError("capture_output_symlink")
    if any(output_dir.iterdir()):
        raise ValueError("capture_output_must_be_empty")
    from friday_evidence.identity import runtime_identity, assert_model_unchanged
    identity = runtime_identity(model_spec)
    started = time.monotonic()
    wall_limit = 90 * limit + 60 * (limit - 1) + 30
    deadline = started + wall_limit
    guard = BudgetGuard(BudgetPolicy(wall_limit_s=wall_limit))
    from ironmule_product.worker import _load
    model, tokenizer, _, _ = _load(model_spec)
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.gemma3_text import clip_residual

    case_deadline = deadline
    def evaluate(array):
        if time.monotonic() >= min(deadline, case_deadline):
            raise TimeoutError("capture_deadline")
        before = time.monotonic()
        mx.eval(array)
        mx.synchronize()
        guard.record_gpu(time.monotonic() - before)
        return array

    session = uuid.uuid4().hex
    cases = []
    for index in range(case_index, case_index + limit):
        repetitions = (2, 8, 16, 24)[index]
        layer_index = _LAYER_BASES[partition] + index
        projection_name = _PROJECTIONS[index]
        guard.before_candidate()
        case_deadline = min(deadline, time.monotonic() + 90)
        # The workload is intentionally public and limited to 512 tokens.
        workload = PUBLIC_WORKLOADS[partition] * repetitions
        ids = tokenizer.encode(workload)
        if not 1 <= len(ids) <= 512:
            raise ValueError("capture_context_out_of_bounds")
        h = model.model.embed_tokens(mx.array([ids]))
        h = h * mx.array(model.args.hidden_size**0.5, mx.bfloat16).astype(h.dtype)
        evaluate(h)
        for number in range(layer_index + 1):
            layer = model.model.layers[number]
            is_global = number % model.args.sliding_window_pattern == model.args.sliding_window_pattern - 1
            mask = create_attention_mask(h, None, window_size=None if is_global else model.args.sliding_window)
            if number < layer_index:
                h = evaluate(layer(h, mask, None))
                guard.required_break()
                continue
            if projection_name == "gate_proj":
                attention = layer.self_attn(layer.input_layernorm(h), mask, None)
                h = evaluate(clip_residual(h, layer.post_attention_layernorm(attention)))
                a_native = evaluate(layer.pre_feedforward_layernorm(h))
                projection = layer.mlp.gate_proj
            else:
                a_native = evaluate(layer.input_layernorm(h))
                projection = getattr(layer.self_attn, projection_name)
        native_output = evaluate(projection(a_native))
        source_output_sha = hashlib.sha256(np.array(evaluate(native_output.astype(mx.float32))).tobytes()).hexdigest()
        guard.required_break()
        weight = evaluate(mx.dequantize(projection.weight, projection.scales, projection.biases,
                           group_size=projection.group_size, bits=projection.bits, mode=projection.mode))
        left = np.array(evaluate(a_native.astype(mx.float32))).reshape(-1, a_native.shape[-1])
        right = np.ascontiguousarray(np.array(evaluate(weight.astype(mx.float32))).T)
        case_id = f"{partition}-{session}-{index}"
        a_name, b_name = f"{case_id}-a.npy", f"{case_id}-b.npy"
        for name, array in ((a_name, left), (b_name, right)):
            descriptor = os.open(output_dir / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
        weight_sha = hashlib.sha256(right.tobytes()).hexdigest()
        case = validate_case({"case_id": case_id, "a_file": a_name, "b_file": b_name,
            "a_sha256": file_sha256(output_dir / a_name), "b_sha256": file_sha256(output_dir / b_name),
            "shape": [left.shape[0], left.shape[1], right.shape[1]], "dtype": "float32",
            "lineage_id": canonical_sha256({"a": hashlib.sha256(left.tobytes()).hexdigest(), "b": weight_sha}),
            "weight_sha256": weight_sha, "prompt_family": f"data1-{partition}-{index}",
            "shape_family": f"gemma1b-{projection_name}-m{left.shape[0]}", "capture_session": session,
            "model_id": MODEL_ID, "model_sha256": identity["model_sha256"],
            "source_kind": "real_model_capture", "partition": partition,
            "model_revision": REVISION, "layer": layer_index,
            "source_output_sha256": source_output_sha,
            "derivation": "native_quantized_projection_inputs_and_dequantized_weight_to_dense_fp32",
            "export_policy": "private_public_workload_only"})
        write_json_new(output_dir / f"{case_id}.json", case)
        cases.append(case)
        guard.finish_candidate()
    assert_model_unchanged(model_spec, identity)
    report = {"schema": "ironmule.capture.v1", "status": "captured", "capture_session": session,
        "partition": partition, "cases": cases, "environment_sha256": identity["environment_sha256"],
        "code_sha256": code_digest(), "budget": guard.summary(), "performance_claim": False}
    write_json_new(output_dir / "capture.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model-spec", type=Path, required=True)
    parser.add_argument("--partition", choices=tuple(PUBLIC_WORKLOADS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--case-index", type=int, default=0)
    args = parser.parse_args(argv)
    capture(load_json(args.model_spec), args.partition, args.output_dir, limit=args.limit, case_index=args.case_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
