#!/usr/bin/env python3
"""Capture real `K=3840` activations and weights from a Gemma 12B decode step.

Everything downstream must run on what the model actually produces, not on synthetic
normals. This performs one short real decode, records the input vector reaching every
quantised projection whose reduction dimension is 3840, and writes the tensors plus a
manifest of checksums.

The weights are not copied: they are addressed by model id, module path and a SHA-256
over their packed bytes, so a later run can prove it used the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
    "a": "Explain in three sentences why unified memory changes how a laptop runs a language model.",
    "b": "Describe how a key value cache grows during autoregressive decoding.",
    "c": "Name three properties of a bandwidth bound computation.",
    "d": "Write one paragraph about how a compiler schedules instructions.",
}
TARGET_K = 3840  # overridden by --target-k


def _digest(array: mx.array) -> str:
    return hashlib.sha256(bytes(memoryview(array))).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--steps", type=int, default=3, help="decode steps before capture")
    parser.add_argument("--layers", type=int, default=4, help="layers whose weights to record")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="a")
    parser.add_argument("--target-k", type=int, default=3840,
                        help="reduction length of the projections to capture")
    parser.add_argument("--out", type=Path, required=True, help="directory for tensors")
    args = parser.parse_args()

    global TARGET_K
    TARGET_K = args.target_k
    model, tokenizer = load(args.model)
    language = getattr(model, "language_model", model)

    captured: dict[str, mx.array] = {}
    originals: dict[int, object] = {}
    paths: dict[int, str] = {}

    def register(module: nn.Module, prefix: str = "") -> None:
        for name, child in module.children().items():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, list):
                for index, item in enumerate(child):
                    if isinstance(item, nn.Module):
                        register(item, f"{path}.{index}")
                continue
            if isinstance(child, nn.QuantizedLinear):
                columns = child.weight.shape[1] * 32 // child.bits
                if columns == TARGET_K:
                    paths[id(child)] = path
            elif isinstance(child, nn.Module):
                register(child, path)

    register(language)

    original_call = nn.QuantizedLinear.__call__

    decoding = {"active": False}

    def recording(self, x, *rest, **kw):
        path = paths.get(id(self))
        # Only single-token decode inputs: the prefill pass carries the whole prompt.
        if (
            decoding["active"]
            and path is not None
            and path not in captured
            and x.size == TARGET_K
        ):
            captured[path] = mx.array(x)
        return original_call(self, x, *rest, **kw)

    nn.QuantizedLinear.__call__ = recording
    try:
        cache = language.make_cache()
        ids = mx.array(tokenizer.encode(PROMPTS[args.prompt]))[None]
        tokens = []
        for step in range(args.steps):
            decoding["active"] = step > 0
            out = language.model(ids, cache=cache)
            logits = language.lm_head(out[:, -1, :])
            token = int(mx.argmax(logits, axis=-1).item())
            tokens.append(token)
            ids = mx.array([[token]])
    finally:
        nn.QuantizedLinear.__call__ = original_call

    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    kept = 0
    for path, vector in sorted(captured.items()):
        layer = path.split(".")[2] if path.startswith("model.layers.") else "other"
        if layer.isdigit() and int(layer) >= args.layers:
            continue
        vector = vector.reshape(1, -1)
        mx.eval(vector)
        name = path.replace(".", "_")
        mx.save(str(args.out / f"activation_{name}.npy"), vector)
        entries.append({
            "module_path": path,
            "kind": "activation",
            "shape": list(vector.shape),
            "dtype": str(vector.dtype).split(".")[-1],
            "sha256": _digest(vector),
            "file": f"activation_{name}.npy",
        })
        kept += 1

    weights = []
    for module_id, path in sorted(paths.items(), key=lambda item: item[1]):
        layer = path.split(".")[2] if path.startswith("model.layers.") else "other"
        if layer.isdigit() and int(layer) >= args.layers:
            continue
        # Resolve the path back to the module so the digest names real bytes.
        node = language
        for part in path.split("."):
            node = node[int(part)] if part.isdigit() else getattr(node, part)
        module = node
        weights.append({
            "module_path": path,
            "kind": "weight",
            "out_features": int(module.weight.shape[0]),
            "in_features": TARGET_K,
            "bits": int(module.bits),
            "group_size": int(module.group_size),
            "weight_sha256": _digest(module.weight),
            "scales_sha256": _digest(module.scales),
            "biases_sha256": _digest(module.biases),
            "weight_bytes": int(module.weight.nbytes),
            "scales_bytes": int(module.scales.nbytes),
            "biases_bytes": int(module.biases.nbytes),
        })

    manifest = {
        "schema": "ironmule.qmv_input_capture.v1",
        "experiment_id": args.out.name,
        "agent": "claude",
        "kind": "capture",
        "status": "captured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "real activations and weight identities for a K=3840 kernel study",
        "model": args.model,
        "prompt_key": args.prompt,
        "prompt_sha256": hashlib.sha256(PROMPTS[args.prompt].encode()).hexdigest(),
        "decode_steps": args.steps,
        "tokens": tokens,
        "target_k": TARGET_K,
        "activations": entries,
        "weights": weights,
        "total_weight_bytes": sum(
            entry["weight_bytes"] + entry["scales_bytes"] + entry["biases_bytes"]
            for entry in weights
        ),
        "mlx_version": mx.__version__,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "activations": kept,
        "weight_modules": len(weights),
        "total_weight_mb": manifest["total_weight_bytes"] / 1e6,
        "out": str(args.out),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
