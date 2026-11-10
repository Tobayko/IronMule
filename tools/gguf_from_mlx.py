"""GG1: a Q8_0 GGUF built from the registered MLX model's own weights, so the ggml engine's gate measures only Q8_0's
rounding (BACKLOG GG, amendment of 2026-11-10).

The registered model is loaded with mlx-lm and widened to float32 as `dequantize` does; each of its tensors is
quantised to the template GGUF's type for that tensor (Q8_0 or F32, through llama.cpp's own `gguf-py`) and written in
place of the template's data, so names, metadata, tokenizer and file layout are the template's. Query and key weights
are permuted to llama.cpp's rotary layout as its `convert_hf_to_gguf.py` does. The tool checks every tensor against the
template's (two quantisations of one original agree closely; a wrong layout does not) and refuses otherwise.
Llama-architecture models only (`general.architecture` = llama).

    PYTHONPATH=SRC/vendor/llama.cpp/gguf-py python tools/gguf_from_mlx.py --snapshot DIR --template T.gguf --out M.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

MIN_CORRELATION = 0.99


def permute(weights: Any, n_head: int, n_head_kv: int | None = None) -> Any:
    """llama.cpp's rotary layout for query and key rows (`convert_hf_to_gguf.LlamaModel.permute`)."""
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2).reshape(weights.shape))


def gguf_name(name: str) -> str | None:
    """The GGUF tensor that holds an mlx-lm (Hugging Face) llama parameter, or None."""
    fixed = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
             "lm_head.weight": "output.weight"}
    if name in fixed:
        return fixed[name]
    parts = name.split(".")
    if len(parts) not in (5, 6) or parts[0] != "model" or parts[1] != "layers":
        return None
    norms = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm"}
    if len(parts) == 5:
        return f"blk.{parts[2]}.{norms[parts[3]]}.weight" if parts[3] in norms and parts[4] == "weight" else None
    layer, group, leaf = parts[2], parts[3], parts[4]
    table = {("self_attn", "q_proj"): "attn_q", ("self_attn", "k_proj"): "attn_k", ("self_attn", "v_proj"): "attn_v",
             ("self_attn", "o_proj"): "attn_output", ("mlp", "gate_proj"): "ffn_gate", ("mlp", "up_proj"): "ffn_up",
             ("mlp", "down_proj"): "ffn_down"}
    if (group, leaf) in table and parts[5] == "weight":
        return f"blk.{layer}.{table[group, leaf]}.weight"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    import gguf
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load

    from ironmule.numeric_plans import dequantize_model

    mx.set_default_device(mx.cpu)
    reader = gguf.GGUFReader(args.template)
    arch = bytes(reader.fields["general.architecture"].parts[-1]).decode()
    if arch != "llama":
        raise SystemExit(f"only llama GGUFs are rebuilt here, not {arch}")
    n_head = int(reader.fields["llama.attention.head_count"].parts[-1][0])
    n_head_kv = int(reader.fields["llama.attention.head_count_kv"].parts[-1][0])
    model, _ = load(args.snapshot)
    dequantize_model(model)
    params = {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())}
    by_gguf = {}
    for name, value in params.items():
        target = gguf_name(name)
        if target is None:
            continue
        if target.endswith("attn_q.weight"):
            value = permute(value, n_head, n_head)
        elif target.endswith("attn_k.weight"):
            value = permute(value, n_head, n_head_kv)
        by_gguf[target] = value
    tensors = {t.name: t for t in reader.tensors}
    missing = sorted(set(tensors) - set(by_gguf))
    if missing:
        raise SystemExit(f"the model has no weights for {missing[:5]}")
    shutil.copyfile(args.template, args.out)
    checks = []
    with open(args.out, "r+b") as handle:
        for name, tensor in tensors.items():
            value = by_gguf[name]
            template = gguf.quants.dequantize(tensor.data, tensor.tensor_type).reshape(value.shape)
            corr = float(np.corrcoef(template.ravel(), value.ravel())[0, 1]) if value.size > 1 else 1.0
            checks.append(corr)
            if not corr >= MIN_CORRELATION:
                raise SystemExit(f"{name} does not match the template's layout (correlation {corr:.4f})")
            if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
                data = value.astype(np.float32).tobytes()
            else:
                data = gguf.quants.quantize(value, tensor.tensor_type).tobytes()
            if len(data) != tensor.n_bytes:
                raise SystemExit(f"{name}: {len(data)} bytes against the template's {tensor.n_bytes}")
            handle.seek(tensor.data_offset)
            handle.write(data)
    digest = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    print(json.dumps({"out": args.out, "sha256": digest, "tensors": len(tensors), "min_correlation": min(checks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
