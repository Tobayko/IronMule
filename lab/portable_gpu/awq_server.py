"""Launch pinned, native HF AWQ weights in vLLM for a practical cross-format comparison."""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
from pathlib import Path

from ironmule_product import compare_servers, compare_tokenizer

from . import fetch_awq


def check_snapshot(args):
    """Use the four shards verified by fetch_awq.verify and the exact Qwen3-32B base tokenizer."""
    receipt = args.awq_manifest
    if not fetch_awq.unchanged(args.awq_dir, receipt):
        raise ValueError("AWQ snapshot differs from the local verification receipt")
    tokenizer = Path(args.tokenizer)
    _, _, config_hash = compare_tokenizer.TOKENIZERS["qwen3-32b"]
    expected = {**compare_tokenizer._QWEN_FILES, "config.json": config_hash}
    if {p.name for p in tokenizer.iterdir()} != expected.keys():
        raise ValueError("native AWQ needs the pinned Qwen3-32B base tokenizer directory")
    for name, known in expected.items():
        with (tokenizer / name).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != known:
                raise ValueError(f"invalid Qwen3-32B tokenizer file: {name}")
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if getattr(args, "tokenizer_sha256", digest) != digest:
        raise ValueError("AWQ and GGUF tokenizer manifests differ")
    return {
        "weight_format": "awq", "awq_repo": fetch_awq.REPO, "awq_revision": fetch_awq.REVISION,
        "awq_manifest_sha256": receipt["manifest_sha256"], "awq_weight_bytes": receipt["shard_bytes"],
        "tokenizer_sha256": digest, "tokenizer_files": expected,
        "tokenizer_repo": fetch_awq.TOKENIZER_REPO, "tokenizer_revision": fetch_awq.TOKENIZER_COMMIT,
    }


def _command(engine, args, port, env, gpu):
    if engine != "vllm" or not gpu:
        raise ValueError("the native AWQ pilot requires a GPU")
    if getattr(args, "vllm_enforce_eager", False):
        raise ValueError("the native AWQ pilot requires the default vLLM execution mode")
    # The checkpoint's config.json declares AWQ. vLLM chooses AWQ or AWQMarlin on the actual device.
    return compare_servers.vllm_command(args, port, env, Path(args.awq_dir).resolve())


def _engine_config(log, model, tokenizer):
    resolved = set()
    with Path(log).open(errors="replace") as lines:
        for line in lines:
            if "Initializing a V1 LLM engine " not in line or f"with config: model={str(model)!r}, " not in line:
                continue
            if f", tokenizer={str(tokenizer)!r}, " not in line:
                continue
            method = re.search(r", quantization=(awq|awq_marlin),", line)
            dtype = re.search(r", dtype=torch\.(float16|bfloat16|float32),", line)
            if method is None or dtype is None:
                raise ValueError("vLLM resolved an unexpected quantization method or dtype")
            resolved.add((method[1], dtype[1]))
    if len(resolved) != 1:
        raise ValueError("missing or ambiguous vLLM engine configuration for the pinned AWQ model and tokenizer")
    return resolved.pop()


@contextlib.contextmanager
def open_awq_server(args, repeat, deadline):
    """Yield a ready native-AWQ vLLM server after checking its served model, backend and placement."""
    with compare_servers.open_prepared_server("vllm", args, repeat, deadline, prepare=check_snapshot, command=_command) as server:
        model = Path(args.awq_dir).resolve()
        if server["version"] != "0.11.0" or server["model_root"] != str(model):
            raise ValueError("vLLM version or advertised AWQ model root differs from the native pilot")
        if server["provider"] != "cuda" or server["placement_verified"] is not True:
            raise ValueError("native AWQ placement on the selected CUDA device is unverified")
        check_snapshot(args)
        quantization, dtype = _engine_config(server["log_file"], model, Path(args.tokenizer).resolve())
        server.update(loaded_quantization=quantization, loaded_dtype=dtype,
                      quantization_evidence="Resolved quantization in the ready vLLM V1 engine configuration",
                      quantization_selection="checkpoint configuration, vLLM automatic backend")
        yield server
