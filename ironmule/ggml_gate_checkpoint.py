"""Private, atomic checkpoints for individual completed GGUF qualification children."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys


def _digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fixture_signature(tokenizer, kg5):
    from .factual_gate import MAX_TOKENS, _chat_ids, _protocol

    return _digest([MAX_TOKENS, [
        [case.key, case.category, case.expected, case.accepted_forms, _chat_ids(tokenizer, case, kg5)]
        for case in _protocol(0, kg5)]])


def directory(cache, *, key, snapshot, gguf, text, flags):
    from .ggml_binary import identity as native_identity
    from .ggml_engine import file_sha256

    # Never write tokenized prompts or model content. The model's existing gate key includes its full SHA.
    native = native_identity().get("sha256")
    if not native or not isinstance(text.get("facts_sha256"), str) or len(text["facts_sha256"]) != 64:
        return None
    snapshot = Path(snapshot).resolve()
    sources = [Path(__file__).with_name(f) for f in
               ("ggml_engine.py", "ggml_gate_checkpoint.py", "gguf_tokenizer.py", "numeric_choice.py", "factual_gate.py",
                "pilot.py", "speculation.py")]
    files = []
    try:
        if snapshot != Path(gguf).resolve():
            if snapshot.is_file():
                files.append(snapshot)
            else:
                # Include custom vocabularies, templates and tokenizer code, including nested assets.
                # The GGUF supplies the weights; Hugging Face model weight shards are not tokenizer inputs.
                cache_root = cache.resolve()
                if cache_root == snapshot:
                    return None
                for p in snapshot.rglob("*"):
                    if (cache_root.is_relative_to(snapshot) and p.is_relative_to(cache_root)
                            or ".git" in p.relative_to(snapshot).parts):
                        continue
                    if p.is_symlink() and p.is_dir():
                        return None
                    if (p.is_file() and p.suffix not in (".gguf", ".safetensors", ".pth", ".pt")
                            and not (p.suffix == ".bin" and p.name.startswith(("pytorch_model", "model-")))):
                        files.append(p)
        for name in os.environ.get("LD_PRELOAD", "").replace(":", " ").split():
            lib = Path(name)
            if not lib.is_absolute() or not lib.is_file():
                return None
            files.append(lib)
        contents = [(str(p), file_sha256(str(p))) for p in sorted(set(files + sources))]
        host = next((p.read_text().strip() for p in (Path("/etc/machine-id"), Path("/proc/sys/kernel/random/boot_id"))
                     if p.is_file()), "")
        if not host:
            return None
        versions = {}
        for name in ("mlx-lm", "transformers", "tokenizers", "numpy", "llama-cpp-python"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
        env = {k: v for k, v in os.environ.items() if k.startswith(
            ("IRONMULE_GGML_", "IRONMULE_PILOT", "IRONMULE_KG", "GGML_", "OMP_", "CUDA_", "HIP_", "ROCR_"))}
        identity = _digest([1, key, native, contents, host, sys.version, versions, flags, text, env,
                            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None])
        cache.mkdir(parents=True, exist_ok=True)
        return cache / f"stages-{identity}"
    except (OSError, ValueError):
        return None


def stage(folder, label, run, counts, valid):
    """Lock, recheck, run and atomically save. Interrupted or failed children produce no reusable record."""
    from ironmule_product.errors import StateError
    from ironmule_product.state import _atomic_write, _locked, _read_json

    def measure():
        output, error = run()
        if output is not None and not error and valid(output):
            counts["measured"] += 1
        return output, error

    if folder is None:
        return measure()
    path = folder / (_digest(label) + ".json")
    with ExitStack() as stack:
        try:
            stack.enter_context(_locked(folder))
        except (StateError, OSError):
            return measure()
        try:
            record = _read_json(path)
            if (record["label"] == label and valid(record["output"])
                    and record["digest"] == _digest([folder.name, label, record["output"]])):
                counts["reused"] += 1
                return record["output"], ""
        except (StateError, OSError, KeyError, ValueError, TypeError):
            pass
        output, error = measure()
        if output is not None and not error and valid(output):
            try:
                _atomic_write(path, {"label": label, "output": output, "digest": _digest([folder.name, label, output])})
            except (StateError, OSError, ValueError):
                pass
        return output, error
