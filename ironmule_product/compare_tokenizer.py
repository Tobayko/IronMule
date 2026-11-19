"""Verify rendered chat prompt IDs for a local, jointly launched server comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.request
from pathlib import Path

# These two Qwen3 releases share the complete tokenizer and chat template.
_QWEN_FILES = {
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
TOKENIZERS = {
    "qwen3-32b": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137",
                  "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb"),
    "qwen3-06b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca",
                  "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"),
}


def fetch_hf_tokenizer(alias: str, dest: Path) -> Path:
    """Fetch just the five pinned files under dest/alias and validate cached copies."""
    repo, revision, config = TOKENIZERS[alias]
    files = {**_QWEN_FILES, "config.json": config}
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / alias
    if not path.exists():
        with tempfile.TemporaryDirectory(dir=dest) as temp:
            stage = Path(temp) / alias
            stage.mkdir()
            for name, expected in files.items():
                url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = response.read(16 << 20)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"unexpected contents for {repo}@{revision}/{name}")
                (stage / name).write_bytes(data)
            try:
                stage.rename(path)
            except OSError:
                if not path.is_dir():
                    raise
    if {p.name for p in path.iterdir()} != files.keys():
        raise ValueError(f"unexpected tokenizer directory contents in {path}")
    for name, expected in files.items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"unexpected tokenizer contents in {path / name}")
    return path


def _json(url, deadline, body=None):
    timeout = deadline - time.monotonic()
    if timeout <= 0:
        raise TimeoutError("tokenizer preflight exhausted the benchmark deadline")
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=min(30, timeout)) as response:
        return json.load(response)


def _ids(value):
    if not isinstance(value, list) or not value or any(type(x) is not int or x < 0 for x in value):
        raise ValueError("tokenizer returned invalid prompt IDs")
    return value


def rendered_prompt_ids(engine, server, messages, deadline, tok=None):
    """Read exact prompt IDs from a caller-verified serving configuration."""
    base, model = server["base_url"].rstrip("/"), server["model_id"]
    served = _json(base + "/v1/models", deadline)
    if not any(m["id"] == model for m in served["data"]):
        raise ValueError(f"server does not expose expected model {model}")
    unique = {}
    digest = hashlib.sha256()
    digests, lengths = [], []
    for conversation in messages:
        if time.monotonic() >= deadline:
            raise TimeoutError("tokenizer preflight exhausted the benchmark deadline")
        key = json.dumps(conversation, sort_keys=True, ensure_ascii=False)
        if key not in unique:
            if engine == "ironmule":
                ids = tok.apply_chat_template(conversation, add_generation_prompt=True, enable_thinking=False)
            elif engine == "llama_cpp":
                rendered = _json(base + "/apply-template", deadline, {"messages": conversation})["prompt"]
                ids = _json(base + "/tokenize", deadline,
                            {"content": rendered, "add_special": True, "parse_special": True})["tokens"]
            else:
                params = {"model": model, "messages": conversation, "add_generation_prompt": True,
                          "add_special_tokens": False, "chat_template_kwargs": {"enable_thinking": False}}
                result = _json(base + "/tokenize", deadline, params)
                ids = _ids(result["tokens"])
                if result.get("count") != len(ids):
                    raise ValueError("vLLM tokenizer count and IDs disagree")
            ids = _ids(ids)
            encoded = json.dumps(ids, separators=(",", ":")).encode()
            unique[key] = (len(ids), hashlib.sha256(encoded).digest())
        count, row_digest = unique[key]
        digest.update(row_digest)
        lengths.append(count)
        digests.append(row_digest.hex())
    return {"verified": True, "digest": digest.hexdigest(), "prompt_ids_sha256": digests,
            "prompt_tokens": lengths, "reason": None}


def preflight(engine, server, args, messages, deadline):
    """Derive IDs for every timed conversation using the active arm's own template and tokenizer.

    The comparative runner must also require equal digests, GGUF identities, and matching
    generation settings across arms before a timing comparison is valid.
    """
    tok = None
    try:
        if not messages:
            raise ValueError("no benchmark conversations supplied")
        if engine not in ("ironmule", "llama_cpp", "vllm"):
            raise ValueError(f"unsupported comparison engine {engine}")
        if not re.fullmatch("[0-9a-f]{64}", server.get("gguf_sha256", "")):
            raise ValueError("missing GGUF identity from local server launcher")
        argv = [os.fspath(v) for v in server["argv"]]
        if str(Path(args.gguf).resolve()) not in argv:
            raise ValueError("launcher did not bind the local GGUF to this server")
        if engine == "ironmule":
            if "--no-thinking" not in argv:
                raise ValueError("local worker must render with thinking disabled")
            from ironmule.gguf_tokenizer import GgufTokenizer

            tok = GgufTokenizer(str(args.gguf))
        if engine == "vllm" and (not args.tokenizer or "--tokenizer" not in argv or
                                Path(argv[argv.index("--tokenizer") + 1]).resolve() != Path(args.tokenizer).resolve()):
            raise ValueError("launcher did not bind the local tokenizer to vLLM")
        return rendered_prompt_ids(engine, server, messages, deadline, tok)
    except Exception as exc:  # noqa: BLE001 - an unavailable arm cannot qualify a comparison
        return {"verified": False, "digest": None, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        if tok is not None:
            tok.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch the pinned Hugging Face tokenizer without weights.")
    parser.add_argument("model", choices=TOKENIZERS)
    parser.add_argument("--dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(fetch_hf_tokenizer(arguments.model, arguments.dir))
