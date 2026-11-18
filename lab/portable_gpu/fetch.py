"""Fetch one pinned public GGUF into a local cache, checking its size and SHA-256."""
import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

# Immutable Hugging Face revisions and LFS digests from their model APIs.
MODELS = {
    "smol": ("bartowski/SmolLM2-135M-Instruct-GGUF", "09816acd5d99df7be770d85ea30822623dab342c",
             "SmolLM2-135M-Instruct-Q4_K_M.gguf", 105454432,
             "2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d"),
    "qwen-small": ("bartowski/Qwen2.5-0.5B-Instruct-GGUF", "41ba88dbac95fed2528c92514c131d73eb5a174b",
                   "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", 397808192,
                   "6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653"),
    "tinyllama": ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "52e7645ba7c309695bec7ac98f4f005b139cf465",
                  "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", 668788096,
                  "9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0"),
    "qwen3-4b": ("Qwen/Qwen3-4B-GGUF", "bc640142c66e1fdd12af0bd68f40445458f3869b",
                 "Qwen3-4B-Q4_K_M.gguf", 2497280256,
                 "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name, dest):
    repo, revision, filename, size, sha = MODELS[name]
    dest.mkdir(exist_ok=True, parents=True)
    path = dest / filename
    if path.exists():
        if path.stat().st_size != size or digest(path) != sha:
            raise ValueError(f"existing {path} differs from the pinned GGUF")
        return path
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    with tempfile.NamedTemporaryFile(dir=dest, prefix=".gguf-", delete=False) as partial:
        try:
            with urllib.request.urlopen(url, timeout=30) as stream:
                shutil.copyfileobj(stream, partial, length=1 << 20)
            partial.close()
            temp = Path(partial.name)
            if temp.stat().st_size != size or digest(temp) != sha:
                raise ValueError(f"download differs from pinned SHA-256 for {name}")
            os.replace(temp, path)
        finally:
            Path(partial.name).unlink(missing_ok=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    print(fetch(args.model, args.dir))
