#!/usr/bin/env bash
# Prepare a fresh RunPod pod (runpod/pytorch CUDA 12.8 devel image) for the Qwen3-32B IronMule/vLLM comparison.
# Usage on the pod: bash /work/IronMule/lab/rp/setup.sh ARCH   (ARCH = CUDA compute capability, e.g. 80 for A100)
# VLLM=0 skips vLLM. Lessons from the first A100 pod: `pip download --no-binary :all:` builds CMake from source
# (single-threaded, minutes), so the llama-cpp-python sdist is fetched directly; vLLM 0.11.0 needs transformers 4.57;
# IronMule's package import needs an MLX backend; the product state must be initialised before `ironmule serve`.
set -euo pipefail
ARCH=${1:-80}
JOBS=${JOBS:-16}
cd /work/IronMule
export PIP_ROOT_USER_ACTION=ignore
mkdir -p /models /work/logs
pip install -q cmake ninja > /work/logs/cmake.log 2>&1

# 1. downloads in the background (verified against pinned SHA-256)
( python lab/portable_gpu/fetch.py qwen3-32b --dir /models > /work/logs/fetch_gguf.log 2>&1; echo "rc=$?" >> /work/logs/fetch_gguf.log ) &
( python lab/portable_gpu/fetch.py qwen3-06b --dir /models > /work/logs/fetch_draft.log 2>&1; echo "rc=$?" >> /work/logs/fetch_draft.log ) &
( python -m ironmule_product.compare_tokenizer qwen3-32b --dir /models/tokenizers > /work/logs/fetch_tok.log 2>&1; echo "rc=$?" >> /work/logs/fetch_tok.log ) &
if [ "${VLLM:-1}" = 1 ]; then
  ( python lab/portable_gpu/fetch_awq.py /models/qwen3-32b-awq > /work/logs/fetch_awq.log 2>&1; echo "rc=$?" >> /work/logs/fetch_awq.log ) &
  ( python -m venv /opt/vllm && /opt/vllm/bin/pip install -q vllm==0.11.0 transformers==4.57.1 > /work/logs/vllm.log 2>&1; echo "rc=$?" >> /work/logs/vllm.log ) &
fi

# 2. IronMule, an MLX backend for its imports, and its ggml build for this GPU
pip install -q -e ".[dev]" "mlx[cpu]>=0.32,<0.33" numpy jinja2 typing-extensions diskcache > /work/logs/pip.log 2>&1
d=~/.cache/ironmule/ggml/ironmule/cuda
mkdir -p $d && cd $d
url=$(python -c "import json,urllib.request;d=json.load(urllib.request.urlopen('https://pypi.org/pypi/llama_cpp_python/0.3.16/json'));print([u['url'] for u in d['urls'] if u['packagetype']=='sdist'][0])")
[ -d llama_cpp_python-0.3.16 ] || (curl -sSL "$url" -o llama_cpp_python-0.3.16.tar.gz && tar xzf llama_cpp_python-0.3.16.tar.gz)
cd /work/IronMule
python tools/build_ggml.py --cuda "$ARCH" --jobs "$JOBS" > /work/logs/build.log 2>&1
tree=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-*/vendor/llama.cpp)
cmake --build "$tree/build" --target llama-bench -j "$JOBS" >> /work/logs/build.log 2>&1
python -c "from pathlib import Path; from ironmule_product.state import ProductStore; ProductStore(Path('/work/state')).setup('server')"
wait
echo SETUP-DONE
