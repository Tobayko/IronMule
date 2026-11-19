#!/usr/bin/env bash
# Prepare a fresh RunPod pod (runpod/pytorch CUDA 12.8 devel image) for the Qwen3-32B IronMule/vLLM comparison.
# Usage on the pod: bash /work/IronMule/lab/rp/setup.sh ARCH   (ARCH = CUDA compute capability, e.g. 80 for A100)
set -euo pipefail
ARCH=${1:-80}
JOBS=${JOBS:-16}
cd /work/IronMule
export PIP_ROOT_USER_ACTION=ignore
mkdir -p /models /work/logs
pip install -q cmake ninja > /work/logs/cmake.log 2>&1

# 1. downloads in the background (verified against pinned SHA-256)
( python lab/portable_gpu/fetch.py qwen3-32b --dir /models > /work/logs/fetch_gguf.log 2>&1; echo "rc=$?" >> /work/logs/fetch_gguf.log ) &
( python lab/portable_gpu/fetch_awq.py /models/qwen3-32b-awq > /work/logs/fetch_awq.log 2>&1; echo "rc=$?" >> /work/logs/fetch_awq.log ) &
( python -m ironmule_product.compare_tokenizer qwen3-32b --dir /models/tokenizers > /work/logs/fetch_tok.log 2>&1; echo "rc=$?" >> /work/logs/fetch_tok.log ) &

# 2. vLLM 0.11.0 in its own venv
( python -m venv /opt/vllm && /opt/vllm/bin/pip install -q vllm==0.11.0 > /work/logs/vllm.log 2>&1; echo "rc=$?" >> /work/logs/vllm.log ) &

# 3. IronMule + its ggml build for this GPU
pip install -q -e ".[dev]" > /work/logs/pip.log 2>&1
pip install -q "numpy" "jinja2" "typing-extensions" "diskcache" >> /work/logs/pip.log 2>&1
python tools/build_ggml.py --cuda "$ARCH" --jobs "$JOBS" > /work/logs/build.log 2>&1
python tools/build_ggml.py --cuda "$ARCH" --jobs "$JOBS" --no-patches > /work/logs/build_stock.log 2>&1
for t in ironmule stock; do
  tree=$(ls -d ~/.cache/ironmule/ggml/$t/cuda/llama_cpp_python-*/vendor/llama.cpp)
  cmake --build "$tree/build" --target llama-bench -j "$JOBS" >> /work/logs/build_bench.log 2>&1
done
wait
echo SETUP-DONE
