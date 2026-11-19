#!/usr/bin/env bash
# Pod-side setup for the Qwen3 GGUF vs vLLM AWQ comparison. Runs from /work after the repo tarball is unpacked
# to /work/IronMule. Four parallel parts, each with its own log and done-marker in /work/done/.
set -u
MODEL=${MODEL:-qwen3-8b}
ARCH=${ARCH:-89}
mkdir -p /work/done /work/logs /models/tokenizers
cd /work/IronMule
command -v patch >/dev/null || (apt-get update -qq && apt-get install -y -qq patch) >/work/logs/apt.log 2>&1

( python3 -m venv /work/im && /work/im/bin/pip install -q -U pip cmake ninja \
  && /work/im/bin/pip install -q -e ".[cpu,dev]" jinja2 typing-extensions diskcache \
  && /work/im/bin/python tools/build_ggml.py --cuda "$ARCH" --jobs "$(nproc)" --dest /work/ggml \
  && touch /work/done/ironmule ) >/work/logs/setup-ironmule.log 2>&1 &

( python3 -m venv /work/vllm && /work/vllm/bin/pip install -q -U pip && /work/vllm/bin/pip install -q vllm==0.11.0 \
  && touch /work/done/vllm ) >/work/logs/setup-vllm.log 2>&1 &

( python3 lab/portable_gpu/fetch.py "$MODEL" --dir /models \
  && python3 -m ironmule_product.compare_tokenizer "$MODEL" --dir /models/tokenizers \
  && touch /work/done/gguf ) >/work/logs/setup-gguf.log 2>&1 &

( python3 -m lab.portable_gpu.fetch_awq --model "$MODEL" "/models/${MODEL}-awq" \
  && touch /work/done/awq ) >/work/logs/setup-awq.log 2>&1 &

wait
ls /work/done
