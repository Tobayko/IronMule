#!/usr/bin/env bash
# Kernel-level probe on the pod: where does Qwen3-32B Q4_K_M spend prefill and decode time on this GPU?
# Usage: bash lab/rp/probe.sh [stock|ironmule]  -> /work/results/probe-<tree>.txt
set -uo pipefail
T=${1:-stock}
B=$(ls -d ~/.cache/ironmule/ggml/$T/cuda/llama_cpp_python-*/vendor/llama.cpp/build/bin)
M=/models/Qwen3-32B-Q4_K_M.gguf
mkdir -p /work/results
OUT=/work/results/probe-$T.txt
export LD_LIBRARY_PATH=$B:${LD_LIBRARY_PATH:-}
{
  echo "== llama-bench prefill and decode, flash attention on"
  "$B/llama-bench" -m "$M" -ngl 99 -fa 1 -p 444,2048 -n 32 -ub 512,2048 -b 4096 -r 3
  echo "== batched-bench: 444-token prompts, 32 new tokens, 1/8/32 parallel (separate KV streams)"
  "$B/llama-batched-bench" -m "$M" -ngl 99 -fa on -c 40960 -b 4096 -ub 512 -npp 444 -ntg 32 -npl 1,8,32
  echo "== same, ubatch 2048"
  "$B/llama-batched-bench" -m "$M" -ngl 99 -fa on -c 40960 -b 4096 -ub 2048 -npp 444 -ntg 32 -npl 8,32
  echo "== same, unified KV"
  "$B/llama-batched-bench" -m "$M" -ngl 99 -fa on -c 40960 -b 4096 -ub 2048 -npp 444 -ntg 32 -npl 8,32 -kvu
} > "$OUT" 2>&1
echo "PROBE-DONE $OUT"
