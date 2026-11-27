#!/usr/bin/env bash
# PF1 A100 session (2026-11-24): the prefill/decode timeline of one 32-client wave, IronMule against vLLM on this pod.
# On a pod after `JOBS=32 bash lab/rp/setup.sh 80`:
#   bash lab/pf/a100_pf1.sh > /work/a100_pf1.log 2>&1
# Workload: lab/pk1/http_bench.py's 32-client mix (16 short prompts of ~33 tokens, 16 documents of ~444), the same
# seeds for every arm, 32-token answers, three waves (repeats, each with its own prompts). Cache conditions: every arm
# is a fresh server process (no KV or prefix cache left from another arm), the repeats' prompts differ, and the reuse
# each engine still found is recorded (IronMule's trace: `reused=` per request; vLLM: the prefix-cache counters of
# /metrics over each wave).
# 1. IronMule without the engine trace (first start: its gate and calibrations): client timeline + the service's
#    stability log; then the factual cases once, apart from the timed waves.
# 2. IronMule with IRONMULE_GGML_TRACE (same state, so the startup reuses step 1's gate): per native call its rows,
#    prompt / decode split and duration, per request its prefill start and end.
# 3. vLLM 0.11.0 (AWQ-Marlin, its defaults: chunked prefill, prefix caching): client timeline + /metrics per wave;
#    the factual cases once. A second vLLM arm without prefix caching only where the first showed more than 5 % hits.
set -uo pipefail
W=/work/pf
mkdir -p $W
cd /work/IronMule
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
M=/models/Qwen3-32B-Q4_K_M.gguf
ST=$W/st
stamp() { echo "== $(date +%H:%M:%S) $*"; }
stamp "$(git log --oneline -1); patches: $(ls $T/vendor/llama.cpp/.ironmule-* | xargs -n1 basename | tr '\n' ' ')"
[ -d $ST ] || python -c "from pathlib import Path; from ironmule_product.state import ProductStore; ProductStore(Path('$ST')).setup('server')"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader

serve() {  # name, extra env...
  local name=$1; shift
  (unset IRONMULE_COMPARE_LOCAL_KEY; env IRONMULE_GGML_SEQ_CTX=2048 IRONMULE_ADMISSION_LEARN=0 "$@" \
     python -m ironmule_cli serve --gguf $M --state-dir $ST --agent on --no-thinking \
     --api-key-env IRONMULE_COMPARE_LOCAL_KEY --host 127.0.0.1 --port 8002 > $W/serve_$name.log 2>&1) &
  local t0=$(date +%s)
  until grep -q '"ready": true' $W/serve_$name.log || ! pgrep -f "[i]ronmule_cli serve" >/dev/null; do sleep 10; done
  stamp "$name startup $(( $(date +%s) - t0 )) s"
}
show() {  # jsonl
  python -c "import json,sys;[print(r.get('label'),r.get('warmup_s') or (r['repeat'],r['width'],r['tok_s'],r['wall_s'],r['ttft_p50_ms'],r['ttft_max_ms'],r.get('quality'),r.get('quality_misses'))) for r in map(json.loads,open(sys.argv[1]))]" $1
}
im_bench() {  # label, extra bench args...
  local label=$1; shift
  python lab/pk1/http_bench.py --url http://127.0.0.1:8002 --engine ironmule --model gguf/Qwen3-32B-Q4_K_M \
    --widths 32 --label $label "$@" > $W/http_$label.jsonl 2>$W/http_$label.err
  show $W/http_$label.jsonl
}

stamp "1. IronMule, no engine trace"
serve im
im_bench im --repeats 3 --timeline $W/tl_im.jsonl
im_bench im_quality --repeats 1 --quality
pkill -f "[i]ronmule_cli serve"; sleep 15

stamp "2. IronMule, engine trace"
serve im_trace IRONMULE_GGML_TRACE=$W/trace.txt
im_bench im_trace --repeats 3 --timeline $W/tl_im_trace.jsonl
pkill -f "[i]ronmule_cli serve"; sleep 15
cp $ST/stability-log.jsonl $W/stability-log.jsonl

vllm_arm() {  # label, extra vLLM args...
  local label=$1; shift
  /opt/vllm/bin/vllm serve /models/qwen3-32b-awq --served-model-name compare --tokenizer /models/tokenizers/qwen3-32b \
    --generation-config vllm --max-model-len 2048 --max-num-seqs 32 --tensor-parallel-size 1 --gpu-memory-utilization 0.88 \
    --host 127.0.0.1 --port 8001 "$@" > $W/serve_$label.log 2>&1 &
  local vp=$! t0=$(date +%s)
  until curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do sleep 10; kill -0 $vp 2>/dev/null || break; done
  stamp "$label startup $(( $(date +%s) - t0 )) s"
  python lab/pk1/http_bench.py --url http://127.0.0.1:8001 --engine vllm --model compare --widths 32 --repeats 3 \
    --label $label --timeline $W/tl_$label.jsonl --metrics $W/metrics_$label.jsonl > $W/http_$label.jsonl 2>$W/http_$label.err
  show $W/http_$label.jsonl
  python lab/pk1/http_bench.py --url http://127.0.0.1:8001 --engine vllm --model compare --widths 32 --repeats 1 \
    --quality --label ${label}_quality > $W/http_${label}_quality.jsonl 2>$W/http_${label}_quality.err
  show $W/http_${label}_quality.jsonl
  kill $vp; wait $vp 2>/dev/null; sleep 10
}
stamp "3. vLLM"
vllm_arm vllm
HITS=$(python - <<'PY'
import json
q = h = 0.0
for r in map(json.loads, open("/work/pf/metrics_vllm.jsonl")):
    for k, v in r["delta"].items():
        if k.startswith("vllm:prefix_cache_queries"):
            q += v
        elif k.startswith("vllm:prefix_cache_hits"):
            h += v
print(f"{h / q:.4f}" if q else "0")
PY
)
stamp "vLLM prefix-cache hit fraction over the timed waves: $HITS"
if python -c "import sys; sys.exit(0 if float('$HITS') > 0.05 else 1)"; then
  vllm_arm vllm_nocache --no-enable-prefix-caching
fi

stamp "analysis"
python lab/pf/analyze.py --timeline $W/tl_im_trace.jsonl --stability $W/stability-log.jsonl --trace $W/trace.txt \
  --out $W/pf_im_trace.json | cut -c1-3000
python lab/pf/analyze.py --timeline $W/tl_im.jsonl --stability $W/stability-log.jsonl --out $W/pf_im.json | cut -c1-1500
for f in $W/tl_vllm*.jsonl; do
  python lab/pf/analyze.py --timeline $f --out ${f%.jsonl}.pf.json | cut -c1-1500
done
grep -h "reused=" $W/trace.txt | sed 's/.*prompt=\([0-9]*\) reused=\([0-9]*\).*/\1 \2/' | sort | uniq -c | sort -rn | head
stamp A100-PF1-DONE
