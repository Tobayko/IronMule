#!/usr/bin/env bash
# Pod 35, phase H after the rule rejected PS1 and DH2 (run 3): fresh starts alternating R (PS1 and DH2 off: the default
# once they are rejected) and P (the build with PS1 and DH2 on, as diagnostics), twice each, the existing benchmark
# (32 clients x 3 waves, 8 clients x 2 with the factual check), then vLLM on the same GPU. Verdict on R as F.
set -uo pipefail
W=/work/ps1
cd /work/IronMule
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
M=/models/Qwen3-32B-Q4_K_M.gguf
ST=$W/st
stamp() { echo "== $(date +%H:%M:%S) $*"; }
pack() { (cd $W && tar czf /work/ps1_h.tgz --exclude='st' --exclude='run*' * 2>/dev/null); }
trap pack EXIT
serve() {
  local name=$1; shift
  (unset IRONMULE_COMPARE_LOCAL_KEY; env IRONMULE_GGML_SEQ_CTX=2048 IRONMULE_ADMISSION_LEARN=0 IRONMULE_GGML_TRACE=$W/trace_$name.txt "$@" \
     python -m ironmule_cli serve --gguf $M --state-dir $ST --agent on --no-thinking \
     --api-key-env IRONMULE_COMPARE_LOCAL_KEY --host 127.0.0.1 --port 8002 > $W/serve_$name.log 2>&1) &
  local t0=$(date +%s)
  until grep -q '"ready": true' $W/serve_$name.log || ! pgrep -f "[i]ronmule_cli serve" >/dev/null; do sleep 5; done
  stamp "$name startup $(( $(date +%s) - t0 )) s"
}
stop_im() { pkill -f "[i]ronmule_cli serve"; sleep 15; }
show() { python -c "import json,sys;[print(r.get('label'),r.get('warmup_s') or (r['repeat'],r['width'],r['tok_s'],r['wall_s'],r['ttft_p50_ms'],r['ttft_max_ms'],r.get('quality'))) for f in sys.argv[1:] for r in map(json.loads,open(f))]" "$@"; }
bench() {
  local url=$1 eng=$2 model=$3 label=$4 width=$5 reps=$6; shift 6
  python lab/pk1/http_bench.py --url $url --engine $eng --model $model --widths $width --repeats $reps --label $label \
    --timeline $W/tl_${label}_$width.jsonl "$@" > $W/http_${label}_$width.jsonl 2>$W/http_${label}_$width.err
  show $W/http_${label}_$width.jsonl; pack
}
im() { bench http://127.0.0.1:8002 ironmule gguf/Qwen3-32B-Q4_K_M "$@"; }
for arm in R1 P1 R2 P2; do
  case $arm in R*) serve $arm IRONMULE_PS1=0 IRONMULE_GGML_DH2=0 ;; P*) serve $arm IRONMULE_PS1=1 IRONMULE_GGML_DH2=1 ;; esac
  im $arm 32 3
  im ${arm}q 8 2 --quality
  stop_im
done
cp $ST/stability-log.jsonl $W/stability-log.jsonl
stamp "vLLM"
/opt/vllm/bin/vllm serve /models/qwen3-32b-awq --served-model-name compare --tokenizer /models/tokenizers/qwen3-32b \
  --generation-config vllm --max-model-len 2048 --max-num-seqs 32 --tensor-parallel-size 1 --gpu-memory-utilization 0.88 \
  --host 127.0.0.1 --port 8001 > $W/serve_vllm.log 2>&1 &
VP=$!
until curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $VP 2>/dev/null || break; done
bench http://127.0.0.1:8001 vllm compare vllm 32 3 --metrics $W/metrics_vllm.jsonl
bench http://127.0.0.1:8001 vllm compare vllmq 8 2 --quality
kill $VP; wait $VP 2>/dev/null
python lab/adm2/verdict.py final $W --final R --ref P | tee $W/final_R.txt
python lab/adm2/verdict.py final $W --final P --ref R | tee $W/final_P.txt
pack
stamp A100-H-DONE
