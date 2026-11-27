#!/usr/bin/env bash
# PK6 / SV5 session (2026-11-25): close the remaining 26 ms a median wave (pod 36: 300.0 against vLLM's 274.8 tok/s,
# target 302.3) with two small changes, each with its own switch; the reference is today's default (PS1 + ps1a, no
# warm-up, DH2 off). On a pod after `JOBS=28 bash lab/rp/setup.sh 80`, one stage per call:
#   bash lab/pk6/a100_pk6.sh e | d | h      (log: /work/pk6_<stage>.log)
#
# Fixed before the pod (scope, rules, cost):
#   Candidates. PK6 (IRONMULE_PK6, native, read at every graph): the prefill's SwiGLU and fused add + RMS norm skip
#     the f32 writes that only matmuls taking the fp16 copy would read; expected -40 ms a 32-client wave (the two
#     kernels' bytes, pod 36 profile scaled to 7,390 rows), values bit-identical. SV5 (IRONMULE_SV5, worker, read at
#     every join): a join's requests rendered as one batch (SV4's path); expected -5 ms in waves whose burst arrives
#     as one join (pod 36: 7-12 ms of one-by-one rendering before the prefill).
#   E  engine, no server: lab/ps1/check.py with PK6 off / on (logits of every output row and greedy tokens; PK6 must
#     give max |difference| 0.0) with IRONMULE_PK6_LOG=1 (skips per graph); then lab/pk6/engine_ab.py, 8 ABBA pairs of
#     32 fresh benchmark prompts (prefill + 31 decode steps) in one process. If E's logits differ or nothing is
#     skipped, PK6 is out of D and H.
#   D  one server start (gate, W4R tuning; no trace), paired waves through the lab switch: R {PK6 0, SV5 0}, P {PK6 1},
#     V {SV5 1}, C {PK6 1, SV5 1}; 8 rounds of 32 clients, 4 of 8 with the factual check. Rule (lab/adm2/verdict.py
#     pick --singles P --combo C): an arm passes against R with its median paired 32-client wave at least 10 ms
#     shorter and shorter in at least 6 of 8 rounds; 8-client median wave not longer by more than 2 %; 32-client
#     first-token p50 not later by more than 5 %; facts 18/18 (or R's count). K = C if P and C pass and C saves at
#     least P's; else P if it passes; else R (and no H). V is reported, not a gate.
#   H  fresh starts R1 K1 K2 R2 (the order turned around in the middle), the existing benchmark with the engine trace
#     (32 clients x 3 waves, first wave counted; 8 clients x 2 with the factual check), then vLLM 0.11.0 AWQ-Marlin on
#     the GPU. Result: median tok/s over all six 32-client waves of K against vLLM's median; +10 % if >= 1.10 x.
#     Gap = median K wave - vLLM's median wave / 1.10, by position too; K against R; first and later waves apart;
#     every merged prefill above 1.15 x its position's median listed; nothing removed from any total.
#     K's switches become the code default only if D keeps them and H's K is not below R.
#   Cost: watchdog at 2.5 h after the pod's creation, not extended. One run per stage; no repeats to cross the line.
set -uo pipefail
STAGE=${1:?stage}
W=/work/pk6
mkdir -p $W
cd /work/IronMule
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
M=/models/Qwen3-32B-Q4_K_M.gguf
ST=$W/st
SW=$W/arms.json
stamp() { echo "== $(date +%H:%M:%S) $*"; }
pack() { (cd $W && tar czf /work/pk6.tgz --exclude='st' * 2>/dev/null); }
trap pack EXIT
[ -d $ST ] || python -c "from pathlib import Path; from ironmule_product.state import ProductStore; ProductStore(Path('$ST')).setup('server')"
pgrep -f "nvidia-smi --query-gpu=timestamp" >/dev/null || nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,memory.used,utilization.gpu \
  --format=csv,noheader -lms 200 > $W/smi.csv 2>&1 &
ENG="IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_W4=0 IRONMULE_CUBLAS_MIN_ROWS=128 IRONMULE_GGML_SEQ_CTX=2048"

serve() {  # name, extra env...
  local name=$1; shift
  (unset IRONMULE_COMPARE_LOCAL_KEY; env IRONMULE_GGML_SEQ_CTX=2048 IRONMULE_ADMISSION_LEARN=0 "$@" \
     python -m ironmule_cli serve --gguf $M --state-dir $ST --agent on --no-thinking \
     --api-key-env IRONMULE_COMPARE_LOCAL_KEY --host 127.0.0.1 --port 8002 > $W/serve_$name.log 2>&1) &
  local t0=$(date +%s)
  until grep -q '"ready": true' $W/serve_$name.log || ! pgrep -f "[i]ronmule_cli serve" >/dev/null; do sleep 5; done
  stamp "$name startup $(( $(date +%s) - t0 )) s"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/$name memory after ready: /"
}
stop_im() { pkill -f "[i]ronmule_cli serve"; sleep 15; }
paired() {  # label width rounds arms...
  local label=$1 width=$2 rounds=$3; shift 3
  python lab/adm2/paired.py --url http://127.0.0.1:8002 --model gguf/Qwen3-32B-Q4_K_M --switch $SW --width $width \
    --rounds $rounds --out $W/paired_$label.jsonl --timeline $W/tl_paired_$label.jsonl "$@" 2>$W/paired_$label.err \
    | grep SUMMARY | tee -a $W/summaries.txt
  pack
}
show() { python -c "import json,sys;[print(r.get('label'),r.get('warmup_s') or (r['repeat'],r['width'],r['tok_s'],r['wall_s'],r['ttft_p50_ms'],r['ttft_max_ms'],r.get('quality'))) for f in sys.argv[1:] for r in map(json.loads,open(f))]" "$@"; }
bench() {
  local url=$1 eng=$2 model=$3 label=$4 width=$5 reps=$6; shift 6
  python lab/pk1/http_bench.py --url $url --engine $eng --model $model --widths $width --repeats $reps --label $label \
    --timeline $W/tl_${label}_$width.jsonl "$@" > $W/http_${label}_$width.jsonl 2>$W/http_${label}_$width.err
  show $W/http_${label}_$width.jsonl; pack
}
im() { bench http://127.0.0.1:8002 ironmule gguf/Qwen3-32B-Q4_K_M "$@"; }
R='{"PK6":0,"SV5":0}'
P='{"PK6":1,"SV5":0}'
V='{"PK6":0,"SV5":1}'
C='{"PK6":1,"SV5":1}'

case $STAGE in
e)
  stamp "$(cat COMMIT); patches: $(ls -a $T/vendor/llama.cpp | grep '^.ironmule-' | tr '\n' ' ')"
  for v in 0 1; do
    env $ENG IRONMULE_PK6=$v IRONMULE_PK6_LOG=1 timeout 900 python lab/ps1/check.py $M $W/check_$v.npz --seqs 32 \
      > $W/check_$v.log 2>&1
    grep -E "^rows|pk6:|Error|Traceback|PK6" $W/check_$v.log | cut -c1-200 | sort | uniq -c | head -8
  done
  python - $W <<'EOF'
import sys, numpy as np
a, b = np.load(f"{sys.argv[1]}/check_0.npz"), np.load(f"{sys.argv[1]}/check_1.npz")
for k in a.files:
    if a[k].shape == b[k].shape and a[k].dtype.kind in "fiu":
        print("E check", k, a[k].shape, "max |on - off|", float(np.abs(a[k].astype(float) - b[k].astype(float)).max()))
EOF
  env $ENG IRONMULE_PK6_LOG=1 timeout 1500 python lab/pk6/engine_ab.py $M $W/engine_ab.jsonl --pairs 8 --tokens 32 \
    > $W/engine_ab.log 2>&1
  grep -E "^WAVE|^SUMMARY|Error|Traceback" $W/engine_ab.log | cut -c1-300
  grep -c "pk6:" $W/engine_ab.log | sed 's/^/E graphs with skips: /'
  grep "pk6:" $W/engine_ab.log | awk '{print $2}' | sort | uniq -c | head -5
  pack
  stamp E-DONE ;;
d)
  stamp "D: one server start, paired R / P / V / C, no trace"
  echo "{}" > $SW
  serve D IRONMULE_LAB_SWITCH=$SW
  paired D_32 32 8 --arms "R=$R" "P=$P" "V=$V" "C=$C"
  paired D_8 8 4 --quality --arms "R=$R" "P=$P" "V=$V" "C=$C"
  K=$(python lab/adm2/verdict.py pick $W/paired_D_32.jsonl $W/paired_D_8.jsonl --ref R --singles P --combo C 2>$W/verdict_D.txt)
  cat $W/verdict_D.txt
  echo "$K" > $W/K.txt
  stamp "D kept: $K"
  stop_im
  cp $ST/stability-log.jsonl $W/stability-log-D.jsonl
  stamp D-DONE ;;
h)
  K=$(cat $W/K.txt)
  case $K in P) KENV="IRONMULE_PK6=1 IRONMULE_SV5=0" ;; C) KENV="IRONMULE_PK6=1 IRONMULE_SV5=1" ;;
    *) stamp "D kept R: no H"; exit 0 ;; esac
  stamp "H: fresh starts R1 K1 K2 R2 (K=$K: $KENV), then vLLM"
  for arm in R1 K1 K2 R2; do
    case $arm in R*) serve $arm IRONMULE_PK6=0 IRONMULE_SV5=0 IRONMULE_GGML_TRACE=$W/trace_$arm.txt ;;
                 K*) serve $arm $KENV IRONMULE_GGML_TRACE=$W/trace_$arm.txt ;; esac
    im $arm 32 3
    im ${arm}q 8 2 --quality
    stop_im
  done
  cp $ST/stability-log.jsonl $W/stability-log-H.jsonl
  /opt/vllm/bin/vllm serve /models/qwen3-32b-awq --served-model-name compare --tokenizer /models/tokenizers/qwen3-32b \
    --generation-config vllm --max-model-len 2048 --max-num-seqs 32 --tensor-parallel-size 1 --gpu-memory-utilization 0.88 \
    --host 127.0.0.1 --port 8001 > $W/serve_vllm.log 2>&1 &
  VP=$!
  until curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $VP 2>/dev/null || break; done
  bench http://127.0.0.1:8001 vllm compare vllm 32 3 --metrics $W/metrics_vllm.jsonl
  bench http://127.0.0.1:8001 vllm compare vllmq 8 2 --quality
  kill $VP; wait $VP 2>/dev/null
  python lab/adm2/verdict.py final $W --final K --ref R | tee $W/final_K.txt
  python lab/ps1/slow_calls.py $W K1 K2 > $W/slow_K.txt 2>&1; python lab/ps1/slow_calls.py $W R1 R2 > $W/slow_R.txt 2>&1
  cat $W/slow_K.txt $W/slow_R.txt
  stamp H-DONE ;;
esac
