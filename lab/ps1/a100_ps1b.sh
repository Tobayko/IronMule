#!/usr/bin/env bash
# PS1 round 2 (2026-11-25): attribute and remove the 0.45-0.55 s of the slow merged prefills, then the 32-client
# benchmark against vLLM. DH2 off in every arm. On a pod after `JOBS=18 bash lab/rp/setup.sh 80`, one stage per call:
#   bash lab/ps1/a100_ps1b.sh d1 | d2 | d3 | h      (log: /work/ps1b_<stage>.log)
#
# Fixed before the pod (scope, rules, cost):
#   D1  engine diagnostics without the server (cuBLAS from 128 rows, no W4R tuning; lab/ps1/growth.py): waves of 8, 16,
#       16, 24, 24, 32, 32, 24, 32 fresh benchmark prompts (the KV cleared; one packed prefill each), TL1/TL2 on, no
#       profiler unless named: a R (PS1 off), b PS1 without the start's warm-up, c b under CUPTI for waves 3 and 4
#       (the first 24-request call and its repeat), d b with CUDA_MODULE_LOADING=EAGER, e PS1 with the warm-up of one
#       merged call of the full micro-batch (IRONMULE_PS1_WARM=2). A wave is slow when its prefill takes more than
#       1.15 x the call of the same request count later in the sequence. The fix taken is the variant without slow
#       waves at the smaller start-up cost; if none removes them, PS1 stays off and the attributed section is reported.
#   D2  one server start (its quality gate and W4R tuning) with PS1 and the chosen fix, the engine trace, TL1 and TL2:
#       the benchmark (32 clients x 3 waves, first wave counted, 8 clients x 2 with the factual check), then paired
#       waves R / S through the lab switch (8 rounds of 32): every merged prefill and every growth after ready() listed.
#   D3  a fresh start without trace or TL (PS1 on at the start, so its warm-up runs): paired waves R / S, 8 rounds of 32
#       clients, 4 of 8 with the factual check. Rule for S: median paired 32-client wave at least 10 ms shorter and
#       shorter in at least 6 of 8 rounds; 8-client median wave not longer by more than 2 %; 32-client first-token p50
#       not later by more than 5 %; facts 18/18 (or R's count).
#   H   fresh starts alternating R (PS1 off: today's default) and S (IRONMULE_PS1=1, the chosen fix), twice each, the
#       existing benchmark with the engine trace on (as in every earlier H), then vLLM 0.11.0 AWQ-Marlin on the GPU.
#       Result as before: median tok/s over all 32-client waves of the kept configuration against vLLM's median, +10 %
#       if >= 1.10 x; gap = median wave - vLLM's median wave / 1.10, and by position; first and later waves and every
#       merged prefill above 1.15 x the median of its wave position reported apart; nothing removed from the totals.
#       PS1 becomes the code default only if D3 passes and H's S is not below R.
#   Cost: watchdog at 2.5 h after the pod's creation, not extended during the run.
set -uo pipefail
STAGE=${1:?stage}
W=/work/ps1b
mkdir -p $W
cd /work/IronMule
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin IRONMULE_GGML_DH2=0
M=/models/Qwen3-32B-Q4_K_M.gguf
ST=$W/st
SW=$W/arms.json
stamp() { echo "== $(date +%H:%M:%S) $*"; }
pack() { (cd $W && tar czf /work/ps1b.tgz --exclude='st' --exclude='*.trace.json' * 2>/dev/null); }
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
R='{"PS1":0}'
S='{"PS1":1}'

case $STAGE in
d1)
  stamp "$(cat COMMIT); patches: $(ls -a $T/vendor/llama.cpp | grep '^.ironmule-' | tr '\n' ' ')"
  for v in a b c d e; do
    case $v in
      a) E="IRONMULE_PS1=0" ;; b) E="IRONMULE_PS1=1 IRONMULE_PS1_WARM=0" ;; c) E="IRONMULE_PS1=1 IRONMULE_PS1_WARM=0" ;;
      d) E="IRONMULE_PS1=1 IRONMULE_PS1_WARM=0 CUDA_MODULE_LOADING=EAGER" ;; e) E="IRONMULE_PS1=1 IRONMULE_PS1_WARM=2" ;;
    esac
    P=""; [ $v = c ] && P="--profile 3,4"
    stamp "D1 $v: $E"
    env $ENG $E IRONMULE_TL1=$W/tl1_$v.txt IRONMULE_TL2=$W/tl2_$v.txt IRONMULE_GGML_TRACE=$W/trace_d1$v.txt \
      timeout 900 python lab/ps1/growth.py $M $W/growth_$v.jsonl $P > $W/growth_$v.log 2>&1
    grep -E "^ENGINE|^WAVE|^GPU|Error|Traceback" $W/growth_$v.log | cut -c1-600
    grep -E "init end|ps1 warm_up" $W/trace_d1$v.txt | head -3
    pack
  done
  stamp D1-DONE ;;
d2)
  stamp "D2: server with PS1 (${FIX:-IRONMULE_PS1_WARM=2}), trace, TL1, TL2"
  echo "{}" > $SW
  serve D2 IRONMULE_PS1=1 ${FIX:-IRONMULE_PS1_WARM=2} IRONMULE_LAB_SWITCH=$SW IRONMULE_GGML_TRACE=$W/trace_D2.txt \
    IRONMULE_TL1=$W/tl1_D2.txt IRONMULE_TL2=$W/tl2_D2.txt
  grep -E "ps1 warm_up|init end" $W/trace_D2.txt | tail -2
  im D2 32 3
  im D2q 8 2 --quality
  paired D2p_32 32 8 --arms "R=$R" "S=$S"
  stop_im
  cp $ST/stability-log.jsonl $W/stability-log-D2.jsonl
  stamp D2-DONE ;;
d3)
  stamp "D3: paired R / S without trace"
  echo "{}" > $SW
  serve D3 IRONMULE_PS1=1 ${FIX:-IRONMULE_PS1_WARM=2} IRONMULE_LAB_SWITCH=$SW
  paired D3_32 32 8 --arms "R=$R" "S=$S"
  paired D3_8 8 4 --quality --arms "R=$R" "S=$S"
  python lab/adm2/verdict.py pick $W/paired_D3_32.jsonl $W/paired_D3_8.jsonl --ref R --singles S 2>$W/verdict_D3.txt | sed 's/^/D3 kept: /'
  cat $W/verdict_D3.txt
  stop_im
  stamp D3-DONE ;;
h)
  stamp "H: fresh starts R / S, then vLLM"
  for arm in R1 S1 R2 S2; do
    case $arm in R*) serve $arm IRONMULE_PS1=0 IRONMULE_GGML_TRACE=$W/trace_$arm.txt ;;
                 S*) serve $arm IRONMULE_PS1=1 ${FIX:-IRONMULE_PS1_WARM=2} IRONMULE_GGML_TRACE=$W/trace_$arm.txt ;; esac
    grep -E "ps1 warm_up" $W/trace_$arm.txt | tail -1
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
  python lab/adm2/verdict.py final $W --final S --ref R | tee $W/final_S.txt
  python lab/adm2/verdict.py final $W --final R --ref S | tee $W/final_R.txt
  stamp H-DONE ;;
esac
