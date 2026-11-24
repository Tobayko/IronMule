#!/usr/bin/env bash
# PF3 A100 session (2026-11-24): PX1 (one kept prefix per packed group) against the reference, and diagnostics of the
# 1.2-2.0 s prefill pauses. On a pod after `JOBS=32 bash lab/rp/setup.sh 80`:
#   bash lab/pf3/a100_pf3.sh > /work/a100_pf3.log 2>&1
# Fixed before the pod (scope, stop rule, cost):
#   D  three server starts with the pause diagnostics (engine trace, IRONMULE_GGML_PAUSE samples and per-call deltas,
#      TL1 per-ubatch timestamps), 32 clients; the first also 8 clients with the factual check. Not used for timing.
#   A  PX1 against the reference in one engine process (lab/pf3/px1_ab.py): 12 ABBA pairs, the benchmark's repeats 1
#      and 2 after a primer of the repeat before, identical prompt work and KV state in both arms, no profiler.
#      Go on to H only if PX1's median paired saving is at least 40 ms and PX1 is faster in at least 10 of 12 pairs.
#   H  HTTP, alternating server starts reference / PX1 / reference / PX1 (32 clients x 3 waves, 8 clients x 2 waves
#      with the factual check), then vLLM on the same GPU (same workload). PX1 becomes the default only if its median
#      32-client wave over both starts is at least 40 ms shorter, its median first token not later by more than 5 %,
#      and its facts 18/18.
#   Cost: watchdog at 2.0 h after the pod's creation (1.39 USD/h, at most 2.78 USD), not extended during the run.
# Throughout: nvidia-smi every 200 ms (clocks, power, throttle reasons), observing only; runtimes are not normalised.
set -uo pipefail
W=/work/pf3
mkdir -p $W
cd /work/IronMule
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
M=/models/Qwen3-32B-Q4_K_M.gguf
ST=$W/st
stamp() { echo "== $(date +%H:%M:%S) $*"; }
pack() { (cd $W && tar czf /work/pf3.tgz --exclude='st' * 2>/dev/null); }
trap pack EXIT
stamp "$(git log --oneline -1); patches: $(ls -a $T/vendor/llama.cpp | grep '^.ironmule-' | tr '\n' ' ')"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,power.limit,driver_version --format=csv,noheader
python -c "import time; print('CLOCKS unix', time.time(), 'monotonic', time.monotonic())" | tee $W/clocks.txt
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,pstate,temperature.gpu,clocks_throttle_reasons.active,utilization.gpu \
  --format=csv,noheader -lms 200 > $W/smi.csv 2>&1 &
cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null | sed 's/^/ptrace_scope /'
which py-spy || pip install -q py-spy > /dev/null 2>&1; which py-spy
[ -d $ST ] || python -c "from pathlib import Path; from ironmule_product.state import ProductStore; ProductStore(Path('$ST')).setup('server')"

serve() {  # name, extra env...
  local name=$1; shift
  (unset IRONMULE_COMPARE_LOCAL_KEY; env IRONMULE_GGML_SEQ_CTX=2048 IRONMULE_ADMISSION_LEARN=0 IRONMULE_GGML_TRACE=$W/trace_$name.txt "$@" \
     python -m ironmule_cli serve --gguf $M --state-dir $ST --agent on --no-thinking \
     --api-key-env IRONMULE_COMPARE_LOCAL_KEY --host 127.0.0.1 --port 8002 > $W/serve_$name.log 2>&1) &
  local t0=$(date +%s)
  until grep -q '"ready": true' $W/serve_$name.log || ! pgrep -f "[i]ronmule_cli serve" >/dev/null; do sleep 5; done
  stamp "$name startup $(( $(date +%s) - t0 )) s"
}
show() {
  python -c "import json,sys;[print(r.get('label'),r.get('warmup_s') or (r['repeat'],r['width'],r['tok_s'],r['wall_s'],r['ttft_p50_ms'],r['ttft_max_ms'],r.get('quality'),r.get('quality_misses'))) for f in sys.argv[1:] for r in map(json.loads,open(f))]" "$@"
}
bench() {  # url engine model label width repeats [extra]
  local url=$1 eng=$2 model=$3 label=$4 width=$5 reps=$6; shift 6
  python lab/pk1/http_bench.py --url $url --engine $eng --model $model --widths $width --repeats $reps --label $label \
    --timeline $W/tl_${label}_$width.jsonl "$@" > $W/http_${label}_$width.jsonl 2>$W/http_${label}_$width.err
  show $W/http_${label}_$width.jsonl
}
im() { bench http://127.0.0.1:8002 ironmule gguf/Qwen3-32B-Q4_K_M "$@"; }
stop_im() { pkill -f "[i]ronmule_cli serve"; sleep 15; }

stamp "D: pause diagnostics, three starts"
for i in 1 2 3; do
  serve D$i IRONMULE_GGML_PAUSE=$W/pause_D$i.txt IRONMULE_GGML_PAUSE_MIN_MS=1000 IRONMULE_GGML_PAUSE_SPY_MS=2900 \
    IRONMULE_TL1=$W/tl1_D$i.txt
  im D$i 32 $([ $i = 1 ] && echo 3 || echo 1)
  [ $i = 1 ] && im D${i}q 8 2 --quality
  stop_im
  grep -c "^# call" $W/pause_D$i.txt 2>/dev/null | sed "s/^/D$i long calls sampled: /"
done

stamp "A: PX1 against the reference in one process"
LIB=$(ls -t $ST/kernels/w4/*.so | head -1); TAB=$(ls -t $ST/kernels/w4/*.table | head -1)
env IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_CUDA_GRAPHS_ROWS=32 IRONMULE_KG_LIB=$LIB IRONMULE_W4_TABLE=$TAB \
  IRONMULE_GGML_SEQ_CTX=2048 IRONMULE_GGML_TRACE=$W/trace_ab.txt python lab/pf3/px1_ab.py $M $W/px1_ab.jsonl --pairs 12 2>&1 \
  | grep "^ENGINE\|^COLD\|^AB\|^SUMMARY\|Error\|Traceback" | cut -c1-700 | tee $W/px1_ab.log
GO=$(python - <<'PY'
import json
s = [json.loads(l) for l in open("/work/pf3/px1_ab.jsonl") if l.startswith('{"summary"')]
ok = bool(s) and s[0]["summary"]["saving_median_ms"] >= 40 and s[0]["summary"]["px1_faster"] >= 10
print(1 if ok else 0)
PY
)
stamp "A verdict: GO=$GO (rule: median saving >= 40 ms and >= 10 of 12 pairs)"

if [ "$GO" = 1 ]; then
  stamp "H: HTTP, reference and PX1 alternating"
  for arm in R1 P1 R2 P2; do
    case $arm in R*) serve $arm IRONMULE_GGML_PX1=0 ;; P*) serve $arm IRONMULE_GGML_PX1=1 ;; esac
    im $arm 32 3
    im ${arm}q 8 2 --quality
    stop_im
  done
  cp $ST/stability-log.jsonl $W/stability-log.jsonl
  stamp "H: vLLM"
  /opt/vllm/bin/vllm serve /models/qwen3-32b-awq --served-model-name compare --tokenizer /models/tokenizers/qwen3-32b \
    --generation-config vllm --max-model-len 2048 --max-num-seqs 32 --tensor-parallel-size 1 --gpu-memory-utilization 0.88 \
    --host 127.0.0.1 --port 8001 > $W/serve_vllm.log 2>&1 &
  VP=$!
  t0=$(date +%s)
  until curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $VP 2>/dev/null || break; done
  stamp "vllm startup $(( $(date +%s) - t0 )) s"
  bench http://127.0.0.1:8001 vllm compare vllm 32 3 --metrics $W/metrics_vllm.jsonl
  bench http://127.0.0.1:8001 vllm compare vllmq 8 2 --quality
  kill $VP; wait $VP 2>/dev/null
else
  cp $ST/stability-log.jsonl $W/stability-log.jsonl
fi
stamp A100-PF3-DONE
