#!/usr/bin/env bash
# PF2 A100 session (2026-11-24): the prefill and its spread between server starts, with the final PD1 state (PD1 on,
# RF1 off). On a pod after `JOBS=32 bash lab/rp/setup.sh 80`:
#   bash lab/pf2/a100_pf2.sh > /work/a100_pf2.log 2>&1
# 1. three IronMule server starts on one state directory (the first runs the quality gate and the W4R tuning, the
#    others reuse both: one tuning configuration throughout), each with the engine trace on: three 32-client waves in a
#    row (the first after the start, then two later ones; fresh prompts in each, the same prompts at the same position
#    in every start), then two 8-client waves with the facts graded. The trace records the start's stages and
#    choices (# init), the gathering window, each native call's host build / llama_decode / readback, and per request
#    when the engine saw it.
# 2. vLLM 0.11.0 with the same prompts (32 x 3, 8 x 2, facts graded).
# 3. diagnostics apart: CUPTI tables of the packed prefill, the first wave after an engine start and a later one
#    (lab/pf2/prefill_gpu.py).
# Throughout: nvidia-smi every 200 ms (SM / memory clock, power, P-state, temperature, throttle reasons), observing only.
set -uo pipefail
W=/work/pf2
mkdir -p $W
cd /work/IronMule
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
M=/models/Qwen3-32B-Q4_K_M.gguf
ST=$W/st
stamp() { echo "== $(date +%H:%M:%S) $*"; }
pack() { (cd $W && tar czf /work/pf2.tgz --exclude='st' *.json *.jsonl *.log *.csv *.txt 2>/dev/null); }
trap pack EXIT
stamp "$(git log --oneline -1); patches: $(ls $T/vendor/llama.cpp/.ironmule-* | xargs -n1 basename | tr '\n' ' ')"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,power.limit,driver_version --format=csv,noheader
python -c "import time; print('CLOCKS unix', time.time(), 'monotonic', time.monotonic())" | tee $W/clocks.txt
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,pstate,temperature.gpu,clocks_throttle_reasons.active,utilization.gpu \
  --format=csv,noheader -lms 200 > $W/smi.csv 2>&1 &
SMI=$!
[ -d $ST ] || python -c "from pathlib import Path; from ironmule_product.state import ProductStore; ProductStore(Path('$ST')).setup('server')"

for i in 1 2 3; do
  stamp "start S$i"
  (unset IRONMULE_COMPARE_LOCAL_KEY; env IRONMULE_GGML_SEQ_CTX=2048 IRONMULE_ADMISSION_LEARN=0 IRONMULE_GGML_TRACE=$W/trace_S$i.txt \
     python -m ironmule_cli serve --gguf $M --state-dir $ST --agent on --no-thinking \
     --api-key-env IRONMULE_COMPARE_LOCAL_KEY --host 127.0.0.1 --port 8002 > $W/serve_S$i.log 2>&1) &
  t0=$(date +%s)
  until grep -q '"ready": true' $W/serve_S$i.log || ! pgrep -f "[i]ronmule_cli serve" >/dev/null; do sleep 5; done
  stamp "S$i startup $(( $(date +%s) - t0 )) s"
  python lab/pk1/http_bench.py --url http://127.0.0.1:8002 --engine ironmule --model gguf/Qwen3-32B-Q4_K_M \
    --widths 32 --repeats 3 --label S$i --timeline $W/tl_S${i}_32.jsonl > $W/http_S${i}_32.jsonl 2>$W/http_S${i}_32.err
  python lab/pk1/http_bench.py --url http://127.0.0.1:8002 --engine ironmule --model gguf/Qwen3-32B-Q4_K_M \
    --widths 8 --repeats 2 --quality --label S${i}q --timeline $W/tl_S${i}_8.jsonl > $W/http_S${i}_8.jsonl 2>$W/http_S${i}_8.err
  python -c "import json,sys;[print(r.get('label'),r.get('warmup_s') or (r['repeat'],r['width'],r['tok_s'],r['wall_s'],r['ttft_p50_ms'],r['ttft_max_ms'],r.get('quality'),r.get('quality_misses'))) for f in sys.argv[1:] for r in map(json.loads,open(f))]" $W/http_S${i}_32.jsonl $W/http_S${i}_8.jsonl
  pkill -f "[i]ronmule_cli serve"; sleep 15
done
cp $ST/stability-log.jsonl $W/stability-log.jsonl

stamp "vLLM"
/opt/vllm/bin/vllm serve /models/qwen3-32b-awq --served-model-name compare --tokenizer /models/tokenizers/qwen3-32b \
  --generation-config vllm --max-model-len 2048 --max-num-seqs 32 --tensor-parallel-size 1 --gpu-memory-utilization 0.88 \
  --host 127.0.0.1 --port 8001 > $W/serve_vllm.log 2>&1 &
VP=$!
t0=$(date +%s)
until curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do sleep 5; kill -0 $VP 2>/dev/null || break; done
stamp "vllm startup $(( $(date +%s) - t0 )) s"
python lab/pk1/http_bench.py --url http://127.0.0.1:8001 --engine vllm --model compare --widths 32 --repeats 3 --label vllm \
  --timeline $W/tl_vllm_32.jsonl --metrics $W/metrics_vllm.jsonl > $W/http_vllm_32.jsonl 2>$W/http_vllm_32.err
python lab/pk1/http_bench.py --url http://127.0.0.1:8001 --engine vllm --model compare --widths 8 --repeats 2 --quality \
  --label vllmq --timeline $W/tl_vllm_8.jsonl > $W/http_vllm_8.jsonl 2>$W/http_vllm_8.err
python -c "import json,sys;[print(r.get('label'),r.get('warmup_s') or (r['repeat'],r['width'],r['tok_s'],r['wall_s'],r['ttft_p50_ms'],r['ttft_max_ms'],r.get('quality'))) for f in sys.argv[1:] for r in map(json.loads,open(f))]" $W/http_vllm_32.jsonl $W/http_vllm_8.jsonl
kill $VP; wait $VP 2>/dev/null; sleep 10

stamp "CUPTI: the packed prefill, first wave after an engine start and a later one"
LIB=$(ls -t $ST/kernels/w4/*.so | head -1); TAB=$(ls -t $ST/kernels/w4/*.table | head -1)
env IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_CUDA_GRAPHS_ROWS=32 IRONMULE_KG_LIB=$LIB IRONMULE_W4_TABLE=$TAB \
  IRONMULE_GGML_TRACE=$W/trace_prof.txt python lab/pf2/prefill_gpu.py $M $W/prof --waves 5 2>&1 \
  | grep "^ENGINE\|^GPU\|^WAVE\|Error\|Traceback" | tee $W/prefill_gpu.log | cut -c1-1500
kill $SMI

stamp "balance"
for i in 1 2 3; do
  for w in 32 8; do
    python lab/pf2/balance.py --timeline $W/tl_S${i}_$w.jsonl --stability $W/stability-log.jsonl --trace $W/trace_S$i.txt \
      --width $w > $W/balance_S${i}_$w.jsonl
  done
  grep "^INIT" $W/balance_S${i}_32.jsonl | tail -1 | cut -c1-600
  grep -v "^INIT" $W/balance_S${i}_32.jsonl | python -c "import json,sys;[print('S$i', d['repeat'], d['wave_s'], d['last_first_token_s'], d['chain_s'], d.get('prefill_calls') and [(c['rows'],c['prompt_rows'],c['filler_rows'],c['ms'],c['b_ms'],c['l_ms'],c['r_ms']) for c in d['prefill_calls']], d.get('gather')) for d in map(json.loads, sys.stdin)]"
done
for w in 32 8; do python lab/pf2/balance.py --timeline $W/tl_vllm_$w.jsonl --width $w > $W/balance_vllm_$w.jsonl; done
stamp A100-PF2-DONE
