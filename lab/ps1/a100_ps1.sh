#!/usr/bin/env bash
# PS1 A100 session (2026-11-25): one native pass for the packed prefill (PS1, llama.cpp patch ps1) and token delivery
# during the next native call (DH2), against the 11f00e4 default (ADM1, ADM2, BF1, PX2 on), then the 32-client HTTP
# benchmark against vLLM on the same GPU. On a pod after `JOBS=24 bash lab/rp/setup.sh 80`:
#   bash lab/ps1/a100_ps1.sh > /work/a100_ps1.log 2>&1
#
# Fixed before the pod (scope, rules, cost):
#   C   correctness of PS1 on the GPU (lab/ps1/check.py, 8 KV streams, unequal prompt and kept-prefix lengths, 8 decode
#       steps): logits and tokens with PS1 off, on, and on with the concat variant (cuBLAS from 128 rows, no W4R tuning).
#       PS1 is dropped from every later arm if it fails: a crash, a non-finite logit, on against concat (the same
#       kernels, outputs written in place against copied) differing by more than 1e-3, or on against off (small
#       passes on MMQ there) differing by more than 1.0 in a logit or in more than one of the 8 top-1 tokens.
#   E   engine A/B in one process (lab/ps1/engine_ab.py: R separate passes with PX2, S PS1 keeping every prefix), 10
#       rotated rounds of the benchmark's 32 prompts after a primer: wall, prefill calls, prompt rows, rows reused,
#       first tokens, identical answers. Diagnostic for the mechanism; the rule below decides.
#   A   one server start (quality gate, W4R tuning) with the engine trace and IRONMULE_LAB_SWITCH; arms alternate wave
#       by wave (lab/adm2/paired.py): R = 11f00e4 default, S = R + PS1, D = R + DH2, SD = R + PS1 + DH2; 8 rounds of
#       32 clients, then 4 rounds of 8 clients with the factual check in the last. No profiler.
#       Rule for a lever arm X against R: median paired 32-client wave at least 10 ms shorter and shorter in at least
#       6 of 8 rounds; 8-client median wave not longer than R's by more than 2 %; 32-client first-token p50 not later
#       by more than 5 %; facts 18/18 (or R's count). F = SD if S and D both pass and SD saves at least the larger
#       single saving; else the passing single arm with the larger saving; else R.
#   H   fresh starts alternating R (PS1 and DH2 off by environment) and F (the default build, no switch set by
#       environment if F keeps both; otherwise the one not kept switched off), twice each: lab/pk1/http_bench.py, 32
#       clients x 3 waves, 8 clients x 2 waves with the factual check; then vLLM 0.11.0 AWQ-Marlin, the same prompts
#       at the same positions. Result: median tok/s over all 32-client waves of F's starts (first waves included)
#       against vLLM's median; +10 % reached if F >= 1.10 x vLLM; gap = median F wave - median vLLM wave / 1.10, and
#       by position. 8 clients, first token and facts against R.
#   Cost: watchdog at 2.5 h after the pod's creation, not extended during the run.
set -uo pipefail
W=/work/ps1
mkdir -p $W
cd /work/IronMule
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
M=/models/Qwen3-32B-Q4_K_M.gguf
ST=$W/st
SW=$W/arms.json
stamp() { echo "== $(date +%H:%M:%S) $*"; }
pack() { (cd $W && tar czf /work/ps1.tgz --exclude='st' --exclude='*.npz' * 2>/dev/null); }
trap pack EXIT
stamp "$(cat COMMIT); patches: $(ls -a $T/vendor/llama.cpp | grep '^.ironmule-' | tr '\n' ' ')"
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit,driver_version --format=csv,noheader
nproc
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,memory.used,utilization.gpu --format=csv,noheader -lms 200 > $W/smi.csv 2>&1 &
[ -d $ST ] || python -c "from pathlib import Path; from ironmule_product.state import ProductStore; ProductStore(Path('$ST')).setup('server')"

stamp "C: PS1 correctness on the GPU"
PS1_OK=1
for v in off on concat; do
  case $v in off) E="IRONMULE_PS1=0" ;; on) E="IRONMULE_PS1=1" ;; concat) E="IRONMULE_PS1=1 IRONMULE_PS1_CONCAT=1" ;; esac
  env $E IRONMULE_PS1_LOG=1 IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_W4=0 IRONMULE_CUBLAS_MIN_ROWS=128 timeout 900 python lab/ps1/check.py $M $W/check_$v.npz \
    > $W/check_$v.log 2>&1 || PS1_OK=0
  grep -E "^rows|^ps1|Error|error" $W/check_$v.log | head -4
done
python - <<PY | tee $W/check.txt
import numpy as np
ok = True
got = {}
for v in ("off", "on", "concat"):
    try:
        got[v] = np.load(f"$W/check_{v}.npz")
    except OSError:
        print(v, "missing"); ok = False
if ok:
    a, b, c = got["off"], got["on"], got["concat"]
    finite = all(np.isfinite(x["logits"]).all() for x in (a, b, c))
    d_bc = float(np.abs(b["logits"] - c["logits"]).max())
    d_ab = float(np.abs(a["logits"] - b["logits"]).max())
    top = int((a["logits"].argmax(1) == b["logits"].argmax(1)).sum())
    print("finite", finite, "on_vs_concat", d_bc, "on_vs_off", d_ab, "top1_equal", top, "of", len(a["logits"]),
          "tokens_equal_on_off", bool((a["tokens"] == b["tokens"]).all()), "prefill_s", float(a["prefill_s"]), float(b["prefill_s"]))
    ok = finite and d_bc <= 1e-3 and d_ab <= 1.0 and top >= len(a["logits"]) - 1
print("PS1_OK", int(ok))
PY
grep -q "PS1_OK 1" $W/check.txt || PS1_OK=0
stamp "C verdict: PS1_OK=$PS1_OK"
pack

REF='{"ADM1":1,"ADM2":1,"BF1":1,"PX2":1,"PS1":0,"DH2":0}'
serve() {  # name, extra env...
  local name=$1; shift
  (unset IRONMULE_COMPARE_LOCAL_KEY; env IRONMULE_GGML_SEQ_CTX=2048 IRONMULE_ADMISSION_LEARN=0 IRONMULE_GGML_TRACE=$W/trace_$name.txt "$@" \
     python -m ironmule_cli serve --gguf $M --state-dir $ST --agent on --no-thinking \
     --api-key-env IRONMULE_COMPARE_LOCAL_KEY --host 127.0.0.1 --port 8002 > $W/serve_$name.log 2>&1) &
  local t0=$(date +%s)
  until grep -q '"ready": true' $W/serve_$name.log || ! pgrep -f "[i]ronmule_cli serve" >/dev/null; do sleep 5; done
  stamp "$name startup $(( $(date +%s) - t0 )) s"
}
stop_im() { pkill -f "[i]ronmule_cli serve"; sleep 15; }
paired() {  # label width rounds arms...
  local label=$1 width=$2 rounds=$3; shift 3
  python lab/adm2/paired.py --url http://127.0.0.1:8002 --model gguf/Qwen3-32B-Q4_K_M --switch $SW --width $width \
    --rounds $rounds --out $W/paired_$label.jsonl --timeline $W/tl_paired_$label.jsonl "$@" 2>$W/paired_$label.err \
    | grep SUMMARY | tee -a $W/summaries.txt
  pack
}
show() {
  python -c "import json,sys;[print(r.get('label'),r.get('warmup_s') or (r['repeat'],r['width'],r['tok_s'],r['wall_s'],r['ttft_p50_ms'],r['ttft_max_ms'],r.get('quality'))) for f in sys.argv[1:] for r in map(json.loads,open(f))]" "$@"
}
bench() {  # url engine model label width repeats [extra]
  local url=$1 eng=$2 model=$3 label=$4 width=$5 reps=$6; shift 6
  python lab/pk1/http_bench.py --url $url --engine $eng --model $model --widths $width --repeats $reps --label $label \
    --timeline $W/tl_${label}_$width.jsonl "$@" > $W/http_${label}_$width.jsonl 2>$W/http_${label}_$width.err
  show $W/http_${label}_$width.jsonl
  pack
}
im() { bench http://127.0.0.1:8002 ironmule gguf/Qwen3-32B-Q4_K_M "$@"; }

stamp "A: first start with the lab switch"
echo "{}" > $SW  # nothing named while the engine starts: defaults hold (PX2 measures its pass costs)
serve S0 IRONMULE_LAB_SWITCH=$SW IRONMULE_GGML_STEPPROF=1
if [ $PS1_OK = 1 ]; then
  ARMS=("R=$REF" 'S={"ADM1":1,"ADM2":1,"BF1":1,"PX2":1,"PS1":1,"DH2":0}' 'D={"ADM1":1,"ADM2":1,"BF1":1,"PX2":1,"PS1":0,"DH2":1}' \
        'SD={"ADM1":1,"ADM2":1,"BF1":1,"PX2":1,"PS1":1,"DH2":1}')
  PICK="--singles S D --combo SD"
else
  ARMS=("R=$REF" 'D={"ADM1":1,"ADM2":1,"BF1":1,"PX2":1,"PS1":0,"DH2":1}')
  PICK="--singles D"
fi
paired A_32 32 8 --arms "${ARMS[@]}"
paired A_8 8 4 --quality --arms "${ARMS[@]}"
F=$(python lab/adm2/verdict.py pick $W/paired_A_32.jsonl $W/paired_A_8.jsonl --ref R $PICK 2>$W/verdict_A.txt)
cat $W/verdict_A.txt
FJ=$(python lab/adm2/verdict.py arm "$F" "${ARMS[@]}")
stamp "A verdict: F=$F $FJ"
grep stepprof $W/trace_S0.txt | tail -3
stop_im
cp $ST/stability-log.jsonl $W/stability-log-S0.jsonl

if [ $PS1_OK = 1 ] && [ "${SKIP_E:-0}" != 1 ]; then  # run 3 of pod 35 skipped it: run 1 measured it already
  stamp "E: engine A/B, R against S"
  LIB=$(ls -t $ST/kernels/w4/*.so | head -1); TAB=$(ls -t $ST/kernels/w4/*.table | head -1)
  env IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_CUDA_GRAPHS_ROWS=32 IRONMULE_KG_LIB=$LIB IRONMULE_W4_TABLE=$TAB \
    IRONMULE_GGML_SEQ_CTX=2048 IRONMULE_GGML_TRACE=$W/trace_E.txt timeout 1500 python lab/ps1/engine_ab.py $M $W/engine_ab.jsonl --rounds 10 2>&1 \
    | grep "^ENGINE\|^SUMMARY\|Error\|Traceback" | cut -c1-900 | tee $W/engine_ab.log
  pack
fi

RENV="IRONMULE_PS1=0 IRONMULE_GGML_DH2=0"
FENV=$(python -c "
import json,sys
d=json.loads(sys.argv[1])
off=[('IRONMULE_PS1=0' if k=='PS1' else 'IRONMULE_GGML_DH2=0') for k in ('PS1','DH2') if not d.get(k)]
print(' '.join(off))" "$FJ")
stamp "H: fresh starts, reference ($RENV) and final (${FENV:-the default build, nothing set})"
for arm in R1 F1 R2 F2; do
  case $arm in R*) serve $arm $RENV ;; F*) serve $arm $FENV ;; esac
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
python lab/adm2/verdict.py final $W --final F --ref R | tee $W/final.txt
pack
stamp A100-PS1-DONE
