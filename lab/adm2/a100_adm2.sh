#!/usr/bin/env bash
# ADM2 A100 session (2026-11-25): the server path (ADM1 admission, ADM2 logs written behind, BF1 batch fill) and PX2
# (PX1's cost decision) against the 584ec37 reference, then the 32-client HTTP benchmark against vLLM on the same GPU.
# On a pod after `JOBS=32 bash lab/rp/setup.sh 80`:
#   bash lab/adm2/a100_adm2.sh > /work/a100_adm2.log 2>&1
#
# Fixed before the pod (scope, rules, cost):
#   S0  one server start (it runs the quality gate and the W4R tuning) with the engine trace and IRONMULE_LAB_SWITCH;
#       the arms alternate inside it wave by wave (lab/adm2/paired.py: an order rotated every round, prompts of
#       their own for every wave, the same shape of work, one discarded wave per width first). No profiler.
#   A1  R (the reference: ADM1, ADM2, PX2, BF1 off), A (ADM1), P (PX2), AP (both): 8 rounds of 32 clients, then 4
#       rounds of 8 clients with the factual check in the last.
#       Rule for a lever arm X against R: median paired 32-client wave at least 10 ms shorter and shorter in at least
#       6 of 8 rounds; 8-client median wave not longer than R's by more than 2 %; 32-client first-token p50 not later
#       than R's by more than 5 %; facts 18/18 (or R's own count). B = AP if A and P both pass and AP's saving is at
#       least the larger single one; else the passing single arm with the larger saving; else R.
#       Defaults after the pod: a switch that F keeps stays on; one it does not keep goes off by default, except PX2,
#       which stays on unless P's median paired 32-client wave is 10 ms or more longer than R's (it is PX1 with a
#       measured price instead of a fixed limit, and the benchmark's waves rarely reach the cases where the two differ).
#   A2  B against C = B + ADM2 + BF1: 8 rounds of 32 clients, 4 of 8 clients; C replaces B by the same rule.
#   H   fresh server starts alternating reference / final (F = the configuration A1 and A2 kept), twice each, the
#       existing benchmark (lab/pk1/http_bench.py: 32 clients x 3 waves, 8 clients x 2 waves with the factual check),
#       then vLLM 0.11.0 AWQ-Marlin on the same GPU, the same prompts at the same positions.
#       Result: the median tok/s over all 32-client waves of F's starts (first waves included, as the 285.3 /
#       279.1 tok/s of pod 33 were) against vLLM's median; +10 % reached if F >= 1.10 x vLLM. Remaining gap in ms:
#       median F wave - median vLLM wave / 1.10. 8 clients: F not slower than the reference by more than 2 %.
#   D   diagnostics only if the pod is younger than 1.7 h: one py-spy record of the worker over 32-client waves of F.
#   Cost: watchdog at 2.5 h after the pod's creation (1.39 USD/h, at most 3.48 USD), not extended during the run.
set -uo pipefail
W=/work/adm2
mkdir -p $W
cd /work/IronMule
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
M=/models/Qwen3-32B-Q4_K_M.gguf
ST=$W/st
SW=$W/arms.json
stamp() { echo "== $(date +%H:%M:%S) $*"; }
pack() { (cd $W && tar czf /work/adm2.tgz --exclude='st' * 2>/dev/null); }
trap pack EXIT
stamp "$(git log --oneline -1 2>/dev/null || cat COMMIT); patches: $(ls -a $T/vendor/llama.cpp | grep '^.ironmule-' | tr '\n' ' ')"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,power.limit,driver_version --format=csv,noheader
nproc
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,pstate,temperature.gpu,clocks_throttle_reasons.active,utilization.gpu \
  --format=csv,noheader -lms 200 > $W/smi.csv 2>&1 &
which py-spy || pip install -q py-spy > /dev/null 2>&1
[ -d $ST ] || python -c "from pathlib import Path; from ironmule_product.state import ProductStore; ProductStore(Path('$ST')).setup('server')"

REF='{"ADM1":0,"ADM2":0,"PX2":0,"BF1":0}'
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
paired() {  # label width rounds arms... (NAME=JSON)
  local label=$1 width=$2 rounds=$3; shift 3
  python lab/adm2/paired.py --url http://127.0.0.1:8002 --model gguf/Qwen3-32B-Q4_K_M --switch $SW --width $width \
    --rounds $rounds --out $W/paired_$label.jsonl --timeline $W/tl_paired_$label.jsonl "$@" 2>$W/paired_$label.err \
    | grep SUMMARY | tee -a $W/summaries.txt
  pack
}
show() {
  python -c "import json,sys;[print(r.get('label'),r.get('warmup_s') or (r['repeat'],r['width'],r['tok_s'],r['wall_s'],r['ttft_p50_ms'],r['ttft_max_ms'],r.get('quality'),r.get('quality_misses'))) for f in sys.argv[1:] for r in map(json.loads,open(f))]" "$@"
}
bench() {  # url engine model label width repeats [extra]
  local url=$1 eng=$2 model=$3 label=$4 width=$5 reps=$6; shift 6
  python lab/pk1/http_bench.py --url $url --engine $eng --model $model --widths $width --repeats $reps --label $label \
    --timeline $W/tl_${label}_$width.jsonl "$@" > $W/http_${label}_$width.jsonl 2>$W/http_${label}_$width.err
  show $W/http_${label}_$width.jsonl
  pack
}
im() { bench http://127.0.0.1:8002 ironmule gguf/Qwen3-32B-Q4_K_M "$@"; }

stamp "S0: first start (gate, tuning) with the lab switch"
echo "{}" > $SW  # nothing named while the engine starts: PX2 measures its pass costs, defaults hold
serve S0 IRONMULE_LAB_SWITCH=$SW
grep -o '"pass_ms": {[^}]*}' $W/serve_S0.log | head -1
grep "pass_costs" $W/trace_S0.txt | tail -1

stamp "A1: R / A / P / AP"
ARMS1=("R=$REF" 'A={"ADM1":1,"ADM2":0,"PX2":0,"BF1":0}' 'P={"ADM1":0,"ADM2":0,"PX2":1,"BF1":0}' \
       'AP={"ADM1":1,"ADM2":0,"PX2":1,"BF1":0}')
paired A1_32 32 8 --arms "${ARMS1[@]}"
paired A1_8 8 4 --quality --arms "${ARMS1[@]}"
B=$(python lab/adm2/verdict.py pick $W/paired_A1_32.jsonl $W/paired_A1_8.jsonl --ref R --singles A P --combo AP)
stamp "A1 verdict: B=$B"
BJ=$(python lab/adm2/verdict.py arm "$B" "${ARMS1[@]}")
CJ=$(python -c "import json,sys; d=json.loads(sys.argv[1]); d.update(ADM2=1, BF1=1); print(json.dumps(d,separators=(',',':')))" "$BJ")
stamp "A2: B=$BJ against C=$CJ"
paired A2_32 32 8 --arms "B=$BJ" "C=$CJ"
paired A2_8 8 4 --quality --arms "B=$BJ" "C=$CJ"
K=$(python lab/adm2/verdict.py pick $W/paired_A2_32.jsonl $W/paired_A2_8.jsonl --ref B --singles C)
FJ=$([ "$K" = C ] && echo "$CJ" || echo "$BJ")
stamp "A2 verdict: kept=$K F=$FJ"
stop_im
cp $ST/stability-log.jsonl $W/stability-log-S0.jsonl
FENV=$(python -c "import json,sys; d=json.loads(sys.argv[1]); print(' '.join(f'IRONMULE_{k}={int(v)}' if k.startswith('ADM') else f'IRONMULE_GGML_{k}={int(v)}' for k,v in d.items()))" "$FJ")
RENV="IRONMULE_ADM1=0 IRONMULE_ADM2=0 IRONMULE_GGML_PX2=0 IRONMULE_GGML_BF1=0"
stamp "H: fresh starts, reference ($RENV) and final ($FENV) alternating"
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
AGE=${POD_AGE_S:-0}
if [ $(( AGE + SECONDS )) -lt 6120 ]; then
  stamp "D: py-spy of the worker over 32-client waves of F"
  serve D1 $FENV
  WP=$(pgrep -f "ironmule_product.worker" | head -1)
  (py-spy record -p $WP -o $W/pyspy_worker.txt -f raw -r 500 -d 20 --nonblocking > $W/pyspy.log 2>&1 &)
  im D1 32 3
  sleep 3
  stop_im
fi
stamp A100-ADM2-DONE
