#!/usr/bin/env bash
# PD1 A100 session (2026-11-24): decode producers (FU1's fused norm, SWIGLU) write the W4R library's fp16 activations
# instead of prep_x; matmul variants and K-splits unchanged (the pod's own W4R table). On a pod after
# `JOBS=32 bash lab/rp/setup.sh 80` (this commit's build, patch pd1 included):
#   bash lab/pd1/a100_pd1.sh > /work/a100_pd1.log 2>&1
# 1. the pod's W4R tuning (one table for every arm).
# 2. correctness: one A/B round with IRONMULE_KG_PD1_CHECK=1 (every producer-written copy compared bit for bit with
#    prep_x's, inside the CUDA graphs), answers compared; its times are not used.
# 3. the decode step without diagnostics: reference and PD1 in one process, alternating, four rounds, 8 / 16 / 24 / 32
#    requests, two prompt sets each (lab/pd1/step_ab.py).
# 4. diagnostics apart: CUPTI kernel tables of 16 decode steps at 32 rows, PD1 off and on (lab/pk1/profile_engine.py).
# 5. rule fixed before the data: HTTP only if PD1's median 32-row step is at least 0.3 ms below the reference's in every
#    round and the check found no differing half. Then IronMule reference / PD1 / reference / PD1 (one repeat each,
#    32 and 8 clients, the first start of each its own quality gate), then vLLM, all with client timelines.
set -uo pipefail
W=/work/pd
mkdir -p $W
cd /work/IronMule
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
E="IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_CUDA_GRAPHS_ROWS=32"
M=/models/Qwen3-32B-Q4_K_M.gguf
stamp() { echo "== $(date +%H:%M:%S) $*"; }
pack() { (cd $W && tar czf /work/pd1.tgz --exclude='*.so' --exclude='st_*' --exclude='w4/*.json' *.json *.log *.jsonl *.env w4/*.table 2>/dev/null); }
trap pack EXIT
stamp "$(git log --oneline -1); patches: $(ls $T/vendor/llama.cpp/.ironmule-* | xargs -n1 basename | tr '\n' ' ')"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,power.limit,driver_version --format=csv,noheader

stamp "1. W4R tuning"
python -m ironmule.w4_kernels --gguf $M --out $W/w4 > $W/w4tune.log 2>&1
LIB=$(ls -t $W/w4/*.so | head -1); TAB=$(ls -t $W/w4/*.table | head -1)
grep -E "^1[24] (5120 25600|25600 5120|8192 5120|1024 5120|5120 8192) (8|16|24|32) " $TAB | sort | tr '\n' ';'; echo

stamp "2. correctness round (bit check on)"
env $E IRONMULE_KG_LIB=$LIB IRONMULE_W4_TABLE=$TAB IRONMULE_KG_PD1_CHECK=1 \
  python lab/pd1/step_ab.py $M $W/check.json --rounds 2 2>&1 | grep "^STEP\|^CHECK\|Error\|Traceback" | cut -c1-300

stamp "3. decode step A/B (no diagnostics)"
env $E IRONMULE_KG_LIB=$LIB IRONMULE_W4_TABLE=$TAB \
  python lab/pd1/step_ab.py $M $W/ab.json --rounds 4 2>&1 | grep "^STEP\|^SUMMARY\|Error\|Traceback" | cut -c1-300

stamp "4. diagnostics: kernel tables, PD1 off / on"
for pd in 0 1; do
  env $E IRONMULE_KG_LIB=$LIB IRONMULE_W4_TABLE=$TAB IRONMULE_KG_PD1=$pd python lab/pk1/profile_engine.py --gguf $M \
    --prompts 32 --prompt-len 200 --decode 16 --width 32 > $W/prof_pd$pd.log 2>&1
  echo "-- PD1=$pd"; sed -n '/== decode/,$p' $W/prof_pd$pd.log | head -42
done

python - <<'PY' > $W/decision.env
import json, sys
ab = json.load(open("/work/pd/ab.json")); ck = json.load(open("/work/pd/check.json"))
rows = ab["rounds"]
rounds = sorted({r["round"] for r in rows})
gain = []
for r in rounds:
    ref = [x["decode_ms_median"] for x in rows if x["round"] == r and x["arm"] == "ref" and x["width"] == 32]
    pd = [x["decode_ms_median"] for x in rows if x["round"] == r and x["arm"] == "pd1" and x["width"] == 32]
    if ref and pd:
        gain.append(min(ref) - max(pd))  # the smaller of the two prompt sets' gains, conservatively
same = all(x.get("tokens_equal_to_ref") == x.get("tokens") for x in ck["rounds"] + rows if "tokens" in x)
ok = ck["pd1_check"]["compared_halves"] > 0 and ck["pd1_check"]["differing_halves"] == 0
print(f"SUMMARY 32-row gain per round (ms, conservative): {[round(g, 3) for g in gain]}; tokens equal: {same}; "
      f"check: {ck['pd1_check']}", file=sys.stderr)
print(f"HTTP={'1' if gain and min(gain) >= 0.3 and ok else ''}")
PY
cat $W/decision.env
. $W/decision.env
if [ -z "${HTTP:-}" ]; then
  stamp "no HTTP run (rule above)"; stamp A100-PD1-DONE; exit 0
fi

stamp "5. HTTP: IronMule reference / PD1 alternating, then vLLM"
serve() {  # name, state suffix, extra env...
  local name=$1 st=$W/st_$2; shift 2
  [ -d $st ] || python -c "from pathlib import Path; from ironmule_product.state import ProductStore; ProductStore(Path('$st')).setup('server')"
  (unset IRONMULE_COMPARE_LOCAL_KEY; env IRONMULE_GGML_SEQ_CTX=2048 IRONMULE_ADMISSION_LEARN=0 "$@" \
     python -m ironmule_cli serve --gguf $M --state-dir $st --agent on --no-thinking \
     --api-key-env IRONMULE_COMPARE_LOCAL_KEY --host 127.0.0.1 --port 8002 > $W/serve_$name.log 2>&1) &
  local t0=$(date +%s)
  until grep -q '"ready": true' $W/serve_$name.log || ! pgrep -f "[i]ronmule_cli serve" >/dev/null; do sleep 10; done
  stamp "$name startup $(( $(date +%s) - t0 )) s"
  python lab/pk1/http_bench.py --url http://127.0.0.1:8002 --engine ironmule --model gguf/Qwen3-32B-Q4_K_M \
    --widths 32,8 --repeats 2 --quality --label $name --timeline $W/tl_$name.jsonl > $W/http_$name.jsonl 2>$W/http_$name.err
  python -c "import json;[print('$name',r.get('warmup_s') or (r['repeat'],r['width'],r['tok_s'],r['wall_s'],r['ttft_p50_ms'],r['ttft_max_ms'],r.get('quality'),r.get('quality_misses'))) for r in map(json.loads,open('$W/http_$name.jsonl'))]"
  pkill -f "[i]ronmule_cli serve"; sleep 15
}
serve ref1 ref
serve pd1 pd IRONMULE_KG_PD1=1
serve ref2 ref
serve pd2 pd IRONMULE_KG_PD1=1
/opt/vllm/bin/vllm serve /models/qwen3-32b-awq --served-model-name compare --tokenizer /models/tokenizers/qwen3-32b \
  --generation-config vllm --max-model-len 2048 --max-num-seqs 32 --tensor-parallel-size 1 --gpu-memory-utilization 0.88 \
  --host 127.0.0.1 --port 8001 > $W/serve_vllm.log 2>&1 &
VP=$!
until curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do sleep 10; kill -0 $VP 2>/dev/null || break; done
python lab/pk1/http_bench.py --url http://127.0.0.1:8001 --engine vllm --model compare --widths 32,8 --repeats 2 \
  --quality --label vllm --timeline $W/tl_vllm.jsonl > $W/http_vllm.jsonl 2>$W/http_vllm.err
python -c "import json;[print('vllm',r.get('warmup_s') or (r['repeat'],r['width'],r['tok_s'],r['wall_s'],r['ttft_p50_ms'],r['ttft_max_ms'],r.get('quality'))) for r in map(json.loads,open('$W/http_vllm.jsonl'))]"
kill $VP
for f in $W/tl_*.jsonl; do python lab/pf/analyze.py --timeline $f --width 32 | grep -v prefill_calls | cut -c1-400; done
stamp A100-PD1-DONE
