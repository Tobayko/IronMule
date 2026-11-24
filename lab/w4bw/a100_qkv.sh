#!/usr/bin/env bash
# A100 session (2026-11-24): QK1 / v13 (q, k, v in one launch) in the served decode step, then over HTTP if it wins.
# On a pod after `JOBS=32 bash lab/rp/setup.sh 80` (models, vLLM, this commit's ggml build with qk1):
#   bash lab/w4bw/a100_qkv.sh > /work/a100_qkv.log 2>&1
# 1. Q/K/V isolated again (lab/w4bw/qkv_bench.py, 8/16/24/32 rows) -> the fused configurations' table.
# 2. Decode step, engine (one process per round, the benchmark's prompts, the pod's own W4R table): QKV off / on / off /
#    on, steps at 8, 24 and 32 rows, answers compared. HTTP only if every "on" round's 32-row step beats every "off"
#    round's (the gain outside the spread).
# 3. HTTP, alternating servers off / on / off / on (one repeat each, same prompts, seeds and 32-token answers; the
#    first start of each variant runs its own quality gate, the second reuses it), the warm-up recorded apart; then
#    vLLM for reference.
set -uo pipefail
W=/work/q
mkdir -p $W
cd /work/IronMule
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
E="IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_CUDA_GRAPHS_ROWS=32"
M=/models/Qwen3-32B-Q4_K_M.gguf
stamp() { echo "== $(date +%H:%M:%S) $*"; }
stamp "$(git log --oneline -1); patches: $(ls $T/vendor/llama.cpp/.ironmule-* | xargs -n1 basename | tr '\n' ' ')"

stamp "1. Q/K/V isolated"
python lab/w4bw/qkv_bench.py $W/iso --tokens 8,16,24,32 2>&1 | grep '^{\|^STEP' | cut -c1-500
QT=$W/iso/qkv.table
cat $QT | head -3

stamp "2. decode step, QKV off/on alternating"
python -m ironmule.w4_kernels --gguf $M --out $W/w4 > $W/w4tune.log 2>&1
LIB=$(ls -t $W/w4/*.so | head -1); TAB=$(ls -t $W/w4/*.table | head -1)
for round in 1 2; do
  for arm in off on; do
    q=0; [ $arm = on ] && q=1
    env $E IRONMULE_KG_LIB=$LIB IRONMULE_W4_TABLE=$TAB IRONMULE_KG_QKV=$q IRONMULE_KG_QKV_TABLE=$QT \
      python lab/w4bw/engine_probe.py $M $W/step_${arm}_$round.json --widths 8,24,32 --seeds 0,1,2 --label ${arm}_$round 2>&1 \
      | grep "^STEPS\|Error\|Traceback"
  done
done
python - <<'PY' > $W/decision.env
import json, sys
W = "/work/q"
runs = {f"{a}_{r}": json.load(open(f"{W}/step_{a}_{r}.json")) for a in ("off", "on") for r in (1, 2)}
for rows in ("8", "24", "32"):
    off = [runs[f"off_{r}"]["decode_ms_by_rows"].get(rows) for r in (1, 2)]
    on = [runs[f"on_{r}"]["decode_ms_by_rows"].get(rows) for r in (1, 2)]
    print(f"SUMMARY step {rows} rows: off {off} on {on}", file=sys.stderr)
ref = runs["off_1"]["tokens"]
for name, run in runs.items():
    same = sum(x == y for k in ref for x, y in zip(run["tokens"][k], ref[k]))
    total = sum(len(v) for v in ref.values())
    print(f"SUMMARY answers {name}: {same}/{total} equal to off_1", file=sys.stderr)
off32 = [runs[f"off_{r}"]["decode_ms_by_rows"]["32"] for r in (1, 2)]
on32 = [runs[f"on_{r}"]["decode_ms_by_rows"]["32"] for r in (1, 2)]
print(f"HTTP={'1' if max(on32) < min(off32) else ''}")
PY
cat $W/decision.env
. $W/decision.env
if [ -z "${HTTP:-}" ]; then
  stamp "the 32-row gain stayed within the spread: no HTTP run"
  stamp A100-QKV-DONE
  exit 0
fi

stamp "3. HTTP, alternating"
serve() {  # name, state dir suffix, extra env
  local name=$1 st=$W/st_$2; shift 2
  [ -d $st ] || python -c "from pathlib import Path; from ironmule_product.state import ProductStore; ProductStore(Path('$st')).setup('server')"
  (unset IRONMULE_COMPARE_LOCAL_KEY; env IRONMULE_GGML_SEQ_CTX=2048 IRONMULE_ADMISSION_LEARN=0 "$@" \
     python -m ironmule_cli serve --gguf $M --state-dir $st --agent on --no-thinking \
     --api-key-env IRONMULE_COMPARE_LOCAL_KEY --host 127.0.0.1 --port 8002 > $W/serve_$name.log 2>&1) &
  local t0=$(date +%s)
  until grep -q '"ready": true' $W/serve_$name.log || ! pgrep -f "[i]ronmule_cli serve" >/dev/null; do sleep 10; done
  stamp "$name startup $(( $(date +%s) - t0 )) s"
  python lab/pk1/http_bench.py --url http://127.0.0.1:8002 --engine ironmule --model gguf/Qwen3-32B-Q4_K_M \
    --widths 1,8,24,32 --repeats 1 --quality --label $name > $W/http_$name.jsonl 2>$W/http_$name.err
  python -c "import json;[print('$name',r.get('warmup_s') or (r['width'],r['tok_s'],r['ttft_p50_ms'],r['ttft_max_ms'],r['wall_s'],r.get('quality'),r.get('quality_misses'))) for r in map(json.loads,open('$W/http_$name.jsonl'))]"
  pkill -f "[i]ronmule_cli serve"; sleep 15
}
serve off1 off
serve on1 on IRONMULE_KG_QKV=1 IRONMULE_KG_QKV_TABLE=$QT
serve off2 off
serve on2 on IRONMULE_KG_QKV=1 IRONMULE_KG_QKV_TABLE=$QT
/opt/vllm/bin/vllm serve /models/qwen3-32b-awq --served-model-name compare --tokenizer /models/tokenizers/qwen3-32b \
  --generation-config vllm --max-model-len 2048 --max-num-seqs 32 --tensor-parallel-size 1 --gpu-memory-utilization 0.88 \
  --host 127.0.0.1 --port 8001 > $W/vllm_serve.log 2>&1 &
VP=$!
until curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do sleep 10; kill -0 $VP 2>/dev/null || break; done
python lab/pk1/http_bench.py --url http://127.0.0.1:8001 --engine vllm --model compare --widths 1,8,24,32 --repeats 2 \
  --quality --label vllm > $W/http_vllm.jsonl 2>$W/http_vllm.err
python -c "import json;[print('vllm',r.get('warmup_s') or (r['repeat'],r['width'],r['tok_s'],r['ttft_p50_ms'],r['ttft_max_ms'],r['wall_s'],r.get('quality'))) for r in map(json.loads,open('$W/http_vllm.jsonl'))]"
kill $VP
stamp A100-QKV-DONE
