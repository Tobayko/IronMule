#!/usr/bin/env bash
# A100 session for the 2026-11-24 experiments (Qwen3-32B Q4_K_M; prepared, not yet run). On a pod after
# `JOBS=32 bash lab/rp/setup.sh 80` (vLLM, the models, this commit's ggml build in /work/IronMule):
#   bash lab/w4bw/a100_20261124.sh > /work/a100_20261124.log 2>&1
# 1. kernels: every W4R variant at 8/16/24/32 tokens on every Qwen3-32B shape (the full 5120 x 25600 gate/up included),
#    PS (v12) against v9 with a bit-for-bit check; then the L2-resident ceiling.
# 2. engine (one process per arm, the benchmark's prompts): (a) the W4R table without and with width 24 at 8/20/24/32
#    requests; (b) v9 against PS on the full 32-row decode step, tokens compared; (c) the prompt-matmul threshold on the
#    8-client mix (fixed 0/128/256/512/1024 and the "equal" and "mix" calibrations), prefill and time to first token.
# 3. HTTP on this pod, same prompts, seeds and answer lengths: the baseline (255a58f), the candidate (this commit, plus
#    PS and/or PK2-M where step 2 showed a gain), vLLM; 1, 8, 24 and 32 clients, three repeats, facts graded.
set -uo pipefail
BASE=255a58f
W=/work/x24
mkdir -p $W
cd /work/IronMule
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
E="IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_CUDA_GRAPHS_ROWS=32"
M=/models/Qwen3-32B-Q4_K_M.gguf
git worktree add -f /work/base $BASE >/dev/null 2>&1 || true
echo "== candidate $(git log --oneline -1), baseline $(git -C /work/base log --oneline -1)"

echo "== 1. kernels"
python lab/w4bw/bench.py $W/bench --tokens 8,16,24,32 2>&1 | grep -v "^ *$" | tail -40
python lab/w4bw/bench.py $W/bench_l2 --tokens 32 --l2 2>&1 | tail -6

echo "== 2a. W4R tables: baseline tuner (no width 24), candidate tuner, candidate tuner with PS"
(cd /work/base && python -m ironmule.w4_kernels --gguf $M --out $W/w4base > $W/w4base.log 2>&1)
python -m ironmule.w4_kernels --gguf $M --out $W/w4cand > $W/w4cand.log 2>&1
IRONMULE_W4_PRE=1 python -m ironmule.w4_kernels --gguf $M --out $W/w4ps > $W/w4ps.log 2>&1
lib() { ls -t $1/*.so | head -1; }
tab() { ls -t $1/*.table | head -1; }
for arm in base cand; do
  env $E IRONMULE_KG_LIB=$(lib $W/w4$arm) IRONMULE_W4_TABLE=$(tab $W/w4$arm) \
    python lab/w4bw/engine_probe.py $M $W/e24_$arm.json --widths 8,20,24,32 --seeds 0,1,2 --label w24_$arm 2>&1 \
    | grep "^STEPS\|^WAVE\|Error"
done
echo "== 2b. v9 against PS, full 32-row decode step"
for arm in cand ps; do
  env $E IRONMULE_KG_LIB=$(lib $W/w4$arm) IRONMULE_W4_TABLE=$(tab $W/w4$arm) \
    python lab/w4bw/engine_probe.py $M $W/eps_$arm.json --widths 32 --seeds 0,1,2 --label ps_$arm 2>&1 \
    | grep "^STEPS\|^WAVE\|Error"
done
echo "== 2c. prompt-matmul threshold on the 8-client mix"
for rows in 0 128 256 512 1024 equal mix; do
  case $rows in equal|mix) P="IRONMULE_PK2_CALIBRATION=$rows";; *) P="IRONMULE_CUBLAS_MIN_ROWS=$rows";; esac
  env $E $P IRONMULE_KG_LIB=$(lib $W/w4cand) IRONMULE_W4_TABLE=$(tab $W/w4cand) \
    python lab/w4bw/engine_probe.py $M $W/epk_$rows.json --widths 8 --seeds 0,1,2,3,4 --label pk2_$rows 2>&1 \
    | grep "^PROBE\|^WAVE\|Error" | cut -c1-400
done
python - <<'PY'
import json
W = "/work/x24"
def load(n):
    try:
        return json.load(open(f"{W}/{n}.json"))
    except OSError:
        return None
def same(a, b):
    if not a or not b:
        return None
    return sum(x == y for k in b["tokens"] for x, y in zip(a["tokens"].get(k, []), b["tokens"][k]))
print("SUMMARY width24 base/cand steps", load("e24_base") and load("e24_base")["decode_ms_by_rows"],
      load("e24_cand") and load("e24_cand")["decode_ms_by_rows"])
print("SUMMARY ps steps cand/ps", load("eps_cand") and load("eps_cand")["decode_ms_by_rows"],
      load("eps_ps") and load("eps_ps")["decode_ms_by_rows"], "same answers", same(load("eps_ps"), load("eps_cand")))
for r in ("0", "128", "256", "512", "1024", "equal", "mix"):
    d = load(f"epk_{r}")
    if d:
        w = d["waves"]
        print("SUMMARY pk2", r, "wall", round(sum(x["wall_s"] for x in w) / len(w), 4), "ttft_p50",
              round(sum(x["ttft_p50_ms"] for x in w) / len(w), 1), "ttft_max",
              round(sum(x["ttft_max_ms"] for x in w) / len(w), 1), "prefill", [x["prefill"] for x in w][:1])
PY

echo "== 3. HTTP: baseline, candidate, vLLM on this pod"
serve() {  # name, package dir, extra env
  local name=$1 dir=$2; shift 2
  pip install -q -e $dir > /dev/null 2>&1
  rm -rf $W/st_$name && python -c "from pathlib import Path; from ironmule_product.state import ProductStore; ProductStore(Path('$W/st_$name')).setup('server')"
  (unset IRONMULE_COMPARE_LOCAL_KEY; cd $dir && env IRONMULE_GGML_SEQ_CTX=2048 IRONMULE_ADMISSION_LEARN=0 "$@" \
     python -m ironmule_cli serve --gguf $M --state-dir $W/st_$name --agent on --no-thinking \
     --api-key-env IRONMULE_COMPARE_LOCAL_KEY --host 127.0.0.1 --port 8002 > $W/serve_$name.log 2>&1) &
  local t0=$(date +%s)
  until grep -q '"ready": true' $W/serve_$name.log || ! pgrep -f "[i]ronmule_cli serve" >/dev/null; do sleep 10; done
  echo "== $name startup $(( $(date +%s) - t0 )) s"
  python lab/pk1/http_bench.py --url http://127.0.0.1:8002 --engine ironmule --model gguf/Qwen3-32B-Q4_K_M \
    --widths 1,8,24,32 --repeats 3 --quality --label $name > $W/http_$name.jsonl 2>$W/http_$name.err
  python -c "import json;[print('$name',r['repeat'],r['width'],r['tok_s'],r['ttft_p50_ms'],r['ttft_p95_ms'],r.get('quality'),r.get('quality_misses')) for r in map(json.loads,open('$W/http_$name.jsonl'))]"
  pkill -f "[i]ronmule_cli serve"; sleep 15
}
serve base /work/base
EXTRA=""
[ -n "${CAND_PS:-}" ] && EXTRA="$EXTRA IRONMULE_W4_PRE=1"
[ -n "${CAND_PK2:-}" ] && EXTRA="$EXTRA IRONMULE_PK2_CALIBRATION=$CAND_PK2"
serve cand /work/IronMule $EXTRA
/opt/vllm/bin/vllm serve /models/qwen3-32b-awq --served-model-name compare --tokenizer /models/tokenizers/qwen3-32b \
  --generation-config vllm --max-model-len 2048 --max-num-seqs 32 --tensor-parallel-size 1 --gpu-memory-utilization 0.88 \
  --host 127.0.0.1 --port 8001 > $W/vllm_serve.log 2>&1 &
VP=$!
until curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do sleep 10; kill -0 $VP 2>/dev/null || break; done
python lab/pk1/http_bench.py --url http://127.0.0.1:8001 --engine vllm --model compare --widths 1,8,24,32 --repeats 3 \
  --quality --label vllm > $W/http_vllm.jsonl 2>$W/http_vllm.err
python -c "import json;[print('vllm',r['repeat'],r['width'],r['tok_s'],r['ttft_p50_ms'],r['ttft_p95_ms'],r.get('quality'),r.get('quality_misses')) for r in map(json.loads,open('$W/http_vllm.jsonl'))]"
kill $VP
echo A100-20261124-DONE
