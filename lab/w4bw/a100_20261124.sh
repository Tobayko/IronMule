#!/usr/bin/env bash
# A100 session for the 2026-11-24 experiments (Qwen3-32B Q4_K_M). On a pod after
# `JOBS=32 bash lab/rp/setup.sh 80` (vLLM, the models, this commit's ggml build in /work/IronMule):
#   bash lab/w4bw/a100_20261124.sh > /work/a100_20261124.log 2>&1
# 1. kernels: every W4R variant at 8/16/24/32 tokens on every Qwen3-32B shape (the full 5120 x 25600 gate/up included),
#    PS (v12) against v9 with a bit-for-bit check.
# 2. engine (one process per arm, the benchmark's prompts), tables from ONE tuning run (lab/w4bw/tables.py):
#    (a) base table (no width 24) against cand (width 24) at 8/20/24/32 requests; (b) PS against cand on the 24- and
#    32-row decode steps, tokens compared; (c) the prompt-matmul path on the 8-client mix: llama.cpp's kernels, fixed
#    128 / 1024 rows, the "equal" and "mix" calibrations; prefill and time to first token.
# 3. HTTP on this pod, same prompts, seeds and answer lengths: the baseline (255a58f), the candidate (this commit, plus
#    PS and/or PK2-M where step 2 showed a clear gain with the same answers), vLLM; 1, 8, 24 and 32 clients, three
#    repeats, facts graded with the misses recorded.
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
stamp() { echo "== $(date +%H:%M:%S) $*"; }
stamp "candidate $(git log --oneline -1), baseline $(git -C /work/base log --oneline -1)"

stamp "1. kernels"
python lab/w4bw/bench.py $W/bench --tokens 8,16,24,32 2>&1 | grep -v "^ *$" | tail -45

stamp "2. one tuning run with PS, three tables"
IRONMULE_W4_PRE=1 python -m ironmule.w4_kernels --gguf $M --out $W/w4 > $W/w4tune.log 2>&1
LIB=$(ls -t $W/w4/*.so | head -1); REC=$(ls -t $W/w4/*.json | head -1)
python lab/w4bw/tables.py $REC $LIB $W/tables
probe() {  # table, label, widths, seeds, extra env...
  local tab=$1 label=$2 widths=$3 seeds=$4; shift 4
  env $E "$@" IRONMULE_KG_LIB=$LIB IRONMULE_W4_TABLE=$W/tables/$tab.table \
    python lab/w4bw/engine_probe.py $M $W/$label.json --widths $widths --seeds $seeds --label $label 2>&1 \
    | grep "^PROBE\|^STEPS\|^WAVE\|Error\|Traceback" | cut -c1-420
}
stamp "2a/2b. width 24 and PS"
probe base e_base 8,20,24,32 0,1,2
probe cand e_cand 8,20,24,32 0,1,2
probe ps e_ps 24,32 0,1,2
stamp "2c. prompt-matmul path on the 8-client mix"
for arm in 0 128 1024 equal mix; do
  case $arm in equal|mix) P="IRONMULE_PK2_CALIBRATION=$arm";; *) P="IRONMULE_CUBLAS_MIN_ROWS=$arm";; esac
  probe cand e_pk_$arm 8 0,1,2,3,4 $P
done
python - <<'PY' > $W/decision.env
import json, statistics as st
W = "/work/x24"
def load(n):
    try:
        return json.load(open(f"{W}/{n}.json"))
    except OSError:
        return None
base, cand, ps = load("e_base"), load("e_cand"), load("e_ps")
def same(a, b):
    keys = [k for k in b["tokens"] if k in a["tokens"]]
    return sum(x == y for k in keys for x, y in zip(a["tokens"][k], b["tokens"][k])), sum(len(b["tokens"][k]) for k in keys)
import sys
log = sys.stderr
print("SUMMARY width24 steps base", base and base["decode_ms_by_rows"], file=log)
print("SUMMARY width24 steps cand", cand and cand["decode_ms_by_rows"], file=log)
use_ps = False
if ps and cand:
    eq, n = same(ps, cand)
    a, b = ps["decode_ms_by_rows"].get("32"), cand["decode_ms_by_rows"].get("32")
    print("SUMMARY ps steps", ps["decode_ms_by_rows"], "answers equal", eq, "of", n, file=log)
    use_ps = bool(a and b and a < 0.98 * b and eq == n)
pk = {}
for arm in ("0", "128", "1024", "equal", "mix"):
    d = load(f"e_pk_{arm}")
    if d:
        w = d["waves"]
        pk[arm] = (st.mean(x["wall_s"] for x in w), st.mean(x["ttft_p50_ms"] for x in w), st.mean(x["ttft_max_ms"] for x in w))
        print("SUMMARY pk2", arm, "wall_s %.4f ttft_p50 %.1f ttft_max %.1f" % pk[arm], "calibration", d["meta"].get("pk2"),
              file=log)
use_mix = "mix" in pk and "equal" in pk and pk["mix"][0] < 0.97 * pk["equal"][0] and pk["mix"][1] <= pk["equal"][1]
print(f"CAND_PS={'1' if use_ps else ''}")
print(f"CAND_PK2={'mix' if use_mix else ''}")
PY
cat $W/decision.env

stamp "3. HTTP: baseline, candidate, vLLM on this pod"
serve() {  # name, package dir, extra env
  local name=$1 dir=$2; shift 2
  pip install -q -e $dir > /dev/null 2>&1
  rm -rf $W/st_$name && python -c "from pathlib import Path; from ironmule_product.state import ProductStore; ProductStore(Path('$W/st_$name')).setup('server')"
  (unset IRONMULE_COMPARE_LOCAL_KEY; cd $dir && env IRONMULE_GGML_SEQ_CTX=2048 IRONMULE_ADMISSION_LEARN=0 "$@" \
     python -m ironmule_cli serve --gguf $M --state-dir $W/st_$name --agent on --no-thinking \
     --api-key-env IRONMULE_COMPARE_LOCAL_KEY --host 127.0.0.1 --port 8002 > $W/serve_$name.log 2>&1) &
  local t0=$(date +%s)
  until grep -q '"ready": true' $W/serve_$name.log || ! pgrep -f "[i]ronmule_cli serve" >/dev/null; do sleep 10; done
  stamp "$name startup $(( $(date +%s) - t0 )) s"
  python lab/pk1/http_bench.py --url http://127.0.0.1:8002 --engine ironmule --model gguf/Qwen3-32B-Q4_K_M \
    --widths 1,8,24,32 --repeats 3 --quality --label $name > $W/http_$name.jsonl 2>$W/http_$name.err
  python -c "import json;[print('$name',r['repeat'],r['width'],r['tok_s'],r['ttft_p50_ms'],r['ttft_p95_ms'],r.get('quality'),r.get('quality_misses')) for r in map(json.loads,open('$W/http_$name.jsonl'))]"
  pkill -f "[i]ronmule_cli serve"; sleep 15
}
serve base /work/base
. $W/decision.env
EXTRA="IRONMULE_GGML_SEQ_CTX=2048"
[ -n "${CAND_PS:-}" ] && EXTRA="$EXTRA IRONMULE_W4_PRE=1"
[ -n "${CAND_PK2:-}" ] && EXTRA="$EXTRA IRONMULE_PK2_CALIBRATION=$CAND_PK2"
stamp "candidate extras: $EXTRA"
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
stamp A100-20261124-DONE
