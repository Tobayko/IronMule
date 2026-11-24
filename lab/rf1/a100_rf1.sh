#!/usr/bin/env bash
# RF1 on the PD1 pod (2026-11-24), after lab/pd1/a100_pd1.sh: the grid K-split matmuls of the 32-row step (v11: down
# Q4_K 256/4, q 256/4, down Q6_K 267/8; w4d6 v Q6_K 214/8 stays on reduce_split as a control) with the partial sums
# reduced inside the kernel. Same W4R table as the PD1 session, a library built from this commit (RF1 changes no
# variant or split).
#   bash lab/rf1/a100_rf1.sh > /work/a100_rf1.log 2>&1
# 1. isolated: prep_x + matmul + whole reduction per call, reference and RF1 alternating, outputs compared bit for bit.
# 2. decode step: PD1 on in both arms, RF1 off / on alternating in one process, four rounds (lab/pd1/step_ab.py).
set -uo pipefail
W=/work/rf
mkdir -p $W
cd /work/IronMule
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
E="IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_CUDA_GRAPHS_ROWS=32"
M=/models/Qwen3-32B-Q4_K_M.gguf
stamp() { echo "== $(date +%H:%M:%S) $*"; }
pack() { (cd $W && tar czf /work/rf1.tgz *.json *.log 2>/dev/null); }
trap pack EXIT
stamp "$(git log --oneline -1)"
TAB=$(ls -t /work/pd/w4/*.table | head -1)
python -c "
import sys; sys.path.insert(0, 'lab/w4bw'); import bench
from pathlib import Path; bench.build(Path('$W/lib'), '80', bench.ROOT / 'native/kernels/w4mma.cu')" > $W/build.log 2>&1
LIB=$W/lib/w4_w4mma_sm80.so
ls -la $LIB
stamp "1. isolated, matmul + reduction"
python lab/rf1/kernel_ab.py $LIB --cases Q4_K:25600:5120:32:256:4,Q4_K:5120:8192:32:256:4,Q6_K:25600:5120:32:267:8,Q6_K:5120:1024:32:214:8 \
  2>&1 | grep "^KERNEL\|Error\|Traceback" | tee $W/kernel.log
stamp "2. decode step, PD1 on in both arms, RF1 off / on"
env $E IRONMULE_KG_LIB=$LIB IRONMULE_W4_TABLE=$TAB python lab/pd1/step_ab.py $M $W/ab.json --rounds 4 --switch rf1 --base pd1 \
  2>&1 | grep "^STEP\|^SUMMARY\|Error\|Traceback" | cut -c1-300 > $W/step.log
python - <<'PY'
import json
d = json.load(open("/work/rf/ab.json")); s = d["summary"]
for w in (8, 16, 24, 32):
    r, p = s[f"ref_{w}"], s[f"pd1_{w}"]
    print(f"STEPS {w}: PD1 {r['median']} ms, PD1+RF1 {p['median']} ms, diff {round(p['median'] - r['median'], 3)}; "
          f"PD1 {r['per_wave']}, PD1+RF1 {p['per_wave']}")
rows = d["rounds"]
print("tokens equal", sum(1 for x in rows if "tokens" in x and x["tokens_equal_to_ref"] == x["tokens"]), "of",
      sum(1 for x in rows if "tokens" in x))
PY
stamp A100-RF1-DONE
