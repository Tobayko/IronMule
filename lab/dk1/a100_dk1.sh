#!/usr/bin/env bash
# DK1 A100 session (2026-11-24): what limits the 32-row decode matmuls (gate/up Q4_K 5120 x 25600, down Q4_K and Q6_K
# 25600 x 5120). On a pod after `VLLM=0 JOBS=32 bash lab/rp/setup.sh 80`:
#   bash lab/dk1/a100_dk1.sh > /work/a100_dk1.log 2>&1
# 1. every W4R variant on the three shapes at 16 and 32 tokens (the winners; times include prep_x and reduce_split).
# 2. v9 ablations on gate/up at 32 tokens (IRONMULE_W4_ABL 0-4 in w4mma.cu, the full kernel's blocks per SM pinned),
#    two alternating rounds, no profiler.
# 3. Nsight Compute on the three winners at 32 tokens (weights from DRAM, x in L2), if the pod allows GPU counters.
# 4. the pod's own W4R tuning; served decode steps at 8 / 16 / 24 / 32 rows without a profiler (engine trace).
# 5. Nsight Compute on one served 32-request wave (CUDA graphs up to 32 rows), every W4R kernel of two decode steps.
set -uo pipefail
W=/work/dk
mkdir -p $W
cd /work/IronMule
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
E="IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_CUDA_GRAPHS_ROWS=32"
M=/models/Qwen3-32B-Q4_K_M.gguf
SHAPES=Q4_K:5120:25600,Q4_K:25600:5120,Q6_K:25600:5120
stamp() { echo "== $(date +%H:%M:%S) $*"; }
stamp "$(git log --oneline -1)"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,power.limit,driver_version --format=csv,noheader
NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
stamp "ncu: ${NCU:-none} $([ -n "$NCU" ] && $NCU --version | tail -1)"

stamp "1. all variants, 16 and 32 tokens"
python lab/w4bw/bench.py $W/b0 --tokens 16,32 --shapes $SHAPES --reps 30 2>&1 | grep '^{' | cut -c1-400
LIB=$W/b0/w4_w4mma_sm80.so
eval "$(python - <<'PY'
import json
d = json.load(open("/work/dk/b0/bench.json"))
names = {(5120, 25600): "G", (25600, 5120, "Q4_K"): "DQ", (25600, 5120, "Q6_K"): "D6"}
for r in d["best"]:
    if r["B"] != 32:
        continue
    key = "G" if r["N"] == 25600 else ("DQ" if r["kind"] == "Q4_K" else "D6")
    print(f'{key}V={r["variant"]} {key}K={r["ksplit"]}')
PY
)"
stamp "winners at 32 tokens: gate/up $GV/$GK, down Q4_K $DQV/$DQK, down Q6_K $D6V/$D6K"

stamp "2. v9 ablations on gate/up (variant 184 = v9 2 warps, 2 tiles, the pod-26 winner; 4 blocks per SM pinned)"
for a in 0 1 2 3 4; do
  (mkdir -p $W/abl$a && W4_NVCC_FLAGS="-DIRONMULE_W4_ABL=$a -DIRONMULE_W4_PIN=39936" python -c "
import sys; sys.path.insert(0, 'lab/w4bw'); import bench
from pathlib import Path; bench.build(Path('$W/abl$a'), '80', bench.ROOT / 'native/kernels/w4mma.cu')" > $W/abl$a.build 2>&1) &
done
wait
for round in 1 2; do
  for a in 0 1 2 3 4; do
    W4_REUSE=1 W4_NO_CHECK=1 python lab/w4bw/bench.py $W/abl$a --tokens 32 --shapes Q4_K:5120:25600 --only 184 --reps 50 2>&1 \
      | grep '"B": 32' | python -c "import json,sys; r=json.loads(sys.stdin.read()); print('ABL $a round $round', r['us'], 'us ks', r['ksplit'], 'err', r['rel_err'], 'next', r['next'])"
  done
done

if [ -n "$NCU" ]; then
  stamp "3. Nsight Compute, isolated winners at 32 tokens"
  prof() {  # name kind K N variant ksplit
    $NCU --set full --cache-control none --clock-control none --kernel-name regex:"w4[df]" --launch-skip 4 --launch-count 1 \
      -f -o $W/ncu_$1 python lab/dk1/ncu_kernel.py $LIB $2 $3 $4 32 $5 $6 > $W/ncu_$1.log 2>&1
    tail -3 $W/ncu_$1.log
    [ -f $W/ncu_$1.ncu-rep ] && $NCU --import $W/ncu_$1.ncu-rep --page details --csv > $W/ncu_$1.csv && \
      $NCU --import $W/ncu_$1.ncu-rep --page raw --csv > $W/ncu_$1.raw.csv
  }
  prof gate Q4_K 5120 25600 $GV $GK
  if grep -q "ERR_NVGPUCTRPERM\|permission" $W/ncu_gate.log; then
    stamp "GPU counters not permitted on this pod: Nsight Compute skipped"; NCU=""
  else
    prof downq4 Q4_K 25600 5120 $DQV $DQK
    prof downq6 Q6_K 25600 5120 $D6V $D6K
    prof gate184 Q4_K 5120 25600 184 1
  fi
fi

stamp "4. tuning and served decode steps"
python -m ironmule.w4_kernels --gguf $M --out $W/w4 > $W/w4tune.log 2>&1
TLIB=$(ls -t $W/w4/*.so | head -1); TAB=$(ls -t $W/w4/*.table | head -1)
grep -E "^Q4_K 25600 5120 32 |^Q4_K 5120 25600 32 |^Q6_K 25600 5120 32 |5120 25600 32|25600 5120 32" $TAB | head; head -2 $TAB
env $E IRONMULE_KG_LIB=$TLIB IRONMULE_W4_TABLE=$TAB python lab/w4bw/engine_probe.py $M $W/steps.json --widths 8,16,24,32 \
  --seeds 0,1 --label base 2>&1 | grep "^STEPS\|^WAVE\|Error\|Traceback" | cut -c1-400

if [ -n "$NCU" ]; then
  stamp "5. Nsight Compute, one served wave (CUDA graphs)"
  env $E IRONMULE_KG_LIB=$TLIB IRONMULE_W4_TABLE=$TAB $NCU --profile-from-start off --kernel-name regex:"w4[df]" \
    --launch-skip 449 --launch-count 898 --cache-control none --clock-control none \
    --section LaunchStats --section Occupancy --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis --section SchedulerStats --section WarpStateStats --section InstructionStats \
    -f -o $W/ncu_engine python lab/dk1/engine_ncu.py $M > $W/ncu_engine.log 2>&1
  tail -5 $W/ncu_engine.log
  [ -f $W/ncu_engine.ncu-rep ] && $NCU --import $W/ncu_engine.ncu-rep --page raw --csv > $W/ncu_engine.raw.csv
fi
cd $W && tar czf /work/dk1.tgz --exclude='*.so' --exclude='w4/*.json' *.log *.csv *.json b0/bench.json steps.json w4/*.table *.ncu-rep 2>/dev/null
ls -la /work/dk1.tgz
stamp A100-DK1-DONE
