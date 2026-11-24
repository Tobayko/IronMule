#!/usr/bin/env bash
# DK1 follow-up on the same pod (2026-11-24), after lab/dk1/a100_dk1.sh: GPU counters are not permitted in RunPod's
# containers (ERR_NVGPUCTRPERM), so the limit of v9 on gate/up at 32 tokens is located by timing alone:
# - ablation 5 (no dequantisation) beside step 2's ablations 0-4 (all at the full kernel's 4 blocks per SM);
# - the full kernel at 4, 3 and 2 blocks per SM (dynamic shared memory 39936 / 54000 / 80000 bytes): if the time grows
#   about in proportion, the kernel is limited by the work in flight (latency); if not, by a pipe.
#   bash lab/dk1/a100_dk1b.sh > /work/a100_dk1b.log 2>&1
set -uo pipefail
W=/work/dk
cd /work/IronMule
stamp() { echo "== $(date +%H:%M:%S) $*"; }
stamp "$(git log --oneline -1)"
build() {  # dir flags
  (mkdir -p $W/$1 && W4_NVCC_FLAGS="$2" python -c "
import sys; sys.path.insert(0, 'lab/w4bw'); import bench
from pathlib import Path; bench.build(Path('$W/$1'), '80', bench.ROOT / 'native/kernels/w4mma.cu')" > $W/$1.build 2>&1) &
}
build abl5 "-DIRONMULE_W4_ABL=5 -DIRONMULE_W4_PIN=39936"
build occ4 "-DIRONMULE_W4_ABL=0 -DIRONMULE_W4_PIN=39936"
build occ3 "-DIRONMULE_W4_ABL=0 -DIRONMULE_W4_PIN=54000"
build occ2 "-DIRONMULE_W4_ABL=0 -DIRONMULE_W4_PIN=80000"
wait
for round in 1 2 3; do
  for d in occ4 occ3 occ2 abl5 abl1 abl2 abl3 abl4; do
    [ -f $W/$d/w4_w4mma_sm80.so ] || continue
    W4_REUSE=1 W4_NO_CHECK=1 python lab/w4bw/bench.py $W/$d --tokens 32 --shapes Q4_K:5120:25600 --only 184 --reps 50 2>&1 \
      | grep '"B": 32' | python -c "import json,sys; r=json.loads(sys.stdin.read()); print('$d round $round', r['us'], 'us ks', r['ksplit'], 'err', r['rel_err'], 'next', r['next'])"
  done
done
stamp A100-DK1B-DONE
