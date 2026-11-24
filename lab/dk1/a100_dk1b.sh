#!/usr/bin/env bash
# DK1 follow-up on the same pod (2026-11-24), after lab/dk1/a100_dk1.sh. GPU counters are not permitted in RunPod's
# containers (ERR_NVGPUCTRPERM), so v9's limit on gate/up at 32 tokens is located by timing alone. Step 2 pinned the
# blocks per SM with dynamic shared memory and a 100 % carveout, which also cut L1 to about 28 KB and slowed the full
# kernel from about 75 to 121 us: v9 reads x through L1. Here:
# - the ablations 0-5 without a pin (each at its own registers' occupancy, the full L1);
# - the full kernel with a 14 KB pin at a 40 % carveout (4 blocks per SM, L1 about 128 KB) and with the 39 KB pin of
#   step 2 (L1 about 28 KB): how much of its time is x missing L1.
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
for a in 0 1 2 3 4 5; do build u$a "-DIRONMULE_W4_ABL=$a"; done
build l128 "-DIRONMULE_W4_ABL=0 -DIRONMULE_W4_PIN=14336 -DIRONMULE_W4_CARVE=40"
wait
for round in 1 2 3; do
  for d in u0 u1 u2 u3 u4 u5 l128 abl0; do
    [ -f $W/$d/w4_w4mma_sm80.so ] || continue
    W4_REUSE=1 W4_NO_CHECK=1 python lab/w4bw/bench.py $W/$d --tokens 32 --shapes Q4_K:5120:25600 --only 184 --reps 50 2>&1 \
      | grep '"B": 32' | python -c "import json,sys; r=json.loads(sys.stdin.read()); print('$d round $round', r['us'], 'us ks', r['ksplit'], 'err', r['rel_err'], 'next', r['next'])"
  done
done
stamp A100-DK1B-DONE
