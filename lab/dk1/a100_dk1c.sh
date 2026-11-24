#!/usr/bin/env bash
# DK1, third pass on the same pod (2026-11-24): v9 (variant 184) on gate/up at 32 tokens, every arm in one process
# (lab/dk1/abl_bench.py): the product build; explicit carveouts for the full kernel (0 % = the largest L1, 50 %, 100 % =
# about 28 KB of L1); the ablations 1-5 without a pin at the largest L1.
#   bash lab/dk1/a100_dk1c.sh > /work/a100_dk1c.log 2>&1
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
build c0 "-DIRONMULE_W4_CARVE=0"
build c50 "-DIRONMULE_W4_CARVE=50"
build c100 "-DIRONMULE_W4_CARVE=100"
for a in 1 2 3 4 5; do build m$a "-DIRONMULE_W4_ABL=$a -DIRONMULE_W4_CARVE=0"; done
wait
L=w4_w4mma_sm80.so
python lab/dk1/abl_bench.py prod=$W/b0/$L c0=$W/c0/$L c50=$W/c50/$L c100=$W/c100/$L 2>&1 | grep '^{'
python lab/dk1/abl_bench.py c0=$W/c0/$L m1=$W/m1/$L m2=$W/m2/$L m3=$W/m3/$L m4=$W/m4/$L m5=$W/m5/$L 2>&1 | grep '^{'
stamp "16 tokens, same arms (v9, 2 warps, 2 tiles, 2 token tiles)"
python lab/dk1/abl_bench.py --cfg 2:30201:2 --tokens 16 prod=$W/b0/$L c0=$W/c0/$L c100=$W/c100/$L 2>&1 | grep '^{'
stamp A100-DK1C-DONE
