#!/bin/bash
# PL1: llama-server (stock, all cores) against IronMule with the Pilot, while 0, 1 or 2 foreign processes keep cores
# busy; Qwen3-4B Q8_0, `burst` with N = 1 and 8, `single`. Usage: busy_cmp.sh OLD_SCRATCHPAD NEW_SCRATCHPAD WEIGHTS
SP=$1; NS=$2; WT=$3; O=$NS/cmp/busy; mkdir -p $O; cd /home/user/IronMule
G=$SP/row1/gguf/Qwen3-4B-Q8_0.gguf; LIB=$SP/row1/llama.cpp/build/bin; export HF_HUB_OFFLINE=1
wait_up() { for i in $(seq 1 1800); do curl -s -o /dev/null -w "%{http_code}" "$1" 2>/dev/null | grep -q 200 && return 0; sleep 2; done; return 1; }
if [ "${PART:-all}" != "ironmule" ]; then
ROW1_GGML=0 FA1=0 LLAMA_CPP_LIB_PATH=$LIB $LIB/llama-server -m $G --host 127.0.0.1 --port 8402 -np 8 -c 16384 --no-webui --jinja > $O/llama.log 2>&1 &
PID=$!; wait_up http://127.0.0.1:8402/health
for h in 0 1 2; do python lab/pl1/busy.py $h python lab/cmp/bench.py http://127.0.0.1:8402 qwen llama-busy$h $O/llama_h$h.json --workloads quick --ns 1,8 >> $O/summary.txt 2>&1; done
kill $PID; wait $PID
fi
[ "${PART:-all}" = "llama" ] && { echo LLAMADONE >> $O/summary.txt; exit 0; }
ST=$NS/cmp/state-ironmule
GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IRONMULE_GGML_BINDINGS=$SP/src/llama_cpp_python-0.3.16 LLAMA_CPP_LIB_PATH=$LIB ROW1_GGML=B16 IRONMULE_PILOT_WEIGHTS=$WT \
  python ironmule_cli.py serve --state-dir $ST --model Qwen/Qwen3-1.7B --gguf $G --agent on --port 8403 > $O/ironmule.log 2>&1 &
PID=$!; wait_up http://127.0.0.1:8403/health
for h in 0 1 2; do python lab/pl1/busy.py $h python lab/cmp/bench.py http://127.0.0.1:8403 Qwen/Qwen3-1.7B ironmule-busy$h $O/ironmule_h$h.json --workloads quick --ns 1,8 >> $O/summary.txt 2>&1; curl -s http://127.0.0.1:8403/health > $O/health_h$h.json; done
kill $PID; wait $PID
echo BUSYDONE >> $O/summary.txt
