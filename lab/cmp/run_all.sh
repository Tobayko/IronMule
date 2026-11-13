#!/bin/bash
# CMP1: every engine on this CPU with the same GGUF (Qwen3-4B Q8_0), one after another, the vLLM build paused while
# one measures. Usage: run_all.sh OLD_SCRATCHPAD NEW_SCRATCHPAD ENGINES (ironmule,llama,ollama,lmstudio)
SP=$1; NS=$2; O=$NS/cmp/results; mkdir -p $O; cd /home/user/IronMule
G=$SP/row1/gguf/Qwen3-4B-Q8_0.gguf; LIB=$SP/row1/llama.cpp/build/bin; W="burst-sys,burst,poisson,single,facts"
export HF_HUB_OFFLINE=1
pause() { for p in $(pgrep -f "cc1plus|ninja|cmake|build_vllm|pip install"); do kill -$1 $p 2>/dev/null; done; }
wait_up() { for i in $(seq 1 1800); do curl -s -o /dev/null -w "%{http_code}" "$1" 2>/dev/null | grep -q 200 && return 0; sleep 2; done; return 1; }
for e in ${3//,/ }; do
  pause STOP
  case $e in
  ironmule)
    ST=$NS/cmp/state-ironmule
    [ -d $ST ] || { python ironmule_cli.py setup --state-dir $ST --mode server >/dev/null; python ironmule_cli.py models add Qwen/Qwen3-1.7B --state-dir $ST >/dev/null; }
    GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IRONMULE_GGML_BINDINGS=$SP/src/llama_cpp_python-0.3.16 LLAMA_CPP_LIB_PATH=$LIB ROW1_GGML=B16 \
      python ironmule_cli.py serve --state-dir $ST --model Qwen/Qwen3-1.7B --gguf $G --agent on --port 8301 > $O/ironmule.log 2>&1 &
    PID=$!; wait_up http://127.0.0.1:8301/health && python lab/cmp/bench.py http://127.0.0.1:8301 Qwen/Qwen3-1.7B ironmule $O/ironmule.json --workloads $W >> $O/summary.txt 2>&1
    curl -s http://127.0.0.1:8301/health > $O/ironmule_health.json; kill $PID; wait $PID ;;
  llama)
    ROW1_GGML=0 FA1=0 LLAMA_CPP_LIB_PATH=$LIB $LIB/llama-server -m $G --host 127.0.0.1 --port 8302 -t 4 -np 32 -c 32768 --no-webui --jinja > $O/llama.log 2>&1 &
    PID=$!; wait_up http://127.0.0.1:8302/health && python lab/cmp/bench.py http://127.0.0.1:8302 qwen llama-server $O/llama.json --workloads $W >> $O/summary.txt 2>&1
    kill $PID; wait $PID ;;
  llamafa)
    ROW1_GGML=0 FA1=0 LLAMA_CPP_LIB_PATH=$LIB $LIB/llama-server -m $G --host 127.0.0.1 --port 8306 -t 4 -np 32 -c 32768 -fa --no-webui --jinja > $O/llamafa.log 2>&1 &
    PID=$!; wait_up http://127.0.0.1:8306/health && python lab/cmp/bench.py http://127.0.0.1:8306 qwen llama-server-fa $O/llamafa.json --workloads $W >> $O/summary.txt 2>&1
    kill $PID; wait $PID ;;
  ollama)
    OLLAMA_MODELS=$NS/cmp/ollama-models OLLAMA_HOST=127.0.0.1:8303 OLLAMA_NUM_PARALLEL=32 OLLAMA_CONTEXT_LENGTH=1024 OLLAMA_KEEP_ALIVE=-1 \
      $NS/cmp/ollama/bin/ollama serve > $O/ollama.log 2>&1 &
    PID=$!; wait_up http://127.0.0.1:8303/api/version && python lab/cmp/bench.py http://127.0.0.1:8303 qwen3-4b-q8 ollama $O/ollama.json --workloads $W >> $O/summary.txt 2>&1
    kill $PID; wait $PID ;;
  ollama16)  # Ollama with 16 slots: with 32 its runner took about 12 GB and was killed for memory on this container
    OLLAMA_MODELS=$NS/cmp/ollama-models OLLAMA_HOST=127.0.0.1:8305 OLLAMA_NUM_PARALLEL=16 OLLAMA_CONTEXT_LENGTH=1024 OLLAMA_KEEP_ALIVE=-1 \
      $NS/cmp/ollama/bin/ollama serve > $O/ollama16.log 2>&1 &
    PID=$!; wait_up http://127.0.0.1:8305/api/version && python lab/cmp/bench.py http://127.0.0.1:8305 qwen3-4b-q8 ollama-16 $O/ollama16.json --workloads $W >> $O/summary.txt 2>&1
    kill $PID; wait $PID ;;
  lmstudio)
    unset BUN_OPTIONS; L=/root/.lmstudio/bin/lms
    $L server start --port 8304 > $O/lmstudio.log 2>&1
    $L load qwen3-4b --parallel 32 --context-length 32768 --gpu off -y --identifier qwen3-4b >> $O/lmstudio.log 2>&1
    python lab/cmp/bench.py http://127.0.0.1:8304 qwen3-4b lmstudio $O/lmstudio.json --workloads $W >> $O/summary.txt 2>&1
    $L unload --all >> $O/lmstudio.log 2>&1; $L server stop >> $O/lmstudio.log 2>&1 ;;
  esac
  pause CONT
done
echo ALLDONE >> $O/summary.txt
