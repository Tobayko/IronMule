#!/bin/bash
# FA1 end to end: a 479-token system prompt on every request (lab/fa1/system.txt), Qwen3-4B Q8_0, this CPU.
# Stock llama-server (ROW1 kernels off, FA1 off) with and without -fa, against IronMule's product (B16 + FA1, the
# engine choosing its width). Usage: e2e.sh OLD_SCRATCHPAD OUT_DIR
SP=$1; O=$2; mkdir -p $O; cd /home/user/IronMule
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IRONMULE_GGML_BINDINGS=$SP/src/llama_cpp_python-0.3.16
export LLAMA_CPP_LIB_PATH=$SP/row1/llama.cpp/build/bin HF_HUB_OFFLINE=1
export ROW1_SYSTEM="$(cat lab/fa1/system.txt)"
G=$SP/row1/gguf/Qwen3-4B-Q8_0.gguf; S=$SP/row1/llama.cpp/build/bin/llama-server
ROW1_GGML=0 FA1=0 python lab/row1/serve.py $S $G stock_fa load:1,8,16 -fa > $O/stock_fa.txt 2>&1
ROW1_GGML=0 FA1=0 python lab/row1/serve.py $S $G stock_nofa load:1,8,16 > $O/stock_nofa.txt 2>&1
ROW1_GGML=B16 FA1=1 ROW1_AUTO_SEQS=1 ROW1_GATE_CACHE=$SP/row1/p7/gate python lab/row1/product.py $G Qwen/Qwen3-1.7B $O/product.json 1,8,16 > $O/product.txt 2>&1
echo ALLDONE >> $O/product.txt
