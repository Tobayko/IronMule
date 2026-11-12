#!/bin/bash
# Round 5 part 8: the engine with prefix reuse and the one-call prompt budget (new defaults) against llama-server -fa
# with the same B16 kernels, Qwen3-4B Q8_0: serve.py's prompts, then the same with a shared ~90-token system prompt.
SP=$1; O=$SP/row1/p8; mkdir -p $O; cd /home/user/IronMule
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IRONMULE_GGML_BINDINGS=$SP/src/llama_cpp_python-0.3.16 ROW1_GGML=B16
G=$SP/row1/gguf/Qwen3-4B-Q8_0.gguf; LIB=$SP/row1/llama.cpp/build/bin
PROD="env LLAMA_CPP_LIB_PATH=$LIB IRONMULE_GGML_SEQS=32 IRONMULE_GGML_SEQ_CTX=1024 ROW1_GATE_CACHE=$SP/row1/p7/gate python lab/row1/product.py $G Qwen/Qwen3-1.7B"
$PROD $O/prod_plain.json 1,8,16,32 >> $O/log.txt 2>&1
SYSTEXT="You are a careful assistant. Answer briefly and correctly, in plain English, without lists. "
export ROW1_SYSTEM="$SYSTEXT$SYSTEXT$SYSTEXT$SYSTEXT$SYSTEXT$SYSTEXT"
ROW1_CTXCAP=32768 python lab/row1/serve.py $LIB/llama-server $G srvB16-4B-sys load:1,8,16,32 -fa --no-mmap > $O/srv_sys.json 2>>$O/log.txt
$PROD $O/prod_sys.json 1,8,16,32 >> $O/log.txt 2>&1
echo ALLDONE >> $O/log.txt
