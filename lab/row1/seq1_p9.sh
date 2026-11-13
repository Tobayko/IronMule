#!/bin/bash
# SEQ1: the product with the width the engine chose itself (default request length 2,048) against the old default of
# 8 sequences, Qwen3-4B Q8_0, B16 kernels, serve.py's prompts.
SP=$1; O=$SP/row1/p9; mkdir -p $O; cd /home/user/IronMule
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IRONMULE_GGML_BINDINGS=$SP/src/llama_cpp_python-0.3.16 ROW1_GGML=B16
export LLAMA_CPP_LIB_PATH=$SP/row1/llama.cpp/build/bin ROW1_GATE_CACHE=$SP/row1/p7/gate
G=$SP/row1/gguf/Qwen3-4B-Q8_0.gguf
IRONMULE_GGML_SEQS=8 python lab/row1/product.py $G Qwen/Qwen3-1.7B $O/prod_fixed8.json 1,8,16,32 >> $O/log.txt 2>&1
ROW1_AUTO_SEQS=1 python lab/row1/product.py $G Qwen/Qwen3-1.7B $O/prod_auto.json 1,8,16,32 >> $O/log.txt 2>&1
echo ALLDONE >> $O/log.txt
