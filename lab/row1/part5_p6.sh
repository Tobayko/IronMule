#!/bin/bash
# Round 5 part 6: Qwen3-4B Q8_0, B16 kernels: llama-server -fa (same kernels) against the product with the old
# prompt budget (96) and with 1024. Usage: part5_p6.sh SCRATCH (holds row1/llama.cpp, src/llama_cpp_python-0.3.16).
SP=$1; O=$SP/row1/p7; mkdir -p $O; cd /home/user/IronMule
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1 IRONMULE_GGML_BINDINGS=$SP/src/llama_cpp_python-0.3.16
G=$SP/row1/gguf/Qwen3-4B-Q8_0.gguf; LIB=$SP/row1/llama.cpp/build/bin
[ -n "$SKIP_SRV" ] || ROW1_GGML=B16 ROW1_CTXCAP=32768 python lab/row1/serve.py $LIB/llama-server $G srvB16-4B load:1,8,16,32 -fa --no-mmap > $O/srv_B16_4B.json 2>>$O/log.txt
for B in 96 1024; do
  LLAMA_CPP_LIB_PATH=$LIB ROW1_GGML=B16 IRONMULE_GGML_SEQS=32 IRONMULE_GGML_SEQ_CTX=1024 IRONMULE_GGML_PROMPT_BUDGET=$B \
    ROW1_GATE_CACHE=$O/gate python lab/row1/product.py $G Qwen/Qwen3-1.7B $O/prod_4B_B16_pb$B.json 1,8,16,32 >> $O/log.txt 2>&1
done
echo ALLDONE >> $O/log.txt
