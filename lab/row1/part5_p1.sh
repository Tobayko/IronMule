#!/bin/bash
# Round 5 part 1: original llama-server -fa + huge pages on both models; the product with B16 on Qwen3-4B.
SP=/tmp/claude-0/-home-user-IronMule/6e4a0a70-e763-5189-b9a9-9d8ce1e4fff1/scratchpad
source $SP/gg3env.sh; export GLIBC_TUNABLES=glibc.malloc.hugetlb=1; O=$SP/row1/p6
ORIG=$SP/lc1_src/llama_cpp_python-0.3.16/vendor/llama.cpp/build/bin/llama-server
cd /home/user/IronMule
python lab/row1/serve.py $ORIG $SP/gg3_gguf/Qwen3-1.7B-Q8_0.gguf orig-1.7B load:1,8,16,32 -fa --no-mmap > $O/orig_1.7B.json 2>>$O/log.txt
ROW1_CTXCAP=32768 python lab/row1/serve.py $ORIG $SP/row1/gguf/Qwen3-4B-Q8_0.gguf orig-4B load:1,8,16,32 -fa --no-mmap > $O/orig_4B.json 2>>$O/log.txt
LLAMA_CPP_LIB_PATH=$SP/row1/llama.cpp/build/bin ROW1_GGML=B16 IRONMULE_GGML_SEQS=32 IRONMULE_GGML_SEQ_CTX=1024 \
  python lab/row1/product.py $SP/row1/gguf/Qwen3-4B-Q8_0.gguf Qwen/Qwen3-1.7B $O/prod_4B_B16.json 1,8,16,32 >> $O/log.txt 2>&1
echo ALLDONE >> $O/log.txt
