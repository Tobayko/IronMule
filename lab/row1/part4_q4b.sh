#!/bin/bash
# Round 4 part 1b: our kernels on the 4-bit files: A4 (exact Q4_0) for the layers, ROW1_OUT=1 (B16) for the Q6_K
# output matrix. Quality, single request, rows table; stock = hook off.
SP=/tmp/claude-0/-home-user-IronMule/6e4a0a70-e763-5189-b9a9-9d8ce1e4fff1/scratchpad
source $SP/gg3env.sh; export LLAMA_CPP_LIB_PATH=$SP/row1/llama.cpp/build/bin GLIBC_TUNABLES=glibc.malloc.hugetlb=1
SRV=$SP/row1/llama.cpp/build/bin/llama-server; O=$SP/row1/p5
cd /home/user/IronMule
run() { local tag=$1 q=$2; shift 2; G=$SP/row1/gguf/Qwen3-1.7B-$q.gguf
  env "$@" python lab/row1/ppl.py $G 2>/dev/null > $O/ppl_${q}_$tag.json
  env "$@" python lab/row1/facts.py $G 2>/dev/null > $O/facts_${q}_$tag.json
  env "$@" ROW1_MMAP=0 ROW1_RS=1,8,16,32 python lab/row1/rows_llama.py $G 2>/dev/null | tail -1 > $O/rows_${q}_$tag.json
  env "$@" python lab/row1/serve.py $SRV $G $q-$tag single --no-mmap > $O/single_${q}_$tag.json 2>>$O/log.txt
  echo "$q $tag done" >> $O/log.txt
}
run stock Q4_0 ROW1_GGML=0
run out Q4_0 ROW1_GGML=0 ROW1_OUT=1
run a4out Q4_0 ROW1_GGML=A ROW1_OUT=1
run stock Q4_K_M ROW1_GGML=0
run out Q4_K_M ROW1_GGML=0 ROW1_OUT=1
echo ALLDONE >> $O/log.txt
