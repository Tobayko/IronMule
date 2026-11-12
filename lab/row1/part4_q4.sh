#!/bin/bash
# Round 4 part 1: stock llama.cpp on Qwen3-1.7B Q4_0 and Q4_K_M (huge pages): rows table, profile, single request.
SP=/tmp/claude-0/-home-user-IronMule/6e4a0a70-e763-5189-b9a9-9d8ce1e4fff1/scratchpad
source $SP/gg3env.sh; export LLAMA_CPP_LIB_PATH=$SP/row1/llama.cpp/build/bin GLIBC_TUNABLES=glibc.malloc.hugetlb=1 ROW1_GGML=0
SRV=$SP/row1/llama.cpp/build/bin/llama-server; O=$SP/row1/p5; mkdir -p $O
cd /home/user/IronMule
for q in Q4_0 Q4_K_M; do G=$SP/row1/gguf/Qwen3-1.7B-$q.gguf
  ROW1_MMAP=0 ROW1_RS=1,8,16,32 python lab/row1/rows_llama.py $G 2>$O/load_$q.log | tail -1 > $O/rows_$q.json
  ROW1_PROF=1 python lab/row1/prof_llama.py $G $O/prof_$q > $O/prof_$q.txt 2>/dev/null
  python lab/row1/serve.py $SRV $G $q single --no-mmap > $O/single_$q.json 2>>$O/log.txt
done
echo ALLDONE >> $O/log.txt
