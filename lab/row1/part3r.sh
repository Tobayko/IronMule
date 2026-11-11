#!/bin/bash
# Round 3 speed tables on 1.7B: rows table and serving (-fa, --no-mmap + hugetlb), stock vs B16 (dynamic work,
# row blocks) vs B8S (segment 128); prefill for B8S.
SP=/tmp/claude-0/-home-user-IronMule/6e4a0a70-e763-5189-b9a9-9d8ce1e4fff1/scratchpad
source $SP/gg3env.sh; export LLAMA_CPP_LIB_PATH=$SP/row1/llama.cpp/build/bin GLIBC_TUNABLES=glibc.malloc.hugetlb=1
SRV=$SP/row1/llama.cpp/build/bin/llama-server; G=$SP/gg3_gguf/Qwen3-1.7B-Q8_0.gguf; O=$SP/row1/p4
cd /home/user/IronMule
for g in 0 B16 B8S; do
  ROW1_SEG=128 ROW1_GGML=$g ROW1_MMAP=0 ROW1_RS=1,8,16,32,48 python lab/row1/rows_llama.py $G 2>/dev/null | tail -1 > $O/rows3_$g.json
  ROW1_SEG=128 ROW1_GGML=$g python lab/row1/serve.py $SRV $G 1.7B-$g-fa-hp load:8,16,32 -fa --no-mmap > $O/serve3_$g.json 2>>$O/log.txt
done
ROW1_SEG=128 ROW1_GGML=B8S python lab/row1/serve.py $SRV $G 1.7B-B8S prefill > $O/prefill3_B8S.json 2>>$O/log.txt
ROW1_GGML=0 python lab/row1/serve.py $SRV $G 1.7B-0-fa prefill -fa > $O/prefill3_0_fa.json 2>>$O/log.txt
ROW1_GGML=B16 python lab/row1/serve.py $SRV $G 1.7B-B16-fa prefill -fa > $O/prefill3_B16_fa.json 2>>$O/log.txt
echo ALLDONE >> $O/log.txt
