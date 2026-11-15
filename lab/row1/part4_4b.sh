#!/bin/bash
# Round 4 part 2: Qwen3-4B Q8_0, stock -fa vs B16 -fa, huge pages; quality, serving, prefill.
SP=${ROW1_WORK:-/tmp/row1-work}
source $SP/gg3env.sh; export LLAMA_CPP_LIB_PATH=$SP/row1/llama.cpp/build/bin GLIBC_TUNABLES=glibc.malloc.hugetlb=1 ROW1_CTXCAP=32768
SRV=$SP/row1/llama.cpp/build/bin/llama-server; G=$SP/row1/gguf/Qwen3-4B-Q8_0.gguf; O=$SP/row1/p5
cd "$(dirname "$0")/../.."
for g in 0 B16; do
  ROW1_GGML=$g python lab/row1/ppl.py $G 2>/dev/null > $O/ppl_4B_$g.json
  ROW1_GGML=$g python lab/row1/facts.py $G 2>/dev/null > $O/facts_4B_$g.json
  ROW1_GGML=$g python lab/row1/serve.py $SRV $G 4B-$g-fa-hp load:1,8,16,32 -fa --no-mmap > $O/serve_4B_$g.json 2>>$O/log4b.txt
  ROW1_GGML=$g python lab/row1/serve.py $SRV $G 4B-$g-hp prefill --no-mmap > $O/prefill_4B_$g.json 2>>$O/log4b.txt
  free -m | head -2 >> $O/log4b.txt
done
echo ALLDONE >> $O/log4b.txt
