#!/bin/bash
# Part 3/4: rows table, serving, single request, prefill; stock (hook off) against A and B16. One job at a time.
SP=/tmp/claude-0/-home-user-IronMule/6e4a0a70-e763-5189-b9a9-9d8ce1e4fff1/scratchpad
source $SP/gg3env.sh; export LLAMA_CPP_LIB_PATH=$SP/row1/llama.cpp/build/bin
SRV=$SP/row1/llama.cpp/build/bin/llama-server
ORIG=$SP/lc1_src/llama_cpp_python-0.3.16/vendor/llama.cpp/build/bin/llama-server
O=$SP/row1/p3; mkdir -p $O; cd /home/user/IronMule
G17=$SP/gg3_gguf/Qwen3-1.7B-Q8_0.gguf; G06=$SP/gg3_gguf/Qwen3-0.6B-Q8_0.gguf
HP="GLIBC_TUNABLES=glibc.malloc.hugetlb=1"
run() { local out=$1; shift; [ -s $O/$out ] || env "$@" > $O/$out 2>>$O/log.txt; echo "$out $(date +%T)" >> $O/progress.txt; }
for m in 1.7B 0.6B; do G=$SP/gg3_gguf/Qwen3-$m-Q8_0.gguf
  for g in 0 A B16; do run rows_${m}_$g.json ROW1_GGML=$g python lab/row1/rows_llama.py $G; done
done
for g in 0 A B16; do run serve_1.7B_$g.json ROW1_GGML=$g python lab/row1/serve.py $SRV $G17 1.7B-$g load:1,8,16,32; done
for g in 0 B16; do run serve_1.7B_${g}_hp.json ROW1_GGML=$g $HP python lab/row1/serve.py $SRV $G17 1.7B-$g-hp load:1,8,16,32 --no-mmap; done
run serve_1.7B_orig.json python lab/row1/serve.py $ORIG $G17 1.7B-orig load:8
for g in 0 A B16; do run serve_0.6B_$g.json ROW1_GGML=$g python lab/row1/serve.py $SRV $G06 0.6B-$g load:1,8,16,32; done
run serve_0.6B_orig.json python lab/row1/serve.py $ORIG $G06 0.6B-orig load:8
for m in 1.7B 0.6B; do G=$SP/gg3_gguf/Qwen3-$m-Q8_0.gguf
  run single_${m}_0.json ROW1_GGML=0 python lab/row1/serve.py $SRV $G $m-0 single
  run single_${m}_0_hp.json ROW1_GGML=0 $HP python lab/row1/serve.py $SRV $G $m-0-hp single --no-mmap
  run single_${m}_B16_hp.json ROW1_GGML=B16 $HP python lab/row1/serve.py $SRV $G $m-B16-hp single --no-mmap
  run single_${m}_A_hp.json ROW1_GGML=A $HP python lab/row1/serve.py $SRV $G $m-A-hp single --no-mmap
  for g in 0 A B16; do
    run prefill_${m}_$g.json ROW1_GGML=$g python lab/row1/serve.py $SRV $G $m-$g prefill
    run prefill_${m}_${g}_b2048.json ROW1_GGML=$g python lab/row1/serve.py $SRV $G $m-$g-b2048 prefill -b 2048 -ub 2048
  done
done
echo ALLDONE >> $O/progress.txt
