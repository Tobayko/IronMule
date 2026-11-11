#!/bin/bash
# Part 2: perplexity and fact check, stock (hook off) and A, B, B16, both models, one job at a time.
SP=/tmp/claude-0/-home-user-IronMule/6e4a0a70-e763-5189-b9a9-9d8ce1e4fff1/scratchpad
source $SP/gg3env.sh; export LLAMA_CPP_LIB_PATH=$SP/row1/llama.cpp/build/bin
cd /home/user/IronMule
for m in Qwen3-0.6B Qwen3-1.7B; do for g in 0 A B B16; do
  [ -s $SP/row1/facts_${m}_$g.json ] || ROW1_GGML=$g python lab/row1/facts.py $SP/gg3_gguf/$m-Q8_0.gguf 2>/dev/null > $SP/row1/facts_${m}_$g.json
  ROW1_GGML=$g python lab/row1/ppl.py $SP/gg3_gguf/$m-Q8_0.gguf 2>/dev/null > $SP/row1/ppl_${m}_$g.json
  echo "$m $g done"
done; done
