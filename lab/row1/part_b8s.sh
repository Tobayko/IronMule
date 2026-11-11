#!/bin/bash
# Round 3: B8s quality (perplexity + facts) for segment sizes 128, 256, 512 on both models.
SP=/tmp/claude-0/-home-user-IronMule/6e4a0a70-e763-5189-b9a9-9d8ce1e4fff1/scratchpad
source $SP/gg3env.sh; export LLAMA_CPP_LIB_PATH=$SP/row1/llama.cpp/build/bin
cd /home/user/IronMule
for m in Qwen3-0.6B Qwen3-1.7B; do for sg in 256 128 512; do
  ROW1_GGML=B8S ROW1_SEG=$sg python lab/row1/ppl.py $SP/gg3_gguf/$m-Q8_0.gguf 2>/dev/null > $SP/row1/p4/ppl_${m}_B8S$sg.json
  ROW1_GGML=B8S ROW1_SEG=$sg python lab/row1/facts.py $SP/gg3_gguf/$m-Q8_0.gguf 2>/dev/null > $SP/row1/p4/facts_${m}_B8S$sg.json
  echo "$m $sg done"
done; done
