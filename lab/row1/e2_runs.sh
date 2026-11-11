#!/bin/bash
# E2 final: alternating repeats of the full-pass timing, huge pages on/off, both models; one JSON line per run.
SP=/tmp/claude-0/-home-user-IronMule/6e4a0a70-e763-5189-b9a9-9d8ce1e4fff1/scratchpad
cd "$(dirname "$0")"
for rep in 1 2 3; do for m in Qwen3-1.7B Qwen3-0.6B; do for hp in 1 0; do
  ROW1_HUGE=$hp ROW1_PF=16384 ROW1_REPS=5 python3 e2_pass.py $SP/gg3_gguf/$m-Q8_0.gguf 2>&1 | tail -1
done; done; done
