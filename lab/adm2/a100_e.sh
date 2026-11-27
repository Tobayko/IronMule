#!/usr/bin/env bash
# ADM2 phase E (2026-11-25, same pod, within the 2.5 h watchdog): F against F + PX2 in one server start (12 paired
# rounds of 32 clients; PX2 stays on by default unless F + PX2's median paired wave is 10 ms or more longer), and the
# host time of a full-width decode step by part (IRONMULE_GGML_STEPPROF, the trace's "# stepprof" lines).
set -uo pipefail
W=/work/adm2
cd /work/IronMule
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
M=/models/Qwen3-32B-Q4_K_M.gguf
ST=$W/st
SW=$W/arms_e.json
echo "{}" > $SW
stamp() { echo "== $(date +%H:%M:%S) $*"; }
(unset IRONMULE_COMPARE_LOCAL_KEY; env IRONMULE_GGML_SEQ_CTX=2048 IRONMULE_ADMISSION_LEARN=0 IRONMULE_GGML_TRACE=$W/trace_E0.txt \
   IRONMULE_GGML_STEPPROF=1 IRONMULE_LAB_SWITCH=$SW \
   python -m ironmule_cli serve --gguf $M --state-dir $ST --agent on --no-thinking \
   --api-key-env IRONMULE_COMPARE_LOCAL_KEY --host 127.0.0.1 --port 8002 > $W/serve_E0.log 2>&1) &
t0=$(date +%s)
until grep -q '"ready": true' $W/serve_E0.log || ! pgrep -f "[i]ronmule_cli serve" >/dev/null; do sleep 5; done
stamp "E0 startup $(( $(date +%s) - t0 )) s"
python lab/adm2/paired.py --url http://127.0.0.1:8002 --model gguf/Qwen3-32B-Q4_K_M --switch $SW --width 32 --rounds 12 \
  --out $W/paired_E_32.jsonl --timeline $W/tl_paired_E_32.jsonl --seed 20261201 \
  --arms 'F={"ADM1":1,"ADM2":1,"PX2":0,"BF1":1}' 'FP={"ADM1":1,"ADM2":1,"PX2":1,"BF1":1}' 2>$W/paired_E_32.err | grep SUMMARY
grep stepprof $W/trace_E0.txt | tail -5
pkill -f "[i]ronmule_cli serve"; sleep 10
(cd $W && tar czf /work/adm2_e.tgz paired_E_32.jsonl tl_paired_E_32.jsonl trace_E0.txt serve_E0.log)
stamp A100-E-DONE
