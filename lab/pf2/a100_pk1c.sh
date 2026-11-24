#!/usr/bin/env bash
# PK1-C A/B on the PF2 pod (2026-11-24): server starts alternating reference (IRONMULE_GGML_PK1C=0) and PK1-C, the PF2
# session's state (gate, W4R tuning) reused, engine trace on; per start three 32-client waves and two 8-client waves
# with the facts graded (same prompts as PF2's starts).
#   bash lab/pf2/a100_pk1c.sh > /work/a100_pk1c.log 2>&1
set -uo pipefail
W=/work/pf2; cd /work/IronMule
T=$(ls -d ~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16)
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
stamp() { echo "== $(date +%H:%M:%S) $*"; }
stamp "$(git log --oneline -1)"
for arm in ref1:0 c1:1 ref2:0 c2:1; do
  name=${arm%%:*}; on=${arm##*:}
  (unset IRONMULE_COMPARE_LOCAL_KEY; env IRONMULE_GGML_SEQ_CTX=2048 IRONMULE_ADMISSION_LEARN=0 IRONMULE_GGML_PK1C=$on \
     IRONMULE_GGML_TRACE=$W/trace_$name.txt python -m ironmule_cli serve --gguf /models/Qwen3-32B-Q4_K_M.gguf --state-dir $W/st \
     --agent on --no-thinking --api-key-env IRONMULE_COMPARE_LOCAL_KEY --host 127.0.0.1 --port 8002 > $W/serve_$name.log 2>&1) &
  t0=$(date +%s)
  until grep -q '"ready": true' $W/serve_$name.log || ! pgrep -f "[i]ronmule_cli serve" >/dev/null; do sleep 5; done
  stamp "$name (PK1C=$on) startup $(( $(date +%s) - t0 )) s"
  python lab/pk1/http_bench.py --url http://127.0.0.1:8002 --engine ironmule --model gguf/Qwen3-32B-Q4_K_M \
    --widths 32 --repeats 3 --label $name --timeline $W/tl_${name}_32.jsonl > $W/http_${name}_32.jsonl 2>$W/http_${name}_32.err
  python lab/pk1/http_bench.py --url http://127.0.0.1:8002 --engine ironmule --model gguf/Qwen3-32B-Q4_K_M \
    --widths 8 --repeats 2 --quality --label ${name}q --timeline $W/tl_${name}_8.jsonl > $W/http_${name}_8.jsonl 2>$W/http_${name}_8.err
  python -c "import json,sys;[print(r.get('label'),r.get('warmup_s') or (r['repeat'],r['width'],r['tok_s'],r['wall_s'],r['ttft_p50_ms'],r['ttft_max_ms'],r.get('quality'),r.get('quality_misses'))) for f in sys.argv[1:] for r in map(json.loads,open(f))]" $W/http_${name}_32.jsonl $W/http_${name}_8.jsonl
  pkill -f "[i]ronmule_cli serve"; sleep 12
done
cp $W/st/stability-log.jsonl $W/stability-log.jsonl
for name in ref1 c1 ref2 c2; do for w in 32 8; do
  python lab/pf2/balance.py --timeline $W/tl_${name}_$w.jsonl --stability $W/stability-log.jsonl --trace $W/trace_$name.txt --width $w \
    | grep -v "^INIT" > $W/balance_${name}_$w.jsonl
  python -c "import json,sys;[print('$name', $w, d['repeat'], d['wave_s'], d['last_first_token_s'], [(c['rows'], round(c['ms'])) for c in d.get('prefill_calls') or []], d.get('gather')) for d in map(json.loads, open(sys.argv[1]))]" $W/balance_${name}_$w.jsonl
done; done
(cd $W && tar czf /work/pk1c.tgz trace_ref1.txt trace_c1.txt trace_ref2.txt trace_c2.txt tl_ref*.jsonl tl_c*.jsonl http_ref*.jsonl http_c*.jsonl balance_ref*.jsonl balance_c*.jsonl serve_ref*.log serve_c*.log stability-log.jsonl s4.log 2>/dev/null; cp /work/s4.log $W/ 2>/dev/null; tar czf /work/pk1c.tgz trace_ref1.txt trace_c1.txt trace_ref2.txt trace_c2.txt tl_ref*.jsonl tl_c*.jsonl http_ref*.jsonl http_c*.jsonl balance_ref*.jsonl balance_c*.jsonl serve_ref*.log serve_c*.log stability-log.jsonl s4.log tl_S4*.jsonl http_S4*.jsonl trace_S4.txt prefill_split.log trace_split.txt)
stamp A100-PK1C-DONE
