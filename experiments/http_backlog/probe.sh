#!/bin/bash
# TEST1: run the 429-saturation test repeatedly and count listen-queue overflows.
# usage: experiments/http_backlog/probe.sh <label> <runs> <cpu-burners>
# Linux only (reads TcpExt ListenOverflows from /proc/net/netstat). Run from the repo root.
set -u
label=$1; runs=$2; burners=$3
PY=${PYTHON:-python}
TEST="tests/engine/test_product_http.py::test_handler_saturation_returns_429_then_recovers"
ov() { awk '/^TcpExt:/{if(!h){split($0,k);h=1}else{split($0,v);for(i in k)if(k[i]=="ListenOverflows")print v[i]}}' /proc/net/netstat; }
pids=()
for _ in $(seq 1 "$burners"); do python3 -c "while True: pass" & pids+=($!); done
for r in $(seq 1 "$runs"); do
  a=$(ov); t0=$(date +%s.%N)
  out=$("$PY" -m pytest -n 0 -p no:cacheprovider "$TEST" 2>&1 | tail -1)
  t1=$(date +%s.%N); b=$(ov)
  printf "%s run=%d burners=%d wall=%.2fs overflows=%d result=%s\n" \
    "$label" "$r" "$burners" "$(echo "$t1-$t0" | bc)" $((b - a)) "$out"
done
for p in "${pids[@]}"; do kill "$p"; done
