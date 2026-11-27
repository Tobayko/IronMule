#!/usr/bin/env bash
# ADM1 local A/B: the stub server pinned to one core (as a busy pod's), the benchmark client on the others.
#   bash lab/adm1/run_ab.sh OUTDIR LABEL [env...]
set -u
O=$1; L=$2; shift 2
mkdir -p $O; rm -rf $O/st_$L
env "$@" taskset -c ${SRV_CPUS:-0} python lab/adm1/stub_server.py --port 8011 --state $O/st_$L --log $O/arr_$L.jsonl --tokens ${TOKENS:-4} > $O/srv_$L.log 2>&1 &
P=$!
until grep -q ready $O/srv_$L.log 2>/dev/null; do sleep 0.2; done
taskset -c ${CLI_CPUS:-1-3} python lab/adm1/burst.py --url http://127.0.0.1:8011 --state $O/st_$L --log $O/arr_$L.jsonl --waves ${WAVES:-15} > $O/burst_$L.jsonl
kill $P; wait $P 2>/dev/null
grep SUMMARY $O/burst_$L.jsonl | sed "s/^/$L /"
