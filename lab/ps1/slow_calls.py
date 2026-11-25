"""PS1: the large prefill calls of every 32-client wave of the H stage (engine traces), by wave position, and those
above 1.15 x the median of their position (reported apart; nothing is removed from any total).

  python lab/ps1/slow_calls.py DIR LABEL...      (DIR holds tl_<label>_32.jsonl and trace_<label>.txt)
"""
import json
import statistics as st
import sys
from collections import defaultdict

d, labels = sys.argv[1], sys.argv[2:]
by_pos = defaultdict(list)
for lab in labels:
    waves = defaultdict(list)
    for line in open(f"{d}/tl_{lab}_32.jsonl"):
        r = json.loads(line)
        waves[(r["repeat"], r["wave_start"])].append(r)
    calls = [(float(p[0]), float(p[1]), int(p[2])) for p in (l.split() for l in open(f"{d}/trace_{lab}.txt"))
             if p and not p[0].startswith("#") and len(p) > 5 and int(p[5]) > 0]
    for (rep, w0), rs in waves.items():
        t0, t1 = min(r["t_send"] for r in rs), max(r["t_done"] for r in rs)
        for t, sec, rows in calls:
            if t0 <= t <= t1 and rows > 300:
                by_pos[rep].append({"label": lab, "rows": rows, "ms": round(1e3 * sec, 1)})
for pos, cs in sorted(by_pos.items()):
    med = st.median(c["ms"] for c in cs)
    slow = [c for c in cs if c["ms"] > 1.15 * med]
    print(json.dumps({"position": pos, "median_ms": med, "calls": cs, "slow": slow}))
