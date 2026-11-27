"""ADM1/PX2 (2026-11-25): one 32-client wave on one time axis, from a client timeline and the engine trace.

  python lab/adm1/anatomy.py TIMELINE TRACE
Per wave (ms from the first send): the last send; when the engine had the whole burst ("gathered"); the large
prefill call's start and length; the decode calls (count, median native call, median start to start); the last
decode call's end; the last token at the client.
"""
import json
import statistics as st
import sys
from collections import defaultdict

tl, tr = sys.argv[1], sys.argv[2]
waves = defaultdict(list)
for line in open(tl):
    r = json.loads(line)
    if r["width"] == int(sys.argv[3] if len(sys.argv) > 3 else 32):
        waves[r["wave_start"]].append(r)
calls, gathered = [], []
for line in open(tr):
    if line.startswith("# gathered"):
        gathered.append(float(line.split()[-1]))
    elif line.strip() and not line.startswith("#"):
        p = line.split()
        calls.append((float(p[0]), float(p[1]), int(p[2]), int(p[3])))
out = []
for w0, rs in sorted(waves.items()):
    t0 = min(r["t_send"] for r in rs)
    end = max(r["t_done"] for r in rs)
    cs = [c for c in calls if t0 <= c[0] <= end]
    pre = [c for c in cs if c[2] > 300]
    dec = [c for c in cs if c[2] <= 64 and pre and c[0] > pre[-1][0]]
    g = [x for x in gathered if t0 - 0.05 <= x <= (pre[-1][0] if pre else end)]
    gaps = [b[0] - a[0] for a, b in zip(dec, dec[1:])]
    row = {"label": rs[0].get("label"), "wave_ms": round(1e3 * (end - t0)), "last_send": round(1e3 * (max(r["t_send"] for r in rs) - t0)),
           "gathered": round(1e3 * (g[-1] - t0)) if g else None,
           "prefill_calls": [(round(1e3 * (c[0] - t0)), round(1e3 * c[1]), c[2]) for c in pre],
           "prefill_end_to_first_decode": round(1e3 * (dec[0][0] - pre[-1][0] - pre[-1][1]), 1) if dec else None,
           "decode_calls": len(dec), "decode_native_med": round(1e3 * st.median(c[1] for c in dec), 2),
           "decode_period_med": round(1e3 * st.median(gaps), 2),
           "decode_span": round(1e3 * (dec[-1][0] + dec[-1][1] - dec[0][0])),
           "last_call_end_to_last_token": round(1e3 * (end - dec[-1][0] - dec[-1][1]), 1)}
    out.append(row)
    print(json.dumps(row))
