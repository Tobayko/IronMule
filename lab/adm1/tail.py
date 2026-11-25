"""ADM1: a 32-client wave's end on one time axis (ms after the engine's last decode call ended): the service's
member_done notes (it read the done frame), its http notes ([DONE] written), and the client's t_done.

  python lab/adm1/tail.py TIMELINE TRACE STABILITY_LOG
"""
import json
import statistics as st
import sys
from collections import defaultdict

tl, tr, sl = sys.argv[1:4]
waves = defaultdict(list)
for line in open(tl):
    r = json.loads(line)
    if r["width"] == 32:
        waves[r["wave_start"]].append(r)
calls = []
for line in open(tr):
    if line.strip() and not line.startswith("#"):
        p = line.split()
        calls.append((float(p[0]), float(p[1]), int(p[2])))
ev = defaultdict(dict)
for line in open(sl):
    e = json.loads(line)
    if e["event"] in ("member_done", "http", "end"):
        ev[e["id"]][e["event"]] = e["t"]
for w0, rs in sorted(waves.items()):
    t0 = min(r["t_send"] for r in rs)
    end = max(r["t_done"] for r in rs)
    last = max((c for c in calls if t0 <= c[0] <= end), key=lambda c: c[0])
    le = last[0] + last[1]
    md = sorted(1e3 * (ev[r["server_id"]]["member_done"] - le) for r in rs if "member_done" in ev[r["server_id"]])
    ht = sorted(1e3 * (ev[r["server_id"]]["http"] - le) for r in rs if "http" in ev[r["server_id"]])
    cl = sorted(1e3 * (r["t_done"] - le) for r in rs)
    lt = sorted(1e3 * (r["chunk_times"][-1] - le) for r in rs)
    f = lambda v: [round(v[0], 1), round(st.median(v), 1), round(v[-1], 1)]  # noqa: E731
    print(json.dumps({"member_done": f(md), "last_chunk_client": f(lt), "http_note": f(ht), "client_done": f(cl)}))
