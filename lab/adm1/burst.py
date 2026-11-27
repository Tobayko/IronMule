"""ADM1: fire the benchmark's 32-client bursts (compare.batch, the same client and prompts) at a running server and
join the client's send times with the stub's arrival log and the service's stability log.

  python lab/adm1/burst.py --url http://127.0.0.1:8010 --state /tmp/st --log /tmp/arrivals.jsonl --waves 5
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

from ironmule_product import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--waves", type=int, default=5)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--tokens", type=int, default=int(__import__("os").environ.get("TOKENS", 4)))
    a = p.parse_args()
    server = {"base_url": a.url, "model_id": "stub/model"}
    deadline = time.monotonic() + 600
    compare.stream("ironmule", server, compare.WARMUP, 4, deadline)
    out = []
    for w in range(a.waves):
        prompts = compare.performance_prompts(20261119 + w * 1009 + a.width, a.width, max(a.width, 8))
        Path(a.log).write_text("")
        w0 = time.monotonic()
        rows, _ = compare.batch("ironmule", server, prompts, a.width, a.tokens, deadline, timeline=True)
        time.sleep(0.2)
        arr = {}
        starts = []
        step_end = None
        for line in open(a.log):
            e = json.loads(line)
            if "last_step_end" in e:
                step_end = e["last_step_end"]
            elif "prefill_start" in e:
                starts.append((e["prefill_start"] - w0, e["n"]))
            else:
                arr[e["id"]] = e["t"] - w0
        admit = {}
        log = Path(a.state) / "stability-log.jsonl"
        for line in (open(log) if log.exists() else ()):
            e = json.loads(line)
            if e["event"] == "admit" and e["t"] >= w0:
                admit[e["id"]] = e["t"] - w0
        sends = sorted(r["t_send"] - w0 for r in rows)
        ad = sorted(admit.get(r["server_id"], float("nan")) for r in rows) if admit else [float("nan")] * len(rows)
        ar = sorted(arr.get(r["server_id"], float("nan")) for r in rows)
        rec = {"wave": w, "send_ms": [round(1e3 * sends[0], 1), round(1e3 * st.median(sends), 1), round(1e3 * sends[-1], 1)],
               "admit_ms": [round(1e3 * ad[0], 1), round(1e3 * st.median(ad), 1), round(1e3 * ad[-1], 1)],
               "worker_ms": [round(1e3 * ar[0], 1), round(1e3 * st.median(ar), 1), round(1e3 * ar[-1], 1)],
               "prefill_starts_ms": [(round(1e3 * t, 1), n) for t, n in starts],
               "tail_ms": round(1e3 * (max(r["t_done"] for r in rows) - step_end), 1) if step_end else None}
        out.append(rec)
        print(json.dumps(rec), flush=True)
    last = [r["worker_ms"][2] for r in out]
    print("SUMMARY " + json.dumps({"last_at_worker_ms_median": st.median(last),
                                   "last_send_ms_median": st.median(r["send_ms"][2] for r in out),
                                   "tail_ms_median": st.median(r["tail_ms"] for r in out if r["tail_ms"] is not None)}))


if __name__ == "__main__":
    main()
