"""PF2: the time balance of each 32- or 8-client wave along its own critical path, and what a server start ran before
serving (2026-11-24).

Joins, on the monotonic clock of one machine, the client timeline (lab/pk1/http_bench.py --timeline), the service's
stability log ("end": t_submit, t_dispatch, t_first, t_end) and the engine trace (IRONMULE_GGML_TRACE: per native call
its rows, prompt / decode / placeholder rows and b= host batch build, l= llama_decode, r= readback in ms; "# job" lines
with seen / prefill / first / end; "# gather start" / "# gathered" notes; "# init" stage notes and choices). The
balance is a chain of consecutive intervals on one time axis (no sums of overlapping or per-request times):
  burst      first send -> last send (client)
  to_engine  last send -> the last request seen by the engine
  gather     gather start -> gathered (the engine's window), if it ran
  to_call    end of the window (or last seen) -> start of the first prefill call
  prefill    first prefill call start -> last prefill call end, itself split into each call's b / l / r and the
             gaps between calls
  to_first   last prefill call end -> the last first token at the client (engine first, service, client)
  decode     the last first token -> wave end (client)

  python lab/pf2/balance.py --timeline TL.jsonl --stability S.jsonl --trace TRACE [--width 32]
"""
import argparse
import json
import statistics as st
from collections import defaultdict


def load_trace(path):
    calls, jobs, notes, inits = [], {}, [], []
    for line in open(path):
        if line.startswith("# job "):
            kv = dict(p.split("=", 1) for p in line[6:].split())
            jobs[kv["tag"]] = {k: float(v) for k, v in kv.items() if k in ("seen", "prefill_start", "prefill_end", "first", "end")}
            jobs[kv["tag"]].update(prompt=int(kv["prompt"]), reused=int(kv["reused"]))
        elif line.startswith("# init result "):
            inits.append(("result", None, json.loads(line[len("# init result "):])))
        elif line.startswith("# init "):
            p = line.split()
            inits.append((p[2], float(p[-1]), None))
        elif line.startswith("# gather start") or line.startswith("# gathered"):
            p = line.split()
            notes.append(("start" if p[1] == "gather" else "end", int(p[-2]), float(p[-1])))
        elif line.strip() and not line.startswith("#"):
            p = line.split()
            c = {"start": float(p[0]), "sec": float(p[1]), "rows": int(p[2]), "seqs": int(p[3]),
                 "prompt_rows": int(p[5]), "decode_rows": int(p[6]), "filler_rows": int(p[7]) if len(p) > 7 and "=" not in p[7] else 0}
            for kv in p[8:]:
                if "=" in kv and kv[0] in "blr" and kv[1] == "=":
                    c[kv[0]] = float(kv[2:])
            calls.append(c)
    return calls, jobs, notes, inits


def init_stages(inits):
    """The stages of each engine start in the trace (the serving engine is the last one before the waves)."""
    starts, cur = [], None
    for name, t, res in inits:
        if name == "warm_up":
            cur = {"stages": [], "result": None}
            starts.append(cur)
        if cur is None:
            continue
        if res is not None:
            cur["result"] = res
        else:
            cur["stages"].append((name, t))
    out = []
    for s in starts:
        st_ = s["stages"]
        out.append({"stages_s": {a[0]: round(b[1] - a[1], 3) for a, b in zip(st_, st_[1:])},
                    "start_t": st_[0][1] if st_ else None, "result": s["result"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--stability", default="")
    ap.add_argument("--trace", default="")
    ap.add_argument("--width", type=int, default=0, help="only this width (0: all)")
    a = ap.parse_args()
    reqs = [json.loads(l) for l in open(a.timeline)]
    server = {}
    if a.stability:
        for l in open(a.stability):
            r = json.loads(l)
            if r.get("event") == "end" and "t_submit" in r:
                server[r["id"]] = r
    calls, jobs, notes, inits = load_trace(a.trace) if a.trace else ([], {}, [], [])
    if inits:
        for i, s in enumerate(init_stages(inits)):
            print("INIT " + json.dumps({"engine_start": i, **s}))
    waves = defaultdict(list)
    for r in reqs:
        if not a.width or r["width"] == a.width:
            waves[(r["label"], r["width"], r["repeat"])].append(r)
    for (label, width, rep), rs in sorted(waves.items()):
        t0 = min(r["t_send"] for r in rs)
        t1 = max(r["t_done"] for r in rs)
        last_send = max(r["t_send"] for r in rs)
        firsts = [r["chunk_times"][0] for r in rs if r["chunk_times"]]
        js = [jobs[r["server_id"]] for r in rs if r["server_id"] in jobs]
        sv = [server[r["server_id"]] for r in rs if r["server_id"] in server]
        out = {"label": label, "width": width, "repeat": rep, "wave_s": round(t1 - t0, 4),
               "prompt_tokens": sum(r["prompt_tokens"] for r in rs), "requests": len(rs)}
        chain = {"burst": last_send - t0}
        if js:
            wn = [n for n in notes if t0 <= n[2] <= t1]
            gs = [n for n in wn if n[0] == "start"]
            ge = [n for n in wn if n[0] == "end"]
            pre = [c for c in calls if t0 <= c["start"] <= t1 and c["prompt_rows"] > 0]
            dec = [c for c in calls if t0 <= c["start"] <= t1 and c["prompt_rows"] == 0]
            first_call = pre[0]["start"] if pre else None
            # the requests the engine had seen when its first prefill call began (all of a gathered burst)
            before = [j["seen"] for j in js if first_call is None or j["seen"] <= first_call] or [j["seen"] for j in js]
            seen = max(before)
            chain["to_engine"] = seen - last_send
            out["seen_before_first_call"] = len(before)
            if gs and ge:
                chain["gather"] = ge[-1][2] - gs[0][2]
                chain["to_call"] = (first_call - ge[-1][2]) if first_call else None
                out["gather"] = {"start_after_t0": round(gs[0][2] - t0, 4), "jobs_at_start": gs[0][1],
                                 "jobs_gathered": ge[-1][1], "windows": len(gs)}
            elif first_call:
                chain["to_call"] = first_call - seen
            if pre:
                end_pre = max(c["start"] + c["sec"] for c in pre)
                chain["prefill"] = end_pre - pre[0]["start"]
                out["prefill_calls"] = [{"at": round(c["start"] - t0, 4), "ms": round(1e3 * c["sec"], 2), "rows": c["rows"],
                                         "seqs": c["seqs"], "prompt_rows": c["prompt_rows"], "decode_rows": c["decode_rows"],
                                         "filler_rows": c["filler_rows"], "b_ms": c.get("b"), "l_ms": c.get("l"),
                                         "r_ms": c.get("r")} for c in pre]
                gaps = [pre[i + 1]["start"] - (pre[i]["start"] + pre[i]["sec"]) for i in range(len(pre) - 1)]
                out["prefill_gaps_ms"] = [round(1e3 * g, 2) for g in gaps]
                chain["to_first"] = max(firsts) - end_pre
                chain["first_engine_after_prefill_ms"] = round(1e3 * (max(j["first"] for j in js) - end_pre), 2)
                if sv:
                    chain["first_service_after_prefill_ms"] = round(1e3 * (max(s["t_first"] for s in sv) - end_pre), 2)
            if dec:
                full = [c for c in dec if c["rows"] == width]
                out["decode"] = {"calls": len(dec), "full_width_calls": len(full),
                                 "full_width_median_ms": round(1e3 * st.median([c["sec"] for c in full]), 2) if full else None}
        chain["decode"] = t1 - max(firsts)
        out["chain_s"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in chain.items() if v is not None}
        out["last_first_token_s"] = round(max(firsts) - t0, 4)
        print(json.dumps(out))


if __name__ == "__main__":
    main()
