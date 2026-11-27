"""PF1: one 32-client wave's timeline, short and long requests apart (2026-11-24).

Joins three records on the monotonic clock of one machine, by the server's request id (the SSE chunks' "id"):
the client's (lab/pk1/http_bench.py --timeline: send, every content chunk, done), the service's stability log ("end"
notes: t_submit, t_dispatch, t_first, t_end) and the engine trace (IRONMULE_GGML_TRACE: one line per native call with
prompt and decode rows, one "# job" line per request with its prefill start and end). For vLLM only the client record
exists. Per wave: the wave time (first send to last done; no sums of medians or of overlapping GPU time), per class
the individual and median time to first token and completion, the end of the last prefill and how many tokens each
short request had received by then, and every GPU call of the wave.

  python lab/pf/analyze.py --timeline TL.jsonl [--stability stability-log.jsonl --trace trace.txt] [--width 32]
"""
import argparse
import json
import statistics as st
from collections import defaultdict

SHORT = 200  # prompt tokens: the benchmark's short prompts have about 33, its long ones about 440


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 1) if xs else None


def load_trace(path):
    calls, jobs = [], {}
    for line in open(path):
        if line.startswith("# job "):
            kv = dict(p.split("=", 1) for p in line[6:].split())
            jobs[kv["tag"]] = {k: (float(v) if k in ("prefill_start", "prefill_end", "first", "end") else v)
                               for k, v in kv.items()}
        elif line.strip() and not line.startswith("#"):
            p = line.split()
            c = {"start": float(p[0]), "sec": float(p[1]), "rows": int(p[2]), "seqs": int(p[3])}
            if len(p) >= 7:
                c.update(prompt_rows=int(p[5]), decode_rows=int(p[6]))
            if len(p) >= 9:
                c.update(filler_rows=int(p[7]))
            calls.append(c)
    return calls, jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--stability", default="")
    ap.add_argument("--trace", default="")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    reqs = [json.loads(l) for l in open(a.timeline)]
    server = {}
    if a.stability:
        for l in open(a.stability):
            r = json.loads(l)
            if r.get("event") == "end" and "t_submit" in r:
                server[r["id"]] = r
    calls, jobs = load_trace(a.trace) if a.trace else ([], {})
    waves = defaultdict(list)
    for r in reqs:
        if r["width"] == a.width:
            waves[(r["label"], r["repeat"])].append(r)
    report = []
    for (label, rep), rs in sorted(waves.items()):
        t0 = min(r["t_send"] for r in rs)
        t1 = max(r["t_done"] for r in rs)
        rows = []
        for r in rs:
            sv, jb = server.get(r["server_id"]), jobs.get(r["server_id"])
            row = {"id": r["server_id"], "prompt": r["prompt_tokens"], "kind": "short" if r["prompt_tokens"] < SHORT else "long",
                   "send": r["t_send"] - t0, "first": r["chunk_times"][0] - t0 if r["chunk_times"] else None,
                   "done": r["t_done"] - t0, "ttft": r["ttft_ms"] / 1e3, "chunks": [c - t0 for c in r["chunk_times"]]}
            if sv:
                row.update(server_queue=(sv["t_dispatch"] - sv["t_submit"]) if sv.get("t_dispatch") else None,
                           server_in=sv["t_submit"] - t0)
            if jb:
                row.update(gpu_first=jb["prefill_start"] - t0 if jb["prefill_start"] else None,
                           prefill_end=jb["prefill_end"] - t0 if jb["prefill_end"] else None,
                           engine_first=jb["first"] - t0 if jb["first"] else None, engine_end=jb["end"] - t0)
            rows.append(row)
        longs = [r for r in rows if r["kind"] == "long"]
        shorts = [r for r in rows if r["kind"] == "short"]
        # the end of the last prefill: the engine's own record where there is one, else the last long request's first
        # token at the client (a prefill ends before its first token is streamed)
        engine_end = max((r["prefill_end"] for r in rows if r.get("prefill_end") is not None), default=None)
        client_end = max(r["first"] for r in longs) if longs else None
        cut = engine_end if engine_end is not None else client_end
        for r in shorts:
            r["tokens_at_last_prefill_end"] = sum(1 for c in r["chunks"] if c <= cut) if cut is not None else None
            r["tokens_at_last_long_first_token"] = sum(1 for c in r["chunks"] if c <= client_end) if client_end else None
        wave_calls = [c for c in calls if t0 <= c["start"] <= t1]
        gpu = [{"at": round(c["start"] - t0, 4), "ms": round(1e3 * c["sec"], 2), "rows": c["rows"], "seqs": c["seqs"],
                "prompt_rows": c.get("prompt_rows"), "decode_rows": c.get("decode_rows"),
                "filler_rows": c.get("filler_rows")} for c in wave_calls]
        summary = {"label": label, "repeat": rep, "wave_s": round(t1 - t0, 4), "requests": len(rows),
                   "last_prefill_end_s": None if cut is None else round(cut, 4),
                   "last_prefill_end_source": "engine" if engine_end is not None else "client (last long first token)",
                   "last_long_first_token_s": None if client_end is None else round(client_end, 4)}
        for name, group in (("short", shorts), ("long", longs)):
            summary[name] = {
                "n": len(group),
                "ttft_s": sorted(round(r["ttft"], 3) for r in group), "ttft_median_s": med([r["ttft"] for r in group]),
                "done_s": sorted(round(r["done"], 3) for r in group), "done_median_s": med([r["done"] for r in group]),
                "server_queue_median_ms": med([1e3 * r["server_queue"] for r in group if r.get("server_queue") is not None]),
                "gpu_first_median_s": med([r["gpu_first"] for r in group if r.get("gpu_first") is not None]),
                "prefill_end_median_s": med([r["prefill_end"] for r in group if r.get("prefill_end") is not None]),
            }
        summary["short"]["tokens_at_last_prefill_end"] = sorted(r["tokens_at_last_prefill_end"] for r in shorts
                                                                if r.get("tokens_at_last_prefill_end") is not None)
        summary["short"]["tokens_at_last_long_first_token"] = sorted(
            r["tokens_at_last_long_first_token"] for r in shorts if r.get("tokens_at_last_long_first_token") is not None)
        summary["gpu_calls"] = gpu
        report.append({"summary": summary, "requests": rows})
        s = dict(summary)
        s.pop("gpu_calls")
        print(json.dumps(s))
        if gpu:
            pre = [g for g in gpu if (g["prompt_rows"] or 0) > 0]
            dec = [g for g in gpu if (g["prompt_rows"] or 0) == 0]
            print(json.dumps({"label": label, "repeat": rep, "prefill_calls": pre,
                              "decode_calls": len(dec), "decode_median_ms": med([g["ms"] for g in dec]),
                              "decode_rows": sorted({g["rows"] for g in dec})}))
    if a.out:
        json.dump(report, open(a.out, "w"))


if __name__ == "__main__":
    main()
