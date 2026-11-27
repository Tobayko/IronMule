"""PF3: what the engine thread did during long prefill calls (2026-11-24).

Joins three diagnostics of one server start on the monotonic clock: the engine trace (one line per native call, with
the pause probe's per-call deltas: thread CPU time, page faults, run delay, cgroup throttling), the TL1 lines
(llama.cpp's per-ubatch host timestamps: graph build/alloc, inputs, launch, output copy) and the probe's samples of
calls of 1 s or more (scheduler state, current system call, wait channel). For each prefill call: its duration
against the expectation from its prompt rows and passes (PF3 fit on pod 32: 0.3347 ms a row, 21.47 ms a pass), the
excess, the ubatch phase that holds the excess, and how the thread spent the call (sample shares by state/syscall).

A call counts as a pause when its excess is 700 ms or more. Classification of the excess, from the evidence only:
  compute        the thread is running or in its launch/sync calls as in normal calls, and no phase stands out
  cpu-wait       run delay or cgroup throttling covers most of the excess
  driver-call    the samples in the long phase sit in an ioctl on /dev/nvidia* (sleeping or uninterruptible)
  other-sync     the samples in the long phase sit in a futex or poll wait
  page-faults    major faults during the call
  open           none of these covers the excess

  python lab/pf3/pauses.py --trace trace_D1.txt --tl1 tl1_D1.txt --pause pause_D1.txt
"""
import argparse
import json
from collections import Counter

ROW_MS, PASS_MS = 0.3347, 21.47


def load_trace(path):
    calls = []
    for line in open(path):
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        kv = {}
        for x in p[8:]:
            if "=" in x:
                k, v = x.split("=", 1)
                try:
                    kv[k] = float(v)
                except ValueError:
                    kv[k] = v
        calls.append({"start": float(p[0]), "sec": float(p[1]), "rows": int(p[2]), "prompt_rows": int(p[5]), **kv})
    return calls


def load_tl1(path):
    out = []
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return out
    for line in lines:
        p = line.split()
        if len(p) < 5 or p[0] != "D":
            continue
        try:
            d = {"start": int(p[1]) / 1e6, "split_ms": int(p[2]) / 1e3, "total_ms": int(p[3]) / 1e3, "ub": []}
            for u in p[5:]:
                f = u.split(":")
                d["ub"].append({"at_ms": int(f[0]) / 1e3, "tok": int(f[1]), "seqs": int(f[2]), "reused": int(f[3]),
                                "build_ms": int(f[4]) / 1e3, "inputs_ms": int(f[5]) / 1e3,
                                "launch_ms": int(f[6]) / 1e3, "out_ms": int(f[7]) / 1e3})
        except (ValueError, IndexError):
            continue  # a line another process interleaved
        out.append(d)
    return out


def load_samples(path):
    calls, cur = [], None
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return calls
    for line in lines:
        if line.startswith("# call "):
            p = line.split()
            cur = {"start": float(p[3]), "sec": float(p[4]), "head": line, "samples": []}
            calls.append(cur)
        elif cur is not None and line.strip():
            p = line.split()
            if len(p) >= 4:
                cur["samples"].append({"t": float(p[0]), "state": p[1], "call": p[2], "wchan": p[3]})
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--tl1", default="")
    ap.add_argument("--pause", default="")
    ap.add_argument("--min-rows", type=int, default=300)
    a = ap.parse_args()
    tl1 = load_tl1(a.tl1) if a.tl1 else []
    samples = load_samples(a.pause) if a.pause else []
    for c in load_trace(a.trace):
        if c["prompt_rows"] < a.min_rows:
            continue
        end = c["start"] + c["sec"]
        d = next((x for x in tl1 if c["start"] - 0.002 <= x["start"] <= end), None)
        passes = len(d["ub"]) if d else None
        expect_ms = ROW_MS * c["rows"] + PASS_MS * (passes or 2)
        excess = 1e3 * c["sec"] - expect_ms
        rec = {"start": round(c["start"], 3), "ms": round(1e3 * c["sec"], 1), "rows": c["rows"], "passes": passes,
               "expect_ms": round(expect_ms, 1), "excess_ms": round(excess, 1),
               **{k: c[k] for k in ("cpu_u", "cpu_s", "minflt", "majflt", "pmajflt", "rundelay", "thr", "thr_ms")
                  if k in c}}
        long_phase = None
        if d:
            rec["split_ms"] = d["split_ms"]
            rec["ubatches"] = [{k: u[k] for k in ("tok", "seqs", "reused", "build_ms", "inputs_ms", "launch_ms",
                                                  "out_ms")} for u in d["ub"]]
            # the phase holding the excess: the single largest of build/inputs/launch/out over the ubatches, beyond
            # what the rows of that ubatch explain for a launch (launch time ~ rows when the queue is full)
            best = (0.0, None)
            for i, u in enumerate(d["ub"]):
                for ph in ("build_ms", "inputs_ms", "launch_ms", "out_ms"):
                    over = u[ph] - (ROW_MS * u["tok"] + PASS_MS if ph == "launch_ms" else 0)
                    if over > best[0]:
                        best = (over, (i, ph, u["at_ms"], u[ph]))
            if best[1]:
                i, ph, at, ms = best[1]
                long_phase = {"ubatch": i, "phase": ph, "from_ms": round(at, 1), "ms": round(ms, 1),
                              "beyond_rows_ms": round(best[0], 1)}
                rec["long_phase"] = long_phase
        s = next((x for x in samples if abs(x["start"] - c["start"]) < 0.01), None)
        if s:
            rec["samples"] = len(s["samples"])
            rec["share"] = {k: round(v / len(s["samples"]), 3) for k, v in Counter(
                f"{x['state']}:{x['call']}" for x in s["samples"]).most_common(6)} if s["samples"] else {}
            if long_phase and s["samples"]:
                # call-relative window of the long phase (the TL1 clock starts a few us after the trace's)
                lo = (d["start"] - c["start"]) + (long_phase["from_ms"] + 0) / 1e3
                in_phase = [x for x in s["samples"] if lo <= x["t"] <= lo + long_phase["ms"] / 1e3 + 0.05]
                rec["phase_share"] = {k: round(v / max(1, len(in_phase)), 3) for k, v in Counter(
                    f"{x['state']}:{x['call']}:{x['wchan']}" for x in in_phase).most_common(5)}
            rec["probe"] = s["head"].split("pyspy=")[-1]
        verdict = "normal"
        if excess >= 700:
            verdict = "open"
            if rec.get("majflt", 0) or rec.get("pmajflt", 0):
                verdict = "page-faults"
            elif sum(v for v in (rec.get("rundelay"), rec.get("thr_ms")) if isinstance(v, float)) >= 0.5 * excess:
                verdict = "cpu-wait"
            elif rec.get("phase_share"):
                top = max(rec["phase_share"], key=rec["phase_share"].get)
                if "ioctl:nvidia" in top:
                    verdict = "driver-call"
                elif ":futex" in top or ":poll" in top or ":ppoll" in top:
                    verdict = "other-sync"
                elif top.startswith("R:running") or top.startswith("R:"):
                    verdict = "compute-or-spin"
        rec["verdict"] = verdict
        print(json.dumps(rec))


if __name__ == "__main__":
    main()
