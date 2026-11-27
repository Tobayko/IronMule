"""ADM2 (2026-11-25): the session's rules, applied as fixed in lab/adm2/a100_adm2.sh before the pod.

  python lab/adm2/verdict.py pick PAIRED_32 PAIRED_8 --ref R --singles A P [--combo AP]   -> the arm kept
  python lab/adm2/verdict.py arm NAME NAME=JSON ...                                      -> that arm's JSON
  python lab/adm2/verdict.py final DIR --final F --ref R                                  -> the HTTP result
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys


def waves(path):
    return [json.loads(line) for line in open(path) if line.startswith('{"arm"')]


def passes(x, ref, w32, w8, out):
    rounds = sorted({r["round"] for r in w32})
    d = []
    for k in rounds:
        a = [r["wall_s"] for r in w32 if r["arm"] == x and r["round"] == k]
        b = [r["wall_s"] for r in w32 if r["arm"] == ref and r["round"] == k]
        if a and b:
            d.append(a[0] - b[0])
    saving = -st.median(d) if d else 0.0
    faster = sum(1 for v in d if v < 0)
    m8 = lambda arm: st.median(r["wall_s"] for r in w8 if r["arm"] == arm)  # noqa: E731
    t32 = lambda arm: st.median(r["ttft_p50_ms"] for r in w32 if r["arm"] == arm)  # noqa: E731
    q = lambda arm: [int(r["quality"].split("/")[0]) for r in w8 if r["arm"] == arm and "quality" in r]  # noqa: E731
    checks = {"saving_ms": round(1e3 * saving, 1), "faster_rounds": f"{faster}/{len(d)}",
              "wave8_ratio": round(m8(x) / m8(ref), 4), "ttft32_ratio": round(t32(x) / t32(ref), 4),
              "facts": q(x), "facts_ref": q(ref)}
    ok = (saving >= 0.010 and faster >= 0.75 * len(d) and checks["wave8_ratio"] <= 1.02
          and checks["ttft32_ratio"] <= 1.05 and min(q(x) or [0]) >= min(18, min(q(ref) or [18])))
    out[x] = {**checks, "passes": ok}
    return ok, saving


def pick(a):
    w32, w8 = waves(a.paired32), waves(a.paired8)
    out = {}
    ok = {x: passes(x, a.ref, w32, w8, out) for x in a.singles + ([a.combo] if a.combo else [])}
    singles = [x for x in a.singles if ok[x][0]]
    chosen = a.ref
    if a.combo and len(singles) == len(a.singles) and ok[a.combo][0] and ok[a.combo][1] >= max(ok[x][1] for x in singles):
        chosen = a.combo
    elif singles:
        chosen = max(singles, key=lambda x: ok[x][1])
    print(json.dumps({"rule": out, "chosen": chosen}), file=sys.stderr)
    print(chosen)


def arm(a):
    for spec in a.specs:
        name, _, values = spec.partition("=")
        if name == a.name:
            print(values)
            return
    sys.exit(f"no arm {a.name}")


def final(a):
    def rows(pattern):
        out = []
        for path in sorted(glob.glob(f"{a.dir}/{pattern}")):
            out += [json.loads(line) for line in open(path) if '"tok_s"' in line]
        return out
    res = {}
    for label, pat32, pat8 in ((a.final, f"http_{a.final}[0-9]_32.jsonl", f"http_{a.final}[0-9]q_8.jsonl"),
                               (a.ref, f"http_{a.ref}[0-9]_32.jsonl", f"http_{a.ref}[0-9]q_8.jsonl"),
                               ("vllm", "http_vllm_32.jsonl", "http_vllmq_8.jsonl")):
        r32, r8 = rows(pat32), rows(pat8)
        if not r32:
            continue
        res[label] = {"tok_s_32": [r["tok_s"] for r in r32], "wall_32": [r["wall_s"] for r in r32],
                      "tok_s_32_median": st.median(r["tok_s"] for r in r32),
                      "wall_32_median": st.median(r["wall_s"] for r in r32),
                      "ttft_p50_32_median": st.median(r["ttft_p50_ms"] for r in r32),
                      "by_position": {str(k): st.median(r["wall_s"] for r in r32 if r["repeat"] == k)
                                      for k in sorted({r["repeat"] for r in r32})},
                      "tok_s_8_median": st.median(r["tok_s"] for r in r8) if r8 else None,
                      "wall_8": [r["wall_s"] for r in r8], "facts_8": [r.get("quality") for r in r8 if r.get("quality")]}
    if a.final in res and "vllm" in res:
        f, v = res[a.final], res["vllm"]
        res["verdict"] = {"ratio_tok_s": round(f["tok_s_32_median"] / v["tok_s_32_median"], 4),
                          "target_reached": f["tok_s_32_median"] >= 1.10 * v["tok_s_32_median"],
                          "gap_ms": round(1e3 * (f["wall_32_median"] - v["wall_32_median"] / 1.10), 1),
                          "gap_ms_by_position": {k: round(1e3 * (f["by_position"][k] - v["by_position"][k] / 1.10), 1)
                                                 for k in f["by_position"] if k in v["by_position"]}}
        if a.ref in res:
            res["verdict"]["final_vs_ref_tok_s"] = round(f["tok_s_32_median"] / res[a.ref]["tok_s_32_median"], 4)
            if f["tok_s_8_median"] and res[a.ref]["tok_s_8_median"]:
                res["verdict"]["final_vs_ref_8"] = round(f["tok_s_8_median"] / res[a.ref]["tok_s_8_median"], 4)
    print(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="op", required=True)
    k = s.add_parser("pick")
    k.add_argument("paired32")
    k.add_argument("paired8")
    k.add_argument("--ref", required=True)
    k.add_argument("--singles", nargs="+", required=True)
    k.add_argument("--combo")
    r = s.add_parser("arm")
    r.add_argument("name")
    r.add_argument("specs", nargs="+")
    f = s.add_parser("final")
    f.add_argument("dir")
    f.add_argument("--final", required=True)
    f.add_argument("--ref", required=True)
    a = p.parse_args()
    {"pick": pick, "arm": arm, "final": final}[a.op](a)


if __name__ == "__main__":
    main()
