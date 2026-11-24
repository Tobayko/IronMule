"""Three W4R tables from one tuning record (`python -m ironmule.w4_kernels` with IRONMULE_W4_PRE=1 measures every
variant, PS included): "base" as the tuner before 2026-11-24 chose (no width 24, no PS), "cand" (width 24, no PS) and
"ps" (width 24, PS allowed). Same measurements, so the arms differ only in what each tuner may choose.

  python lab/w4bw/tables.py RECORD.json LIB.so OUT_DIR
"""
import json
import sys
from pathlib import Path

from ironmule import w4_kernels as wk


def best(measured, keep):
    out = {}
    for r in measured:
        if r["rc"] != 0 or r["rel_err"] > wk.ERROR_BOUND or not keep(r):
            continue
        k = (r["type"], r["N"], r["K"], r["B"])
        if k not in out or r["us"] < out[k]["us"]:
            out[k] = r
    return list(out.values())


def main():
    record, lib, out = json.load(open(sys.argv[1])), sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    pre = {v["id"]: v.get("pre", 0) for v in wk.variants(wk._lib(lib))}
    arms = {"base": lambda r: r["B"] != 24 and not pre[r["variant"]],
            "cand": lambda r: not pre[r["variant"]],
            "ps": lambda r: True}
    for name, keep in arms.items():
        entries = best(record["measured"], keep)
        (out / f"{name}.table").write_text(wk.table_lines({"entries": entries}))
        ps = sum(1 for e in entries if pre[e["variant"]])
        print(json.dumps({"table": name, "entries": len(entries), "ps_entries": ps,
                          "w24_entries": sum(1 for e in entries if e["B"] == 24)}))


if __name__ == "__main__":
    main()
