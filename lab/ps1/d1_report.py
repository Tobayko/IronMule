"""PS1 diagnostics: per wave of lab/ps1/growth.py, the prefill call and its TL1 sections (ms) and the TL2 growth events.

TL1 ubatch fields: offset:tokens:seqs:reused:build:inputs:launch:out:graph:cpu:shape (us); build includes graph (the
graph's construction) and the scheduler's allocation (build - graph); out waits for the GPU and copies the outputs.

  python lab/ps1/d1_report.py growth_b.jsonl [growth_e.jsonl ...]
"""
import json
import sys


def ubatch(line):
    p = line.split()
    split, total = int(p[2]), int(p[3])
    ubs = []
    for u in p[5:]:
        f = u.split(":")
        build, graph = int(f[4]), int(f[8])
        ubs.append({"rows": int(f[1]), "split_ms": round(split / 1e3, 1), "graph_ms": round(graph / 1e3, 1),
                    "alloc_ms": round((build - graph) / 1e3, 1), "inputs_ms": round(int(f[5]) / 1e3, 1),
                    "launch_ms": round(int(f[6]) / 1e3, 1), "out_ms": round(int(f[7]) / 1e3, 1),
                    "cpu_ms": round(int(f[9]) / 1e3, 1), "rects": f[10].rstrip(",")})
    return {"total_ms": round(total / 1e3, 1), "ubatches": ubs}


for path in sys.argv[1:]:
    print("##", path)
    for line in open(path):
        r = json.loads(line)
        if "engine" in r:
            print("ENGINE", json.dumps(r["engine"]))
            continue
        big = [ubatch(x) for x in r["tl1"] if int(x.split()[4]) >= 1]
        big = sorted((b for b in big if any(u["rows"] > 64 for u in b["ubatches"])),
                     key=lambda b: -max(u["rows"] for u in b["ubatches"]))
        grow = [" ".join(x.split()[2:]) for x in r["tl2"]]
        print(json.dumps({"wave": r["wave"], "requests": r["requests"], "prefill_ms": [c["ms"] for c in r["prefill"]],
                          "tl1": big[:1], "growth": grow}))
