"""CMP1 tables from bench.py's JSON files: median tokens/s and mean first token of the 3 blocks per engine, workload
and N, the Poisson and single-request rows, the facts, and each engine's ratio to IronMule. Usage: table.py DIR"""
import json
import statistics
import sys
from pathlib import Path

folder = Path(sys.argv[1])
runs = {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json")) if p.stem != "ironmule_health"}
order = [k for k in ("ironmule", "llama", "llamafa", "ollama", "lmstudio", "vllm", "sglang") if k in runs]
names = {k: runs[k].get("label", k) for k in order}
for workload in ("burst-sys", "burst"):
    ns = sorted({int(n) for k in order for n in runs[k].get(workload, {})})
    print(f"\n{workload}: tokens/s (mean first token)\n")
    print("| Engine | " + " | ".join(f"N={n}" for n in ns) + " |")
    print("| --- |" + " --- |" * len(ns))
    for k in order:
        cells = []
        for n in ns:
            blocks = runs[k].get(workload, {}).get(str(n))
            if not blocks:
                cells.append("-")
                continue
            tok = statistics.median(b["tok_s"] for b in blocks)
            ttft = statistics.median(b["ttft_mean_s"] for b in blocks)
            base = runs["ironmule"].get(workload, {}).get(str(n)) if "ironmule" in runs else None
            ratio = (statistics.median(b["tok_s"] for b in base) / tok) if base and k != "ironmule" else None
            cells.append(f"{tok:.1f} ({ttft:.2f} s)" + (f", IronMule {ratio:.2f}x" if ratio else ""))
        print(f"| {names[k]} | " + " | ".join(cells) + " |")
print("\nsingle request, Poisson 0.5/s, facts\n")
print("| Engine | single: decode tok/s, first token | Poisson: tok/s, first token p50 / p90, end to end p90 | facts |")
print("| --- | --- | --- | --- |")
for k in order:
    s, p, f = runs[k].get("single"), (runs[k].get("poisson") or {}).get("0.5"), runs[k].get("facts")
    sc = f"{s['decode_tok_s']:.1f}, {s['ttft_s']:.2f} s" if s else "-"
    pc = (f"{p['tok_s']:.1f}, {p['ttft_p50']:.2f} / {p['ttft_p90']:.2f} s, {p['e2e_p90']:.1f} s" if p else "-")
    fc = f"{f['correct']}/{f['of']}" if f else "-"
    print(f"| {names[k]} | {sc} | {pc} | {fc} |")
