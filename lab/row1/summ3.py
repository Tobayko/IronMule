"""Summarize part 3/4 outputs in $SP/row1/p3 (median of 3 blocks / runs)."""
import json, glob, os, statistics as st
O = "/tmp/claude-0/-home-user-IronMule/6e4a0a70-e763-5189-b9a9-9d8ce1e4fff1/scratchpad/row1/p3"
def last(f): return json.loads(open(f).read().strip().splitlines()[-1])
for f in sorted(glob.glob(f"{O}/serve_*.json")):
    d = last(f); cells = []
    for k in ("N1", "N8", "N16", "N32"):
        if k in d: cells.append(f"{k} {st.median(b['tok_s'] for b in d[k]):.1f} tok/s ttft {st.median(b['ttft_mean_s'] for b in d[k]):.2f}s")
    print(os.path.basename(f), " | ".join(cells))
for f in sorted(glob.glob(f"{O}/single_*.json")):
    d = last(f); print(os.path.basename(f), f"{st.median(b['tok_s'] for b in d['single']):.1f} tok/s", [round(b['tok_s'], 1) for b in d['single']])
for f in sorted(glob.glob(f"{O}/prefill_*.json")):
    d = last(f)
    print(os.path.basename(f), " | ".join(f"{k}: {st.median(r['prompt_tok_s'] for r in d[k]):.0f} tok/s (runs {[round(r['prompt_tok_s']) for r in d[k]]}) wall {st.median(r['wall_s'] for r in d[k]):.2f}s" for k in ("P512", "P2000")))
