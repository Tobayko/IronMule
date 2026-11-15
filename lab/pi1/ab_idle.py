"""Same process, same engine: steps with the Pilot and without it (e.pilot = None), alternating, idle machine."""
import json, sys, time
sys.path.insert(0, __import__("os").path.abspath(__file__ + "/../../.."))
from ironmule.ggml_engine import GgmlEngine, Job
gguf = sys.argv[1]
e = GgmlEngine(gguf, seqs=16, seq_ctx=2048, speculate=True, flash_attn=True, shared=True, cap=32)
e.collective = None
pilot = e.pilot
ask = ["Write a short story about a lighthouse keeper.", "Explain how a refrigerator keeps food cold.",
       "List five facts about the planet Mars.", "Give step by step instructions to make tea."]
ids = [e.tokenize(f"<|im_start|>user\n{q} variant {i} /no_think<|im_end|>\n<|im_start|>assistant\n") for i in range(64) for q in ask[i % 4:i % 4 + 1]]
e.run([Job(ids[0], 8)])
rows = []
k = 0
for rep in range(4):
    for arm in ("pilot", "none"):
        e.pilot = pilot if arm == "pilot" else None
        for n in (1, 8):
            js = [Job(ids[(k + i) % len(ids)], 64) for i in range(n)]; k += n
            t0 = time.monotonic_ns(); e.run(js); dt = (time.monotonic_ns() - t0) / 1e9
            r = {"rep": rep, "arm": arm, "n": n, "tok_s": round(sum(len(j.tokens) for j in js) / dt, 2), "threads": e.threads}
            rows.append(r); print(json.dumps(r), flush=True)
st = pilot.status() if pilot else {}
print(json.dumps({"decide_us_mean": st.get("decide_us_mean")}))
for arm in ("pilot", "none"):
    for n in (1, 8):
        v = [r["tok_s"] for r in rows if r["arm"] == arm and r["n"] == n]
        print(arm, n, round(sum(v) / len(v), 2), v)
