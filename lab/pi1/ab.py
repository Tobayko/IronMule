"""PI1: the Pilot against no Pilot on one engine, alternating, idle machine. Usage: ab.py GGUF (IRONMULE_PILOT_WEIGHTS)"""
import sys, time, json
sys.path.insert(0, "/home/user/IronMule")
from ironmule.ggml_engine import GgmlEngine, Job
e = GgmlEngine(sys.argv[1], seqs=16, seq_ctx=2048, speculate=True, flash_attn=True, shared=True, cap=32)
e.collective = None
pilot = e.pilot
ask = ["Write a short story about a lighthouse keeper.", "Explain how a refrigerator keeps food cold."]
ids = [e.tokenize(f"<|im_start|>user\n{ask[i%2]} variant {i} /no_think<|im_end|>\n<|im_start|>assistant\n") for i in range(16)]
e.run([Job(ids[0], 8)])
for rnd in range(3):
    for arm in ("pilot", "none"):
        e.pilot = pilot if arm == "pilot" else None
        for n in (1, 8):
            js = [Job(ids[i], 32) for i in range(n)]
            t0 = time.monotonic(); e.run(js); dt = time.monotonic() - t0
            print(json.dumps({"rnd": rnd, "arm": arm, "n": n, "tok_s": round(sum(len(j.tokens) for j in js) / dt, 2), "threads": e.threads}), flush=True)
