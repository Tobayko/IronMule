"""PL1 on the real machine: the engine with its defaults against the engine with the Pilot, under foreign CPU load.
Qwen3-4B Q8_0, shared pool (cap 32), N chat requests at once (64 tokens), H busy foreign processes the whole time,
or ("dyn") one that switches on and off every 6 s. Prints tokens/s, mean first token and what the Pilot chose.
Usage: real.py GGUF MODE(default|pilot) OUT.json"""
import json, multiprocessing as mp, os, sys, time
sys.path.insert(0, "/home/user/IronMule")
def hog(stop, on):
    x = 0
    while not stop.is_set():
        if on.is_set():
            for _ in range(100000): x += 1
        else:
            time.sleep(0.01)
if __name__ == "__main__":
    gguf, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.environ["IRONMULE_PILOT"] = "1" if mode == "pilot" else "0"
    from ironmule.ggml_engine import GgmlEngine, Job
    e = GgmlEngine(gguf, seqs=16, seq_ctx=2048, speculate=True, flash_attn=True, shared=True, cap=32)
    e.collective = None
    ask = ["Write a short story about a lighthouse keeper.", "Explain how a refrigerator keeps food cold.",
           "List five facts about the planet Mars.", "Give step by step instructions to make tea."]
    ids = [e.tokenize(f"<|im_start|>user\n{q} variant {i} /no_think<|im_end|>\n<|im_start|>assistant\n")
           for i in range(64) for q in ask[i % 4:i % 4 + 1]]
    e.run([Job(ids[0], 8)])
    res = {"mode": mode, "rows": []}
    for hogs in (0, 1, 2, "dyn"):
        stop, on = mp.Event(), mp.Event()
        n_h = 1 if hogs == "dyn" else hogs
        ps = [mp.Process(target=hog, args=(stop, on), daemon=True) for _ in range(n_h)]
        [p.start() for p in ps]
        if hogs != "dyn": on.set()
        time.sleep(1.0)
        for n in (1, 8, 16):
            t_start = time.monotonic()
            jobs, tok, wall, ttft = [], 0, 0.0, []
            for block in range(2 if n > 1 else 3):
                if hogs == "dyn":
                    (on.set if int((time.monotonic() - t_start) / 6) % 2 == 0 else on.clear)()
                js = [Job(ids[(block * n + i) % len(ids)], 64) for i in range(n)]
                t0 = time.monotonic_ns(); e.run(js); dt = (time.monotonic_ns() - t0) / 1e9
                tok += sum(len(j.tokens) for j in js); wall += dt; ttft += [(j.first_ns - t0) / 1e9 for j in js]
            st = e.summary().get("pilot") or {}
            row = {"hogs": hogs, "n": n, "tok_s": round(tok / wall, 2), "ttft": round(sum(ttft) / len(ttft), 3),
                   "threads_now": e.threads, "pilot": {k: st.get(k) for k in ("decision", "foreign_cores", "decide_us_mean")}}
            res["rows"].append(row); print(json.dumps(row), flush=True)
        stop.set(); on.set(); [p.join() for p in ps]
    open(out, "w").write(json.dumps(res, indent=1))
    e.close()
