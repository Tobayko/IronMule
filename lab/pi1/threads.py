"""PI1 probe: what a decode step costs with T threads (and busy-wait level) while H other busy processes run on this
CPU, for R rows at a short context; Qwen3-4B Q8_0, shared-pool engine. Usage: threads.py GGUF OUT.json"""
import json, multiprocessing as mp, os, statistics, sys, time
sys.path.insert(0, "/home/user/IronMule")
def hog(stop):
    x = 0
    while not stop.is_set():
        for _ in range(100000): x += 1
if __name__ == "__main__":
    from ironmule.ggml_engine import GgmlEngine, CHECK_TEXT
    gguf, out = sys.argv[1], sys.argv[2]
    e = GgmlEngine(gguf, seqs=16, seq_ctx=1024, speculate=False, flash_attn=True)
    base = (e.tokenize(CHECK_TEXT) * 8)[:64]
    res = []
    for hogs in (0, 1, 2, 3):
        stop = mp.Event(); ps = [mp.Process(target=hog, args=(stop,), daemon=True) for _ in range(hogs)]
        [p.start() for p in ps]; time.sleep(0.5)
        for threads in (1, 2, 3, 4):
            for poll in (50, 0):
                if threads == 1 and poll == 0: continue
                t0 = time.perf_counter(); e.set_threads(threads, poll); switch = time.perf_counter() - t0
                for rows in (1, 16):
                    e._clear()
                    e._decode([(t, j, s, j == len(base) - 1) for s in range(rows) for j, t in enumerate(base)][:1024])
                    times = []
                    for k in range(6):
                        t1 = time.perf_counter(); e._decode([(base[k], 64 + k, s, True) for s in range(rows)]); times.append(time.perf_counter() - t1)
                    row = {"hogs": hogs, "threads": threads, "poll": poll, "rows": rows, "step_s": statistics.median(times[1:]), "switch_ms": 1000 * switch}
                    res.append(row); print(json.dumps(row), flush=True)
        stop.set(); [p.join() for p in ps]
    e.set_threads(4, 50); e.close()
    open(out, "w").write(json.dumps(res, indent=1))
