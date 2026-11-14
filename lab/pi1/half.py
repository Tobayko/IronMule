"""PI1 probe: a decode step's seconds with 3 or 4 threads while one foreign process is busy a fraction D of the time
(on/off in 20 ms periods), and the foreign load the Pilot's reader sees. Usage: half.py GGUF OUT.json"""
import json, multiprocessing as mp, statistics, sys, time
sys.path.insert(0, "/home/user/IronMule")
def hog(stop, duty):
    while not stop.is_set():
        t = time.perf_counter()
        while time.perf_counter() - t < 0.02 * duty: pass
        time.sleep(0.02 * (1 - duty))
if __name__ == "__main__":
    from ironmule.ggml_engine import GgmlEngine, CHECK_TEXT
    from ironmule.pilot import CpuLoad
    gguf, out = sys.argv[1], sys.argv[2]
    e = GgmlEngine(gguf, seqs=16, seq_ctx=1024, speculate=False, flash_attn=True)
    base = (e.tokenize(CHECK_TEXT) * 8)[:64]
    res = []
    for duty in (0.0, 0.25, 0.5, 0.75, 1.0):
        stop = mp.Event(); p = mp.Process(target=hog, args=(stop, duty), daemon=True)
        if duty: p.start()
        time.sleep(0.5)
        for threads in (3, 4):
            e.set_threads(threads, 50)
            for rows in (1, 8):
                e._clear()
                e._decode([(t, j, s, j == len(base) - 1) for s in range(rows) for j, t in enumerate(base)][:1024])
                load = CpuLoad(); times = []
                for k in range(12):
                    t1 = time.perf_counter(); e._decode([(base[k % 8], 64 + k, s, True) for s in range(rows)]); times.append(time.perf_counter() - t1)
                    load.update(0.0)
                row = {"duty": duty, "threads": threads, "rows": rows, "step_s": round(statistics.median(times[2:]), 4), "foreign_seen": round(load.foreign, 2)}
                res.append(row); print(json.dumps(row), flush=True)
        stop.set()
        if duty: p.join()
    e.set_threads(4, 50); e.close()
    open(out, "w").write(json.dumps(res, indent=1))
