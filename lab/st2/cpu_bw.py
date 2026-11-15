"""ST2 (CPU): is decode bound by memory bandwidth? One-row and 8-row decode steps of the same model as Q8_0 and
Q4_K_M at 1-4 threads, the weight bytes each step reads, and the bandwidth this machine's memory gives (a STREAM-like
copy and read with 1-4 processes). If tokens per second follow the bytes (Q4_K_M about 1.7x Q8_0) and flatten once the
threads reach the memory's bandwidth, the CPU path is memory bound: fewer bytes, not more SIMD.
Usage: cpu_bw.py OUT.json GGUF [GGUF ...]"""
import json, multiprocessing as mp, os, statistics, sys, time
sys.path.insert(0, __import__("os").path.abspath(__file__ + "/../../.."))
import numpy as np


def stream(kind, n_bytes, reps, out):
    a = np.ones(n_bytes // 8)
    b = np.empty_like(a)
    t = time.perf_counter()
    for _ in range(reps):
        if kind == "read":
            a.sum()
        else:
            np.copyto(b, a)
    out.put((time.perf_counter() - t) / reps)


def bandwidth(procs, kind, n_bytes=1 << 29, reps=6):
    q = mp.Queue()
    ps = [mp.Process(target=stream, args=(kind, n_bytes, reps, q)) for _ in range(procs)]
    [p.start() for p in ps]
    times = [q.get() for _ in ps]
    [p.join() for p in ps]
    moved = n_bytes * (1 if kind == "read" else 2) * procs
    return moved / max(times) / 1e9


def main():
    out_path, ggufs = sys.argv[1], sys.argv[2:]
    from ironmule.ggml_engine import GgmlEngine, CHECK_TEXT
    res = {"cpu": os.popen("lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Core|L3'").read(),
           "bandwidth_GBps": {f"{kind}_{p}": round(bandwidth(p, kind), 1) for kind in ("read", "copy") for p in (1, 2, 4)},
           "models": {}}
    print(json.dumps(res["bandwidth_GBps"]), flush=True)
    for gguf in ggufs:
        size = os.path.getsize(gguf)
        e = GgmlEngine(gguf, seqs=8, seq_ctx=512, speculate=False, flash_attn=True)
        base = (e.tokenize(CHECK_TEXT) * 8)[:64]
        rows_out = []
        for threads in (1, 2, 3, 4):
            e.set_threads(threads, 50)
            for rows in (1, 8):
                e._clear()
                e._decode([(t, j, s, j == len(base) - 1) for s in range(rows) for j, t in enumerate(base)])
                times = []
                for k in range(12):
                    t1 = time.perf_counter()
                    e._decode([(base[k % 8], 64 + k, s, True) for s in range(rows)])
                    times.append(time.perf_counter() - t1)
                step = statistics.median(times[2:])
                row = {"threads": threads, "rows": rows, "step_s": round(step, 4), "tok_s": round(rows / step, 2),
                       "weights_GBps": round(size / step / 1e9, 1)}
                rows_out.append(row)
                print(os.path.basename(gguf), json.dumps(row), flush=True)
        e.close()
        res["models"][os.path.basename(gguf)] = {"bytes": size, "rows": rows_out}
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
