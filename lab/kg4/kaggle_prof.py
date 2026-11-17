"""KG4 probe on a Kaggle T4: where one engine step of 1 / 8 / 16 / 32 requests spends its time, before any kernel work.

The engine as `ironmule serve` runs it on a GPU (Qwen3-4B Q4_K_M, every layer on GPU 0, one KV stream per request,
flash attention, 32 sequences of 2,048 cells), MG2's CUDA build (the dataset `ironmule-mg2-cuda-build`, no kg1 hook).
Speculation off, so that every step is one row per request.

  steps    per (requests, context): 32 sequences' worth of random prompt tokens filled, then 24 timed decode steps of
           one token per request, split into llama_decode's return, the first logits row ready (the GPU done) and the
           argmax of every row on the host; once with logits for every row, once for one row only
  kernels  torch.profiler (CUPTI) over 8 steps at 1 / 8 / 32 requests, context 300: every CUDA kernel and copy the
           process ran, summed by name, and the GPU's busy time against the steps' wall time
Writes /kaggle/working/kg4_prof.json. MG2_COMMIT names the commit to fetch. KG4_LOCAL=dir: a CPU smoke (no kernels).
"""
import json, os, random, sys, time
from pathlib import Path

COMMIT = os.environ["MG2_COMMIT"]
LOCAL = os.environ.get("KG4_LOCAL")
W = Path(LOCAL or "/kaggle/working")
OUT = {"commit": COMMIT, "started": time.time(), "steps": [], "kernels": {}}


def save():
    (W / "kg4_prof.json").write_text(json.dumps(OUT, indent=1))


def main():
    import subprocess

    OUT["gpu"] = subprocess.run("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader",
                                shell=True, capture_output=True, text=True).stdout
    repo = Path(__file__).resolve().parents[2]
    if LOCAL:
        src = (repo / "lab/mg2/kaggle_gpu.py").read_bytes()
    else:
        src = __import__("urllib.request").request.urlopen(
            f"https://raw.githubusercontent.com/Tobayko/IronMule/{COMMIT}/lab/mg2/kaggle_gpu.py", timeout=60).read()
    mg2 = type(sys)("mg2")
    exec(compile(src, "mg2", "exec"), mg2.__dict__)
    if LOCAL:
        mg2.SRC = Path(os.environ["IRONMULE_GGML_BINDINGS"]).parent
        mg2.LIB = Path(os.environ["LLAMA_CPP_LIB_PATH"])
        (W / "pkg").mkdir(exist_ok=True)
        if not (W / "pkg/ironmule").exists():
            (W / "pkg/ironmule").symlink_to(repo / "ironmule")
        gguf = os.environ["KG4_GGUF"]
    else:
        mg2.build()
        gguf = str(W / "Qwen3-4B-Q4_K_M.gguf")
        if not Path(gguf).exists():
            mg2.sh(f"cd {W} && curl -sSL -o q.part https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/"
                   f"Qwen3-4B-Q4_K_M.gguf && mv q.part {gguf}")
    for k, v in mg2.env("0").items():
        os.environ[k] = v
    os.environ["IRONMULE_PILOT"] = "0"
    os.environ["IRONMULE_KG"] = "0"
    sys.path.insert(0, str(W / "pkg"))
    from ironmule.ggml_engine import GgmlEngine

    seqs = 4 if LOCAL else 32
    e = GgmlEngine(gguf, seqs=seqs, seq_ctx=2048, speculate=False, flash_attn=True, shared=False)
    OUT["engine"] = {"gpu_layers": e.gpu_layers, "settings": e.settings, "n_vocab": e.n_vocab, "threads": e.threads}
    L, np = e.L, e.np
    rnd = random.Random(7)
    top = {}

    def fill(n, ctx):
        L.llama_memory_clear(e.memory, True)
        top.clear()
        for s in range(n):
            ids = [rnd.randrange(1000, 20000) for _ in range(ctx)]
            for a in range(0, ctx, 512):
                part = ids[a:a + 512]
                e._decode([(t, a + i, s, a + i == ctx - 1) for i, t in enumerate(part)])
            top[s] = ctx

    def step(n, every):
        b = e.batch
        b.n_tokens = n
        for s in range(n):
            b.token[s], b.pos[s], b.n_seq_id[s], b.logits[s] = rnd.randrange(1000, 20000), top[s], 1, int(every or s == n - 1)
            b.seq_id[s][0] = s
            top[s] += 1
        t0 = time.perf_counter()
        if L.llama_decode(e.ctx, b) != 0:
            raise RuntimeError("llama_decode failed")
        t1 = time.perf_counter()
        first = L.llama_get_logits_ith(e.ctx, n - 1 if not every else 0)
        t2 = time.perf_counter()
        rows = range(n) if every else (n - 1,)
        for i in rows:
            int(np.ctypeslib.as_array(L.llama_get_logits_ith(e.ctx, i), shape=(e.n_vocab,)).argmax())
        t3 = time.perf_counter()
        del first
        return t1 - t0, t2 - t1, t3 - t2

    grid = [(n, c) for n in ((1, 2, 4) if LOCAL else (1, 8, 16, 32)) for c in ((64,) if LOCAL else (300, 1000))]
    for every in (True, False):
        for n, ctx in grid:
            fill(n, ctx)
            for _ in range(3):
                step(n, every)
            parts = [step(n, every) for _ in range(4 if LOCAL else 24)]
            parts.sort(key=sum)
            mid = parts[len(parts) // 4: len(parts) - len(parts) // 4]
            row = {"n": n, "ctx": ctx, "logits_every_row": every,
                   "step_ms": round(1e3 * sum(map(sum, mid)) / len(mid), 3),
                   "decode_call_ms": round(1e3 * sum(p[0] for p in mid) / len(mid), 3),
                   "wait_gpu_ms": round(1e3 * sum(p[1] for p in mid) / len(mid), 3),
                   "argmax_ms": round(1e3 * sum(p[2] for p in mid) / len(mid), 3)}
            row["tok_s"] = round(n / row["step_ms"] * 1e3, 1)
            OUT["steps"].append(row)
            print(json.dumps(row), flush=True)
            save()
    if not LOCAL:
        import torch
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.init()
        for n in (1, 8, 32):
            fill(n, 300)
            for _ in range(3):
                step(n, True)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                t0 = time.perf_counter()
                for _ in range(8):
                    step(n, True)
                wall = time.perf_counter() - t0
            kernels = {}
            spans = []
            for ev in prof.events():
                if ev.device_type is not None and str(ev.device_type).endswith("CUDA"):
                    k = kernels.setdefault(ev.name, [0, 0.0])
                    k[0] += 1
                    k[1] += ev.time_range.end - ev.time_range.start
                    spans.append((ev.time_range.start, ev.time_range.end))
            spans.sort()
            busy, end = 0.0, -1.0
            for a, z in spans:  # the union of the device intervals (us)
                if z > end:
                    busy += z - max(a, end)
                    end = z
            ranked = sorted(kernels.items(), key=lambda kv: -kv[1][1])
            OUT["kernels"][str(n)] = {"wall_ms_per_step": round(1e3 * wall / 8, 3),
                                      "gpu_busy_ms_per_step": round(busy / 1e3 / 8, 3),
                                      "top": [[name[:120], c // 8, round(us / 8, 1)] for name, (c, us) in ranked[:40]],
                                      "total_us_per_step": round(sum(v[1] for v in kernels.values()) / 8, 1),
                                      "launches_per_step": sum(v[0] for v in kernels.values()) // 8}
            print(json.dumps({"n": n, **{k: v for k, v in OUT["kernels"][str(n)].items() if k != "top"}}), flush=True)
            save()
    e.close()
    OUT["finished"] = time.time()
    save()


if __name__ == "__main__":
    main()
