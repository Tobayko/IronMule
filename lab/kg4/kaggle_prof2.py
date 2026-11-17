"""KG4 step 4 on a Kaggle T4: why the tensor-core kernels save 5.4 ms of Q4_K matmuls per 32-token step in isolation
but 1-3 ms in the engine. A CUPTI kernel profile (torch.profiler) of engine steps with the library on and off.

KG2's build of llama.cpp with the kg1 hook (the kernel output `ironmule-kg2-e2e`), the library tuned and built as
ironmule/kernel_gen.py does at MG2_COMMIT, the engine as `ironmule serve` runs it on a GPU (one KV stream per request,
flash attention, 32 sequences, speculation off). Per arm ("kg": every token count on, "off": mask 0) and request count
(8, 32; context 300): 24 timed steps of one token per request, then 8 steps under the profiler: every kernel and copy
summed by name, the GPU's busy time, and the gaps between kernels on the stream.
Writes /kaggle/working/kg4_prof2.json. MG2_COMMIT names the commit.
"""
import glob, json, os, random, subprocess, sys, time, urllib.request
from pathlib import Path

COMMIT = os.environ["MG2_COMMIT"]
W = Path("/kaggle/working")
OUT = {"commit": COMMIT, "started": time.time(), "steps": [], "kernels": {}}


def save():
    (W / "kg4_prof2.json").write_text(json.dumps(OUT, indent=1))


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def main():
    OUT["gpu"] = subprocess.run("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader",
                                shell=True, capture_output=True, text=True).stdout
    cuda = next(iter(sorted(glob.glob("/usr/local/cuda-12*"), reverse=True)), "/usr/local/cuda")
    os.environ["PATH"] = f"{cuda}/bin:{os.environ['PATH']}"
    tar = next(iter(glob.glob("/kaggle/input/**/kg1_build.tar.gz", recursive=True)), None)
    src = Path("/tmp/kgsrc")
    src.mkdir(exist_ok=True)
    sh(f"cd {src} && tar xzf {tar}")
    mg2 = type(sys)("mg2")
    code = urllib.request.urlopen(f"https://raw.githubusercontent.com/Tobayko/IronMule/{COMMIT}/lab/mg2/kaggle_gpu.py",
                                  timeout=60).read()
    exec(compile(code, "mg2", "exec"), mg2.__dict__)
    mg2.SRC, mg2.LIB = src, src / "llama_cpp_python-0.3.16/vendor/llama.cpp/build/bin"
    mg2.build()
    for k, v in mg2.env("0").items():
        os.environ[k] = v
    os.environ["LD_LIBRARY_PATH"] = f"{mg2.LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["IRONMULE_PILOT"] = "0"
    sys.path.insert(0, str(W / "pkg"))
    gguf = Path("/tmp/Qwen3-4B-Q4_K_M.gguf")
    sh(f"curl -sSL -o {gguf} https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf")
    from ironmule import kernel_gen  # noqa: E402

    table = kernel_gen.tune(str(gguf), Path("/tmp/kg"), log=lambda m: None)
    OUT["entries_32"] = [e for e in table["entries"] if e["B"] in (8, 32)]
    lib = kernel_gen.build(table, Path("/tmp/kg/kg4.so"))
    os.environ["IRONMULE_KG_LIB"] = str(lib)
    from ironmule import ggml_engine as ge  # noqa: E402

    ge.KG_LIB = ""  # no startup choice: the arms set the mask themselves
    e = ge.GgmlEngine(str(gguf), seqs=32, seq_ctx=2048, speculate=False, flash_attn=True, shared=False)
    import ctypes

    klib = ctypes.CDLL(str(lib))
    klib.ironmule_kg_set_mask.argtypes = [ctypes.c_uint32]
    L = e.L
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

    def step(n):
        rows = []
        for s in range(n):
            rows.append((rnd.randrange(1000, 20000), top[s], s, True))
            top[s] += 1
        t = time.perf_counter()
        e._decode(rows)
        return time.perf_counter() - t

    import torch
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.init()
    for rep in range(2):
        for arm, mask in ((("kg", 0xFFFFFFFF), ("off", 0)) if rep == 0 else (("off", 0), ("kg", 0xFFFFFFFF))):
            klib.ironmule_kg_set_mask(mask)
            for n in (8, 32):
                fill(n, 300)
                for _ in range(3):
                    step(n)
                ts = sorted(step(n) for _ in range(24))[6:18]
                row = {"arm": arm, "rep": rep, "n": n, "step_ms": round(1e3 * sum(ts) / len(ts), 3)}
                OUT["steps"].append(row)
                print(json.dumps(row), flush=True)
                if rep:
                    continue
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                    t0 = time.perf_counter()
                    for _ in range(8):
                        step(n)
                    wall = time.perf_counter() - t0
                kernels, spans = {}, []
                for ev in prof.events():
                    if ev.device_type is not None and str(ev.device_type).endswith("CUDA"):
                        k = kernels.setdefault(ev.name, [0, 0.0])
                        k[0] += 1
                        k[1] += ev.time_range.end - ev.time_range.start
                        spans.append((ev.time_range.start, ev.time_range.end))
                spans.sort()
                busy, end, gaps = 0.0, -1.0, []
                for a, z in spans:
                    if end >= 0 and a > end:
                        gaps.append(a - end)
                    if z > end:
                        busy += z - max(a, end)
                        end = z
                gaps.sort()
                ranked = sorted(kernels.items(), key=lambda kv: -kv[1][1])
                OUT["kernels"][f"{arm}_{n}"] = {
                    "wall_ms_per_step": round(1e3 * wall / 8, 3), "gpu_busy_ms_per_step": round(busy / 1e3 / 8, 3),
                    "gap_ms_per_step": round(sum(gaps) / 1e3 / 8, 3), "gaps": len(gaps) // 8,
                    "gaps_over_20us": sum(g > 20 for g in gaps) // 8,
                    "top": [[name[:140], c // 8, round(us / 8, 1)] for name, (c, us) in ranked[:45]],
                    "launches_per_step": sum(v[0] for v in kernels.values()) // 8}
                print(json.dumps({"arm": arm, "n": n, **{k: v for k, v in OUT["kernels"][f"{arm}_{n}"].items()
                                                         if k != "top"}}), flush=True)
                save()
    e.close()
    OUT["finished"] = time.time()
    save()


if __name__ == "__main__":
    main()
