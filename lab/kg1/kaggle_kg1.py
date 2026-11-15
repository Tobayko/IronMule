"""KG1 on a Kaggle T4: every generated Q4_K kernel variant (gen.py) against llama.cpp's own CUDA kernel, on the real
Q4_K weights of Qwen3-4B Q4_K_M, for 1-16 activation rows. Needs SG1 v2's CUDA build of llama.cpp 4227c9b (the kernel
output `ironmule-sg1-single`, attached as input) for llama.cpp's kernel; the variants are compiled here with nvcc.

Per tensor shape and batch: llama.cpp's time and result, each variant's time, its error against a float64 reference
and against llama.cpp. Then the time per token of all Q4_K matmuls of the model with llama.cpp's kernel and with the
fastest variant per shape and batch. Writes /kaggle/working/kg1.json. MG2_COMMIT names the commit to fetch.
"""
import collections, glob, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

import numpy as np

COMMIT = os.environ["MG2_COMMIT"]
W = Path("/kaggle/working")
SRC = W / "src"
OUT = {"commit": COMMIT, "started": time.time()}
BS = (1, 2, 4, 8, 16)


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def save():
    (W / "kg1.json").write_text(json.dumps(OUT, indent=1))


def fetch(path):
    return urllib.request.urlopen(f"https://raw.githubusercontent.com/Tobayko/IronMule/{COMMIT}/{path}", timeout=60).read()


def main():
    OUT["gpu"] = subprocess.run("nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem "
                                "--format=csv,noheader", shell=True, capture_output=True, text=True).stdout
    SRC.mkdir(parents=True, exist_ok=True)
    tar = next(iter(glob.glob("/kaggle/input/**/sg1_build.tar.gz", recursive=True)), None)
    OUT["build_from"] = tar
    sh(f"cd {SRC} && tar xzf {tar}")
    sh(f"cd {SRC} && pip download llama_cpp_python==0.3.16 --no-binary :all: --no-deps -d sdist -q && "
       f"mkdir -p sd && tar xzf sdist/llama_cpp_python-0.3.16.tar.gz -C sd")
    tree = SRC / "sd/llama_cpp_python-0.3.16/vendor/llama.cpp"
    lib = SRC / "llama_cpp_python-0.3.16/vendor/llama.cpp/build/bin"
    sys.path.insert(0, str(tree / "gguf-py"))
    import gguf  # noqa: E402 - llama.cpp's own reader, from the same tree

    cuda = next(iter(sorted(glob.glob("/usr/local/cuda-12*"), reverse=True)), "/usr/local/cuda")
    nvcc = f"{cuda}/bin/nvcc"
    (W / "gen.py").write_bytes(fetch("lab/kg1/gen.py"))
    (W / "ref_ggml.c").write_bytes(fetch("lab/kg1/ref_ggml.c"))
    sh(f"cd {W} && {sys.executable} gen.py > bench.cu")
    t0 = time.time()
    sh(f"cd {W} && {nvcc} -O3 -std=c++17 -arch=sm_75 -o bench bench.cu > nvcc.log 2>&1")
    OUT["nvcc_s"] = round(time.time() - t0, 1)
    sh(f"cd {W} && gcc -O2 -o ref_ggml ref_ggml.c -I{tree}/ggml/include -L{lib} -lggml -lggml-base -lggml-cuda "
       f"-Wl,-rpath,{lib}")
    save()

    gguf_path = W / "Qwen3-4B-Q4_K_M.gguf"
    sh(f"curl -sSL -o {gguf_path} https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf")
    reader = gguf.GGUFReader(str(gguf_path))
    counts, picks = collections.Counter(), {}
    for t in reader.tensors:
        kind = t.tensor_type.name
        K, N = int(t.shape[0]), int(t.shape[1]) if len(t.shape) > 1 else 1
        counts[(kind, K, N)] += 1
        if kind == "Q4_K" and (K, N) not in picks:
            picks[(K, N)] = t
    OUT["tensor_counts"] = {f"{k[0]} {k[1]}x{k[2]}": v for k, v in sorted(counts.items())}
    rng = np.random.default_rng(0)
    xs = {}
    for K in sorted({k for k, _ in picks}):
        x = rng.standard_normal((16, K)).astype(np.float32)
        xs[K] = W / f"x_{K}.bin"
        x.tofile(xs[K])
    OUT["shapes"] = []
    env = {**os.environ, "LD_LIBRARY_PATH": f"{lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    for (K, N), t in sorted(picks.items()):
        name = t.name
        wpath = W / f"w_{K}_{N}.bin"
        wpath.write_bytes(np.asarray(t.data).tobytes())
        prefix = W / f"ggml_{K}_{N}"
        ref = subprocess.run([str(W / "ref_ggml"), str(wpath), str(N), str(K), str(xs[K]), str(prefix), "8", "50",
                              *map(str, BS)], capture_output=True, text=True, env=env)
        rows = [json.loads(l) for l in ref.stdout.splitlines() if l.startswith("{")]
        bench = subprocess.run([str(W / "bench"), str(wpath), str(N), str(K), str(xs[K]), str(prefix), "8", "50", name],
                               capture_output=True, text=True, env=env)
        rows += [json.loads(l) for l in bench.stdout.splitlines() if l.startswith("{")]
        OUT["shapes"].append({"tensor": name, "K": K, "N": N, "rows": rows,
                              "errors": (ref.stderr[-600:] if ref.returncode else "") +
                                        (bench.stderr[-600:] if bench.returncode else "")})
        save()
    # per token of the model: all its Q4_K matmuls, llama.cpp's kernel against the fastest correct variant
    summary = {}
    for B in BS:
        g_us = v_us = 0.0
        best = {}
        for s in OUT["shapes"]:
            n = counts[("Q4_K", s["K"], s["N"])]
            g = [r["us"] for r in s["rows"] if r.get("ggml") and r["B"] == B]
            v = [r for r in s["rows"] if "L" in r and r["B"] == B and r["rel_err"] < 1e-4]
            if not g or not v:
                continue
            top = min(v, key=lambda r: r["us"])
            best[f"{s['K']}x{s['N']}"] = {k: top[k] for k in ("L", "R", "W", "us", "GBps", "rel_err", "rel_to_ggml")}
            best[f"{s['K']}x{s['N']}"]["ggml_us"] = g[0]
            g_us += n * g[0]
            v_us += n * top["us"]
        summary[B] = {"ggml_us_per_token": round(g_us, 1), "kg1_us_per_token": round(v_us, 1),
                      "speedup": round(g_us / v_us, 3) if v_us else None, "best": best}
    OUT["summary"] = summary
    OUT["finished"] = time.time()
    save()


main()
