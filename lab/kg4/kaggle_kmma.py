"""KG4 step 2 on a Kaggle T4: lab/kg4/kmma.cu (Q4_K x int8 on tensor cores) against llama.cpp's own kernels, per
matrix shape of Qwen3-4B Q4_K_M and token count (1, 8, 16, 24, 32), no engine.

MG2's CUDA build (the dataset `ironmule-mg2-cuda-build`) gives llama.cpp's kernels through lab/kg1/ref_ggml.c
(ggml_mul_mat on the CUDA backend, its time per matmul including its own activation quantization); nvcc compiles
kmma.cu here. Every kmma variant is checked against float64 beside llama.cpp's result on the same data and timed the
same way (8 weight copies, so that the 4 MB L2 holds none of them). The summary weights each shape by how often the
model has it: llama.cpp's Q4_K time per step against the faster of the two per shape.
Writes /kaggle/working/kg4_kmma.json. MG2_COMMIT names the commit.
"""
import collections, glob, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

import numpy as np

COMMIT = os.environ["MG2_COMMIT"]
W = Path("/kaggle/working")
T = Path("/tmp/kg4")
T.mkdir(parents=True, exist_ok=True)
OUT = {"commit": COMMIT, "started": time.time()}
BS = (1, 8, 16, 24, 32)


def save():
    (W / "kg4_kmma.json").write_text(json.dumps(OUT, indent=1))


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def fetch(path):
    return urllib.request.urlopen(f"https://raw.githubusercontent.com/Tobayko/IronMule/{COMMIT}/{path}", timeout=60).read()


def main():
    OUT["gpu"] = subprocess.run("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem "
                                "--format=csv,noheader", shell=True, capture_output=True, text=True).stdout
    for name in ("lab/kg4/kmma.cu", "lab/kg1/ref_ggml.c", "lab/mg2/kaggle_gpu.py"):
        (T / Path(name).name).write_bytes(fetch(name))
    mg2 = type(sys)("mg2")
    exec(compile((T / "kaggle_gpu.py").read_bytes(), "mg2", "exec"), mg2.__dict__)
    mg2.build()
    lib = mg2.LIB
    sd = T / "sd"
    sd.mkdir(exist_ok=True)
    sh(f"cd {sd} && pip download llama_cpp_python==0.3.16 --no-binary :all: --no-deps -d . -q "
       f"&& tar xzf llama_cpp_python-0.3.16.tar.gz")
    tree = sd / "llama_cpp_python-0.3.16/vendor/llama.cpp"
    sys.path.insert(0, str(tree / "gguf-py"))
    import gguf  # noqa: E402

    cuda = next(iter(sorted(glob.glob("/usr/local/cuda-12*"), reverse=True)), "/usr/local/cuda")
    t0 = time.time()
    sh(f"cd {T} && {cuda}/bin/nvcc -O3 -std=c++17 -arch=sm_75 -o kmma kmma.cu > nvcc.log 2>&1")
    OUT["nvcc_s"] = round(time.time() - t0, 1)
    sh(f"cd {T} && gcc -O2 -o ref_ggml ref_ggml.c -I{tree}/ggml/include -L{lib} -lggml -lggml-base -lggml-cuda "
       f"-Wl,-rpath,{lib}")
    gpath = T / "Qwen3-4B-Q4_K_M.gguf"
    sh(f"curl -sSL -o {gpath} https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf")
    reader = gguf.GGUFReader(str(gpath))
    counts, picks = collections.Counter(), {}
    for t in reader.tensors:
        if len(t.shape) != 2:
            continue
        key = (t.tensor_type.name, int(t.shape[0]), int(t.shape[1]))
        counts[key] += 1
        picks.setdefault(key, t)
    OUT["matrices"] = {f"{k[0]} {k[1]}x{k[2]}": v for k, v in sorted(counts.items())}
    env = {**os.environ, "LD_LIBRARY_PATH": f"{lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    rng = np.random.default_rng(0)
    OUT["ggml"], OUT["kmma"] = [], []
    for (kind, K, N), t in sorted(picks.items()):
        if kind not in ("Q4_K", "Q6_K"):
            continue
        xpath = T / f"x_{K}.bin"
        if not xpath.exists():
            rng.standard_normal((32, K)).astype(np.float32).tofile(xpath)
        wpath = T / f"w_{kind}_{K}_{N}.bin"
        raw = np.asarray(t.data)
        raw.tofile(wpath)
        copies = 2 if raw.nbytes > 100e6 else 8
        ref = subprocess.run([str(T / "ref_ggml"), str(wpath), str(N), str(K), str(xpath), str(T / f"ggml_{kind}_{K}_{N}"),
                              str(copies), "30", *map(str, BS)], capture_output=True, text=True,
                             env={**env, "REF_TYPE": "12" if kind == "Q4_K" else "14"})
        OUT["ggml"] += [{"kind": kind, "count": counts[(kind, K, N)], **json.loads(l)} for l in ref.stdout.splitlines()
                        if l.startswith("{")]
        if ref.returncode:
            OUT.setdefault("errors", []).append(ref.stderr[-500:])
        if kind == "Q4_K":
            run = subprocess.run([str(T / "kmma"), str(wpath), str(N), str(K), str(xpath), str(T / f"ggml_{kind}_{K}_{N}"),
                                  str(copies), "30", *map(str, BS)], capture_output=True, text=True, env=env)
            OUT["kmma"] += [{"count": counts[(kind, K, N)], **json.loads(l)} for l in run.stdout.splitlines()
                            if l.startswith("{")]
            if run.returncode:
                OUT.setdefault("errors", []).append(run.stderr[-500:] + run.stdout[-500:])
        save()
    summary = {}
    for B in BS:
        g = {(r["K"], r["N"]): r for r in OUT["ggml"] if r["B"] == B and r["kind"] == "Q4_K"}
        q6 = sum(r["us"] * r["count"] for r in OUT["ggml"] if r["B"] == B and r["kind"] == "Q6_K")
        g_us = k_us = 0.0
        best = {}
        for (K, N), r in g.items():
            vs = [v for v in OUT["kmma"] if "us" in v and v["B"] == B and (v["K"], v["N"]) == (K, N)
                  and v["rel_err"] <= 1.25 * max(v["ggml_rel_err"], 1e-4)]
            top = min(vs, key=lambda v: v["us"]) if vs else None
            best[f"{K}x{N}"] = {"ggml_us": r["us"], "kmma_us": top and top["us"], "rel_err": top and top["rel_err"],
                                "ggml_rel_err": top and top["ggml_rel_err"], "GBps": top and top["GBps"],
                                "variant": top and [top["T"], top["R"], top["W"], top["KB"], top["SK"]]}
            g_us += r["us"] * r["count"]
            k_us += min(top["us"] if top else r["us"], r["us"]) * r["count"]
        summary[B] = {"q4k_ggml_us": round(g_us, 1), "q4k_best_us": round(k_us, 1), "q6k_ggml_us": round(q6, 1),
                      "best": best}
    OUT["summary"] = summary
    OUT["finished"] = time.time()
    save()
    print(json.dumps({B: {k: v for k, v in s.items() if k != "best"} for B, s in summary.items()}), flush=True)


if __name__ == "__main__":
    main()
