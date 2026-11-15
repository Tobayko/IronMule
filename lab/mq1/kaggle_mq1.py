"""ST2 (GPU) on a Kaggle T4: where one request's decode step goes, and whether MQ1 (Q4_K repacked for streaming, the
Marlin principle without Ampere's cp.async) makes it at least 15 % faster. No build: SG1 v2's llama.cpp build (the
kernel output `ironmule-sg1-single`, attached as input) gives llama.cpp's kernels; nvcc compiles MQ1 here.

1. Profile: the engine decoding one request (Qwen3-4B Q4_K_M, 128 steps) under Nsight Systems where it is installed
   (the kernels' share of the step); llama.cpp's time for every quantized matrix shape of the model, Q4_K and Q6_K, at
   1 / 2 / 4 tokens (lab/kg1/ref_ggml.c) times how often the model has it; the engine's step for 1-4 tokens; and
   llama-server's one-request rate.
2. MQ1: every variant on every Q4_K shape, checked against float64 and timed like llama.cpp's.
3. The rule fixed before the data: MQ1 goes on to an end-to-end run only if, with the fastest variant per shape in
   place of llama.cpp's Q4_K kernels, one request's step would be at least 15 % shorter.
Writes /kaggle/working/mq1.json. MG2_COMMIT names the commit.
"""
import collections, glob, json, os, shutil, statistics, subprocess, sys, time, urllib.request
from pathlib import Path

import numpy as np

COMMIT = os.environ["MG2_COMMIT"]
W = Path("/kaggle/working")
T = Path("/tmp/mq1")
T.mkdir(parents=True, exist_ok=True)
OUT = {"commit": COMMIT, "started": time.time(), "rule": "one request's step at least 15 % shorter"}


def save():
    (W / "mq1.json").write_text(json.dumps(OUT, indent=1))


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def fetch(path):
    return urllib.request.urlopen(f"https://raw.githubusercontent.com/Tobayko/IronMule/{COMMIT}/{path}", timeout=60).read()


def repack(raw: np.ndarray, N: int, K: int):
    """Q4_K rows -> (qs in weight order, 16 bytes per 32 weights; the 16-byte block headers)."""
    blocks = raw.reshape(N * K // 256, 144)
    hdr = np.ascontiguousarray(blocks[:, :16])
    q = blocks[:, 16:].reshape(-1, 4, 32)  # 4 chunks of 64 weights, 32 bytes each
    w = np.empty((blocks.shape[0], 4, 64), dtype=np.uint8)
    w[:, :, :32] = q & 0xF  # weights 64c + l
    w[:, :, 32:] = q >> 4  # weights 64c + 32 + l
    w = w.reshape(-1, 256)
    qs = (w[:, 0::2] | (w[:, 1::2] << 4)).astype(np.uint8)  # byte i: weights 2i and 2i + 1
    return np.ascontiguousarray(qs), hdr


def main():
    OUT["gpu"] = subprocess.run("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem "
                                "--format=csv,noheader", shell=True, capture_output=True, text=True).stdout
    tar = next(iter(glob.glob("/kaggle/input/**/sg1_build.tar.gz", recursive=True)), None)
    OUT["build_from"] = tar
    src = T / "src"
    src.mkdir(exist_ok=True)
    sh(f"cd {src} && tar xzf {tar} && pip download llama_cpp_python==0.3.16 --no-binary :all: --no-deps -d sd -q "
       f"&& tar xzf sd/llama_cpp_python-0.3.16.tar.gz -C sd")
    tree = src / "sd/llama_cpp_python-0.3.16/vendor/llama.cpp"
    lib = src / "llama_cpp_python-0.3.16/vendor/llama.cpp/build/bin"
    sys.path.insert(0, str(tree / "gguf-py"))
    import gguf  # noqa: E402

    cuda = next(iter(sorted(glob.glob("/usr/local/cuda-12*"), reverse=True)), "/usr/local/cuda")
    nvcc = f"{cuda}/bin/nvcc"
    for name in ("lab/mq1/mq1.cu", "lab/kg1/ref_ggml.c", "lab/mg2/kaggle_gpu.py"):
        (T / Path(name).name).write_bytes(fetch(name))
    t0 = time.time()
    sh(f"cd {T} && {nvcc} -O3 -std=c++17 -arch=sm_75 -o mq1 mq1.cu > nvcc.log 2>&1")
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
    save()
    env = {**os.environ, "LD_LIBRARY_PATH": f"{lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    rng = np.random.default_rng(0)
    xs = {}
    for _, K, _ in picks:
        if K not in xs:
            xs[K] = T / f"x_{K}.bin"
            rng.standard_normal((4, K)).astype(np.float32).tofile(xs[K])
    # 1b. llama.cpp's kernels per matrix shape
    OUT["ggml"] = []
    for (kind, K, N), t in sorted(picks.items()):
        if kind not in ("Q4_K", "Q6_K"):
            continue
        wpath = T / f"w_{kind}_{K}_{N}.bin"
        raw = np.asarray(t.data)
        raw.tofile(wpath)
        copies = 2 if raw.nbytes > 100e6 else 8
        ref = subprocess.run([str(T / "ref_ggml"), str(wpath), str(N), str(K), str(xs[K]), str(T / f"ggml_{kind}_{K}_{N}"),
                              str(copies), "30", "1", "2", "4"], capture_output=True, text=True,
                             env={**env, "REF_TYPE": "12" if kind == "Q4_K" else "14"})
        for line in ref.stdout.splitlines():
            if line.startswith("{"):
                OUT["ggml"].append({"kind": kind, "count": counts[(kind, K, N)], **json.loads(line)})
        if ref.returncode:
            OUT.setdefault("errors", []).append(ref.stderr[-500:])
        save()
    # 2. MQ1 on every Q4_K shape
    OUT["mq1"] = []
    for (kind, K, N), t in sorted(picks.items()):
        if kind != "Q4_K":
            continue
        raw = np.asarray(t.data)
        qs, hdr = repack(raw, N, K)
        (T / "qs.bin").write_bytes(qs.tobytes())
        (T / "hdr.bin").write_bytes(hdr.tobytes())
        run = subprocess.run([str(T / "mq1"), str(T / f"w_{kind}_{K}_{N}.bin"), str(T / "qs.bin"), str(T / "hdr.bin"),
                              str(N), str(K), str(xs[K]), str(T / f"ggml_{kind}_{K}_{N}"), "8", "30", t.name],
                             capture_output=True, text=True, env=env)
        OUT["mq1"] += [{"count": counts[(kind, K, N)], **json.loads(l)} for l in run.stdout.splitlines()
                       if l.startswith("{")]
        if run.returncode:
            OUT.setdefault("errors", []).append(run.stderr[-500:])
        save()
    # 1c. the engine's step and llama-server's rate (MG2's harness on this build)
    mg2 = type(sys)("mg2")
    exec(compile((T / "kaggle_gpu.py").read_bytes(), str(T / "kaggle_gpu.py"), "exec"), mg2.__dict__)
    mg2.SRC, mg2.LIB = src, lib
    facts = mg2.build()
    del facts
    probe = T / "probe.py"
    probe.write_text(r'''
import json, sys, time
from ironmule.ggml_engine import GgmlEngine
e = GgmlEngine(sys.argv[1], seqs=1, seq_ctx=2048, speculate=False, flash_attn=True)
print(json.dumps({"step_ms": {t: round(v * 1000, 2) for t, v in e._step_prices(range(1, 5), repeats=9).items()}}), flush=True)
bos = max(0, int(e.L.llama_vocab_bos(e.vocab)))
e._clear(); e._decode([(bos, j, 0, j == 63) for j in range(64)])
t0 = time.perf_counter()
for k in range(128):
    e._decode([(bos, 64 + k, 0, True)])
print(json.dumps({"decode_128_ms": round((time.perf_counter() - t0) * 1000, 1)}), flush=True)
''')
    penv = mg2.env("0")
    nsys = shutil.which("nsys") or next(iter(glob.glob("/usr/local/cuda*/bin/nsys") + glob.glob("/opt/nvidia/nsight-systems*/bin/nsys")), None)
    OUT["nsys"] = nsys
    if nsys:
        prof = subprocess.run(f"{nsys} profile -t cuda -o {T}/prof --force-overwrite true {sys.executable} {probe} {gpath}",
                              shell=True, capture_output=True, text=True, env=penv)
        OUT["probe"] = [l for l in prof.stdout.splitlines() if l.startswith("{")]
        stats = subprocess.run(f"{nsys} stats --report cuda_gpu_kern_sum --format csv {T}/prof.nsys-rep", shell=True,
                               capture_output=True, text=True)
        OUT["kernels_csv"] = stats.stdout[-12000:]
    else:
        prof = subprocess.run([sys.executable, str(probe), str(gpath)], capture_output=True, text=True, env=penv)
        OUT["probe"] = [l for l in prof.stdout.splitlines() if l.startswith("{")]
    save()
    proc, port = mg2.server(str(gpath), 1, "0", "none")
    try:
        prompts = mg2.template(port, mg2.TEXTS[:1], None)
        mg2.block_servers([port], prompts, 128)
        OUT["server_1"] = [mg2.block_servers([port], prompts, 128)[0] for _ in range(3)]
    finally:
        mg2.stop(proc)
    # 3. the projection and the rule
    summary = {}
    for B in (1, 2, 4):
        g = {(r["K"], r["N"]): r for r in OUT["ggml"] if r["B"] == B and r["kind"] == "Q4_K"}
        q6 = sum(r["us"] * r["count"] for r in OUT["ggml"] if r["B"] == B and r["kind"] == "Q6_K")
        gerr = {(r["K"], r["N"]): r["ggml_rel_err"] for r in OUT["mq1"] if "ggml_rel_err" in r and r["B"] == B}
        g_us = m_us = 0.0
        best = {}
        for (K, N), r in g.items():
            vs = [v for v in OUT["mq1"] if "W" in v and v["B"] == B and (v["K"], v["N"]) == (K, N)
                  and v["rel_err"] <= max(1e-4, gerr.get((K, N), 1e-4))]
            if not vs:
                continue
            top = min(vs, key=lambda v: v["us"])
            best[f"{K}x{N}"] = {"ggml_us": r["us"], "mq1_us": top["us"], "variant": [top["W"], top["R"], top["U"]],
                                "rel_err": top["rel_err"]}
            g_us += r["us"] * r["count"]
            m_us += min(top["us"], r["us"]) * r["count"]
        summary[B] = {"q4k_ggml_us": round(g_us, 1), "q4k_mq1_us": round(m_us, 1), "q6k_ggml_us": round(q6, 1),
                      "best": best}
    step = None
    for line in OUT.get("probe", []):
        step = json.loads(line).get("step_ms", step)
    if step:
        for B in (1, 2, 4):
            s = summary[B]
            step_us = 1000 * float(step[str(B)] if str(B) in step else step[B])
            s["step_us"] = step_us
            s["projected_step_us"] = round(step_us - s["q4k_ggml_us"] + s["q4k_mq1_us"], 1)
            s["projected_gain"] = round(step_us / s["projected_step_us"] - 1, 3)
    OUT["summary"] = summary
    OUT["decision"] = ("go on" if summary.get(1, {}).get("projected_gain", 0) >= 0.15 else "stop: under 15 %")
    OUT["finished"] = time.time()
    save()


main()
