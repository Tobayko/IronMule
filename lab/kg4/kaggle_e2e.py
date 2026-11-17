"""KG4 step 3 on a Kaggle T4: the tensor-core kernels (ironmule/kernel_gen.py, KG_VERSION 3) inside the engine, end to
end, against the same engine with llama.cpp's own kernels.

1. KG2's build of llama.cpp 4227c9b with gr1 and the kg1 hook (the kernel output `ironmule-kg2-e2e`, attached as input:
   kg1_build.tar.gz), no rebuild.
2. The library tuned for this GPU and Qwen3-4B Q4_K_M and built (every variant on the model's own Q4_K matrices,
   checked against float64; the fastest correct one per shape and token count).
3. One engine as `ironmule serve` runs it on a GPU (one KV stream per request, flash attention, 32 sequences of 2,048
   cells), IRONMULE_KG_LIB set; the engine keeps the library for the token counts it measures faster at startup. The
   library is switched per arm through its mask: "kg" (the engine's own choice) and "off" (mask 0), alternating:
     steps   one token per request, 1 / 8 / 16 / 32 requests at context 300 and 1,000 (24 timed steps, middle half)
     blocks  1 / 8 / 32 chat requests at once, 128 tokens, greedy, the engine's padding and speculation as shipped
     facts   the 50 facts of lab/row1/facts.py 8 at a time through both, checked as MG2 checks them, and the answers
             compared token for token
Writes /kaggle/working/kg4_e2e.json. MG2_COMMIT names the commit.
"""
import glob, json, os, random, subprocess, sys, time, urllib.request
from pathlib import Path

COMMIT = os.environ["MG2_COMMIT"]
W = Path("/kaggle/working")
OUT = {"commit": COMMIT, "started": time.time(), "steps": [], "blocks": []}


def save():
    (W / "kg4_e2e.json").write_text(json.dumps(OUT, indent=1))


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def main():
    OUT["gpu"] = subprocess.run("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader",
                                shell=True, capture_output=True, text=True).stdout
    cuda = next(iter(sorted(glob.glob("/usr/local/cuda-12*"), reverse=True)), "/usr/local/cuda")
    os.environ["PATH"] = f"{cuda}/bin:{os.environ['PATH']}"
    tar = next(iter(glob.glob("/kaggle/input/**/kg1_build.tar.gz", recursive=True)), None)
    OUT["build_from"] = tar
    src = Path("/tmp/kgsrc")  # where KG2 built it: its libraries find each other by that absolute path
    src.mkdir(exist_ok=True)
    sh(f"cd {src} && tar xzf {tar}")
    mg2 = type(sys)("mg2")
    code = urllib.request.urlopen(f"https://raw.githubusercontent.com/Tobayko/IronMule/{COMMIT}/lab/mg2/kaggle_gpu.py",
                                  timeout=60).read()
    exec(compile(code, "mg2", "exec"), mg2.__dict__)
    mg2.SRC, mg2.LIB = src, src / "llama_cpp_python-0.3.16/vendor/llama.cpp/build/bin"
    facts = mg2.build()  # the build is there: only the package at COMMIT and the facts are fetched
    for k, v in mg2.env("0").items():
        os.environ[k] = v
    os.environ["LD_LIBRARY_PATH"] = f"{mg2.LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["IRONMULE_PILOT"] = "0"
    sys.path.insert(0, str(W / "pkg"))
    gguf = Path("/tmp/Qwen3-4B-Q4_K_M.gguf")
    sh(f"curl -sSL -o {gguf} https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf")
    from ironmule import kernel_gen  # noqa: E402

    t0 = time.time()
    table = kernel_gen.tune(str(gguf), Path("/tmp/kg"), log=lambda m: print(m, flush=True))
    (W / "kg4_tune_measured.json").write_text(json.dumps(table["measured"]))
    OUT["tune"] = {k: v for k, v in table.items() if k != "measured"}
    lib = kernel_gen.build(table, Path("/tmp/kg/kg4.so"))
    OUT["tune_s"] = round(time.time() - t0, 1)
    save()
    proc, port = mg2.server(str(gguf), 8, "0", "none")
    try:
        prompts = mg2.template(port, mg2.TEXTS, None)
        fact_prompts = mg2.template(port, [q + " Answer in one short sentence." for q, _ in facts], "/no_think")
    finally:
        mg2.stop(proc)
    os.environ["IRONMULE_KG_LIB"] = str(lib)
    from ironmule import ggml_engine as ge  # noqa: E402

    ge.KG_LIB = str(lib)
    e = ge.GgmlEngine(str(gguf), seqs=32, seq_ctx=2048, flash_attn=True, shared=False)
    OUT["kg"] = e.kg
    OUT["pad_to"] = {str(k): v for k, v in e.pad_to.items()}
    save()
    import ctypes

    klib = ctypes.CDLL(str(lib))
    klib.ironmule_kg_set_mask.argtypes = klib.ironmule_kg_set_mask6.argtypes = [ctypes.c_uint32]
    klib.ironmule_kg_get_mask.restype = klib.ironmule_kg_get_mask6.restype = ctypes.c_uint32
    chosen = (klib.ironmule_kg_get_mask(), klib.ironmule_kg_get_mask6())
    OUT["mask"] = chosen
    arms = (("kg", chosen), ("off", (0, 0)))
    pads = {"kg": dict(e.pad_to)}

    def use(arm, mask):  # the library's masks and the padding the engine measures for them
        klib.ironmule_kg_set_mask(mask[0])
        klib.ironmule_kg_set_mask6(mask[1])
        if arm not in pads:
            pads[arm] = e.measure_padding()
        e.pad_to = pads[arm]

    use("off", (0, 0))
    OUT["pad_to_off"] = {str(k): v for k, v in pads["off"].items()}
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

    for rep in range(2):
        for arm, mask in (arms if rep == 0 else arms[::-1]):
            use(arm, mask)
            for n in (1, 8, 16, 32):
                for ctx in (300, 1000):
                    fill(n, ctx)
                    for _ in range(3):
                        step(n)
                    ts = sorted(step(n) for _ in range(24))[6:18]
                    row = {"arm": arm, "rep": rep, "n": n, "ctx": ctx, "step_ms": round(1e3 * sum(ts) / len(ts), 3)}
                    row["tok_s"] = round(n / row["step_ms"] * 1e3, 1)
                    OUT["steps"].append(row)
                    print(json.dumps(row), flush=True)
            save()
    L.llama_memory_clear(e.memory, True)
    e._kept.clear()
    e._free = list(range(e.seqs))
    tok = lambda p: e.tokenize(p)  # noqa: E731
    answers = {}
    for rep in range(2):
        for arm, mask in (arms if rep == 0 else arms[::-1]):
            use(arm, mask)
            for n in (1, 8, 32):
                jobs = [ge.Job(tok(prompts[(rep * 7 + i) % len(prompts)]), 128) for i in range(n)]
                t = time.perf_counter()
                e.run(jobs)
                wall = time.perf_counter() - t
                row = {"arm": arm, "rep": rep, "n": n, "tok_s": round(sum(len(j.tokens) for j in jobs) / wall, 2),
                       "wall": round(wall, 2)}
                OUT["blocks"].append(row)
                print(json.dumps(row), flush=True)
            if rep == 0:
                got = []
                for k in range(0, len(fact_prompts), 8):
                    js = [ge.Job(tok(p), 48) for p in fact_prompts[k:k + 8]]
                    e.run(js)
                    got += [j.tokens for j in js]
                answers[arm] = got
            save()
    e.close()
    OUT["same_tokens"] = sum(a == b for a, b in zip(answers["kg"], answers["off"]))
    proc, port = mg2.server(str(gguf), 1, "0", "none")
    try:
        OUT["facts"] = {arm: mg2.correct([mg2.detok_server(port, i) for i in v], facts) for arm, v in answers.items()}
    finally:
        mg2.stop(proc)
    OUT["finished"] = time.time()
    save()


if __name__ == "__main__":
    main()
