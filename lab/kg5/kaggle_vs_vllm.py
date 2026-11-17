"""KG5 on a Kaggle T4: IronMule with its generated kernels against vLLM, in one session, the same prompts.

Input: the dataset `ironmule-kg5-build` (kg5_build.tar.gz: llama.cpp 4227c9b built for sm_75 with gr1, kg1, kg5 and
gr2, built off Kaggle with the CUDA 12.8 runtime beside it, glibc 2.34 or later). No build here.

1. The kernel library tuned for this GPU and Qwen3-4B Q4_K_M (ironmule/kernel_gen.py at MG2_COMMIT).
2. IronMule's engine as `ironmule serve` runs it on a GPU (one KV stream per request, flash attention, 32 sequences of
   2,048 cells, padding and speculation as shipped, Pilot off), arms switched in one process, each on the padding it
   measures for itself:
     base   the library off (llama.cpp's own kernels)
     kg4    the matmul kernels where the engine measures them faster, llama.cpp's attention
     kg5    and the library's decode attention where measured faster
   and in a second process with IRONMULE_CUDA_GRAPHS=2 (CUDA graphs for steps of several tokens too):
     kg5g   as kg5
   Blocks of 1 / 8 / 32 requests at once, 128 tokens, greedy, the MG2 prompts templated by llama-server, two rounds in
   alternating order; the 50 facts of lab/row1/facts.py 8 at a time through base and kg5 (and kg5g).
3. vLLM with Qwen3-4B-AWQ (4-bit, float16) on the same GPU through its OpenAI server, the same templated prompts, the
   same blocks (3 rounds) and the 50 facts.
Writes /kaggle/working/kg5_vs_vllm.json. MG2_COMMIT names the commit.
"""
import glob, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

COMMIT = os.environ["MG2_COMMIT"]
W = Path("/kaggle/working")
OUT = {"commit": COMMIT, "started": time.time(), "ironmule": {}, "vllm": {}}
NS, TOKENS = (1, 8, 32), 128


def save():
    (W / "kg5_vs_vllm.json").write_text(json.dumps(OUT, indent=1))


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def setup():
    src = Path("/tmp/kg5src")
    if not (src / "llama_cpp_python-0.3.16").exists():
        tar = next(iter(glob.glob("/kaggle/input/**/kg5_build.tar.gz", recursive=True)))
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
    return mg2, facts


def engine_arms(arms, prompts, fact_prompts):
    """The engine in this process; the library switched per arm. Returns blocks, facts' token ids and the measurements."""
    import ctypes

    from ironmule import ggml_engine as ge

    lib = os.environ["IRONMULE_KG_LIB"]
    ge.KG_LIB = lib
    e = ge.GgmlEngine(os.environ["KG5_GGUF"], seqs=32, seq_ctx=2048, flash_attn=True, shared=False)
    k = ctypes.CDLL(lib)
    for name in ("ironmule_kg_set_mask", "ironmule_kg_set_mask6", "ironmule_kg_set_fa"):
        getattr(k, name).argtypes = [ctypes.c_uint32]
    for name in ("ironmule_kg_get_mask", "ironmule_kg_get_mask6", "ironmule_kg_get_fa"):
        getattr(k, name).restype = ctypes.c_uint32
    chosen = (k.ironmule_kg_get_mask(), k.ironmule_kg_get_mask6(), k.ironmule_kg_get_fa())
    setting = {"base": (0, 0, 0), "kg4": (chosen[0], chosen[1], 0), "kg5": chosen, "kg5g": chosen}
    pads = {}
    res = {"kg": e.kg, "chosen": list(chosen), "blocks": [], "facts": {}}

    def use(arm):
        m4, m6, fa = setting[arm]
        k.ironmule_kg_set_mask(m4); k.ironmule_kg_set_mask6(m6); k.ironmule_kg_set_fa(fa)
        if arm not in pads:
            pads[arm] = e.measure_padding()
        e.pad_to = pads[arm]

    tok = e.tokenize
    for rep in range(2):
        for arm in (arms if rep == 0 else arms[::-1]):
            use(arm)
            for n in NS:
                jobs = [ge.Job(tok(prompts[(rep * 11 + i) % len(prompts)]), TOKENS) for i in range(n)]
                t, t_ns = time.perf_counter(), time.monotonic_ns()
                e.run(jobs)
                wall = time.perf_counter() - t
                row = {"arm": arm, "rep": rep, "n": n, "tok_s": round(sum(len(j.tokens) for j in jobs) / wall, 2),
                       "ttft": round(sum((j.first_ns - t_ns) / 1e9 for j in jobs) / n, 3), "wall": round(wall, 2)}
                res["blocks"].append(row)
                print(json.dumps(row), flush=True)
            if rep == 0 and arm != "kg4":
                got = []
                for i in range(0, len(fact_prompts), 8):
                    js = [ge.Job(tok(p), 48) for p in fact_prompts[i:i + 8]]
                    e.run(js)
                    got += [j.tokens for j in js]
                res["facts"][arm] = got
    res["pads"] = {a: {str(x): y for x, y in p.items()} for a, p in pads.items()}
    e.close()
    return res


def main():
    if os.environ.get("KG5_CHILD"):  # the second engine process (CUDA graphs on from the start)
        mg2, _ = setup()
        cfg = json.loads(Path(os.environ["KG5_CHILD"]).read_text())
        res = engine_arms(["kg5g"], cfg["prompts"], cfg["fact_prompts"])
        Path(os.environ["KG5_CHILD"] + ".out").write_text(json.dumps(res))
        return
    OUT["gpu"] = subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", shell=True,
                                capture_output=True, text=True).stdout
    mg2, facts = setup()
    gguf = Path("/tmp/Qwen3-4B-Q4_K_M.gguf")
    if not gguf.exists():
        sh(f"curl -sSL -o {gguf} https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf")
    os.environ["KG5_GGUF"] = str(gguf)
    from ironmule import kernel_gen

    t0 = time.time()
    table = kernel_gen.tune(str(gguf), Path("/tmp/kg"), log=lambda m: None)
    lib = kernel_gen.build(table, Path("/tmp/kg/kg5.so"))
    OUT["tune_s"] = round(time.time() - t0, 1)
    OUT["tune_entries"] = len(table["entries"])
    os.environ["IRONMULE_KG_LIB"] = str(lib)
    proc, port = mg2.server(str(gguf), 8, "0", "none")
    try:
        prompts = mg2.template(port, mg2.TEXTS, None)
        fact_prompts = mg2.template(port, [q + " Answer in one short sentence." for q, _ in facts], "/no_think")
    finally:
        mg2.stop(proc)
    save()
    try:
        OUT["ironmule"]["main"] = engine_arms(["base", "kg4", "kg5"], prompts, fact_prompts)
    except Exception as exc:  # noqa: BLE001 - recorded; vLLM still runs
        OUT["ironmule"]["main_error"] = f"{type(exc).__name__}: {exc}"[:800]
    save()
    cfg = W / "kg5_child.json"
    cfg.write_text(json.dumps({"prompts": prompts, "fact_prompts": fact_prompts}))
    child = subprocess.run([sys.executable, __file__], env={**os.environ, "KG5_CHILD": str(cfg),
                                                            "IRONMULE_CUDA_GRAPHS": "2"},
                           capture_output=True, text=True, timeout=3600)
    OUT["ironmule"]["graphs_log"] = child.stdout[-3000:] + child.stderr[-3000:]
    if Path(str(cfg) + ".out").exists():
        OUT["ironmule"]["graphs"] = json.loads(Path(str(cfg) + ".out").read_text())
    save()
    # the facts through llama-server's detokenizer, as MG2 checks them
    proc, port = mg2.server(str(gguf), 1, "0", "none")
    try:
        for part in ("main", "graphs"):
            for arm, ids in (OUT["ironmule"].get(part) or {}).get("facts", {}).items():
                OUT["ironmule"].setdefault("facts_right", {})[arm] = mg2.correct(
                    [mg2.detok_server(port, i) for i in ids], facts)
    finally:
        mg2.stop(proc)
    save()
    # vLLM AWQ on the same GPU, the same prompts
    vllm = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--served-model-name", "Qwen/Qwen3-4B",
            "--max-model-len", "4096", "--gpu-memory-utilization", "0.9", "--port", "8400", "--model",
            "Qwen/Qwen3-4B-AWQ", "--quantization", "awq", "--dtype", "half"]
    try:
        from transformers import AutoTokenizer  # vLLM's answers counted with the model's own tokenizer, as MG2 does

        hf = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        count = lambda text: len(hf.encode(text, add_special_tokens=False))  # noqa: E731
        sh(f"pip install -q vllm >> {W}/pip.log 2>&1")
        proc = mg2.python_server(vllm, 8400, {"CUDA_VISIBLE_DEVICES": "0"})
        try:
            for n in NS:
                mg2.block_openai(8400, prompts[:n], TOKENS, "Qwen/Qwen3-4B", count)  # a warm-up block
                OUT["vllm"][str(n)] = [mg2.block_openai(8400, prompts[:n], TOKENS, "Qwen/Qwen3-4B", count)[0]
                                       for _ in range(3)]
                print(json.dumps({"vllm": n, "rows": OUT["vllm"][str(n)]}), flush=True)
                save()
            answers = []
            for i in range(0, len(fact_prompts), 8):
                answers += mg2.block_openai(8400, fact_prompts[i:i + 8], 48, "Qwen/Qwen3-4B", count)[1]
            OUT["vllm"]["facts_right"] = mg2.correct(answers, facts)
        finally:
            mg2.kill_group(proc)
    except Exception as exc:  # noqa: BLE001
        OUT["vllm"]["error"] = f"{type(exc).__name__}: {exc}"[:800]
    OUT["finished"] = time.time()
    save()


if __name__ == "__main__":
    main()
