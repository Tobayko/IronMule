"""KG1 v3 on a Kaggle T4: the generated kernels inside the engine, end to end.

1. Builds llama.cpp 4227c9b with IronMule's patches (tools/build_ggml.py --cuda 75: fa1, row1, gr1 and the kg1 hook),
   about an hour, and keeps the build as this run's output for later runs.
2. Tunes the kernel library for this GPU and Qwen3-4B Q4_K_M (ironmule/kernel_gen.py: every variant on the model's
   own Q4_K matrices, the fastest correct one per shape and token count) and builds it.
3. Per arm, 1 / 2 / 4 / 8 requests at once (128 tokens, one warm-up block and 3 blocks, greedy, SP4 off):
   llama-server (stock kernels: the hook stays idle without IRONMULE_KG_LIB), the engine with the shared pool and
   PAD1, and the same engine with IRONMULE_KG_LIB (it keeps the library only for the token counts it measures faster).
4. Answers: the 50 facts through the engine with and without the library, token for token against each other.
Writes /kaggle/working/kg1_e2e.json. MG2_COMMIT names the commit to fetch.
"""
import glob, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

COMMIT = os.environ["MG2_COMMIT"]
W = Path("/kaggle/working")
OUT = {"commit": COMMIT, "started": time.time()}
NS = (1, 2, 4, 8)


def save():
    (W / "kg1_e2e.json").write_text(json.dumps(OUT, indent=1))


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def main():
    OUT["gpu"] = subprocess.run("nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader",
                                shell=True, capture_output=True, text=True).stdout
    sh(f"cd {W} && curl -sSL https://github.com/Tobayko/IronMule/archive/{COMMIT}.tar.gz | tar xz && mv IronMule-{COMMIT} repo")
    repo = W / "repo"
    cuda = next(iter(sorted(glob.glob("/usr/local/cuda-12*"), reverse=True)), "/usr/local/cuda")
    os.environ["PATH"] = f"{cuda}/bin:{os.environ['PATH']}"
    tar = next(iter(glob.glob("/kaggle/input/**/kg1_build.tar.gz", recursive=True)), None)
    dest = W / "ggml"
    if tar:
        dest.mkdir(parents=True, exist_ok=True)
        sh(f"cd {dest} && tar xzf {tar}")
        OUT["build_from"] = tar
    else:
        t0 = time.time()
        built = subprocess.run(f"cd {repo} && {sys.executable} tools/build_ggml.py --cuda 75 --dest {dest}", shell=True,
                               capture_output=True, text=True)
        OUT["build_s"] = round(time.time() - t0, 1)
        OUT["build_tail"] = (built.stdout + built.stderr)[-1500:]
        save()
        sh(f"cd {dest} && tar czf {W}/kg1_build.tar.gz ironmule/cuda/llama_cpp_python-0.3.16/vendor/llama.cpp/build/bin "
           f"ironmule/cuda/llama_cpp_python-0.3.16/llama_cpp ironmule/cuda/llama_cpp_python-0.3.16/vendor/llama.cpp/gguf-py")
    source = dest / "ironmule/cuda/llama_cpp_python-0.3.16"
    # MG2's harness, pointed at this build
    src = (repo / "lab/mg2/kaggle_gpu.py").read_bytes()
    os.environ["MG2_COMMIT"] = COMMIT
    mg2 = type(sys)("mg2")
    exec(compile(src, str(repo / "lab/mg2/kaggle_gpu.py"), "exec"), mg2.__dict__)
    mg2.SRC, mg2.LIB = source.parent, source / "vendor/llama.cpp/build/bin"
    facts = mg2.build()  # the build exists: only the engine files and the facts
    os.environ["IRONMULE_GGML_BINDINGS"] = str(source)
    name = "Qwen3-4B-Q4_K_M.gguf"
    gguf = W / name
    sh(f"curl -sSL -o {gguf} https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/{name}")
    # 2. tune and build the library
    sys.path.insert(0, str(W / "pkg"))
    from ironmule import kernel_gen  # noqa: E402

    t0 = time.time()
    try:
        table = kernel_gen.tune(str(gguf), W / "kg", log=lambda m: print(m, flush=True))
        lib = kernel_gen.build(table, W / "kg" / "kg_qwen3_4b.so")
        OUT["tune"] = {k: v for k, v in table.items() if k != "measured"}
        (W / "kg1_tune_measured.json").write_text(json.dumps(table["measured"]))
    except Exception as exc:  # noqa: BLE001 - recorded; the engine arms without the library still run
        OUT["tune_error"] = f"{type(exc).__name__}: {exc}"[:1500]
        lib = None
    OUT["tune_s"] = round(time.time() - t0, 1)
    save()
    # 3. arms
    proc, port = mg2.server(str(gguf), 8, "0", "none")
    try:
        prompts = mg2.template(port, mg2.TEXTS, None)
        fact_prompts = mg2.template(port, [q + " Answer in one short sentence." for q, _ in facts], "/no_think")
    finally:
        mg2.stop(proc)
    arms = OUT.setdefault("arms", {})
    plan = [("server", None), ("engine", None)] + ([("engine_kg", str(lib))] if lib else [])
    for rep in ("a", "b"):
        for arm, kg in plan:
            key = f"{arm}_{rep}"
            rows = arms.setdefault(key, {})
            for n in NS:
                os.environ["IRONMULE_KG_LIB"] = kg or ""
                try:
                    if arm == "server":
                        started = mg2.server(str(gguf), n, "0", "none")
                        run = lambda pr: mg2.block_servers([started[1]], pr, mg2.TOKENS)  # noqa: E731
                    else:
                        eng = mg2.Engines(str(gguf), n, ["0"], shared=True)
                        run = lambda pr: eng.block(pr, mg2.TOKENS)  # noqa: E731
                    try:
                        run(prompts[:n])
                        rows[str(n)] = [run(prompts[:n])[0] for _ in range(3)]
                        if arm != "server":
                            rows[f"diag_{n}"] = eng.last
                    finally:
                        mg2.stop(started[0]) if arm == "server" else eng.close()
                    mg2.log(key, n, json.dumps([round(b["tok_s"], 1) for b in rows[str(n)]]))
                except Exception as exc:  # noqa: BLE001
                    rows[str(n)] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                save()
    os.environ["IRONMULE_KG_LIB"] = ""
    # 4. answers
    answers = {}
    for arm, kg in plan[1:]:
        os.environ["IRONMULE_KG_LIB"] = kg or ""
        for n in (1, 8):
            eng = mg2.Engines(str(gguf), n, ["0"], shared=True)
            ids = []
            for k in range(0, len(fact_prompts), n):
                ids += eng.block(fact_prompts[k:k + n], 48)[1]
            OUT.setdefault("facts_diag", {})[f"{arm}_{n}"] = eng.last
            eng.close()
            answers[f"{arm}_{n}"] = ids
    os.environ["IRONMULE_KG_LIB"] = ""
    proc, port = mg2.server(str(gguf), 1, "0", "none")
    try:
        OUT["facts"] = {k: mg2.correct([mg2.detok_server(port, i) for i in v], facts) for k, v in answers.items()}
    finally:
        mg2.stop(proc)
    for n in (1, 8):
        a, b = answers.get(f"engine_{n}"), answers.get(f"engine_kg_{n}")
        if a and b:
            OUT.setdefault("same_tokens", {})[str(n)] = sum(x == y for x, y in zip(a, b))
    OUT["finished"] = time.time()
    save()


main()
