"""SG1 on a Kaggle T4: why one request on one GPU is slower through IronMule's engine than through llama-server (MG2 v5:
46.1 against 52.9 tok/s on Qwen3-4B), and whether CUDA graphs fix it.

llama.cpp 4227c9b turns CUDA graphs off on GPUs older than Ampere (`ggml-cuda.cu`: cc < GGML_CUDA_CC_AMPERE), so on a T4
every one-token decode step launches hundreds of kernels one by one; vLLM captures graphs there. GR1
(`native/ggml_patches/gr1`) keeps them on when IRONMULE_CUDA_GRAPHS=1, nothing else changes. This run reuses MG2's
build (the dataset `ironmule-mg2-cuda-build`), applies GR1, recompiles `ggml-cuda.cu` only, and measures per model one
request on GPU 0 (one warm-up block and 3 blocks of 128 tokens, greedy, SP4 off as in MG2):

  server_g0 / server_g1         llama-server -t 4, graphs off / on (twice each, alternating: _a, _b)
  engine_pilot_g0 / _g1         the engine as the product runs it (the Pilot picks threads), graphs off / on
  engine_nospec_g1a             graphs on, no speculation

v1 (2026-11-15) also swept 1, 2 and 4 threads (no difference with every layer on the GPU) and found the cached build
without `ggml-cuda.cu`; v2 builds afresh with GR1 applied before cmake (the dataset left off, about an hour).

and whether answers change: every arm's tokens against its graphs-off twin, and the 50 facts one request at a time
(the graph path) through the engine and llama-server with graphs off and on. Writes /kaggle/working/sg1.json.
"""
import json, os, runpy, subprocess, sys, time, urllib.request
from pathlib import Path

COMMIT = os.environ["MG2_COMMIT"]
W = Path("/kaggle/working")
src = urllib.request.urlopen(f"https://raw.githubusercontent.com/Tobayko/IronMule/{COMMIT}/lab/mg2/kaggle_gpu.py",
                             timeout=60).read()
(W / "kaggle_gpu.py").write_bytes(src)
mg2 = type(sys)("mg2")
exec(compile(src, str(W / "kaggle_gpu.py"), "exec"), mg2.__dict__)
OUT = {"commit": COMMIT, "started": time.time(), "models": {}}


def save():
    (W / "sg1.json").write_text(json.dumps(OUT, indent=1))


def with_env(extra, make):
    old = {k: os.environ.get(k) for k in extra}
    os.environ.update(extra)
    try:
        return make()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def patch_and_rebuild():
    tree = mg2.SRC / "llama_cpp_python-0.3.16/vendor/llama.cpp"
    diff = W / "gr1.diff"
    diff.write_bytes(urllib.request.urlopen(
        f"https://raw.githubusercontent.com/Tobayko/IronMule/{COMMIT}/native/ggml_patches/gr1/ggml-cuda.cu.diff",
        timeout=60).read())
    t0 = time.time()
    # the copied build tree is older than nothing: make every output newer than the sources, then patch one file
    mg2.sh(f"cd {tree} && find build -type f -exec touch {{}} + && patch ggml/src/ggml-cuda/ggml-cuda.cu {diff} && "
           f"cmake --build build --target llama-server llama -j 4 > {W}/gr1_build.log 2>&1")
    OUT["rebuild_s"] = round(time.time() - t0, 1)
    mg2.log("GR1 rebuilt", OUT["rebuild_s"])


def patch_during_build():
    """Without the cached build (SG1 v2): MG2's build() downloads the tree and runs cmake; GR1 goes in right before."""
    diff = W / "gr1.diff"
    diff.write_bytes(urllib.request.urlopen(
        f"https://raw.githubusercontent.com/Tobayko/IronMule/{COMMIT}/native/ggml_patches/gr1/ggml-cuda.cu.diff",
        timeout=60).read())
    plain = mg2.sh

    def sh(cmd, **kw):
        if isinstance(cmd, str) and "cmake -B build" in cmd:
            root = cmd.split("cd ", 1)[1].split(" &&", 1)[0]
            plain(f"cd {root} && patch ggml/src/ggml-cuda/ggml-cuda.cu {diff}")
            OUT["gr1_patched_before_build"] = True
        return plain(cmd, **kw)
    mg2.sh = sh


def engine_arm(gguf, prompts, extra, speculate=True):
    eng = with_env(extra, lambda: mg2.Engines(gguf, 1, ["0"], speculate=speculate))
    try:
        eng.block(prompts, mg2.TOKENS)
        blocks, texts = [], None
        for _ in range(3):
            b, texts = eng.block(prompts, mg2.TOKENS)
            blocks.append(b)
        return {"blocks": blocks, "texts": texts, "diag": eng.last, "summary": eng.summary}
    finally:
        eng.close()


def server_arm(gguf, prompts, extra):
    proc, port = with_env(extra, lambda: mg2.server(gguf, 1, "0", "none"))
    try:
        mg2.block_servers([port], prompts, mg2.TOKENS)
        blocks, texts = [], None
        for _ in range(3):
            b, texts = mg2.block_servers([port], prompts, mg2.TOKENS)
            blocks.append(b)
        return {"blocks": blocks, "texts": texts}
    finally:
        mg2.stop(proc)


def run_model(repo, name, facts):
    gguf = str(W / name)
    if not Path(gguf).exists():
        mg2.sh(f"cd {W} && curl -sSL -o {name}.part https://huggingface.co/{repo}/resolve/main/{name} && mv {name}.part {name}")
    m = OUT["models"].setdefault(name, {"arms": {}})
    proc, port = mg2.server(gguf, 1, "0", "none")
    try:
        system = "/no_think" if "Qwen3" in name else None
        prompts = mg2.template(port, mg2.TEXTS[:1], None)
        fact_prompts = mg2.template(port, [q + " Answer in one short sentence." for q, _ in facts], system)
    finally:
        mg2.stop(proc)
    g = lambda on: {"IRONMULE_CUDA_GRAPHS": "1" if on else "0"}  # noqa: E731
    arms = []
    for rep in ("a", "b"):  # graphs off and on alternate twice: v1 saw the same setting differ by up to 3 tok/s
        arms += [(f"server_g0{rep}", "server", g(0), True), (f"server_g1{rep}", "server", g(1), True),
                 (f"engine_pilot_g0{rep}", "engine", g(0), True), (f"engine_pilot_g1{rep}", "engine", g(1), True)]
    arms.append(("engine_nospec_g1a", "engine", g(1), False))
    texts = {}
    for arm, kind, extra, spec in arms:
        try:
            r = server_arm(gguf, prompts, extra) if kind == "server" else engine_arm(gguf, prompts, extra, spec)
            texts[arm] = r.pop("texts")
            m["arms"][arm] = r
            mg2.log(name, arm, json.dumps([{k: round(v, 3) for k, v in b.items()} for b in r["blocks"]]))
        except Exception as exc:  # noqa: BLE001 - recorded, the run goes on
            m["arms"][arm] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            mg2.log(name, arm, "ERROR", exc)
        save()
    m["same_tokens_as_g0_twin"] = {arm: texts.get(arm) == texts.get(arm.replace("_g1", "_g0"))
                                   for arm in texts if "_g1" in arm and arm.replace("_g1", "_g0") in texts}
    m["same_tokens_as_engine_pilot_g0a"] = {arm: texts[arm] == texts.get("engine_pilot_g0a")
                                           for arm in texts if arm.startswith("engine")}
    # the facts one request at a time, so that every step is a one-token step (the graph path)
    answers = {}
    for arm, extra in (("engine_g0", g(0)), ("engine_g1", g(1))):
        try:
            eng = with_env(extra, lambda: mg2.Engines(gguf, 1, ["0"], speculate=True))
            answers[arm] = [eng.block([p], 48)[1][0] for p in fact_prompts]
            eng.close()
        except Exception as exc:  # noqa: BLE001
            m[f"facts_{arm}"] = f"error: {exc}"[:200]
    for arm, extra in (("server_g0", g(0)), ("server_g1", g(1))):
        try:
            proc, port = with_env(extra, lambda: mg2.server(gguf, 1, "0", "none"))
            answers[arm] = [mg2.block_servers([port], [p], 48)[1][0] for p in fact_prompts]
            mg2.stop(proc)
        except Exception as exc:  # noqa: BLE001
            m[f"facts_{arm}"] = f"error: {exc}"[:200]
    proc, port = mg2.server(gguf, 1, "0", "none")
    try:
        for arm, ans in answers.items():
            words = [mg2.detok_server(port, a) for a in ans] if arm.startswith("engine") else ans
            m[f"facts_{arm}"] = mg2.correct(words, facts)
    finally:
        mg2.stop(proc)
    if "engine_g0" in answers and "engine_g1" in answers:
        m["facts_engine_same_tokens"] = sum(a == b for a, b in zip(answers["engine_g0"], answers["engine_g1"]))
    if "server_g0" in answers and "server_g1" in answers:
        m["facts_server_same_text"] = sum(a == b for a, b in zip(answers["server_g0"], answers["server_g1"]))
    mg2.log(name, "facts", {k: v for k, v in m.items() if k.startswith("facts")})
    save()


def main():
    OUT["nvidia_smi"] = subprocess.run("nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv,noheader",
                                       shell=True, capture_output=True, text=True).stdout
    patch_during_build()
    facts = mg2.build()
    OUT["build_cached"] = mg2.REPORT.get("build_cached")
    OUT["build_s"] = round(mg2.REPORT.get("build_s", 0), 1)
    if OUT["build_cached"]:
        try:
            patch_and_rebuild()
        except Exception as exc:  # noqa: BLE001 - the arms still run, on the stock build
            OUT["rebuild_error"] = f"{type(exc).__name__}: {exc}"[:300]
    cu = mg2.SRC / "llama_cpp_python-0.3.16/vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu"
    OUT["gr1_in_source"] = cu.exists() and "IRONMULE_CUDA_GRAPHS" in cu.read_text()
    save()
    if os.environ.get("SG1_KEEP_BUILD") == "1":  # the patched build as this run's output, for later runs
        mg2.sh(f"cd {mg2.SRC} && tar czf {W}/sg1_build.tar.gz llama_cpp_python-0.3.16/vendor/llama.cpp/build/bin "
               f"llama_cpp_python-0.3.16/llama_cpp")
    for repo, name in mg2.MODELS:
        try:
            run_model(repo, name, facts)
        except Exception as exc:  # noqa: BLE001
            OUT["models"].setdefault(name, {})["error"] = f"{type(exc).__name__}: {exc}"[:300]
            mg2.log(name, "ERROR", exc)
        save()
        for f in W.glob("*.gguf"):
            f.unlink()
    OUT["finished"] = time.time()
    save()


main()
