"""MG2 on a Kaggle "GPU T4 x2" session: IronMule's ggml engine on CUDA, one GPU, and two GPUs, against llama-server
built from the same llama.cpp tree, on the open-weight models most people run (Qwen3, Gemma 3, Mistral, Llama 3.1).

Builds llama-cpp-python 0.3.16's vendored llama.cpp (4227c9b) with CUDA for the T4 (sm_75), downloads the IronMule
engine files at a pushed commit, downloads ungated Q4_K_M GGUFs, and measures per model and N concurrent streamed
requests (N = 1, 8, 32; one warm-up block and 3 blocks; 128 tokens; greedy; the model's chat template):

  server_1gpu    llama-server -ngl 99 -sm none -fa, -np N, -c N*2048 (one GPU)
  server_split   llama-server -ngl 99 -sm layer (the model split over both GPUs, llama.cpp's multi-GPU way)
  server_x2      two llama-servers, one per GPU, requests alternated (stock llama.cpp deployed per GPU)
  ironmule_1gpu  IronMule's ggml engine (continuous batching, one call per step, prefix reuse, speculation from the
                 context) on one GPU, N sequences of 2,048 tokens
  ironmule_x2    one engine per GPU (`ironmule serve --gpus all`), requests split evenly

Answers: the 50 checkable facts of lab/row1/facts.py through llama-server and through the engine (both greedy).
Writes /kaggle/working/mg2.json. Usage on Kaggle: the kernel runs this file with the commit to fetch in MG2_COMMIT.
"""
import glob, http.client, json, os, socket, subprocess, sys, threading, time, urllib.request
from pathlib import Path

W = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(os.environ.get("MG2_WORK", "/tmp/mg2"))
W.mkdir(parents=True, exist_ok=True)
COMMIT = os.environ.get("MG2_COMMIT", "main")
MODELS = [("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
          ("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
          ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
          ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")]
NS = [int(x) for x in os.environ.get("MG2_NS", "1,8,32").split(",")]
TOKENS, CTX = 128, 2048
PYTHON_INPUTS, PYTHON_INPUTS_COUNT = ((), ()), None
REPORT: dict = {"commit": COMMIT, "started": time.time(), "models": {}}
LOG = open(W / "mg2.log", "a")


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


def save():
    (W / "mg2.json").write_text(json.dumps(REPORT, indent=1))


def sh(cmd, **kw):
    log("$", cmd if isinstance(cmd, str) else " ".join(cmd))
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


# --- build ---------------------------------------------------------------------------------------------------------
SRC = W / "src"
LIB = SRC / "llama_cpp_python-0.3.16/vendor/llama.cpp/build/bin"


def build():
    t0 = time.time()
    if not (LIB / "libllama.so").exists():
        SRC.mkdir(exist_ok=True)
        sh(f"cd {SRC} && pip download llama_cpp_python==0.3.16 --no-binary :all: --no-deps -d . -q && tar xzf llama_cpp_python-0.3.16.tar.gz")
        root = SRC / "llama_cpp_python-0.3.16/vendor/llama.cpp"
        # Kaggle's default toolkit is CUDA 13; llama.cpp 4227c9b predates it, so a CUDA 12 toolkit is preferred when
        # one is installed. CUDA::cuda_driver needs libcuda: the toolkit's stub, or the driver itself.
        cuda = next(iter(sorted(glob.glob("/usr/local/cuda-12*"), reverse=True)), "/usr/local/cuda")
        driver = next((p for p in [*glob.glob(f"{cuda}/lib64/stubs/libcuda.so"),
                                   *glob.glob(f"{cuda}/targets/x86_64-linux/lib/stubs/libcuda.so"),
                                   *glob.glob("/usr/local/cuda/lib64/stubs/libcuda.so"),
                                   *glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so*"),
                                   *glob.glob("/usr/local/nvidia/lib64/libcuda.so*")]), None)
        REPORT["cuda"] = {"toolkit": cuda, "driver_lib": driver, "toolkits": glob.glob("/usr/local/cuda*")}
        sh(f"cd {root} && cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DGGML_CUDA=ON "
           f"-DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=ON -DCUDAToolkit_ROOT={cuda} "
           f"-DCMAKE_CUDA_COMPILER={cuda}/bin/nvcc -DCMAKE_LIBRARY_PATH={cuda}/lib64/stubs "
           + (f"-DCUDA_cuda_driver_LIBRARY={driver} " if driver else "") + "> build.log 2>&1 && "
           f"cmake --build build --target llama-server llama -j 4 >> build.log 2>&1")
    REPORT["build_s"] = time.time() - t0
    pkg = W / "pkg/ironmule"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for name in ("ggml_engine.py", "speculation.py", "ggml_agent.py", "pilot.py", "draft_model.py", "pilot_weights.npz"):
        url = f"https://raw.githubusercontent.com/Tobayko/IronMule/{COMMIT}/ironmule/{name}"
        (pkg / name).write_bytes(urllib.request.urlopen(url, timeout=60).read())
    facts = urllib.request.urlopen(f"https://raw.githubusercontent.com/Tobayko/IronMule/{COMMIT}/lab/row1/facts.py",
                                   timeout=60).read().decode()
    return eval(facts[facts.index("Q = [") + 4:facts.index("]\nge.") + 1])


def env(gpus=None):
    e = {**os.environ, "IRONMULE_GGML_BINDINGS": str(SRC / "llama_cpp_python-0.3.16"), "LLAMA_CPP_LIB_PATH": str(LIB),
         "PYTHONPATH": str(W / "pkg")}
    if gpus is not None:
        e["CUDA_VISIBLE_DEVICES"] = gpus
    return e


# --- llama-server --------------------------------------------------------------------------------------------------
def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def server(gguf, n, gpus, split):
    port = free_port()
    cmd = [str(LIB / "llama-server"), "-m", gguf, "--host", "127.0.0.1", "--port", str(port), "-np", str(n),
           "-c", str(n * CTX), "-ngl", "99", "-sm", split, "-fa", "--temp", "0", "--no-webui", "-t", "4"]
    proc = subprocess.Popen(cmd, env=env(gpus), stdout=subprocess.DEVNULL, stderr=open(W / "server.err", "a"))
    for _ in range(600):
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5); c.request("GET", "/health")
            if c.getresponse().status == 200:
                return proc, port
        except Exception:
            pass
        if proc.poll() is not None:
            break
        time.sleep(1)
    proc.kill()
    raise RuntimeError("llama-server did not start")


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def post(port, path, body, timeout=3600):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    return c.getresponse()


def template(port, texts, system=None):
    out = []
    for text in texts:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        out.append(json.loads(post(port, "/apply-template", {"messages": msgs}).read())["prompt"])
    return out


def ask(port, prompt, limit, t0, out, i):
    r = post(port, "/completion", {"prompt": prompt, "n_predict": limit, "temperature": 0, "stream": True,
                                   "cache_prompt": True})
    first, n, text = None, 0, []
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        data = json.loads(line[5:])
        if data.get("content"):
            first = first if first is not None else time.perf_counter() - t0
            text.append(data["content"])
        if data.get("stop"):
            n = data.get("tokens_predicted", n)
            break
        n += 1 if data.get("content") is not None else 0
    out[i] = {"first_s": first, "tokens": n, "text": "".join(text)}


def ask_openai(port, prompt, limit, t0, out, i, model):
    """vLLM / SGLang: the same templated prompt through /v1/completions, streamed."""
    r = post(port, "/v1/completions", {"model": model, "prompt": prompt, "max_tokens": limit, "temperature": 0,
                                       "stream": True})
    first, text = None, []
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        piece = (json.loads(line[5:]).get("choices") or [{}])[0].get("text") or ""
        if piece:
            first = first if first is not None else time.perf_counter() - t0
            text.append(piece)
    out[i] = {"first_s": first, "tokens": None, "text": "".join(text)}


def block_openai(port, prompts, limit, model, count):
    out, t0 = [None] * len(prompts), time.perf_counter()
    th = [threading.Thread(target=ask_openai, args=(port, prompts[i], limit, t0, out, i, model)) for i in range(len(prompts))]
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.perf_counter() - t0
    tokens = sum(count(o["text"]) for o in out)
    return {"tok_s": tokens / wall, "ttft_s": sum(o["first_s"] or wall for o in out) / len(out), "wall_s": wall,
            "tokens": tokens}, [o["text"] for o in out]


def python_server(command, port, env_extra=None, timeout=1800):
    proc = subprocess.Popen(command, env={**os.environ, **(env_extra or {})}, stdout=open(W / "pyserver.log", "a"),
                            stderr=subprocess.STDOUT)
    for _ in range(timeout // 2):
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5); c.request("GET", "/v1/models")
            if c.getresponse().status == 200:
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError("server exited")
        time.sleep(2)
    proc.kill()
    raise RuntimeError("server did not start")


def run_python_engines(prompts, fact_prompts, facts, count):
    """vLLM and SGLang with Qwen3-4B in float16 (a T4 has no bfloat16), on one GPU, the same templated prompts."""
    for name, install, command in (
            ("vllm_1gpu", "pip install -q vllm", [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model",
             "Qwen/Qwen3-4B", "--dtype", "half", "--max-model-len", "4096", "--gpu-memory-utilization", "0.9",
             "--port", "8400"]),
            ("sglang_1gpu", "pip install -q 'sglang[all]'", [sys.executable, "-m", "sglang.launch_server", "--model-path",
             "Qwen/Qwen3-4B", "--dtype", "half", "--attention-backend", "triton", "--port", "8400",
             "--mem-fraction-static", "0.85"])):
        m = REPORT["models"].setdefault("Qwen3-4B-hf-fp16", {"arms": {}})
        rows = m["arms"].setdefault(name, {})
        try:
            sh(install + " >> " + str(W / "pip.log") + " 2>&1")
            proc = python_server(command, 8400, {"CUDA_VISIBLE_DEVICES": "0"})
        except Exception as exc:  # noqa: BLE001 - recorded, the run goes on
            rows["error"] = f"{type(exc).__name__}: {exc}"[:300]
            log(name, "ERROR", exc); save(); continue
        try:
            for n in NS:
                block_openai(8400, prompts[:n], TOKENS, "Qwen/Qwen3-4B", count)
                rows[str(n)] = [block_openai(8400, prompts[:n], TOKENS, "Qwen/Qwen3-4B", count)[0] for _ in range(3)]
                log(name, n, json.dumps(rows[str(n)]))
                save()
            answers = []
            for k in range(0, len(fact_prompts), 8):
                answers += block_openai(8400, fact_prompts[k:k + 8], 48, "Qwen/Qwen3-4B", count)[1]
            m["facts_" + name] = correct(answers, facts)
        except Exception as exc:  # noqa: BLE001
            rows["error"] = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            proc.kill()
        save()


def block_servers(ports, prompts, limit):
    out, t0 = [None] * len(prompts), time.perf_counter()
    th = [threading.Thread(target=ask, args=(ports[i % len(ports)], prompts[i], limit, t0, out, i))
          for i in range(len(prompts))]
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.perf_counter() - t0
    return {"tok_s": sum(o["tokens"] for o in out) / wall, "ttft_s": sum(o["first_s"] or wall for o in out) / len(out),
            "wall_s": wall, "tokens": sum(o["tokens"] for o in out)}, [o["text"] for o in out]


# --- IronMule's engine, in a child per GPU ----------------------------------------------------------------------------
ENGINE_CHILD = r"""
import json, sys, time
from ironmule.ggml_engine import GgmlEngine, Job
cfg = json.loads(sys.stdin.readline())
e = GgmlEngine(cfg["gguf"], seqs=cfg["seqs"], seq_ctx=cfg["ctx"], speculate=cfg.get("speculate", True), flash_attn=True)
print(json.dumps({"ready": True, "summary": e.summary()}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req.get("quit"): break
    jobs = [Job(e.tokenize(p), req["limit"]) for p in req["prompts"]]
    start = time.monotonic_ns()
    while time.monotonic_ns() < req["at_ns"]: pass
    t0 = time.monotonic_ns()
    e.run(jobs)
    t1 = time.monotonic_ns()
    print(json.dumps({"t0": t0, "t1": t1, "tokens": [len(j.tokens) for j in jobs],
                      "first": [(j.first_ns - t0) / 1e9 for j in jobs], "texts": [j.tokens for j in jobs],
                      "steps": e.steps}), flush=True)
"""


class Engines:
    def __init__(self, gguf, seqs, gpus_list, speculate=True):
        self.procs = []
        for gpus in gpus_list:
            p = subprocess.Popen([sys.executable, "-c", ENGINE_CHILD], env=env(gpus), stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=open(W / "engine.err", "a"), text=True)
            p.stdin.write(json.dumps({"gguf": gguf, "seqs": seqs, "ctx": CTX, "speculate": speculate}) + "\n")
            p.stdin.flush()
            self.procs.append(p)
        self.summary = [json.loads(self._line(p)) for p in self.procs]

    @staticmethod
    def _line(p):
        line = p.stdout.readline()
        if not line:
            raise RuntimeError("engine child died")
        return line

    def block(self, prompts, limit):
        at = time.monotonic_ns() + int(0.3e9)
        parts = [prompts[i::len(self.procs)] for i in range(len(self.procs))]
        for p, part in zip(self.procs, parts):
            p.stdin.write(json.dumps({"prompts": part, "limit": limit, "at_ns": at}) + "\n"); p.stdin.flush()
        rows = [json.loads(self._line(p)) for p in self.procs]
        wall = (max(r["t1"] for r in rows) - min(r["t0"] for r in rows)) / 1e9
        tokens = sum(sum(r["tokens"]) for r in rows)
        firsts = [f for r in rows for f in r["first"]]
        texts = [None] * len(prompts)
        for k, r in enumerate(rows):
            for j, ids in enumerate(r["texts"]):
                texts[k + j * len(self.procs)] = ids
        return {"tok_s": tokens / wall, "ttft_s": sum(firsts) / len(firsts), "wall_s": wall, "tokens": tokens}, texts

    def close(self):
        for p in self.procs:
            try:
                p.stdin.write(json.dumps({"quit": True}) + "\n"); p.stdin.flush(); p.wait(timeout=60)
            except Exception:
                p.kill()


# --- runs ---------------------------------------------------------------------------------------------------------------
TOPICS = ["a lighthouse keeper", "a lost robot", "a mountain village", "a river in winter", "a clockmaker",
          "a desert caravan", "a library at night", "a sailing ship", "a city garden", "an old bakery", "a train journey",
          "a forest fire watch", "a chess player", "a beekeeper", "a snowstorm", "a market day"]
TEXTS = [f"Write a short story about {t}." for t in TOPICS * 4]


def detok_server(port, ids):
    return json.loads(post(port, "/detokenize", {"tokens": ids}).read())["content"]


def correct(answers, facts):
    return sum(any(k in (a or "").lower() for k in keys) for a, (_, keys) in zip(answers, facts))


def run_model(repo, name, facts):
    gguf = str(W / name)
    if not Path(gguf).exists():
        sh(f"cd {W} && curl -sSL -o {name}.part https://huggingface.co/{repo}/resolve/main/{name} && mv {name}.part {name}")
    m = REPORT["models"].setdefault(name, {"repo": repo, "arms": {}})
    proc, port = server(gguf, 8, "0", "none")
    try:
        system = "/no_think" if "Qwen3" in name else None
        prompts = template(port, TEXTS, None)
        fact_prompts = template(port, [q + " Answer in one short sentence." for q, _ in facts], system)
        if "Qwen3-4B" in name:  # the same templated prompts for vLLM and SGLang, counted with the model's tokenizer
            global PYTHON_INPUTS, PYTHON_INPUTS_COUNT
            PYTHON_INPUTS = (prompts, fact_prompts)
            from transformers import AutoTokenizer  # the same tokenizer as the GGUF's; llama-server is stopped by then

            hf = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
            PYTHON_INPUTS_COUNT = lambda text: len(hf.encode(text, add_special_tokens=False))  # noqa: E731
        ans = []
        for k in range(0, len(fact_prompts), 8):
            ans += block_servers([port], fact_prompts[k:k + 8], 48)[1]
        m["facts_server"] = correct(ans, facts)
    finally:
        stop(proc)
    for arm, kind, gpus, split in (("server_1gpu", "server", ["0"], "none"), ("server_split", "server", ["0,1"], "layer"),
                                   ("server_x2", "server", ["0", "1"], "none"), ("ironmule_1gpu", "engine", ["0"], None),
                                   ("ironmule_x2", "engine", ["0", "1"], None)):
        rows = m["arms"].setdefault(arm, {})
        for n in NS:
            if str(n) in rows:
                continue
            try:
                each = max(1, n // len(gpus)) if kind == "engine" or arm == "server_x2" else n
                if kind == "server":
                    started = [server(gguf, each, g, split) for g in gpus]
                    ports = [p for _, p in started]
                    run = lambda pr: block_servers(ports, pr, TOKENS)
                else:
                    eng = Engines(gguf, each, gpus)
                    run = lambda pr: eng.block(pr, TOKENS)
                try:
                    run(prompts[:n])  # warm-up block
                    blocks = [run(prompts[:n])[0] for _ in range(3)]
                finally:
                    if kind == "server":
                        [stop(p) for p, _ in started]
                    else:
                        eng.close()
                rows[str(n)] = blocks
                log(name, arm, n, json.dumps([{k: round(v, 3) for k, v in b.items()} for b in blocks]))
            except Exception as exc:  # noqa: BLE001 - a failed arm is recorded and the run goes on
                rows[str(n)] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                log(name, arm, n, "ERROR", exc)
            save()
    # the engine's answers to the facts, decoded by llama-server's tokenizer
    try:
        eng = Engines(gguf, 8, ["0"], speculate=True)
        texts = []
        for k in range(0, len(fact_prompts), 8):
            _, ids = eng.block(fact_prompts[k:k + 8], 48)
            texts += ids
        eng.close()
        proc, port = server(gguf, 1, "0", "none")
        try:
            m["facts_ironmule"] = correct([detok_server(port, ids) for ids in texts], facts)
        finally:
            stop(proc)
    except Exception as exc:  # noqa: BLE001
        m["facts_ironmule"] = f"error: {exc}"[:200]
    log(name, "facts", m.get("facts_server"), m.get("facts_ironmule"))
    save()


def main():
    REPORT["nvidia_smi"] = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                                          capture_output=True, text=True).stdout
    save()
    facts = build()
    log("built", REPORT["build_s"])
    for index, (repo, name) in enumerate(MODELS):
        try:
            run_model(repo, name, facts)
            if index == 0 and os.environ.get("MG2_PYTHON_ENGINES", "1") == "1":
                run_python_engines(*PYTHON_INPUTS, facts, PYTHON_INPUTS_COUNT)
        except Exception as exc:  # noqa: BLE001
            REPORT["models"].setdefault(name, {})["error"] = f"{type(exc).__name__}: {exc}"[:300]
            log(name, "ERROR", exc)
        save()
        for f in W.glob("*.gguf"):
            f.unlink()  # the session's disk
    REPORT["finished"] = time.time()
    save()


if __name__ == "__main__":
    main()
