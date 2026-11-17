"""KG3 end to end on a Kaggle T4: `ironmule serve --gguf` itself, one KV stream per request (the GPU default since KG3)
against the shared pool (`IRONMULE_GGML_STREAMS=0`), through the HTTP API.

1. MG2's CUDA build of llama.cpp (the dataset `ironmule-mg2-cuda-build`), the repository at MG2_COMMIT installed with
   `pip install -e .[cpu]`, `ironmule setup --mode server`, Qwen3-4B Q4_K_M.
2. Four server starts in the order streams, pool, pool, streams on one state directory (the gate and the width are
   measured at the first start and reused); `/health` of each is kept.
3. Per start: 1, 8 and 32 chat requests at once (`/v1/chat/completions`, 128 tokens, greedy, the reply's usage
   counted), and the 50 facts of lab/row1/facts.py 8 at a time.
Writes /kaggle/working/kg3_serve.json. LOCAL=dir runs the same on this machine (CPU: the pool arm only).
"""
import glob, http.client, json, os, socket, subprocess, sys, threading, time
from pathlib import Path

COMMIT = os.environ.get("MG2_COMMIT", "main")
LOCAL = os.environ.get("KG3_LOCAL")
W = Path(LOCAL or "/kaggle/working")
OUT = {"commit": COMMIT, "started": time.time(), "arms": []}


def save():
    (W / "kg3_serve.json").write_text(json.dumps(OUT, indent=1))


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def call(port, path, body=None, timeout=3600):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST" if body is not None else "GET", path, json.dumps(body) if body is not None else None,
              {"Content-Type": "application/json"})
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"{}")


def chat(port, model, text, limit, out, i):
    t0 = time.perf_counter()
    status, body = call(port, "/v1/chat/completions", {"model": model, "messages": [
        {"role": "system", "content": "/no_think"}, {"role": "user", "content": text}],
        "max_tokens": limit, "temperature": 0})
    choice = (body.get("choices") or [{}])[0]
    out[i] = {"status": status, "tokens": (body.get("usage") or {}).get("completion_tokens", 0),
              "text": (choice.get("message") or {}).get("content") or "", "seconds": time.perf_counter() - t0}


def block(port, model, texts, limit):
    out = [None] * len(texts)
    t0 = time.perf_counter()
    th = [threading.Thread(target=chat, args=(port, model, t, limit, out, i)) for i, t in enumerate(texts)]
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.perf_counter() - t0
    tokens = sum(o["tokens"] for o in out)
    return {"tok_s": round(tokens / wall, 2), "tokens": tokens, "wall": round(wall, 2),
            "statuses": sorted({o["status"] for o in out})}, [o["text"] for o in out]


def main():
    if LOCAL:
        repo = Path(__file__).resolve().parents[2]
        gguf = os.environ["KG3_GGUF"]
    else:
        OUT["gpu"] = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True,
                                    capture_output=True, text=True).stdout
        sh(f"cd /tmp && curl -sSL https://github.com/Tobayko/IronMule/archive/{COMMIT}.tar.gz | tar xz "
           f"&& mv IronMule-{COMMIT} repo")
        repo = Path("/tmp/repo")
        sh(f"{sys.executable} -m pip install -q -e '{repo}[cpu]'")
        cached = next(iter(glob.glob("/kaggle/input/**/mg2_build.tar.gz", recursive=True)), None)
        unpacked = next(iter(glob.glob("/kaggle/input/**/build/bin/libllama.so", recursive=True)), None)
        src = W / "src"  # where MG2 built it: its libraries find each other by that absolute path
        src.mkdir(exist_ok=True)
        if cached:
            sh(f"cd {src} && tar xzf {cached}")
        else:
            sh(f"cp -r {Path(unpacked).parents[4]} {src}/")
        os.environ["IRONMULE_GGML_BINDINGS"] = str(src / "llama_cpp_python-0.3.16")
        os.environ["LLAMA_CPP_LIB_PATH"] = str(src / "llama_cpp_python-0.3.16/vendor/llama.cpp/build/bin")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [os.environ["LLAMA_CPP_LIB_PATH"],
                                                                    os.environ.get("LD_LIBRARY_PATH")]))
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gguf = str(W / "Qwen3-4B-Q4_K_M.gguf")
        if not Path(gguf).exists():
            sh(f"curl -sSL -o {gguf} https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf")
    os.environ["IRONMULE_KG"] = "0"  # MG2's build has no kg1 hook: the generated kernels stay off in both arms
    text = (repo / "lab/row1/facts.py").read_text()
    facts = eval(text[text.index("Q = [") + 4:text.index("]\nge.") + 1])
    state = W / "state"
    sh(f"{sys.executable} -m ironmule_cli setup --mode server --state-dir {state}")
    model = "gguf/" + Path(gguf).stem
    topics = ["a lighthouse", "the moon", "bread", "rivers", "a violin", "glaciers", "trains", "bees"]
    asks = [f"Ticket {k}: write a short story about {topics[k % len(topics)]}." for k in range(200)]
    order = [("streams", "1")] if LOCAL else [("streams", "1"), ("pool", "0"), ("pool", "0"), ("streams", "1")]
    used = 0
    for rep, (arm, flag) in enumerate(order):
        port = free_port()
        log = open(W / f"serve_{rep}_{arm}.log", "w")
        env = {**os.environ, "IRONMULE_GGML_STREAMS": flag}
        t0 = time.time()
        proc = subprocess.Popen([sys.executable, "-m", "ironmule_cli", "serve", "--gguf", gguf, "--port", str(port),
                                 "--state-dir", str(state)], env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        row = {"arm": arm, "rep": rep}
        try:
            for _ in range(1800):
                try:
                    if call(port, "/v1/models", timeout=5)[0] == 200:
                        break
                except Exception:
                    pass
                if proc.poll() is not None:
                    raise RuntimeError(f"serve exited ({proc.returncode})")
                time.sleep(2)
            row["start_s"] = round(time.time() - t0, 1)
            row["health"] = call(port, "/health")[1]
            for n in ((1, 2) if LOCAL else (1, 8, 32)):
                texts = asks[used:used + n]
                used += n
                row[f"n{n}"] = block(port, model, texts, 16 if LOCAL else 128)[0]
                print(json.dumps({"arm": arm, "rep": rep, "n": n, **row[f"n{n}"]}), flush=True)
            got = []
            for k in range(0, 8 if LOCAL else len(facts), 8):
                got += block(port, model, [q + " Answer in one short sentence." for q, _ in facts[k:k + 8]], 48)[1]
            row["facts"] = sum(any(key in (a or "").lower() for key in keys) for a, (_, keys) in zip(got, facts))
        except Exception as exc:  # noqa: BLE001 - recorded, the next arm still runs
            row["error"] = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            try:
                os.killpg(proc.pid, 15)
                proc.wait(timeout=60)
            except ProcessLookupError:  # it had exited already
                pass
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, 9)
            log.close()
            row["log_tail"] = (W / f"serve_{rep}_{arm}.log").read_text()[-3000:]
        OUT["arms"].append(row)
        save()
    OUT["finished"] = time.time()
    save()


if __name__ == "__main__":
    main()
