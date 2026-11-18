"""Internal worker for bench.py, run in a process group bounded by the stage deadline."""
import ast
import concurrent.futures
import ctypes
import http.client
import json
import math
import os
import random
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def counter():
    try:
        symbol = ctypes.CDLL(None).ironmule_kg5_launches
    except AttributeError:
        symbol = ctypes.CDLL(str(Path(os.environ["LLAMA_CPP_LIB_PATH"]) / "libggml-cuda.so")).ironmule_kg5_launches
    symbol.argtypes, symbol.restype = [], ctypes.c_uint64
    return symbol


def steps(gguf, out, rep):
    import numpy as np

    # As in KG4, the isolated ggml timing needs just these Python modules. Importing
    # IronMule's top-level MLX runtime here would require the full HTTP-stage dependencies.
    ironmule = ModuleType("ironmule")
    ironmule.__path__ = [str(ROOT / "ironmule")]
    sys.modules["ironmule"] = ironmule
    from ironmule.ggml_engine import GgmlEngine

    hits = counter()
    rng = random.Random(570 + rep)
    engine = GgmlEngine(str(gguf), seqs=32, seq_ctx=1280, flash_attn=True, shared=False, speculate=False)
    try:
        if engine.gpu_layers == 0:
            raise RuntimeError("engine reports CPU execution")
        engine._clear()
        for seq in range(32):
            for start in (0, 512):
                count = min(512, 1000 - start)
                engine._decode([(rng.randrange(1000, 20000), start + i, seq, i == count - 1)
                                for i in range(count)])
        times = []
        for pos in range(1000, 1015):
            rows = [(rng.randrange(1000, 20000), pos, s, True) for s in range(32)]
            if pos == 1003:
                before = hits()
            started = time.perf_counter()
            engine._decode(rows)  # copies the logits to host, synchronizing the device in both arms
            if pos >= 1003:
                times.append(1e3 * (time.perf_counter() - started))
        launches = hits() - before
        logits = [np.ctypeslib.as_array(engine.L.llama_get_logits_ith(engine.ctx, i),
                                        shape=(engine.n_vocab,)).copy() for i in (0, 7, 16, 31)]
        path = out.with_suffix(".npy")
        np.save(path, np.stack(logits))
        return {"hits": launches, "logits": str(path), "step_ms": times}
    finally:
        engine.close()


def messages(text):
    return [{"role": "system", "content": "/no_think"}, {"role": "user", "content": text}]


def long_prompt(tokenizer, prefix, question):
    # Give every client enough history for the intended allocated KV size, while leaving
    # room for the answer. Measure the actual GGUF chat template, not whitespace counts.
    def render(n):
        return f"{prefix}. Background notes:" + " neutral" * n + "\nQuestion: " + question

    base = len(tokenizer.apply_chat_template(messages(render(0))))
    prompt = render(max(0, 972 - base))
    size = len(tokenizer.apply_chat_template(messages(prompt)))
    if not 960 <= size <= 990:
        raise ValueError(f"expected a prompt near 972 tokens, got {size}")
    return prompt


def stream(port, model, prompt, limit):
    start = time.perf_counter()
    first = count = None
    pieces = []
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps({"model": model, "messages": messages(prompt),
            "temperature": 0, "max_tokens": limit, "stream": True}), {"Content-Type": "application/json"})
        res = conn.getresponse()
        if res.status != 200:
            raise RuntimeError(f"chat HTTP {res.status}: {res.read(300)!r}")
        for raw in res:
            data = raw.strip()
            if data == b"data: [DONE]":
                break
            if data.startswith(b"data:"):
                chunk = json.loads(data[5:])
                text = chunk["choices"][0]["delta"].get("content", "")
                if text:
                    if first is None:
                        first = time.perf_counter() - start
                    pieces.append(text)
                if "usage" in chunk:
                    count = chunk["usage"]["completion_tokens"]
        else:
            raise RuntimeError("incomplete chat stream")
        if not count or first is None:
            raise RuntimeError("empty completion")
        return first, count, "".join(pieces)
    finally:
        conn.close()


def questions():
    # The existing fixture also runs a model when imported. Only read its literal question list.
    module = ast.parse((ROOT / "lab/row1/facts.py").read_text())
    for row in module.body:
        if isinstance(row, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "Q" for t in row.targets):
            return ast.literal_eval(row.value)
    raise RuntimeError("fact questions missing")


def instrumented_server_environment(tmp):
    # The actual serving engine lives in an isolated child. Wrap its entry point in the
    # benchmark's temporary package to allow a launch-counter snapshot between HTTP blocks.
    pkg = tmp / "ironmule_product"
    pkg.mkdir()
    for source in (ROOT / "ironmule_product").iterdir():
        if source.name not in ("worker.py", "__pycache__"):
            (pkg / source.name).symlink_to(source)
    (pkg / "worker.py").write_text(
        "import os, runpy, signal, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from worker import counter\n"
        "hits = counter()\n"
        f"record = Path({str(tmp / 'launches')!r})\n"
        "def snapshot(*_):\n"
        "    stage = record.with_suffix('.tmp')\n"
        "    stage.write_text(str(hits()))\n"
        "    stage.replace(record)\n"
        "signal.signal(signal.SIGUSR1, snapshot)\n"
        f"Path({str(tmp / 'pid')!r}).write_text(str(os.getpid()))\n"
        f"runpy.run_path({str(ROOT / 'ironmule_product/worker.py')!r}, run_name='__main__')\n"
    )
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp), str(ROOT)])}


def serving_count(folder):
    record = folder / "launches"
    record.unlink(missing_ok=True)
    os.kill(int((folder / "pid").read_text()), signal.SIGUSR1)
    for _ in range(120):
        if record.exists():
            return int(record.read_text())
        time.sleep(0.05)
    raise RuntimeError("serving worker did not return a launch count")


def http_arm(gguf, out, rep):
    tmp = out.with_suffix(".server")
    tmp.mkdir()
    # State is separate per arm and reused across repetitions of that arm only. Gates always
    # run via the product, including the independent stock-reference evaluation.
    state = out.parent / f"state-{os.environ['IRONMULE_KG5']}"
    if not state.exists():
        subprocess.run([sys.executable, "-m", "ironmule_cli", "setup", "--mode", "server", "--state-dir", state],
                       check=True)
        if cache := os.environ.get("KG5_BENCH_GATE_CACHE"):
            gate = state / "ggml-gate"
            gate.mkdir(exist_ok=True)
            for cached in Path(cache).glob("*.json"):
                shutil.copyfile(cached, gate / cached.name)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = instrumented_server_environment(tmp)
    model = f"gguf/{gguf.stem}"
    proc = subprocess.Popen([sys.executable, "-m", "ironmule_cli", "serve", "--gguf", gguf, "--port", str(port),
                             "--state-dir", state], cwd=tmp, env=env)
    try:
        while proc.poll() is None:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            try:
                conn.request("GET", "/v1/models")
                if conn.getresponse().status == 200:
                    break
            except OSError:
                pass
            finally:
                conn.close()
            time.sleep(0.2)
        else:
            raise RuntimeError(f"server exited {proc.returncode}")
        stream(port, model, "Name a colour.", 8)
        from ironmule.gguf_tokenizer import GgufTokenizer

        tokenizer = GgufTokenizer(str(gguf))
        try:
            asks = [long_prompt(tokenizer, f"Story {rep}-{i}", f"Write a new story about train number {i}.")
                    for i in range(32)]
            cases = questions()
            # The last fourteen questions repeat the start of the suite to keep all 32
            # streams active in both batches. A repeated question must pass each time.
            cases += cases[:64 - len(cases)]
            prompts = [long_prompt(tokenizer, f"Document {rep}-{i}", q + " Answer in one short sentence.")
                       for i, (q, _) in enumerate(cases)]
        finally:
            tokenizer.close()

        before = serving_count(tmp)
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            start = time.perf_counter()
            jobs = [pool.submit(stream, port, model, p, 48) for p in asks]
            rows = [job.result() for job in jobs]
            wall = time.perf_counter() - start
            after = serving_count(tmp)
            launches = after - before
            facts, fact_hits = {}, []
            for first in (0, 32):
                jobs = [pool.submit(stream, port, model, p, 48) for p in prompts[first:first + 32]]
                for (q, expected), job in zip(cases[first:first + 32], jobs):
                    answer = job.result()[2].lower()
                    facts[q] = facts.get(q, True) and any(key in answer for key in expected)
                seen = serving_count(tmp)
                fact_hits.append(seen - after)
                after = seen
        ttft = sorted(row[0] for row in rows)
        return {"tok_s": sum(row[1] for row in rows) / wall, "ttft_median_s": statistics.median(ttft),
                "ttft_p95_s": ttft[math.ceil(0.95 * len(ttft)) - 1], "hits": launches, "facts": facts,
                "fact_hits": fact_hits}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    stage, gguf, out, rep = sys.argv[1:]
    result = (steps if stage == "steps" else http_arm)(Path(gguf), Path(out), int(rep))
    Path(out).write_text(json.dumps(result))
