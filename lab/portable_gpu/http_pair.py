"""Run sequential, deadline-bounded stock llama-server and IronMule product HTTP arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from bench import ROOT, configure, save

sys.path.insert(0, str(ROOT / "lab/ag10"))
from http_transition import question, score


def endpoint(url, route, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    return urlopen(Request(url + route, data=data, headers={"Content-Type": "application/json"}), timeout=20)


def chat(url, model, prompt, limit):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "seed": 71,
            "temperature": 0, "top_p": 1, "max_tokens": limit, "stream": True}
    first, pieces, finished = None, [], False
    start = time.monotonic()
    with endpoint(url, "/v1/chat/completions", body) as response:
        for raw in response:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            if line[5:].strip() == b"[DONE]":
                finished = True
                break
            chunk = json.loads(line[5:])
            if "error" in chunk:
                raise RuntimeError(str(chunk["error"]))
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                fragment = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if fragment:
                    first = first or time.monotonic()
                    pieces.append(fragment)
    if not finished or first is None:
        raise RuntimeError("incomplete SSE or empty visible completion")
    return "".join(pieces), (first - start) * 1000


def prompt_ids(url, tokenizer, prompt, role):
    messages = [{"role": "user", "content": prompt}]
    if role == "product":
        return tokenizer.apply_chat_template(messages, enable_thinking=False)
    with endpoint(url, "/apply-template", {"messages": messages}) as response:
        rendered = json.load(response)["prompt"]
    # Pinned stock server uses add_special=true and parse_special=true for chat prompts.
    with endpoint(url, "/tokenize", {"content": rendered, "add_special": True, "parse_special": True}) as response:
        return json.load(response)["tokens"]


def measure(args, url, tok, model):
    rows = []
    # Warm one prompt outside the measurements, then use new questions in every measured block.
    prompt, _ = question("warmup", 0, 12)
    chat(url, model, prompt, args.tokens)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for repeat in range(2):
            for index, (phase, width) in enumerate((("short", 1), ("documents", 4), ("short_again", 1))):
                tasks = [question(phase, 12 * repeat + index + i, 12) for i in range(width)]
                prompts = [prompt_ids(url, tok, task[0], args.arm) for task in tasks]
                start = time.monotonic()
                futures = [pool.submit(chat, url, model, text, args.tokens) for text, _ in tasks]
                outputs = [f.result() for f in futures]
                elapsed = time.monotonic() - start
                texts = [out[0] for out in outputs]
                lengths = [len(tok.encode(text)) for text in texts]
                rows.append({"phase": phase, "repeat": repeat, "concurrency": width,
                             "prompt_ids_sha256": [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts],
                             "prompt_tokens": list(map(len, prompts)), "texts": texts, "visible_tokens": lengths,
                             "correct": [score(text, task[1])[0] for text, task in zip(texts, tasks)],
                             "ttft_ms": [out[1] for out in outputs],
                             "elapsed_s": elapsed, "visible_tokens_per_second": sum(lengths) / elapsed})
    return rows


def native_maps(pid):
    from ironmule.ggml_binary import _digest, _stat_id

    result = {}
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or int(fields[2], 16) != 0:
            continue
        path = Path(fields[5])
        if not re.fullmatch(r"lib(?:llama|ggml[^/]*)\.so(?:\.\d+)*", path.name):
            continue
        st = path.stat()
        major, minor = (int(x, 16) for x in fields[3].split(":"))
        if (st.st_dev, st.st_ino) != (os.makedev(major, minor), int(fields[4])):
            raise ValueError(f"mapped native library changed: {path.name}")
        result[path.name] = _digest(str(path), _stat_id(st))
    return result


def serving_pid(process, role):
    if role == "stock":
        return process.pid
    # The CLI holds the HTTP socket. Its isolated product worker runs the model.
    workers = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if not re.search(rf"^PPid:\s+{process.pid}$", (proc / "status").read_text(), re.MULTILINE):
                continue
            if any(arg.endswith(b"/ironmule_product/worker.py") for arg in (proc / "cmdline").read_bytes().split(b"\0")):
                workers.append(int(proc.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    if len(workers) != 1:
        raise ValueError(f"expected one live GGUF worker, observed {len(workers)}")
    return workers[0]


def run_arm(args):
    configure(args.provider)
    from ironmule.ggml_engine import file_sha256, machine_key
    from ironmule.gguf_tokenizer import GgufTokenizer, gguf_model_id
    from ironmule_devices import ggml_devices, verify_ggml

    devices = verify_ggml(args.provider)
    if args.provider != "cpu" and (len(devices) != 1 or len(ggml_devices()) != 1):
        raise ValueError("select exactly one GPU")
    tok = GgufTokenizer(str(args.gguf))
    model = gguf_model_id(str(args.gguf))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ, IRONMULE_GGML_SEQS="4", IRONMULE_GGML_SEQ_CTX=str(args.seq_ctx), IRONMULE_GGML_THREADS="2",
               IRONMULE_GGML_SHARED="0", IRONMULE_GGML_AGENT="0", IRONMULE_KG="0",
               IRONMULE_GGML_PREFIX_REUSE="1", IRONMULE_GGML_STREAMS="0", IRONMULE_PILOT_LEARN="0")
    if args.arm == "product":
        cli = [sys.executable, "-m", "ironmule_cli"]
        subprocess.run([*cli, "setup", "--mode", "server", "--state-dir", str(args.state_dir)], env=env,
                       check=True, stdout=subprocess.DEVNULL)
        cmd = [*cli, "serve", "--gguf", str(args.gguf), "--state-dir", str(args.state_dir), "--port", str(port),
               "--speculate", "off", "--no-thinking"]
    else:
        cmd = [str(args.stock), "-m", str(args.gguf), "--alias", model, "--host", "127.0.0.1", "--port", str(port),
               "-t", "2", "-tb", "2", "-np", "4", "-c", str(4 * args.seq_ctx), "-b", str(args.seq_ctx),
               "-ub", "512", "--jinja",
               "--chat-template-kwargs", '{"enable_thinking":false}', "--no-webui",
               "-ngl", "0" if args.provider == "cpu" else "999"]
    log = args.out / f"{args.arm}.log"
    with log.open("wb") as handle:
        start = time.monotonic()
        process = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=handle, stderr=handle)
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{args.arm} exited {process.returncode}. See {log}")
                try:
                    with endpoint(url, "/health") as response:
                        health = json.load(response)
                        ready = (health.get("ready") is True if args.arm == "product" else health.get("status") == "ok")
                        if ready:
                            break
                except (OSError, URLError):
                    pass
                time.sleep(.2)
            startup = time.monotonic() - start
            if args.arm == "product" and (health.get("execution") != "numeric@ggml" or
                                         health.get("ggml", {}).get("gate", {}).get("passes") is not True):
                raise ValueError("expected a qualified product GGUF worker")
            libraries = native_maps(serving_pid(process, args.arm))
            if not any(name.startswith("libllama.so") for name in libraries):
                raise ValueError("could not identify the running server's loaded libllama")
            rows = measure(args, url, tok, model)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            tok.close()
    layer_match = re.findall(rb"offloaded (\d+)/(\d+) layers to GPU", log.read_bytes())
    offload = (args.provider == "cpu" or
               (args.arm == "product" and str(health.get("device")).lower() == "gpu")
               or any(int(x) == int(y) and int(x) > 0 for x, y in layer_match))
    report = {"arm": args.arm, "provider": args.provider, "devices": devices, "machine": machine_key(),
              "model_sha256": file_sha256(str(args.gguf)), "backend_confirmed": bool(offload),
              "server_device": health.get("device"), "native_sha256": libraries, "stock_sha256": file_sha256(str(args.stock)),
              "model": model, "seq_ctx": args.seq_ctx, "startup_s": startup, "rows": rows}
    save(args.out / (args.arm + ".json"), report)


def compare(product, stock):
    if any(product[k] != stock[k] for k in ("machine", "model_sha256", "provider", "stock_sha256", "native_sha256", "seq_ctx")):
        raise ValueError("server arms have different hardware, software or model identities")
    rows = []
    for a, b in zip(product["rows"], stock["rows"], strict=True):
        if any(a[k] != b[k] for k in ("phase", "repeat", "concurrency")):
            raise ValueError("server arms ran different workloads")
        rows.append({"phase": a["phase"], "repeat": a["repeat"],
                     "prompt_tokens_identical": a["prompt_ids_sha256"] == b["prompt_ids_sha256"],
                     "identical_texts": a["texts"] == b["texts"], "all_correct": all(a["correct"]) and all(b["correct"]),
                     "product_over_stock_tps": a["visible_tokens_per_second"] / b["visible_tokens_per_second"],
                     "product_over_stock_median_ttft": statistics.median(a["ttft_ms"]) / statistics.median(b["ttft_ms"])})
    return {"schema": "ironmule.http-stock-pair.v1", "rows": rows,
            "paired_quality_passed": all(r["all_correct"] and r["identical_texts"] and r["prompt_tokens_identical"]
                                         for r in rows), "backend_confirmed": all(a["backend_confirmed"] for a in (product, stock))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--stock", type=Path, required=True)
    p.add_argument("--provider", choices=("cpu", "cuda", "rocm"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--state-dir", type=Path, required=True, help="dedicated local product state (quality gate cache)")
    p.add_argument("--first", choices=("stock", "product"), default="stock")
    p.add_argument("--budget-s", type=int, default=300, help="total wall time for both arms")
    p.add_argument("--seq-ctx", type=int, choices=(512, 1024, 2048), default=2048)
    p.add_argument("--tokens", default=16, type=int)
    p.add_argument("--arm", choices=("stock", "product"), help=argparse.SUPPRESS)
    args = p.parse_args()
    if not (args.gguf.is_file() and args.stock.is_file() and 1 <= args.tokens <= 128 and args.budget_s > 0):
        p.error("provide existing GGUF and stock executable, 1..128 output tokens, and a positive deadline")
    for key in ("out", "gguf", "state_dir", "stock"):
        setattr(args, key, getattr(args, key).resolve())
    if args.arm:
        run_arm(args)
        return
    args.out.mkdir(exist_ok=False, parents=True)
    deadline = time.monotonic() + args.budget_s
    for label in (args.first, "product" if args.first == "stock" else "stock"):
        save(args.out / "status.json", {"status": "running", "arm": label})
        proc = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--arm", label], start_new_session=True)
        try:
            code = proc.wait(timeout=max(.001, deadline - time.monotonic()))
            if code:
                raise RuntimeError(f"{label} arm exited {code}")
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            save(args.out / "status.json", {"status": "incomplete", "arm": label, "error": type(exc).__name__})
            raise
        finally:
            # Includes gate and worker subprocesses even when the supervising worker times out.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    arms = [json.loads((args.out / f"{arm}.json").read_text()) for arm in ("product", "stock")]
    save(args.out / "comparison.json", compare(*arms))
    save(args.out / "status.json", {"status": "complete"})


if __name__ == "__main__":
    main()
