"""Kaggle harness for BACKLOG U1, HW1 and HW2: unquantized checkpoints on a "GPU T4 x2" session.

Runs as one Kaggle script kernel, or locally with `--smoke` (CPU, tiny sizes) to check the
harness itself; a smoke report is never evidence. It writes one JSON report with token counts,
times, identities and SHA-256 digests of answers (to compare answers without keeping them), but
no prompt or answer text.

  env  hardware and software inventory
  u1   per unquantized checkpoint, each plan (stock bfloat16, float16, float32 and U2's mixed16) in
       its own process: the numeric-plan gate's perplexity text (`numeric_choice.nll_per_chunk`) and served seconds per
       128 tokens on a short and a long prompt (`numeric_choice._speed`), then the gate's bootstrap
       ratio against stock; and the exact knobs against stock, paired (`ab.run`, token identity)
  hw1  `ironmule serve --engine tuned` on one card against `--gpus all`, in the order one, two, two,
       one, at 1, 4 and 8 concurrent clients: completion tokens per second, first-content p50/p95,
       and every answer's digest, which must agree for the same prompt in every arm and level
  hw2  whether MLX's NCCL backend comes up across the two cards (`mlx.launch`, an all_sum check)

Kill criteria are BACKLOG's; this tool only measures. To run it on Kaggle (Kaggle CLI, with
`KAGGLE_API_TOKEN` or `KAGGLE_USERNAME`/`KAGGLE_KEY` set), from a pushed commit:

    python tools/kaggle_whole_machine.py --write-kernel kernel --owner <user> --commit <sha>
    kaggle kernels push -p kernel --accelerator NvidiaTeslaT4
    kaggle kernels status <user>/ironmule-whole-machine
    kaggle kernels output <user>/ironmule-whole-machine -p results
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import queue
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

U1_MODELS = ("mlx-community/Qwen3-1.7B-bf16", "mlx-community/gemma-3-1b-it-bf16",
             "mlx-community/Llama-3.2-1B-Instruct-bf16", "mlx-community/Qwen3-4B-bf16")
HW1_MODEL = "mlx-community/Qwen3-1.7B-bf16"
# Qwen 3 has mlx-lm's `shard`; the 8B (about 16 GB in bfloat16) fits two T4s and not one.
HW2_MODELS = ("mlx-community/Qwen3-1.7B-bf16", "mlx-community/Qwen3-4B-bf16", "mlx-community/Qwen3-8B-bf16")
SMOKE_MODEL = "Qwen/Qwen3-0.6B"
# `mixed16` (BACKLOG U2) exists only in this tool: float16 matmuls with the checkpoint's activations.
PLANS = (None, "float16", "float32", "mixed16")
EXACT_KNOBS = {"compiled_fixed_cache": True, "head_skip_prefill": True, "readback_every": 4}
# Generic questions; the report keeps only their index, never the text or the answers.
PROMPTS = ("Explain how a refrigerator keeps food cold.", "What causes the seasons on Earth?",
           "Describe how a bicycle gear system works.", "Why is the sky blue during the day?",
           "Summarize the water cycle in a few sentences.", "How does a compiler differ from an interpreter?",
           "What is the difference between weather and climate?", "Explain what a hash table is.",
           "How do vaccines train the immune system?", "Why do metals conduct electricity?",
           "Describe the life cycle of a star like the Sun.", "What is photosynthesis?",
           "How does a binary search work?", "Explain the greenhouse effect.",
           "What does a CPU cache do?", "How do airplanes stay in the air?")


def _run(command: list[str], timeout: float = 120, **kwargs: Any) -> dict[str, Any]:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"returncode": None, "error": f"{type(exc).__name__}: {exc}"[:400]}
    return {"returncode": done.returncode, "stdout": done.stdout[-4000:], "stderr": done.stderr[-2000:]}


def environment() -> dict[str, Any]:
    from importlib import metadata

    versions = {}
    for name in ("ironmule", "mlx", "mlx-lm", "mlx-cuda", "nvidia-nccl-cu12"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    gpus = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,pcie.link.gen.max",
                 "--format=csv,noheader"])
    topology = _run(["nvidia-smi", "topo", "-m"])
    memory = Path("/proc/meminfo").read_text().splitlines()[:3] if Path("/proc/meminfo").exists() else []
    return {"python": platform.python_version(), "machine": platform.machine(), "cpus": os.cpu_count(),
            "meminfo": memory, "versions": versions, "gpus": gpus.get("stdout", "").strip().splitlines(),
            "topology": topology.get("stdout", "")}


def _download(model: str) -> str:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                         "*.jinja", "tokenizer*"]))
    return path.name


_CHILD = r"""
import json, sys, time
import mlx.core as mx
import mlx.nn as nn
from ironmule.runtime import BASELINE
from ironmule.tune import load_engine, _close_engine
from ironmule import numeric_choice


class Mixed16Linear(nn.Linear):
    # U2: the matmul on float16 tensor cores; activations, norms and the residual keep the checkpoint's type.
    def __call__(self, x):
        y = x.astype(mx.float16) @ self["weight"].T
        return (y + self["bias"] if "bias" in self else y).astype(x.dtype)


class Mixed16Embedding(nn.Embedding):
    # The lookup keeps the checkpoint's weights; a tied head multiplies a float16 copy.
    def as_linear(self, x):
        return (x.astype(mx.float16) @ self._float16_weight.T).astype(x.dtype)


def mixed16(model):
    swapped = 0
    for _, module in model.named_modules():
        if type(module) is nn.Linear:
            module.weight = module.weight.astype(mx.float16)
            if "bias" in module:
                module.bias = module.bias.astype(mx.float16)
            module.__class__ = Mixed16Linear
            swapped += 1
        elif type(module) is nn.Embedding:
            module._float16_weight = module.weight.astype(mx.float16)
            mx.eval(module._float16_weight)
            module.__class__ = Mixed16Embedding
    mx.eval(model.parameters())
    return swapped


model, plan, chunks = sys.argv[1], sys.argv[2] or None, int(sys.argv[3])
started = time.perf_counter()
try:
    engine, tokenizer = load_engine(model, BASELINE, compute_dtype=None if plan == "mixed16" else plan)
    if plan == "mixed16":
        swapped = mixed16(engine.model)
except Exception as exc:
    print("@@U1 " + json.dumps({"plan": plan, "refused": f"{type(exc).__name__}: {exc}"[:300]}), flush=True)
    raise SystemExit(0)
load_s = time.perf_counter() - started
try:
    nll = numeric_choice.nll_per_chunk(engine, numeric_choice._text_ids(tokenizer), chunks=chunks)
    speed = numeric_choice._speed(engine, tokenizer)
    print("@@U1 " + json.dumps({"plan": plan, "load_s": load_s, "nll": nll, "speed_s": speed,
                                "nonfinite_chunks": sum(1 for value in nll if value != value or value == float("inf")),
                                "swapped": swapped if plan == "mixed16" else None,
                                "peak_bytes": mx.get_peak_memory()}), flush=True)
finally:
    _close_engine(engine)
"""


def _plan_in_child(model: str, plan: str | None, chunks: int) -> dict[str, Any]:
    done = _run([sys.executable, "-c", _CHILD, model, plan or "", str(chunks)], timeout=3600)
    lines = [line for line in done.get("stdout", "").splitlines() if line.startswith("@@U1 ")]
    if not lines:
        return {"plan": plan, "failed": done.get("error") or done.get("stderr", "")[-600:],
                "returncode": done.get("returncode")}
    return json.loads(lines[-1][len("@@U1 "):])


def u1(models: tuple[str, ...], *, chunks: int, plans: tuple[str | None, ...], pairs: int,
       max_tokens: int) -> list[dict[str, Any]]:
    from ironmule import ab
    from ironmule.numeric_choice import ratio_interval
    from ironmule.runtime import BASELINE, Knobs

    rows = []
    for model in models:
        row: dict[str, Any] = {"model": model, "revision": _download(model), "plans": {}}
        for plan in plans:
            row["plans"][plan or "stock"] = _plan_in_child(model, plan, chunks)
        stock = row["plans"]["stock"]
        for name, result in row["plans"].items():
            if name == "stock" or "nll" not in result or "nll" not in stock:
                continue
            result["gate"] = ratio_interval(stock["nll"], result["nll"])
            result["speed_ratio"] = {key: result["speed_s"][key] / stock["speed_s"][key] for key in stock["speed_s"]}
        for result in row["plans"].values():
            result.pop("nll", None)  # per-chunk likelihoods are summarised by the gate
        try:
            paired = ab.run({"baseline": BASELINE, "candidate": Knobs(**EXACT_KNOBS)}, processes=pairs,
                            repeats=3, warmup=2, max_tokens=max_tokens, model=model)
            row["exact_knobs"] = {"knobs": EXACT_KNOBS, "processes": paired["processes"],
                                  "token_identity": paired["token_identity"],
                                  "token_count_identity": paired["token_count_identity"],
                                  "deterministic": paired["deterministic"],
                                  "ratios": paired["ratios"]["candidate/baseline"]}
        except Exception as exc:  # noqa: BLE001 - recorded, the other models still run
            row["exact_knobs"] = {"failed": f"{type(exc).__name__}: {exc}"[:400]}
        rows.append(row)
        print(json.dumps({"u1": model, "done": True}), flush=True)
    return rows


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _ironmule(*arguments: str, state: Path, timeout: float = 600) -> dict[str, Any]:
    code = "import sys, ironmule_cli; sys.exit(ironmule_cli.main(sys.argv[1:]))"
    return _run([sys.executable, "-c", code, *arguments, "--state-dir", str(state)], timeout=timeout)


def _chat(port: int, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "stream": True})
    started = time.perf_counter()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    try:
        connection.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            return {"status": response.status, "total_s": time.perf_counter() - started}
        first, text, usage = None, [], None
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: "):])
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                first = first if first is not None else time.perf_counter() - started
                text.append(content)
            usage = chunk.get("usage") or usage
        answer = "".join(text)
        return {"status": 200, "first_s": first, "total_s": time.perf_counter() - started,
                "completion_tokens": (usage or {}).get("completion_tokens"),
                "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest()}
    finally:
        connection.close()


def _level(port: int, model: str, clients: int, requests: int, max_tokens: int) -> dict[str, Any]:
    work: queue.Queue = queue.Queue()
    for index in range(requests):
        work.put(index % len(PROMPTS))
    results: list[dict[str, Any]] = []
    lock = threading.Lock()

    def client() -> None:
        while True:
            try:
                index = work.get_nowait()
            except queue.Empty:
                return
            result = _chat(port, model, PROMPTS[index], max_tokens)
            with lock:
                results.append({"prompt": index, **result})

    started = time.perf_counter()
    threads = [threading.Thread(target=client) for _ in range(clients)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    ok = [r for r in results if r.get("status") == 200]
    firsts = sorted(r["first_s"] for r in ok if r.get("first_s") is not None)
    tokens = sum(r.get("completion_tokens") or 0 for r in ok)

    def quantile(values: list[float], q: float) -> float | None:
        return values[min(len(values) - 1, int(q * len(values)))] if values else None

    return {"clients": clients, "requests": requests, "ok": len(ok), "wall_s": wall,
            "tokens_per_s": tokens / wall if wall > 0 else None, "completion_tokens": tokens,
            "first_p50_s": quantile(firsts, 0.5), "first_p95_s": quantile(firsts, 0.95),
            "answers": {str(r["prompt"]): r.get("answer_sha256") for r in ok}}


def _serve(model: str, state: Path, gpus: str | None, levels: tuple[int, ...], requests_per_client: int,
           max_tokens: int) -> dict[str, Any]:
    port = _free_port()
    code = "import sys, ironmule_cli; sys.exit(ironmule_cli.main(sys.argv[1:]))"
    command = [sys.executable, "-c", code, "serve", "--model", model, "--engine", "tuned", "--port", str(port),
               "--state-dir", str(state)] + (["--gpus", gpus] if gpus else [])
    server = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 900
        health = None
        while time.monotonic() < deadline and server.poll() is None:
            try:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                connection.request("GET", "/health")
                health = json.loads(connection.getresponse().read())
                connection.close()
                if health.get("ready"):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        if not health or not health.get("ready"):
            return {"gpus": gpus, "failed": "server not ready", "returncode": server.poll()}
        _chat(port, model, PROMPTS[0], max_tokens)  # warm-up, not recorded
        out = {"gpus": gpus, "devices": [d.get("cuda_device") for d in health.get("devices", [])] or None,
               "levels": [_level(port, model, clients, clients * requests_per_client, max_tokens)
                          for clients in levels]}
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        connection.request("GET", "/health")
        final = json.loads(connection.getresponse().read())
        connection.close()
        out["per_device_completed"] = [d.get("completed_requests") for d in final.get("devices", [])] or None
        return out
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def hw1(model: str, *, arms: tuple[str | None, ...], levels: tuple[int, ...], requests_per_client: int,
        max_tokens: int, work: Path) -> dict[str, Any]:
    revision = _download(model)
    state = work / "hw1-state"
    setup = _ironmule("setup", "--mode", "server", state=state)
    added = _ironmule("models", "add", model, state=state)
    if added.get("returncode") != 0:
        return {"model": model, "revision": revision, "failed": "models add", "detail": added, "setup": setup}
    runs = [_serve(model, state, gpus, levels, requests_per_client, max_tokens) for gpus in arms]
    digests: dict[str, set[str]] = {}
    for run in runs:
        for level in run.get("levels", []):
            for prompt, digest in level["answers"].items():
                digests.setdefault(prompt, set()).add(digest)
    for run in runs:
        for level in run.get("levels", []):
            level.pop("answers")
    return {"model": model, "revision": revision, "order": [run["gpus"] or "one" for run in runs], "runs": runs,
            "answers_identical": bool(digests) and all(len(found) == 1 for found in digests.values()),
            "prompts_compared": len(digests)}


_PROBE = r"""
import json, os
import mlx.core as mx
out = {"rank_env": {k: os.environ.get(k) for k in ("MLX_RANK", "CUDA_VISIBLE_DEVICES", "OMPI_COMM_WORLD_RANK")}}
try:
    group = mx.distributed.init(backend=os.environ.get("HW2_BACKEND", "nccl"), strict=True)
    value = mx.distributed.all_sum(mx.ones(4) * (group.rank() + 1))
    mx.eval(value)
    out.update(rank=group.rank(), size=group.size(), all_sum=value.tolist())
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"[:400]
print("@@HW2 " + json.dumps(out), flush=True)
"""


_DECODE = r"""
import hashlib, json, os, sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
model_id, tokens = sys.argv[1], int(sys.argv[2])
group = (mx.distributed.init(backend=os.environ.get("HW2_BACKEND", "nccl"), strict=True)
         if os.environ.get("MLX_RANK") is not None else None)
out = {"rank": group.rank() if group else 0, "size": group.size() if group else 1}
try:
    if group is not None and group.size() > 1:
        # Lazy, so each card only ever holds its shard: an 8B in bfloat16 does not fit one T4 whole.
        try:
            from mlx_lm.utils import sharded_load
            model, tokenizer = sharded_load(model_id, tensor_group=group)
            out["loader"] = "sharded_load"
        except Exception as exc:
            out["sharded_load_error"] = f"{type(exc).__name__}: {exc}"[:300]
            model, tokenizer = load(model_id, lazy=True)
            model.shard(group)
            out["loader"] = "lazy_load_then_shard"
    else:
        model, tokenizer = load(model_id)
    mx.eval(model.parameters())
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Explain how a refrigerator keeps food cold."}],
                                           add_generation_prompt=True)
    runs = []
    for index in range(4):  # the first run compiles and is not recorded
        ids, last = [], None
        started = time.perf_counter()
        for last in stream_generate(model, tokenizer, prompt, max_tokens=tokens, sampler=make_sampler(temp=0.0)):
            ids.append(last.token)
        if index:
            runs.append({"generation_tps": last.generation_tps, "prompt_tps": last.prompt_tps, "tokens": len(ids),
                         "total_s": time.perf_counter() - started,
                         "digest": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
    out.update(runs=runs, peak_bytes=mx.get_peak_memory())
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"[:400]
print("@@TP " + json.dumps(out), flush=True)
"""


def _launcher() -> str | None:
    launcher = shutil.which("mlx.launch") or str(Path(sys.executable).with_name("mlx.launch"))
    return launcher if Path(launcher).exists() else None


def _decode(work: Path, model: str, tokens: int, backend: str | None) -> list[dict[str, Any]]:
    """One process on one card (`backend` None), or two ranks sharded across two cards."""
    script = work / "hw2_decode.py"
    script.write_text(_DECODE)
    if backend is None:
        done = _run([sys.executable, str(script), model, str(tokens)], timeout=1800,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("HW2_SINGLE_DEVICE", "0")})
    else:
        done = _run([_launcher(), "--backend", backend, "-n", "2", "--env", f"HW2_BACKEND={backend}", "--",
                     sys.executable, str(script), model, str(tokens)], timeout=1800)
    ranks = [json.loads(line[len("@@TP "):]) for line in done.get("stdout", "").splitlines()
             if line.startswith("@@TP ")]
    return ranks or [{"failed": done.get("error") or done.get("stderr", "")[-800:], "returncode": done.get("returncode")}]


def hw2(work: Path, backend: str = "nccl", models: tuple[str, ...] = (), tokens: int = 128) -> dict[str, Any]:
    install = (_run([sys.executable, "-m", "pip", "install", "-q", "nvidia-nccl-cu12"], timeout=900)
               if backend == "nccl" else {})
    probe = work / "hw2_probe.py"
    probe.write_text(_PROBE)
    launcher = _launcher()
    if launcher is None:
        return {"install": install.get("returncode"), "failed": "mlx.launch not found"}
    done = _run([launcher, "--backend", backend, "-n", "2", "--env", f"HW2_BACKEND={backend}", "--",
                 sys.executable, str(probe)],
                timeout=600)
    ranks = [json.loads(line[len("@@HW2 "):]) for line in done.get("stdout", "").splitlines()
             if line.startswith("@@HW2 ")]
    out: dict[str, Any] = {"install": install.get("returncode"), "launch_returncode": done.get("returncode"),
                           "ranks": ranks, "stderr_tail": done.get("stderr", "")[-1200:],
                           "works": len(ranks) == 2 and all(rank.get("all_sum") == [3.0] * 4 for rank in ranks),
                           "decode": []}
    if not out["works"]:
        return out
    for model in models:
        revision = _download(model)
        single = _decode(work, model, tokens, None)
        sharded = _decode(work, model, tokens, backend)
        row: dict[str, Any] = {"model": model, "revision": revision, "single": single, "sharded": sharded}
        try:
            one = statistics.median(run["generation_tps"] for run in single[0]["runs"])
            two = statistics.median(run["generation_tps"] for run in next(r for r in sharded if r.get("rank") == 0)["runs"])
            row["decode_speedup"] = two / one
            row["same_tokens"] = ({run["digest"] for run in single[0]["runs"]}
                                  == {run["digest"] for r in sharded for run in r["runs"]})
        except (KeyError, StopIteration, TypeError, ValueError, statistics.StatisticsError):
            row["decode_speedup"] = None  # one arm failed (e.g. the model does not fit one card)
        out["decode"].append(row)
    return out


REPOSITORY = "https://github.com/Tobayko/IronMule"
_BOOTSTRAP = '''"""IronMule U1/HW1/HW2 on this Kaggle session; written by tools/kaggle_whole_machine.py."""
from __future__ import annotations
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "ironmule[cuda] @ git+{repository}@{commit}"], check=True)
'''


def write_kernel(directory: Path, owner: str, commit: str, phases: str) -> Path:
    """A private script kernel that installs IronMule at `commit` and runs this file's phases.

    Push it with `kaggle kernels push -p DIR --accelerator NvidiaTeslaT4` (Kaggle's T4 session has
    two cards); the report lands in the kernel's output as `whole_machine_report.json`."""
    if not commit or any(c not in "0123456789abcdef" for c in commit) or len(commit) != 40:
        raise ValueError("commit must be a full 40-character lowercase SHA, so the kernel runs exactly it")
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_text()
    entry = (f"\nsys.exit(main(['--out', '/kaggle/working/whole_machine_report.json', "
             f"'--phases', {phases!r}]))\n")
    body = (source.replace('\nif __name__ == "__main__":\n    sys.exit(main())\n', "")
            .replace("from __future__ import annotations\n", "", 1))
    (directory / "run.py").write_text(_BOOTSTRAP.format(repository=REPOSITORY, commit=commit) + body + entry)
    metadata = {"id": f"{owner}/ironmule-whole-machine", "title": "ironmule-whole-machine", "code_file": "run.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1) + "\n")
    return directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--phases", default="env,u1,hw1,hw2")
    parser.add_argument("--smoke", action="store_true", help="CPU, tiny sizes: checks this tool, measures nothing")
    parser.add_argument("--write-kernel", type=Path, metavar="DIR",
                        help="write a Kaggle kernel folder instead of running (needs --owner and --commit)")
    parser.add_argument("--owner", help="Kaggle user name that will own the kernel")
    parser.add_argument("--commit", help="the pushed IronMule commit the kernel installs")
    args = parser.parse_args(argv)
    if args.write_kernel is not None:
        if not args.owner or not args.commit:
            parser.error("--write-kernel needs --owner and --commit")
        print(json.dumps({"kernel": str(write_kernel(args.write_kernel, args.owner, args.commit, args.phases))}))
        return 0
    if args.out is None:
        parser.error("--out is required to run")
    phases = set(args.phases.split(","))
    work = Path(tempfile.mkdtemp(prefix="ironmule-whole-machine-"))
    report: dict[str, Any] = {"schema": "ironmule.kaggle_whole_machine.v1", "smoke": args.smoke,
                              "started_unix": time.time(), "phases": sorted(phases)}

    def save() -> None:
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")

    if "env" in phases:
        report["env"] = environment()
        save()
    if "u1" in phases:
        if args.smoke:
            report["u1"] = u1((SMOKE_MODEL,), chunks=2, plans=(None, "float32"), pairs=2, max_tokens=8)
        else:
            report["u1"] = u1(U1_MODELS, chunks=16, plans=PLANS, pairs=4, max_tokens=128)
        save()
    if "hw1" in phases:
        if args.smoke:
            report["hw1"] = hw1(SMOKE_MODEL, arms=(None,), levels=(1, 2), requests_per_client=1, max_tokens=8,
                                work=work)
        else:
            report["hw1"] = hw1(HW1_MODEL, arms=(None, "all", "all", None), levels=(1, 4, 8),
                                requests_per_client=4, max_tokens=128, work=work)
        save()
    if "hw2" in phases:
        if args.smoke:
            report["hw2"] = hw2(work, backend="ring", models=(SMOKE_MODEL,), tokens=8)
        else:
            report["hw2"] = hw2(work, models=HW2_MODELS, tokens=128)
        save()
    report["finished_unix"] = time.time()
    save()
    print(json.dumps({"report": str(args.out)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
