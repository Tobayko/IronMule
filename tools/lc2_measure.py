"""LC2: IronMule against `llama-server` and vLLM on one Kaggle T4, Qwen3-1.7B, over HTTP (BACKLOG PL1, LC2).

Arms, one server at a time on GPU 0: I, `ironmule serve --agent on` on `mlx-community/Qwen3-1.7B-bf16`; L,
`llama-server` built here from llama.cpp 4227c9b (vendored in `llama-cpp-python` 0.3.16) with CUDA, Qwen's Q8_0
GGUF; Lf, the same with flash attention; V, vLLM 0.10.1.1 in its own virtual environment, float16. Order I L Lf V I.
Load: CPU6's 8 prompts at once, 64 tokens, 5 blocks; quiet: 16 Dolly prompts from index 14,200, 128 tokens. The
report is written after every cell; `--analyse REPORT` applies the reading fixed in the entry before the data.

    python tools/lc2_measure.py --write-kernel kernel/ --owner OWNER --commit FULL_SHA
    python tools/lc2_measure.py --out lc2.json                 # on the T4 session (the kernel runs this)
    python tools/lc2_measure.py --local-smoke --server PATH/llama-server --gguf M.gguf --out smoke.json
    python -I tools/lc2_measure.py --analyse lc2.json
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/Tobayko/IronMule"
MLX_MODEL = "mlx-community/Qwen3-1.7B-bf16"
HF_MODEL = "Qwen/Qwen3-1.7B"
GGUF_REPO, GGUF_FILE = "Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf"
LLAMA_CPP_PYTHON = "0.3.16"  # its sdist vendors llama.cpp 4227c9b, LC1's build
VLLM = "0.10.1.1"
ORDER = ("I", "L", "Lf", "V", "I")
LOAD_BLOCKS, LOAD_LIMIT = 5, 64
QUIET_FIRST, QUIET_SIZE, QUIET_LIMIT = 14_200, 16, 128
WARM_FIRST = 14_280  # warm-up prompts, which no block uses
HOLDS, RESAMPLES = 0.95, 2000
HARD_LIMIT_S = 3 * 3600.0
CPU6_PROMPTS = [f"List {3 + i % 3} {topic}, one short line each." for i, topic in enumerate(
    ("animals that live in the sea", "uses for a paperclip", "reasons the sky looks blue", "steps to boil an egg",
     "facts about the moon", "ways to save water at home", "tips for learning a language", "parts of a bicycle"))]
DOLLY_URL = ("https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/"
             "databricks-dolly-15k.jsonl")


# ---------------------------------------------------------------- prompts and the client


def dolly_prompts(work: Path) -> tuple[list[str], str]:
    """The project's shuffled Dolly pool (`training/rl2_headroom.py`'s), so indices mean what they mean elsewhere."""
    sys.path[:0] = [str(ROOT / "training")]
    import rl2_headroom

    return rl2_headroom.dolly_prompts(work)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def ask(port: int, model: str, text: str, limit: int, offered: float, usage_option: bool) -> dict[str, Any]:
    """One streamed chat completion: the first token's and the end's time since `offered`, tokens, text."""
    payload: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": limit,
                               "temperature": 0, "stream": True}
    if usage_option:  # IronMule refuses unknown fields and sends its usage anyway
        payload["stream_options"] = {"include_usage": True}
    first, pieces, usage, error = None, [], None, None
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
        conn.request("POST", "/v1/chat/completions", json.dumps(payload), {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read()[:200]!r}")
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            for choice in chunk.get("choices") or []:
                content = (choice.get("delta") or {}).get("content") or (choice.get("delta") or {}).get(
                    "reasoning_content")
                if content:
                    first = first if first is not None else time.perf_counter() - offered
                    pieces.append(content)
            if chunk.get("usage"):
                usage = chunk["usage"]
        conn.close()
    except Exception as exc:  # noqa: BLE001 - recorded, and it makes the cell incomplete
        error = f"{type(exc).__name__}: {exc}"[:300]
    tokens = (usage or {}).get("completion_tokens") or len(pieces)
    return {"first_s": first, "end_s": time.perf_counter() - offered, "tokens": tokens, "text": "".join(pieces),
            "error": error}


def run_cell(port: int, model: str, texts: list[str], at_once: bool, limit: int, usage_option: bool) -> dict[str, Any]:
    out: dict[int, dict[str, Any]] = {}
    if at_once:
        offered = time.perf_counter()
        threads = [threading.Thread(target=lambda i=i: out.__setitem__(
            i, ask(port, model, texts[i], limit, offered, usage_option))) for i in range(len(texts))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        for i, text in enumerate(texts):
            out[i] = ask(port, model, text, limit, time.perf_counter(), usage_option)
    rows = [out[i] for i in range(len(texts))]
    return {"at_once": at_once, "requests": rows,
            "complete": all(r["error"] is None and r["tokens"] and r["first_s"] is not None for r in rows)}


# ---------------------------------------------------------------- the servers


def commands(arm: str, port: int, paths: dict[str, str]) -> tuple[list[str], str, bool]:
    """Each arm's server command, the model name its requests carry, and whether they ask for usage."""
    if arm == "I":
        return ([paths["ironmule"], "serve", "--model", paths["ironmule_model"], "--agent", "on", "--host",
                 "127.0.0.1", "--port", str(port), "--state-dir", paths["state"], *paths.get("ironmule_extra", [])],
                paths["ironmule_model"], False)
    if arm in ("L", "Lf"):
        return ([paths["llama_server"], "-m", paths["gguf"], "--host", "127.0.0.1", "--port", str(port),
                 "-np", "8", "-c", str(8 * 2048), "--temp", "0", "--no-webui", *paths.get("llama_extra", []),
                 *(["-fa"] if arm == "Lf" else [])], "llama", True)
    if arm == "V":
        return ([paths["vllm_python"], "-m", "vllm.entrypoints.openai.api_server", "--model", HF_MODEL, "--dtype",
                 "half", "--max-num-seqs", "8", "--max-model-len", "2048", "--gpu-memory-utilization", "0.85",
                 "--host", "127.0.0.1", "--port", str(port), "--seed", "0"], HF_MODEL, True)
    raise ValueError(arm)


def _ready(port: int, arm: str) -> bool:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        response = conn.getresponse()
        body = response.read()
        conn.close()
        if response.status != 200:
            return False
        return json.loads(body).get("ready") is True if arm == "I" else True
    except (OSError, ValueError):
        return False


def _health(port: int) -> Any:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request("GET", "/health")
        return json.loads(conn.getresponse().read())
    except (OSError, ValueError) as exc:
        return {"error": type(exc).__name__}


def _tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def run_arm(arm: str, run: int, paths: dict[str, str], plan: list[tuple[str, list[str], bool, int]], warm: list[str],
            work: Path, save: Any, deadline: float) -> dict[str, Any]:
    """One arm: start its server, warm it (8 at once, then one alone, unmeasured), run `plan`'s cells, stop it."""
    port = _free_port()
    command, model, usage_option = commands(arm, port, paths)
    log = work / f"{arm}{run}.log"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("LC2_DEVICE", "0")}
    record: dict[str, Any] = {"arm": arm, "run": run, "command": command, "cells": [], "started": False}
    with open(log, "wb") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    began = time.perf_counter()
    try:
        while not _ready(port, arm):
            if process.poll() is not None or time.perf_counter() - began > 3600 or time.time() > deadline:
                record["error"] = "the server did not become ready"
                record["log_tail"] = _tail(log)
                return record
            time.sleep(1.0)
        record.update(started=True, startup_s=time.perf_counter() - began)
        run_cell(port, model, warm[:8], True, 32, usage_option)
        run_cell(port, model, warm[8:9], False, 32, usage_option)
        for kind, texts, at_once, limit in plan:
            if time.time() > deadline:
                record["error"] = "the hard limit ended the arm"
                break
            cell = run_cell(port, model, texts, at_once, limit, usage_option)
            cell["kind"] = kind
            record["cells"].append(cell)
            save()
            if not cell["complete"]:
                record["log_tail"] = _tail(log)
        record["health"] = _health(port) if arm == "I" else None
    finally:
        try:
            os.killpg(process.pid, 15)
            process.wait(timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, 9)
            except OSError:
                pass
        time.sleep(5.0)  # the card's memory is free before the next server
    return record


# ---------------------------------------------------------------- preparing the session


def _sh(command: list[str], cwd: Path | None = None, log: Path | None = None) -> None:
    with open(log, "ab") if log else open(os.devnull, "wb") as handle:
        subprocess.run(command, cwd=cwd, check=True, stdout=handle, stderr=subprocess.STDOUT)


def prepare(work: Path, report: dict[str, Any]) -> dict[str, str]:
    """On the T4 session: build `llama-server` with CUDA, install vLLM into its own environment, fetch the models and
    register IronMule's. Each step's time and failure is recorded; a failed step leaves its arms out."""
    from huggingface_hub import hf_hub_download, snapshot_download

    steps = report.setdefault("prepare", {})
    paths: dict[str, str] = {"state": str(work / "state")}
    bin_dir = Path(sys.executable).parent

    def step(name: str, action: Any) -> None:
        started = time.perf_counter()
        try:
            action()
            steps[name] = {"ok": True, "seconds": round(time.perf_counter() - started, 1)}
        except Exception as exc:  # noqa: BLE001 - recorded; the arms that need it are left out
            steps[name] = {"ok": False, "seconds": round(time.perf_counter() - started, 1),
                           "error": f"{type(exc).__name__}: {exc}"[:500], "log_tail": _tail(work / f"{name}.log")}

    def build_llama() -> None:
        _sh([sys.executable, "-m", "pip", "download", "--no-deps", "--no-binary", ":all:", "-d", str(work / "src"),
             f"llama-cpp-python=={LLAMA_CPP_PYTHON}"], log=work / "llama.log")
        archive = next((work / "src").glob("llama_cpp_python-*.tar.gz"))
        with tarfile.open(archive) as tar:
            tar.extractall(work / "src")  # noqa: S202 - the pinned sdist from PyPI
        source = next(p for p in (work / "src").glob("llama_cpp_python-*/vendor/llama.cpp") if p.is_dir())
        import shutil

        if shutil.which("cmake") is None:
            _sh([sys.executable, "-m", "pip", "install", "-q", "cmake"], log=work / "llama.log")
        nvcc = Path("/usr/local/cuda/bin/nvcc")
        if shutil.which("nvcc") is None and nvcc.exists():
            os.environ["CUDACXX"] = str(nvcc)
            os.environ["PATH"] = f"{nvcc.parent}{os.pathsep}{os.environ['PATH']}"
        _sh(["cmake", "-B", "build", "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75", "-DLLAMA_CURL=OFF",
             "-DCMAKE_BUILD_TYPE=Release"], cwd=source, log=work / "llama.log")
        _sh(["cmake", "--build", "build", "--target", "llama-server", "-j", str(os.cpu_count() or 4)], cwd=source,
            log=work / "llama.log")
        paths["llama_server"] = str(source / "build" / "bin" / "llama-server")
        version = subprocess.run([paths["llama_server"], "--version"], capture_output=True, text=True)
        report["llama_server"] = (version.stdout + version.stderr).strip()[-300:]

    def install_vllm() -> None:
        _sh([sys.executable, "-m", "venv", str(work / "vllm")], log=work / "vllm.log")
        python = work / "vllm" / "bin" / "python"
        _sh([str(python), "-m", "pip", "install", "-q", f"vllm=={VLLM}"], log=work / "vllm.log")
        paths["vllm_python"] = str(python)

    def models() -> None:
        paths["gguf"] = hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir=str(work / "gguf"))
        snapshot_download(HF_MODEL)
        snapshot_download(MLX_MODEL)

    def ironmule() -> None:
        paths["ironmule"] = str(bin_dir / "ironmule")
        paths["ironmule_model"] = MLX_MODEL
        _sh([paths["ironmule"], "setup", "--mode", "server", "--state-dir", paths["state"]], log=work / "ironmule.log")
        _sh([paths["ironmule"], "models", "add", MLX_MODEL, "--state-dir", paths["state"]], log=work / "ironmule.log")

    step("models", models)
    step("ironmule", ironmule)
    step("llama", build_llama)
    step("vllm", install_vllm)
    return paths


def _machine() -> dict[str, Any]:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True)
    return {"machine": platform.machine(), "cpus": os.cpu_count(), "gpus": smi.stdout.strip().splitlines(),
            "python": platform.python_version()}


# ---------------------------------------------------------------- the run


def measure(out: Path, work: Path, paths: dict[str, str] | None, report: dict[str, Any], *, arms: tuple[str, ...] = ORDER,
            smoke: bool = False) -> dict[str, Any]:
    deadline = time.time() + HARD_LIMIT_S
    texts, digest = dolly_prompts(work)
    report["dolly_sha256"] = digest

    def save() -> None:
        out.write_text(json.dumps(report))

    if paths is None:
        paths = prepare(work, report)
        save()
    quiet = texts[QUIET_FIRST + 50:QUIET_FIRST + 52] if smoke else texts[QUIET_FIRST:QUIET_FIRST + QUIET_SIZE]
    plan = ([("load", CPU6_PROMPTS, True, 16 if smoke else LOAD_LIMIT)] * (1 if smoke else LOAD_BLOCKS)
            + [("quiet", quiet, False, 16 if smoke else QUIET_LIMIT)])
    warm = texts[WARM_FIRST:WARM_FIRST + 9]
    needs = {"I": "ironmule", "L": "llama_server", "Lf": "llama_server", "V": "vllm_python"}
    report.setdefault("arms", [])
    seen: dict[str, int] = {}
    for arm in arms:
        seen[arm] = seen.get(arm, 0) + 1
        if needs[arm] not in paths:
            report["arms"].append({"arm": arm, "run": seen[arm], "started": False,
                                   "error": f"not prepared: {needs[arm]}", "cells": []})
            save()
            continue
        record = run_arm(arm, seen[arm], paths, plan, warm, work, save, deadline)
        report["arms"].append(record)
        save()
    report["analysis"] = None if smoke else analyse(report)
    save()
    return report


# ---------------------------------------------------------------- the reading fixed before the data


def _interval(values: list[float], seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(RESAMPLES))
    return [statistics.fmean(values), means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]]


def _rate(row: dict[str, Any]) -> float | None:
    span = row["end_s"] - row["first_s"] if row["first_s"] is not None else 0
    return (row["tokens"] - 1) / span if row["tokens"] and row["tokens"] > 1 and span > 0 else None


def _block_rate(cell: dict[str, Any]) -> float:
    return sum(r["tokens"] for r in cell["requests"]) / max(r["end_s"] for r in cell["requests"])


def _profile(record: dict[str, Any]) -> dict[str, Any] | None:
    """An arm run's per-block load rates and first tokens, and per-prompt quiet rates and first tokens."""
    cells = record.get("cells") or []
    load = [c for c in cells if c["kind"] == "load"]
    quiet = [c for c in cells if c["kind"] == "quiet"]
    if len(load) != LOAD_BLOCKS or len(quiet) != 1 or not all(c["complete"] for c in cells):
        return None
    return {"load": [_block_rate(c) for c in load],
            "load_first": [statistics.fmean(r["first_s"] for r in c["requests"]) for c in load],
            "quiet": [_rate(r) for r in quiet[0]["requests"]],
            "quiet_first": [r["first_s"] for r in quiet[0]["requests"]]}


def analyse(report: dict[str, Any]) -> dict[str, Any]:
    """BACKLOG LC2's reading: I (the mean of its two runs) over each other arm, paired by block and by prompt; MLX
    holds its own against an engine when both lower bounds are at least 0.95; the fastest engine by the geometric
    mean of its load and quiet rates; the owner's 1.2 goal for I against L."""
    runs: dict[str, list[dict[str, Any]]] = {}
    failed = {}
    for record in report.get("arms", []):
        profile = _profile(record)
        if profile is None:
            failed[f"{record['arm']}{record['run']}"] = record.get("error") or "incomplete cells"
        else:
            runs.setdefault(record["arm"], []).append(profile)
    if len(runs.get("I", [])) != 2:
        return {"valid": False, "problems": ["IronMule's two runs are not both complete"], "failed": failed}
    i1, i2 = runs["I"]
    mean = {key: [(a + b) / 2 if None not in (a, b) else None for a, b in zip(i1[key], i2[key])] for key in i1}
    aa = {"load": statistics.fmean(abs(b / a - 1) for a, b in zip(i1["load"], i2["load"])),
          "quiet": statistics.fmean(abs(b / a - 1) for a, b in zip(i1["quiet"], i2["quiet"]) if None not in (a, b))}

    def speed(profile: dict[str, Any]) -> float:
        rates = [x for x in profile["quiet"] if x is not None]
        return math.sqrt(statistics.fmean(profile["load"]) * statistics.fmean(rates))

    against = {}
    for arm in ("L", "Lf", "V"):
        if arm not in runs:
            continue
        other = runs[arm][0]
        load = _interval([a / b for a, b in zip(mean["load"], other["load"])])
        quiet = _interval([a / b for a, b in zip(mean["quiet"], other["quiet"]) if None not in (a, b)])
        first = _interval([a / b for a, b in zip(mean["quiet_first"], other["quiet_first"])])
        load_first = _interval([a / b for a, b in zip(mean["load_first"], other["load_first"])])
        against[arm] = {"load_tokens_per_s": load, "quiet_tokens_per_s": quiet, "quiet_first": first,
                        "load_first": load_first, "holds_its_own": load[1] >= HOLDS and quiet[1] >= HOLDS}
    speeds = {"I": speed(mean)} | {arm: speed(runs[arm][0]) for arm in against}
    fastest = max(speeds, key=speeds.get)
    clear = fastest != "I" and speeds[fastest] / speeds["I"] - 1 > max(aa.values())
    goal = against.get("L")
    return {"valid": True, "failed": failed, "i_aa": aa, "speeds": speeds, "against": against, "fastest": fastest,
            "next": (f"the agent on {fastest} (its own entry)" if clear else "MLX stays the CUDA base"),
            "goal_1_2_against_l": (None if goal is None else
                                   goal["load_tokens_per_s"][1] >= 1.2 and goal["quiet_tokens_per_s"][1] >= 1.2)}


# ---------------------------------------------------------------- the kernel


_BOOTSTRAP = '''"""IronMule LC2 on this Kaggle T4 session; written by tools/lc2_measure.py."""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ironmule[cuda] @ git+{repository}@{commit}"],
               check=True)
subprocess.run(["git", "clone", "-q", "{repository}", "/kaggle/working/ironmule"], check=True)
subprocess.run(["git", "-C", "/kaggle/working/ironmule", "checkout", "-q", "{commit}"], check=True)
sys.exit(subprocess.run([sys.executable, "/kaggle/working/ironmule/tools/lc2_measure.py", "--out",
                         "/kaggle/working/lc2_report.json", "--commit", "{commit}"]).returncode)
'''


def write_kernel(directory: Path, owner: str, commit: str) -> Path:
    """A private script kernel on one T4 session that installs IronMule at `commit` and runs this file from it."""
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("commit must be a full 40-character lowercase SHA, so the kernel runs exactly it")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.py").write_text(_BOOTSTRAP.format(repository=REPOSITORY, commit=commit))
    metadata = {"id": f"{owner}/ironmule-lc2", "title": "ironmule-lc2", "code_file": "run.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
                "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
                "machine_shape": "NvidiaTeslaT4"}
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1) + "\n")
    return directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--analyse", type=Path)
    parser.add_argument("--write-kernel", type=Path)
    parser.add_argument("--owner")
    parser.add_argument("--local-smoke", action="store_true",
                        help="this container's CPU: I on SmolLM2 and L with --server/--gguf, tiny cells, not counted")
    parser.add_argument("--server")
    parser.add_argument("--gguf")
    args = parser.parse_args(argv)
    if args.analyse:
        print(json.dumps(analyse(json.loads(args.analyse.read_text())), indent=1))
        return 0
    if args.write_kernel:
        if not args.owner or not args.commit:
            parser.error("--write-kernel needs --owner and --commit")
        print(write_kernel(args.write_kernel, args.owner, args.commit))
        return 0
    if not args.out:
        parser.error("--out is needed")
    work = args.out.resolve().parent / "lc2_work"
    work.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"protocol": "lc2" + ("-local-smoke" if args.local_smoke else ""),
                              "commit": args.commit, "machine": None, "vllm": VLLM, "llama_cpp_python": LLAMA_CPP_PYTHON,
                              "models": {"I": MLX_MODEL, "L": f"{GGUF_REPO}/{GGUF_FILE}", "V": HF_MODEL}}
    if args.local_smoke:
        import shutil

        state = work / "state"
        ironmule = shutil.which("ironmule") or str(Path(sys.executable).parent / "ironmule")
        model = "mlx-community/SmolLM2-135M-Instruct-8bit"
        subprocess.run([ironmule, "setup", "--mode", "server", "--state-dir", str(state)], check=True,
                       capture_output=True)
        subprocess.run([ironmule, "models", "add", model, "--state-dir", str(state)], capture_output=True)
        paths = {"ironmule": ironmule, "ironmule_model": model, "state": str(state), "llama_server": args.server,
                 "gguf": args.gguf, "llama_extra": ["-t", "4"]}
        measure(args.out, work, paths, report, arms=("I", "L", "Lf", "I"), smoke=True)
    else:
        report["machine"] = _machine()
        measure(args.out, work, None, report)
    print(json.dumps(report.get("analysis"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
