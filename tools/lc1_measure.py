"""LC1: IronMule's CPU setup against llama.cpp's `llama-server` on the same CPU and model (BACKLOG LC1).

Arm I is the agent's CPU6-A2 setup through `ProductService` (`tools/sp2_measure.py`'s arm A: the CPU stacked plan
at width 8 on OpenBLAS, speculation off); arm L is `llama-server` with the same model as a Q8_0 GGUF, 8 parallel
slots, greedy, over `/v1/chat/completions` with streaming. Load: CPU6's 8 prompts at once, 64 tokens, 5 blocks of
I L I; quiet: 16 Dolly prompts one at a time, 128 tokens, one block of I L I. `--analyse REPORT` applies the fixed,
descriptive analysis.

    python tools/lc1_measure.py --server PATH/llama-server --gguf MODEL.gguf --dolly DIR --out lc1.json
    python -I tools/lc1_measure.py --analyse lc1.json
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOAD_BLOCKS, LOAD_LIMIT = 5, 64
QUIET_FIRST, QUIET_SIZE, QUIET_LIMIT = 13_400, 16, 128
ORDER = ("I", "L", "I")
RESAMPLES = 2000


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ask_llama(port: int, text: str, limit: int, offered: float) -> dict[str, Any]:
    body = json.dumps({"messages": [{"role": "user", "content": text}], "max_tokens": limit, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True}})
    first, pieces, usage, error = None, [], None, None
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
        conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            for choice in chunk.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    first = first if first is not None else time.perf_counter() - offered
                    pieces.append(content)
            if chunk.get("usage"):
                usage = chunk["usage"]
        conn.close()
    except Exception as exc:  # noqa: BLE001 - recorded, and it makes the cell incomplete
        error = type(exc).__name__
    tokens = (usage or {}).get("completion_tokens") or len(pieces)
    return {"first_s": first, "end_s": time.perf_counter() - offered, "tokens": tokens, "text": "".join(pieces),
            "error": error}


def run_llama(server: str, gguf: str, texts: list[str], at_once: bool, limit: int) -> dict[str, Any]:
    """One arm on a fresh `llama-server`: the prompts one after another, or all offered at once."""
    port = _free_port()
    process = subprocess.Popen([server, "-m", gguf, "--host", "127.0.0.1", "--port", str(port), "-t", "4",
                                "-np", "8", "-c", str(8 * 2048), "--temp", "0", "--no-webui"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    started = time.perf_counter()
    try:
        while True:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request("GET", "/health")
                if conn.getresponse().status == 200:
                    break
            except OSError:
                pass
            if process.poll() is not None or time.perf_counter() - started > 600:
                raise RuntimeError("llama-server did not become ready")
            time.sleep(0.5)
        startup = time.perf_counter() - started
        out: dict[int, dict[str, Any]] = {}
        if at_once:
            offered = time.perf_counter()
            threads = [threading.Thread(target=lambda i=i: out.__setitem__(i, _ask_llama(port, texts[i], limit, offered)))
                       for i in range(len(texts))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            for i, text in enumerate(texts):
                out[i] = _ask_llama(port, text, limit, time.perf_counter())
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
    rows = [out[i] for i in range(len(texts))]
    return {"arm": "L", "at_once": at_once, "startup_s": startup, "requests": rows,
            "complete": all(r["error"] is None and r["tokens"] and r["first_s"] is not None for r in rows)}


def _interval(values: list[float], seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(RESAMPLES))
    return [statistics.fmean(values), means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]]


def _rate(row: dict[str, Any]) -> float | None:
    span = row["end_s"] - row["first_s"] if row["first_s"] is not None else 0
    return (row["tokens"] - 1) / span if row["tokens"] and row["tokens"] > 1 and span > 0 else None


def analyse(report: dict[str, Any]) -> dict[str, Any]:
    """LC1's fixed, descriptive analysis: L over I, with I's A/A noise."""
    load, quiet = report.get("load", []), report.get("quiet", [])
    complete = all(cell["complete"] for block in load + quiet for cell in block)
    out: dict[str, Any] = {"complete": complete, "load_blocks": len(load), "quiet_blocks": len(quiet)}

    def block_rate(cell):
        return sum(r["tokens"] for r in cell["requests"]) / max(r["end_s"] for r in cell["requests"])

    def block_first(cell):
        return statistics.fmean(r["first_s"] for r in cell["requests"])

    if load:
        out["load"] = {
            "i_tokens_per_s": statistics.fmean((block_rate(i1) + block_rate(i2)) / 2 for i1, _, i2 in load),
            "l_tokens_per_s": statistics.fmean(block_rate(l_) for _, l_, _ in load),
            "l_over_i_tokens_per_s": _interval([block_rate(l_) / ((block_rate(i1) + block_rate(i2)) / 2)
                                                for i1, l_, i2 in load]),
            "l_over_i_first": _interval([block_first(l_) / ((block_first(i1) + block_first(i2)) / 2)
                                         for i1, l_, i2 in load]),
            "i_aa_tokens_per_s": statistics.fmean(abs(block_rate(i2) / block_rate(i1) - 1) for i1, _, i2 in load)}
    if quiet:
        tps, first, i_rates, l_rates = [], [], [], []
        for i1, l_, i2 in quiet:
            for r1, rl, r2 in zip(i1["requests"], l_["requests"], i2["requests"]):
                x1, xl, x2 = _rate(r1), _rate(rl), _rate(r2)
                if None in (x1, xl, x2):
                    continue
                tps.append(xl / ((x1 + x2) / 2))
                first.append(rl["first_s"] / ((r1["first_s"] + r2["first_s"]) / 2))
                i_rates.append((x1 + x2) / 2)
                l_rates.append(xl)
        out["quiet"] = {"prompts": len(tps), "i_tokens_per_s": statistics.fmean(i_rates),
                        "l_tokens_per_s": statistics.fmean(l_rates), "l_over_i_tokens_per_s": _interval(tps),
                        "l_over_i_first": _interval(first)}
    out["answers_equal"] = sum(r_l["text"] == r_i["text"] for block in load + quiet
                               for r_i, r_l in zip(block[0]["requests"], block[1]["requests"]))
    out["answers"] = sum(len(block[1]["requests"]) for block in load + quiet)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", help="the llama-server binary")
    parser.add_argument("--gguf", help="the model as GGUF")
    parser.add_argument("--dolly", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--analyse", type=Path)
    parser.add_argument("--smoke", action="store_true", help="one tiny block of each, not the protocol's run")
    args = parser.parse_args(argv)
    if args.analyse:
        print(json.dumps(analyse(json.loads(args.analyse.read_text())), indent=1))
        return 0
    sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import hashlib

    import cpu6_measure
    import rl2_headroom
    import sp2_measure

    texts, digest = rl2_headroom.dolly_prompts(args.dolly)
    spec = sp2_measure._spec()
    gguf_sha = hashlib.sha256(Path(args.gguf).read_bytes()).hexdigest()
    version = subprocess.run([args.server, "--version"], capture_output=True, text=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    blocks, quiet_size, load_limit, quiet_limit = (1, 2, 16, 16) if args.smoke else (
        LOAD_BLOCKS, QUIET_SIZE, LOAD_LIMIT, QUIET_LIMIT)
    report: dict[str, Any] = {"protocol": "lc1" + ("-smoke" if args.smoke else ""), "commit": commit,
                              "ironmule_model": f"{sp2_measure.MODEL}@{sp2_measure.REVISION}",
                              "gguf_sha256": gguf_sha, "llama_server": (version.stdout + version.stderr).strip()[-300:],
                              "dolly_sha256": digest, "machine": {"machine": platform.machine(),
                                                                  "cpus": os.cpu_count()},
                              "load": [], "quiet": []}
    load_texts = cpu6_measure.prompts()
    quiet_first = QUIET_FIRST + (1_000 if args.smoke else 0)
    quiet_texts = texts[quiet_first:quiet_first + quiet_size]
    plan = [("load", load_texts, True, load_limit)] * blocks + [("quiet", quiet_texts, False, quiet_limit)]
    for kind, chunk, at_once, limit in plan:
        cells = []
        for arm in ORDER:
            cell = (sp2_measure.run_arm(spec, "A", chunk, at_once, limit) if arm == "I"
                    else run_llama(args.server, args.gguf, chunk, at_once, limit))
            cell["arm"] = arm
            cells.append(cell)
            print(json.dumps({kind: len(report[kind]), "arm": arm, "complete": cell["complete"],
                              "tokens": sum(r["tokens"] or 0 for r in cell["requests"]),
                              "seconds": round(max(r["end_s"] for r in cell["requests"]), 1)}), flush=True)
        report[kind].append(cells)
        if args.out:
            args.out.write_text(json.dumps(report))
    report["analysis"] = analyse(report)
    if args.out:
        args.out.write_text(json.dumps(report))
    print(json.dumps(report["analysis"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
