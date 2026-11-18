"""Run short chats, documents and short chats again against the same warm GGUF server.

Start a fresh IronMule server with --speculate off, IRONMULE_PILOT=0, IRONMULE_GGML_SHARED=0
and IRONMULE_GGML_AGENT=1. Repeat with a fresh state directory containing just the same
ggml-gate cache, with IRONMULE_GGML_AGENT=0. This isolates AG10 from the step-level Pilot.
Run one arm at a time. The pause after each wave gives AG10 idle time to dream. Health
reports the most recent completed request, so the next wave observes any intervening switch.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from urllib.request import Request, urlopen

FACTS = (("France", "Paris"), ("Italy", "Rome"), ("Germany", "Berlin"),
         ("Japan", "Tokyo"), ("Spain", "Madrid"), ("Canada", "Ottawa"),
         ("Portugal", "Lisbon"), ("Austria", "Vienna"))
PHASES = ("short", "documents", "short_again")


def question(phase: str, index: int, lines: int) -> tuple[str, str]:
    country, answer = FACTS[index % len(FACTS)]
    prefix = f"Record {phase}-{index}. "
    if phase == "documents":
        prefix += "\n".join(f"Station {i} recorded {(index * 7 + i * 11) % 89} visitors."
                            for i in range(lines)) + "\n"
    return prefix + f"What is the capital of {country}? Answer with just the city name.", answer


def score(text: str, expected: str) -> tuple[bool, bool]:
    """Conservative factual match; also report compliance with the requested city-only format."""
    response = text.strip(" .!\r\n\t").casefold()
    country = next(country for country, city in FACTS if city == expected)
    bare = response == expected.casefold()
    full = response == f"the capital of {country} is {expected}".casefold()
    return bare or full, bare


def health(url: str) -> dict:
    with urlopen(url + "/health", timeout=5) as response:
        return json.load(response)


def chat(url: str, model: str, prompt: str, expected: str, limit: int) -> dict:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
               "max_tokens": limit, "stream": True}
    request = Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    start = time.monotonic()
    first, pieces, usage, done = None, [], None, False
    with urlopen(request, timeout=125) as response:
        for raw in response:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise RuntimeError(str(chunk["error"]))
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if text:
                    if first is None:
                        first = time.monotonic() - start
                    pieces.append(text)
    if not done or first is None or not usage or usage["completion_tokens"] <= 0:
        raise ValueError("incomplete streamed response or missing token count")
    text = "".join(pieces)
    correct, city_only = score(text, expected)
    return {"ttft_s": first, "prompt_tokens": usage["prompt_tokens"], "tokens": usage["completion_tokens"],
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text,
            "correct": correct, "city_only": city_only}


def measure(args: argparse.Namespace) -> dict:
    url = args.url.rstrip("/")
    initial = health(url)
    if not initial["ready"] or initial["execution"] != "numeric@ggml":
        raise RuntimeError("the GGUF server must be ready")
    model = initial["loaded_model"]
    warm, answer = question("warmup", 0, args.lines)
    chat(url, model, warm, answer, args.max_tokens)
    ggml = health(url)["ggml"]
    if bool(ggml.get("agent")) != (args.arm == "agent") or ggml.get("pilot", {}).get("active"):
        raise RuntimeError("the serving agent or Pilot does not match the requested arm")
    if ggml.get("shared_pool"):
        raise RuntimeError("both arms must use separate KV streams")
    report = {"model": model, "arm": args.arm, "starting_ggml": ggml,
              "protocol": {k: getattr(args, k) for k in ("requests", "width", "lines", "max_tokens", "idle")},
              "phases": {}}
    with ThreadPoolExecutor(max_workers=args.width) as pool:
        for phase in PHASES:
            waves, all_rows = [], []
            for start in range(0, args.requests, args.width):
                questions = [question(phase, i, args.lines) for i in range(start, min(start + args.width, args.requests))]
                begin = time.monotonic()
                jobs = [pool.submit(chat, url, model, q, a, args.max_tokens) for q, a in questions]
                rows = [job.result() for job in jobs]
                elapsed = time.monotonic() - begin
                status = health(url)["ggml"].get("agent", {})
                waves.append({"seconds": elapsed, "serving": status.get("serving"),
                              "decisions": status.get("last_decisions", [])})
                all_rows.extend(rows)
                time.sleep(args.idle)
            times = sorted(row["ttft_s"] for row in all_rows)
            value = {"requests": len(times), "correct": sum(row["correct"] for row in all_rows),
                     "city_only": sum(row["city_only"] for row in all_rows),
                     "completion_tok_s_active": sum(r["tokens"] for r in all_rows) / sum(w["seconds"] for w in waves),
                     "mean_prompt_tokens": statistics.mean(r["prompt_tokens"] for r in all_rows),
                     "ttft_mean_s": statistics.mean(times), "ttft_p95_s": times[math.ceil(.95 * len(times)) - 1],
                     "outputs": [r["output_sha256"] for r in all_rows],
                     "examples": [{"response": row["text"], "expected": question(phase, i, args.lines)[1]}
                                  for i, row in enumerate(all_rows[:2])],
                     "waves": waves}
            report["phases"][phase] = value
            print(phase, {k: v for k, v in value.items() if k not in ("waves", "outputs", "examples")}, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("agent", "fixed"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--requests", type=int, default=64, help="requests per phase")
    parser.add_argument("--width", type=int, default=16, help="concurrent requests per wave")
    parser.add_argument("--lines", type=int, default=96, help="document length in synthetic records")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--idle", type=float, default=5, help="seconds between waves, excluded from active throughput")
    args = parser.parse_args()
    if min(args.requests, args.width, args.lines, args.max_tokens) < 1 or not math.isfinite(args.idle) or args.idle < 0:
        parser.error("sizes must be positive and idle time finite and nonnegative")
    if args.out.exists():
        parser.error("choose a new output file")
    result = measure(args)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
