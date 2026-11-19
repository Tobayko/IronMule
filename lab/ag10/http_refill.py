"""Stream one long answer alongside two waves of short requests against an already running GGUF service."""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen


def chat(url, model, prompt, limit, origin, received=None):
    started = time.monotonic()
    request = Request(url + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                      data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                                       "max_tokens": limit, "temperature": 0, "stream": True}).encode())
    first = terminal = usage = None
    text = hashlib.sha256()
    with urlopen(request, timeout=130) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            choice = chunk["choices"][0]
            content = choice["delta"].get("content")
            if content:
                if first is None:
                    first = time.monotonic()
                    if received:
                        received.set()
                text.update(content.encode())
            if choice.get("finish_reason"):
                terminal = choice["finish_reason"]
                usage = chunk["usage"]
        else:
            raise RuntimeError("SSE stream ended without [DONE]")
    if first is None or terminal not in ("length", "stop") or not usage:
        raise RuntimeError("incomplete answer")
    return {"id": chunk["id"], "start_s": started - origin, "end_s": time.monotonic() - origin,
            "ttft_s": first - started, "usage": usage, "finish": terminal, "answer_sha256": text.hexdigest()}


def measure(url, state, anchor_tokens=384, first_finished=31):
    if not 1 <= first_finished <= 31 or not 1 <= anchor_tokens <= 8192:
        raise ValueError("invalid first-wave threshold or anchor output limit")
    def health():
        with urlopen(url + "/health", timeout=4) as response:
            return json.load(response)

    initial = health()
    if not initial["ready"] or initial["execution"] != "numeric@ggml":
        raise RuntimeError("expected a ready GGUF worker")
    model = initial["loaded_model"]
    chat(url, model, "What is the capital of Japan?", 4, time.monotonic())
    clock = time.monotonic()
    anchor_started = threading.Event()
    anchor_prompt = "List every integer from 1 to 1000 in ascending order, separated by commas. Start at 1."
    waves = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        anchor = pool.submit(chat, url, model, anchor_prompt, anchor_tokens, clock, anchor_started)
        if not anchor_started.wait(30):
            raise RuntimeError("the first long-answer token did not arrive")
        index = 0
        pending = []
        for count in (31, 7):
            started = time.monotonic() - clock
            jobs = []
            for _ in range(count):
                prompt = ("Begin with the word Paris. " + "Read this sample. " * (index % 5)
                          + f"Sample {index}.")
                jobs.append(pool.submit(chat, url, model, prompt, 2 + index % 3, clock))
                index += 1
            if not pending:
                for completed, future in enumerate(as_completed(jobs), start=1):
                    future.result()
                    if completed == first_finished:
                        break
            pending.append((started, jobs))
        for started, jobs in pending:
            rows = [job.result() for job in jobs]
            ttfts = sorted(row["ttft_s"] for row in rows)
            waves.append({"count": len(jobs), "wall_s": max(r["end_s"] for r in rows) - started, "rows": rows,
                          "p95_ttft_s": ttfts[math.ceil(.95 * len(rows)) - 1]})
        long_row = anchor.result()
    ended = max([long_row["end_s"]] + [r["end_s"] for w in waves for r in w["rows"]])
    tokens = long_row["usage"]["completion_tokens"] + sum(
        r["usage"]["completion_tokens"] for wave in waves for r in wave["rows"])
    final = health()
    groups = {}
    if state:
        ids = {long_row["id"]} | {r["id"] for w in waves for r in w["rows"]}
        for line in (state / "stability-log.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "member_done" and event.get("id") in ids:
                groups[event["id"]] = event["group"]
        if set(groups) != ids:
            raise RuntimeError("the service lifecycle journal did not account for every response")
    return {"model": model, "anchor_limit": anchor_tokens, "first_finished": first_finished, "wall_s": ended, "tokens_per_second": tokens / ended,
            "ggml": final["ggml"], "long": long_row, "waves": waves,
            "groups": groups, "completed": final["completed_requests"] - initial["completed_requests"] - 1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--state", type=Path, help="local server state to verify group membership")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--anchor-tokens", type=int, default=384)
    parser.add_argument("--first-finished", type=int, default=31,
                        help="send the second wave after this many first-wave responses have finished")
    args = parser.parse_args()
    with args.out.open("x") as target:
        json.dump(measure(args.url.rstrip("/"), args.state, args.anchor_tokens, args.first_finished), target, indent=2)
        target.write("\n")
