"""Wave tail probe: fire WIDTH streaming requests at once and print each request's end on the system's monotonic
clock, which the engine trace (IRONMULE_GGML_TRACE) also uses, so the tail after the last decode call can be read off.

  python lab/sv/tail_probe.py --url http://127.0.0.1:8002 --model gguf/NAME --width 32 --waves 3 --trace FILE
"""
from __future__ import annotations

import argparse
import json
import threading
import time

from ironmule_product import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--waves", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--trace", default="")
    a = p.parse_args()
    server = {"base_url": a.url, "model_id": a.model}
    deadline = time.monotonic() + 3600
    compare.stream("ironmule", server, compare.WARMUP, 8, deadline)
    for wave in range(a.waves):
        prompts = compare.performance_prompts(20261119 + wave, a.width, max(a.width, 8))
        ends, starts = [None] * a.width, time.monotonic()

        def run(i):
            compare.stream("ironmule", server, prompts[i], a.max_tokens, deadline)
            ends[i] = time.monotonic()

        threads = [threading.Thread(target=run, args=(i,)) for i in range(a.width)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        out = {"wave": wave, "start": starts, "first_end": min(ends), "last_end": max(ends)}
        if a.trace:  # the last decode call of this wave in the engine trace
            calls = [tuple(float(x) for x in line.split()[:3]) for line in open(a.trace)
                     if line.strip() and not line.startswith("#")]
            calls = [c for c in calls if starts <= c[0] <= out["last_end"]]
            if calls:
                last = max(c[0] + c[1] for c in calls)
                out.update(calls=len(calls), last_call_end=last, tail_ms=round(1e3 * (out["last_end"] - last), 1),
                           first_call_ms=round(1e3 * (calls[0][0] - starts), 1))
        print(json.dumps(out), flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
