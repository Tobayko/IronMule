"""Exercise placement through a running GGUF server with a known public question every six seconds."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args()

    def fetch(path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(args.url.rstrip("/") + path, data, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    model = next(row["id"] for row in fetch("/v1/models")["data"] if row.get("loaded"))
    start = time.monotonic()
    for index in range(13):
        if index:
            time.sleep(6)
        before = fetch("/health")["placement"]
        result = fetch("/v1/chat/completions", {"model": model, "messages": [
            {"role": "user", "content": "What is the capital of France?"}], "max_tokens": 12})
        print(json.dumps({"elapsed_seconds": round(time.monotonic() - start, 2),
                          "correct": "paris" in result["choices"][0]["message"]["content"].lower(),
                          "paired_trials_before": before["paired_trials"],
                          "idle_delay_seconds": before["idle_delay_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
