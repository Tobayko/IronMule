"""PERF1-K: aggregate throughput of `ironmule serve` with and without `--batch-width`, and its answers.

The flag was removed again after PERF1-K1 (ledger PERF1-K); this is the harness that measured it.

Usage: python batch_serve.py MODEL_ID OUT.json PLAN CONFIG [CONFIG ...]
       PLAN: bf16 | float32 | native;  CONFIG: ref | b8   (e.g. ref b8 ref b8)

Per CONFIG, in the given order: start the registered model with `ironmule serve` (the plan's
`--compute-dtype`, and `--batch-width 8` for b8), wait for /ready, then one warm-up and three
measured rounds. A round releases eight `stream: false` chat requests at once (perf1's server
prompts, `max_tokens` 128, greedy) and records the wall time until the last answer, every
answer's text and completion tokens, and /health afterwards; then the server is stopped.
"""
import json
import subprocess
import sys
import threading
import time
import urllib.request

PROMPTS = (  # perf1.py SERVER_PROMPTS
    "Explain how a refrigerator moves heat out of its interior, step by step.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Summarise the causes of the French Revolution in one paragraph.",
    "What is the difference between TCP and UDP? Give two examples where each is the better choice.",
    "Draft a polite email asking a landlord to repair a broken heating system before winter.",
    "List five practical tips for learning a new language as an adult, and explain why each works.",
    "Describe the water cycle to a ten-year-old.",
    "Compare electric cars and petrol cars on cost, maintenance and environmental impact.",
)
URL = "http://127.0.0.1:8080"
MAX_TOKENS, ROUNDS = 128, 3


def get(path):
    with urllib.request.urlopen(URL + path, timeout=10) as response:
        return json.loads(response.read())


def ask(model, prompt, results, index, barrier):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": MAX_TOKENS, "stream": False}).encode()
    request = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    barrier.wait()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.loads(response.read())
        results[index] = {"text": payload["choices"][0]["message"]["content"],
                          "completion_tokens": payload["usage"]["completion_tokens"],
                          "finish_reason": payload["choices"][0]["finish_reason"], "ended": time.time()}
    except Exception as exc:  # noqa: BLE001 - a failed request is a result
        results[index] = {"error": f"{type(exc).__name__}: {exc}"[:300], "ended": time.time()}


def round_(model):
    results = [None] * len(PROMPTS)
    barrier = threading.Barrier(len(PROMPTS) + 1)
    threads = [threading.Thread(target=ask, args=(model, prompt, results, index, barrier))
               for index, prompt in enumerate(PROMPTS)]
    for thread in threads:
        thread.start()
    barrier.wait()
    started = time.time()
    for thread in threads:
        thread.join()
    wall = max(row["ended"] for row in results) - started
    tokens = sum(row.get("completion_tokens", 0) for row in results)
    return {"wall_s": round(wall, 3), "completion_tokens": tokens, "aggregate_tps": round(tokens / wall, 3),
            "answers": results, "health": get("/health")}


def main(model, out, plan, *configs):
    report = {"schema": "ironmule.perf1k-serve.v1", "model_id": model, "plan": plan, "prompts": len(PROMPTS),
              "max_tokens": MAX_TOKENS, "starts": [], "performance_claim": False}
    for config in configs:
        command = ["ironmule", "serve", "--model", model]
        if plan != "bf16":
            command += ["--compute-dtype", plan]
        if config == "b8":
            command += ["--batch-width", "8"]
        log = open(f"{out}.{len(report['starts'])}-{config}.log", "w")
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        start = {"config": config, "command": command[1:], "rounds": []}
        try:
            deadline = time.time() + 600
            while time.time() < deadline:
                try:
                    if get("/ready").get("ready"):
                        break
                except OSError:
                    pass
                time.sleep(2)
            start["ready_health"] = get("/health")
            start["warmup"] = round_(model)
            start["rounds"] = [round_(model) for _ in range(ROUNDS)]
        finally:
            server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
            log.close()
        report["starts"].append(start)
        print(config, [r["aggregate_tps"] for r in start["rounds"]],
              start["rounds"][-1]["health"].get("last_batch_size") if start["rounds"] else None, flush=True)
        with open(out, "w") as stream:
            json.dump(report, stream, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
