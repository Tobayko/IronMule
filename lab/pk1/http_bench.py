"""PK1 HTTP benchmark of one running OpenAI-compatible server with the deployment comparison's workload.

Uses ironmule_product.compare (the same streaming client, prompts and metrics as production_compare.py), so numbers are
comparable to the paired harness. The server must already listen on --url.

  python lab/pk1/http_bench.py --url http://127.0.0.1:8001 --engine vllm --model compare --widths 1,8,32 --repeats 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

from ironmule.factual_gate import MAX_TOKENS, cases, correct
from ironmule_product import compare


def scrape(url):
    """The server's Prometheus samples as {name{labels}: value} (empty where it has no /metrics)."""
    import urllib.request

    try:
        text = urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10).read().decode()
    except OSError:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("vllm:"):
            name, _, value = line.rpartition(" ")
            try:
                out[name] = float(value)
            except ValueError:
                pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--engine", required=True, choices=("vllm", "ironmule", "llama_cpp"))
    p.add_argument("--model", default="compare")
    p.add_argument("--widths", default="1,8,32")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261119)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--quality", action="store_true", help="also grade the factual cases at each width")
    p.add_argument("--label", default="")
    p.add_argument("--timeline", default="", help="PF1: one JSON line per request (send, every token, server id)")
    p.add_argument("--metrics", default="", help="PF1: FILE for the change of the server's /metrics over each wave "
                   "(vLLM: steps and their token counts, prefix-cache queries and hits)")
    a = p.parse_args()
    server = {"base_url": a.url, "model_id": a.model}
    deadline = time.monotonic() + 3600
    warm = time.perf_counter()
    compare.stream(a.engine, server, compare.WARMUP, 8, deadline)
    print(json.dumps({"label": a.label, "engine": a.engine, "warmup_s": round(time.perf_counter() - warm, 3)}), flush=True)
    for rep in range(a.repeats):
        for width in [int(w) for w in a.widths.split(",")]:
            seed = a.seed + rep * 1009 + width
            prompts = compare.performance_prompts(seed, width, max(width, 8))
            before = scrape(a.url) if a.metrics else None
            wave_start = time.monotonic()
            rows, elapsed = compare.batch(a.engine, server, prompts, width, a.max_tokens, deadline,
                                          timeline=bool(a.timeline))
            if a.timeline:
                with open(a.timeline, "a") as fh:
                    for index, r in enumerate(rows):
                        fh.write(json.dumps({"label": a.label, "engine": a.engine, "repeat": rep, "width": width,
                                             "index": index, "wave_start": wave_start, "server_id": r["server_id"],
                                             "prompt_tokens": r["prompt_tokens"], "tokens": r["tokens"],
                                             "ttft_ms": r["ttft_ms"], "t_send": r["t_send"], "t_done": r["t_done"],
                                             "chunk_times": r["chunk_times"]}) + "\n")
            if a.metrics:
                after = scrape(a.url)
                delta = {k: round(v - before.get(k, 0.0), 6) for k, v in after.items() if v != before.get(k, 0.0)}
                with open(a.metrics, "a") as fh:
                    fh.write(json.dumps({"label": a.label, "repeat": rep, "width": width, "delta": delta}) + "\n")
            m = compare.performance_metrics(rows, elapsed)
            out = {"label": a.label, "engine": a.engine, "repeat": rep, "width": width,
                   "tok_s": round(m["tokens_per_second"], 2), "wall_s": round(m["wall_s"], 3),
                   "ttft_p50_ms": round(m["ttft_median_ms"], 1), "ttft_p95_ms": round(m["ttft_p95_ms"], 1),
                   "ttft_max_ms": round(max(r["ttft_ms"] for r in rows), 1),  # the last request's wait for its first token
                   "tokens": m["completion_tokens"], "prompt_tokens": sum(m["performance_prompt_tokens"]),
                   "texts": hashlib.sha256(json.dumps([r["text"] for r in rows]).encode()).hexdigest()[:16]}
            if a.engine == "ironmule":  # the server's own intervals for this wave: queue, dispatch to first token
                try:
                    import urllib.request
                    live = json.load(urllib.request.urlopen(a.url + "/v1/ironmule/live", timeout=10))["runtime"]
                    recent = sorted(live["samples"], key=lambda s: s["at"])[-len(rows):]
                    for key in ("queue_ms", "dispatch_to_first_ms", "first_to_end_ms"):
                        vals = sorted(s[key] for s in recent if s.get(key) is not None)
                        if vals:
                            out[key] = [round(vals[len(vals) // 2], 1), round(vals[-1], 1)]
                except Exception as exc:  # noqa: BLE001 - the timing record is optional
                    out["live_error"] = type(exc).__name__
            if a.quality:
                qs = cases(seed)
                qrows, _ = compare.batch(a.engine, server, [c.prompt for c in qs], width, MAX_TOKENS, deadline)
                out["quality"] = f"{sum(correct(c, r['text']) for c, r in zip(qs, qrows))}/{len(qs)}"
                # the synthetic fact questions and their wrong answers, so a miss can be read (no user text)
                out["quality_misses"] = [{"key": c.key, "expected": c.expected, "answer": r["text"][:300]}
                                         for c, r in zip(qs, qrows) if not correct(c, r["text"])]
            print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
