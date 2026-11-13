"""CMP1: one benchmark for every OpenAI-compatible server (IronMule, llama-server, Ollama, LM Studio, vLLM, SGLang).

Usage: bench.py URL MODEL LABEL OUT.json [--workloads burst-sys,burst,poisson,single,facts] [--ns 1,8,16,32]

Workloads (all greedy, streamed chat completions, `/no_think` for Qwen3's template):
  burst-sys  N requests at once, each with the same 479-token system prompt (lab/fa1/system.txt) and its own short
             question, 64 tokens; one warm-up block and 3 measured blocks, every block with new questions (the system
             prompt repeats, as in a chat app or RAG front end)
  burst      the same without a system prompt
  poisson    open-loop arrivals for 90 s at RATE requests/s (--rate), system prompt, 64 tokens: completed tokens/s,
             first-token p50/p90/p99, end-to-end p50/p90
  single     one request alone, 256 tokens, 3 runs: decode tokens/s after the first token, and the first token
  facts      lab/row1/facts.py's 50 questions, 48 tokens, 8 at a time: how many contain an expected answer

Tokens are counted by re-tokenizing every answer with the same tokenizer (Qwen3's, HF snapshot in IRONMULE_CMP_TOKENIZER
or Qwen/Qwen3-1.7B's cache), so engines that report usage differently are counted alike.
"""

from __future__ import annotations

import argparse
import glob
import http.client
import json
import random
import statistics
import sys
import threading
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = (ROOT / "lab/fa1/system.txt").read_text()
TOPICS = ["a lighthouse keeper", "a lost robot", "a mountain village", "a river in winter", "a clockmaker",
          "a desert caravan", "a library at night", "a sailing ship", "a city garden", "an old bakery", "a train journey",
          "a forest fire watch", "a chess player", "a beekeeper", "a snowstorm", "a market day", "a violin maker",
          "a moon base", "a fisherman", "a school teacher", "a radio operator", "a glass blower", "a postman",
          "a tea plantation", "a castle ruin", "an island ferry", "a night nurse", "a bridge builder", "a museum guard",
          "a street musician", "a coal miner", "a weather station", "a kite festival", "a bicycle courier", "a toy shop",
          "a canal boat", "a stone mason", "a zoo keeper", "a printing press", "a hot air balloon", "a mapmaker"]


def tokenizer():
    import os

    from transformers import AutoTokenizer

    path = os.environ.get("IRONMULE_CMP_TOKENIZER") or glob.glob(
        str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*"))[0]
    return AutoTokenizer.from_pretrained(path)


class Client:
    def __init__(self, url: str, model: str):
        parts = urllib.parse.urlparse(url)
        self.host, self.port, self.prefix, self.model = parts.hostname, parts.port or 80, parts.path.rstrip("/"), model

    def chat(self, messages, max_tokens: int, t0: float | None = None) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True}
        started = time.perf_counter()
        t0 = started if t0 is None else t0
        conn = http.client.HTTPConnection(self.host, self.port, timeout=3600)
        conn.request("POST", f"{self.prefix}/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read()[:200]!r}")
        first, pieces = None, []
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            try:
                chunk = json.loads(line[5:])
            except ValueError:
                continue
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if text:
                    first = first if first is not None else time.perf_counter()
                    pieces.append(text)
        end = time.perf_counter()
        return {"text": "".join(pieces), "first": (first or end) - t0, "start": started - t0, "end": end - t0}


def messages(question: str, system: bool):
    return ([{"role": "system", "content": SYSTEM}] if system else []) + [{"role": "user",
                                                                          "content": f"{question} /no_think"}]


def burst(client, tok, n: int, system: bool, block_index: int, limit: int = 64) -> dict:
    out, t0 = [None] * n, time.perf_counter()

    def one(i):
        topic = TOPICS[(block_index * n + i) % len(TOPICS)]
        out[i] = client.chat(messages(f"Write a short story about {topic}, variant {block_index}-{i}.", system), limit, t0)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = max(o["end"] for o in out)
    tokens = sum(len(tok.encode(o["text"])) for o in out)
    return {"tok_s": tokens / wall, "tokens": tokens, "wall_s": wall,
            "ttft_mean_s": statistics.mean(o["first"] for o in out), "ttft_max_s": max(o["first"] for o in out)}


def poisson(client, tok, rate: float, seconds: float = 90.0, limit: int = 64, seed: int = 0) -> dict:
    rng, t, arrivals = random.Random(seed), 0.0, []
    while t < seconds:
        t += rng.expovariate(rate)
        arrivals.append(t)
    out, t0 = [None] * len(arrivals), time.perf_counter()

    def one(i, at):
        delay = at - (time.perf_counter() - t0)
        if delay > 0:
            time.sleep(delay)
        topic = TOPICS[i % len(TOPICS)]
        r = client.chat(messages(f"Describe {topic} in a few sentences, request {i}.", True), limit, t0)
        r["arrival"] = at
        out[i] = r

    threads = [threading.Thread(target=one, args=(i, at)) for i, at in enumerate(arrivals)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = max(o["end"] for o in out)
    ttft = sorted(o["first"] - o["arrival"] for o in out)
    e2e = sorted(o["end"] - o["arrival"] for o in out)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    return {"rate": rate, "requests": len(out), "tok_s": sum(len(tok.encode(o["text"])) for o in out) / wall,
            "ttft_p50": q(ttft, 0.5), "ttft_p90": q(ttft, 0.9), "ttft_p99": q(ttft, 0.99),
            "e2e_p50": q(e2e, 0.5), "e2e_p90": q(e2e, 0.9), "wall_s": wall}


def single(client, tok, limit: int = 256) -> dict:
    runs = []
    for k in range(4):
        r = client.chat(messages(f"Explain in detail how a mechanical clock works, part {k}.", False), limit)
        n = len(tok.encode(r["text"]))
        runs.append({"ttft_s": r["first"], "decode_tok_s": (n - 1) / max(1e-9, r["end"] - r["first"]), "tokens": n})
    return {"runs": runs[1:], "decode_tok_s": statistics.median(x["decode_tok_s"] for x in runs[1:]),
            "ttft_s": statistics.median(x["ttft_s"] for x in runs[1:])}


def facts(client) -> dict:
    src = (ROOT / "lab/row1/facts.py").read_text()
    questions = eval(src[src.index("Q = [") + 4:src.index("]\nge.") + 1])
    ok, answers = 0, []
    for k in range(0, len(questions), 8):
        group, out = questions[k:k + 8], [None] * 8

        def one(i):
            out[i] = client.chat(messages(group[i][0], False), 48)["text"]

        threads = [threading.Thread(target=one, args=(i,)) for i in range(len(group))]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        for (q, expected), a in zip(group, out):
            a = (a or "").replace("<think>", "").replace("</think>", "").strip()
            good = any(e in a.lower() for e in expected)
            ok += good
            answers.append({"q": q, "ok": good, "a": a[:120]})
    return {"correct": ok, "of": len(questions), "answers": answers}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("model")
    parser.add_argument("label")
    parser.add_argument("out", type=Path)
    parser.add_argument("--workloads", default="burst-sys,burst,single,facts")
    parser.add_argument("--ns", default="1,8,16,32")
    parser.add_argument("--rate", type=float, default=0.5)
    args = parser.parse_args(argv)
    client, tok = Client(args.url, args.model), tokenizer()
    result = json.loads(args.out.read_text()) if args.out.exists() else {"label": args.label}
    for workload in args.workloads.split(","):
        if workload in ("burst-sys", "burst"):
            for n in [int(x) for x in args.ns.split(",")]:
                burst(client, tok, n, workload == "burst-sys", 99)  # warm-up block
                blocks = [burst(client, tok, n, workload == "burst-sys", b) for b in range(3)]
                result.setdefault(workload, {})[str(n)] = blocks
                print(args.label, workload, n, [round(b["tok_s"], 1) for b in blocks],
                      [round(b["ttft_mean_s"], 2) for b in blocks], flush=True)
        elif workload == "poisson":
            result.setdefault("poisson", {})[str(args.rate)] = r = poisson(client, tok, args.rate)
            print(args.label, "poisson", json.dumps({k: round(v, 2) for k, v in r.items()}), flush=True)
        elif workload == "single":
            result["single"] = r = single(client, tok)
            print(args.label, "single", round(r["decode_tok_s"], 2), "tok/s, first", round(r["ttft_s"], 3), flush=True)
        elif workload == "facts":
            result["facts"] = r = facts(client)
            print(args.label, "facts", r["correct"], "/", r["of"], flush=True)
        args.out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
