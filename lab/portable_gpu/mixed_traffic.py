"""Replay interleaved tool-result histories, long document jobs and short lookups over real HTTP."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import threading
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

from ironmule_product import compare, compare_tokenizer

SCHEMA = "ironmule.heldout-mixed-traffic.v1"
WORDS = ("ALDER", "BIRCH", "CEDAR", "ELM", "FIR", "HAZEL", "LARCH", "MAPLE", "OAK", "PINE", "SPRUCE", "WILLOW")


@dataclass
class Request:
    key: str
    workload: str
    messages: list
    limit: int
    expected: str


def requests(seed):
    rng = random.Random(f"{SCHEMA}:{seed}")
    longs, short, agents = [], [], []
    for i in range(2):
        values = [rng.choice(WORDS) for _ in range(48)]
        keep = set(rng.sample(range(len(values)), 24))
        data = "\n".join(f"Row {n + 1}: {'KEEP' if n in keep else 'SKIP'} {word}" for n, word in enumerate(values))
        task = "Output every KEEP value in row order, separated by spaces. Omit SKIP rows and all other text.\n" + data
        longs.append(Request(f"document-{i}", "long_document", compare.conversation(task), 128,
                             " ".join(word for n, word in enumerate(values) if n in keep)))
    for i in range(8):
        code, item = rng.choice(WORDS), rng.randrange(1000, 10000)
        prompt = f"The dispatch record for shipment {item} says route {code}. Return only its route."
        short.append(Request(f"short-{i}", "short_lookup", compare.conversation(prompt), 12, code))
    for i in range(4):
        item, code = f"T{i}-{rng.randrange(100, 1000)}", rng.choice(WORDS)
        query = f"Find the route for case {item}. Request the record by replying only LOOKUP {item}."
        command = "LOOKUP " + item
        messages = compare.conversation(query)
        first = Request(f"agent-{i}-call", "agent_call", messages, 16, command)
        reply = json.dumps({"tool": "route_lookup", "case": item, "route": code}, separators=(",", ":"))
        history = messages + [{"role": "assistant", "content": command},
                              {"role": "user", "content": f"Tool result: {reply}\nReturn only the route value."}]
        second = Request(f"agent-{i}-result", "agent_result", history, 12, code)
        agents.append((first, second))
    return longs, short, agents


def measure(engine, server, seed, deadline, *, gguf=None):
    longs, short, agents = requests(seed)
    plan = longs + short + [turn for pair in agents for turn in pair]
    conversations = [r.messages for r in plan]
    if engine == "ironmule":
        from ironmule.gguf_tokenizer import GgufTokenizer

        tok = GgufTokenizer(str(gguf))
        try:
            proof = compare_tokenizer.rendered_prompt_ids(engine, server, conversations, deadline, tok)
        finally:
            tok.close()
    else:
        proof = compare_tokenizer.rendered_prompt_ids(engine, server, conversations, deadline)
    counts = dict(zip((r.key for r in plan), proof["prompt_tokens"], strict=True))
    long_visible = threading.Event()
    start = time.monotonic()

    def run(req):
        issued = time.monotonic() - start
        result = compare.stream(engine, server, req.messages, req.limit, deadline,
                                request_id=f"mixed-{seed}-{req.key}",
                                on_first_visible=long_visible.set if req.workload == "long_document" else None)
        ended = time.monotonic() - start
        if result["prompt_tokens"] != counts[req.key]:
            raise compare.Unverified("served usage does not match the verified prompt")
        response = result["text"]
        if result["visible_ttft_ms"] is None or "<think>" in response or "</think>" in response:
            raise compare.Unverified("expected visible content from the no-thinking deployment")
        return {"id": req.key, "class": req.workload, "issued_s": issued, "ended_s": ended,
                "visible_ttft_ms": result["visible_ttft_ms"], "service_ms": 1000 * (ended - issued),
                "prompt_tokens": result["prompt_tokens"], "tokens": result["tokens"], "finish": result["finish_reason"],
                "text": response, "correct": response.casefold().split() == req.expected.casefold().split()}

    def run_agent(pair):
        # A fixed tool-call history gives equivalent input even after a wrong first reply.
        # Score the live tool-call reply and final extraction separately and jointly.
        return [run(req) for req in pair]

    pool = ThreadPoolExecutor(max_workers=len(longs) + len(short) + len(agents))
    try:
        docs = [pool.submit(run, req) for req in longs]
        while not long_visible.wait(min(.05, compare.remaining(deadline))):
            for fut in docs:
                if fut.done():
                    fut.result()  # propagate a failed long request without spending the whole deadline
        calls = [pool.submit(run_agent, pair) for pair in agents]
        short_futures = [pool.submit(run, req) for req in short]
        rows = [fut.result(timeout=compare.remaining(deadline)) for fut in docs + short_futures]
        for fut in calls:
            rows.extend(fut.result(timeout=compare.remaining(deadline)))
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    by_id = {row["id"]: row for row in rows}
    docs = [by_id[r.key] for r in longs]
    overlapping = sum(any(d["issued_s"] <= by_id[r.key]["issued_s"] < d["ended_s"] for d in docs) for r in short)
    classes = {}
    for label in ("agent_call", "agent_result", "long_document", "short_lookup"):
        members = [row for row in rows if row["class"] == label]
        elapsed = max(row["ended_s"] for row in members) - min(row["issued_s"] for row in members)
        def p95(name):
            return sorted(row[name] for row in members)[math.ceil(.95 * len(members)) - 1]
        classes[label] = {"requests": len(members), "active_span_s": elapsed,
                          "output_tok_s": sum(row["tokens"] for row in members) / elapsed,
                          "requests_s": len(members) / elapsed, "visible_ttft_p95_ms": p95("visible_ttft_ms"),
                          "service_p95_ms": p95("service_ms"), "correct": sum(row["correct"] for row in members)}
    workflow_pass = sum(all(by_id[r.key]["correct"] for r in pair) for pair in agents)
    elapsed = max(row["ended_s"] for row in rows)
    return {"schema": SCHEMA, "seed": seed, "conversation_sha256": hashlib.sha256(
                json.dumps(conversations, separators=(",", ":")).encode()).hexdigest(),
            "prompt_id_digest": proof["digest"], "short_starts_during_long": overlapping,
            "fully_correct_workflows": workflow_pass, "workflows": len(agents), "wall_s": elapsed,
            "output_tok_s": sum(row["tokens"] for row in rows) / elapsed, "classes": classes, "requests": rows}


def pair(iron, reference):
    if (iron["seed"] != reference["seed"] or iron["conversation_sha256"] != reference["conversation_sha256"]
            or iron["prompt_id_digest"] != reference["prompt_id_digest"]):
        raise compare.Unverified("held-out request messages or rendered token IDs differ between deployments")
    keys = [(r["id"], r["prompt_tokens"]) for r in iron["requests"]]
    if keys != [(r["id"], r["prompt_tokens"]) for r in reference["requests"]]:
        raise compare.Unverified("the arms did not serve the same requests")
    if min(iron["short_starts_during_long"], reference["short_starts_during_long"]) == 0:
        return {"status": "no_overlap"}
    return {"status": "measured",
            "tokens_and_stops_match": all((a["tokens"], a["finish"]) == (b["tokens"], b["finish"])
                                          for a, b in zip(iron["requests"], reference["requests"], strict=True)),
            "changed_correctness": [a["id"] for a, b in zip(iron["requests"], reference["requests"], strict=True)
                                    if a["correct"] != b["correct"]],
            "ratio_order": ["ironmule_gguf", "vllm_awq"],
            "by_class": {label: {key: ours[key] / reference["classes"][label][key] for key in
                         ("output_tok_s", "requests_s", "visible_ttft_p95_ms", "service_p95_ms")}
                         for label, ours in iron["classes"].items()}}


def main(argv=None):
    """A standalone smoke against a local GGUF service; only the deployment runner produces paired evidence."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="local no-thinking IronMule service")
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--seed", type=int, default=12090427)
    p.add_argument("--budget-seconds", type=int, default=45)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if not 1 <= a.budget_seconds <= 300:
        p.error("use a budget between 1 and 300 seconds")
    deadline = time.monotonic() + a.budget_seconds
    address = urlsplit(a.url)
    if address.scheme != "http" or address.hostname not in ("127.0.0.1", "localhost", "::1"):
        p.error("use a loopback HTTP endpoint")
    with urlopen(a.url.rstrip("/") + "/v1/models", timeout=compare.remaining(deadline)) as res:
        models = [row for row in json.load(res)["data"] if row.get("loaded")]
    from ironmule.gguf_tokenizer import gguf_revision

    revision = gguf_revision(str(a.gguf))
    if len(models) != 1 or models[0].get("revision") != revision:
        raise compare.Unverified("local GGUF digest differs from the loaded standalone model")
    model = models[0]["id"]
    with a.output.open("x") as handle:
        data = measure("ironmule", {"base_url": a.url, "model_id": model}, a.seed, deadline, gguf=a.gguf)
        data["model"] = {"id": model, "revision": revision}
        json.dump(data, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
