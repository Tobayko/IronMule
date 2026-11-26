"""Private pipe worker for the ordinary GGUF product bridge on its qualified four-core CPU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path


def emit(value):
    print(json.dumps(value, allow_nan=False, separators=(",", ":")), flush=True)


def serve(bridge, tokenizer, prompts, limit):
    start = time.monotonic_ns()
    first = [None] * len(prompts)
    streams = [tokenizer.detokenizer for _ in prompts]

    def callback(i):
        def emit_token(token):
            streams[i].add_token(token)
            if first[i] is None and streams[i].last_segment:
                first[i] = (time.monotonic_ns() - start) / 1e9
        return emit_token

    ids = [tokenizer.encode(p) for p in prompts]
    if any(len(tokens) + limit > bridge.request_limit() for tokens in ids):
        raise ValueError("request exceeds the worker context")
    learning = bridge._engine.pilot.learning
    before = (learning.candidate_status or {}).get("qualified_decisions", 0)
    rows = bridge.generate_many(ids, limit, on_tokens=[callback(i) for i in range(len(ids))])
    if any(t is None for t in first) or any(r.tokens != s.tokens for r, s in zip(rows, streams, strict=True)):
        raise ValueError("incomplete streamed response")
    return {"answers": [{"tokens": r.tokens, "text": r.text, "finish_reason": r.finish_reason} for r in rows],
            "prompt_tokens": [len(x) for x in ids], "ttfts_s": first, "wall_s": (time.monotonic_ns() - start) / 1e9,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "scratch_uses": (learning.candidate_status or {}).get("qualified_decisions", 0) - before}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("gguf", "head", "order"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--arm", choices=("candidate", "reference"), required=True)
    p.add_argument("--profile", type=Path)
    p.add_argument("--matched-helper", type=Path)
    args = p.parse_args()
    from ironmule.ggml_binary import identity
    from ironmule.ggml_engine import file_sha256, machine_key
    from ironmule.gguf_tokenizer import GgufTokenizer
    from ironmule_product.ggml_bridge import GgmlBridge
    from training.pilot_threads import qualify_candidate

    b = json.loads(args.order.read_text())["binding"]
    if (identity() != b["native"] or machine_key() != b["machine"] or sorted(os.sched_getaffinity(0)) != b["affinity"]
            or file_sha256(str(args.head)) != b["scratch_thread_head"]["weights_sha256"]):
        raise ValueError("foreign native runtime, affinity or checkpoint")
    with tempfile.TemporaryDirectory() as state, closing(GgufTokenizer(str(args.gguf))) as tok:
        gate = {"sequences": {"sequences": 4}, "settings": b["engine_settings"]}
        with closing(GgmlBridge(str(args.gguf), tok, speculate=False, gate=gate)) as bridge:
            e = bridge._engine
            if (bridge.gguf_sha256 != b["model_sha256"] or e.pilot.cores != 4 or e.seqs != 4 or e.seq_ctx != 512
                    or e.gpu_layers or e.settings != b["engine_settings"] or e.pilot.weights_path is None
                    or e.prefix_reuse != b["prefix_reuse"] or file_sha256(str(e.pilot.weights_path)) != b["pilot_weights_sha256"]):
                raise ValueError("foreign product runtime binding")
            extra = {}
            expected = b["scratch_thread_head"]
            if args.profile:
                spec = importlib.util.spec_from_file_location("matched", args.matched_helper)
                matched = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(matched)
                profile = matched.read(args.profile)
                matched.verify_profile(profile, matched.read(args.order))
                expected = profile["binding"]["scratch_thread_head"]
                try:
                    q = matched.requalify(e, args.gguf, args.head, Path(state), profile)
                except ValueError as exc:
                    if args.arm != "reference":
                        raise
                    # This dedicated fallback still serves the unchanged base Pilot when the
                    # active load regime cannot re-earn the former target's thread offset.
                    if e.pilot.learning is None:
                        raise
                    e.pilot.learning.weights.clear()
                    e.pilot.learning.margins.clear()
                    q = {"binding": {"qualified_offsets": {}}, "fresh": {}, "search_and_warmup_s": None}
                    extra["abstention"] = str(exc)[:250]
                extra.update(profile_sha256=matched.sha(args.profile), costs=dict(e.pilot.costs), fresh=q["fresh"])
            else:
                q = qualify_candidate(e, args.gguf, args.head, Path(state), profile_clients=4)
            qualified = bool(q["binding"]["qualified_offsets"]) and q["binding"] == {k: expected[k] for k in q["binding"]}
            if args.arm == "candidate" and not qualified:
                raise ValueError("product worker could not requalify the measured head")
            emit({"ready": True, "qualified": qualified, "qualification": q["binding"], "search_s": q["search_and_warmup_s"],
                  "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **extra})
            for line in sys.stdin:
                item = json.loads(line)
                prompts, limit = item["prompts"], item["max_tokens"]
                if not 1 <= len(prompts) <= 4 or not all(isinstance(t, str) and 0 < len(t) <= 4096 for t in prompts) or not 1 <= limit <= 64:
                    raise ValueError("invalid worker input")
                emit(serve(bridge, tok, prompts, limit))


if __name__ == "__main__":
    main()
