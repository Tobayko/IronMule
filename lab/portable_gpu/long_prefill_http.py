"""Require the native long-prefill receipt and identify actual qualifying work in the HTTP waves."""

from __future__ import annotations

import hashlib
import json

from ironmule_product import compare


def _qualified(receipt, **scope):
    from ironmule.long_prefill_study import qualified_receipt

    return qualified_receipt(receipt, **scope)


def read_receipt(path):
    with path.open("rb") as stream:
        raw = stream.read(1_000_001)
    if len(raw) > 1_000_000:
        raise compare.Unverified("native long-444 receipt exceeds the worker limit")
    receipt = json.loads(raw)
    if not isinstance(receipt, dict) or not _qualified(receipt):
        raise compare.Unverified("native long-444 study did not qualify for HTTP")
    return receipt, hashlib.sha256(raw).hexdigest()


def verify_local(receipt, gguf_sha):
    from ironmule.long_prefill_study import local_identity

    if not _qualified(receipt, gguf_sha256=gguf_sha, **local_identity()):
        raise compare.Unverified("native receipt differs from the local GPU, GGUF, code or native runtime")


def verify_worker(receipt, arm, server, ggml, scope):
    layout, stats = ggml["experiment_binding"], ggml["prefill_batch"]
    expected = "long-444" if arm == "long-444" else "off"
    if (stats.get("mode") != expected or (ggml.get("gate") or {}).get("passes") is not True
            or (ggml.get("stream_qualification") or {}).get("passes") is not True
            or any(layout.get(k) != receipt[k] for k in ("seq_ctx", "seqs", "threads", "settings"))
            or not _qualified(receipt, gguf_sha256=server["gguf_sha256"], cuda_uuid=layout["cuda_uuid"],
                              implementation_sha256=scope["implementation_sha256"],
                              native_runtime_sha256=scope["native_runtime_sha256"])):
        raise compare.Unverified("loaded worker differs from the qualified long-444 study")


def wave_hits(arm, stats, previous, deltas):
    # Every counted grouped call issues one cold rectangular group of two or four 444-token prompts.
    def counts(snapshot):
        if any(type(snapshot.get(k)) is not int or snapshot[k] < 0 for k in ("calls", "prompts", "rows")):
            raise compare.Unverified("invalid grouped prefill counter")
        raw = snapshot.get("long_shape_hits", {} if arm == "stock" else None)
        if arm == "stock" and raw == {}:
            return {"2": 0, "4": 0}
        if (not isinstance(raw, dict) or raw.keys() != {"2", "4"}
                or any(type(n) is not int or n < 0 for n in raw.values())):
            raise compare.Unverified("invalid scoped shape counters")
        return raw

    before, after = counts(previous), counts(stats)
    hits = {k: after[k] - before[k] for k in before}
    prompts = 2 * hits["2"] + 4 * hits["4"]
    if (min(hits.values()) < 0 or sum(hits.values()) != deltas["calls"]
            or prompts != deltas["prompts"] or 444 * prompts != deltas["rows"]
            or (prompts > 0) != (arm == "long-444")):
        raise compare.Unverified("scoped grouped calls, prompts or rows did not match the requested wave")
    return hits


def pairs(report, *, repeats=None):
    # production_compare verifies every served input length, the local tokenizer, placement and
    # artifact stability when its server context exits. Match the two fully verified deployments.
    deployments = {(a["repeat"], a["deployment"]) for a in report["deployments"] if a["prompt_ids_verified"]}
    pairs = []
    for rep in range(report["protocol"]["repeats"] if repeats is None else repeats):
        for width in (8, 32):
            cells = {r["deployment"]: r for r in report["measurements"]
                     if r["repeat"] == rep and r["concurrency"] == width}
            if set(cells) != {"stock", "long-444"} or not all((rep, arm) in deployments for arm in cells):
                raise compare.Unverified("missing verified deployment")
            baseline, trial = cells["stock"], cells["long-444"]
            base_scope, test_scope = (r["profile"]["scope"] for r in (baseline, trial))
            if (not baseline.get("prompt_ids_sha256") or baseline["prompt_ids_sha256"] != trial.get("prompt_ids_sha256")
                    or baseline["request_id_prefix"] != trial["request_id_prefix"]
                    or base_scope["settings_device_sha256"] != test_scope["settings_device_sha256"]
                    or not baseline["quality"] or not trial["quality"]):
                raise compare.Unverified("paired HTTP requests, worker settings or factual cases differ")
            qa, qb = baseline["quality"]["answers"], trial["quality"]["answers"]
            if not qa or qa.keys() != qb.keys():
                raise compare.Unverified("factual case identities differ")
            pairs.append({
                "repeat": rep, "concurrency": width,
                "output_tokens_and_stops_match": baseline["generation"] == trial["generation"],
                "outputs_match": baseline["profile"]["outputs_sha256"] == trial["profile"]["outputs_sha256"],
                "lost_answers": [name for name in qa if qa[name] and not qb[name]],
                "gained_answers": [name for name in qa if qb[name] and not qa[name]],
                "quality_correct": {"stock": sum(qa.values()), "long-444": sum(qb.values())},
                "quality_total": len(qa),
                "long_shape_hits": trial["profile"]["batch_delta"]["long_shape_hits"],
                "long_over_stock": {k: trial[k] / baseline[k] for k in
                                    ("tokens_per_second", "requests_per_second", "ttft_p95_ms")}})
    return pairs
