"""Leave one full native HTTP repeat out when testing split vs grouped prompt processing."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

ARMS = ("stock", "batch")  # Both are IronMule.
MIN_RATE = 1.03


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def require(value, message):
    if not value:
        raise ValueError(message)


def positive(value):
    return type(value) in (int, float) and 0 < value < math.inf


def index_unique(rows, fields):
    result = {}
    for row in rows:
        if row.get("deployment") not in ARMS:
            continue
        key = tuple(row[k] for k in fields)
        require(key not in result, "duplicate_measurement")
        result[key] = row
    return result


def answers(result, seed):
    quality = result["quality"]
    expected = {f"{seed}:{category}-{i}" for category in ("arithmetic", "extraction", "facts") for i in range(6)}
    values = quality["answers"]
    require(values.keys() == expected and all(type(v) is bool for v in values.values())
            and quality["total"] == len(expected) and quality["passed"] == sum(values.values()), "invalid_factual_scores")
    return values


def arm_values(result, deployment, seed, count, max_tokens):
    identity = deployment["identity"]
    profile = result["profile"]
    implementation = identity.get("implementation_sha256")
    require(implementation is None or sha(implementation), "invalid_implementation")
    require(deployment.get("prompt_ids_verified") is True and sha(deployment.get("prompt_token_digest"))
            and identity.get("placement_verified") is True and identity.get("autonomous") is True
            and identity.get("provider") == "cuda", "unverified_deployment")
    common = [identity.get(k) for k in ("version", "provider", "device_name", "gguf_sha256", "tokenizer_sha256")]
    require(all(isinstance(v, str) and v for v in common) and all(sha(v) for v in common[-2:]),
            "unverified_model_device")
    common.append(implementation)
    require(profile["samples"] == result["requests"] == count and profile["shared"] is False
            and type(profile["context"]) is int and type(profile["sequences"]) is int
            and profile["context"] > max_tokens and profile["sequences"] >= 1, "invalid_execution_layout")
    lengths = result["performance_prompt_tokens"]
    require(len(lengths) == count and all(type(n) is int and 0 < n <= profile["context"] - max_tokens for n in lengths),
            "invalid_prompt_work")
    generation = result["generation"]
    require(len(generation) == count and all(isinstance(x, list) and len(x) == 2
            and type(x[0]) is int and 1 <= x[0] <= max_tokens and x[1] in ("length", "stop") for x in generation),
            "invalid_generation")
    require(sum(x[0] for x in generation) == result["completion_tokens"] and positive(result["wall_s"])
            and positive(result["tokens_per_second"]) and positive(result["ttft_p95_ms"])
            and math.isclose(result["tokens_per_second"], result["completion_tokens"] / result["wall_s"], rel_tol=1e-5),
            "invalid_timing")
    scores = answers(result, seed)
    ids = result["prompt_ids_sha256"]
    require(len(ids) == count + len(scores) and all(sha(v) for v in ids)
            and sha(profile["outputs_sha256"]), "invalid_token_proof")
    counters = profile["batch_delta"]
    require(all(type(counters[k]) is int and counters[k] >= 0 for k in ("calls", "prompts", "rows")),
            "invalid_dispatch_counter")
    scope = profile.get("scope")
    if scope is not None:
        require(all(sha(scope.get(k)) for k in ("settings_device_sha256", "arm_sha256", "implementation_sha256", "native_runtime_sha256"))
                and scope["implementation_sha256"] == identity.get("implementation_sha256"), "invalid_worker_scope")
    return common, scope, scores


def pair(rep, width, rows, deployments, order, protocol):
    data = [rows[(arm, rep, width)] for arm in ARMS]
    arms = [deployments[(arm, rep)] for arm in ARMS]
    count = max(width, protocol["requests"])
    seed = protocol["seed"] + rep * 1009 + width
    a, b = [arm_values(r, d, seed, count, protocol["max_tokens"]) for r, d in zip(data, arms)]
    split, grouped = data
    ap, bp = split["profile"], grouped["profile"]
    require(a[0] == b[0] and arms[0]["prompt_token_digest"] == arms[1]["prompt_token_digest"],
            "different_deployment_or_prompt_proof")
    require(split["request_id_prefix"] == grouped["request_id_prefix"] == f"qwen3-{protocol['seed']}-{rep}-{width}"
            and split["prompt_ids_sha256"] == grouped["prompt_ids_sha256"]
            and split["performance_prompt_tokens"] == grouped["performance_prompt_tokens"], "different_input")
    require(all(ap[k] == bp[k] for k in ("context", "sequences", "schedule"))
            and ap["context"] == protocol["context"], "different_scheduler_or_context")
    require(not any(ap["batch_delta"][k] for k in ("calls", "rows", "prompts")), "split_executed_grouping")
    scope = None
    if a[1] is not None or b[1] is not None:
        require(a[1] and b[1] and a[1]["settings_device_sha256"] == b[1]["settings_device_sha256"]
                and a[1]["native_runtime_sha256"] == b[1]["native_runtime_sha256"],
                "different_worker_scope")
        scope = [a[1]["settings_device_sha256"], a[1]["arm_sha256"], b[1]["arm_sha256"]]
    histogram = sorted(Counter(v.bit_length() - 1 for v in split["performance_prompt_tokens"]).items())
    work = {"concurrency": width, "prompt_length_log2_bins": histogram, "requests": count,
            "max_tokens": protocol["max_tokens"], "context": ap["context"], "sequences": ap["sequences"],
            "scope": "worker_verified" if scope else "single_report_only"}
    hits = bp["batch_delta"]
    rate = grouped["tokens_per_second"] / split["tokens_per_second"]
    p95 = grouped["ttft_p95_ms"] / split["ttft_p95_ms"]
    reasons = []
    if not (hits["calls"] > 0 and hits["prompts"] >= 2 * hits["calls"] and hits["rows"] >= hits["prompts"]):
        reasons.append("no_verified_grouped_work")
    if split["generation"] != grouped["generation"] or ap["outputs_sha256"] != bp["outputs_sha256"]:
        reasons.append("different_output")
    by_category = {}
    for category in ("arithmetic", "extraction", "facts"):
        keys = [k for k in a[2] if f":{category}-" in k]
        by_category[category] = {"total": len(keys), "split": sum(a[2][k] for k in keys),
                                 "grouped": sum(b[2][k] for k in keys),
                                 "lost": sum(a[2][k] and not b[2][k] for k in keys)}
    if any(category["lost"] for category in by_category.values()):
        reasons.append("lost_correct_answer")
    if any(not category["split"] for category in by_category.values()):
        reasons.append("uninformative_reference_category")
    if rate < MIN_RATE:
        reasons.append("throughput_margin")
    if p95 > 1:
        reasons.append("p95_regression")
    binding = dict(zip(("version", "provider", "device_name", "gguf_sha256", "tokenizer_sha256", "implementation_sha256"), a[0]))
    if scope:
        binding.update(settings_device_sha256=scope[0], split_arm_sha256=scope[1], grouped_arm_sha256=scope[2],
                       implementation_sha256=a[1]["implementation_sha256"],
                       native_runtime_sha256=a[1]["native_runtime_sha256"])
    return {"repeat": rep, "workload": work, "binding": binding,
            "binding_sha256": checksum([binding, ap["schedule"]]),
            "grouped_first": order.index(("batch", rep)) < order.index(("stock", rep)),
            "timed_prompt_hashes": split["prompt_ids_sha256"][:count],
            "performance_trace_sha256": checksum(split["prompt_ids_sha256"][:count]),
            "throughput_ratio": rate, "p95_ratio": p95, "batch_delta": hits,
            "tokens_per_second": {"split": split["tokens_per_second"], "grouped": grouped["tokens_per_second"]},
            "ttft_p95_ms": {"split": split["ttft_p95_ms"], "grouped": grouped["ttft_p95_ms"]},
            "quality_by_category": by_category,
            "split_correct": sum(a[2].values()), "grouped_correct": sum(b[2].values()), "reasons": reasons}


def folds(rows):
    out = []
    for test in rows:
        train = [x for x in rows if x["repeat"] != test["repeat"]]
        learned = (len(train) >= 2 and len({x["grouped_first"] for x in train}) == 2
                   and all(not x["reasons"] for x in train))
        out.append({"test_repeat": test["repeat"], "training_repeats": [x["repeat"] for x in train],
                    "choice": "grouped" if learned else "split",
                    "test_throughput_ratio_vs_split": test["throughput_ratio"] if learned else 1.0,
                    "test_p95_ratio_vs_split": test["p95_ratio"] if learned else 1.0,
                    "test_rejections": test["reasons"] if learned else []})
    return out


def evaluate(report):
    require(report.get("schema") == "ironmule.prefill-http.v1", "requires_native_prefill_http_report")
    out = {"schema": "ironmule.prefill-selection.v1", "status": "insufficient_evidence", "groups": [], "excluded": []}
    if report.get("status") not in ("measured", "partial", "incomplete") or report.get("stage") != "finished":
        out["reason"] = "unfinished_or_failed_native_measurement"
        return out
    protocol = report["protocol"]
    require(1 <= protocol["repeats"] <= 9 and set(ARMS) <= set(protocol["arms"]), "missing_native_prefill_protocol")
    require(type(protocol["requests"]) is int and 1 <= protocol["requests"] <= 128
            and type(protocol["max_tokens"]) is int and 1 <= protocol["max_tokens"] <= 2048
            and type(protocol["seed"]) is int and 0 <= protocol["seed"] < 2**32
            and len(set(protocol["concurrency"])) == len(protocol["concurrency"])
            and all(type(w) is int and 1 <= w <= 64 for w in protocol["concurrency"]), "invalid_protocol")
    rows = index_unique(report["measurements"], ("deployment", "repeat", "concurrency"))
    deployments = index_unique(report["deployments"], ("deployment", "repeat"))
    out["native_arms"] = {
        name: {"timed_cells": sum(key[0] == name for key in rows),
               "verified_repeats": sorted(rep for (arm, rep), row in deployments.items()
                                          if arm == name and row.get("prompt_ids_verified") is True)}
        for name in ARMS}
    known_errors = {"actual shared-pool placement differs from requested arm": "requested_kv_layout_unavailable",
                    "actual engine toggle differs from requested arm": "grouped_path_unavailable",
                    "separate KV streams are unverified": "unverified_split_streams",
                    "missing same-worker prefill identity": "unverified_worker",
                    "execution configuration changed during the measured wave": "worker_changed"}
    out["arm_failures"] = [{"deployment": row["deployment"], "repeat": row["repeat"],
                            "reason": known_errors.get(row.get("reason"), "measurement_failure")}
                           for row in report.get("errors", []) if row.get("deployment") in ARMS]
    if report["status"] == "incomplete":
        out["reason"] = "unfinished_or_failed_native_measurement"
        out["verified_cells"] = []
        for (arm, rep, width), row in rows.items():
            deployment = deployments.get((arm, rep))
            if not deployment or deployment.get("prompt_ids_verified") is not True:
                continue
            try:
                identity, _, scores = arm_values(row, deployment, protocol["seed"] + rep * 1009 + width,
                                                 max(width, protocol["requests"]), protocol["max_tokens"])
            except (ValueError, KeyError, TypeError, IndexError):
                continue
            out["verified_cells"].append({
                "deployment": arm, "repeat": rep, "concurrency": width, "device_name": identity[2],
                "gguf_sha256": identity[3], "tokenizer_sha256": identity[4], "requests": row["requests"],
                "tokens_per_second": row["tokens_per_second"], "ttft_p95_ms": row["ttft_p95_ms"],
                "correct": sum(scores.values()), "questions": len(scores)})
        return out
    groups = defaultdict(list)
    for width in protocol["concurrency"]:
        for rep in range(protocol["repeats"]):
            try:
                row = pair(rep, width, rows, deployments, list(deployments), protocol)
                groups[checksum([row["binding_sha256"], row["workload"]])].append(row)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                reason = str(exc) if isinstance(exc, ValueError) else "missing_or_invalid_measurement"
                out["excluded"].append({"repeat": rep, "concurrency": width, "reason": reason})
    for members in groups.values():
        trace_shapes = [set(x.pop("timed_prompt_hashes")) for x in members]
        fresh = sum(map(len, trace_shapes)) == len(set.union(*trace_shapes))
        scope = members[0]["binding"]
        for member in members:
            del member["binding"]
        tested = folds(members) if fresh else []
        proposed = [x for x in tested if x["choice"] == "grouped"]
        group = {"workload": members[0]["workload"], "binding": scope, "pair_evidence": members, "folds": tested,
                 "decision": "retain_split"}
        if not fresh or len(members) < 3:
            group["decision"] = "insufficient_independent_rounds"
        elif (len(proposed) >= 2 and all(not x["reasons"] for x in members)
              and all(not f["test_rejections"] for f in proposed)):
            group["decision"] = ("candidate_for_new_local_holdout" if group["workload"]["scope"] == "worker_verified"
                                 else "candidate_requires_bound_remeasurement")
        out["groups"].append(group)
    if not groups:
        out["reason"] = "no_verified_split_grouped_pairs"
    if any(len(g["folds"]) >= 3 for g in out["groups"]):
        out["status"] = "cross_validated_offline"
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    blob = args.report.read_bytes()
    result = evaluate(json.loads(blob))
    result["source_sha256"] = hashlib.sha256(blob).hexdigest()
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
