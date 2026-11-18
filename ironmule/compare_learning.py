"""Use a measured rival gap to select and locally test one GGML admission budget."""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import statistics
import subprocess
import sys
import time

SCHEMA = "ironmule.competitor-response.v1"
WORDS = "gold blue green pink red brown silver orange yellow white black purple violet cyan".split()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def engine_binding(engine, model):
    from . import ggml_engine as ge

    return {"machine": ge.machine_key(), "gguf_sha256": model, "engine_version": ge.ENGINE_VERSION,
            "gpu_layers": engine.gpu_layers, "seqs": engine.seqs, "seq_ctx": engine.seq_ctx, "threads": engine.threads,
            "shared": engine.shared, "cap": engine.cap, "speculate": engine.speculate, "max_rows": engine.max_rows,
            "settings": engine.settings, "prefix_reuse": engine.prefix_reuse, "prompt_budget": ge.PROMPT_BUDGET,
            "pilot": ge.file_sha256(str(engine.pilot.weights_path)) if engine.pilot else None,
            "has_drafter": engine.drafter is not None, "collective": engine.collective is not None,
            "kv_defrag": ge.DEFRAG, "kv_keep": ge.KEEP, "split": ge.SPLIT, "padding": ge.PAD,
            "kg_sha256": ge.file_sha256(ge.KG_LIB) if ge.KG_LIB else None,
            "implementation": digest([ge.file_sha256(str(Path(__file__).with_name(name))) for name in
                                      ("compare_learning.py", "ggml_engine.py", "pilot.py", "speculation.py")])}


def shape(lengths, limit):
    return ",".join(str(n // 16) for n in sorted(lengths)) + "/" + str(limit)


def rule_directory(state, binding):
    return Path(state) / "competitor-response" / digest(binding)


def read_rules(state, binding):
    try:
        record = json.loads((rule_directory(state, binding) / "qualified.json").read_text())
        if record["binding"] == binding:
            return {k: v for k, v in record["budgets"].items() if type(v) is int and 0 < v <= binding["seq_ctx"]}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return {}


def select(report):
    """Require independently verified prompt IDs and concrete, same-device, same-model measurements."""
    if report.get("schema") != "ironmule.compare.v1":
        return None
    arms = {(a["engine"], a["repeat"]): a for a in report.get("arms", [])}
    runs = {(r["engine"], r["repeat"], r["concurrency"]): r for r in report.get("runs", [])}
    gaps = {}
    for (name, repeat, width), rival in runs.items():
        if name not in ("llama_cpp", "vllm") or not 2 <= width <= 16:
            continue
        ours = runs.get(("ironmule", repeat, width))
        if not ours or not ours.get("quality") or not rival.get("quality"):
            continue
        a, b = arms.get(("ironmule", repeat), {}), arms.get((name, repeat), {})
        aid, bid = a.get("identity", {}), b.get("identity", {})
        ta, tb = a.get("tokenization", {}), b.get("tokenization", {})
        model = aid.get("gguf_sha256")
        if (not re.fullmatch("[0-9a-f]{64}", str(model))
                or not re.fullmatch("[0-9a-f]{64}", str(ta.get("digest")))
                or not re.fullmatch("[0-9a-f]{64}", str(aid.get("tokenizer_sha256")))
                or not (ta.get("verified") is True and tb.get("verified") is True and ta["digest"] == tb.get("digest"))
                or not all(aid.get(k) and aid[k] == bid.get(k) for k in ("gguf_sha256", "tokenizer_sha256", "device"))
                or aid.get("autonomous") is not True or aid.get("placement_verified") is not True
                or bid.get("placement_verified") is not True or not (aid.get("version") and bid.get("version"))
                or name == "llama_cpp" and bid.get("baseline_stock") is not True
                or aid["device"] != report.get("device")):
            continue
        lengths = ours.get("performance_prompt_tokens")
        if not isinstance(lengths, list) or len(lengths) < width or lengths != rival.get("performance_prompt_tokens"):
            continue
        rates = [ours["tokens_per_second"], rival["tokens_per_second"], ours["ttft_p95_ms"], rival["ttft_p95_ms"]]
        if not all(type(v) in (float, int) and 0 < v < math.inf for v in rates):
            continue
        if not all(type(n) is int and n > 0 for n in lengths):
            continue
        outputs = ours.get("generation")
        if (not isinstance(outputs, list) or len(outputs) != len(lengths) or outputs != rival.get("generation")
                or not all(isinstance(v, list) and len(v) == 2 and type(v[0]) is int and v[0] > 0
                           and v[1] in ("stop", "length") for v in outputs)):
            continue
        key = (name, width, model, digest(aid.get("ggml")))
        gaps.setdefault(key, []).append((rates[0] / rates[1], rates[2] / rates[3], lengths[:width], aid.get("ggml")))
    ranked = []
    for (name, width, model, _), observations in gaps.items():
        rate = statistics.median(v[0] for v in observations)
        latency = statistics.median(v[1] for v in observations)
        gap = max(1.03 / rate, latency)
        if gap <= 1:
            continue
        ranked.append((gap, {"opponent": name, "concurrency": width, "model": model, "throughput_ratio": rate,
                             "p95_ttft_ratio": latency, "paired_repeats": len(observations),
                             "focus": "ttft" if latency > 1.03 / rate else "throughput",
                             "prompt_lengths": observations[0][2], "binding": observations[0][3]}))
    return max(ranked, key=lambda row: row[0])[1] if ranked else None


def _sample(engine, tasks, limit, budget, key):
    from .ggml_engine import Job

    engine._clear()
    engine.competitor_budgets = {key: budget} if budget else {}
    engine.competitor_budget_hits = 0
    steps = []
    engine.observer = lambda rows, prompt, kv, seconds: steps.append((rows, prompt))
    jobs = [Job(ids, limit) for ids, _ in tasks]
    start = time.monotonic_ns()
    engine.run(jobs)
    elapsed = (time.monotonic_ns() - start) / 1e9
    if not all(j.reason in ("eos", "length") and j.first_ns > start for j in jobs):
        raise ValueError("native holdout did not complete")
    ttft = sorted((j.first_ns - start) / 1e6 for j in jobs)
    outputs = [(j.tokens, j.reason) for j in jobs]
    correct = []
    for job, (_, word) in zip(jobs, tasks):
        # The first generated word must answer the held-out copy task.
        ids = (engine.L.llama_token * len(job.tokens))(*job.tokens)
        buf = ctypes.create_string_buffer(4096)
        used = engine.L.llama_detokenize(engine.vocab, ids, len(ids), buf, len(buf), True, False)
        if used < 0:
            raise ValueError("unreadable holdout answer")
        first = re.search(r"\w+", buf.raw[:used].decode("utf8", errors="replace").casefold())
        correct.append(bool(first and first.group() == word))
    return {"tokens_per_second": sum(len(j.tokens) for j in jobs) / elapsed, "median_ttft_ms": statistics.median(ttft),
            "p95_ttft_ms": ttft[math.ceil(.95 * len(ttft)) - 1], "output_digest": digest(outputs),
            "correct": sum(correct), "requests": len(jobs), "budget_applied": engine.competitor_budget_hits,
            "execution_shape": digest(steps)}


def trial(payload):
    import random
    from .ggml_engine import GgmlEngine, file_sha256, ggml_devices

    b, gguf = payload["binding"], payload["gguf"]
    if file_sha256(gguf) != b["gguf_sha256"]:
        raise ValueError("GGUF changed")
    opened = {k: b[k] for k in ("threads", "seqs", "seq_ctx", "speculate", "shared", "cap", "max_rows")}
    engine = GgmlEngine(gguf, **opened, **b["settings"])
    try:
        provider = payload["provider"]
        if engine_binding(engine, b["gguf_sha256"]) != b:
            raise ValueError("current native engine differs from measured incumbent")
        if provider == "cpu" and engine.gpu_layers or (provider != "cpu" and
                (not engine.gpu_layers or [d["provider"] for d in ggml_devices(engine.L)] != [provider])):
            raise ValueError("local native backend differs from comparison")
        rng = random.Random(payload["seed"])
        lengths, limit = payload["lengths"], payload["limit"]
        prefix = engine.tokenize(" Notes for the copy task. " * max(lengths))
        suffixes = {word: engine.tokenize(f"\nColor: red\nCopy: red\nColor: blue\nCopy: blue\nColor: {word}\nCopy:")
                    for word in WORDS}
        repeats = []
        for rep in range(4):
            tasks = []
            for length in lengths:
                word = rng.choice(WORDS)
                suffix = suffixes[word]
                if length < len(suffix):
                    raise ValueError("prompt too short for an independent copy task")
                tasks.append((prefix[:length - len(suffix)] + suffix, word))
            order = ("incumbent", "candidate") if rep % 2 == 0 else ("candidate", "incumbent")
            pair = {}
            for arm in order:
                # Each arm starts with freshly initialized speculation, prefix and Pilot state.
                if engine is None:
                    engine = GgmlEngine(gguf, **opened, **b["settings"])
                pair[arm] = _sample(engine, tasks, limit, payload[arm], payload["key"])
                engine.close()
                engine = None
            repeats.append({"order": order, **pair})
        return repeats
    finally:
        if engine is not None:
            engine.close()


def rejections(pairs):
    reasons = set()
    if len(pairs) != 4:
        reasons.add("incomplete_holdout")
    for p in pairs:
        a, b = p["incumbent"], p["candidate"]
        if not (a["budget_applied"] or b["budget_applied"]) or a["execution_shape"] == b["execution_shape"]:
            reasons.add("unchanged_execution")
        if a["output_digest"] != b["output_digest"] or min(a["correct"], b["correct"]) != a["requests"]:
            reasons.add("quality")
        if b["tokens_per_second"] < 1.03 * a["tokens_per_second"]:
            reasons.add("throughput_margin")
        if b["median_ttft_ms"] > a["median_ttft_ms"] or b["p95_ttft_ms"] > a["p95_ttft_ms"]:
            reasons.add("first_token_regression")
    return sorted(reasons)


def respond(report, *, gguf, state, provider, deadline):
    """Bounded child owns the native context. Only a complete local holdout can change the scoped rule."""
    from ironmule_product.state import _atomic_write, _ensure_directory

    choice = select(report)
    result = {"schema": SCHEMA, "status": "deferred", "selection": choice}
    if not choice:
        return {**result, "reason": "no verified rival gap at a supported concurrent workload"}
    b = choice["binding"]
    if (not b or b.get("gguf_sha256") != choice["model"] or b.get("has_drafter")
            or b.get("prompt_budget") != 0 or provider not in ("cpu", "cuda", "rocm")):
        return {**result, "reason": "missing or unsupported native serving configuration"}
    lengths, limit = choice["prompt_lengths"], report["protocol"]["max_tokens"]
    if type(limit) is not int or not 1 <= limit <= 128 or max(lengths) + limit > b["seq_ctx"]:
        return {**result, "reason": "workload outside supported context"}
    key = shape(lengths, limit)
    directory = rule_directory(state, b)
    for path in (Path(state), directory.parent, directory):
        _ensure_directory(path)
    with (directory / "trial.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {**result, "reason": "another local trial is running"}
        budgets = read_rules(state, b)
        baseline = budgets.get(key, 0)
        source = digest({k: report.get(k) for k in ("schema", "device", "artifacts", "protocol", "runs")})
        history = directory / (digest([source, key]) + ".json")
        if history.exists():
            return json.loads(history.read_text())
        # Try an untested budget before repeating a losing experiment in the same workload.
        n = len(lengths)
        options = [sum(sorted(lengths)[:max(1, n // 2)]), min(lengths)]
        if choice["focus"] == "ttft":
            options.reverse()
        if baseline:
            options.append(0)
        counts = {v: 0 for v in options if v != baseline and v < b["seq_ctx"]}
        for path in directory.glob("*.json"):
            old = json.loads(path.read_text())
            option = old.get("candidate_budget")
            if old.get("workload_key") == key and old.get("incumbent_budget") == baseline and option in counts:
                counts[option] += 1
        candidate = min(counts, key=counts.get, default=None)
        if candidate is None:
            return {**result, "reason": "no distinct bounded prompt budget"}
        result.update(source_sha256=source, workload_key=key, incumbent_budget=baseline, candidate_budget=candidate)
        payload = {"binding": b, "gguf": str(gguf), "lengths": lengths, "limit": limit, "provider": provider,
                   "incumbent": baseline, "candidate": candidate, "key": key, "seed": secrets.randbits(64)}
        result["holdout_seed"] = payload["seed"]
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {**result, "reason": "comparison budget exhausted"}
            child = subprocess.run([sys.executable, "-m", "ironmule.compare_learning"], input=json.dumps(payload),
                                   text=True, capture_output=True, timeout=remaining)
            if child.returncode:
                return {**result, "reason": "local trial unavailable", "detail": child.stderr.splitlines()[-1][:300]
                        if child.stderr else str(child.returncode)}
            pairs = json.loads(child.stdout)
        except subprocess.TimeoutExpired:
            return {**result, "reason": "comparison budget exhausted"}
        reasons = rejections(pairs)
        passed = not reasons
        result.update(status="qualified" if passed else "retained_incumbent", reasons=reasons, holdout=pairs)
        # Keep the measured evidence before exposing a winner to any worker restarting concurrently.
        _atomic_write(history, result)
        if passed:
            if candidate:
                budgets[key] = candidate
            else:
                budgets.pop(key, None)
            _atomic_write(directory / "qualified.json", {"binding": b, "budgets": budgets})
        return result


if __name__ == "__main__":
    print(json.dumps(trial(json.load(sys.stdin)), allow_nan=False))
