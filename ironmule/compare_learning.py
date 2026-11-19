"""Learn GGML prompt admission from repeated native request cohorts and local holdouts."""
from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import statistics
import subprocess
import sys
import time

SCHEMA = "ironmule.self-admission.v1"
WORDS = "gold blue green pink red brown silver orange yellow white black purple violet cyan".split()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def implementation():
    names = ("compare_learning.py", "ggml_engine.py", "pilot.py", "pilot_learning.py", "speculation.py")
    return digest([hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names])


def engine_binding(engine, model):
    from . import ggml_engine as ge

    host = None
    for path in (Path("/etc/machine-id"), Path("/proc/sys/kernel/random/boot_id")):
        try:
            value = path.read_text().strip()
        except OSError:
            continue
        if value:
            host = digest(value)
            break
    provider, cuda_uuid = "cpu", None
    if engine.gpu_layers:
        devices = ge.ggml_devices(engine.L)
        provider = devices[0]["provider"] if len(devices) == 1 else None
        if provider == "cuda":
            from .kernel_gen import _visible_cuda_devices
            visible = _visible_cuda_devices()
            cuda_uuid = visible[0]["uuid"] if len(visible) == 1 else None
    threads_head = getattr(engine.pilot, "learning", None)
    return {"thread_weights": dict(threads_head.weights) if threads_head is not None else None,
            "machine": ge.machine_key(), "host_sha256": host,
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "gguf_sha256": model, "engine_version": ge.ENGINE_VERSION, "provider": provider, "cuda_uuid": cuda_uuid,
            "gpu_layers": engine.gpu_layers, "seqs": engine.seqs, "seq_ctx": engine.seq_ctx, "threads": engine.threads,
            "shared": engine.shared, "cap": engine.cap, "speculate": engine.speculate, "max_rows": engine.max_rows,
            "settings": engine.settings, "prefix_reuse": engine.prefix_reuse, "prompt_budget": ge.PROMPT_BUDGET,
            "pilot": ge.file_sha256(str(engine.pilot.weights_path)) if engine.pilot else None,
            "has_drafter": engine.drafter is not None, "collective": engine.collective is not None,
            "kv_defrag": ge.DEFRAG, "kv_keep": ge.KEEP, "split": ge.SPLIT, "padding": ge.PAD,
            "kg_sha256": ge.file_sha256(ge.KG_LIB) if ge.KG_LIB else None,
            "implementation": implementation()}


def shape(lengths, limit):
    return ",".join(str(n // 16) for n in sorted(lengths)) + "/" + str(limit)


@dataclass(frozen=True)
class Observation:
    """One completed, simultaneously admitted cohort. Contains no tokens, text or external measurements."""

    sample_id: int
    prompt_lengths: tuple[int, ...]
    max_tokens: int
    tokens_per_second: float
    ttft_p95_ms: float


def rule_directory(state, binding):
    return Path(state) / "self-admission" / digest(binding)


def _ledger_path(directory, key):
    return directory / ("workload-" + digest([SCHEMA, key]) + ".json")


def _valid_record(record, binding, key):
    return (isinstance(record, dict) and record.get("schema") == SCHEMA and record.get("binding") == binding
            and record.get("workload_key") == key and isinstance(record.get("trials"), dict))


def _measured(result, source, key):
    return (isinstance(result, dict) and result.get("schema") == SCHEMA and result.get("source_sha256") == source
            and result.get("workload_key") == key and isinstance(result.get("selection"), dict)
            and result["selection"].get("source_sha256") == source and isinstance(result.get("holdout"), list)
            and result.get("status") in ("retained_incumbent", "qualified"))


def _active_budget(record):
    source = record.get("active_source")
    decision = record["trials"].get(source)
    if (_measured(decision, source, record["workload_key"]) and decision["status"] == "qualified"
            and not rejections(decision["holdout"]) and type(decision.get("candidate_budget")) is int):
        return decision["candidate_budget"]
    return 0


def read_rules(state, binding):
    rules = {}
    if not binding.get("host_sha256") or binding.get("implementation") != implementation():
        return rules
    directory = rule_directory(state, binding)
    for path in directory.glob("workload-*.json"):
        try:
            record = json.loads(path.read_text())
            key = record["workload_key"]
            if not isinstance(key, str) or path != _ledger_path(directory, key) or not _valid_record(record, binding, key):
                continue
            budget = _active_budget(record)
            if 0 < budget < binding["seq_ctx"]:
                rules[key] = budget
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return rules


def _positive_number(value):
    return type(value) in (float, int) and 0 < value < math.inf


def select(observations, *, binding):
    """Rank recurring admission queues by their observed aggregate first-token delay."""
    if not isinstance(observations, (tuple, list)) or not isinstance(binding, dict):
        return None
    groups, seen = {}, set()
    for row in observations:
        if (type(row) is not Observation or type(row.sample_id) is not int or row.sample_id < 0
                or row.sample_id in seen):
            continue
        lengths, limit = row.prompt_lengths, row.max_tokens
        if (type(lengths) is not tuple or not 2 <= len(lengths) <= 64 or type(limit) is not int or limit < 1
                or any(type(n) is not int or not 0 < n <= binding["seq_ctx"] - limit for n in lengths)
                or not _positive_number(row.tokens_per_second) or not _positive_number(row.ttft_p95_ms)):
            continue
        if len(lengths) < 4 and 2 * max(lengths) < 3 * min(lengths):
            continue
        seen.add(row.sample_id)
        groups.setdefault(shape(lengths, limit), []).append(row)
    recurring = [rows for rows in groups.values() if len(rows) >= 2]
    if not recurring:
        return None

    def queue_cost(rows):
        return (len(rows[0].prompt_lengths) * statistics.median(r.ttft_p95_ms for r in rows),
                len(rows), shape(rows[0].prompt_lengths, rows[0].max_tokens))

    rows = sorted(max(recurring, key=queue_cost), key=lambda r: r.sample_id)
    last = rows[-1]
    return {"focus": "admission", "samples": len(rows), "concurrency": len(last.prompt_lengths),
            "prompt_lengths": sorted(last.prompt_lengths), "max_tokens": last.max_tokens,
            "tokens_per_second": statistics.median(r.tokens_per_second for r in rows),
            "ttft_p95_ms": statistics.median(r.ttft_p95_ms for r in rows),
            "source_sha256": digest([asdict(r) for r in rows])}


def _sample(engine, tasks, limit, budget, key):
    from .ggml_engine import Job

    engine._clear()
    engine.admission_budgets = {key: budget} if budget else {}
    engine.admission_budget_hits = 0
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
            buf = ctypes.create_string_buffer(-used)
            used = engine.L.llama_detokenize(engine.vocab, ids, len(ids), buf, len(buf), True, False)
        if used < 0:
            raise ValueError("unreadable holdout answer")
        first = re.search(r"\w+", buf.raw[:used].decode("utf8", errors="replace").casefold())
        correct.append(bool(first and first.group() == word))
    return {"tokens_per_second": sum(len(j.tokens) for j in jobs) / elapsed, "median_ttft_ms": statistics.median(ttft),
            "p95_ttft_ms": ttft[math.ceil(.95 * len(ttft)) - 1], "output_digest": digest(outputs),
            "correct": sum(correct), "requests": len(jobs), "budget_applied": engine.admission_budget_hits,
            "execution_shape": digest(steps)}


class BindingMismatch(ValueError):
    def __init__(self, actual, expected):
        # Names come from engine_binding(), never from report keys or paths.
        self.fields = sorted(k for k in actual if k not in expected or expected[k] != actual[k])
        if expected.keys() - actual.keys():
            self.fields.append("unexpected_field")
        super().__init__("current native engine differs from measured incumbent: " + ", ".join(self.fields))


def trial(payload):
    import random

    b, gguf = payload["binding"], payload["gguf"]
    # The fresh child inherits the calling IronMule worker's visibility and loaded-kernel
    # environment. Verify the native device and the entire engine binding on this host.
    os.environ["IRONMULE_GGML_SEQS"] = str(b["seqs"])
    os.environ["IRONMULE_GGML_SEQ_CTX"] = str(b["seq_ctx"])
    if not b["kg_sha256"]:
        os.environ["IRONMULE_KG_LIB"] = ""
    affinity = b.get("cpu_affinity")
    if affinity and hasattr(os, "sched_setaffinity"):
        if not set(affinity).issubset(os.sched_getaffinity(0)):
            raise ValueError("the serving worker's CPUs are unavailable")
        os.sched_setaffinity(0, affinity)
    from .ggml_engine import GgmlEngine, SETTING_KEYS, _kernel_digest, file_sha256

    if file_sha256(gguf) != b["gguf_sha256"]:
        raise ValueError("GGUF changed")
    if b.get("kg_sha256") and _kernel_digest() != b["kg_sha256"]:
        raise BindingMismatch({"kg_sha256": None}, {"kg_sha256": b["kg_sha256"]})
    opened = {k: b[k] for k in ("threads", "seqs", "seq_ctx", "speculate", "shared", "cap", "max_rows")}
    settings = {k: b["settings"][k] for k in SETTING_KEYS if k in b["settings"]}
    def open_engine():
        engine = GgmlEngine(gguf, **opened, **settings)
        if engine.pilot is not None and b.get("thread_weights") is not None:
            from .pilot_learning import ThreadPolicy
            engine.pilot.learning = ThreadPolicy(engine.pilot, b["thread_weights"])
        return engine

    engine = open_engine()
    try:
        actual = engine_binding(engine, b["gguf_sha256"])
        if actual != b:
            raise BindingMismatch(actual, b)
        rng = random.Random(payload["seed"])
        lengths, limit = payload["lengths"], payload["limit"]
        prefix = engine.tokenize(" Notes for the copy task. " * max(lengths))
        suffixes = {}
        for template in ("\nColor: red\nCopy: red\nColor: blue\nCopy: blue\nColor: {}\nCopy:", "\nRepeat {}:"):
            suffixes = {word: engine.tokenize(template.format(word)) for word in WORDS}
            if max(map(len, suffixes.values())) <= min(lengths):
                break
        else:
            raise ValueError("prompt too short for an independent copy task")
        repeats = []
        for rep in range(4):
            tasks = []
            for length in lengths:
                word = rng.choice(WORDS)
                suffix = suffixes[word]
                tasks.append((prefix[:length - len(suffix)] + suffix, word))
            order = ("incumbent", "candidate") if rep % 2 == 0 else ("candidate", "incumbent")
            pair = {}
            for arm in order:
                # Each arm starts with freshly initialized speculation, prefix and Pilot state.
                if engine is None:
                    engine = open_engine()
                pair[arm] = _sample(engine, tasks, limit, payload[arm], payload["key"])
                engine.close()
                engine = None
            repeats.append({"order": order, **pair})
        return repeats
    finally:
        if engine is not None:
            engine.close()


def rejections(pairs):
    expected = [["incumbent", "candidate"], ["candidate", "incumbent"]] * 2
    if not isinstance(pairs, list) or [p.get("order") if isinstance(p, dict) else None for p in pairs] != expected:
        return ["incomplete_holdout"]
    reasons = set()
    for pair in pairs:
        a, b = pair.get("incumbent"), pair.get("candidate")
        for arm in (a, b):
            if (not isinstance(arm, dict) or not all(_positive_number(arm.get(k)) for k in
                    ("tokens_per_second", "median_ttft_ms", "p95_ttft_ms"))
                    or arm["p95_ttft_ms"] < arm["median_ttft_ms"]
                    or type(arm.get("requests")) is not int or not 2 <= arm["requests"] <= 64
                    or type(arm.get("correct")) is not int or not 0 <= arm["correct"] <= arm["requests"]
                    or type(arm.get("budget_applied")) is not int or arm["budget_applied"] < 0
                    or not all(isinstance(arm.get(k), str) and re.fullmatch("[a-f0-9]{64}", arm[k])
                               for k in ("output_digest", "execution_shape"))):
                return ["invalid_measurement"]
        if not (a["budget_applied"] or b["budget_applied"]) or a["execution_shape"] == b["execution_shape"]:
            reasons.add("unchanged_execution")
        if (a["requests"] != b["requests"] or a["output_digest"] != b["output_digest"]
                or min(a["correct"], b["correct"]) != a["requests"]):
            reasons.add("quality")
        if b["tokens_per_second"] < 1.03 * a["tokens_per_second"]:
            reasons.add("throughput_margin")
        if b["median_ttft_ms"] > a["median_ttft_ms"] or b["p95_ttft_ms"] > a["p95_ttft_ms"]:
            reasons.add("first_token_regression")
    return sorted(reasons)


def _run_child(payload, deadline, interrupted):
    command = [sys.executable, *(["-I"] if sys.flags.isolated else []), "-m", "ironmule.compare_learning"]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True, start_new_session=True) as process:
        data = json.dumps(payload)
        try:
            while True:
                if interrupted is not None and interrupted():
                    raise InterruptedError
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                try:
                    output, _ = process.communicate(data, timeout=min(remaining, 0.1))
                    return process.returncode, output
                except subprocess.TimeoutExpired:
                    data = None
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()


def respond(observations, *, binding, gguf, state, deadline, interrupted=None):
    """Choose from own request cohorts. Atomically publish only after four matched local trials."""
    from ironmule_product.state import _atomic_write, _ensure_directory

    result = {"schema": SCHEMA, "status": "deferred", "selection": None}
    if (not isinstance(binding, dict) or not binding.get("host_sha256")
            or binding.get("implementation") != implementation() or binding.get("has_drafter")
            or binding.get("prompt_budget") != 0 or type(binding.get("seq_ctx")) is not int
            or not re.fullmatch("[0-9a-f]{64}", str(binding.get("gguf_sha256")))
            or binding.get("provider") not in ("cpu", "cuda", "rocm")
            or binding["provider"] == "cuda" and not re.fullmatch("[a-f0-9]{32}", str(binding.get("cuda_uuid")))):
        return {**result, "reason": "missing or unsupported native serving configuration"}
    choice = select(observations, binding=binding)
    result["selection"] = choice
    if not choice:
        return {**result, "reason": "no recurring concurrent admission workload"}
    lengths, limit = choice["prompt_lengths"], choice["max_tokens"]
    key, source = shape(lengths, limit), choice["source_sha256"]
    directory = rule_directory(state, binding)
    for path in (Path(state), directory.parent, directory):
        _ensure_directory(path)
    with (directory / "trial.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {**result, "reason": "another local trial is running"}
        path = _ledger_path(directory, key)
        try:
            record = json.loads(path.read_text())
            if not _valid_record(record, binding, key):
                return {**result, "reason": "invalid local admission history"}
        except FileNotFoundError:
            record = {"schema": SCHEMA, "binding": binding, "workload_key": key, "trials": {}}
        except (OSError, ValueError):
            return {**result, "reason": "unreadable local admission history"}
        trials = record["trials"]
        if _measured(trials.get(source), source, key):
            return trials[source]
        baseline = _active_budget(record)
        tested = {old["candidate_budget"] for trial_source, old in trials.items()
                  if _measured(old, trial_source, key) and old.get("incumbent_budget") == baseline
                  and type(old.get("candidate_budget")) is int}
        options = [min(lengths), sum(lengths[:max(1, len(lengths) // 2)])]
        if baseline:
            options.append(0)
        candidate = next((v for v in options if v != baseline and v < binding["seq_ctx"] and v not in tested), None)
        if candidate is None:
            return {**result, "reason": "candidate budgets already measured for this workload"}
        seed = secrets.randbits(64)
        result.update(source_sha256=source, workload_key=key, incumbent_budget=baseline, candidate_budget=candidate,
                      holdout_seed=seed)
        if interrupted is not None and interrupted():
            return {**result, "reason": "foreground work resumed"}
        if time.monotonic() >= deadline:
            return {**result, "reason": "local trial budget exhausted"}
        payload = {"binding": binding, "gguf": str(gguf), "lengths": lengths, "limit": limit,
                   "incumbent": baseline, "candidate": candidate, "key": key, "seed": seed}
        try:
            code, output = _run_child(payload, deadline, interrupted)
            if interrupted is not None and interrupted():
                raise InterruptedError
            if code == 3:
                fields = json.loads(output).get("binding_mismatch")
                if isinstance(fields, list) and len(fields) <= 40 and all(
                        isinstance(k, str) and re.fullmatch("[a-z0-9_]{1,32}", k) for k in fields):
                    return {**result, "reason": "current native engine differs from measured incumbent",
                            "binding_mismatch": fields}
            if code:
                return {**result, "reason": "local trial unavailable", "exit_code": code}
            pairs = json.loads(output)
        except InterruptedError:
            return {**result, "reason": "foreground work resumed"}
        except TimeoutError:
            return {**result, "reason": "local trial budget exhausted"}
        except (OSError, ValueError):
            return {**result, "reason": "local trial unavailable"}
        reasons = rejections(pairs)
        if not reasons and any(p["incumbent"]["requests"] != len(lengths) for p in pairs):
            reasons = ["invalid_measurement"]
        if "invalid_measurement" in reasons or "incomplete_holdout" in reasons:
            return {**result, "reason": reasons[0]}
        result.update(status="retained_incumbent" if reasons else "qualified", reasons=reasons, holdout=pairs)
        if interrupted is not None and interrupted():
            return {**result, "status": "deferred", "reason": "foreground work resumed"}
        trials[source] = result
        if not reasons:
            record["active_source"] = source
        _atomic_write(path, record)
        return result


if __name__ == "__main__":
    try:
        print(json.dumps(trial(json.load(sys.stdin)), allow_nan=False))
    except BindingMismatch as exc:
        print(json.dumps({"binding_mismatch": exc.fields}))
        sys.exit(3)
