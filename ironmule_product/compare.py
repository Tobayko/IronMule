"""Paired local HTTP comparison of the autonomous IronMule server, llama-server and vLLM."""
from __future__ import annotations

import argparse
import fcntl
import http.client
import json
import math
import os
import platform
import secrets
import statistics
import sys
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .state import ProductStore, _atomic_write

ENGINES = ("ironmule", "llama_cpp", "vllm")
PROTOCOL = "chat-gguf-v1"
WARMUP = "Write two sentences describing how to read a map."


class Unverified(ValueError):
    """The arms cannot support a controlled comparison."""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def conversation(text):
    return [{"role": "user", "content": text}]


def performance_prompts(seed, width, count):
    """A balanced mix of short and document requests; paired arms see identical messages."""
    import random

    rng = random.Random(seed * 997 + width)
    topics = ("bridge inspection", "tree survey", "railway schedule", "weather station")
    prompts = []
    for index in range(count):
        key = rng.randrange(10000, 99999)
        topic = topics[index % len(topics)]
        if index % 2:
            records = " ".join(f"Site {key + j} measured {rng.randrange(20, 80)} units." for j in range(32))
            prompt = f"Field notes on {topic}. {records} Summarize the variation in four complete sentences."
        else:
            prompt = f"Assignment {key}. Describe the work involved in a {topic} in four complete sentences."
        prompts.append(prompt)
    return prompts


def workloads(args, repeat):
    from ironmule.factual_gate import cases

    data = {}
    for width in args.concurrency:
        seed = args.seed + repeat * 1009 + width
        perf = performance_prompts(seed, width, max(width, args.requests))
        quality = cases(seed)
        data[width] = (perf, quality)
    return data


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("comparison budget exhausted")
    return value


def stream(engine, server, prompt, limit, deadline):
    """Require successful OpenAI SSE termination and server-generated token counts in each arm."""
    parsed = urllib.parse.urlsplit(server["base_url"])
    if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("comparison launch must provide a local HTTP endpoint")
    body = {"model": server["model_id"], "messages": conversation(prompt), "stream": True,
            "max_tokens": limit, "temperature": 0, "top_p": 1}
    if engine != "ironmule":  # IronMule always includes usage in the final streaming event.
        body["stream_options"] = {"include_usage": True}
    if engine == "vllm":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    started = time.perf_counter()
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=min(remaining(deadline), 120))
    first = usage = finish = None
    pieces = []
    try:
        conn.connect()
        connected_socket = conn.sock  # HTTPConnection drops its own reference on Connection: close.
        conn.request("POST", parsed.path.rstrip("/") + "/v1/chat/completions", json.dumps(body),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"{engine} chat endpoint returned HTTP {response.status}: {response.read(500)!r}")
        while True:
            connected_socket.settimeout(min(remaining(deadline), 120))
            raw = response.readline(1 << 20)
            if not raw:
                raise RuntimeError(f"{engine} stream ended without [DONE]")
            if len(raw) >= 1 << 20:
                raise RuntimeError("oversized SSE line")
            line = raw.strip()
            if line == b"data: [DONE]":
                break
            if line.startswith(b"data:"):
                chunk = json.loads(line[5:])
                if chunk.get("error"):
                    raise RuntimeError(f"{engine} returned a stream error")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    text = choice.get("delta", {}).get("content") or ""
                    if text:
                        if first is None:
                            first = time.perf_counter() - started
                        pieces.append(text)
                    finish = choice.get("finish_reason") or finish
        if finish not in ("stop", "length") or first is None or not usage:
            raise RuntimeError(f"{engine} did not return visible content, finish reason and usage")
        n, p = usage.get("completion_tokens"), usage.get("prompt_tokens")
        if type(n) is not int or not 0 < n <= limit or type(p) is not int or p <= 0:
            raise RuntimeError(f"{engine} returned invalid token counts")
        return {"text": "".join(pieces), "tokens": n, "prompt_tokens": p, "ttft_ms": 1000 * first, "finish_reason": finish}
    finally:
        conn.close()


def batch(engine, server, prompts, width, limit, deadline):
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=width) as pool:
        futures = [pool.submit(stream, engine, server, prompt, limit, deadline) for prompt in prompts]
        rows = [future.result(timeout=remaining(deadline)) for future in futures]
    return rows, time.perf_counter() - started


def check_usage(prompts, rows, token_counts):
    for index, (prompt, row) in enumerate(zip(prompts, rows, strict=True)):
        if token_counts[prompt] != row["prompt_tokens"]:
            raise Unverified(f"served prompt {index} used {row['prompt_tokens']} tokens, "
                             f"but token verification expected {token_counts[prompt]}")


def evaluate(report):
    protocol = report["protocol"]
    runs = {(r["engine"], r["repeat"], r["concurrency"]): r for r in report["runs"]}
    arms = {(a["engine"], a["repeat"]): a for a in report.get("arms", [])}
    engines = protocol["engines"]
    checks, reasons = [], set()
    if protocol["repeats"] < 3 or len(protocol["concurrency"]) < 2:
        reasons.add("insufficient_repeats_or_loads")
    for rep in range(protocol["repeats"]):
        for width in protocol["concurrency"]:
            tested = runs.get(("ironmule", rep, width))
            for engine in engines[1:]:
                reference = runs.get((engine, rep, width))
                if not tested or not reference or not tested.get("quality") or not reference.get("quality"):
                    reasons.add("missing_arm")
                    continue
                ratio = tested["tokens_per_second"] / reference["tokens_per_second"]
                latency = tested["ttft_p95_ms"] / reference["ttft_p95_ms"]
                scores, control = tested["quality"]["answers"], reference["quality"]["answers"]
                lost = [k for k in control if control[k] and not scores.get(k)]
                reasons_here = []
                candidate_identity = arms.get(("ironmule", rep), {})
                opponent_identity = arms.get((engine, rep), {})
                if not candidate_identity.get("tokenization", {}).get("digest") or (
                        candidate_identity["tokenization"]["digest"] != opponent_identity.get("tokenization", {}).get("digest")):
                    reasons_here.append("prompt_ids_unmatched")
                if not all(a.get("identity", {}).get("placement_verified") is True
                           for a in (candidate_identity, opponent_identity)):
                    reasons_here.append("unverified_device_placement")
                if engine == "llama_cpp" and opponent_identity.get("identity", {}).get("baseline_stock") is not True:
                    reasons_here.append("stock_provenance_unverified")
                if tested["generation"] != reference["generation"]:
                    reasons_here.append("generated_lengths_or_stops_differ")
                if ratio < 1.03:
                    reasons_here.append("throughput_margin")
                if latency > 1:
                    reasons_here.append("first_token_regressed")
                if tested["requests_per_second"] < reference["requests_per_second"]:
                    reasons_here.append("request_rate_regressed")
                if lost or scores.keys() != control.keys():
                    reasons_here.append("quality_regressed")
                if not all(any(v for k, v in control.items() if f":{category}-" in k)
                           for category in ("facts", "arithmetic", "extraction")):
                    reasons_here.append("uninformative_quality_reference")
                if tested["performance_prompt_tokens"] != reference["performance_prompt_tokens"] or (
                        tested["quality"]["prompt_tokens"] != reference["quality"]["prompt_tokens"]):
                    reasons_here.append("served_prompt_lengths_differ")
                checks.append({"opponent": engine, "repeat": rep, "concurrency": width, "throughput_ratio": ratio,
                               "ttft_p95_ratio": latency, "lost_answers": lost, "reasons": reasons_here})
                reasons.update(reasons_here)
    incomplete = reasons.intersection({"missing_arm", "insufficient_repeats_or_loads", "served_prompt_lengths_differ",
                                       "prompt_ids_unmatched", "stock_provenance_unverified", "unverified_device_placement",
                                       "generated_lengths_or_stops_differ"})
    status = "incomplete" if incomplete else ("does_not_qualify" if reasons else "qualifies")
    return {"status": status, "engines": list(engines), "reasons": sorted(reasons), "comparisons": checks}


def run(args, report, save, deadline):
    from ironmule.factual_gate import MAX_TOKENS, correct

    from .code_identity import sources_digest
    from .compare_servers import open_server, prepare_artifacts, refresh_identity
    from .compare_tokenizer import preflight

    prepare_artifacts(args)
    report["protocol"]["runner_sha256"] = sources_digest()
    report["artifacts"] = {"gguf_sha256": args.gguf_sha256, "tokenizer_sha256": args.tokenizer_sha256}
    ref_device = None
    for rep in range(args.repeats):
        data = workloads(args, rep)
        texts = [WARMUP]
        for performance, quality in data.values():
            texts.extend(performance)
            texts.extend(q.prompt for q in quality)
        offset = rep % len(args.engines)
        for engine in args.engines[offset:] + args.engines[:offset]:
            report["current"] = {"engine": engine, "repeat": rep, "concurrency": None}
            save("starting")
            remaining(deadline)
            try:
                with open_server(engine, args, rep, deadline) as server:
                    save("preflight")
                    if not server.get("version") or server.get("gguf_sha256") != args.gguf_sha256 or (
                            server.get("tokenizer_sha256") != args.tokenizer_sha256):
                        raise Unverified("server artifact or version identity unavailable")
                    device = server.get("device")
                    if not isinstance(device, str) or not device or (ref_device and device != ref_device):
                        raise Unverified("hardware identity unavailable or changed between arms")
                    ref_device = report["device"] = device
                    arm = {"engine": engine, "repeat": rep, "identity": server,
                           "tokenization": {"verified": False, "digest": None}}
                    report["arms"].append(arm)
                    save("preflight")
                    probe = preflight(engine, server, args, [conversation(WARMUP)], deadline)
                    if probe.get("verified") is not True or len(probe.get("prompt_tokens", [])) != 1:
                        raise Unverified(f"{engine} warmup tokenization failed: {probe.get('reason')}")
                    save("warmup")
                    warm = stream(engine, server, WARMUP, min(8, args.max_tokens), deadline)
                    check_usage([WARMUP], [warm], {WARMUP: probe["prompt_tokens"][0]})
                    served_lengths = [warm["prompt_tokens"]]
                    refresh_identity(engine, server, deadline)
                    if engine == "ironmule" and server.get("autonomous") is not True:
                        raise Unverified("autonomous IronMule serving was not verified after warmup")
                    for width, (prompts, cases) in data.items():
                        report["current"]["concurrency"] = width
                        save("benchmark")
                        rows, wall = batch(engine, server, prompts, width, args.max_tokens, deadline)
                        served_lengths.extend(r["prompt_tokens"] for r in rows)
                        tokens = sum(row["tokens"] for row in rows)
                        firsts = sorted(row["ttft_ms"] for row in rows)
                        result = {"engine": engine, "repeat": rep, "concurrency": width, "requests": len(rows),
                                  "completion_tokens": tokens, "tokens_per_second": tokens / wall,
                                  "requests_per_second": len(rows) / wall, "wall_s": wall,
                                  "generation": [[r["tokens"], r["finish_reason"]] for r in rows],
                                  "performance_prompt_tokens": [r["prompt_tokens"] for r in rows],
                                  "ttft_median_ms": statistics.median(firsts),
                                  "ttft_p95_ms": firsts[math.ceil(.95 * len(firsts)) - 1], "quality": None}
                        report["runs"].append(result)
                        save("quality")
                        rows, _ = batch(engine, server, [q.prompt for q in cases], width, MAX_TOKENS, deadline)
                        served_lengths.extend(r["prompt_tokens"] for r in rows)
                        answers = {q.key: correct(q, row["text"]) for q, row in zip(cases, rows)}
                        result["quality"] = {"passed": sum(answers.values()), "total": len(answers), "answers": answers,
                                             "prompt_tokens": [r["prompt_tokens"] for r in rows]}
                        save("quality")
                    # Tokenize the timed inputs only now, so a server's tokenizer or template
                    # cache cannot be primed by our verifier before its first timed request.
                    save("preflight")
                    proof = preflight(engine, server, args, [conversation(t) for t in texts], deadline)
                    if proof.get("verified") is not True or len(proof.get("digest") or "") != 64:
                        raise Unverified(f"{engine} prompt verification failed: {proof.get('reason')}")
                    if proof.get("prompt_tokens") != served_lengths:
                        raise Unverified(f"{engine} served prompt lengths differ from its verified token IDs")
                    arm["tokenization"] = {"verified": True, "digest": proof["digest"]}
                    save("preflight")
            except Exception as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError("comparison budget exhausted") from exc
                code = ("identity_unverified" if isinstance(exc, Unverified) or report["stage"] == "preflight"
                        else "launch_failed" if report["stage"] == "starting" else "measurement_failed")
                report["failures"].append({**report["current"], "stage": report["stage"], "code": code,
                                           "error": str(exc)[:600]})
                save(report["stage"])
    save("evaluating")
    report["verdict"] = evaluate(report)


def arguments(argv):
    p = argparse.ArgumentParser(prog="ironmule compare", description=__doc__)
    p.add_argument("--engines", default=",".join(ENGINES), help="ironmule plus llama_cpp and/or vllm")
    p.add_argument("--gguf", required=True, type=Path, help="a single local GGUF weight file used by all three engines")
    p.add_argument("--tokenizer", required=True, type=Path, help="matching offline Hugging Face tokenizer/config folder")
    p.add_argument("--llama-server", type=Path, help="stock llama-server executable")
    p.add_argument("--stock-manifest", type=Path, help="attested clean-upstream executable and library SHA manifest")
    p.add_argument("--vllm", type=Path, help="vLLM executable")
    p.add_argument("--ironmule", type=Path, help="IronMule executable; defaults to this interpreter's CLI")
    p.add_argument("--output", type=Path, help="live JSON report; defaults to benchmark-progress.json in product state")
    p.add_argument("--state-dir", type=Path, help="IronMule product state root")
    p.add_argument("--agent-dir", type=Path, help="dedicated persistent IronMule benchmark agent state")
    p.add_argument("--log-dir", type=Path, help="server log directory")
    p.add_argument("--gpu-memory-utilization", type=float, default=.88)
    p.add_argument("--seed", type=int, help="reproduce a workload; a new seed is generated by default")
    p.add_argument("--concurrency", default="1,8,32", help="comma-separated positive client counts")
    p.add_argument("--repeats", type=int, default=3, help="rotate engine order on each repeat")
    p.add_argument("--requests", type=int, default=8, help="minimum performance requests per load, at least concurrency")
    p.add_argument("--max-tokens", type=int, default=32, help="maximum completion tokens in the timed traffic")
    p.add_argument("--budget-seconds", type=int, default=2700, help="whole comparison wall-time budget")
    p.add_argument("--context", type=int, default=2048, help="context tokens per sequence")
    p.add_argument("--startup-timeout", type=int, default=300, help="maximum time per server start, within total budget")
    args = p.parse_args(argv)
    selected = args.engines.split(",")
    if (len(selected) < 2 or selected[0] != "ironmule" or set(selected) - set(ENGINES)
            or len(selected) != len(set(selected))):
        p.error("--engines needs ironmule followed by one or both distinct competitors")
    args.engines = tuple(selected)
    if "llama_cpp" in selected and not args.llama_server or "vllm" in selected and not args.vllm:
        p.error("supply the executable for each selected competitor")
    try:
        args.concurrency = [int(value) for value in args.concurrency.split(",")]
    except ValueError:
        p.error("--concurrency expects comma-separated integers")
    if not 1 <= len(args.concurrency) <= 8 or len(set(args.concurrency)) != len(args.concurrency) or (
            not all(1 <= value <= 64 for value in args.concurrency)):
        p.error("choose between one and eight distinct concurrency levels in 1..64")
    if not (1 <= args.repeats <= 9 and 1 <= args.requests <= 128 and 1 <= args.max_tokens <= 2048
            and args.budget_seconds > 0 and args.startup_timeout > 0 and args.context > args.max_tokens
            and 0 < args.gpu_memory_utilization <= 1):
        p.error("invalid benchmark bounds (see --help)")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        p.error("--seed must be an unsigned 32-bit integer")
    args.seed = secrets.randbits(32) if args.seed is None else args.seed
    args.state_dir = ProductStore(args.state_dir).root
    args.output = (args.output or Path(os.environ.get("IRONMULE_BENCH_PROGRESS") or
                                     args.state_dir / "benchmark-progress.json")).expanduser().absolute()
    return args


def learn(args, report, save, deadline):
    """Keep the comparison verdict if a subsequent local admission experiment cannot run."""
    from ironmule.compare_learning import respond

    arm = next((a["identity"] for a in report["arms"] if a["engine"] == "ironmule"
                and a["identity"].get("autonomous") is True), None)
    if not arm:
        return
    command = arm.get("argv", [])
    state = args.agent_dir
    if not state and "--state-dir" in command:
        state = Path(command[command.index("--state-dir") + 1])
    if state is None:
        report["learning"] = {"status": "deferred", "reason": "no bound local IronMule state"}
        return
    save("learning")
    provider = arm.get("provider") or ("cuda" if str(arm.get("device")).startswith("GPU-") else "cpu")
    try:
        report["learning"] = respond(report, gguf=args.gguf.resolve(), state=state, provider=provider,
                                     deadline=deadline)
    except (OSError, RuntimeError, ValueError) as exc:
        report["learning"] = {"status": "error", "reason": str(exc)[:400]}


def main(argv=None):
    args = arguments(argv)
    started = time.monotonic()
    deadline = started + args.budget_seconds
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    report = {"schema": "ironmule.compare.v1", "protocol": {"id": PROTOCOL, "seed": args.seed,
              "engines": list(args.engines), "repeats": args.repeats, "concurrency": args.concurrency, "requests": args.requests,
              "max_tokens": args.max_tokens, "context": args.context, "budget_seconds": args.budget_seconds},
              "run_id": uuid.uuid4().hex, "status": "running", "stage": "initializing", "started_at": utc_now(),
              "model": args.gguf.name, "device": "unverified", "host_arch": platform.machine(),
              "current": {}, "arms": [], "runs": [], "failures": [], "verdict": {"status": "pending", "reasons": []}}

    def save(stage):
        report.update(stage=stage, updated_at=utc_now(), elapsed_seconds=round(time.monotonic() - started, 3))
        _atomic_write(args.output, report)

    with args.output.with_name("." + args.output.name + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Another comparison is writing {args.output}", file=sys.stderr)
            return 1
        save("initializing")
        try:
            run(args, report, save, deadline)
            learn(args, report, save, deadline)
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - persist the actual partial result and stage
            code = "identity_unverified" if isinstance(exc, Unverified) else {
                "starting": "launch_failed", "initializing": "invalid_input", "preflight": "identity_unverified",
                "learning": "learning_failed"}.get(report["stage"], "measurement_failed")
            timed_out = isinstance(exc, TimeoutError) or time.monotonic() >= started + args.budget_seconds
            if isinstance(exc, KeyboardInterrupt):
                code = "interrupted"
            elif timed_out:
                code = "budget_exhausted"
            report.update(status="timeout" if timed_out else "error", error_code=code, error=str(exc)[:600],
                          verdict={"status": "incomplete", "reasons": [code]})
            save(report["stage"])
            print(f"{code}: {report['error']} (report {args.output})", file=sys.stderr)
            if isinstance(exc, KeyboardInterrupt):
                return 130
            return 124 if timed_out else 1
        report["status"] = "error" if report["failures"] else "complete"
        if report["failures"]:
            report["error_code"] = report["failures"][-1]["code"]
        save("finished")
    print(f"{report['verdict']['status']}: {args.output}")
    return 1 if report["failures"] else 0
