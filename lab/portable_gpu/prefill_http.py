"""Compare split-stream, grouped-prompt and shared-pool GGUF serving on the same physical GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

from ironmule_product import compare, compare_servers, compare_tokenizer
from ironmule_product.state import _atomic_write
from lab.portable_gpu import fetch, production_compare

ARMS = {"stock": ("1", "0"), "batch": ("1", "1"), "shared": ("0", "0")}


def read(server, path, deadline):
    return compare_servers._json(server["base_url"], path, min(5, compare.remaining(deadline)))


class Profile:
    """Read the product's numeric request samples outside the client measurement interval."""
    def __init__(self, arm):
        self.arm = arm

    def before(self, server, deadline):
        live = read(server, "/v1/ironmule/live", deadline)
        ggml = read(server, "/health", deadline)["ggml"]
        layout = ggml.get("experiment_binding") or {}
        stats = ggml.get("prefill_batch") or {}
        if stats.get("enabled") is not (self.arm == "batch"):
            raise compare.Unverified("actual engine toggle differs from requested arm")
        if layout.get("shared") is not (self.arm == "shared"):
            raise compare.Unverified("actual shared-pool placement differs from requested arm")
        if self.arm != "shared" and layout.get("settings", {}).get("kv_unified") is not False:
            raise compare.Unverified("separate KV streams are unverified")
        return live["observed_at"], stats

    def after(self, server, deadline, checkpoint, rows):
        cutoff, previous = checkpoint
        until = min(time.monotonic() + 4, deadline)
        expected = sorted((r["prompt_tokens"], r["tokens"]) for r in rows)
        while True:
            live = read(server, "/v1/ironmule/live", deadline)["runtime"]
            samples = [s for s in live["samples"] if s["at"] > cutoff]
            if sorted((s["prompt_tokens"], s["completion_tokens"]) for s in samples) == expected:
                break
            if time.monotonic() >= until:
                raise compare.Unverified("live samples disagree with the measured wave")
            time.sleep(.04)
        health = read(server, "/health", deadline)
        ggml = health["ggml"]
        binding = ggml["experiment_binding"]
        stats = ggml["prefill_batch"]
        timings = {}
        for key in ("queue_ms", "dispatch_to_first_ms", "first_to_end_ms"):
            values = sorted(s[key] for s in samples)
            if not all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in values):
                raise compare.Unverified("invalid server interval")
            timings[key] = {"median": statistics.median(values), "p95": values[math.ceil(.95 * len(values)) - 1]}
        deltas = {k: stats[k] - previous[k] for k in ("calls", "rows", "prompts")}
        if min(deltas.values()) < 0:
            raise compare.Unverified("worker counters restarted within a wave")
        outputs = [(r["text"], r["tokens"], r["finish_reason"]) for r in rows]
        return {"intervals": timings, "samples": len(samples), "batch_delta": deltas, "shared": binding["shared"],
                "sequences": binding["seqs"], "context": binding["seq_ctx"], "schedule": health.get("schedule"),
                "outputs_sha256": hashlib.sha256(json.dumps(outputs, ensure_ascii=True).encode()).hexdigest()}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("gguf", "tokenizer", "state-dir", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--ironmule", type=Path, help="optional production CLI executable in its own environment")
    p.add_argument("--provider", choices=("cuda",), default="cuda")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--repeats", type=int, choices=range(1, 10), default=3)
    p.add_argument("--budget-seconds", type=int, default=2400)
    p.add_argument("--startup-timeout", type=int, default=900)
    p.add_argument("--log-dir", type=Path)
    a = p.parse_args()
    if len(set(a.arms)) != len(a.arms) or "stock" not in a.arms or min(a.budget_seconds, a.startup_timeout) <= 0:
        p.error("include stock, use distinct arms and positive time limits")
    a.seed, a.context, a.max_tokens, a.requests, a.concurrency = 3591431646, 2048, 32, 8, [1, 8, 32]
    return a


def run(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.close(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    started = time.monotonic()
    deadline = started + args.budget_seconds
    report = {"schema": "ironmule.prefill-http.v1", "status": "running", "deployments": [], "measurements": [],
              "protocol": {k: getattr(args, k) for k in ("arms", "repeats", "concurrency", "seed", "context",
                                                         "requests", "max_tokens")}, "errors": [], "current": {}}

    def save(stage):
        # Full process identity includes a device UUID and local argv. Only publish the verified summaries.
        public = {k: v for k, v in report.items() if k not in ("deployments", "device")}
        public["deployments"] = []
        for arm in report["deployments"]:
            identity = arm["identity"]
            safe = {k: identity.get(k) for k in ("version", "provider", "device_name", "placement_verified",
                        "gguf_sha256", "tokenizer_sha256", "implementation_sha256", "autonomous", "startup_seconds")}
            public["deployments"].append({**arm, "identity": safe})
        _atomic_write(args.output, dict(public, stage=stage, elapsed_seconds=round(time.monotonic()-started, 3)))

    save("binding")
    failed = set()
    try:
        artifacts = compare_servers.prepare_artifacts(args)
        _, _, _, size, sha = fetch.MODELS["qwen3-32b"]
        config = compare_tokenizer.TOKENIZERS["qwen3-32b"][2]
        if (args.gguf.stat().st_size != size or args.gguf_sha256 != sha or
                artifacts["tokenizer_files"] != {**compare_tokenizer._QWEN_FILES, "config.json": config}):
            raise compare.Unverified("expected official pinned model and tokenizer")
        for rep in range(args.repeats):
            data = compare.workloads(args, rep)
            start = rep % len(args.arms)
            for name in args.arms[start:] + args.arms[:start]:
                if name in failed:
                    continue
                os.environ["IRONMULE_GGML_STREAMS"], os.environ["IRONMULE_GGML_PREFILL_BATCH"] = ARMS[name]
                args.agent_dir = args.state_dir / name
                report["current"] = {"deployment": name, "repeat": rep}
                save("starting")
                try:
                    production_compare.measure(args, "ironmule_gguf", rep, data, report, save, deadline,
                                               profile=Profile(name), label=name)
                except (compare.Unverified, compare_servers.StartupError) as exc:
                    error = exc.report_fields() if isinstance(exc, compare_servers.StartupError) else {"reason": str(exc)[:180]}
                    report["errors"].append({"deployment": name, "repeat": rep, **error})
                    failed.add(name)
                    save("arm_unavailable")
                    if name == "stock":
                        raise
        report["status"] = "partial" if failed else "measured"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "incomplete"
        report["failure"] = {"type": type(exc).__name__}
        raise
    finally:
        save("finished")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(run(arguments()))
