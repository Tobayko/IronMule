"""Compare split-stream, grouped-prompt and shared-pool GGUF serving on the same physical GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
from pathlib import Path

from ironmule_product import compare, compare_servers, compare_tokenizer
from ironmule_product.state import _atomic_write
from lab.portable_gpu import fetch, production_compare

ARMS = {"stock": ("1", "0"), "batch": ("1", "1"), "shared": ("0", "0"), "long-444": ("1", "long-444")}


def read(server, path, deadline):
    return compare_servers._json(server["base_url"], path, min(5, compare.remaining(deadline)))


def _scope(binding, server):
    # machine_key contains the expected grouped-prompt toggle. Preserve its hash
    # per arm and compare the rest of the model/device/execution contract directly.
    placement = server.get("device") or ""
    uuid = re.fullmatch(r"GPU-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})", placement)
    if (not uuid or binding.get("provider") != "cuda" or not binding.get("host_sha256")
            or not binding.get("implementation") or not binding.get("machine")
            or not re.fullmatch(r"[a-f0-9]{64}", binding.get("study_native_runtime_sha256") or "")
            or binding.get("cuda_uuid") != uuid.group(1).replace("-", "").lower()
            or binding.get("gguf_sha256") != server.get("gguf_sha256")):
        raise compare.Unverified("missing same-worker prefill identity")
    raw = {k: v for k, v in binding.items() if k != "machine"}
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"settings_device_sha256": digest(raw), "arm_sha256": digest(binding),
            "native_runtime_sha256": binding["study_native_runtime_sha256"],
            "implementation_sha256": binding["implementation"]}


class Profile:
    """Read the product's numeric request samples outside the client measurement interval."""
    def __init__(self, arm, receipt=None):
        self.arm = arm
        self.receipt = receipt
        self.placement = {}

    def before(self, server, deadline):
        live = read(server, "/v1/ironmule/live", deadline)
        ggml = read(server, "/health", deadline)["ggml"]
        layout = ggml.get("experiment_binding") or {}
        stats = ggml.get("prefill_batch") or {}
        self.placement = {
            "shared": layout.get("shared"), "sequences": layout.get("seqs"), "context": layout.get("seq_ctx"),
            "kv_unified": layout.get("settings", {}).get("kv_unified"), "grouping_enabled": stats.get("enabled"),
            "planned_sequences": (ggml.get("sequences") or {}).get("sequences"),
            "stream_qualification": ggml.get("stream_qualification"),
        }
        if stats.get("enabled") is not (self.arm in ("batch", "long-444")):
            raise compare.Unverified("actual engine toggle differs from requested arm")
        if layout.get("shared") is not (self.arm == "shared"):
            raise compare.Unverified("actual shared-pool placement differs from requested arm")
        if self.arm != "shared" and layout.get("settings", {}).get("kv_unified") is not False:
            raise compare.Unverified("separate KV streams are unverified")
        scope = _scope(layout, server)
        if self.receipt is not None:
            from lab.portable_gpu import long_prefill_http

            long_prefill_http.verify_worker(self.receipt, self.arm, server, ggml, scope)
        return live["observed_at"], stats, scope

    def after(self, server, deadline, checkpoint, rows):
        cutoff, previous, scope = checkpoint
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
        if (_scope(binding, server) != scope or stats.get("enabled") is not (self.arm in ("batch", "long-444"))
                or binding.get("shared") is not (self.arm == "shared")):
            raise compare.Unverified("execution configuration changed during the measured wave")
        timings = {}
        for key in ("queue_ms", "dispatch_to_first_ms", "first_to_end_ms"):
            values = sorted(s[key] for s in samples)
            if not all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in values):
                raise compare.Unverified("invalid server interval")
            timings[key] = {"median": statistics.median(values), "p95": values[math.ceil(.95 * len(values)) - 1]}
        deltas = {k: stats[k] - previous[k] for k in ("calls", "rows", "prompts")}
        if min(deltas.values()) < 0:
            raise compare.Unverified("worker counters restarted within a wave")
        if self.receipt is not None:
            from lab.portable_gpu import long_prefill_http

            long_prefill_http.verify_worker(self.receipt, self.arm, server, ggml, scope)
            deltas["long_shape_hits"] = long_prefill_http.wave_hits(self.arm, stats, previous, deltas)
        outputs = [(r["text"], r["tokens"], r["finish_reason"]) for r in rows]
        return {"scope": scope, "intervals": timings, "samples": len(samples), "batch_delta": deltas, "shared": binding["shared"],
                "sequences": binding["seqs"], "context": binding["seq_ctx"], "schedule": health.get("schedule"),
                "outputs_sha256": hashlib.sha256(json.dumps(outputs, ensure_ascii=True).encode()).hexdigest()}


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("gguf", "tokenizer", "state-dir", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--ironmule", type=Path, help="optional production CLI executable in its own environment")
    p.add_argument("--provider", choices=("cuda",), default="cuda")
    p.add_argument("--arms", nargs="+", choices=ARMS)
    p.add_argument("--long-native-receipt", type=Path,
                   help="successful local long-444 native study; runs only stock and long-444 at widths 8 and 32")
    p.add_argument("--repeats", type=int, choices=range(1, 10))
    p.add_argument("--budget-seconds", type=int)
    p.add_argument("--startup-timeout", type=int, default=900)
    p.add_argument("--log-dir", type=Path)
    a = p.parse_args(argv)
    a.arms = a.arms or (["stock", "long-444"] if a.long_native_receipt else ["stock", "batch", "shared"])
    a.repeats = a.repeats or (2 if a.long_native_receipt else 3)
    if a.budget_seconds is None:
        a.budget_seconds = 900 if a.long_native_receipt else 2400
    if ((a.long_native_receipt and set(a.arms) != {"stock", "long-444"})
            or ("long-444" in a.arms and not a.long_native_receipt)):
        p.error("the long-444 follow-up requires its native receipt and the two arms stock and long-444")
    if len(set(a.arms)) != len(a.arms) or "stock" not in a.arms or min(a.budget_seconds, a.startup_timeout) <= 0:
        p.error("include stock, use distinct arms and positive time limits")
    a.seed, a.context, a.max_tokens, a.requests, a.concurrency = 3591431646, 2048, 32, 8, [1, 8, 32]
    if a.long_native_receipt:
        a.seed, a.concurrency = 2228013061, [8, 32]  # independent of the historical comparison and native trial
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
            safe["implementation_sha256"] = (identity.get("ggml") or {}).get("implementation")
            public["deployments"].append({**arm, "identity": safe})
        _atomic_write(args.output, dict(public, stage=stage, elapsed_seconds=round(time.monotonic()-started, 3)))

    save("binding")
    unavailable = set()
    native_path = getattr(args, "long_native_receipt", None)
    receipt = None
    knobs = ("IRONMULE_PREFILL_STUDY", "IRONMULE_GGML_STREAMS", "IRONMULE_GGML_PREFILL_BATCH",
             "IRONMULE_GGML_PREFILL_RECEIPT")
    original_env = {k: os.environ.get(k) for k in knobs}
    try:
        if native_path is not None:
            from lab.portable_gpu import long_prefill_http

            receipt, report["native_receipt_sha256"] = long_prefill_http.read_receipt(native_path)
        artifacts = compare_servers.prepare_artifacts(args)
        _, _, _, size, sha = fetch.MODELS["qwen3-32b"]
        config = compare_tokenizer.TOKENIZERS["qwen3-32b"][2]
        if (args.gguf.stat().st_size != size or args.gguf_sha256 != sha or
                artifacts["tokenizer_files"] != {**compare_tokenizer._QWEN_FILES, "config.json": config}):
            raise compare.Unverified("expected official pinned model and tokenizer")
        if receipt is not None:
            long_prefill_http.verify_local(receipt, args.gguf_sha256)
        for rep in range(args.repeats):
            data = compare.workloads(args, rep)
            start = rep % len(args.arms)
            for name in args.arms[start:] + args.arms[:start]:
                if name in unavailable:
                    continue
                os.environ["IRONMULE_PREFILL_STUDY"] = "1"
                os.environ["IRONMULE_GGML_STREAMS"], os.environ["IRONMULE_GGML_PREFILL_BATCH"] = ARMS[name]
                if receipt is not None:
                    if name == "long-444":
                        if long_prefill_http.read_receipt(native_path)[1] != report["native_receipt_sha256"]:
                            raise compare.Unverified("native receipt changed during the study")
                        os.environ["IRONMULE_GGML_PREFILL_RECEIPT"] = str(native_path.resolve(strict=True))
                    else:
                        os.environ.pop("IRONMULE_GGML_PREFILL_RECEIPT", None)
                args.agent_dir = args.state_dir / name
                report["current"] = {"deployment": name, "repeat": rep}
                save("starting")
                profile = Profile(name, receipt)
                try:
                    production_compare.measure(args, "ironmule_gguf", rep, data, report, save, deadline,
                                               profile=profile, label=name)
                except (compare.Unverified, RuntimeError, OSError) as exc:
                    if isinstance(exc, compare_servers.StartupError):
                        error = exc.report_fields()
                    elif isinstance(exc, compare.Unverified):
                        error = {"reason": str(exc)[:180]}
                    else:
                        # A streaming exception can contain a response body. Keep only a bounded error class.
                        error = {"reason": "request transport or worker error", "type": type(exc).__name__}
                    if profile.placement:
                        error["placement"] = profile.placement
                    report["errors"].append({"deployment": name, "repeat": rep, **error})
                    if isinstance(exc, (compare.Unverified, compare_servers.StartupError)):
                        unavailable.add(name)
                    save("arm_unavailable")
                    if receipt is not None or (name == "stock" and name in unavailable):
                        raise
                    compare.remaining(deadline)
            if receipt is not None:
                report["pairs"] = long_prefill_http.pairs(report, repeats=rep + 1)
                save("paired")
        if receipt is not None:
            if long_prefill_http.read_receipt(native_path)[1] != report["native_receipt_sha256"]:
                raise compare.Unverified("native receipt changed during the study")
        report["status"] = "partial" if report["errors"] else "measured"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "incomplete"
        report["failure"] = {"type": type(exc).__name__}
        if native_path is not None and isinstance(exc, compare.Unverified):
            report["failure"]["reason"] = str(exc)[:180]
        raise
    finally:
        for name, value in original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        save("finished")
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(run(arguments()))
