"""Measure the production width controller against itself before and after the observer fix.
Supply clean before/ and after/ source trees and the same GGUF/bindings. Lab invocation
of idle() follows each synthetic wave; read timings as native, not product HTTP latency.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time

import bench
from switch_probe import record, serve

REVISIONS = {"before": ("acc1a3c", "57dd84f7808cca3fd6b368366baac8ca9a4d5afb6bd48764769986abf36429d6"),
             "after": ("3b1d717", "2a1a627e6b53e1584ec5f2c554173d730f79e031e4f02223085d0c92f1878b4a")}
WIDTHS = (8, 32)
CONTEXT, OUTPUT = 128, 3
PHASES = (("short", 8, 32), ("documents", 32, 96), ("short_again", 8, 32)) * 2


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def worker(args):
    bench.ROOT = (args.trees / args.side).resolve()
    os.environ.update(IRONMULE_GGML_THREADS="2", IRONMULE_GGML_SHARED="0", IRONMULE_GGML_STREAMS="0")
    bench.configure(args.provider)
    from ironmule import ggml_engine as ge
    from ironmule.ggml_agent import GgmlAgent, configs_from_measurement
    from ironmule.ggml_binary import identity
    from ironmule_devices import ggml_devices, verify_ggml

    if Path(ge.__file__).resolve() != bench.ROOT / "ironmule/ggml_engine.py":
        raise ValueError("wrong source tree")
    devices = verify_ggml(args.provider)
    if args.provider == "cuda" and (len(devices) != 1 or len(ggml_devices()) != 1):
        raise ValueError("expected exactly one visible CUDA device")
    native = identity()
    if not native.get("sha256"):
        raise ValueError("native identity unavailable")
    device = devices[0].get("description") if devices else "CPU"
    result = {"native": native, "device": device, "machine_sha256": digest(ge.machine_key())}
    if args.stage == "calibrate":
        result["calibration"] = ge.measure_widths(str(args.gguf), {}, CONTEXT, WIDTHS, steps=3, repeats=2, long_ctx=96)
    else:
        calibration = json.loads(args.calibration.read_text())["calibration"]
        configs = configs_from_measurement([calibration], CONTEXT)
        if set(configs) != {"w8", "w32"}:
            raise ValueError("missing local width calibrations")
        engine = ge.GgmlEngine(str(args.gguf), seqs=8, seq_ctx=CONTEXT, speculate=False, threads=2, shared=False)
        try:
            agent = GgmlAgent(None, {"device": device, "gpu_layers": engine.gpu_layers}, configs, "w8",
                              seed=args.repeat, price_switches=False)
            engine.observer = agent.observe_step
            warm = bench.prompts(engine, "calibration", 32, 4) * 2
            for _ in range(6):
                _, completed = serve(engine, warm, OUTPUT, time.monotonic_ns())
                record(agent, completed)
            result["waves"] = []
            for idx, (phase, width, length) in enumerate(PHASES):
                # Holdout targets differ from calibration. Rotation is identical across revisions.
                tasks = bench.prompts(engine, "holdout" if idx % 2 == 0 else "replay", length, 4)
                tasks = (tasks[idx % 4:] + tasks[:idx % 4]) * (width // 4)
                arrived = time.monotonic_ns()
                selected = agent.idle()
                if selected and selected != agent.current:
                    before = time.monotonic()
                    engine.reconfigure(configs[selected]["seqs"], configs[selected]["settings"])
                    agent.switched(selected, seconds=time.monotonic() - before)
                decision_s = (time.monotonic_ns() - arrived) / 1e9
                if engine.seqs != configs[agent.current]["seqs"]:
                    raise ValueError("active configuration disagrees with controller")
                measured, completed = serve(engine, tasks, OUTPUT, arrived)
                record(agent, completed)
                requests = measured["requests"]
                ttft = sorted(r["ttft_ms"] for r in requests)
                tokens = sum(len(r["tokens"]) for r in requests)
                result["waves"].append({"phase": phase, "arrivals": width, "prompt_tokens": length,
                    "request_ids_sha256": digest([ids for ids, _ in tasks]), "output_ids_stops_sha256":
                    digest([(r["tokens"], r["stop"]) for r in requests]), "tokens": tokens,
                    "correct": sum(r["correct"] for r in requests), "config": agent.current,
                    "decision": (agent.history or [{}])[-1], "decision_and_switch_s": decision_s,
                    "elapsed_s": measured["elapsed_s"], "tokens_per_second": tokens / measured["elapsed_s"],
                    "ttft_p95_ms": ttft[math.ceil(.95 * width) - 1]})
            agent.learn()
            result["samples_by_config"] = agent.live
            result["config_weights"] = {k: v.theta for k, v in agent.costs.items()}
        finally:
            engine.close()
    bench.save(args.save, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--trees", type=Path, required=True, help="directory containing before and after")
    p.add_argument("--provider", choices=("cpu", "cuda"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget-s", type=int, default=480)
    for k in ("stage", "side"):
        p.add_argument("--" + k, help=argparse.SUPPRESS)
    for k in ("calibration", "save"):
        p.add_argument("--" + k, type=Path, help=argparse.SUPPRESS)
    p.add_argument("--repeat", type=int, default=0, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.stage:
        worker(args)
        return
    if args.output.exists() or not args.gguf.is_file() or args.budget_s < 1:
        p.error("existing GGUF, fresh output and positive timeout required")
    root = args.trees.resolve()
    for side, (_, expected) in REVISIONS.items():
        if hashlib.sha256((root / side / "ironmule/ggml_engine.py").read_bytes()).hexdigest() != expected:
            raise ValueError("incorrect " + side + " engine")
    files = [{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / side / "ironmule").glob("*.py")
              if p.name != "ggml_engine.py"} for side in REVISIONS]
    if files[0] != files[1] or "ggml_agent.py" not in files[0]:
        raise ValueError("remaining source differs across comparison")
    deadline = time.monotonic() + args.budget_s
    report = {"schema": "ironmule.controller-observer-pair.v1", "status": "running", "revisions": REVISIONS,
              "model_sha256": hashlib.file_digest(args.gguf.open("rb"), "sha256").hexdigest(),
              "controller_sha256": files[0]["ggml_agent.py"], "provider": args.provider, "pairs": [], "arms": [],
              "protocol": {"widths": WIDTHS, "context": CONTEXT, "output_limit": OUTPUT, "warm_requests": 48,
                           "order": ["before", "after", "after", "before"], "idle_charged_to_wave": True,
                           "pricing": "production unpriced switches", "shared_startup_calibration": True,
                           "scope": "held-out native waves with production controller, prompted idle boundaries"}}
    bench.save(args.output, report)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            def run(stage, side, rep):
                save = tmp / (stage + side + str(rep) + ".json")
                log = tmp / "native.log"
                with log.open("wb") as stream:
                    cmd = [sys.executable, __file__, *sys.argv[1:], "--stage", stage, "--side", side, "--repeat", str(rep),
                           "--save", str(save), "--calibration", str(tmp / "calibrateafter0.json")]
                    completed = subprocess.run(cmd, stderr=stream, timeout=max(.001, deadline-time.monotonic()))
                if completed.returncode:
                    print(log.read_text(errors="replace")[-2000:], file=sys.stderr)
                    raise RuntimeError("child failed")
                data = json.loads(save.read_text())
                counts = re.findall(rb"offloaded (\d+)/(\d+) layers to GPU", log.read_bytes())
                if args.provider == "cuda" and not (counts and all(int(a)==int(b)>0 for a,b in counts)):
                    raise ValueError("no confirmation of full GPU offload")
                data["full_offload"] = [list(map(int, pair)) for pair in counts]
                return data

            calibration = run("calibrate", "after", 0)
            report["calibration"] = calibration
            bench.save(args.output, report)
            for rep, sides in enumerate((("before","after"), ("after","before"))):
                for side in sides:
                    result = run("arm", side, rep)
                    if any(result[k] != calibration[k] for k in ("native", "device", "machine_sha256")):
                        raise ValueError("runtime binding changed")
                    report["arms"].append({"repeat": rep, "source": side, **result})
                    bench.save(args.output, report)
                old, new = [next(row["waves"] for row in report["arms"] if row["source"]==side and row["repeat"]==rep)
                            for side in ("before", "after")]
                for i, (a, b) in enumerate(zip(old, new, strict=True)):
                    if a["request_ids_sha256"] != b["request_ids_sha256"]:
                        raise ValueError("different request tokens")
                    report["pairs"].append({"repeat": rep, "wave": i, "phase": a["phase"],
                        "same_output_ids_stops": a["output_ids_stops_sha256"]==b["output_ids_stops_sha256"],
                        "correct_before": a["correct"], "correct_after": b["correct"], "requests": a["arrivals"],
                        "width_before": a["config"], "width_after": b["config"],
                        "after_over_before_tokens_per_second": b["tokens_per_second"] / a["tokens_per_second"],
                        "after_over_before_ttft_p95": b["ttft_p95_ms"] / a["ttft_p95_ms"]})
                bench.save(args.output, report)
            report["status"] = "measured"
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        report.update(status="incomplete", reason=type(exc).__name__)
        raise
    finally:
        bench.save(args.output, report)


if __name__ == "__main__":
    main()
