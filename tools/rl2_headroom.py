"""RL2-H: does a whole-machine serving decision have state-dependent headroom (BACKLOG RL2)?

A learner can only beat the best fixed setting where different states want different settings.
RSI3-LIVE-P3 and OPT1 lost to the best fixed setting, so before a whole-machine learner is trained
this measures the upper bound any policy could reach, on the same recorded traces, against the A/A
noise of the reference (as OPT2-CAL did for the background-thread decision).

Fixed before data (BACKLOG RL2-H, 2026-10-28):

  arms      every setting a learner could pick: placement across GPUs (`shortest`, HW1's
            join-the-shortest-queue; `fill`, the lowest-index GPU with fewer than `width` requests
            outstanding, else the shortest) times the live learner's schedule catalogue (group width
            1, 2, 4, 8; gather wait 0, 50, 150 ms; width 1 never waits). The reference is
            `shortest/w8/g0`, the stored default policy at the tuned engine's group capacity of 8. With
            one GPU placement does not exist and only the schedule is measured.
  levels    offered load at 0.2, 0.6 and 0.95 of the reference's capacity (requests completed per
            second from a burst of 16 per GPU), equal weights.
  traces    per level, six traces of 16 Poisson arrivals with fixed seeds; prompt and output limit
            (64 or 128 tokens) are drawn from the same seed. Every arm replays the same traces.
  order     per (trace, level) block: the reference, every other arm in an order rotated by the
            block and reversed on odd blocks, the reference again (an A/A pair spanning the block).
  cost      per request, the live learner's reward (docs/LIVE_LEARNING.md): seconds in the system
            from the scheduled arrival + time to first content, or (1 + 1) x deadline + 1 s if it
            failed, timed out or was refused; a cell's cost is the mean over its trace.
  headroom  oracle = sum over levels of weight x the cheapest arm there; fixed = the cheapest single
            arm over all levels; H = 1 - oracle / fixed. In sample, and held out: arms chosen on the
            other traces, scored on the left-out one. Per decision, the oracle may vary only that
            decision; the other stays at the best fixed arm's value.
  noise     the 95th percentile (nearest rank) of the A/A pairs' relative differences.
  verdict   a decision has headroom when the 2.5 % bootstrap bound of its held-out H (2000 resamples
            of the held-out traces, seed 0) exceeds the noise; otherwise a fixed policy ships and no
            learner is trained for that decision. That fixed policy is the reference unless the best
            fixed arm is cheaper by more than the noise (wording added after run 1; the rule is unchanged).
  validity  for each request, every arm's answer digest must agree (grouped batch-1 is exact); a
            mismatch, a non-finite cost, a failed calibration or fewer than two complete traces per
            level makes the run invalid, and an invalid run releases no learner.

The report keeps token counts, times and SHA-256 digests of answers, never prompt or answer text.
To run it on Kaggle (T4 x2) from a pushed commit:

    python tools/rl2_headroom.py --write-kernel kernel --owner <user> --commit <sha>
    kaggle kernels push -p kernel --accelerator NvidiaTeslaT4
    kaggle kernels output <user>/ironmule-rl2-headroom -p results
    python tools/rl2_headroom.py --analyse results/rl2_headroom_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable

MODEL = "mlx-community/Qwen3-1.7B-bf16"
SMOKE_MODEL = "Qwen/Qwen3-0.6B"
WIDTHS = (1, 2, 4, 8)
GATHERS_MS = (0, 50, 150)
REFERENCE = "shortest/w8/g0"
LOADS = (0.2, 0.6, 0.95)
WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
TRACES = 6
REQUESTS = 16
MAX_TOKENS = (64, 128)
RESAMPLES = 2000
DECISIONS = ("joint", "placement", "schedule")
# Generic questions; the report keeps only their index, never the text or the answers.
PROMPTS = ("Explain how a refrigerator keeps food cold.", "What causes the seasons on Earth?",
           "Describe how a bicycle gear system works.", "Why is the sky blue during the day?",
           "Summarize the water cycle in a few sentences.", "How does a compiler differ from an interpreter?",
           "What is the difference between weather and climate?", "Explain what a hash table is.",
           "How do vaccines train the immune system?", "Why do metals conduct electricity?",
           "Describe the life cycle of a star like the Sun.", "What is photosynthesis?",
           "How does a binary search work?", "Explain the greenhouse effect.",
           "What does a CPU cache do?", "How do airplanes stay in the air?")


# ---------------------------------------------------------------- the plan, fixed before data

def arm_name(placement: str, width: int, gather_ms: int) -> str:
    return f"{placement}/w{width}/g{gather_ms}"


def parse_arm(name: str) -> tuple[str, int, int]:
    placement, width, gather = name.split("/")
    return placement, int(width[1:]), int(gather[1:])


def arms(gpus: int) -> list[str]:
    """Every setting, the reference first; width 1 never waits, and one GPU has no placement."""
    schedules = [(w, g) for w in WIDTHS for g in GATHERS_MS if w > 1 or g == 0]
    placements = ("shortest", "fill") if gpus > 1 else ("shortest",)
    names = [arm_name(p, w, g) for p in placements for w, g in schedules]
    names.remove(REFERENCE)
    return [REFERENCE, *names]


def trace(level: int, index: int, rate: float, requests: int = REQUESTS,
          max_tokens: tuple[int, ...] | None = None) -> list[dict[str, Any]]:
    """Poisson arrivals at `rate` per second with prompt and output limit from the same fixed seed."""
    rng = random.Random(1000 * level + index)
    offset, items = 0.0, []
    for _ in range(requests):
        items.append({"offset_s": round(offset, 6), "prompt": rng.randrange(len(PROMPTS)),
                      "max_tokens": rng.choice(max_tokens or MAX_TOKENS)})
        offset += rng.expovariate(rate)
    return items


def block_order(names: list[str], block: int) -> list[tuple[str, int]]:
    """(arm, run): the reference, the others rotated by the block (reversed on odd blocks), the reference again."""
    others = names[1:]
    shift = block % len(others) if others else 0
    others = others[shift:] + others[:shift]
    if block % 2:
        others.reverse()
    return [(names[0], 0), *((name, 0) for name in others), (names[0], 1)]


# ---------------------------------------------------------------- analysis (no hardware)

def request_cost(request: dict[str, Any], deadline_s: float) -> float:
    """The live learner's reward contract: time in system + time to first content, or the failure penalty."""
    if request.get("outcome") != "ok":
        return (1 + 1.0) * deadline_s + 1.0
    first = request["first_s"] if request.get("first_s") is not None else request["end_s"]  # no visible text
    return float(request["end_s"]) + float(first)


def _cell_costs(rows: list[dict[str, Any]], deadline_s: float) -> dict[tuple[int, int, str, int], float]:
    return {(row["level"], row["trace"], row["arm"], row["run"]):
            statistics.fmean(request_cost(r, deadline_s) for r in row["requests"]) for row in rows}


def _choose(table: dict[tuple[int, int, str], float], levels: list[int], traces: list[int],
            names: list[str], decision: str) -> tuple[dict[int, str], str]:
    """Per level, the cheapest arm the decision may vary, and the cheapest single arm, on `traces`."""
    def mean(level: int, name: str) -> float:
        return statistics.fmean(table[level, t, name] for t in traces)

    fixed = min(names, key=lambda n: (sum(WEIGHTS[level] * mean(level, n) for level in levels), n))
    placement, width, gather = parse_arm(fixed)
    if decision == "placement":
        allowed = [n for n in names if parse_arm(n)[1:] == (width, gather)]
    elif decision == "schedule":
        allowed = [n for n in names if parse_arm(n)[0] == placement]
    else:
        allowed = list(names)
    return {level: min(allowed, key=lambda n: (mean(level, n), n)) for level in levels}, fixed


def _score(table: dict[tuple[int, int, str], float], levels: list[int], traces: list[int],
           oracle: dict[int, str], fixed: str) -> tuple[float, float]:
    """Mean weighted cost of the oracle's arms and of the fixed arm over `traces`."""
    def cost(pick: Any) -> float:
        return statistics.fmean(sum(WEIGHTS[level] * table[level, t, pick(level)] for level in levels)
                                for t in traces)

    return cost(lambda level: oracle[level]), cost(lambda level: fixed)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def analyse(report: dict[str, Any], resamples: int = RESAMPLES, seed: int = 0) -> dict[str, Any]:
    """Headroom per decision, in sample and held out, against the reference's A/A noise; and validity."""
    rows, deadline = report.get("rows", []), float(report.get("deadline_s") or 120)
    names = report.get("arms") or []
    levels = list(range(len(report.get("rates") or LOADS)))
    problems: list[str] = []
    if report.get("calibration", {}).get("failed"):
        problems.append("calibration failed: " + str(report["calibration"]["failed"]))
    digests: dict[tuple[int, int, int], set[str]] = {}
    for row in rows:
        for request in row["requests"]:
            if request.get("outcome") == "ok" and request.get("digest"):
                digests.setdefault((row["level"], row["trace"], request["index"]), set()).add(request["digest"])
    mismatched = sorted(key for key, found in digests.items() if len(found) > 1)
    if mismatched:
        problems.append(f"{len(mismatched)} requests answered differently across arms (first: {list(mismatched[0])})")
    cells = _cell_costs(rows, deadline)
    if any(not math.isfinite(value) for value in cells.values()):
        problems.append("a cell cost is not finite")
    complete = [t for t in sorted({row["trace"] for row in rows})
                if all((level, t, name, 0) in cells and (level, t, REFERENCE, 1) in cells
                       for level in levels for name in names)]
    if len(complete) < 2 or not names:
        problems.append(f"{len(complete)} complete traces; at least two are needed in every level")
    out: dict[str, Any] = {"schema": "ironmule.rl2_headroom.analysis.v1", "arms": len(names),
                           "complete_traces": complete, "weights": list(WEIGHTS),
                           "mismatched_requests": len(mismatched)}
    if problems:
        return {**out, "valid": False, "problems": problems, "verdict": "invalid: no learner is released"}
    table = {(level, t, name): cells[level, t, name, 0] for level in levels for t in complete for name in names}
    pairs = [(cells[level, t, REFERENCE, 0], cells[level, t, REFERENCE, 1]) for level in levels for t in complete]
    relative = [abs(a - b) / ((a + b) / 2) for a, b in pairs]
    noise = _quantile(relative, 0.95)
    rng = random.Random(seed)
    decisions: dict[str, Any] = {}
    for decision in DECISIONS if any(parse_arm(n)[0] == "fill" for n in names) else ("joint",):
        oracle, fixed = _choose(table, levels, complete, names, decision)
        o, f = _score(table, levels, complete, oracle, fixed)
        held = []
        for t in complete:  # leave one trace out: choose on the others, score on it
            train = [u for u in complete if u != t]
            held.append(_score(table, levels, [t], *_choose(table, levels, train, names, decision)))
        h_out = 1 - sum(x for x, _ in held) / sum(y for _, y in held)
        boot = []
        for _ in range(resamples):
            sample = [held[rng.randrange(len(held))] for _ in held]
            boot.append(1 - sum(x for x, _ in sample) / sum(y for _, y in sample))
        low = _quantile(boot, 0.025)
        decisions[decision] = {
            "fixed": fixed, "oracle": {str(level): oracle[level] for level in levels},
            "headroom_in_sample": 1 - o / f, "headroom_held_out": h_out,
            "headroom_held_out_ci95": [low, _quantile(boot, 0.975)],
            "verdict": "headroom: train and qualify a learner" if low > noise else "no headroom: a fixed policy ships"}
    reference = sum(WEIGHTS[level] * statistics.fmean(table[level, t, REFERENCE] for t in complete) for level in levels)
    best = decisions["joint"]["fixed"]
    best_cost = sum(WEIGHTS[level] * statistics.fmean(table[level, t, best] for t in complete) for level in levels)
    per_level = {str(level): {name: statistics.fmean(table[level, t, name] for t in complete) for name in names}
                 for level in levels}
    return {**out, "valid": True, "problems": [], "aa_noise": noise, "aa_pairs": len(pairs),
            "reference_cost": reference, "best_fixed": best, "best_fixed_vs_reference": best_cost / reference,
            "decisions": decisions, "mean_cost_per_level": per_level,
            "verdict": ("headroom for: " + ", ".join(d for d, v in decisions.items() if v["verdict"].startswith("headroom"))
                        if any(v["verdict"].startswith("headroom") for v in decisions.values())
                        else f"no headroom: the fixed policy {best} ships, no learner" if best_cost / reference < 1 - noise
                        else f"no headroom: the reference stays ({best} is within the A/A noise of it), no learner")}


# ---------------------------------------------------------------- measurement (hardware)

def _placement_pool(pool: Any) -> Any:
    """The started pool, whose placement can also `fill`: the lowest-index GPU with fewer than
    `fill_width` requests outstanding, else HW1's shortest queue. Only this tool uses `fill`."""
    from ironmule_product.device_pool import DevicePool

    class PlacementPool(DevicePool):
        placement, fill_width = "shortest", 8

        def _order(self) -> list[int]:
            order = super()._order()
            if self.placement != "fill":
                return order
            room = sorted(index for index in order if self.shards[index].outstanding() < self.fill_width)
            return room + [index for index in order if index not in room]

    return PlacementPool(pool.shards, pool.devices)


def set_arm(pool: Any, name: str) -> None:
    from ironmule_product.serving_policy import ServingPolicy

    placement, width, gather = parse_arm(name)
    pool.placement, pool.fill_width = placement, width
    for shard in pool.shards:
        shard.policy = ServingPolicy(max_width=width, gather_ms=gather, source="rl2-h")


def wait_idle(pool: Any, timeout_s: float = 120) -> bool:
    until = time.monotonic() + timeout_s
    while time.monotonic() < until:
        health = pool.health()
        if health.get("queued_requests", 0) == 0 and health.get("active_requests", 0) == 0:
            return True
        time.sleep(0.05)
    return False


def replay(pool: Any, model: str, items: list[dict[str, Any]], deadline_s: float) -> list[dict[str, Any]]:
    """Send each request at its scheduled offset through the pool; times are from the scheduled arrival."""
    from ironmule_product.errors import Overloaded, ProductError, RequestTimeout
    from ironmule_product.types import GenerationRequest

    results: list[dict[str, Any]] = [{} for _ in items]
    start = time.monotonic() + 0.05

    def client(index: int, item: dict[str, Any]) -> None:
        scheduled = start + item["offset_s"]
        out: dict[str, Any] = {"index": index, "outcome": "failed", "first_s": None, "end_s": None,
                               "digest": None, "tokens": 0}
        ids: list[Any] = []
        request = GenerationRequest(model, (("user", PROMPTS[item["prompt"]]),), max_tokens=item["max_tokens"],
                                    stream=True)
        try:
            for event in pool.stream(request):
                if event.get("type") == "token":
                    if out["first_s"] is None and event.get("text"):
                        out["first_s"] = time.monotonic() - scheduled
                    ids.append(event.get("token_id") if event.get("token_id") is not None else event.get("text"))
                elif event.get("type") == "done":
                    out["outcome"], out["tokens"] = "ok", int(event.get("completion_tokens") or 0)
        except Overloaded:
            out["outcome"] = "rejected"
        except RequestTimeout:
            out["outcome"] = "timeout"
        except ProductError as exc:
            out["error"] = exc.code
        out["end_s"] = time.monotonic() - scheduled
        if out["outcome"] == "ok":
            out["digest"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        results[index] = out

    threads = []
    for index, item in enumerate(items):
        delay = start + item["offset_s"] - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        thread = threading.Thread(target=client, args=(index, item), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join(deadline_s + 60)
    return results


def calibrate(pool: Any, model: str, deadline_s: float, max_tokens: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Requests completed per second under the reference from a burst of 16 per GPU at once."""
    set_arm(pool, REFERENCE)
    burst = [{**item, "offset_s": 0.0} for item in trace(9, 0, 1.0, 16 * len(pool.shards), max_tokens)]
    done = replay(pool, model, burst, deadline_s)
    ok = [r for r in done if r["outcome"] == "ok"]
    outcomes: dict[str, int] = {}
    for r in done:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    out: dict[str, Any] = {"requests": len(done), "ok": len(ok), "outcomes": outcomes}
    if len(ok) < len(done) / 2 or not ok:
        return {**out, "failed": "fewer than half of the calibration burst completed"}
    return {**out, "makespan_s": max(r["end_s"] for r in ok), "capacity_rps": len(ok) / max(r["end_s"] for r in ok)}


def measure(pool: Any, model: str, *, deadline_s: float, traces: int = TRACES, requests: int = REQUESTS,
            loads: tuple[float, ...] = LOADS, names: list[str] | None = None,
            max_tokens: tuple[int, ...] | None = None, on_row: Any = None) -> dict[str, Any]:
    """The plan above on a started pool: calibration, then every (trace, level) block in order.
    `names` and `max_tokens` shrink it for a smoke run; a measurement keeps the defaults."""
    pool = _placement_pool(pool)
    names = names or arms(len(pool.shards))
    warm = replay(pool, model, [{"offset_s": 0.0, "prompt": 0, "max_tokens": 8}] * len(pool.shards), deadline_s)
    report: dict[str, Any] = {"arms": names, "devices": list(pool.devices), "deadline_s": deadline_s,
                              "warmup_ok": sum(r["outcome"] == "ok" for r in warm), "rows": []}
    report["calibration"] = calibrate(pool, model, deadline_s, max_tokens)
    if report["calibration"].get("failed"):
        return report
    report["rates"] = [load * report["calibration"]["capacity_rps"] for load in loads]
    block = 0
    for t in range(traces):
        for level, rate in enumerate(report["rates"]):
            items = trace(level, t, rate, requests, max_tokens)
            for name, run in block_order(names, block):
                if not wait_idle(pool):
                    report["stopped"] = f"the pool did not drain before block {block}"
                    return report
                set_arm(pool, name)
                started = time.time()
                row = {"level": level, "trace": t, "arm": name, "run": run, "block": block,
                       "started_unix": started, "requests": replay(pool, model, items, deadline_s)}
                report["rows"].append(row)
                if on_row is not None:
                    on_row(report)
            block += 1
    return report


def _ironmule(*arguments: str, state: Path, timeout: float = 600) -> dict[str, Any]:
    code = "import sys, ironmule_cli; sys.exit(ironmule_cli.main(sys.argv[1:]))"
    try:
        done = subprocess.run([sys.executable, "-c", code, *arguments, "--state-dir", str(state)],
                              capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"returncode": None, "error": f"{type(exc).__name__}: {exc}"[:400]}
    return {"returncode": done.returncode, "stdout": done.stdout[-2000:], "stderr": done.stderr[-2000:]}


def environment() -> dict[str, Any]:
    from importlib import metadata

    versions = {}
    for name in ("ironmule", "mlx", "mlx-lm", "mlx-cuda"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        gpus = ""
    return {"python": platform.python_version(), "machine": platform.machine(), "cpus": os.cpu_count(),
            "versions": versions, "gpus": gpus.strip().splitlines()}


def run(model: str, work: Path, *, smoke: bool, on_row: Any = None) -> dict[str, Any]:
    """Register `model`, start one worker per GPU (the tuned engine's grouped path) and measure."""
    from huggingface_hub import snapshot_download

    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.device_pool import cuda_devices, start_pool
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore

    revision = Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                             "*.jinja", "tokenizer*"])).name
    state = work / "rl2h-state"
    setup = _ironmule("setup", "--mode", "server", state=state)
    added = _ironmule("models", "add", model, state=state)
    if added.get("returncode") != 0:
        return {"model": model, "revision": revision, "failed": "models add", "setup": setup, "added": added}
    store = ProductStore(state)
    spec = store.model(model)
    devices = cuda_devices() or [None]
    pool = start_pool(store, spec, devices, lambda device: MLXWorkerClient(
        spec, execution_variant="current_engine", cuda_device=device), ProductService)
    try:
        deadline = float(store.settings()["request_timeout_s"] or 120)
        sizes = ({"traces": 2, "requests": 3, "max_tokens": (4, 8),
                  "names": [n for n in arms(len(devices)) if parse_arm(n)[1:] in ((8, 0), (1, 0))]}
                 if smoke else {})
        result = measure(pool, model, deadline_s=deadline, on_row=on_row, **sizes)
    finally:
        pool.close()
    return {"model": model, "revision": revision, **result}


REPOSITORY = "https://github.com/Tobayko/IronMule"
_BOOTSTRAP = '''"""IronMule RL2-H on this Kaggle session; written by tools/rl2_headroom.py."""
from __future__ import annotations
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "ironmule[cuda] @ git+{repository}@{commit}"], check=True)
'''


def write_kernel(directory: Path, owner: str, commit: str) -> Path:
    """A private script kernel that installs IronMule at `commit` and runs this file; the report
    lands in the kernel's output as `rl2_headroom_report.json`."""
    if not commit or any(c not in "0123456789abcdef" for c in commit) or len(commit) != 40:
        raise ValueError("commit must be a full 40-character lowercase SHA, so the kernel runs exactly it")
    directory.mkdir(parents=True, exist_ok=True)
    body = (Path(__file__).read_text().replace('\nif __name__ == "__main__":\n    sys.exit(main())\n', "")
            .replace("from __future__ import annotations\n", "", 1))
    entry = "\nsys.exit(main(['--out', '/kaggle/working/rl2_headroom_report.json']))\n"
    (directory / "run.py").write_text(_BOOTSTRAP.format(repository=REPOSITORY, commit=commit) + body + entry)
    metadata = {"id": f"{owner}/ironmule-rl2-headroom", "title": "ironmule-rl2-headroom", "code_file": "run.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1) + "\n")
    return directory


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the measurement report here")
    parser.add_argument("--model", default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny sizes: checks this tool, measures nothing")
    parser.add_argument("--analyse", type=Path, metavar="REPORT", help="print the analysis of a finished report")
    parser.add_argument("--write-kernel", type=Path, metavar="DIR",
                        help="write a Kaggle kernel folder instead of running (needs --owner and --commit)")
    parser.add_argument("--owner")
    parser.add_argument("--commit")
    args = parser.parse_args(None if argv is None else list(argv))
    if args.write_kernel is not None:
        if not args.owner or not args.commit:
            parser.error("--write-kernel needs --owner and --commit")
        print(json.dumps({"kernel": str(write_kernel(args.write_kernel, args.owner, args.commit))}))
        return 0
    if args.analyse is not None:
        report = json.loads(args.analyse.read_text())
        print(json.dumps(analyse(report.get("rl2h", report)), indent=1, sort_keys=True))
        return 0
    if args.out is None:
        parser.error("--out is required to run")
    report: dict[str, Any] = {"schema": "ironmule.rl2_headroom.v1", "smoke": args.smoke, "started_unix": time.time(),
                              "env": environment()}

    def save(partial: dict[str, Any] | None = None) -> None:
        if partial is not None:
            report["rl2h"] = partial
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")

    save()
    work = Path(tempfile.mkdtemp(prefix="ironmule-rl2h-"))
    model = args.model or (SMOKE_MODEL if args.smoke else MODEL)
    report["rl2h"] = run(model, work, smoke=args.smoke, on_row=save)
    report["analysis"] = analyse(report["rl2h"])
    report["finished_unix"] = time.time()
    save()
    print(json.dumps({"report": str(args.out), "verdict": report["analysis"]["verdict"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
