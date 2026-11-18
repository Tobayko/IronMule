"""Repeat local, isolated server comparisons within a persistent time allowance."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import secrets
import shutil
import sys
import time
from pathlib import Path

from . import compare
from .state import _atomic_write


def _fingerprint(args, workload):
    """Cheap change detection. The comparison itself hashes the actual artifacts."""
    from ironmule import compare_learning, ggml_engine

    from . import compare_servers, compare_tokenizer

    executables = [args.ironmule or sys.executable, args.llama_server, args.vllm]
    modules = (compare, compare_servers, compare_tokenizer, compare_learning, ggml_engine)
    sources = [args.gguf, *(Path(module.__file__) for module in modules)]
    native_path = os.environ.get("LLAMA_CPP_LIB_PATH")
    if native_path:
        sources.extend(Path(native_path).glob("*.so*"))
    sources.extend(p for p in args.tokenizer.rglob("*") if p.is_file()
                   and not any(x.startswith(".") for x in p.relative_to(args.tokenizer).parts))
    for item in executables:
        if item:
            sources.append(Path(shutil.which(str(item)) or str(item)))
    stamps = []
    for source in sorted(sources):
        try:
            info = source.stat()
            stamps.append((str(source.resolve()), info.st_size, info.st_mtime_ns))
        except FileNotFoundError:
            stamps.append((str(source), None, None))
    # Quotas and affinity alter the available CPU even when the host name stays the same.
    quota = Path("/sys/fs/cgroup/cpu.max")
    identity = (compare_servers._device(os.environ.copy(), "ironmule"), platform.machine(), stamps, workload,
                quota.read_text().strip() if quota.is_file() else None,
                sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _watch(config, workload, args, folder):
    path = folder / "watch.json"
    state = json.loads(path.read_text()) if path.exists() else {"charged_seconds": 0, "cycles": 0}
    if not isinstance(state.get("charged_seconds"), int) or state["charged_seconds"] < 0:
        raise ValueError("invalid comparison watch accounting")
    while state["charged_seconds"] < config.total_seconds:
        fingerprint = _fingerprint(args, workload)
        wait = state.get("next_at", 0) - time.time()
        if fingerprint == state.get("fingerprint") and wait > 0 and not state.get("recheck"):
            time.sleep(min(60, wait))
            continue
        allowance = min(args.budget_seconds, config.total_seconds - state["charged_seconds"])
        cycle = state["cycles"] + 1
        # Reserve the full run before starting. A killed watch cannot overspend by restarting.
        previous = state["charged_seconds"]
        state.update(cycles=cycle, charged_seconds=previous + allowance, fingerprint=fingerprint,
                     recheck=False, next_at=time.time() + config.interval_seconds)
        _atomic_write(path, state)
        seed = (args.seed + cycle - 1) % (2**32) if config.fixed_seed else secrets.randbits(32)
        options = [*workload, "--seed", str(seed), "--budget-seconds", str(allowance),
                   "--agent-dir", str(args.agent_dir or folder / "agent")]
        try:
            before = json.loads(args.output.read_text()).get("run_id")
        except FileNotFoundError:
            before = None
        start = time.monotonic()
        code = compare.main(options)
        elapsed = max(1, math.ceil(time.monotonic() - start))
        state.update(charged_seconds=previous + elapsed, next_at=time.time() + config.interval_seconds,
                     last_code=code)
        if args.output.exists():
            report = json.loads(args.output.read_text())
            # A separately launched compare can own the live output file.
            ours = report.get("run_id") != before and report.get("protocol", {}).get("seed") == seed
            if ours and report.get("status") in {"complete", "error", "timeout"}:
                _atomic_write(folder / f"cycle-{cycle:06d}.json", report)
                state.update(last_run=report["run_id"], recheck=report.get("learning", {}).get("status") == "qualified")
        _atomic_write(path, state)
        if code == 130:
            return code
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="ironmule compare-watch", description=__doc__,
                                     epilog="Use -- followed by the normal ironmule compare options.")
    parser.add_argument("--total-seconds", required=True, type=int, help="cumulative comparison time across restarts")
    parser.add_argument("--interval-seconds", type=int, default=86400, help="time between scheduled comparisons")
    split = argv.index("--") if "--" in argv else len(argv)
    config = parser.parse_args(argv[:split])
    if min(config.total_seconds, config.interval_seconds) < 1 or split + 1 >= len(argv):
        parser.error("positive budgets and a comparison command after -- are required")
    workload = argv[split + 1:]
    config.fixed_seed = any(x == "--seed" or x.startswith("--seed=") for x in workload)
    args = compare.arguments(workload)
    folder = args.output.parent / (args.output.name + ".watch")
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (folder / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("A watch is already supervising this comparison file", file=sys.stderr)
            return 1
        try:
            return _watch(config, workload, args, folder)
        except KeyboardInterrupt:
            return 130
