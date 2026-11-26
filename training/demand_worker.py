"""Consume one serving demand ticket on a dedicated evaluator with a bounded CPU allocation."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ironmule_product.experiment_queue import host_id, validate
from training.code_trial import ROOT, save


def serving(directory):
    with (directory / ("serving-" + host_id())).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def consume(args):
    queue = args.queue.resolve(strict=True)
    state = args.state.resolve()
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wait
    pending = queue / "pending.json"
    with (queue / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not pending.exists():
            if time.monotonic() >= deadline:
                return "empty"
            time.sleep(.25)
        ticket = validate(json.loads(pending.read_bytes()))
        if serving(queue):
            return "deferred: serving process owns this host"
        run_id = "demand-" + ticket["id"]
        run_dir = state / "runs" / run_id
        if (run_dir / "trial/author/request.json").exists() and not (run_dir / "trial/author/response.json").exists():
            return "awaiting_author"
        command = [sys.executable, "-m", "training.optimization_run", "--gguf", str(args.gguf.resolve(strict=True)),
                   "--provider", args.provider, "--state", str(state), "--run-id", run_id,
                   "--demand-summary", str(pending)]
        if args.thread_candidate:
            command.extend(["--thread-candidate", str(args.thread_candidate.resolve(strict=True))])
        # The timeout guard lives in a separate process so a lost queue consumer still has
        # a wall-clock limit. Normal serving never runs in this process group.
        command = ["timeout", "--kill-after=5", str(args.seconds), *command]
        cpus = sorted(os.sched_getaffinity(0))[:args.cores]
        env = {**os.environ, "IRONMULE_CODE_AUTHOR": "hosted", "OMP_NUM_THREADS": str(len(cpus)),
               "OPENBLAS_NUM_THREADS": "1"}
        env.pop("IRONMULE_EXPERIMENT_QUEUE", None)
        log = state / (run_id + ".log")
        with log.open("ab") as out:
            child = subprocess.Popen(["nice", "-n", "10", "taskset", "-c", ",".join(map(str, cpus)), *command],
                                     cwd=ROOT, env=env, stdout=out, stderr=out, start_new_session=True)
            try:
                while child.poll() is None:
                    if serving(queue):
                        return "deferred: serving resumed"
                    time.sleep(.2)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        if (run_dir / "trial/author/request.json").exists() and not (run_dir / "trial/author/response.json").exists():
            return "awaiting_author"
        if child.returncode or not (run_dir / "complete.json").exists():
            raise RuntimeError(f"evaluation incomplete; ticket retained, see {log.name}")
        save(run_dir / "demand.json", ticket)
        pending.unlink()
        return "complete"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", required=True, type=Path)
    p.add_argument("--state", required=True, type=Path)
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--thread-candidate", type=Path)
    p.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--cores", type=int, choices=range(1, 33), default=2, metavar="1..32")
    p.add_argument("--seconds", type=int, choices=range(1, 1801), default=300, metavar="1..1800")
    p.add_argument("--wait", type=int, choices=range(0, 301), default=60, metavar="0..300")
    print(consume(p.parse_args()))


if __name__ == "__main__":
    main()
