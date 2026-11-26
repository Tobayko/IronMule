"""One four-core CPU-only Pilot experiment during an existing pod's GPU-idle interval."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SMOL = "2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d"


def note(path, record):
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(record, indent=2) + "\n")
    partial.replace(path)


def terminate(signum, _frame):
    raise InterruptedError(f"stopped by signal {signum}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--cpus", type=int, nargs=4, required=True)
    p.add_argument("--seconds", type=int, required=True, choices=range(1, 361), metavar="1..360")
    args = p.parse_args()
    cpus = set(args.cpus)
    if len(cpus) != 4 or not cpus.issubset(os.sched_getaffinity(0)):
        p.error("select four distinct CPUs allowed by this host")
    topology = []
    for cpu in sorted(cpus):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        topology.append(tuple((base / field).read_text().strip() for field in ("physical_package_id", "core_id")))
    if len(set(topology)) != 4:
        p.error("select separate physical cores")
    os.sched_setaffinity(0, cpus)
    os.nice(10)
    os.chdir(ROOT)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--", "ironmule", "ironmule_product", "training",
                                "lab/rp/cpu_pilot.py", "lab/portable_gpu/bench.py", "ironmule_devices.py"]):
        raise ValueError("run from a clean frozen experiment checkout")
    gguf = args.gguf.resolve(strict=True)
    with gguf.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != SMOL:
            raise ValueError("expected pinned SmolLM2-135M Q4_K_M")
    state = args.state.resolve()
    state.mkdir(mode=0o700, parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "HOME", "LD_LIBRARY_PATH", "IRONMULE_GGML_BINDINGS", "LLAMA_CPP_LIB_PATH", "LANG")}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", IRONMULE_GGML_GPU_LAYERS="0",
               CUDA_VISIBLE_DEVICES="", IRONMULE_PILOT="1", PYTHONPATH=str(ROOT))
    record = {"schema": "ironmule.runpod.cpu-pilot.v1", "commit": commit, "model_sha256": SMOL,
              "cpus": sorted(cpus), "physical_cores": topology, "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
              "budget_seconds": args.seconds, "new_resource_bookings_usd": 0,
              "cost_source": "Occupied lease time is billed in the RunPod campaign journal.", "hypothesis": "Thread costs on this EPYC may differ from the Xeon trial.",
              "steps": []}
    start = time.monotonic()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    phases = [
        ("collect", ["training.pilot_threads", "collect", "--slots", "4", "--context-bands", "1", "4", "6",
                     "--widths", "1", "4", "--gguf", gguf, "--trials", state / "trials.json", "--state", state / "local"]),
        ("fit", ["training.pilot_archive", "--incoming", state / "trials.json", "--state", state / "local"]),
        ("serve", ["training.serving_trial", "--gguf", gguf, "--require-provider", "cpu", "--clients", "4",
                   "--threads", "2", "--waves", "4", "--tokens", "3", "--thread-candidate", state / "head.npz",
                   "--out", state / "serving.json", "--profile-dir", state / "waves"]),
    ]
    try:
        for name, argv in phases:
            step = {"name": name, "started_s": time.monotonic() - start}
            record["steps"].append(step)
            note(state / "campaign.json", record)
            if name == "serve":
                from shutil import copyfile

                trained = json.loads((state / "fit.log").read_text())
                if trained["state"] not in ("candidate published", "checkpoint unchanged"):
                    record["result"] = trained["state"]
                    return 1
                pilot_dir = state / "local" / "pilot"
                copyfile(pilot_dir / trained["checkpoint"], state / "head.npz")
                report = pilot_dir / (trained["binding"] + ".training") / (trained["fit_revision"] + ".json")
                copyfile(report, state / "replay.json")
            with (state / f"{name}.log").open("wb") as log:
                child = subprocess.Popen([sys.executable, "-m", *argv], stdout=log, stderr=subprocess.STDOUT,
                                         env=env, close_fds=True, start_new_session=True)
                try:
                    code = child.wait(timeout=max(0, args.seconds - (time.monotonic() - start)))
                finally:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
            step.update(returncode=code, elapsed_s=time.monotonic() - start - step["started_s"])
            if code:
                record["result"] = "stage failed or candidate unqualified"
                return code
        record["result"] = "qualified choices served"
        return 0
    except (InterruptedError, subprocess.TimeoutExpired) as exc:
        record["result"] = type(exc).__name__
        return 124
    finally:
        record["elapsed_s"] = time.monotonic() - start
        note(state / "campaign.json", record)


if __name__ == "__main__":
    raise SystemExit(main())
