#!/usr/bin/env python3
"""Why the 12B autotuner stopped, measured on the process instead of guessed from it.

The 12B tune printed `confirming the screening winner with a paired A/B ...` and produced
nothing after it. That is consistent with a signal, but the run captured no return code
and no wait status, so nothing about *how* it ended was recorded. The unified log for that
window shows the kernel reaping idle processes once a second while the compressor grew to
`23.08 GB` on a `34.36 GB` machine -- pressure, not a cause.

This starts **one** confirmation-shaped child on a fixed, deliberately small budget and
records what the aborted run did not: the return code, the decoded wait status, unbuffered
output as it was produced, start and end, the child's own peak RSS from `RUSAGE_CHILDREN`,
MLX active and peak from inside the child, and free memory, swap, swapin/swapout/pageout
counters and load around it.

**No timing statement can be read out of this file.** One repeat, eight tokens and a
resource probe every two seconds are a lifecycle instrument, not a benchmark. Condition `A`
and condition `B` differ only in `wired_fraction`, and `B` surviving where `A` does not is
one observation, not a cause: the machine is warmer, more fragmented and more swapped when
`B` runs, which is the conservative direction but not a proof.

Nothing here changes the inference path. The child loads through the same `load_engine`
and runs behind the same Q3f no-detach guard as a real confirmation child; only the
measurement around it is new. No foreign process is signalled, nothing is purged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "mlx-community/gemma-3-12b-it-4bit"
MEASURED_SOURCES = ("ironmule/ab.py", "ironmule/tune.py", "ironmule/runtime.py",
                    "ironmule/q3f_child_guard.py", "tools/b60_12b_child_diagnosis.py")

# The budget, fixed here before anything runs. It is a fraction of a confirmation:
# CONFIRM_PROCESSES=6 and CONFIRM_REPEATS=7 become one child and one repeat.
MAX_TOKENS = 8
REPEATS = 1
WARMUP = 1
CHILD_TIMEOUT_S = 900.0
SAMPLE_INTERVAL_S = 2.0
LOAD_CEILING = 4.0          # the same ceiling every other harness here uses
MIN_FREE_PERCENT = 10.0     # refuse to start a second child into an already tight machine
RECOVERY_TIMEOUT_S = 600.0

# What the screening actually recorded, and what it did not. The tune died before
# `save_profile`, and the surviving stdout was piped through a tail, so only its last
# three decisions are on record. The first six search steps are NOT known.
SCREENING_RECORD = {
    "source": "task output of the aborted run, 2026-09-10 13:31 local, truncated by a tail",
    "known_decisions": [
        {"knob": "capacity_slack", "value": 128, "ratio": 0.9236, "verdict": "rejected: no gain"},
        {"knob": "wired_fraction", "value": 0.6, "ratio": 0.9034, "verdict": "kept"},
        {"knob": "fuse_projections", "value": True, "ratio": 0.8978, "verdict": "kept"},
    ],
    "unknown": ("compiled_fixed_cache", "fused_argmax", "head_skip_prefill",
                "prefill_into_fixed", "readback_every", "speculate_k"),
    "reconstruction_rule": (
        "the six unknown steps take the values the confirmed 4B profile on this same "
        "machine kept, and the three known decisions are applied on top. This is a "
        "RECONSTRUCTION, not the frozen screening winner, and no result from it may be "
        "written back into any profile"
    ),
}

# 4B confirmed profile knobs on this fingerprint, plus the two 12B keeps.
CONDITION_A = {
    "compiled_fixed_cache": True, "fused_argmax": True, "head_skip_prefill": True,
    "prefill_into_fixed": False, "readback_every": 2, "speculate_k": 0,
    "speculate_ngram": 3, "capacity_slack": 0,
    "wired_fraction": 0.6, "fuse_projections": True, "k3840_matvec": False,
}
CONDITION_B = dict(CONDITION_A, wired_fraction=0.0)
CONDITIONS = {"A": CONDITION_A, "B": CONDITION_B}

PREREGISTRATION = {
    "experiment": "B60_12b_child_diagnosis",
    "question": (
        "how does a single 12B confirmation-shaped child end, and does the wired-limit "
        "knob the screening kept change that ending"
    ),
    "not_a_performance_study": (
        "one child, one repeat, eight tokens, and a resource probe every two seconds "
        "inside the run. No ratio, tokens-per-second or latency from this file means "
        "anything and none is computed"
    ),
    "model": MODEL,
    "budget": {"max_tokens": MAX_TOKENS, "repeats": REPEATS, "warmup": WARMUP,
               "children_per_condition": 1, "child_timeout_seconds": CHILD_TIMEOUT_S},
    "conditions": {"A": CONDITION_A, "B": CONDITION_B},
    "condition_order": (
        "fixed A then B, no rotation. A first on purpose: B then runs on the warmer, more "
        "fragmented machine A left behind, so a surviving B is observed under conditions "
        "at least as hard as A's, never easier"
    ),
    "screening_record": SCREENING_RECORD,
    "captured": [
        "Popen return code", "decoded wait status and signal name", "unbuffered stdout",
        "unbuffered stderr", "start and end wall clock", "child peak RSS via RUSAGE_CHILDREN",
        "MLX active and peak bytes reported from inside the child", "free memory percent",
        "swap used bytes", "swapins", "swapouts", "pageouts", "load average",
    ],
    "probe_placement": (
        "every probe runs in the parent process while it waits on the child. No probe runs "
        "inside the child and none runs inside a dispatch. The parent holds no model"
    ),
    "gates_before_each_child": (
        f"no other model process (gpu_busy), 1-minute load below {LOAD_CEILING}, system "
        f"free memory at or above {MIN_FREE_PERCENT} per cent"
    ),
    "condition_b_precondition": (
        "B starts only if the A child started, was reaped, and left the machine recovered "
        "to the same start gates within ten minutes. A child that could not be reaped "
        "stops the study"
    ),
    "causality_rule": (
        "B surviving where A does not is one observation of one machine state. It is "
        "reported as such and is not written up as a cause. No condition is repeated to "
        "reach a preferred ending"
    ),
    "safety": (
        "no foreign process is signalled, nothing is purged, no system setting is changed, "
        "and no threshold in any existing harness is moved. Only the one child this file "
        "starts is ever terminated, and only on the fixed timeout"
    ),
    "forbidden_outputs": (
        "no profile is written, no stack is composed, and no 12B result is qualified from "
        "this file. Its only product is how the process ended and what the machine did"
    ),
}

_FREE_PERCENT = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
_VM_STAT_FIELD = {
    "swapins": re.compile(r"Swapins:\s*([0-9]+)"),
    "swapouts": re.compile(r"Swapouts:\s*([0-9]+)"),
    "pageouts": re.compile(r"Pageouts:\s*([0-9]+)"),
    "pages_free": re.compile(r"Pages free:\s*([0-9]+)"),
    "pages_occupied_by_compressor": re.compile(r"Pages occupied by compressor:\s*([0-9]+)"),
}

# The child: the same guard, the same loader, one arm, and its own memory report.
CHILD_BOOTSTRAP = (
    "import importlib.util,json,os,sys;"
    "guard_path=os.path.realpath(sys.argv[2]);"
    "ab_path=os.path.realpath(sys.argv[3]);"
    "harness_path=os.path.realpath(sys.argv[4]);"
    "guard_spec=importlib.util.spec_from_file_location('ironmule.q3f_child_guard',guard_path);"
    "guard=importlib.util.module_from_spec(guard_spec);"
    "sys.modules['ironmule.q3f_child_guard']=guard;"
    "guard_spec.loader.exec_module(guard);"
    "guard.assert_source_surface(ab_path);"
    "guard.install();"
    "spec_module=importlib.util.spec_from_file_location('b60_child',harness_path);"
    "child_module=importlib.util.module_from_spec(spec_module);"
    "spec_module.loader.exec_module(child_module)\n"
    "try:\n"
    "    result=child_module.child_main(json.loads(sys.argv[1]))\n"
    "except BaseException as exc:\n"
    "    print('@!'+repr(exc),flush=True); raise\n"
    "else:\n"
    "    print('@@'+json.dumps(result,sort_keys=True,allow_nan=False),flush=True)\n"
    "finally:\n"
    "    guard.uninstall()"
)


def child_main(spec: dict) -> dict:
    """One arm, in this process, reporting its own MLX memory at every stage."""
    import mlx.core as mx

    from ironmule.runtime import Knobs
    from ironmule.tune import _eos_ids, load_engine, prompt_ids, DEFAULT_PROMPT

    stages: list[dict] = []

    def stage(label: str) -> None:
        stages.append({"stage": label, "monotonic": time.monotonic(),
                       "mlx_active_bytes": int(mx.get_active_memory()),
                       "mlx_peak_bytes": int(mx.get_peak_memory()),
                       "mlx_cache_bytes": int(mx.get_cache_memory()),
                       "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)})

    mx.reset_peak_memory()
    stage("before_load")
    print("@stage before_load", flush=True)
    engine = None
    try:
        engine, tokenizer = load_engine(spec["model"], Knobs(**spec["knobs"]))
        stage("after_load")
        print("@stage after_load", flush=True)
        ids = prompt_ids(tokenizer, spec.get("prompt", DEFAULT_PROMPT))
        eos = _eos_ids(tokenizer)
        for _ in range(spec["warmup"]):
            engine.generate(ids, spec["max_tokens"], eos)
        stage("after_warmup")
        print("@stage after_warmup", flush=True)
        runs = [engine.generate(ids, spec["max_tokens"], eos) for _ in range(spec["repeats"])]
        stage("after_repeats")
        print("@stage after_repeats", flush=True)
        tokens = [list(map(int, run["logical_tokens"])) for run in runs]
        result = {
            "pid": os.getpid(),
            "prompt_tokens": len(ids),
            "logical_tokens": tokens[0],
            "token_count": len(tokens[0]),
            "deterministic": all(t == tokens[0] for t in tokens),
            "stages": stages,
        }
    finally:
        if engine is not None:
            close = getattr(engine, "close", None)
            if close is not None:
                close()
    stage("after_close")
    result["stages"] = stages
    from ironmule import q3f_child_guard
    result["guard"] = q3f_child_guard.ledger()
    return result


def _command(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def sample(label: str, child_pid: int | None = None) -> dict:
    """One resource probe, taken in the parent while it waits. Never inside the child."""
    from ironmule.hw import swap_used_bytes

    vm = _command(["vm_stat"])
    row: dict = {"label": label, "at": datetime.now(timezone.utc).isoformat(),
                 "monotonic": time.monotonic(),
                 "swap_used_bytes": swap_used_bytes(),
                 "load_average": list(os.getloadavg()),
                 "parent_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    for name, pattern in _VM_STAT_FIELD.items():
        found = pattern.search(vm)
        row[name] = int(found.group(1)) if found else None
    free = _FREE_PERCENT.search(_command(["memory_pressure", "-Q"]))
    row["memory_free_percent"] = float(free.group(1)) if free else None
    if child_pid is not None:
        rss = _command(["ps", "-o", "rss=", "-p", str(child_pid)]).strip()
        row["child_rss_bytes"] = int(rss) * 1024 if rss.isdigit() else None
    return row


def source_binding() -> dict:
    digests, running = {}, hashlib.sha256()
    for name in MEASURED_SOURCES:
        data = (PROJECT_ROOT / name).read_bytes()
        digests[name] = hashlib.sha256(data).hexdigest()
        running.update(name.encode())
        running.update(data)
    return {"files": digests, "combined_sha256": running.hexdigest()}


def _decode_status(returncode: int | None) -> dict:
    """What ended the child, from the status Popen already decoded, not from a guess."""
    if returncode is None:
        return {"returncode": None, "exited": False, "signalled": False,
                "signal": None, "signal_name": None,
                "interpretation": "the child was never reaped"}
    if returncode >= 0:
        return {"returncode": returncode, "exited": True, "signalled": False,
                "signal": None, "signal_name": None,
                "interpretation": f"exited normally with status {returncode}"}
    number = -returncode
    try:
        name = signal.Signals(number).name
    except ValueError:
        name = None
    return {"returncode": returncode, "exited": False, "signalled": True,
            "signal": number, "signal_name": name,
            "interpretation": f"killed by signal {number} ({name or 'unknown'})"}


def gates_ok() -> tuple[bool, str, dict]:
    from ironmule.tune import gpu_busy

    probe = sample("gate")
    busy = gpu_busy()
    if busy:
        return False, f"another model process is running ({busy})", probe
    if probe["load_average"][0] > LOAD_CEILING:
        return False, (f"1-minute load average is {probe['load_average'][0]:.2f}, above the "
                       f"{LOAD_CEILING} ceiling"), probe
    free = probe["memory_free_percent"]
    if free is not None and free < MIN_FREE_PERCENT:
        return False, f"system free memory is {free:.1f} per cent, below {MIN_FREE_PERCENT}", probe
    return True, "", probe


def run_child(condition: str, model: str = MODEL) -> dict:
    """Start one child, sample around it, and record exactly how it ended."""
    spec = {"model": model, "knobs": CONDITIONS[condition], "max_tokens": MAX_TOKENS,
            "repeats": REPEATS, "warmup": WARMUP}
    guard_path = str(PROJECT_ROOT / "ironmule" / "q3f_child_guard.py")
    ab_path = str(PROJECT_ROOT / "ironmule" / "ab.py")
    harness_path = str(Path(__file__).resolve())
    before_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    samples = [sample("before_start")]
    started_at = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", CHILD_BOOTSTRAP, json.dumps(spec),
         guard_path, ab_path, harness_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
             "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"},
    )
    stop = threading.Event()

    def sampler() -> None:
        while not stop.wait(SAMPLE_INTERVAL_S):
            samples.append(sample("during", process.pid))

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()                       # only this file's own child, on the fixed budget
        stdout, stderr = process.communicate()
    finally:
        stop.set()
        thread.join(timeout=5)

    ended_monotonic = time.monotonic()
    samples.append(sample("after_end"))
    after_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    status = _decode_status(process.returncode)
    marker = next((line[2:] for line in stdout.splitlines() if line.startswith("@@")), None)
    record: dict = {
        "condition": condition,
        "knobs": CONDITIONS[condition],
        "pid": process.pid,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": ended_monotonic - started_monotonic,
        "timed_out": timed_out,
        "process_end": status,
        "stdout": stdout,
        "stderr": stderr,
        "child_peak_rss_bytes_rusage": int(after_children),
        "child_peak_rss_bytes_rusage_before": int(before_children),
        "stage_markers": [line for line in stdout.splitlines() if line.startswith("@stage")],
        "child_record": json.loads(marker) if marker else None,
        "samples": samples,
    }
    return record


def recovered() -> tuple[bool, dict]:
    """Wait for the machine to come back to the start gates after a child, or give up."""
    deadline = time.monotonic() + RECOVERY_TIMEOUT_S
    while True:
        ok, reason, probe = gates_ok()
        if ok:
            return True, probe
        if time.monotonic() >= deadline:
            probe["blocked_by"] = reason
            return False, probe
        time.sleep(15)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preregister", type=Path)
    parser.add_argument("--conditions", default="A,B",
                        help="fixed order; 'A' alone runs only the single-child step")
    parser.add_argument("--model", default=MODEL,
                        help="smoke-test escape only: a small model proves the child "
                             "plumbing without spending a 12B attempt on a bootstrap bug. "
                             "The record always names the model that actually ran")
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once  # imported here so the
    # child, which executes this module under the guard, never loads it

    if args.preregister:
        write_once(args.preregister, json.dumps(
            {"preregistration": PREREGISTRATION, "source_binding": source_binding(),
             "written_at": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True))
        print(f"preregistration written to {args.preregister}")
        return 0

    wanted = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown condition(s): {unknown}")

    runs: list[dict] = []
    stopped = None
    for index, condition in enumerate(wanted):
        if index:
            ok, probe = recovered()
            if not ok:
                stopped = {"after": wanted[index - 1], "reason": probe.get("blocked_by"),
                           "probe": probe}
                break
        ok, reason, probe = gates_ok()
        if not ok:
            stopped = {"before": condition, "reason": reason, "probe": probe}
            break
        print(f"condition {condition} starting", flush=True)
        run = run_child(condition, args.model)
        runs.append(run)
        print(f"condition {condition}: {run['process_end']['interpretation']}, "
              f"{run['duration_s']:.1f} s", flush=True)
        if run["process_end"]["returncode"] != 0 and len(wanted) > index + 1:
            # A child that did not exit cleanly is the finding. Do not start the next one
            # into a machine whose state this one may have left behind.
            stopped = {"after": condition,
                       "reason": "the child did not exit cleanly; the study stops here"}
            break

    record = {
        "experiment": PREREGISTRATION["experiment"],
        "preregistration": PREREGISTRATION,
        "source_binding": source_binding(),
        "environment": {"platform": platform.platform(),
                        "python": sys.version.split()[0],
                        "model": args.model,
                        "is_the_preregistered_model": args.model == MODEL},
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "stopped": stopped,
        "verdict": ("NO_CHILD_RAN" if not runs
                    else "ALL_CHILDREN_SURVIVED"
                    if all(r["process_end"]["returncode"] == 0 for r in runs)
                    else "A_CHILD_DID_NOT_EXIT_CLEANLY"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": record["verdict"], "stopped": stopped,
                      "runs": [{"condition": r["condition"],
                                "end": r["process_end"],
                                "duration_s": round(r["duration_s"], 1),
                                "child_peak_rss_gb": round(r["child_peak_rss_bytes_rusage"] / 1e9, 2),
                                "stages": [s["stage"] for s in
                                           (r["child_record"] or {}).get("stages", [])]}
                               for r in runs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
