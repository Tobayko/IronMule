"""Bounded information trial on the previously identified Xeon CPU and SmolLM2 model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

from training.code_author import write_once
from training.trial_release import require

PROTOCOL = Path(__file__).resolve().parents[1] / "autonomous/results/xeon-affinity-load/protocol.json"
CPU_STAT = Path("/sys/fs/cgroup/cpu.stat")
CLOCK_HZ = os.sysconf("SC_CLK_TCK")


def usage():
    return int(dict(line.split() for line in CPU_STAT.read_text().splitlines())["usage_usec"]) / 1e6


def proc_seconds(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().split(") ")[1].split()
    return (int(fields[11]) + int(fields[12])) / CLOCK_HZ


@contextmanager
def competitors(regime):
    code = ("import hashlib,os,sys; os.sched_setaffinity(0,{int(sys.argv[1])}); "
            "b=b'foreign'*16384\nwhile True: hashlib.sha256(b).digest()")
    with ExitStack() as clean:
        workers = []
        if regime == "occupied":
            for cpu in (1, 2):
                proc = subprocess.Popen([sys.executable, "-c", code, str(cpu)])
                clean.callback(proc.wait)
                clean.callback(proc.terminate)
                workers.append(proc.pid)
            time.sleep(0.15)
        yield workers


def utc_usage_preflight():
    deltas = []
    for _ in range(2):
        start, t0 = usage(), time.monotonic()
        time.sleep(2)
        deltas.append((usage() - start) / (time.monotonic() - t0))
    require(max(deltas) < 0.20, "control host busy before trial")
    return deltas


def classify(value):
    if math.isfinite(value) and value <= 0.5:
        return "quiet"
    if math.isfinite(value) and value >= 1.25:
        return "occupied"
    return "unknown"


def fit(rows):
    fixed = range(1, 5)
    profile = {}
    valid = [r for r in rows if r["eligible"]]
    for regime in ("quiet", "occupied", "pooled"):
        times = {n: [r["wall_s"] for r in valid if r["arm"] == n and
                     (regime == "pooled" or r["regime"] == regime)] for n in fixed}
        enough = all(len(z) == (4 if regime == "pooled" else 2) for z in times.values())
        profile[regime] = min(fixed, key=lambda n: (statistics.median(times[n]), -n)) if enough else 4
    return profile


def summary(rows):
    arms = [1, 2, 3, 4, "pilot", "measured"]
    out = {}
    for arm in arms:
        own = [r for r in rows if r["arm"] == arm]
        times = sorted(y for r in own for y in r["ttfts_s"])
        out[str(arm)] = {"wall_s": sum(r["wall_s"] for r in own), "p95_ttft_s": times[math.ceil(.95*len(times))-1],
                         "by_seed_s": {str(s):sum(r["wall_s"] for r in own if r["seed"] == s) for s in {r["seed"] for r in own}},
                         "eligible_waves": sum(r["eligible"] for r in own)}
    return out


def measure(args):
    from ironmule.ggml_engine import GgmlEngine, file_sha256
    from ironmule.gguf_tokenizer import GgufTokenizer
    from ironmule.pilot_learning import ThreadLearning
    from training.serving_trial import fixture, run_wave

    start = time.monotonic()
    plan = json.loads(PROTOCOL.read_text())
    require(not args.out.exists() and sorted(os.sched_getaffinity(0)) == plan["cpu_scope"], "scope unavailable")
    quiet = utc_usage_preflight()
    gguf_sha = file_sha256(str(args.gguf))
    require(gguf_sha == plan["gguf_sha256"], "different GGUF")
    rows = []
    with tempfile.TemporaryDirectory() as state, closing(GgmlEngine(str(args.gguf), seqs=4, seq_ctx=512,
            speculate=False)) as engine, closing(GgufTokenizer(str(args.gguf), engine.L)) as tokenizer:
        pilot = engine.pilot
        require(engine.gpu_layers == 0 and pilot.cores == 4 and pilot.load.cpus == frozenset(plan["cpu_scope"]), "foreign scope")
        learning = ThreadLearning(engine, Path(state), gguf_sha, file_sha256(str(pilot.weights_path)))
        require(learning.binding == plan["binding"], "existing affinity-qualified machine or model changed")
        engine.prefix_reuse = False
        profile = None
        phase_times = {}
        for role in ("train", "holdout"):
            phase_started = time.monotonic()
            for regime, seed, arms in plan[role]["order"]:
                expected = None
                with competitors(regime) as children:
                    for arm in arms:
                        engine._clear()
                        pilot.load.update()
                        cpu0 = [proc_seconds(p) for p in children]
                        tick = time.monotonic()
                        time.sleep(.22)
                        dt = time.monotonic() - tick
                        burn = [(proc_seconds(p) - c) / dt for p,c in zip(children,cpu0)]
                        t0 = time.monotonic()
                        observed = pilot.load.update()
                        inferred = classify(observed)
                        picked = arm if isinstance(arm, int) else (
                            profile.get(inferred, profile["pooled"]) if arm == "measured" else None)
                        active = set()
                        original_decide = pilot.decide
                        original_decode = engine._decode
                        def decide(*a, **kw):
                            proposal = original_decide(*a, **kw)
                            if picked is not None:
                                proposal["threads"] = picked
                            return proposal
                        def decode(*a, **kw):
                            active.add(engine.threads)
                            return original_decode(*a, **kw)
                        pilot.decide, engine._decode = decide, decode
                        try:
                            before = pilot.decisions
                            if picked is not None:
                                engine.set_threads(picked)
                            charged_setup = time.monotonic() - t0
                            work = run_wave(engine, tokenizer, fixture(seed, 4, plan[role]["workload"]), 3, {})
                            wall = time.monotonic() - t0
                        finally:
                            pilot.decide, engine._decode = original_decide, original_decode
                        require(work["correct"] == work["completed"] == 4 and work["reused_prompt_tokens"] == 0
                                and pilot.decisions > before and active and (picked is None or active == {picked}),
                                "wrong work, unsupported action or real native thread geometry differs")
                        if expected is None:
                            expected = work["output_sha256"]
                        require(work["output_sha256"] == expected, "answers or stops changed across arms")
                        rows.append({"role":role,"regime":regime,"seed":seed,"arm":arm,"selected_static":picked,
                                "observed":observed if math.isfinite(observed) else None,"arrival_regime":inferred,
                                "child_cpu_per_wall":burn,"native_threads":sorted(active),
                                "eligible":inferred==regime and (not children or sum(burn)>=1.2),
                                "controller_calls":pilot.decisions-before,"wall_s":wall,
                                "ttfts_s":[t+charged_setup for t in work["ttfts_s"]],"work":work})
                print(role,regime,seed,flush=True)
            if role == "train":
                profile = fit(rows)
            phase_times[role] = time.monotonic() - phase_started
        scores = summary([r for r in rows if r["role"]=="holdout"])
        measured = scores["measured"]
        best = min(range(1,5), key=lambda n:scores[str(n)]["wall_s"])
        reference = scores[str(best)]
        clean = all(r["eligible"] for r in rows)
        fast = measured["wall_s"] <= reference["wall_s"]*.98 and all(
               measured["by_seed_s"][s] < scores[str(n)]["by_seed_s"][s] for n in range(1,5) for s in measured["by_seed_s"])
        gate = clean and profile["quiet"] != profile["occupied"] and fast and measured["p95_ttft_s"] <= reference["p95_ttft_s"]
        report = {"schema":"ironmule.xeon-affinity-load-observation.v1","protocol_sha256":file_sha256(str(PROTOCOL)),
                  "worker_sha256":file_sha256(__file__),"binding":learning.binding,"profile":learning.policy.profile,
                  "native_sha256":file_sha256(str(Path(os.environ["LLAMA_CPP_LIB_PATH"])/"libllama.so")),
                  "engine_sha256":file_sha256(sys.modules["ironmule.ggml_engine"].__file__),
                  "pilot_sha256":file_sha256(sys.modules["ironmule.pilot"].__file__),"utc_s":time.time(),
                  "cgroup_cpu_max":Path("/sys/fs/cgroup/cpu.max").read_text().strip(),"quiet_preflight":quiet,
                  "profile_by_regime":profile,"phase_wall_s":phase_times,"elapsed_s":time.monotonic()-start,
                  "cgroup_cost_usd":0,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "rows":rows,"holdout":scores,"best_fixed":best,"all_loads_verified":clean,
                  "load_dependent_gate_passed":gate,"first_horizon_s_saved":reference["wall_s"]-measured["wall_s"]-phase_times["train"],
                  "activation":None}
        write_once(args.out, report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    measure(p.parse_args())
