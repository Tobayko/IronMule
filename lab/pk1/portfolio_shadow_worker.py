"""Fresh complete original calibration and a post-delivery independent sentinel in a disposable Qwen worker."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path


def resources():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    data = {
        "utc_s": time.time(),
        "mono_s": time.monotonic(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "load_1_5_15": list(os.getloadavg()),
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
    }
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,pstate,clocks.sm,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=1,
        )
        data["nvml_snapshot"] = proc.stdout.strip().splitlines()
    except (ValueError, OSError, subprocess.SubprocessError):
        data["nvml_snapshot"] = None
    return data


def sentinel(engine, repeats):
    bos = max(0, int(engine.L.llama_vocab_bos(engine.vocab)))
    start = time.perf_counter()
    engine._clear()
    engine._decode([(bos, j, 0, j == 63) for j in range(64)])
    times = []
    try:
        for _ in range(1 + repeats):
            tick = time.perf_counter()
            engine._decode([(bos, 64, 0, True)])
            times.append(time.perf_counter() - tick)
            engine.L.llama_memory_seq_rm(engine.memory, 0, 64, -1)
    finally:
        engine._clear()
    return {
        "width": 1,
        "warmups": 1,
        "timing_samples_s": times,
        "price_s": statistics.median(times[1:]),
        "elapsed_s": time.perf_counter() - start,
    }


def main():
    import calibration_stability_probe as original
    import serving_trial

    from ironmule import ggml_engine as ge
    from ironmule.pilot import Pilot

    dest = Path(sys.argv[sys.argv.index("--output") + 1])
    source = Path(ge.__file__).resolve()
    before = resources()
    obs = {}
    phases = iter(("cold", "warm", "factual"))
    phase = None
    reads = {}
    wave = serving_trial.run_wave
    close = ge.GgmlEngine.close
    scope = os.environ["IRONMULE_PORTFOLIO_OBSERVATION"]

    def wrapped(*a, **kw):
        nonlocal phase
        phase = next(phases)
        try:
            return wave(*a, **kw)
        finally:
            phase = None

    def observer(previous, fields):
        def get(self, key):
            if phase is not None and key in fields:
                frame = sys._getframe(1)
                where = frame.f_code.co_qualname
                if where.startswith(("Pilot.", "GgmlEngine.")):
                    entry = reads.setdefault(phase, {}).setdefault(
                        key,
                        {
                            "first_monotonic_s": time.monotonic(),
                            "first_caller": where,
                            "line": frame.f_lineno,
                            "count": 0,
                        },
                    )
                    entry["count"] += 1
            return previous(self, key)

        return get

    def finish(self):
        try:
            obs["post_work_runtime"] = {
                "threads": self.threads,
                "poll": getattr(self, "poll", ge.POLL),
                "settings": self.settings,
                "seqs": self.seqs,
                "retained_prefixes": len(self._kept),
                "geometry": {
                    k: getattr(self, k)
                    for k in (
                        "steps",
                        "padded",
                        "pack_calls",
                        "pack_rows",
                        "reused_tokens",
                    )
                },
            }
            obs["after_work"] = resources()
            if scope == "post-delivery-shadow":
                start = time.perf_counter()
                samples = {
                    k: sentinel(self, n) for k, n in (("passes", 3), ("padding", 5))
                }
                tick = time.perf_counter()
                passes = self.measure_pass_costs()
                pass_s = time.perf_counter() - tick
                tick = time.perf_counter()
                dispatch = self.measure_padding()
                pad_s = time.perf_counter() - tick
                obs["shadow"] = {
                    "standalone": samples,
                    "complete": {
                        "passes": dict(passes.points),
                        "padding": self.pad_prices,
                    },
                    "dispatch": dispatch,
                    "full_method_s": {"passes": pass_s, "padding": pad_s},
                    "elapsed_s": time.perf_counter() - start,
                    "state": "after all original deliveries, before closing this process; sentinels and full methods clear logical KV; GPU graph/residency effects remain unmeasured",
                }
        finally:
            close(self)

    e_get, p_get = ge.GgmlEngine.__getattribute__, Pilot.__getattribute__
    ge.GgmlEngine.__getattribute__ = observer(e_get, {"pad_to", "pass_costs"})
    Pilot.__getattribute__ = observer(p_get, {"costs"})
    ge.GgmlEngine.close, serving_trial.run_wave = finish, wrapped
    try:
        original.main()
    finally:
        ge.GgmlEngine.__getattribute__, Pilot.__getattribute__ = e_get, p_get
        ge.GgmlEngine.close, serving_trial.run_wave = close, wave
    result = json.loads(dest.read_bytes())
    result.update(
        portfolio_context=scope,
        post_work=obs,
        resources_before=before,
        serving_field_reads=reads,
        original_probe_sha256=result["profiler_sha256"],
        profiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        engine_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    dest.write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
