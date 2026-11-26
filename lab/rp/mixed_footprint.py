"""Collect the optional constrained W4 screen after the external L40S comparison."""

import argparse
import json
from pathlib import Path
import shlex
import time

from lab.pk6.mixed_progress import footprint, grade
from lab.rp import native_campaign as jobs, warm_repeat as sync
from lab.rp.accounting import save


def sync_attempts(pod, out):
    """Mirror an atomically rewritten JSON map while preserving every completed attempt."""
    remote = "/work/mixed-footprint/attempts.json"
    if jobs.shell(pod, f"if test -f {remote}; then echo present; fi") != "present":
        return
    raw = jobs.call(pod, "GET", jobs.endpoint("get", remote), raw=True)
    current = json.loads(raw)
    dest = out / "attempts.json"
    prior = json.loads(dest.read_bytes()) if dest.exists() else {}
    if (
        not isinstance(current, dict)
        or any(
            k not in ("R1", "M1", "M2", "R2") or not (type(v) is int or v == "timeout")
            for k, v in current.items()
        )
        or any(k not in current or current[k] != v for k, v in prior.items())
    ):
        raise ValueError("completed W4 attempt changed")
    if current != prior or not dest.exists():
        save(dest, current)


def collect(pod_id, primary, out):
    pod = jobs.lease(pod_id)
    if pod["deadline"] - time.time() < 6 * 60:
        return "not enough existing lease time"
    if sync.status(pod, "current-vllm").get("running"):
        raise RuntimeError("external comparator is still running")
    samples = grade.telemetry_samples(primary / "device-load.csv")
    peaks = []
    for arm in ("R1", "R2"):
        gates = [
            json.loads(l)["ggml_gate"]
            for l in (primary / f"serve-{arm}.log").read_text().splitlines()
            if l.startswith('{"ggml_gate":')
        ]
        if not gates or gates[-1]["kernel_mode"] != "stock" or not gates[-1]["passes"]:
            return "native reference did not qualify stock"
        waves = grade.lines(primary / f"waves-{arm}.jsonl")
        peaks.append(grade.sampled_resources(samples, waves)["peak_gpu_mib"])
    out.mkdir(mode=0o700, exist_ok=True)
    jobs.upload(pod, "/work/footprint_runner.py", Path(footprint.__file__).read_bytes())
    jobs.upload(
        pod,
        "/work/footprint_inventory.json",
        (primary / "w4-inventory.json").read_bytes(),
    )
    cmd = """set -eu
cd /work/IronMule
export IRONMULE_GGML_BINDINGS="$HOME/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16"
export LLAMA_CPP_LIB_PATH="$IRONMULE_GGML_BINDINGS/vendor/llama.cpp/build/bin"
export PYTHONPATH="$PWD"
set -- /work/state/kernels/w4/*/dispatch.table
test "$#" -eq 1
exec python /work/footprint_runner.py --primary /work/mixed-recovered --out /work/mixed-footprint \\
  --gguf /models/Qwen3-32B-Q4_K_M.gguf --inventory /work/footprint_inventory.json --table "$1" --stock-peak-mib """ + shlex.quote(
        str(max(peaks))
    )
    jobs.job(pod, "bounded-w4-screen", cmd)
    files = ["selection.json", "outcome.json", "selected.table"]
    for a in ("R1", "M1", "M2", "R2"):
        files.extend([f"worker-{a}.json", f"log-{a}.txt"])
    while time.time() < pod["deadline"] - 24:
        try:
            status = sync.status(pod, "bounded-w4-screen")
            sync.collect(pod, out, target="/work/mixed-footprint", files=files)
            sync_attempts(pod, out)
            if status.get("rc") is not None and not status.get("running"):
                (out / "control-end.json").write_text(json.dumps(status))
                return "collected" if (out / "outcome.json").exists() else "incomplete"
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(4)
    return "deadline"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pod-id", required=True)
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(collect(a.pod_id, a.primary, a.out))
