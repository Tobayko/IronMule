"""Broker the second A100 calibration experiment and retain each immutable completed arm off-pod."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import time
from urllib.parse import urlencode

from lab.pk6 import calibration_grade as grader
from lab.pk6 import startup_search
from lab.rp import native_campaign as target
from training.code_author import write_once

REMOTE = PurePosixPath("/work/calibration")
JOB = "calibration-qualified-target"
RECORDS = ["aa-0", "aa-1", *(f"{arm}-{seed}" for seed in grader.SEEDS for arm in ("reference", "candidate"))]
ROOT = Path(__file__).resolve().parents[2]


def prepare(pod, order):
    candidate = startup_search.materialize(order)
    if hashlib.sha256(candidate).hexdigest() != grader.CANDIDATE:
        raise ValueError("source approval differs from the actual author reply")
    source = startup_search.warm_result.SOURCE
    target.shell(pod, f"""set -eu
test "$(git -C /work/IronMule rev-parse HEAD)" = {source}
test ! -e /work/calibration/out
dst=/work/calibration
install -d -m 0755 "$dst/reference" "$dst/candidate" "$dst/stock" "$dst/out"
git -C /work/IronMule archive {source} | tar -x -C "$dst/reference"
cp -a "$dst/reference/." "$dst/candidate/"
cp -a /root/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16 "$dst/bindings"
cp -a /work/state/kernels /work/state/ggml-gate /work/state/settings.json /work/state/models.json "$dst/stock/"
chmod -R a+rX "$dst" /models""")
    target.upload(pod,str(REMOTE/"candidate"/startup_search.METHOD_PATH),candidate)
    for source_name, remote_name in (("calibration_arm","arm"),("calibration_grade","calibration_grade"),
                                     ("calibration_series","calibration_series")):
        target.upload(pod,str(REMOTE/(remote_name+".py")),(ROOT/f"lab/pk6/{source_name}.py").read_bytes())
    target.shell(pod,"chmod 0444 /work/calibration/candidate/ironmule/ggml_engine.py && "
                     "PYTHONPYCACHEPREFIX=/work/calibration/bytecode python -m py_compile /work/calibration/candidate/ironmule/ggml_engine.py")


def collect(pod, local):
    # JSON appears atomically only after the server exits and the trace has been copied.
    target.snapshot(pod, REMOTE/"out", [r+".json" for r in RECORDS],local)
    completed=[r for r in RECORDS if (local/(r+".json")).is_file()]
    target.snapshot(pod, REMOTE/"out",[r+".trace" for r in completed],local)
    for r in completed:
        info=json.loads((local/(r+".json")).read_bytes())
        if hashlib.sha256((local/(r+".trace")).read_bytes()).hexdigest()!=info["trace_sha256"]:
            raise ValueError("different trace or completed arm")
    return completed


def run(pod_id, local, order, seconds):
    deadline=time.monotonic()+seconds
    local.mkdir(mode=0o700,parents=True,exist_ok=True)
    while time.monotonic()<deadline:
        pod=target.lease(pod_id)
        status=target.call(pod,"GET","/job?"+urlencode({"name":JOB,"tail":10}))
        if not status["running"] and status["rc"] is None:
            prepare(pod, order)
            target.job(pod,JOB,"""set -eu
cd /work/calibration/reference
export PYTHONPATH=/work/calibration/reference
python /work/calibration/arm.py /work/calibration reference 98600 aa-0
python /work/calibration/arm.py /work/calibration reference 98600 aa-1
exec python /work/calibration/calibration_series.py /work/calibration --seconds 2850""")
        completed=collect(pod,local)
        if status["rc"] is not None and not status["running"]:
            try:
                result=grader.grade(local) if "aa-1" in completed else {"status":"incomplete"}
            except (KeyError,ValueError) as error:
                result={"status":"rejected","reason":str(error)}
            if status["rc"]!=0 and result["status"] not in ("rejected","incomplete"):
                result={"status":"incomplete","observed":result}
            result.update(job_exit=status["rc"],completed_files=completed)
            write_once(local/"result.json",result)
            return result
        time.sleep(min(10,max(0,deadline-time.monotonic())))
    return {"status":"poll window expired","completed":completed}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pod-id",required=True)
    p.add_argument("--state",required=True,type=Path)
    p.add_argument("--order",required=True,type=Path)
    p.add_argument("--seconds",type=int,required=True,choices=range(1,7201),metavar="1..7200")
    a=p.parse_args()
    print(json.dumps(run(a.pod_id,a.state,a.order,a.seconds)),flush=True)
