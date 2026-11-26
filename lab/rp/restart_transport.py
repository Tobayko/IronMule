"""Snapshot an incomplete original HTTP arm and optionally run one 120-second reference-only diagnostic."""
from __future__ import annotations

import argparse
import json
import shlex
import time
import urllib.parse
from pathlib import Path, PurePosixPath

from lab.pk1 import restart_http, restart_transport as probe
from lab.rp.native_campaign import call, exclusive_measurement, job, lease, shell, snapshot, upload
from training import scoped_release as release
from training.code_author import write_once
from training.code_trial import read

ROOT = Path(__file__).resolve().parents[2]
FILES = ("opened.json","requests.json","worker.json","result.json","stability-tail.jsonl","worker-stderr.log")


def original(local):
    decision = read(local/"decision.json")
    if (decision != read(local/"product-qualify.json") or decision != {"passed":False,"reason":"ironmule returned a stream error"}
            or not read(local/"withdrawn.json")["withdrawn"] or list(local.glob("reference-9*.json")) or list(local.glob("candidate-9*.json"))):
        raise ValueError("different first incomplete product arm")
    held = ("decision.json","product-qualify.json","withdrawn.json","gate-reference.json","gate-candidate.json",
            "product-binding.json","gate-devices-reference.json","gate-devices-candidate.json")
    source = ("ironmule_product/compare.py","ironmule_product/http_server.py","lab/pk1/restart_http.py","lab/pk1/native_http.py")
    row = {"schema":"ironmule.http-stream-failure.v1","attempt":"product-v3","source_sha256":read(local/"gate-reference.json")["source_sha256"],
           "arm":"reference","seed":restart_http.SEEDS[0], "complete_arm_receipts":0,"speed_label":False,
           "original_artifacts":{name:release.checksum(local/name) for name in held},
           "observation":decision["reason"], "http_status":200, "sse_error_received":True,
           "minimum_requests_with_service_event":1, "minimum_basis":"The original handler validates a token or done before committing 200; the original client inspects SSE errors only after HTTP 200.",
           "phase":None, "completed_correct_answers":None, "native_error":None,
           "missing":"Phase, request counts and native cause were not persisted by the original arm. Its three batches publish one combined receipt. The client discarded the SSE error object; the original worker tail and temporary stability store were not archived.",
           "diagnostic_sources":{name:release.checksum(ROOT/name) for name in source},
           "ordinary_worker_selected_for_promotion":False}
    log = local/"private-http-qualify-job.local.json"
    if log.exists():
        result = read(log)
        row["original_job"] = {"running":result["running"],"rc":result["rc"],"tail_bytes":len(result["tail"].encode()),"sha256":release.checksum(log)}
    write_once(local/"transport-diagnostic.json",row)
    return row


def cycle(a):
    local=a.state/"product-v3"
    original(local)
    binding=read(local/"product-binding.json")
    cfg=read(a.state/"target.json")
    pod=lease(cfg["pod_id"])
    order=read(a.state/"order.json")["order_id"]
    remote=PurePosixPath(cfg["state_root"])/("restart-"+order)
    digest=release.checksum(probe.__file__)
    folder=remote/("transport-diagnostic-"+digest[:16])
    copy=local/("diagnostic-"+digest[:16])
    target=PurePosixPath(cfg["framework"])/"lab/pk1/restart_transport.py"
    if a.stage:
        upload(pod,target,Path(probe.__file__).read_bytes())
        shell(pod,shlex.join(["chmod","0444",str(target)]))
    if shell(pod,shlex.join(["sha256sum",str(target)])).split()[0]!=digest:
        raise ValueError("different diagnostic observer on target")
    if a.stage:
        return {"stage":"observer staged","sha256":digest}
    copy.mkdir(exist_ok=True)
    # The original raw arm and source qualification stay immutable. Observer output has its own key.
    snapshot(pod,folder,["result.json"],copy)
    if (copy/"result.json").exists():
        snapshot(pod,folder,FILES,copy)
        result=read(copy/"result.json")
        if (release.checksum(copy/"requests.json") != result["request_report_sha256"] or
                (copy/"worker.json").is_file() and release.checksum(copy/"worker.json") != result["worker_report_sha256"]):
            raise ValueError("incomplete or changed diagnostic export")
        return {"stage":"diagnostic completed","raw":str(copy),"result":result}
    name="transport-"+digest[:16]
    status=call(pod,"GET","/job?"+urllib.parse.urlencode({"name":name,"tail":4}))
    if not status.get("running") and status.get("rc") not in (None,0,75):
        snapshot(pod,folder,[f for f in FILES if f!="result.json"],copy)
        final={"stage":"diagnostic stopped","rc":status["rc"],"complete_arm":False,
               "artifacts":{p.name:release.checksum(p) for p in copy.iterdir() if p.name in FILES}}
        write_once(copy/"incomplete.json",final)
        return final
    if not a.start:
        return {"stage":"not started","folder":str(folder)}
    if pod["deadline"]-time.time()<155:
        raise RuntimeError("insufficient time for bounded diagnostic")
    common={"state":remote/"active","trial":remote/"trial","candidate-source":remote/"proposal.py",
       "checkout":cfg["checkout"],"gguf":cfg["gguf"],"worker-root":cfg["worker_root"],"reference-lib":cfg["reference_lib"],
       "table":cfg["table"],"geometry":remote/"product-v3/geometry.json","head":remote/"product-v3/startup-head.npz",
       "out":folder,"receipt-sha256":binding["release_sha256"],"geometry-sha256":binding["geometry_sha256"],
       "head-sha256":binding["head_sha256"],"binding":binding["evaluator_binding"],"gate-sha256":release.checksum(local/"gate-reference.json")}
    argv=["timeout","--kill-after=2","120",cfg["python"],"-m","lab.pk1.restart_transport"]
    for key,value in common.items():
        argv.extend(("--"+key,str(value)))
    dest=shlex.quote(cfg["framework"])
    cmd="set -e; . "+shlex.quote(cfg["env"])+"; cd "+dest+"; export PYTHONPATH="+dest+"; "+exclusive_measurement(cfg,argv,300)
    return {"stage":job(pod,name,cmd),"folder":str(folder)}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state",required=True,type=Path)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument("--original",action="store_true",help="CPU-only export of the original terminal failure")
    modes.add_argument("--stage",action="store_true",help="copy the observer without running a model")
    modes.add_argument("--start",action="store_true",help="submit one bounded isolated diagnostic arm")
    a=p.parse_args()
    print(json.dumps(original(a.state/"product-v3") if a.original else cycle(a)),flush=True)
