"""Publish qualified ordered source, fit the existing Pilot head and drive fresh target activation."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np

from lab.pk6 import ordered_grade as grade
from lab.pk6 import ordered_quality
from lab.pk6.startup_search import materialize
from training import pilot_restart_profiles as pilot
from training import scoped_release as release
from training.code_author import write_once
from training.code_trial import read, save
from training.trial_release import require, switch_lock

ADAPTER="qwen-ordered-pk2"
release.ADAPTERS[ADAPTER]=("8cf0a3106f09a3338eb53c28047af7e36fc7b340","GgmlEngine.measure_pk2")
ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/"lab/autonomous/results/a100-dynamic-calibration"


def evaluator():
    paths=[ROOT/"lab/pk6"/name for name in ("ordered_grade.py","ordered_arm.py","ordered_worker.py","calibration_arm.py","calibration_grade.py")]
    paths.extend([ROOT/"lab/pk1/restart_arm.py", ROOT/"lab/pk1/native_trial.py",ROOT/"training/serving_trial.py",ROOT/"lab/rp/restricted_worker.py",FOLDER/"ordered/protocol.json"])
    return grade.canonical({str(p.relative_to(ROOT)):release.checksum(p) for p in paths})


def qualification(archive):
    require(release.checksum(ROOT/"lab/pk6/ordered_grade.py")=="0ecfc27e3b29e57722a70150b45aaac7138c30c4616bdee0b01057f862940f49"
            and release.checksum(FOLDER/"ordered/protocol.json")=="421ba29c5c877437eebcb5fa61d9290d1aed443c61d7376738babd0a44e972eb",
            "different prospective thresholds or scorer")
    verdict=grade.grade(archive)
    require(verdict["status"]=="qualified","source failed the ordered target gate")
    ref=read(archive/"aa-0.json")
    for name in ["aa-0","aa-1",*(f"{a}-{s}" for s in (*grade.MIXED,*grade.FACTS) for a in ("reference","candidate"))]:
        row=read(archive/(name+".json"))
        require(row["evaluator_sha256"]==release.checksum(ROOT/"lab/pk6/ordered_arm.py")
                and row["worker_sha256"]==release.checksum(ROOT/"lab/pk6/ordered_worker.py")
                and row["harness_sha256"]==release.checksum(ROOT/"lab/pk1/restart_arm.py")
                and row["sampled_trace_sha256"]==release.checksum(archive/(name+".trace")),
                "different worker, native measurement or trace")
    return {"passed":True,"reference_sha256":grade.REFERENCE,"candidate_sha256":grade.CANDIDATE,
            "evaluator_sha256":evaluator(),"binding":{"runtime":ref["binding"],"target":ref["target"],
            "vendor":ref["cuda_vendor"],"mask_bits":ref["mask_bits"],"pilot_sha256":ref["pilot_sha256"],
            "pk2_min_rows":ref["pk2"]["min_rows"],"order_sha256":release.checksum(FOLDER/"order.json")},
            "metrics":verdict}


def publish(state,archive):
    q=qualification(archive)
    proposal=materialize(FOLDER)
    require(hashlib.sha256(proposal).hexdigest()==grade.CANDIDATE,"different source proposal")
    state.mkdir(parents=True,exist_ok=True)
    source=state/"candidate.py"
    if not source.exists():
        source.write_bytes(proposal)
    require(source.read_bytes()==proposal,"conflicting source")
    return release.publish(state,read(FOLDER/"order.json"),source,adapter=ADAPTER,verify=lambda:q,
                           rationale=read(FOLDER/"author/response.json")["proposal"]["rationale"],submit=True)


def fit(state, archive):
    start=time.monotonic()
    q=qualification(archive)
    order=read(FOLDER/"order.json")
    pairs=q["metrics"]["pairs"]
    times=[(p["reference_s"],p["candidate_s"]) for p in pairs]
    before,heads,margin,score=pilot.make_heads(times[:6],times[6:],20851)
    arrays={f"{tag}{i}_{k}":v for tag,ensemble in (("prior",before),("head",heads))
            for i,h in enumerate(ensemble) for k,v in zip(pilot.PARAMS,h,strict=True)}
    out=io.BytesIO()
    np.savez(out,**arrays,scope=np.asarray(pilot.binding(order,q)),schema=np.asarray(pilot.SCHEMA),
             margin=np.asarray(margin),holdout_admitted=np.asarray(score["holdout_correct_profile"]))
    state.mkdir(parents=True,exist_ok=True)
    path=state/"ordered-head.npz"
    if not path.is_file():
        with path.open("xb") as stream:
            stream.write(out.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
    require(path.read_bytes()==out.getvalue(),"different trained checkpoint")
    report={"order":order["order_id"],"seed":20851,"weights_sha256":release.checksum(path),
            "parameters_per_network":sum(v.size for v in heads[0]),"ensemble_size":len(heads),
            "fit_and_grade_s":time.monotonic()-start,"metrics":score,
            "rule_comparator":"Always use the already qualified candidate on this exact target; it pays the guarded local PK2 measurements on each start."}
    if (state/"fit.json").exists():
        old=read(state/"fit.json")
        require(old["weights_sha256"]==report["weights_sha256"] and old["metrics"]==score,"conflicting fit")
        return old
    write_once(state/"fit.json",report)
    return report


def bound(row, receipt):
    q=receipt["qualification"]
    b=q["binding"]
    require(row["target"]==b["target"] and row["binding"]==b["runtime"] and row["cuda_vendor"]==b["vendor"] and
            row["mask_bits"]==b["mask_bits"] and row["pilot_sha256"]==b["pilot_sha256"] and row["pk2"]["min_rows"]==b["pk2_min_rows"] and
            q["evaluator_sha256"]==evaluator(), "deployment target, evaluator or local profile changed")


def measured(root,receipt,arm,seed,name,factual):
    runtime=receipt["qualification"]["binding"]["runtime"]
    kernel,=(root/"stock/kernels/w4").glob("*/kernel.so")
    require(release.checksum(kernel)==runtime["w4_sha256"] and release.checksum(kernel.with_name("dispatch.table"))==runtime["table_sha256"],
            "different local W4 artifacts")
    model=Path("/models/Qwen3-32B-Q4_K_M.gguf")
    stat=model.stat()
    require([stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]==runtime["model_stat"],
            "local model identity changed")
    for actual, expected in [(root/"ordered_arm.py",ROOT/"lab/pk6/ordered_arm.py"),
                             (root/"calibration_arm.py",ROOT/"lab/pk6/calibration_arm.py"),
                             (root/"restricted_worker.py",ROOT/"lab/rp/restricted_worker.py"),
                             *[(root/"trusted"/n,ROOT/"lab/pk1"/n) for n in ("restart_arm.py","native_trial.py")],
                             (root/"trusted/serving_trial.py",ROOT/"training/serving_trial.py"),
                             (root/"trusted/ordered_worker.py",ROOT/"lab/pk6/ordered_worker.py")]:
        require(release.checksum(actual)==release.checksum(expected),"different executable worker harness")
    release.check_tree(receipt,root/arm,arm=="candidate")
    if not (root/"ordered"/(name+".json")).exists():
        args=[sys.executable,str(root/"ordered_arm.py"),str(root),arm,str(seed),name]
        if factual:
            args.append("--facts-first")
        subprocess.run(args,cwd=root/"reference",env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1",
                       "PYTHONPATH":str(root/"reference")+":"+str(root/"trusted")},check=True)
    row=read(root/"ordered"/(name+".json"))
    grade.validate(row,arm,seed,factual)
    bound(row,receipt)
    return row


def activate(args):
    def fresh(receipt):
        require(receipt["qualification"]["evaluator_sha256"]==evaluator(),"changed trusted evaluator")
        quality=ordered_quality.ensure(args.root.parent/"calibration",receipt["qualification"]["binding"]["runtime"]["settings"])
        pairs=[]
        for index,(seed,factual) in enumerate(((99301,False),(99401,True))):
            rows={}
            for arm in (("reference","candidate") if index==0 else ("candidate","reference")):
                rows[arm]=measured(args.root,receipt,arm,seed,f"fresh-{arm}-{seed}",factual)
            pair=grade.price(rows["reference"],rows["candidate"],seed,factual)
            require(pair["rate"]>=1.02 and max(pair["warm_wall"],pair["warm_p95"])<=1.02
                    and max(pair["rss_ratio"],pair["gpu_ratio"])<=1.05,"fresh serving regression")
            pairs.append(pair)
        return {**receipt["qualification"],"metrics":{"fresh":pairs,"numerical_quality":quality}}
    return release.activate(args.state,args.control,args.pin,fresh)


def deliver(args):
    started=time.monotonic()
    receipt=release.pinned(args.state,args.pin)
    selected=release.selected(args.state,args.pin)
    decision_start=time.perf_counter()
    family="factual-first-18" if args.facts_first else "mixed-first-32"
    try:
        data=args.weights.read_bytes()
        require(len(data)<1<<20 and hashlib.sha256(data).hexdigest()==args.weights_pin
                and receipt["adapter"]==ADAPTER, "different registered source or weights")
        with np.load(io.BytesIO(data),allow_pickle=False) as model:
            require(str(model["schema"])==pilot.SCHEMA and str(model["scope"])==pilot.binding(receipt["order"],receipt["qualification"]),
                    "unbound checkpoint")
            tag="prior" if args.ablate else "head"
            networks=[[model[f"{tag}{i}_{k}"] for k in pilot.PARAMS] for i in range(3)]
            support=family=="mixed-first-32" or bool(model["holdout_admitted"])
            allowed=pilot.NAMES if selected=="candidate" else ("reference",)
            decision=pilot.Pilot.startup_profile(networks,pilot.features(family),pilot.NAMES,allowed,"reference",
                                                in_support=support,margin_error=float(model["margin"]))
    except (OSError,ValueError,KeyError,TypeError,EOFError):
        decision={"profile":"reference","reason":"unavailable qualified startup model"}
    decision.update(cpu_ms=1000*(time.perf_counter()-decision_start),family=family,ablate_weights=args.ablate)
    arm=decision["profile"]
    prefix="ablation" if args.ablate else "selected"
    request_id=uuid.uuid4().hex
    try:
        name=f"{prefix}-{arm}-{args.seed}-{request_id}"
        row=measured(args.root,receipt,arm,args.seed,name,args.facts_first)
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        release.rollback(args.state,f"real worker verification failed: {exc}"[:300])
        if arm=="reference":
            raise
        arm="reference"
        name=f"fallback-{arm}-{args.seed}-{request_id}"
        row=measured(args.root,receipt,arm,args.seed,name,args.facts_first)
    first=row["factual_first"][0]["completed_s"] if args.facts_first else row["cold_s"]
    elapsed=time.monotonic()-started
    outside_worker=max(0.,elapsed-row["ledger_elapsed_s"])
    monitor=None
    if arm=="candidate" and not args.ablate:
        family=[p for p in receipt["qualification"]["metrics"]["pairs"] if p["facts_first"]==args.facts_first]
        limit=1.02*max(p["reference_s"] for p in family)
        late=first+row["broker_to_process_s"]>limit
        path=args.state/"deployment"/("facts-monitor.json" if args.facts_first else "mixed-monitor.json")
        with switch_lock(path.parent):
            previous=read(path) if path.is_file() else {}
            count=(previous.get("consecutive_late",0) if previous.get("pin")==args.pin else 0)+1 if late else 0
            monitor={"pin":args.pin,"consecutive_late":count,"limit_s":limit,"work":name}
            save(path,monitor)
        if count>=2:
            release.rollback(args.state,"two completed ordered batches exceed the locally measured reference envelope")
    return {"arm":arm,"decision":decision,"seed":args.seed,"source_sha256":row["source_sha256"],
            "complete_first_work_s":first+row["broker_to_process_s"]+outside_worker,
            "complete_request_s":elapsed,"outside_worker_s":outside_worker,
            "work_sha256":grade.canonical([row["cold"]["answers"],row["warm"]["answers"],row["factual"]]),
            "completed":[row["cold"]["completed"],row["warm"]["completed"],len(row["facts"])],
            "work_id":name,"monitor":monitor}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("publish","fit","activate","deliver","exercise","rollback"))
    p.add_argument("--state",type=Path,required=True)
    p.add_argument("--archive",type=Path)
    p.add_argument("--root",type=Path)
    p.add_argument("--control",type=Path)
    p.add_argument("--pin")
    p.add_argument("--weights",type=Path)
    p.add_argument("--weights-pin")
    p.add_argument("--seed",type=int,default=99501)
    p.add_argument("--facts-first",action="store_true")
    p.add_argument("--ablate",action="store_true")
    args=p.parse_args()
    if args.action in ("fit","publish"):
        result=(fit if args.action=="fit" else publish)(args.state,args.archive)
    elif args.action=="activate":
        result=activate(args)
    elif args.action=="rollback":
        result=release.rollback(args.state)
    elif args.action=="exercise":
        def verify_delivery(state):
            return deliver(argparse.Namespace(**{**vars(args),"state":state}))
        result=release.exercise(args.state,args.pin,verify_delivery)
    else:
        result=deliver(args)
    print(json.dumps(result,allow_nan=False))


if __name__=="__main__":
    main()
