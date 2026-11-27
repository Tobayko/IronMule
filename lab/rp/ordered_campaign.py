"""Resume the ordered GPU campaign, publish only a passing source and exercise the learned deployment."""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import time
from urllib.parse import urlencode

from lab.pk6 import ordered_grade as grade
from lab.pk6 import ordered_release as release
from lab.pk6 import startup_search
from lab.rp import calibration_campaign as http
from lab.rp import native_campaign as target
from training.code_author import consume_cli, read_request, read_response, write_once
from training.code_trial import git, read
from training.trial_release import require
from training.scoped_release import checksum

ROOT=Path(__file__).resolve().parents[2]
REMOTE=PurePosixPath("/work/calibration")
RECORDS=["aa-0","aa-1",*(f"{a}-{s}" for s in (*grade.MIXED,*grade.FACTS) for a in ("reference","candidate"))]
DEPLOY="/work/ordered-release"


def status(pod,name):
    return target.call(pod,"GET","/job?"+urlencode({"name":name,"tail":6}))


def commissioned(args, state, deadline):
    result = startup_search.commission(args.series, args.observations, state)
    if result["status"] == "deferred":
        return result
    if result["status"] == "requested":
        if args.author == "cli":
            consume_cli(state)
        elif args.author == "hosted":
            if not target.wait_for_author_response(result["request"], deadline):
                return result
        else:
            original = release.FOLDER/"author"
            request_id, _ = read_request(state/"author")
            read_response(original, request_id)
            write_once(state/"author/response.json",read(original/"response.json"))
    # This prospective evaluator was registered for exactly one isolated source response.
    if (read(state/"order.json") != read(release.FOLDER/"order.json")
            or startup_search.materialize(state) != startup_search.materialize(release.FOLDER)):
        return {"status":"new source requires its own registered target evaluator", "order":str(state/"order.json")}
    return {"status":"ready"}


def stage(pod, order):
    if target.shell(pod,"test -d /work/calibration/reference && echo yes || echo no")=="no":
        http.prepare(pod,order)
    target.shell(pod,"install -d -m 0755 /work/calibration/trusted")
    scripts={
        "ordered_arm.py":"lab/pk6/ordered_arm.py","calibration_arm.py":"lab/pk6/calibration_arm.py",
        "ordered_grade.py":"lab/pk6/ordered_grade.py","calibration_grade.py":"lab/pk6/calibration_grade.py",
        "ordered_series.py":"lab/pk6/ordered_series.py","restricted_worker.py":"lab/rp/restricted_worker.py",
        "trusted/ordered_worker.py":"lab/pk6/ordered_worker.py","trusted/native_trial.py":"lab/pk1/native_trial.py",
        "trusted/serving_trial.py":"training/serving_trial.py","trusted/restart_arm.py":"lab/pk1/restart_arm.py"}
    for remote,local in scripts.items():
        target.upload(pod,str(REMOTE/remote),(ROOT/local).read_bytes())
    target.shell(pod,"chmod -R go-w /work/calibration/reference /work/calibration/candidate /work/calibration/bindings /work/calibration/stock /work/calibration/trusted")
    target.job(pod,"ordered-qualification","""set -eu
cd /work/calibration/reference
export PYTHONPATH=/work/calibration/reference:/work/calibration/trusted PYTHONDONTWRITEBYTECODE=1
exec 8>/work/calibration/ordered.lock
flock -n 8 || exit 75
python /work/calibration/ordered_arm.py /work/calibration reference 99100 aa-0
python /work/calibration/ordered_arm.py /work/calibration reference 99100 aa-1
exec python /work/calibration/ordered_series.py /work/calibration --seconds 2400""")


def collect(pod,local):
    target.snapshot(pod,REMOTE/"ordered",[n+".json" for n in RECORDS],local)
    done=[n for n in RECORDS if (local/(n+".json")).is_file()]
    target.snapshot(pod,REMOTE/"ordered",[n+".trace" for n in done],local)
    for n in done:
        require(read(local/(n+".json"))["sampled_trace_sha256"]==checksum(local/(n+".trace")),"changed target trace")
    return done


def deploy(pod,local,trained,publication):
    framework=git("rev-parse","HEAD")
    new=read(Path(publication["receipt"]))["publication"]["candidate"]
    base=read(release.FOLDER/"order.json")["base_commit"]
    require(all(re.fullmatch(r"[0-9a-f]{40}",k) for k in (framework,base,new)),"unbound cloud checkout")
    target.shell(pod,f"""set -eu
repo=/work/IronMule
dst={DEPLOY}
install -d -m 0755 "$dst" "$dst/inputs"
git -C "$repo" fetch -q https://github.com/Tobayko/IronMule.git {framework} {new}
for entry in reference:{base} candidate:{new} framework:{framework}; do
 name=${{entry%:*}}
 rev=${{entry#*:}}
 if [ ! -f "$dst/$name/.revision" ]; then
   mkdir -p "$dst/$name"
   git -C "$repo" archive "$rev" | tar -x -C "$dst/$name"
   chmod -R go-w "$dst/$name"
   echo "$rev" > "$dst/$name/.revision"
 fi
 test "$(cat "$dst/$name/.revision")" = "$rev"
done
for name in ordered_arm.py calibration_arm.py restricted_worker.py trusted bindings stock; do
 test -e "$dst/$name" || cp -al "/work/calibration/$name" "$dst/$name"
done
install -d "$dst/ordered" "$dst/proofs" """)
    target.upload(pod,DEPLOY+"/inputs/control.json",Path(publication["receipt"]).read_bytes())
    target.upload(pod,DEPLOY+"/inputs/ordered-head.npz",(local/"ordered-head.npz").read_bytes())
    pin=publication["receipt_sha256"]
    weights=trained["weights_sha256"]
    require(all(re.fullmatch(r"[0-9a-f]{64}",k) for k in (pin,weights)),"unbound deployment files")
    cmd=f"""set -eu
cd {DEPLOY}/framework
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
d={DEPLOY}
exec 8>"$d/activation.lock"
flock -n 8 || exit 75
action() {{
 out=$1
 shift
 if [ ! -s "$d/proofs/$out.json" ]; then
   python -m lab.pk6.ordered_release "$@" --state "$d/state" --root "$d" --control "$d/inputs/control.json" --pin {pin} --weights "$d/inputs/ordered-head.npz" --weights-pin {weights} > "$d/proofs/$out.incoming"
   mv "$d/proofs/$out.incoming" "$d/proofs/$out.json"
 fi
}}
action activation activate
python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["arm"]=="candidate" else 2)' "$d/proofs/activation.json"
action random deliver --seed 99502 --ablate
action trained deliver --seed 99502
action recovery exercise --seed 99501
"""
    target.job(pod,"ordered-activation",cmd)


def run(a):
    local=a.state.resolve()
    local.mkdir(parents=True,mode=0o700,exist_ok=True)
    raw=local/"ordered"
    deadline=time.monotonic()+a.seconds
    source=commissioned(a,local,deadline)
    if source["status"]!="ready":
        return source
    while time.monotonic()<deadline:
        pod=target.lease(a.pod_id)
        job=status(pod,"ordered-qualification")
        if job["rc"] is None and not job["running"]:
            stage(pod,local)
        completed=collect(pod,raw)
        if job["rc"] is None or job["running"]:
            time.sleep(8)
            continue
        try:
            verdict=grade.grade(raw)
        except (ValueError,KeyError,FileNotFoundError) as err:
            verdict={"status":"rejected" if isinstance(err,ValueError) else "incomplete","reason":str(err)}
        write_once(local/"campaign-verdict.json",{"verdict":verdict,"completed":completed,"job_exit":job["rc"]})
        if job["rc"]!=0 or verdict["status"]!="qualified" or not a.publish:
            return verdict
        trained=release.fit(local,raw)
        pub=release.publish(local,raw)
        pod=target.lease(a.pod_id)
        active=status(pod,"ordered-activation")
        if not active["running"] and active["rc"] is None:
            deploy(pod,local,trained,pub)
        while time.monotonic()<deadline:
            pod=target.lease(a.pod_id)
            active=status(pod,"ordered-activation")
            target.snapshot(pod,PurePosixPath(DEPLOY)/"proofs",[k+".json" for k in ("activation","random","trained","recovery")],local/"proofs")
            target.snapshot(pod,PurePosixPath(DEPLOY)/"ordered",[f"fresh-{arm}-{s}.json" for s in (99301,99401) for arm in ("reference","candidate")],local/"fresh")
            names=target.shell(pod,"if test -d /work/ordered-release/ordered; then find /work/ordered-release/ordered -type f -name '*.json' -printf '%f\\n'; fi").splitlines()
            names=[n for n in names if re.fullmatch(r"(?:fresh-(?:candidate|reference)-\d+|(?:selected|ablation|fallback)-(?:candidate|reference)-\d+-[0-9a-f]{32})\.json",n)]
            if names:
                target.snapshot(pod,PurePosixPath(DEPLOY)/"ordered",names+[n[:-5]+".trace" for n in names],local/"fresh")
            target.snapshot(pod,REMOTE/"numeric",["observed.json"],local/"numerical")
            if not active["running"] and active["rc"] is not None:
                target.snapshot(pod,PurePosixPath(DEPLOY)/"state/deployment",
                                ["control.json","selection.json","target-check.json","rollback-exercise.json",
                                 "mixed-monitor.json","facts-monitor.json"],local/"target-state")
                result={"publication":pub,"target_exit":active["rc"],"trained_weights_sha256":trained["weights_sha256"],
                        "activation":read(local/"proofs/activation.json") if (local/"proofs/activation.json").exists() else None}
                write_once(local/"deployment-outcome.json",result)
                return result
            time.sleep(8)
        break
    return {"status":"poll window elapsed; existing reservations and target outcomes persist"}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state",type=Path,required=True)
    p.add_argument("--pod-id",required=True)
    p.add_argument("--seconds",type=int,required=True,choices=range(1,7201),metavar="1..7200")
    p.add_argument("--publish",action="store_true")
    p.add_argument("--series",type=Path,default=release.FOLDER.with_name("a100-pr57-warm-repeat"))
    p.add_argument("--observations",type=Path,default=release.FOLDER)
    p.add_argument("--author",choices=("hosted","cli","archived"),default="hosted",
                   help="Use a separate hosted/CLI Codex author, or replay the registered reply by request digest.")
    print(json.dumps(run(p.parse_args())),flush=True)
