"""Submit identical caller-supplied prompts to fresh CLI processes across a persisted withdrawal."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from ironmule.factual_gate import MAX_TOKENS, cases, correct
from lab.pk6 import ordered_grade
from training.code_author import write_once
from training.code_trial import read
from training.trial_release import require


def check(root):
    state = root / "state"
    proofs = root / "client-checks"
    observed = read(proofs / "gate.json")
    if not observed["passed"]:
        return {"passed": False, "reason": "caller entry has no admitted candidate"}
    pin = hashlib.sha256((root / "inputs/control.json").read_bytes()).hexdigest()
    weights = root / "inputs/ordered-head.npz"
    weight_pin = hashlib.sha256(weights.read_bytes()).hexdigest()
    questions = cases(99604)
    request = proofs / "input-99604.json"
    write_once(request, {"prompts":[q.prompt for q in questions],"max_tokens":MAX_TOKENS,"workload_seed":99604})
    with tempfile.TemporaryDirectory(dir=root) as temp:
        shadow = Path(temp)
        dest = shadow / "deployment"
        dest.mkdir()
        for name in ("control.json","target-check.json","selection.json","ordered-client.json"):
            shutil.copyfile(state / "deployment" / name, dest / name)
        def submit(arm):
            path=proofs/f"caller-{arm}-99604.json"
            request_sha=hashlib.sha256(request.read_bytes()).hexdigest()
            if path.exists():
                kept=read(path)
                require(kept["pin"]==pin and kept["request_sha256"]==request_sha and kept["response"]["arm"]==arm
                        and kept["process_to_complete_reply_s"]>=kept["response"]["complete_s"]
                        and len(kept["response"]["answers"])==18 and all(correct(q,r["answer"]) for q,r in
                        zip(questions,kept["response"]["answers"],strict=True)), "different saved caller work")
                return kept
            cmd = [sys.executable,"-m","lab.pk6.ordered_client","--root",str(root),"--state",str(shadow),
                   "--weights",str(weights),"--weights-pin",weight_pin,"--pin",pin,"--requests",str(request)]
            start = time.perf_counter()
            completed = subprocess.run(cmd,stdout=subprocess.PIPE,check=True,timeout=240)
            elapsed = time.perf_counter()-start
            result = json.loads(completed.stdout)
            require(result["arm"]==arm and len(result["answers"])==18 and all(correct(q,r["answer"])
                    for q,r in zip(questions,result["answers"],strict=True)), "incorrect caller recovery")
            entry = {"process_to_complete_reply_s":elapsed,"response":result,"request_sha256":request_sha,"pin":pin}
            write_once(path,entry)
            return entry
        before = submit("candidate")
        subprocess.run([sys.executable,"-m","lab.pk6.ordered_release","rollback","--state",str(shadow)],
                       check=True,stdout=subprocess.DEVNULL)
        after = submit("reference")
        require(before["response"]["answers"]==after["response"]["answers"],"changed caller answers after restart")
    result = {"passed":True,"pin":pin,"candidate_s":before["process_to_complete_reply_s"],
              "reference_s":after["process_to_complete_reply_s"],"answers":18,
              "output_sha256":ordered_grade.canonical(before["response"]["answers"])}
    write_once(proofs/"caller-recovery.json",result)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    print(json.dumps(check(parser.parse_args().root),allow_nan=False))
