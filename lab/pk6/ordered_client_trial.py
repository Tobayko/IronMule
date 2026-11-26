"""Check the caller-provided JSON entry on both registered families before routing requests."""
import argparse
import json
from pathlib import Path

from ironmule.factual_gate import cases, MAX_TOKENS
from ironmule_product.compare import performance_prompts
from lab.pk6.ordered_client import serve
from lab.pk6.ordered_grade import canonical
from training.code_author import write_once
from training.code_trial import read, save
from training.trial_release import require


def trial(args):
    out = args.root/"client-checks"
    out.mkdir(exist_ok=True)
    gate = args.state/"deployment"/"ordered-client.json"
    if gate.exists():
        return read(gate)
    pairs = []
    for seed, width, arms in ((99601,32,("reference","candidate")),(99602,18,("candidate","reference"))):
        request = {"workload_seed":seed, "max_tokens":32 if width==32 else MAX_TOKENS,
                   "prompts":performance_prompts(seed,32,32) if width==32 else [q.prompt for q in cases(seed)]}
        write_once(out/f"input-{seed}.json",request)
        records = {}
        for arm in arms:
            path = out/f"{arm}-{seed}.json"
            if not path.exists():
                write_once(path,serve(args,read(out/f"input-{seed}.json"),trial_arm=arm))
            records[arm] = read(path)
        ref, cand = records["reference"], records["candidate"]
        same = (ref["answers"] == cand["answers"] and ref["generated_tokens"] == cand["generated_tokens"]
                and len(ref["answers"]) == width)
        rate = ref["complete_s"]/cand["complete_s"]
        passed = same and ref["arm"]=="reference" and cand["arm"]=="candidate" and rate>=1.02
        pairs.append({"seed":seed,"family":cand["family"],"reference_s":ref["complete_s"],"candidate_s":cand["complete_s"],
                      "rate":rate,"exact_answers":same,"output_sha256":canonical(cand["answers"]), "passed":passed})
        if not passed:
            break
    passed = len(pairs)==2 and all(p["passed"] for p in pairs)
    result = {"passed":passed,"pin":args.pin,"pairs":pairs,"families":[p["family"] for p in pairs] if passed else []}
    save(gate,result)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","state","weights"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--pin",required=True)
    p.add_argument("--weights-pin",required=True)
    p.set_defaults(ablate=False)
    print(json.dumps(trial(p.parse_args()),allow_nan=False))
