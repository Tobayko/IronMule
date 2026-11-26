"""Independently regrade the prospective same-device ordinary-server calibration pairs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from ironmule.factual_gate import cases, correct

REFERENCE = "80a41c59bfe8088b305529cca5164134c25d4b5a0ce9572877cb25a2b7d269fd"
CANDIDATE = "f60951799571833dfa663227751a41d9a36a88b54405518240df762cdbf340f7"
BOARD = "112d8a87ea0f7aab9de32ad3b2f2d65318c3bce1044ee30b75f78c528c923e2b"
KERNEL = "19af79dc15afb226112953cf140579037270d21337aaee0d3206499d35cef6c8"
TABLE = "1232da533c2619cee981d6751598e593a1cb8f285da5ac2c2e01a2fa8b1bdaf1"
SEEDS = list(range(98601,98607)) + [98701,98702]


def require(value, error):
    if not value:
        raise ValueError(error)


def validated(row, arm, seed, facts_first):
    facts = cases(seed)
    phases = row["waves"]
    require(row["arm"] == arm and row["seed"] == seed and row["factual_first"] is facts_first, "different trial")
    require(row["binding"]["device_sha256"] == BOARD and row["kernel_sha256"] == KERNEL and row["table_sha256"] == TABLE
            and row["sha256"] == (REFERENCE if arm == "reference" else CANDIDATE), "different target or source")
    require(row["pilot"] is True and row["width"] == 32 and row["health"]["passes"] is True
            and row["health"]["ratio"] <= 1.005 and row["peak_rss_kib"] > 0 and row["peak_gpu_mib"] > 0, "incomplete resources or quality")
    require([x["role"] for x in phases] == (["facts","cold","warm"] if facts_first else ["cold","warm","facts"]), "wrong work order")
    require(row["pk2_calls"] in ((20,) if arm == "reference" else (15,20)) and row["pk2"]["calibration"] == "mix"
            and row["pk2"]["rows"] == 1908 and set(row["pk2"]["burst_ms"]) == {"0","128","256","512","1024"}
            and "likelihood ratio" in row["pk2"]["reason"], "missing local prices or NLL")
    previous = row["startup_s"]
    for wave in phases:
        answers = wave["answers"]
        expected = hashlib.sha256(json.dumps(answers,sort_keys=True).encode()).hexdigest()
        require(expected == wave["work_sha256"] and len(answers) == len(wave["ttft_ms"]) and all(
            a["finish_reason"] in ("stop","length") and a["prompt_tokens"] > 0 and 0 < a["tokens"] <= 32 for a in answers), "invalid work")
        require(0 < wave["wall_s"] <= wave["delivered_s"] - previous + .02 and all(
                0 < value <= wave["wall_s"]*1000 + 1 for value in wave["ttft_ms"]), "invalid timing")
        previous = wave["delivered_s"]
        if wave["role"] == "facts":
            require(len(answers) == len(facts) == 18 and wave["correct"] == [True]*18 and all(
                correct(q,a["text"]) for q,a in zip(facts,answers,strict=True)), "incorrect factual answer")
        else:
            require(len(answers) == 32 and all(a["tokens"] == 32 for a in answers), "unequal mixed work")
    return {wave["role"]: wave for wave in phases}


def p95(wave):
    values=sorted(wave["ttft_ms"])
    return values[math.ceil(.95*len(values))-1]


def pair(ref, cand, seed, facts_first, same=False):
    r=validated(ref,"reference",seed,facts_first)
    c=validated(cand,"reference" if same else "candidate",seed,facts_first)
    require(all(ref[k] == cand[k] for k in ("binding","base","width","health","fixture_sha256","evaluator_sha256")),
            "different quality, target or evaluator")
    require(ref["pk2"]["min_rows"] == cand["pk2"]["min_rows"] and
            all(r[k]["answers"] == c[k]["answers"] for k in r), "selected profile or delivered work changed")
    first = "facts" if facts_first else "cold"
    return {"seed":seed,"facts_first":facts_first,"rate": r[first]["delivered_s"]/c[first]["delivered_s"],
            "reference_s":r[first]["delivered_s"],"candidate_s":c[first]["delivered_s"],
            "warm_wall":c["warm"]["wall_s"]/r["warm"]["wall_s"],"warm_p95":p95(c["warm"])/p95(r["warm"]),
            "rss_ratio":cand["peak_rss_kib"]/ref["peak_rss_kib"],"gpu_ratio":cand["peak_gpu_mib"]/ref["peak_gpu_mib"],
            "calibration_calls":[ref["pk2_calls"],cand["pk2_calls"]]}


def grade(folder):
    def load(name):
        return json.loads((folder/(name+".json")).read_bytes())
    aa=pair(load("aa-0"),load("aa-1"),98600,False,same=True)
    noise=max(aa["rate"],1/aa["rate"])-1
    pairs=[]
    for seed in SEEDS:
        if not all((folder/f"{arm}-{seed}.json").is_file() for arm in ("reference","candidate")):
            return {"status":"pending","aa_spread":noise,"pairs":pairs,"next_seed":seed}
        entry=pair(load(f"reference-{seed}"),load(f"candidate-{seed}"),seed,seed>=98700)
        pairs.append(entry)
        if entry["rate"] <= 1 or max(entry["rss_ratio"],entry["gpu_ratio"]) > 1.05:
            return {"status":"rejected","reason":"mandatory paired cold win or resource bound failed","aa_spread":noise,"pairs":pairs}
    mixed=pairs[:6]
    factual=pairs[6:]
    passed=(median(p["rate"] for p in mixed) >= 1.02 and median(p["rate"] for p in mixed)-1 > noise
            and median(p["rate"] for p in factual) >= 1.02 and any(r["calibration_calls"]==[20,15] for r in pairs)
            and all(median(r[k] for r in cohort) <= 1.02 for k in ("warm_wall","warm_p95") for cohort in (mixed,factual)))
    return {"status":"qualified" if passed else "rejected","reason":"complete prospective aggregate gate",
            "aa_spread":noise,"pairs":pairs}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    print(json.dumps(grade(p.parse_args().folder),indent=2,allow_nan=False))
