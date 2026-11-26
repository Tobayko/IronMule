"""Verify the complete work, target binding and prospective gate of the ordered calibration lane."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from ironmule.factual_gate import cases, correct

try:
    from .calibration_grade import BOARD, CANDIDATE, KERNEL, REFERENCE, TABLE, require
except ImportError:  # Standalone pinned grader on the target.
    from calibration_grade import BOARD, CANDIDATE, KERNEL, REFERENCE, TABLE, require

AA = 99100
MIXED = tuple(range(99101,99107))
FACTS = (99201,99202)
EXPECTED_GGUF = "efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689"


def p95(values):
    return sorted(values)[math.ceil(.95*len(values))-1]


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def validate(row, arm, seed, factual):
    b=row["binding"]
    require(row["target"]["device_sha256"]==BOARD and row["arm"]==arm and row["seed"]==seed
            and row["source_sha256"]==(REFERENCE if arm=="reference" else CANDIDATE), "unbound ordered source")
    require(b["model_sha256"]==EXPECTED_GGUF and b["w4_sha256"]==KERNEL and b["table_sha256"]==TABLE
            and b["seqs"]==32 and b["seq_ctx"]==2048 and b["gpu_layers"]==999 and b["speculate"] is True
            and b["settings"]=={"flash_attn":True,"kv_unified":False,"kv_f32":False} and bool(b["native"]["sha256"])
            and bool(row["cuda_vendor"]) and not row["reused"] and row["pilot_decisions"]>0, "different execution profile")
    pk=row["pk2"]
    require(row["pk2_calls"] in ((20,) if arm=="reference" else (15,20))
            and pk["rows"]==1908 and pk["calibration"]=="mix" and "likelihood ratio " in pk["reason"]
            and 0 < float(pk["reason"].split("likelihood ratio ")[1]) <= 1.005
            and set(pk["burst_ms"])=={"0","128","256","512","1024"} and all(ms>0 for ms in pk["burst_ms"].values()), "missing fresh NLL or local calibration")
    require(row["facts"]==[True]*18 and len(row["factual"])==18 and all(correct(q,a["answer"]) for q,a in zip(cases(seed),row["factual"],strict=True)), "incorrect facts")
    for name in ("cold","warm"):
        wave=row[name]
        require(wave["completed"]==len(wave["answers"])==len(wave["ttfts_s"])==32 and wave["tokens"]==1024
                and all(len(a["output_ids"])==32 and a["stop"] in ("length","eos") and a["prompt_digest"] for a in wave["answers"])
                and wave["output_sha256"]==canonical(wave["answers"])
                and all(0 < t <= wave["wall_s"] for t in wave["ttfts_s"]), "incomplete timed batch")
    require(row["initialized_s"]>0 and row["initialized_s"]+row["cold"]["wall_s"] <=row["cold_s"]+.02 and
            row["cold_s"]+row["warm"]["wall_s"] <= row["broker_elapsed_s"]+.05 and
            len(row["device_used_bytes"])==4 and min(row["device_used_bytes"])>0 and row["peak_rss_kib"]>0, "invalid resource observation")
    batches=row["factual_first"]
    require(len(batches)==(2 if factual else 0), "different first workload")
    for qseed,batch in zip((seed,seed^419) if factual else (),batches,strict=True):
        require(batch["seed"]==qseed and batch["correct"]==[True]*18 and len(batch["answers"])==len(batch["ttfts_s"])==18
                and batch["wall_s"]<batch["completed_s"]<row["cold_s"] and all(0<t<=batch["wall_s"] for t in batch["ttfts_s"])
                and all(correct(q,a["answer"]) and a["output_ids"] and a["stop"] in ("length","eos")
                         for q,a in zip(cases(qseed),batch["answers"],strict=True)), "incorrect first facts")


def price(ref,cand,seed,factual=False,same=False):
    validate(ref,"reference",seed,factual)
    validate(cand,"reference" if same else "candidate",seed,factual)
    keys=("binding","target","worker_sha256","harness_sha256","evaluator_sha256","mask_bits","cuda_vendor","pilot_sha256","factual")
    require(all(ref[k]==cand[k] for k in keys) and ref["pk2"]["min_rows"]==cand["pk2"]["min_rows"]
            and all(ref[k]["answers"]==cand[k]["answers"] and ref[k]["geometry"]==cand[k]["geometry"] for k in ("cold","warm"))
            and all(r["answers"]==c["answers"] for r,c in zip(ref["factual_first"],cand["factual_first"],strict=True)), "changed native work, hardware or output")
    def first(row):
        return row["broker_to_process_s"] + (row["factual_first"][0]["completed_s"] if factual else row["cold_s"])
    rw,cw=(r["factual_first"][1] if factual else r["warm"] for r in (ref,cand))
    return {"seed":seed,"facts_first":factual,"reference_s":first(ref),"candidate_s":first(cand),
            "rate":first(ref)/first(cand),"warm_wall":cw["wall_s"]/rw["wall_s"],
            "warm_p95":p95(cw["ttfts_s"])/p95(rw["ttfts_s"]),"rss_ratio":cand["peak_rss_kib"]/ref["peak_rss_kib"],
            "gpu_ratio":max(cand["device_used_bytes"])/max(ref["device_used_bytes"]),
            "calibration_calls":[ref["pk2_calls"],cand["pk2_calls"]]}


def grade(folder):
    def load(name):
        return json.loads((folder/(name+".json")).read_bytes())
    aa=price(load("aa-0"),load("aa-1"),AA,same=True)
    spread=max(aa["rate"],1/aa["rate"])-1
    pairs=[]
    for seed in (*MIXED,*FACTS):
        if not all((folder/f"{a}-{seed}.json").is_file() for a in ("reference","candidate")):
            return {"status":"pending","next_seed":seed,"pairs":pairs,"aa_spread":spread}
        entry=price(load(f"reference-{seed}"),load(f"candidate-{seed}"),seed,seed in FACTS)
        pairs.append(entry)
        if entry["rate"]<=1 or max(entry["rss_ratio"],entry["gpu_ratio"])>1.05:
            return {"status":"rejected","reason":"mandatory paired cold win or resource bound failed","pairs":pairs,"aa_spread":spread}
    mixed,facts=pairs[:6],pairs[6:]
    passed=(median(x["rate"] for x in mixed)>=1.02 and median(x["rate"] for x in mixed)-1>spread
            and median(x["rate"] for x in facts)>=1.02 and any(x["calibration_calls"]==[20,15] for x in pairs)
            and all(median(p[k] for p in group)<=1.02 for group in (mixed,facts) for k in ("warm_wall","warm_p95")))
    return {"status":"qualified" if passed else "rejected","reason":"complete prospective ordered gate","pairs":pairs,"aa_spread":spread}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    print(json.dumps(grade(p.parse_args().folder),allow_nan=False,indent=2))
