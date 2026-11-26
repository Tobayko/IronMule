"""Launch one read-only, restricted native worker and retain its full answers, timings and trace."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from calibration_arm import card, sha, tree_digest, FILES
from restricted_worker import run


def serve(root,arm,seed,name,factual):
    evidence=root/"ordered"
    evidence.mkdir(mode=0o700,exist_ok=True)
    final=evidence/(name+".json")
    if final.exists():
        return
    tree=root/arm
    if sha(tree/"ironmule/ggml_engine.py")!=FILES[arm] or tree_digest(tree)!=tree_digest(root/"reference"):
        raise ValueError("different source tree")
    hw=card()
    trust=root/"trusted"
    gguf=Path("/models/Qwen3-32B-Q4_K_M.gguf")
    lib,= (root/"stock/kernels/w4").glob("*/kernel.so")
    table=lib.with_name("dispatch.table")
    with tempfile.TemporaryDirectory(dir=root) as temp:
        output=Path(temp)
        os.chown(output,60000,60000)
        env={"PYTHONPATH":str(tree),"PATH":"/usr/local/bin:/usr/bin:/bin:/usr/local/cuda/bin",
             "IRONMULE_GGML_BINDINGS":str(root/"bindings"),"LLAMA_CPP_LIB_PATH":str(root/"bindings/vendor/llama.cpp/build/bin"),
             "OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","IRONMULE_PILOT":"1","IRONMULE_GGML_GPU_LAYERS":"999",
             "IRONMULE_KG_LIB":str(lib),"IRONMULE_W4_TABLE":str(table),"IRONMULE_PS1":"1","IRONMULE_PS1_WARM":"0",
             "IRONMULE_PS1_CONCAT":"0","IRONMULE_PK6":"1","IRONMULE_GGML_DH2":"0","IRONMULE_SV5":"0",
             "IRONMULE_GGML_PACK":"1","IRONMULE_GGML_UBATCH":"8192","IRONMULE_CUDA_GRAPHS_ROWS":"32",
             "IRONMULE_GGML_TRACE":str(output/"trace.txt")}
        env["LD_LIBRARY_PATH"]=env["LLAMA_CPP_LIB_PATH"]+":"+os.environ.get("LD_LIBRARY_PATH","")
        for key in ("CUDA_VISIBLE_DEVICES",):
            if key in os.environ:
                env[key]=os.environ[key]
        if arm=="candidate":
            env["IRONMULE_TRIAL_LOCAL_CALIBRATION"]="1"
        allowed=[tree,trust,gguf,lib,table,root/"bindings"]
        argv=[sys.executable,str(trust/"ordered_worker.py"),"--gguf",str(gguf),"--output",str(output/"result.json"),"--seed",str(seed)]
        if factual:
            argv.append("--factual-first")
        env["IRONMULE_TRIAL_START_MONOTONIC"]=str(time.monotonic())
        proc=run(argv,env=env,cwd=output,output=output,readonly=allowed,timeout=180)
        (evidence/(name+".log")).write_text(proc.stdout)
        if proc.returncode:
            raise RuntimeError(f"restricted worker exited {proc.returncode}: {proc.stdout[-800:]}")
        from native_trial import copy_result
        raw=evidence/(name+".worker")
        copy_result(output/"result.json",raw,60000)
        copy_result(output/"trace.txt",evidence/(name+".trace"),60000)
        row=json.loads(raw.read_bytes())
        row.update(target=hw,arm=arm,evaluator_sha256=sha(Path(__file__)),worker_sha256=sha(trust/"ordered_worker.py"),
                   sampled_trace_sha256=sha(evidence/(name+".trace")),ledger_elapsed_s=time.monotonic()-float(env["IRONMULE_TRIAL_START_MONOTONIC"]))
        tmp=final.with_suffix(".incoming")
        tmp.write_text(json.dumps(row,sort_keys=True,allow_nan=False))
        tmp.replace(final)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("arm",choices=list(FILES))
    p.add_argument("seed",type=int)
    p.add_argument("name")
    p.add_argument("--facts-first",action="store_true")
    a=p.parse_args()
    serve(a.root,a.arm,a.seed,a.name,a.facts_first)
