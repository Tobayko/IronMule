"""Submit caller-provided text to the target-bound native worker and return its actual answers."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from ironmule.factual_gate import cases, correct, MAX_TOKENS
from ironmule_product.compare import performance_prompts
from lab.pk1.native_trial import copy_result
from lab.pk6 import calibration_arm
from lab.pk6 import ordered_release as deploy
from lab.rp import restricted_worker
from training import scoped_release as release
from training.trial_release import require, switch_lock
from training.code_trial import read, save


def family(request):
    prompts = request["prompts"]
    limit = request["max_tokens"]
    seed = request.get("workload_seed")
    require(isinstance(prompts,list) and 1 <= len(prompts) <= 32 and all(
        isinstance(p,str) and 0 < len(p.encode()) <= 16384 for p in prompts)
        and type(limit) is int and 1 <= limit <= 128, "invalid ordered request")
    if type(seed) is int and 0 <= seed < 2**31:
        if len(prompts) == 32 and limit == 32 and prompts == performance_prompts(seed,32,32):
            return "mixed-first-32"
        if len(prompts) == 18 and limit == MAX_TOKENS and prompts == [q.prompt for q in cases(seed)]:
            return "factual-first-18"
    return "unknown"


def monitor_completion(state, pin, family, elapsed):
    directory = state/"deployment"
    gate_file = directory/"ordered-client.json"
    with switch_lock(directory):
        gate = read(gate_file)
        if not gate["passed"] or gate["pin"] != pin:
            return
        matching = [p["reference_s"] for p in gate["pairs"] if p["family"] == family]
        if not matching:
            return
        limit = 1.02 * max(matching)
        path = directory/("ordered-client-"+family+".json")
        previous = read(path) if path.exists() else {}
        count = previous.get("consecutive_late",0) if previous.get("pin") == pin else 0
        count = count + 1 if elapsed > limit else 0
        value = {"pin":pin,"limit_s":limit,"complete_s":elapsed,"consecutive_late":count}
        save(path,value)
        if count >= 2:
            save(gate_file,{**gate,"passed":False,"reason":"two complete requests exceeded the reference time envelope",
                           "withdrawal":value})
        return value


def serve(args, request, *, trial_arm=None):
    started = time.monotonic()
    kind = family(request)
    root = args.root.resolve(strict=True)
    receipt = release.pinned(args.state, args.pin)
    bound = receipt["qualification"]["binding"]
    runtime = bound["runtime"]
    selected = release.selected(args.state,args.pin)
    if trial_arm is not None:
        require(trial_arm in ("reference","candidate") and selected == "candidate", "client trial requires a fresh eligible source")
        selected = trial_arm
    else:
        client_gate = args.state/"deployment"/"ordered-client.json"
        try:
            gate = json.loads(client_gate.read_bytes())
            if not gate["passed"] or gate["pin"] != args.pin or kind not in gate["families"]:
                selected = "reference"
        except (OSError,ValueError,KeyError,TypeError):
            selected = "reference"
    decision = deploy.choose_profile(args, receipt, selected, kind)
    require(calibration_arm.card() == bound["target"], "device differs from registered ordered deployment")
    model = Path("/models/Qwen3-32B-Q4_K_M.gguf")
    st = model.stat()
    require([st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns] == runtime["model_stat"],
            "use a locally qualified model")
    kernel, = (root/"stock/kernels/w4").glob("*/kernel.so")
    table = kernel.with_name("dispatch.table")
    require(release.checksum(kernel) == runtime["w4_sha256"] and release.checksum(table) == runtime["table_sha256"],
            "local kernel artifacts changed")
    worker = Path(__file__).with_name("ordered_client_worker.py")
    bindings = root/"bindings"
    native = bindings/"vendor/llama.cpp/build/bin"
    def run(arm):
        tree = root/arm
        release.check_tree(receipt, tree, arm == "candidate")
        env = {**runtime["flags"], "PATH":"/usr/local/bin:/usr/bin:/bin:/usr/local/cuda/bin",
               "PYTHONPATH":str(tree)+":"+str(root/"trusted"), "IRONMULE_GGML_BINDINGS":str(bindings),
               "LLAMA_CPP_LIB_PATH":str(native), "LD_LIBRARY_PATH":str(native)+":"+os.environ.get("LD_LIBRARY_PATH",""),
               "IRONMULE_PILOT":"1", "IRONMULE_GGML_GPU_LAYERS":"999","IRONMULE_KG_LIB":str(kernel),
               "IRONMULE_W4_TABLE":str(table),"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        if arm == "candidate":
            env["IRONMULE_TRIAL_LOCAL_CALIBRATION"] = "1"
        with tempfile.TemporaryDirectory(dir=root) as scratch, tempfile.TemporaryDirectory(dir=root) as temp:
            inputs, output = Path(scratch), Path(temp)
            inputs.chmod(0o711)
            path = inputs/"request.json"
            path.write_text(json.dumps({"prompts":request["prompts"],"max_tokens":request["max_tokens"],
                                       "binding":bound,"kernel":str(kernel),"model":str(model),"arm":arm}))
            path.chmod(0o444)
            os.chown(output,60000,60000)
            cmd = [sys.executable,str(worker),str(path)]
            process = restricted_worker.run(cmd,env=env,cwd=output,output=output,timeout=200,
                      readonly=[tree,root/"trusted",model,kernel,table,bindings,path,worker])
            require(process.returncode == 0, "ordered worker failed: "+process.stdout[-700:])
            dest = inputs/"answers.json"
            copy_result(output/"response.json",dest,60000)
            result = json.loads(dest.read_bytes())
        require(len(result["answers"]) == len(request["prompts"]) and result["pilot_decisions"] > 0,
                "incomplete ordered delivery")
        if kind == "factual-first-18":
            require(all(correct(q,a["answer"]) for q,a in zip(cases(request["workload_seed"]),result["answers"],strict=True)),
                    "factual quality withdrawal")
        return result
    arm = decision["profile"]
    try:
        result = run(arm)
    except (OSError,RuntimeError,ValueError,KeyError) as exc:
        if arm != "candidate":
            raise
        if trial_arm is not None:
            raise
        save(args.state/"deployment"/"ordered-client.json",{"passed":False,"pin":args.pin,
             "reason":f"ordered client withdrew candidate: {exc}"[:300]})
        arm = "reference"
        result = run(arm)
    monitoring = None
    if arm == "candidate" and trial_arm is None:
        monitoring = monitor_completion(args.state,args.pin,kind,time.monotonic()-started)
    return {**result,"arm":arm,"decision":decision,"family":kind,"complete_s":time.monotonic()-started,
            "monitor":monitoring}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","state","weights","requests"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--pin",required=True)
    p.add_argument("--weights-pin",required=True)
    p.set_defaults(ablate=False)
    args=p.parse_args()
    print(json.dumps(serve(args,json.loads(args.requests.read_bytes())),allow_nan=False))
