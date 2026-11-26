"""Run each original GGUF gate subprocess in its own protected Linux process."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


STAGES = ("GATE_IDS", "TUNE_CHILD", "GATE_CANDIDATE")
MARKER = "IRONMULE_PROTECTED_GATE_CUDA "
MAX_PACKET = 1024 * 1024
STOCK = {"IRONMULE_KG_LIB":"", "IRONMULE_KG5":"0", "IRONMULE_GGML_PREFILL_BATCH":"0", "IRONMULE_W4":"0",
         "IRONMULE_GGML_PACK":"0", "IRONMULE_GGML_UBATCH":"0", "IRONMULE_CUBLAS_MIN_ROWS":"0"}


def fingerprint(code):
    return hashlib.sha256(code.encode()).hexdigest()


def filtered(env):
    return {k:v for k,v in env.items() if k.startswith("IRONMULE_") or k in ("PYTHONPATH","HF_HUB_OFFLINE")}


def install_driver(ge, channel):
    original = subprocess.run
    stages = {getattr(ge, key):key for key in STAGES}

    def route(command, *a, **kw):
        if not (isinstance(command, (list, tuple)) and len(command) == 4 and command[:2] == [sys.executable, "-c"]
                and command[2] in stages):
            return original(command, *a, **kw)
        if a or not kw.get("capture_output") or kw.get("text") is not True or kw.get("check") is not False:
            raise ValueError("unsupported original gate child protocol")
        request = {"stage":stages[command[2]], "code":fingerprint(command[2]), "gguf":command[3],
                   "input":kw.get("input"), "env":filtered(kw["env"])}
        channel.write(json.dumps(request, separators=(",",":"))+"\n")
        channel.flush()
        response = json.loads(sys.stdin.readline(MAX_PACKET))
        return subprocess.CompletedProcess(command, response["returncode"], response["stdout"], response["stderr"])

    subprocess.run = route
    return original


def checked_request(layout, request):
    from training.trial_release import require

    base = layout.env
    stage = request["stage"]
    require(stage in STAGES and request["gguf"] == str(layout.args.gguf) and
            request["code"] == layout.gate_scripts[stage], "different gate program or model")
    env = {**base, "PYTHONPATH": base["PYTHONPATH"].split(os.pathsep)[0]+os.pathsep+base["PYTHONPATH"],
           "IRONMULE_CUDA_GRAPHS_ROWS":"1"}
    if stage == "GATE_CANDIDATE":
        data = json.loads(request["input"])
        require(data["snapshot"] == str(layout.args.gguf) and data["kg5_shape"] is False and
                type(data["reference"]) is bool and type(data["shared"]) is bool and
                data["settings"].keys() == {"flash_attn","kv_unified","kv_f32"} and
                all(type(v) is bool for v in data["settings"].values()) and
                2049 <= len(data["ids"]) <= 8193 and all(type(v) is int and v >= 0 for v in data["ids"]),
                "different gate workload or child settings")
        if data["reference"]:
            env.update(STOCK)
    else:
        require(request["input"] is None, "unexpected gate subprocess input")
    require(request["env"] == {**filtered(env), "HF_HUB_OFFLINE":"1"}, "different gate subprocess environment")
    return env


def exchange(driver, layout):
    from lab.rp.restricted_worker import open_worker
    from training.trial_release import require

    from training.code_trial import read, save
    records = read(layout.args.gate_driver_record) if layout.args.gate_driver_record.exists() else []
    calls = 0
    while line := driver.stdout.readline(MAX_PACKET):
        require(len(line)<MAX_PACKET and calls<16, "bounded gate protocol exceeded")
        calls += 1
        req = json.loads(line)
        env = checked_request(layout,req)
        left = int(layout.args.deadline-time.monotonic())
        require(left>0,"protected gate deadline expired")
        argv = [layout.command[0], str(layout.profile.parent / Path(__file__).name), req["stage"], req["code"], req["gguf"]]
        with open_worker(argv,env=env,cwd=layout.out,output=layout.out,readonly=layout.readonly,
                         timeout=min(left,3600),uid=layout.args.worker_uid,gid=layout.args.worker_uid,
                         text=False,stderr=subprocess.PIPE) as child:
            stdout, log = child.communicate(None if req["input"] is None else req["input"].encode(),timeout=left)
            status=child.returncode
        stderr=log.decode(errors="replace")
        proof=[json.loads(s[len(MARKER):]) for s in stderr.splitlines() if s.startswith(MARKER)]
        require(len(stdout)<MAX_PACKET//2 and len(proof)==1 and proof[0]["stage"]==req["stage"]
                and proof[0]["provider"]=="cuda" and proof[0]["gpu_layers"]==999,
                "protected gate stage has no actual CUDA registry sentinel: "+stderr[-350:])
        records.append({**proof[0],"child_status":status,"code_sha256":req["code"]})
        save(layout.args.gate_driver_record, records)
        answer={"returncode":status,"stdout":stdout.decode(errors="strict"),"stderr":stderr[-65536:]}
        driver.stdin.write((json.dumps(answer,separators=(",",":"))+"\n").encode())
        driver.stdin.flush()
    return records


def child_main(stage, expected, model):
    os.environ["HF_HUB_OFFLINE"]="1"
    from ironmule import ggml_engine as ge
    from ironmule_devices import ggml_devices

    if stage not in STAGES or fingerprint(getattr(ge,stage)) != expected:
        raise ValueError("original protected gate script differs")
    bindings=ge.bindings()
    bindings.llama_backend_init()
    devices=ggml_devices(bindings)
    if len(devices)!=1 or devices[0]["provider"]!="cuda" or ge.gpu_layers(bindings)!=999:
        raise RuntimeError("fresh protected GGUF process has no registered CUDA offload device")
    proof={"stage":stage,"provider":devices[0]["provider"],"device":devices[0]["description"],"gpu_layers":999}
    print(MARKER+json.dumps(proof),file=sys.stderr,flush=True)
    sys.argv=["-c",model]
    exec(compile(getattr(ge,stage),"<string>","exec"),{"__name__":"__main__"})


if __name__=="__main__":
    child_main(*sys.argv[1:])
