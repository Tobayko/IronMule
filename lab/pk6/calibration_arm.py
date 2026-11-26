"""Trusted external client for one complete, isolated PR57 startup and serving workload."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.request import urlopen

BASE = "8cf0a3106f09a3338eb53c28047af7e36fc7b340"
DEVICE = "112d8a87ea0f7aab9de32ad3b2f2d65318c3bce1044ee30b75f78c528c923e2b"
FILES = {"reference": "80a41c59bfe8088b305529cca5164134c25d4b5a0ce9572877cb25a2b7d269fd",
         "candidate": "f60951799571833dfa663227751a41d9a36a88b54405518240df762cdbf340f7"}
GGUF = "efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689"
MODEL = "gguf/Qwen3-32B-Q4_K_M"
from ironmule.factual_gate import cases, correct, MAX_TOKENS
from ironmule_product import compare


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def store(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w") as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def card():
    fields = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True).strip().split(", ")
    if len(fields) != 3 or hashlib.sha256(fields[0].encode()).hexdigest() != DEVICE:
        raise ValueError("different physical device")
    return {"device_sha256": DEVICE, "name": fields[1], "driver": fields[2]}


def tree_digest(tree):
    h = hashlib.sha256()
    for pkg in ("ironmule","ironmule_product"):
        for p in sorted((tree/pkg).rglob("*.py")):
            if p.relative_to(tree).as_posix() != "ironmule/ggml_engine.py":
                h.update(p.relative_to(tree).as_posix().encode()+b"\0"+bytes.fromhex(sha(p)))
    return h.hexdigest()


def serve(root, arm, seed, run_id, factual):
    base = root/"reference"
    tree = root/arm
    final = root/"out"/(run_id+".json")
    if final.exists():
        return
    source = sha(tree/"ironmule/ggml_engine.py")
    if source != FILES[arm] or tree_digest(tree) != tree_digest(base):
        raise ValueError("unbound source")
    hw = card()
    with tempfile.TemporaryDirectory(dir=root) as scratch:
        state = Path(scratch)/"state"
        shutil.copytree(root/"stock",state)
        for p in (state, *state.rglob("*")):
            os.chown(p,60000,60000)
            p.chmod(0o700 if p.is_dir() else 0o600)
        os.chmod(scratch,0o755)
        env = {"HOME": str(state), "TMPDIR":str(state), "PATH":"/usr/local/bin:/usr/bin:/bin:/usr/local/cuda/bin",
               "PYTHONPATH":str(tree),"PYTHONDONTWRITEBYTECODE":"1","OPENBLAS_NUM_THREADS":"1",
               "IRONMULE_GGML_BINDINGS":str(root/"bindings"),
               "LLAMA_CPP_LIB_PATH":str(root/"bindings/vendor/llama.cpp/build/bin"), "IRONMULE_GGML_SEQ_CTX":"2048",
               "IRONMULE_PS1":"1","IRONMULE_PK6":"1","IRONMULE_GGML_DH2":"0","IRONMULE_SV5":"0",
               "IRONMULE_ADMISSION_LEARN":"0","IRONMULE_GGML_TRACE":str(state/"trace.txt")}
        env["LD_LIBRARY_PATH"] = env["LLAMA_CPP_LIB_PATH"] + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        if arm == "candidate":
            env["IRONMULE_TRIAL_LOCAL_CALIBRATION"]="1"
        gguf=Path("/models/Qwen3-32B-Q4_K_M.gguf")
        command=[sys.executable,"-m","ironmule_cli","serve","--gguf",str(gguf),"--state-dir",str(state),"--agent","on","--no-thinking",
                 "--host","127.0.0.1","--port","8135","--api-key-env","IRONMULE_EXPERIMENT_LOCAL_KEY"]
        url="http://127.0.0.1:8135"
        server={"base_url":url,"model_id":MODEL}
        high=[0,0]
        done=threading.Event()
        started=time.monotonic()
        with (root/"out"/(run_id+".log")).open("ab") as log:
            proc=subprocess.Popen(command,cwd=tree,env=env,stdout=log,stderr=log,user=60000,group=60000,
                                  extra_groups=[],start_new_session=True)
            def monitor():
                while not done.is_set():
                    try:
                        status=Path(f"/proc/{proc.pid}/status").read_text().splitlines()
                        high[0]=max(high[0],int(next(l for l in status if l.startswith("VmHWM:")).split()[1]))
                        mem=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True,timeout=3)
                        high[1]=max(high[1],int(mem.strip()))
                    except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
                        pass
                    done.wait(2)
            watcher=threading.Thread(target=monitor,daemon=True)
            watcher.start()
            try:
                while time.monotonic()-started < 420 and proc.poll() is None:
                    try:
                        with urlopen(url+"/health",timeout=2) as r:
                            health=json.load(r)
                        if health["ready"] and health["ggml"]["gate"]["passes"]:
                            break
                    except (OSError,ValueError,KeyError):
                        pass
                    time.sleep(.3)
                else:
                    raise RuntimeError(f"server failed to become ready (exit {proc.poll()})")
                if health["ggml"]["sequences"]["sequences"]!=32:
                    raise ValueError("source failed to qualify the same target width")
                output={"binding":hw,"arm":arm,"seed":seed,"factual_first":factual,"base":BASE,
                        "sha256":source,"fixture_sha256":sha(base/"ironmule_product/compare.py"),
                        "health":health["ggml"]["gate"],"width":32,"startup_s":time.monotonic()-started,"waves":[]}
                for role in (["facts","cold","warm"] if factual else ["cold","warm","facts"]):
                    qs=cases(seed)
                    prompts=[q.prompt for q in qs] if role=="facts" else compare.performance_prompts(seed if role=="cold" else seed^317,32,32)
                    rows,wall=compare.batch("ironmule",server,prompts,8 if role=="facts" else 32,MAX_TOKENS if role=="facts" else 32,time.monotonic()+120)
                    delivered=time.monotonic()-started
                    reduced=[{k:r[k] for k in ("text","prompt_tokens","tokens","finish_reason")} for r in rows]
                    out={"role":role,"delivered_s":delivered,"wall_s":wall,"answers":reduced,
                         "ttft_ms":[r["ttft_ms"] for r in rows],"work_sha256":hashlib.sha256(json.dumps(reduced,sort_keys=True).encode()).hexdigest()}
                    if role=="facts":
                        out["correct"]=[correct(q,r["text"]) for q,r in zip(qs,rows,strict=True)]
                    output["waves"].append(out)
                output.update(peak_rss_kib=high[0],peak_gpu_mib=high[1],worker_complete_s=time.monotonic()-started)
            finally:
                done.set()
                watcher.join(5)
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:
                        proc.wait(10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL)
                        proc.wait()
                if (state/"trace.txt").is_file():
                    shutil.copyfile(state/"trace.txt", root/"out"/(run_id+".trace"))
        last=(root/"out"/(run_id+".trace")).read_text().splitlines()
        a=max(i for i,l in enumerate(last) if l.startswith("# init measure_pk2 begin"))
        b=next(i for i in range(a+1,len(last)) if last[i].startswith("# init "))
        details=json.loads(next(l for l in last[b:] if l.startswith("# init result ")).removeprefix("# init result "))
        output.update(pk2_calls=sum(bool(l) and not l.startswith("#") for l in last[a+1:b]),
                      pk2=details["pk2"],pilot=details["pilot"],trace_sha256=sha(root/"out"/(run_id+".trace")),
                      evaluator_sha256=sha(Path(__file__)),kernel_sha256=sha(next((state/"kernels/w4").glob("*/kernel.so"))),
                      table_sha256=sha(next((state/"kernels/w4").glob("*/dispatch.table"))))
        store(final,output)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("arm",choices=list(FILES))
    p.add_argument("seed",type=int)
    p.add_argument("run_id")
    p.add_argument("--facts-first",action="store_true")
    a=p.parse_args()
    serve(a.root,a.arm,a.seed,a.run_id,a.facts_first)
