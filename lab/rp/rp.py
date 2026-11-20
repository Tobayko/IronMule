"""Drive RunPod pods for IronMule GPU work: deploy, talk to lab/rp/agent.py, stop, track spend.

State (pod id, token, start time, spend ledger) lives in $RP_STATE (default ~/.cache/ironmule/rp.json), never in git.

  python lab/rp/rp.py deploy --gpu "NVIDIA A100-SXM4-80GB" [--disk 200] [--cloud COMMUNITY]
  python lab/rp/rp.py wait                       # until the agent answers
  python lab/rp/rp.py put LOCAL REMOTE
  python lab/rp/rp.py get REMOTE LOCAL
  python lab/rp/rp.py run "nvidia-smi"           # short command
  python lab/rp/rp.py job NAME "long command"    # background; then: python lab/rp/rp.py log NAME
  python lab/rp/rp.py stop                       # terminate the pod and book its cost
  python lab/rp/rp.py spend
Authentication to api.runpod.io is injected by the session proxy.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(os.environ.get("RP_STATE", Path.home() / ".cache/ironmule/rp.json"))
AGENT = Path(__file__).with_name("agent.py")
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
BUDGET = float(os.environ.get("RP_BUDGET", 25.0))  # owner: 20 USD, then 5 more (2026-11-20)


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {"spent": 0.0, "pods": []}


def save(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))
    os.chmod(STATE, 0o600)


def curl(args, data=None, timeout=100):
    cmd = ["curl", "-sS", "-m", str(timeout), *args]
    done = subprocess.run(cmd, input=data, capture_output=True)
    if done.returncode:
        raise RuntimeError(done.stderr.decode()[-500:])
    return done.stdout


def gql(query, variables=None):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    out = json.loads(curl(["-X", "POST", "https://api.runpod.io/graphql", "-H", "Content-Type: application/json",
                           "--data-binary", "@-"], body, 60))
    if out.get("errors"):
        raise RuntimeError(json.dumps(out["errors"])[:1000])
    return out["data"]


def current(state):
    pod = state.get("pod")
    if not pod:
        sys.exit("no pod")
    return pod


def agent(pod, method, path, data=None, timeout=100, raw=False):
    url = f"https://{pod['id']}-8000.proxy.runpod.net{path}"
    args = ["-X", method, url, "-H", f"X-Token: {pod['token']}"]
    if data is not None:
        args += ["--data-binary", "@-", "-H", "Content-Type: application/octet-stream"]
    out = curl(args, data, timeout)
    return out if raw else json.loads(out)


def deploy(a):
    state = load()
    if state.get("pod"):
        sys.exit(f"pod already running: {state['pod']['id']}")
    if state["spent"] >= BUDGET:
        sys.exit(f"budget spent: {state['spent']:.2f}")
    token = secrets.token_hex(24)
    agent_b64 = base64.b64encode(AGENT.read_bytes()).decode()
    start = "bash -c 'echo $AGENT_B64 | base64 -d > /agent.py && exec python3 /agent.py'"
    q = """mutation($i: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $i) {
           id costPerHr machine { gpuDisplayName } } }"""
    i = {"cloudType": a.cloud, "gpuCount": a.count, "gpuTypeId": a.gpu, "name": "ironmule-lab",
         "imageName": a.image, "containerDiskInGb": a.disk, "volumeInGb": 0, "minVcpuCount": a.vcpu,
         "minMemoryInGb": a.mem, "ports": "8000/http", "dockerArgs": start, "startSsh": False,
         "env": [{"key": "AGENT_TOKEN", "value": token}, {"key": "AGENT_B64", "value": agent_b64}]}
    pod = gql(q, {"i": i})["podFindAndDeployOnDemand"]
    state["pod"] = {"id": pod["id"], "token": token, "cost": pod["costPerHr"], "gpu": a.gpu, "started": time.time()}
    save(state)
    print(json.dumps({k: v for k, v in state["pod"].items() if k != "token"}))


def wait(a):
    pod = current(load())
    end = time.time() + a.timeout
    while time.time() < end:
        try:
            if agent(pod, "GET", "/health", timeout=15).get("ok"):
                print("agent ready")
                return
        except Exception:
            pass
        time.sleep(10)
    sys.exit("agent not ready")


def stop(a):
    state = load()
    pod = current(state)
    gql("mutation($i: PodTerminateInput!) { podTerminate(input: $i) }", {"i": {"podId": pod["id"]}})
    hours = (time.time() - pod["started"]) / 3600
    cost = hours * pod["cost"]
    state["spent"] += cost
    state["pods"].append({"id": pod["id"], "gpu": pod["gpu"], "hours": round(hours, 3), "usd": round(cost, 3)})
    state["pod"] = None
    save(state)
    print(f"terminated {pod['id']} after {hours:.2f} h, {cost:.2f} USD; total {state['spent']:.2f} of {BUDGET}")


def spend(a):
    state = load()
    pod = state.get("pod")
    live = (time.time() - pod["started"]) / 3600 * pod["cost"] if pod else 0
    print(json.dumps({"booked": round(state["spent"], 3), "live": round(live, 3), "budget": BUDGET,
                      "pod": pod and {k: v for k, v in pod.items() if k != "token"}, "history": state["pods"]}))


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="op", required=True)
    d = s.add_parser("deploy")
    d.add_argument("--gpu", required=True)
    d.add_argument("--count", type=int, default=1)
    d.add_argument("--cloud", default="COMMUNITY")
    d.add_argument("--image", default=IMAGE)
    d.add_argument("--disk", type=int, default=200)
    d.add_argument("--vcpu", type=int, default=8)
    d.add_argument("--mem", type=int, default=60)
    w = s.add_parser("wait")
    w.add_argument("--timeout", type=int, default=900)
    for name in ("put", "get"):
        x = s.add_parser(name)
        x.add_argument("src")
        x.add_argument("dst")
    r = s.add_parser("run")
    r.add_argument("cmd")
    r.add_argument("--timeout", type=int, default=90)
    j = s.add_parser("job")
    j.add_argument("name")
    j.add_argument("cmd")
    lg = s.add_parser("log")
    lg.add_argument("name")
    lg.add_argument("--tail", type=int, default=40)
    k = s.add_parser("kill")
    k.add_argument("name")
    s.add_parser("stop")
    s.add_parser("spend")
    a = p.parse_args()
    if a.op in ("deploy", "wait", "stop", "spend"):
        return {"deploy": deploy, "wait": wait, "stop": stop, "spend": spend}[a.op](a)
    pod = current(load())
    if a.op == "put":
        print(agent(pod, "POST", "/put?path=" + a.dst, Path(a.src).read_bytes(), timeout=600))
    elif a.op == "get":
        Path(a.dst).write_bytes(agent(pod, "GET", "/get?path=" + a.src, raw=True, timeout=600))
        print(f"saved {a.dst}")
    elif a.op == "run":
        out = agent(pod, "POST", "/run", json.dumps({"cmd": a.cmd, "timeout": a.timeout}).encode())
        print(out["out"])
        print(f"[rc={out['rc']}]")
    elif a.op == "job":
        print(agent(pod, "POST", "/job", json.dumps({"cmd": a.cmd, "name": a.name}).encode()))
    elif a.op == "log":
        out = agent(pod, "GET", f"/job?name={a.name}&tail={a.tail}")
        print(out["tail"])
        print(f"[running={out['running']} rc={out['rc']}]")
    elif a.op == "kill":
        print(agent(pod, "POST", "/kill", json.dumps({"name": a.name}).encode()))


if __name__ == "__main__":
    main()
