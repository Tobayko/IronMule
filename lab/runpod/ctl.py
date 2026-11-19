"""Control a RunPod lab pod from here: create, wait, put, run, job, get, terminate, list, spend.

The RunPod key is never read here: the session proxy injects it for api.runpod.io (a dummy bearer is sent).
Pod state (id, channel token, start time, price) lives in RUNPOD_STATE (default: ~/.ironmule-runpod.json), outside
the repository. Every create and terminate is appended to the spend ledger in that file.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.runpod.io/graphql"
STATE = Path(os.environ.get("RUNPOD_STATE", Path.home() / ".ironmule-runpod.json"))
AGENT = Path(__file__).with_name("agent.py")
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def gql(query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(API, body, {"Content-Type": "application/json", "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    if out.get("errors"):
        raise SystemExit(f"RunPod API error: {out['errors']}")
    return out["data"]


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"pods": {}, "ledger": []}


def save(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=1))


def pod_url(pod: str) -> str:
    return f"https://{pod}-8000.proxy.runpod.net"


def call(pod: str, method: str, path: str, data: bytes | None = None, timeout: int = 90):
    token = load()["pods"][pod]["token"]
    req = urllib.request.Request(pod_url(pod) + path, data, {"X-Token": token}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def create(a) -> None:
    token = secrets.token_urlsafe(24)
    env = [{"key": "TOKEN", "value": token},
           {"key": "AGENT_B64", "value": base64.b64encode(AGENT.read_bytes()).decode()}]
    inp = {"cloudType": a.cloud, "gpuCount": 1, "volumeInGb": 0, "containerDiskInGb": a.disk,
           "minVcpuCount": a.vcpu, "minMemoryInGb": a.mem, "gpuTypeId": a.gpu, "name": a.name,
           "imageName": a.image, "ports": "8000/http", "env": env, "startSsh": False,
           "dockerArgs": "bash -c 'mkdir -p /work && echo $AGENT_B64 | base64 -d > /agent.py && exec python3 /agent.py'"}
    if a.cuda:
        inp["allowedCudaVersions"] = a.cuda.split(",")
    q = """mutation($i: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: $i) {
           id costPerHr machine { gpuDisplayName } } }"""
    pod = gql(q, {"i": inp})["podFindAndDeployOnDemand"]
    state = load()
    state["pods"][pod["id"]] = {"token": token, "started": time.time(), "cost_per_hr": pod["costPerHr"],
                                "gpu": pod["machine"]["gpuDisplayName"], "name": a.name}
    state["ledger"].append({"event": "create", "pod": pod["id"], "t": time.time(), "cost_per_hr": pod["costPerHr"],
                            "gpu": pod["machine"]["gpuDisplayName"]})
    save(state)
    print(json.dumps(pod))


def wait(a) -> None:
    end = time.time() + a.timeout
    while time.time() < end:
        try:
            if json.loads(call(a.pod, "GET", "/ping", timeout=15)).get("ok"):
                print("ready")
                return
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(10)
    raise SystemExit("pod channel did not answer in time")


def put(a) -> None:
    print(call(a.pod, "PUT", "/file?path=" + urllib.request.quote(a.remote), Path(a.local).read_bytes(),
               timeout=600).decode())


def get(a) -> None:
    Path(a.local).write_bytes(call(a.pod, "GET", "/file?path=" + urllib.request.quote(a.remote), timeout=600))
    print(a.local)


def run(a) -> None:
    print(call(a.pod, "POST", "/run", json.dumps({"name": a.name, "cmd": a.cmd}).encode()).decode())


def job(a) -> None:
    out = json.loads(call(a.pod, "GET", f"/job?name={a.name}&tail={a.tail}"))
    print(out["tail"])
    print(f"--- known={out['known']} rc={out['rc']}")


def terminate(a) -> None:
    gql("mutation($i: PodTerminateInput!) { podTerminate(input: $i) }", {"i": {"podId": a.pod}})
    state = load()
    pod = state["pods"].pop(a.pod, {})
    hours = (time.time() - pod.get("started", time.time())) / 3600
    state["ledger"].append({"event": "terminate", "pod": a.pod, "t": time.time(), "hours": hours,
                            "est_usd": hours * pod.get("cost_per_hr", 0)})
    save(state)
    left = [p["id"] for p in gql("query { myself { pods { id } } }")["myself"]["pods"]]
    print(f"terminated {a.pod}: {hours:.2f} h, est ${hours * pod.get('cost_per_hr', 0):.2f}; remaining pods: {left}")


def listing(a) -> None:
    me = gql("query { myself { clientBalance currentSpendPerHr pods { id name desiredStatus costPerHr "
             "runtime { uptimeInSeconds } machine { gpuDisplayName } } } }")["myself"]
    print(json.dumps(me, indent=1))


def spend(a) -> None:
    state = load()
    done = sum(e.get("est_usd", 0) for e in state["ledger"] if e["event"] == "terminate")
    live = sum((time.time() - p["started"]) / 3600 * p["cost_per_hr"] for p in state["pods"].values())
    print(f"terminated pods: ${done:.2f}; running pods so far: ${live:.2f}; total est ${done + live:.2f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    c.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE", "ALL"])
    c.add_argument("--image", default=IMAGE)
    c.add_argument("--disk", type=int, default=80)
    c.add_argument("--vcpu", type=int, default=8)
    c.add_argument("--mem", type=int, default=30)
    c.add_argument("--cuda", default="12.8,12.9")
    c.add_argument("--name", default="ironmule-lab")
    c.set_defaults(fn=create)
    for name, fn, extra in [("wait", wait, [("--timeout", int, 900)]), ("put", put, []), ("get", get, []),
                            ("run", run, []), ("job", job, [("--tail", int, 40)]), ("terminate", terminate, [])]:
        s = sub.add_parser(name)
        s.add_argument("pod")
        if name in ("put",):
            s.add_argument("local"), s.add_argument("remote")
        if name == "get":
            s.add_argument("remote"), s.add_argument("local")
        if name in ("run", "job"):
            s.add_argument("name")
        if name == "run":
            s.add_argument("cmd")
        for flag, kind, default in extra:
            s.add_argument(flag, type=kind, default=default)
        s.set_defaults(fn=fn)
    sub.add_parser("list").set_defaults(fn=listing)
    sub.add_parser("spend").set_defaults(fn=spend)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
