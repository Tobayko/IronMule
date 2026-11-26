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
Use RUNPOD_API_KEY on the control machine or a credential-injecting proxy.
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
import urllib.request
from pathlib import Path

from accounting import (CLEANUP_SECONDS, DISK_USD_PER_GB_HOUR, MAX_BUDGET, event, held,
                        limit, locked, money, reserve, save as save_atomic)

STATE = Path(os.environ.get("RP_STATE", Path.home() / ".cache/ironmule/rp.json"))
AGENT = Path(__file__).with_name("agent.py")
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {"spent": 0.0, "pods": []}


def save(state):
    save_atomic(STATE, state)


def curl(args, data=None, timeout=100):
    cmd = ["curl", "-sS", "-m", str(timeout), *args]
    done = subprocess.run(cmd, input=data, capture_output=True)
    if done.returncode:
        raise RuntimeError(done.stderr.decode()[-500:])
    return done.stdout


def gql(query, variables=None):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    headers = {"Content-Type": "application/json"}
    if os.environ.get("RUNPOD_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["RUNPOD_API_KEY"]
    req = urllib.request.Request("https://api.runpod.io/graphql", data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        out = json.load(response)
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


def account():
    user = gql("{myself{clientBalance currentSpendPerHr pods{id name costPerHr}}}")["myself"]
    if not user or not isinstance(user.get("pods"), list):
        raise ValueError("authenticated account telemetry unavailable")
    money(user["clientBalance"])
    money(user["currentSpendPerHr"])
    return user


def book(a):
    state = load()
    if state.get("pod"):
        raise ValueError("an owned pod or unresolved create already exists")
    if a.count < 1 or a.disk < 1 or a.minutes < 1 or money(a.max_hourly) == 0:
        raise ValueError("expected positive count, disk, minutes and maximum hourly rate")
    user = account()
    rates = gql("""query($gpu:String!, $secure:Boolean!, $n:Int!) {
        gpuTypes(input:{id:$gpu}){id lowestPrice(input:{secureCloud:$secure,gpuCount:$n}){
            uninterruptablePrice stockStatus}}}""",
                {"gpu": a.gpu, "secure": a.cloud == "SECURE", "n": a.count})["gpuTypes"]
    quote = next((x["lowestPrice"] for x in rates if x["id"] == a.gpu), None)
    if not quote or not quote.get("uninterruptablePrice"):
        raise ValueError("no matching on-demand quote")
    disk_rate = a.disk * DISK_USD_PER_GB_HOUR
    quote_rate = money(quote["uninterruptablePrice"]) * a.count + disk_rate
    if quote_rate > a.max_hourly:
        raise ValueError("quoted compute and disk exceed hourly cap")
    amount = (a.minutes * 60 + CLEANUP_SECONDS) / 3600 * a.max_hourly
    reserve(state, amount, user["clientBalance"])
    name = "ironmule-" + secrets.token_hex(10)
    token = secrets.token_hex(24)
    started = time.time()
    pod = {"name": name, "token": token, "cost": a.max_hourly, "disk_rate": disk_rate,
           "reserved": amount, "gpu": a.gpu, "started": started,
           "deadline": int(started + a.minutes * 60), "id": None,
           "balance_before": user["clientBalance"]}
    state["pod"] = pod
    event(state, "reserve", name=name, usd=amount, hourly_quote=quote_rate,
          stock=quote["stockStatus"], balance=user["clientBalance"], deadline=pod["deadline"])
    save(state)
    # Acquire an independent reaper before the first paid mutation. It reconciles
    # an ambiguous create by exact unique name, even if this process dies.
    try:
        launch_watch(name)
    except OSError:
        state["pod"] = None
        event(state, "watch_launch_failed", name=name)
        save(state)
        raise
    start = ("bash -c 'echo $WATCH_B64 | base64 -d > /rp_watch.py && "
             "(python3 /rp_watch.py &) && echo $AGENT_B64 | base64 -d > /agent.py && exec python3 /agent.py'")
    q = """mutation($i: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $i) {
           id costPerHr machine { gpuDisplayName } } }"""
    env = {"AGENT_TOKEN": token, "AGENT_B64": base64.b64encode(AGENT.read_bytes()).decode(),
           "WATCH_B64": base64.b64encode(AGENT.with_name("pod_watchdog.py").read_bytes()).decode(),
           "RP_DEADLINE": str(pod["deadline"])}
    i = {"cloudType": a.cloud, "gpuCount": a.count, "gpuTypeId": a.gpu, "name": name,
         "imageName": a.image, "containerDiskInGb": a.disk, "volumeInGb": 0, "minVcpuCount": a.vcpu,
         "minMemoryInGb": a.mem, "ports": "8000/http", "dockerArgs": start, "startSsh": False,
         "env": [{"key": k, "value": v} for k, v in env.items()]}
    obj = gql(q, {"i": i})["podFindAndDeployOnDemand"]
    pod["id"] = obj["id"]
    save(state)
    pod["cost"] = money(obj["costPerHr"]) + disk_rate
    event(state, "created", name=name, id=pod["id"], hourly=pod["cost"])
    save(state)
    if pod["cost"] > a.max_hourly:
        cleanup(state)
        raise ValueError("provider exceeded quoted maximum, terminated")
    print(json.dumps(public(pod)))


def public(pod):
    return {k: v for k, v in pod.items() if k != "token"}


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


def cleanup(state):
    pod = current(state)
    user = account()
    if not pod["id"]:
        matches = [p for p in user["pods"] if p.get("name") == pod["name"]]
        if len(matches) != 1:
            raise ValueError("create unresolved; reservation retained, no retry")
        pod["id"] = matches[0]["id"]
        pod["cost"] = money(matches[0]["costPerHr"]) + pod["disk_rate"]
        save(state)
    if any(p["id"] == pod["id"] for p in user["pods"]):
        gql("mutation($i: PodTerminateInput!) { podTerminate(input: $i) }", {"i": {"podId": pod["id"]}})
        user = account()
    if any(p["id"] == pod["id"] for p in user["pods"]):
        raise ValueError("termination not yet confirmed")
    hours = max(0, time.time() - pod["started"]) / 3600
    estimate = (hours + CLEANUP_SECONDS / 3600) * money(pod["cost"])
    # Keep the cleanup/billing tail charged even after disappearance. A delayed
    # account update or minimum charge cannot free this reserve prematurely.
    delta = max(0, pod.get("balance_before", user["clientBalance"]) - user["clientBalance"])
    booked = max(estimate, delta)
    state["spent"] += booked
    state["pods"].append({"id": pod["id"], "gpu": pod["gpu"], "hours": hours,
                          "usd": booked, "estimated_with_tail": estimate, "account_delta": delta})
    state["pod"] = None
    event(state, "removed", id=pod["id"], booked=booked, balance=user["clientBalance"])
    save(state)
    print(f"confirmed removal of {pod['id']}, charged including tail {booked:.4f} USD")


def launch_watch(name):
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "watch", name],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)


def watch(name):
    while True:
        with locked(STATE):
            state = load()
            pod = state.get("pod")
            if not pod or pod.get("name") != name:
                return
            if time.time() >= pod["deadline"]:
                try:
                    cleanup(state)
                    return
                except (ValueError, OSError, RuntimeError):
                    # Provider uncertainty must retain the reservation across retries.
                    pass
        time.sleep(10)


def spend(state):
    pod = state.get("pod")
    live = max(0, time.time() - pod["started"]) / 3600 * pod["cost"] if pod else 0
    print(json.dumps({"booked_with_tail": state["spent"], "live_estimate": live,
                      "reserved": held(state), "budget": limit(state),
                      "pod": pod and public(pod), "history": state["pods"]}))


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="op", required=True)
    d = s.add_parser("deploy")
    d.add_argument("--gpu", required=True)
    d.add_argument("--count", type=int, default=1)
    d.add_argument("--cloud", choices=("SECURE", "COMMUNITY"), default="COMMUNITY")
    d.add_argument("--minutes", type=int, default=45)
    d.add_argument("--max-hourly", type=float, default=3.0)
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
    s.add_parser("account")
    s.add_parser("resume")
    s.add_parser("watch").add_argument("name")
    s.add_parser("extend").add_argument("reason")
    for op in ("reserve-external", "settle-external"):
        ext = s.add_parser(op)
        ext.add_argument("name")
        ext.add_argument("usd", type=money)
    a = p.parse_args()
    if a.op == "wait":
        return wait(a)
    if a.op == "watch":
        return watch(a.name)
    if a.op == "account":
        user = account()
        print(json.dumps({"balance": user["clientBalance"], "hourly": user["currentSpendPerHr"],
                          "pods": len(user["pods"])}))
        return
    with locked(STATE):
        state = load()
        if a.op == "stop":
            return cleanup(state)
        if a.op == "spend":
            return spend(state)
        if a.op == "resume":
            pod = state.get("pod")
            if pod:
                if time.time() >= pod["deadline"]:
                    return cleanup(state)
                launch_watch(pod["name"])
            return spend(state)
        if a.op == "deploy":
            return book(a)
        if a.op == "extend":
            if not a.reason.strip():
                raise ValueError("record concrete additional need")
            state["budget"] = MAX_BUDGET
            event(state, "extend", reason=a.reason)
            return save(state)
        if a.op in ("reserve-external", "settle-external"):
            external = state.setdefault("external", {})
            if a.op == "reserve-external":
                if a.name in external:
                    raise ValueError("reservation exists")
                reserve(state, a.usd)
                external[a.name] = a.usd
            else:
                if a.name not in external:
                    raise ValueError("no such reservation")
                del external[a.name]
                state["spent"] += a.usd
            event(state, a.op, name=a.name, usd=a.usd)
            return save(state)
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
