"""Run commands in a RunPod pod through RunPod's hosted MCP server (`exec-pod`), and manage the pod.

The session proxy injects the RunPod key for mcp.runpod.io. Pod id comes from $RP_POD.

  python lab/rp/mcp_pod.py exec "nvidia-smi"            # short command (<= 25 s)
  python lab/rp/mcp_pod.py bg NAME "long command"       # nohup in background, log /work/logs/NAME.log
  python lab/rp/mcp_pod.py log NAME [N]                 # tail of that log + whether it still runs
  python lab/rp/mcp_pod.py tool NAME '{"json": "args"}' # any MCP tool (create-pod, delete-pod, get-pod, ...)
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

URL = "https://mcp.runpod.io/"
H = ["-H", "Content-Type: application/json", "-H", "Accept: application/json, text/event-stream"]


def session():
    out = subprocess.run(["curl", "-sS", "-m", "60", "-D", "-", "-X", "POST", URL, *H, "--data-binary", "@-"],
                         input=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                             "protocolVersion": "2025-06-18", "capabilities": {},
                             "clientInfo": {"name": "ironmule", "version": "0"}}}).encode(),
                         capture_output=True).stdout.decode()
    sid = next((line.split(":", 1)[1].strip() for line in out.splitlines() if line.lower().startswith("mcp-session-id")), None)
    subprocess.run(["curl", "-sS", "-m", "30", "-X", "POST", URL, *H, "-H", f"Mcp-Session-Id: {sid}",
                    "-d", '{"jsonrpc":"2.0","method":"notifications/initialized"}'], capture_output=True)
    return sid


def tool(name, arguments, timeout=120):
    sid = session()
    out = subprocess.run(["curl", "-sS", "-m", str(timeout), "-X", "POST", URL, *H, "-H", f"Mcp-Session-Id: {sid}",
                          "--data-binary", "@-"],
                         input=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                           "params": {"name": name, "arguments": arguments}}).encode(),
                         capture_output=True).stdout.decode()
    data = [line[5:].strip() for line in out.splitlines() if line.startswith("data:")]
    reply = json.loads(data[-1] if data else out[out.index("{"):])
    if "error" in reply:
        raise RuntimeError(json.dumps(reply["error"])[:800])
    text = reply["result"]["content"][0]["text"]
    try:
        return json.loads(text), reply["result"].get("isError", False)
    except ValueError:
        return text, reply["result"].get("isError", False)


def execute(command, timeout=25):
    res, err = tool("exec-pod", {"id": os.environ["RP_POD"],
                                 "body": {"command": ["bash", "-lc", command], "timeoutSeconds": timeout}})
    return res, err


def main():
    op = sys.argv[1]
    if op == "exec":
        res, err = execute(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 25)
        if isinstance(res, dict) and "stdout" in res:
            sys.stdout.write(res.get("stdout", ""))
            sys.stdout.write(res.get("stderr", ""))
            print(f"[exit={res.get('exitCode')} timedOut={res.get('timedOut')}]")
        else:
            print(json.dumps(res)[:2000], err)
    elif op == "bg":
        name, cmd = sys.argv[2], sys.argv[3]
        wrapped = (f"mkdir -p /work/logs; nohup bash -c {shlex.quote(cmd + '; echo EXIT=$?')} "
                   f"> /work/logs/{name}.log 2>&1 & echo $! > /work/logs/{name}.pid; echo started {name}")
        res, _ = execute(wrapped, 10)
        print(res.get("stdout") if isinstance(res, dict) else res)
    elif op == "log":
        name, n = sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "30"
        res, _ = execute(f"tail -n {n} /work/logs/{name}.log; kill -0 $(cat /work/logs/{name}.pid) 2>/dev/null "
                         f"&& echo '[running]' || echo '[done]'", 20)
        print(res.get("stdout", "") + res.get("stderr", "") if isinstance(res, dict) else res)
    elif op == "tool":
        res, err = tool(sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})
        print(json.dumps(res, indent=1)[:6000], "ERROR" if err else "")


if __name__ == "__main__":
    main()
