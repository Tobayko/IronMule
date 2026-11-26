"""Independent deadline enforcer. Runs on the pod with RunPod's pod-scoped key."""
import json
import os
import time
import urllib.error
import urllib.request

deadline = int(os.environ["RP_DEADLINE"])
while time.time() < deadline:
    time.sleep(min(30, max(0, deadline - time.time())))

while True:
    try:
        query = "mutation($i:PodTerminateInput!){podTerminate(input:$i)}"
        body = {"query": query, "variables": {"i": {"podId": os.environ["RUNPOD_POD_ID"]}}}
        request = urllib.request.Request(
            "https://api.runpod.io/graphql", data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            json.load(response)
    except (KeyError, OSError, ValueError):
        pass
    time.sleep(10)
