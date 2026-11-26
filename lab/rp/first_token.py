"""Copy the first-token trial off an already reserved live pod, including incomplete waves."""
import argparse
from pathlib import Path
import time

from lab.pk6.first_token.trial import ARMS
from lab.rp import native_campaign, warm_repeat


def collect(pod_id, state):
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    files = ["target.json", "completed.json", "device-load.csv"]
    for arm in ARMS:
        files += [f"{p}-{arm}.json" for p in ("start", "ready", "end")]
        files += [f"http_{arm}_32.jsonl", f"http_{arm}q_8.jsonl", f"timeline-{arm}.jsonl",
                  f"trace-{arm}.txt", f"serve-{arm}.log"]
    while True:
        pod = native_campaign.lease(pod_id)
        job = warm_repeat.status(pod, "first-token-trial")
        warm_repeat.collect(pod, state, target="/work/first-token", files=files)
        if job.get("rc") is not None and not job.get("running"):
            if job["rc"] or not (state / "completed.json").exists():
                raise RuntimeError("target trial ended before all four arms completed; partial data copied")
            return
        time.sleep(8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--state", required=True, type=Path)
    args = parser.parse_args()
    collect(args.pod_id, args.state)
