"""Run each prospectively assigned worker once and recompute the gate between pairs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from ordered_grade import FACTS, grade


def run(root, limit):
    deadline = time.monotonic()+limit
    while True:
        result = grade(root/"ordered")
        print(json.dumps(result),flush=True)
        if result["status"] != "pending":
            return
        seed = result["next_seed"]
        first_reference = (seed % 2 == 1) != (seed in FACTS)
        for arm in (("reference","candidate") if first_reference else ("candidate","reference")):
            if time.monotonic()+180 > deadline:
                raise TimeoutError("ordered campaign reservation")
            args=[sys.executable,str(root/"ordered_arm.py"),str(root),arm,str(seed),f"{arm}-{seed}"]
            if seed in FACTS:
                args.append("--facts-first")
            subprocess.run(args,check=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--seconds",type=int,required=True)
    a=p.parse_args()
    run(a.root,a.seconds)
