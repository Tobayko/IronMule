"""Run the predeclared pairs after the reference A/A trials; stop at a terminal verdict."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from calibration_grade import grade


def run(root, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        result=grade(root/"out")
        print(json.dumps(result),flush=True)
        if result["status"]!="pending":
            return
        seed=result["next_seed"]
        arms=("reference","candidate") if seed%2 else ("candidate","reference")
        for arm in arms:
            if time.monotonic()+180>end:
                raise TimeoutError("qualification time reservation")
            name=f"{arm}-{seed}"
            cmd=[sys.executable,str(root/"arm.py"),str(root),arm,str(seed),name]
            if seed>=98700:
                cmd.append("--facts-first")
            subprocess.run(cmd,check=True)
    raise TimeoutError("qualification time reservation")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--seconds",type=int,required=True)
    a=p.parse_args()
    run(a.root,a.seconds)
