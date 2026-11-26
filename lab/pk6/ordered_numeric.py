"""Fresh stock 16-chunk and QG1 qualification of the exact authored source on the bound board."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("cache",type=Path)
    a=p.parse_args()
    from ironmule import ggml_engine as engine
    expected="f60951799571833dfa663227751a41d9a36a88b54405518240df762cdbf340f7"
    if hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest()!=expected or os.environ.get("IRONMULE_TRIAL_LOCAL_CALIBRATION")!="1":
        raise ValueError("different source or inactive treatment")
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    board=hashlib.sha256(gpu.encode()).hexdigest()
    if board!="112d8a87ea0f7aab9de32ad3b2f2d65318c3bce1044ee30b75f78c528c923e2b":
        raise ValueError("different physical calibration target")
    model="/models/Qwen3-32B-Q4_K_M.gguf"
    start=time.monotonic()
    verdict=engine.gate(model,model,cache=a.cache,root=a.root/"candidate")
    result={"time_s":time.monotonic()-start,"device_sha256":board,"source_sha256":expected,"gate":verdict}
    target=a.cache/"observed.json"
    temp=target.with_suffix(".incoming")
    temp.write_text(json.dumps(result,sort_keys=True,allow_nan=False))
    temp.replace(target)


if __name__=="__main__":
    main()
