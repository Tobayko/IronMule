"""Apply the ordered serving fixture to locally calibrated PK2 on the new A100."""
import json
import os
from pathlib import Path
import sys
import time

import native_trial
import restart_arm


def main():
    # The fixed-128 v2 fixture uses exactly the same engine settings and workload.
    # This lane requires a full local threshold decision on every worker.
    if os.environ.get("IRONMULE_CUBLAS_MIN_ROWS") is not None:
        raise ValueError("a fixed threshold suppresses the local experiment")
    native_trial.FLAGS = {k:v for k,v in native_trial.FLAGS.items() if k != "CUBLAS_MIN_ROWS"}
    out = Path(sys.argv[sys.argv.index("--output")+1])
    broker = float(os.environ["IRONMULE_TRIAL_START_MONOTONIC"])
    born = int(Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19]) / os.sysconf("SC_CLK_TCK")
    if not -.02 <= born-broker < 60:
        raise ValueError("worker launch time differs")
    restart_arm.main()
    row = json.loads(out.read_bytes())
    trace = Path(os.environ["IRONMULE_GGML_TRACE"]).read_text().splitlines()
    a = max(i for i,l in enumerate(trace) if l.startswith("# init measure_pk2 begin "))
    b = next(i for i in range(a+1,len(trace)) if trace[i].startswith("# init "))
    decision = json.loads(next(l for l in trace[b:] if l.startswith("# init result ")).removeprefix("# init result "))
    row.update(pk2=decision["pk2"], pk2_calls=sum(not l.startswith("#") and bool(l) for l in trace[a+1:b]),
               broker_to_process_s=max(0,born-broker),broker_elapsed_s=time.monotonic()-broker)
    out.write_text(json.dumps(row,allow_nan=False))


if __name__ == "__main__":
    main()
