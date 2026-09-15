"""PORT1: rerun one `ironmule tune` confirmation child exactly as ab.run starts it, with stderr.

Usage: python probe_ab.py MODEL_ID  (run 5: the 4B child exited 1 and ab.run keeps no stderr)
"""
import json
import os
import subprocess
import sys

from ironmule import ab
from ironmule.runtime import BASELINE, Knobs

spec = {"arms": {"baseline": BASELINE.as_dict(), "candidate": Knobs(head_skip_prefill=True).as_dict()},
        "repeats": 1, "warmup": 0, "max_tokens": 8, "model": sys.argv[1], "order": ["baseline", "candidate"]}
here = os.path.dirname(os.path.abspath(ab.__file__))
proc = subprocess.run([sys.executable, "-c", ab.CHILD_BOOTSTRAP, json.dumps(spec),
                       os.path.join(here, "q3f_child_guard.py"), os.path.abspath(ab.__file__)],
                      capture_output=True, text=True, cwd=os.path.dirname(here), env={**os.environ, **ab.CHILD_ENV})
print("exit", proc.returncode, {k: os.environ.get(k) for k in ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")})
print("stdout", proc.stdout[-600:])
print("stderr", proc.stderr[-3000:])
