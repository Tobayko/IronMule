"""Run the stock quality gate outside serving and bind its result to the ordered profile."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from lab.pk6 import ordered_grade as grade
from training.trial_release import require


def validate(observed, settings):
    gate = observed["gate"]
    require(observed["device_sha256"] == grade.BOARD and observed["source_sha256"] == grade.CANDIDATE,
            "quality measurement belongs to a different device or source")
    require(gate.get("ran") and gate.get("passes"), "native numerical qualification failed")
    options = [*gate["tried"], gate.get("streams"), gate.get("shared")]
    for option in options:
        if not option or option.get("settings") != settings or not option.get("passes"):
            continue
        facts = option.get("factual", {})
        if (option["chunks"] >= 16 and option["interval"][1] <= min(gate["bound"], 1.005)
                and facts.get("passes") and all(facts.get(key) == 18 for key in
                  ("total", "reference_correct", "repeat_correct", "candidate_correct"))):
            return {"settings": settings, "chunks": option["chunks"], "interval": option["interval"],
                    "ratio": option["ratio"], "correct": facts["candidate_correct"], "seconds": observed["time_s"],
                    "evidence_sha256": grade.canonical(observed)}
    raise ValueError("the deployed profile lacks its 16-chunk and factual quality receipt")


def ensure(root, settings):
    # Both the independent collector and the deployment process serialize their own GPU work here.
    cache = root / "numeric"
    cache.mkdir(mode=0o700, exist_ok=True)
    with (root / "numeric.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = cache / "observed.json"
        if not result.exists():
            os.chown(cache, 60000, 60000)
            tree = root / "candidate"
            bindings = root / "bindings"
            native = bindings / "vendor/llama.cpp/build/bin"
            kernel, = (root / "stock/kernels/w4").glob("*/kernel.so")
            env = {"PATH": os.environ["PATH"], "HOME": str(cache), "PYTHONPATH": str(tree),
                   "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                   "LD_LIBRARY_PATH": str(native) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
                   "IRONMULE_TRIAL_LOCAL_CALIBRATION": "1", "IRONMULE_GGML_BINDINGS": str(bindings),
                   "LLAMA_CPP_LIB_PATH": str(native), "IRONMULE_GGML_GPU_LAYERS": "999",
                   "IRONMULE_KG_LIB": str(kernel), "IRONMULE_W4_TABLE": str(kernel.with_name("dispatch.table")),
                   "IRONMULE_PS1": "1", "IRONMULE_PS1_WARM": "0", "IRONMULE_PS1_CONCAT": "0",
                   "IRONMULE_PK6": "1", "IRONMULE_GGML_PACK": "1", "IRONMULE_GGML_UBATCH": "8192"}
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
            worker = Path(__file__).with_name("ordered_numeric.py")
            subprocess.run([sys.executable, str(worker), str(root), str(cache)], cwd=tree, env=env, user=60000,
                           group=60000, extra_groups=[], check=True, timeout=1200)
        return validate(json.loads(result.read_bytes()), settings)
