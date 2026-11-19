"""Start the normal GGUF server on the T4 on which the exact KG5 shape qualified."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from . import bench
else:
    import bench


def read_verdict(steps_data, http_data, identity):
    """Keep the fast path disabled unless both complete, opposite-order experiments passed."""
    for stage, data in (("steps", steps_data), ("http", http_data)):
        report = json.loads(data)
        if (report["stage"] != stage or report["status"] != "passed" or report["shape"] != bench.SHAPE
                or any(report[k] != value for k, value in identity.items())):
            raise ValueError(f"{stage} evidence does not qualify this model, binary and device")
        arms = report["arms"]
        if (len(report["pairs"]) != 2 or not all(pair["passed"] is True for pair in report["pairs"])
                or [(arm["rep"], arm["arm"]) for arm in arms] != [(0, "off"), (0, "on"), (1, "on"), (1, "off")]
                or not all((arm["hits"] > 0 if arm["arm"] == "on" else arm["hits"] == 0) for arm in arms)):
            raise ValueError(f"missing real off/on and on/off passes for {stage}")
        if stage == "http":
            if report["steps_sha256"] != hashlib.sha256(steps_data).hexdigest():
                raise ValueError("HTTP evidence was collected with a different step qualification")
            # Re-evaluate the recorded HTTP arms, including observed dispatch during the factual checks.
            for rep in (0, 1):
                by_arm = {arm["arm"]: arm for arm in arms if arm["rep"] == rep}
                if not bench.evaluate_http(**by_arm)["passed"]:
                    raise ValueError("HTTP throughput, latency or quality regressed")
        elif not all(pair["largest_sample_kl"] <= 0.005 and pair["step_saving"] >= 0.1 - 1e-12
                     for pair in report["pairs"]):
            raise ValueError("step evidence did not meet its numerical or speed threshold")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("steps-report", "http-report", "gguf", "bindings", "lib-dir", "preload", "kg-lib", "state-dir"):
        parser.add_argument("--" + flag, type=Path, required=flag not in ("preload", "kg-lib", "state-dir"))
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    # Bind the actual files (including an optional KG2 interposer or tuned KG4 library)
    # and keep the same physical GPU visible to the normal product server.
    deadline = time.monotonic() + 180
    try:
        steps = args.steps_report.read_bytes()
        http = args.http_report.read_bytes()
        qualified_uuid = json.loads(http)["uuid"]
        devices = bench.process(["nvidia-smi", "--query-gpu=uuid,name,compute_cap,driver_version",
                                 "--format=csv,noheader"], deadline, stdout=subprocess.PIPE, text=True)
        matches = [fields for row in devices.splitlines()
                   if (fields := [v.strip() for v in row.split(",")])[0] == qualified_uuid]
        if len(matches) != 1 or len(matches[0]) != 4:
            raise ValueError("the measured GPU is unavailable")
        uuid, gpu, capability, driver = matches[0]
        if capability != "7.5" or "T4" not in gpu:
            raise ValueError("KG5 qualification is scoped to a T4")
        read_verdict(steps, http, {
            "uuid": uuid, "gpu": gpu, "driver": driver, "code_sha256": bench.current_code_digest("http"),
            "kg4": bool(args.kg_lib), "inputs_sha256": bench.fingerprint(args, deadline),
        })
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, TimeoutError, bench.Rejected) as exc:
        parser.exit(2, f"KG5 qualification refused: {exc}\n")

    lib_dir = str(args.lib_dir.resolve())
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid, "IRONMULE_KG5": "1", "IRONMULE_KG5_SHAPE": "1024,32,8",
           "IRONMULE_GGML_BINDINGS": str(args.bindings.resolve()), "LLAMA_CPP_LIB_PATH": lib_dir,
           "LD_LIBRARY_PATH": lib_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""),
           "IRONMULE_KG_LIB": str(args.kg_lib.resolve()) if args.kg_lib else "",
           "IRONMULE_KG": "0", "IRONMULE_GGML_STREAMS": "1", "IRONMULE_GGML_SEQS": "32",
           "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(bench.ROOT)}
    env["LD_PRELOAD"] = str(args.preload.resolve()) if args.preload else ""
    command = [sys.executable, "-m", "ironmule_cli", "serve", "--gguf", str(args.gguf.resolve()),
               "--port", str(args.port)]
    if args.state_dir:
        command.extend(["--state-dir", str(args.state_dir.resolve())])
    os.chdir(bench.ROOT)
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
