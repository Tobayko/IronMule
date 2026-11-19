"""Run cheap checks first, saving each verdict before spending time on the next stage."""

import json
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = Path("/kaggle/working")


def bounded(argv, deadline, env):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("session budget exhausted")
    with subprocess.Popen(argv, cwd=ROOT, env=env, start_new_session=True) as proc:
        try:
            result = proc.wait(timeout=remaining)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise
    if result:
        raise subprocess.CalledProcessError(result, argv)


def run(job, work=WORK):
    from bench import main as benchmark

    started = time.monotonic()
    deadline = started + job["budget_s"]
    result = {"commit": job["commit"], "through": job["through"], "budget_s": job["budget_s"],
              "status": "running", "stages": {}}
    summary = work / "kg5-session.json"

    def save():
        result["elapsed_s"] = round(time.monotonic() - started, 2)
        tmp = summary.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(summary)

    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]),
           "CUDA_VISIBLE_DEVICES": "0", "MG2_COMMIT": job["commit"]}
    cuda12 = sorted(Path("/usr/local").glob("cuda-12*/bin/nvcc"), reverse=True)
    if cuda12:
        env["PATH"] = os.pathsep.join([str(cuda12[0].parent), env.get("PATH", "")])
    os.environ.update({k: env[k] for k in ("PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "MG2_COMMIT")})
    args = []
    stage = "preflight"
    try:
        save()
        for key in ("gguf", "kg2_archive", "preload", "kg_lib", "bindings", "lib_dir", "llama_source", "wheelhouse", "gate_cache"):
            if key in job and not Path(job[key]).exists():
                raise FileNotFoundError(f"missing attached input {key}: {job[key]}")
        if job["through"] == "http" and "wheelhouse" not in job:
            bounded([sys.executable, "-I", "-c", "import mlx.core, mlx_lm, jinja2"], deadline, env)
        with tempfile.TemporaryDirectory() as temp:
            for stage in ("smoke", "steps", "http"):
                if stage == "steps":
                    libs, bindings = job.get("lib_dir"), job.get("bindings")
                    if "kg2_archive" in job:
                        bounded([sys.executable, __file__, "--extract", job["kg2_archive"], temp], deadline, env)
                        bindings = str(Path(temp) / "llama_cpp_python-0.3.16")
                        libs = str(Path(bindings) / "vendor/llama.cpp/build/bin")
                    args = ["--gguf", job["gguf"], "--lib-dir", libs, "--bindings", bindings]
                    if "kg_lib" in job:
                        args += ["--kg-lib", job["kg_lib"]]
                    preload = job.get("preload")
                    if "llama_source" in job:
                        preload = str(Path(temp) / "kg5-probe.so")
                        bounded([sys.executable, "native/ggml_patches/kg5/build_probe.py", "--source",
                                 job["llama_source"], "--lib-dir", libs, "--output", preload], deadline, env)
                    if preload:
                        args += ["--preload", preload]
                if stage == "http":
                    if "wheelhouse" in job:
                        # Only a qualifying kernel earns time to install the HTTP runtime.
                        venv = Path(temp) / "http-runtime"
                        bounded([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], deadline, env)
                        python = str(venv / "bin" / "python")
                        bounded([python, "-m", "pip", "install", "--no-index", "--only-binary=:all:",
                                 "--disable-pip-version-check", "--find-links", job["wheelhouse"],
                                 "mlx[cpu]>=0.32,<0.33", "mlx-lm>=0.31.3,<0.32"], deadline, env)
                        os.environ["KG5_HTTP_PYTHON"] = python
                    if "gate_cache" in job:
                        args += ["--gate-cache", job["gate_cache"]]
                    args += ["--steps-report", str(work / "kg5-steps.json")]
                report = work / f"kg5-{stage}.json"
                seconds = min({"smoke": 90, "steps": 360, "http": 600}[stage], int(deadline - time.monotonic()))
                if seconds < 1:
                    raise TimeoutError("no budget left for " + stage)
                # Run the stage in this supervisor so only one owner terminates and reaps its workers.
                rc = benchmark(["--stage", stage, "--output", str(report), "--budget-s", str(seconds), *args])
                if rc:
                    raise subprocess.CalledProcessError(rc, stage)
                verdict = json.loads(report.read_text())["status"]
                if verdict != "passed":
                    raise ValueError(f"{stage} reported {verdict}")
                result["stages"][stage] = verdict
                save()
                if stage == job["through"]:
                    break
        result["status"] = "passed"
        return 0
    except (TimeoutError, subprocess.TimeoutExpired):
        result.update(status="timeout", stopped_at=stage)
        return 124
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        code = exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1
        result.update(status={2: "rejected", 124: "timeout"}.get(code, "error"), stopped_at=stage,
                      reason=str(exc))
        return code if code > 0 else 1
    finally:
        save()


if __name__ == "__main__":
    if sys.argv[1] == "--extract":
        with tarfile.open(sys.argv[2]) as archive:
            archive.extractall(sys.argv[3], filter="data")
    else:
        sys.exit(run(json.loads(Path(sys.argv[1]).read_text())))
