"""Bounded KG5 experiments on one T4. Each stage is a separate invocation.

smoke compiles/runs the standalone numeric check. steps uses cached llama-cpp-python bindings,
GPU libraries and a Qwen3-4B Q4_K_M GGUF. http requires a passing steps report on those inputs
and the project's Python dependencies. Its results compare the KG5 candidate with the same
IronMule build. Nothing is downloaded or rebuilt except the standalone smoke executable.
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHAPE = {"kv_rows": 1024, "streams": 32, "kv_heads": 8}


class Rejected(Exception):
    """A completed experiment failed its acceptance criterion."""


def process(command, deadline, **kwargs):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("stage budget exhausted")
    with subprocess.Popen(command, start_new_session=True, **kwargs) as proc:
        try:
            output, _ = proc.communicate(timeout=remaining)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
        if proc.returncode:
            raise RuntimeError(f"{Path(str(command[0])).name} exited {proc.returncode}")
        return output


def fingerprint(args, deadline):
    paths = [args.gguf, args.lib_dir / "libggml-cuda.so", args.lib_dir / "libllama.so",
             args.bindings / "llama_cpp" / "llama_cpp.py"]
    # Match loader inputs as well as libllama itself. Include actual bytes of every
    # distinct GGML DSO once when the cache includes both SONAME links and real files.
    known = {path.resolve() for path in paths}
    for family in ("libggml.so*", "libggml-base.so*", "libggml-cpu*.so*"):
        for path in sorted(args.lib_dir.glob(family)):
            if path.resolve() not in known:
                paths.append(path)
                known.add(path.resolve())
    for path in sorted((args.bindings / "llama_cpp").rglob("*.py")):
        if path.resolve() not in known:
            paths.append(path)
    if args.preload:
        paths.append(args.preload)
    if args.kg_lib:
        paths.append(args.kg_lib)
    return [line.split()[0] for line in process(["sha256sum", *paths], deadline, stdout=subprocess.PIPE,
                                               text=True).splitlines()]


def evaluate_steps(off, on):
    import numpy as np

    if on["hits"] <= 0 or off["hits"]:
        raise Rejected("KG5 dispatch counter did not distinguish the measured arms")
    # A distributional smoke gate on the same four teacher-forced next-token predictions.
    # Soft-target KL, not perplexity on a corpus. Full factual testing follows only on a win.
    ref = np.load(off["logits"]).astype(np.float64)
    test = np.load(on["logits"]).astype(np.float64)
    if ref.shape != test.shape or ref.ndim != 2 or ref.shape[0] != 4:
        raise Rejected("unexpected sampled-logit dimensions")
    if not np.isfinite(ref).all() or not np.isfinite(test).all():
        raise Rejected("nonfinite sampled logits")

    def log_softmax(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    ref, test = log_softmax(ref), log_softmax(test)
    kl = float((np.exp(ref) * (ref - test)).sum(axis=1).max())
    saving = 1 - statistics.median(on["step_ms"]) / statistics.median(off["step_ms"])
    return {"largest_sample_kl": kl, "step_saving": saving, "passed": kl <= 0.005 and saving >= 0.1 - 1e-12}


def evaluate_http(off, on):
    if off["hits"] or on["hits"] <= 0:
        raise Rejected("KG5 dispatch not confirmed during the candidate HTTP workload")
    if off["fact_hits"] != [0, 0] or len(on["fact_hits"]) != 2 or min(on["fact_hits"]) <= 0:
        raise Rejected("KG5 dispatch not confirmed in both 32-client factual batches")
    if len(off["facts"]) != 50 or off["facts"].keys() != on["facts"].keys():
        raise Rejected("incomplete factual suite")
    regressions = [q for q in off["facts"] if off["facts"][q] and not on["facts"][q]]
    return {"regressions": regressions, "passed": not regressions and
            on["tok_s"] > off["tok_s"] and on["ttft_median_s"] <= off["ttft_median_s"] and
            on["ttft_p95_s"] <= off["ttft_p95_s"]}


def arguments(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=["smoke", "steps", "http"])
    p.add_argument("--output", type=Path, help="default: /kaggle/working/kg5-STAGE.json")
    p.add_argument("--budget-s", type=int, help="wall time ceiling for the whole stage")
    p.add_argument("--gguf", type=Path)
    p.add_argument("--bindings", type=Path)
    p.add_argument("--lib-dir", type=Path)
    p.add_argument("--preload", type=Path, help="optional cached-build KG5 interposer")
    p.add_argument("--kg-lib", type=Path, help="optional pre-tuned KG4 matmul library for both arms")
    p.add_argument("--gate-cache", type=Path, help="seed HTTP states from previously measured ggml-gate JSONs")
    p.add_argument("--steps-report", type=Path)
    args = p.parse_args(argv)
    if args.output is None:
        args.output = Path(f"/kaggle/working/kg5-{args.stage}.json")
    if args.steps_report and args.output.resolve() == args.steps_report.resolve():
        p.error("--output must differ from --steps-report")
    if args.budget_s is None:
        args.budget_s = {"smoke": 90, "steps": 360, "http": 600}[args.stage]
    if args.budget_s < 1:
        p.error("--budget-s must be positive")
    if args.stage != "smoke" and not all([args.gguf, args.lib_dir, args.bindings]):
        p.error("cached --gguf, --lib-dir and --bindings are required")
    if args.stage == "http" and not args.steps_report:
        p.error("--stage http needs --steps-report")
    return args


def run(args, report, tmp, deadline, save):
    gpu = process(["nvidia-smi", "--query-gpu=uuid,name,compute_cap,driver_version",
                   "--format=csv,noheader"], deadline, stdout=subprocess.PIPE, text=True).splitlines()
    if not gpu:
        raise RuntimeError("no GPU visible")
    uuid, name, capability, driver = [field.strip() for field in gpu[0].split(",")]
    if capability != "7.5" or "T4" not in name:
        raise RuntimeError(f"this experiment targets a T4, found {name} ({capability})")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid}
    report.update(gpu=name, uuid=uuid, driver=driver, shape=SHAPE)
    save()
    if args.stage == "smoke":
        binary = tmp / "check"
        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        process([nvcc, "-std=c++17", "-O3", "-arch=sm_75", ROOT / "native/ggml_patches/kg5/check.cu",
                 "-o", binary], deadline, env=env)
        report["check_output"] = process([binary], deadline, env=env, stdout=subprocess.PIPE, text=True)
        return

    worker = ROOT / "lab/kg5/worker.py"
    report["inputs_sha256"] = fingerprint(args, deadline)
    report["kg4"] = bool(args.kg_lib)
    if args.stage == "http":
        payload = args.steps_report.read_bytes()
        prior = json.loads(payload)
        identity = ("inputs_sha256", "gpu", "uuid", "driver", "shape", "code_sha256", "kg4")
        if prior.get("stage") != "steps" or prior.get("status") != "passed" or any(
            prior.get(key) != report[key] for key in identity
        ):
            raise Rejected("steps report does not qualify these inputs for HTTP")
        report["steps_sha256"] = hashlib.sha256(payload).hexdigest()
    env.update(IRONMULE_KG5_SHAPE="1024,32,8", IRONMULE_GGML_BINDINGS=str(args.bindings.resolve()),
               LLAMA_CPP_LIB_PATH=str(args.lib_dir.resolve()),
               LD_LIBRARY_PATH=f"{args.lib_dir.resolve()}:{env.get('LD_LIBRARY_PATH', '')}",
               IRONMULE_KG_LIB=str(args.kg_lib.resolve()) if args.kg_lib else "", IRONMULE_KG="0",
               IRONMULE_GGML_STREAMS="1", IRONMULE_GGML_SEQS="32", PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1")
    if args.preload:
        env["LD_PRELOAD"] = str(args.preload.resolve())
    if args.stage == "http" and args.gate_cache:
        if not args.gate_cache.is_dir():
            raise FileNotFoundError(args.gate_cache)
        env["KG5_BENCH_GATE_CACHE"] = str(args.gate_cache.resolve())
    if args.stage == "steps":
        env["IRONMULE_PILOT"] = "0"  # raw teacher-forced decode never asks the Pilot
    for rep, order in enumerate([("off", "on"), ("on", "off")]):
        pair = {}
        for arm in order:
            out = tmp / f"{arm}{rep}.json"
            python = env.get("KG5_HTTP_PYTHON", sys.executable) if args.stage == "http" else sys.executable
            process([python, worker, args.stage, args.gguf.resolve(), out, str(rep)], deadline,
                    env={**env, "IRONMULE_KG5": str(int(arm == "on"))})
            row = json.loads(out.read_text())
            pair[arm] = row
            report["arms"].append({"rep": rep, "arm": arm, **{k: v for k, v in row.items() if k != "logits"}})
            save()
        verdict = (evaluate_steps if args.stage == "steps" else evaluate_http)(**pair)
        report["pairs"].append(verdict)
        save()
        if not verdict["passed"]:
            raise Rejected(f"pair {rep + 1} below criterion")


def current_code_digest(stage):
    code = hashlib.sha256()
    sources = [Path(__file__), Path(__file__).with_name("worker.py")]
    if stage != "smoke":
        sources.extend(sorted((ROOT / "ironmule").glob("*.py")))
        sources.extend(sorted((ROOT / "ironmule_product").glob("*.py")))
        sources.extend([ROOT / "ironmule_cli.py", ROOT / "lab/row1/facts.py"])
    for path in sources:
        code.update(path.read_bytes())
    return code.hexdigest()


def main(argv=None):
    args = arguments(argv)
    start = time.monotonic()
    report = {"stage": args.stage, "code_sha256": current_code_digest(args.stage), "budget_s": args.budget_s,
              "status": "running", "arms": [], "pairs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["elapsed_s"] = round(time.monotonic() - start, 2)
        temp = args.output.with_suffix(".json.tmp")
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(args.output)

    try:
        with tempfile.TemporaryDirectory(prefix="kg5-") as folder:
            run(args, report, Path(folder), start + args.budget_s, save)
        report["status"] = "passed"
        rc = 0
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        report.update(status="timeout", reason=str(exc))
        rc = 124
    except Rejected as exc:
        report.update(status="rejected", reason=str(exc))
        rc = 2
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        report.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        rc = 1
    save()
    print(f"KG5 {args.stage}: {report['status']} ({report['elapsed_s']}s), {args.output}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
