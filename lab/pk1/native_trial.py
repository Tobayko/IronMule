"""Screen a new W4 CUDA binary and qualify its 32-client engine workload against the locally tuned incumbent."""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import resource
import shutil
import stat
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

FLAGS = {"PS1": "1", "PS1_WARM": "0", "PK6": "1", "DH2": "0", "SV5": "0", "GGML_PACK": "1",
         "GGML_UBATCH": "8192", "CUBLAS_MIN_ROWS": "128", "CUDA_GRAPHS_ROWS": "32"}
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
SCHEMA = "ironmule.w4_target_trial.v1"


def checksum(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arm(args):
    from serving_trial import digest, run_wave

    from ironmule import ggml_engine as ge
    from ironmule import gguf_tokenizer, kernel_gen, pilot, w4_kernels
    from ironmule.factual_gate import MAX_TOKENS, cases, correct
    from ironmule.ggml_binary import identity
    from ironmule.gguf_tokenizer import GgufTokenizer
    from ironmule.kernel_gen import fingerprint
    from ironmule_devices import ggml_devices
    from ironmule_product.compare import performance_prompts
    from lab.portable_gpu.bench import save

    start = time.monotonic()
    if any(os.environ.get("IRONMULE_" + k) != v for k, v in FLAGS.items()):
        raise ValueError("wrong engine reference settings")
    lib, table = (Path(os.environ[k]) for k in ("IRONMULE_KG_LIB", "IRONMULE_W4_TABLE"))
    perf = [(TEMPLATE.format(prompt), "") for prompt in performance_prompts(args.seed, 32, 32)]
    with closing(ge.GgmlEngine(str(args.gguf), seqs=32, seq_ctx=2048, flash_attn=True, speculate=True)) as eng:
        mapped = Path("/proc/self/maps").read_text().splitlines()
        executable = any(len(fields := line.split(maxsplit=5)) == 6 and "x" in fields[1]
                         and Path(fields[5]).resolve() == lib.resolve() for line in mapped)
        if not executable or 32 not in eng.kg.get("tokens_served", []) or eng.pilot is None or eng.gpu_layers != 999:
            raise RuntimeError("full GPU configuration, active Pilot and width-32 W4 dispatch required")
        def used_memory():
            devices = ggml_devices(eng.L)
            if len(devices) != 1 or devices[0]["provider"] != "cuda":
                raise ValueError("exactly one CUDA device required")
            return devices[0]["total_memory"] - devices[0]["free_memory"]

        memory = [used_memory()]
        with closing(GgufTokenizer(str(args.gguf), eng.L)) as tok:
            warm = [(TEMPLATE.format(prompt), "") for prompt in performance_prompts(args.seed ^ 317, 32, 32)]
            run_wave(eng, tok, warm, 32, {})
            memory.append(used_memory())
            begin = eng.pilot.decisions
            result = run_wave(eng, tok, perf, 32, {})
            decisions = eng.pilot.decisions - begin
            memory.append(used_memory())
            latency = sorted(result["ttfts_s"])
            questions = cases(args.seed)
            factual = run_wave(eng, tok, [(TEMPLATE.format(q.prompt), "") for q in questions], MAX_TOKENS, {})
            memory.append(used_memory())
            passed = [correct(case, row["answer"]) for case, row in zip(questions, factual["answers"], strict=True)]
        loaded = identity()
        if not loaded.get("sha256") or not decisions:
            raise ValueError("missing runtime identity or Pilot decisions")
        hardware = fingerprint()
        # The .cu variation is the only intended treatment; separately bind both loaded artifacts.
        runtime_sources = {m.__name__.replace(".", "/") + ".py": checksum(m.__file__)
                           for m in (ge, gguf_tokenizer, pilot, kernel_gen, w4_kernels)}
        binding = {"fp": hardware, "model_sha256": checksum(args.gguf), "native": loaded, "runtime_sources": runtime_sources,
                   "pilot": checksum(eng.pilot.weights_path), "engine": eng.settings, "flags": FLAGS,
                   "affinity": sorted(os.sched_getaffinity(0)), "threads": eng.threads}
    if result["completed"] != 32 or not latency:
        raise ValueError("incomplete fixed workload")
    def deliveries(wave):
        return [{k: a[k] for k in ("output_ids", "prompt_digest", "answer", "stop")} for a in wave["answers"]]

    answers, factual_answers = deliveries(result), deliveries(factual)
    save(args.output, {"schema": SCHEMA, "profiled": False, "binding": binding, "workload": {"seed": args.seed,
                       "clients": 32, "tokens": 32, "context": 2048, "warmup_waves": 1, "quality_cases": len(passed)},
                       "treatment": {"binary": checksum(lib), "table": checksum(table)},
                       "evaluator_sha256": checksum(__file__), "pilot_decisions": decisions,
                       "wall_s": result["wall_s"], "ttft_s": latency[30], "completed": result["completed"],
                       "tokens": result["tokens"], "output_sha256": digest(answers),
                       "facts": passed, "quality_answers": factual_answers, "answers": answers,
                       "fact_digest": digest([r["output_ids"] for r in factual["answers"]]),
                       "device_used_bytes_after_phases": memory,
                       "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       "total_elapsed_s": time.monotonic() - start})


def build(args):
    from ironmule import w4_kernels as wk
    from ironmule.kernel_gen import compile_cuda, compute_capability

    arch = compute_capability()
    if arch is None or int(arch) < 80:
        raise ValueError("W4 requires an eligible CUDA device")
    compile_cuda(args.candidate_source.read_text(), args.output, arch, include=wk.SOURCE.parent)


def screen(args):
    """Test the actual model's matrices and the local table against float64 and the incumbent binary."""
    import numpy as np

    from ironmule import w4_kernels as wk
    from ironmule.kernel_gen import q4k_tensors
    from lab.portable_gpu.bench import save

    ref = wk._lib(args.reference_lib)
    candidate = wk._lib(args.candidate_lib)
    if wk.variants(ref) != wk.variants(candidate):
        raise ValueError("this evaluator requires a separately compiled candidate preserving the dispatch ABI")
    measurements = []
    table = [list(map(int, line.split())) for line in args.table.read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    for kind, code, block in (("Q4_K", 12, 144), ("Q6_K", 14, 210)):
        tensors = q4k_tensors(str(args.gguf), kind=kind)
        for ty, n, k, batch, variant, split in table:
            if ty != code or batch != 32:
                continue
            raw = np.ascontiguousarray(np.asarray(tensors[(k, n)]["data"])).reshape(-1)
            rows = min(n, 512)
            weights = wk.dequant(raw[:rows * (k // 256) * block], kind, rows, k)
            x = np.random.default_rng(572).standard_normal((batch, k)).astype(np.float32)
            expect = x.astype(np.float64) @ weights.T
            outputs, timings = [], []
            for lib in (ref, candidate):
                y = np.full((batch, n), np.nan, dtype=np.float32)
                us = ctypes.c_double()
                rc = lib.im_w4_bench(variant, raw.ctypes.data, raw.nbytes, n, k, x.ctypes.data, batch,
                                     split, max(1, min(8, 2_000_000_000 // raw.nbytes)), 30,
                                     ctypes.byref(us), y.ctypes.data)
                error = float(np.linalg.norm(y[:, :rows] - expect) / max(np.linalg.norm(expect), 1e-30))
                if rc or not np.isfinite(y).all() or not error <= wk.ERROR_BOUND:
                    raise ValueError(f"failed float64 check on {(kind, n, k, batch)}")
                outputs.append(y)
                timings.append({"microseconds": us.value, "relative_error": error})
            if not np.array_equal(*outputs):
                raise ValueError(f"candidate changed tensor values on {(kind, n, k, batch)}")
            measurements.append({"shape": [code, n, k, batch], "variant": variant, "ksplit": split, "arms": timings})
    if not measurements:
        raise ValueError("no width-32 matrices checked")
    save(args.output, {"schema": SCHEMA, "stage": "screen", "measurements": measurements,
                       "all_elements_identical": True, "model": checksum(args.gguf),
                       "reference": checksum(args.reference_lib), "candidate": checksum(args.candidate_lib),
                       "table": checksum(args.table), "evaluator_sha256": checksum(__file__)})


def validate_screen(row, binding, candidate_hash, table):
    dispatch = [tuple(map(int, line.split())) for line in table.read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    expected = sorted(r for r in dispatch if r[3] == 32)
    observed = sorted((*m["shape"], m["variant"], m["ksplit"]) for m in row["measurements"])
    if (row.get("schema") != SCHEMA or row.get("stage") != "screen" or row.get("all_elements_identical") is not True
            or row.get("evaluator_sha256") != checksum(__file__) or row.get("model") != binding["gguf"]
            or row.get("reference") != binding["w4_sha256"] or row.get("candidate") != candidate_hash
            or row.get("table") != binding["table_sha256"] or not expected or observed != expected):
        raise ValueError("screen receipt does not match the active width-32 dispatch table and binaries")


def author(args):
    from training.code_author import hosted_proposal, write_once
    from training.experiment_planner import _historical_notes

    order = json.loads(args.order.read_text())
    target = order["target"]
    if order["evaluator"] != SCHEMA or target["path"] != "native/kernels/w4mma.cu":
        raise ValueError("unsupported native order")
    source = args.checkout / target["path"]
    if checksum(source) != order["binding"]["source_sha256"]:
        raise ValueError("attributed source differs from checkout")
    text = source.read_text()
    lines = text.splitlines(keepends=True)
    start = target["line"] - 1
    if start < 0 or start >= len(lines) or target["function"] + "(" not in lines[start]:
        raise ValueError("source location differs from device attribution")
    end = next(i + 1 for i in range(start + 1, len(lines)) if lines[i].rstrip() == "}")
    signature = "".join(lines[start:end]).split("{", 1)[0].strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reply = hosted_proposal(args.output.parent, order, text, knowledge=_historical_notes(target, None))
    updated = reply["definition"].strip()
    if updated.split("{", 1)[0].strip() != signature or not updated.endswith("}"):
        raise ValueError("only the attributed kernel definition and its existing signature are accepted")
    if any(line.startswith("}") for line in updated.splitlines()[:-1]):
        raise ValueError("response contains code outside the measured kernel")
    candidate = "".join(lines[:start]) + updated + "\n" + "".join(lines[end:])
    if candidate == text:
        raise ValueError("worker proposed keeping the existing kernel")
    if args.output.exists() and args.output.read_text() != candidate:
        raise ValueError("conflicting candidate for an existing order")
    args.output.write_text(candidate)
    write_once(args.output.parent / "authorship.json", {"order": order["order_id"],
               "candidate_sha256": checksum(args.output), "rationale": reply["rationale"]})


def validate_arm(row, binding, treatment, seed):
    if (row["schema"] != SCHEMA or row["evaluator_sha256"] != checksum(__file__)
            or row["treatment"] != treatment or row["binding"]["native"] != binding["native"]
            or row["binding"]["fp"]["devices"] != binding["fp"]["devices"]
            or any(row["binding"]["runtime_sources"].get(name) != value for name, value in binding["serving_sources"].items())
            or row["binding"]["model_sha256"] != binding["gguf"] or row["completed"] != 32
            or row["workload"] != {"seed": seed, "clients": 32, "tokens": 32, "context": 2048,
                                   "warmup_waves": 1, "quality_cases": 18}
            or row["pilot_decisions"] < 1 or row["facts"] != [True] * 18):
        raise ValueError("qualification lost hardware binding, quality, or expected work")


def verify_outputs(row, seed):
    from ironmule.factual_gate import cases, correct
    from training.serving_trial import digest

    answers = row["answers"]
    facts = row["quality_answers"]
    if (len(answers) != 32 or len(facts) != 18 or row["output_sha256"] != digest(answers)
            or sum(len(a["output_ids"]) for a in answers) != row["tokens"]
            or any(not a["output_ids"] or a["stop"] not in ("length", "eos") for a in [*answers, *facts])
            or row["fact_digest"] != digest([a["output_ids"] for a in facts])
            or not all(correct(q, a["answer"]) for q, a in zip(cases(seed), facts, strict=True))):
        raise ValueError("returned work or independently graded factual answers failed")


def copy_result(product, output, uid):
    """Import a single bounded regular file from the completed user process."""
    fd = os.open(product, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as src:
        info = os.fstat(fd)
        limit = (512 if output.suffix == ".so" else 2) * (1 << 20)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != uid
                or not 0 < info.st_size <= limit):
            raise ValueError("worker returned an invalid output artifact")
        with NamedTemporaryFile(dir=output.parent, delete=False) as dest:
            temp = Path(dest.name)
            try:
                remaining = info.st_size
                while remaining:
                    chunk = src.read(min(remaining, 1 << 20))
                    if not chunk:
                        raise ValueError("truncated worker output")
                    dest.write(chunk)
                    remaining -= len(chunk)
                if src.read(1):
                    raise ValueError("worker output changed during collection")
                dest.flush()
                os.fsync(dest.fileno())
                os.replace(temp, output)
            finally:
                temp.unlink(missing_ok=True)


def restricted_launch(args, output, arguments, mounts, env):
    from lab.rp.restricted_worker import run
    from training.code_trial import runtime_env

    staging = args.worker_root.resolve(strict=True)
    with TemporaryDirectory(dir=staging) as scratch:
        stage = Path(scratch)
        stage.chmod(0o755)
        inputs = stage / "inputs"
        inputs.mkdir(mode=0o755)
        paths = {"/trusted/native_trial.py": str(inputs / "native_trial.py"),
                 "/models/" + args.gguf.name: str(args.gguf.resolve())}
        for script in [Path(__file__).resolve(), Path(__file__).resolve().parents[2] / "training/serving_trial.py"]:
            shutil.copyfile(script, inputs / script.name)
            (inputs / script.name).chmod(0o444)
        for source, logical in mounts:
            name = Path(logical).name
            path = inputs / name
            shutil.copyfile(source, path)
            path.chmod(0o444)
            paths[logical] = str(path)
        writable = stage / "out"
        writable.mkdir(mode=0o700)
        os.chown(writable, args.worker_uid, args.worker_uid)
        paths["/output/" + output.name] = str(writable / output.name)
        command = [str(Path(sys.executable).resolve()), *(paths.get(arg, arg) for arg in arguments)]
        defaults = {k: v for k, v in runtime_env("cuda").items() if k not in ("HOME", "TMPDIR")}
        environment = {**defaults, **{k: paths.get(v, v) for k, v in env.items()},
                       "PYTHONPATH": str(args.checkout.resolve())}
        readonly = [inputs, args.checkout.resolve(), args.gguf.resolve(),
                    *(Path(environment[k]).resolve() for k in ("IRONMULE_GGML_BINDINGS", "LLAMA_CPP_LIB_PATH"))]
        runtime = [Path(sys.prefix), Path(sys.base_prefix),
                   *(Path(entry) for entry in environment.get("LD_LIBRARY_PATH", "").split(":") if entry)]
        for entry in runtime:
            real = entry.resolve(strict=True)
            if not real.is_relative_to("/usr") and real not in readonly:
                readonly.append(real)
        proc = run(command, env=environment, cwd=writable, uid=args.worker_uid, gid=args.worker_uid,
                   output=writable, readonly=readonly, timeout=args.timeout)
        if proc.returncode:
            raise RuntimeError(f"restricted worker exited {proc.returncode}: {(proc.stdout or '')[-500:]}")
        copy_result(writable / output.name, output, args.worker_uid)


def launch(args, output, arguments, mounts, env):
    from training.code_trial import sandbox

    if args.worker_root is not None:
        return restricted_launch(args, output, arguments, mounts, env)
    with TemporaryDirectory(dir=output.parent) as tmp:
        cmd = sandbox(args.checkout.resolve(), args.gguf.resolve(), Path(tmp), "cuda")
        prefix = cmd[:-1]
        for path, dest in [(Path(__file__).resolve(), "/trusted/native_trial.py"), *mounts]:
            prefix += ["--ro-bind", str(path.resolve()), dest]
        for key, value in env.items():
            prefix += ["--setenv", key, value]
        subprocess.run([*prefix, cmd[-1], *arguments], timeout=args.timeout, check=True)
        product = Path(tmp) / output.name
        os.replace(product, output)



def launch_arm(args, binary, seed, out):
    """Run the fixed benchmark once in a fresh, constrained process. Caller binds and checks the evidence."""
    logical_table = "/kernel/dispatch.table"
    logical_lib = "/kernel/active.so"
    env = {"IRONMULE_" + k: v for k, v in FLAGS.items()}
    launch(args, out, ["/trusted/native_trial.py", "--gguf", "/models/" + args.gguf.name, "--mode", "arm",
                       "--seed", str(seed), "--output", "/output/" + out.name],
           [(binary, logical_lib), (args.table, logical_table)],
           {**env, "IRONMULE_KG_LIB": logical_lib, "IRONMULE_W4_TABLE": logical_table})
    return json.loads(out.read_text())


def judge_native(pairs):
    from training.code_trial import judge

    if any(ref["fact_digest"] != cand["fact_digest"] for ref, cand in pairs):
        return {"passed": False, "reason": "factual answers differ between arms"}
    scored = []
    for pair in pairs:
        for row in pair:
            if len(row["facts"]) != 18 or not all(row["facts"]):
                return {"passed": False, "reason": "factual grade failed"}
            verify_outputs(row, row["workload"]["seed"])
        # All 18 facts were graded above. judge checks complete equivalence of the 32 timed requests.
        scored.append(tuple({**row, "correct": 32} for row in pair))
    result = judge(scored)
    device_ratios = [max(c["device_used_bytes_after_phases"]) / max(r["device_used_bytes_after_phases"])
                     for r, c in pairs]
    result["gpu_memory_ratios_at_phase_boundaries"] = device_ratios
    if max(device_ratios) > 1.05:
        result.update(passed=False, reason="GPU memory at phase boundaries exceeded the reference by over 5%")
    return result


def qualify(args):
    from lab.portable_gpu.bench import save
    from training.code_trial import PAIRS

    order = json.loads(args.order.read_text())
    binding = order["binding"]
    order_id = hashlib.sha256(json.dumps({k: v for k, v in order.items() if k != "order_id"}, sort_keys=True).encode()).hexdigest()
    if (order["profile_sha256"] != checksum(args.order.with_suffix(".profile.json"))
            or order["order_id"] != order_id or order["evaluator"] != SCHEMA or order["workload"]["decode_rows"] != 32
            or checksum(args.gguf) != binding["gguf"] or checksum(args.reference_lib) != binding["w4_sha256"]
            or checksum(args.table) != binding["table_sha256"]
            or checksum(args.checkout / "native/kernels/w4mma.cu") != binding["source_sha256"]
            or checksum(args.candidate_source) == binding["source_sha256"]):
        raise ValueError("artifacts differ from the attributed reference, or candidate unchanged")
    state = args.output.parent
    state.mkdir(parents=True, exist_ok=True)
    context = {"order": order["order_id"], "base_commit": order["base_commit"],
               "order_sha256": checksum(args.order), "profile_sha256": order["profile_sha256"], "candidate_source": checksum(args.candidate_source),
               "harness": checksum(__file__), "reference": checksum(args.reference_lib), "table": checksum(args.table)}

    if (state / "context.json").exists():
        if json.loads((state / "context.json").read_text()) != context:
            raise ValueError("different order in saved experiment directory")
    else:
        save(state / "context.json", context)
    cand = state / "candidate.so"
    base = ["/trusted/native_trial.py", "--gguf", "/models/" + args.gguf.name]
    started = time.monotonic()
    environment = {"IRONMULE_" + k: v for k, v in FLAGS.items()}
    # Keep the original build, table and operating flags. Compile a different library only for this isolated worker.
    try:
        build_record = state / "build.json"
        if not build_record.exists():
            launch(args, cand, [*base, "--mode", "build", "--candidate-source", "/kernel/proposal.cu",
                          "--output", "/output/candidate.so"],
                   [(args.candidate_source, "/kernel/proposal.cu")],
                   {"PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"})
            save(build_record, {"source": context["candidate_source"], "binary": checksum(cand)})
        if json.loads(build_record.read_text()) != {"source": context["candidate_source"], "binary": checksum(cand)}:
            raise ValueError("saved candidate binary no longer matches its build")
        tab = (args.table, "/kernel/dispatch.table")
        screening = state / "screen.json"
        if not screening.exists():
            launch(args, screening, [*base, "--mode", "screen", "--output", "/output/screen.json",
                               "--reference-lib", "/kernel/reference.so", "--candidate-lib", "/kernel/candidate.so",
                               "--table", "/kernel/dispatch.table"],
                   [(args.reference_lib, "/kernel/reference.so"), (cand, "/kernel/candidate.so"), tab], environment)
        validate_screen(json.loads(screening.read_text()), binding, checksum(cand), args.table)
        pairs = []
        for i in range(PAIRS):
            rows = {}
            for label in (("ref", "cand") if i % 2 == 0 else ("cand", "ref")):
                binary = args.reference_lib if label == "ref" else cand
                out = state / f"{label}-{i}.json"
                if not out.exists():
                    launch_arm(args, binary, 92741 + i, out)
                row = json.loads(out.read_text())
                validate_arm(row, binding, {"binary": checksum(binary), "table": context["table"]}, 92741 + i)
                rows[label] = row
            pairs.append((rows["ref"], rows["cand"]))
        result = judge_native(pairs)
    except Exception as exc:
        save(args.output, {"schema": SCHEMA, "context": context, "passed": False, "reason": str(exc)[:600],
                           "incumbent": context["reference"], "elapsed_s": time.monotonic() - started})
        raise
    receipts = {p.name: checksum(p) for p in [state / "build.json", state / "screen.json",
                *(state / f"{label}-{i}.json" for i in range(PAIRS) for label in ("ref", "cand"))]}
    save(args.output, {"schema": SCHEMA, "context": context, **result, "measurements_sha256": receipts,
                       "candidate": checksum(cand),
                       "incumbent": context["reference"], "elapsed_s": time.monotonic() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("arm", "screen", "qualify", "author", "build"), default="qualify")
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", type=Path)
    parser.add_argument("--reference-lib", type=Path)
    parser.add_argument("--candidate-lib", type=Path)
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--worker-root", type=Path, help="existing root-owned worker staging directory for UID isolation")
    parser.add_argument("--worker-uid", type=int, default=60000)
    args = parser.parse_args()
    if args.mode == "qualify":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with (args.output.parent / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            qualify(args)
    else:
        {"arm": arm, "screen": screen, "author": author, "build": build}[args.mode](args)


if __name__ == "__main__":
    main()
