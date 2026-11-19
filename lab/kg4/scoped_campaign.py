"""Run stock, narrowly scoped KG4 and closing stock in a frozen offline capsule on one L40S."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

MODEL = "efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689"
MODEL_BYTES = 19762149024
R1 = "da0ff490ce4336e9b684166cfbe6486f7a9058ee68832114a9543d0b75c219cf"
PINNED = {
    "manifest.json": "4ff1a2150e6fdcfe41723287bd0dbc38e6f385d30004b02c27c452d22882df77",
    "r1.so": R1,
    "product/SHA256SUMS": "779986b71c720feefe810db2128403b4711d0f1febf55909c7dba21f2074e69b",
    "runtime/SHA256SUMS": "0bef56c59addefb852c38a310523b068f84d4779dc1fea4178ee0c4d5d142dde",
    "runtime/lib/libggml-cuda.so": "bbc59e06670907330a9e39b111451560f7318153fdd0ce45d3d7d7afc9ffc816",
    "native.py": "44152f646eb0ecc6554e3fedccf3985f5bae3d3e67353e908220f48d0978d19f",
    "http_kernel.py": "17dadc6038583698cc02a2977fb15a839321cf46701e67a3cac0dce7484aa293",
    "select_layout.py": "7303aeb791ec532836ca08b262a4b61497a5bca253da4189f647d98b59473fbb",
    "run.py": "f738f4c8191c525f4ce7c25a46499638e6c79022028bbf3ea6b078d9861683e1",
}
SEED = 1178421
PUBLIC = {"report.json": "kg4-status.json", "native-qualification.json": "kg4-native.json",
          "scoped-http.json": "kg4-http.json",
          **{f"native-{arm}.json": f"kg4-native-{arm}.json" for arm in ("stock", "scoped", "stock-closing")}}


def sha(path):
    with path.open("rb") as handle:
        hashed = hashlib.sha256()
        for chunk in iter(lambda: handle.read(2**20), b""):
            hashed.update(chunk)
        return hashed.hexdigest()


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def prepare(base, product, model, source):
    if model.stat().st_size != MODEL_BYTES or sha(model) != MODEL:
        raise ValueError("unexpected GGUF")
    for path, digest in PINNED.items():
        if sha(base / path) != digest:
            raise ValueError(f"base artifact mismatch {path}")
    for folder in ("runtime", "product"):
        subprocess.run(["sha256sum", "--quiet", "-c", "SHA256SUMS"], cwd=base / folder, check=True)
    # Worker children launch in isolated Python mode. Install this snapshot into a new interpreter.
    product.mkdir()
    (product / "wheels").symlink_to(base / "product/wheels", target_is_directory=True)
    (product / "bindings").symlink_to(base / "runtime/bindings", target_is_directory=True)
    (product / "src").symlink_to(source, target_is_directory=True)
    shutil.copyfile(base / "product/prepare_product.sh", product / "prepare_product.sh")
    subprocess.run(["bash", "prepare_product.sh"], cwd=product, check=True)


def descendants(pid):
    """Remember nested HTTP worker/server process groups while their parents still exist on Linux."""
    pending, found = [pid], {}
    while pending:
        parent = pending.pop()
        try:
            entries = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
        except OSError:
            continue
        for child in map(int, entries):
            try:
                stamp = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[19]
            except OSError:
                continue
            if child not in found:
                found[child] = stamp
                pending.append(child)
    return found


def stop(process):
    children = descendants(process.pid)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            children.update(descendants(process.pid))
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    for pid, stamp in children.items():
        try:
            if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19] == stamp:
                os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, FileNotFoundError):
            pass


def run(args):
    base, out = args.capsule.resolve(), args.out.resolve()
    gguf, tok = args.gguf.resolve(), args.tokenizer.resolve()
    program = Path(__file__).resolve()
    source = program.parents[2]
    out.mkdir(mode=0o700)
    work = out / "work"
    work.mkdir(mode=0o700)
    product = work / "product"
    python = product / "product-venv/bin/python"
    started = time.monotonic()
    deadline = started + args.budget_seconds
    stages = {}
    report = {"schema": "ironmule.kg4-scoped-campaign.v1", "status": "running", "stages": stages,
              "native_eligible": False, "http_eligible": False, "seed": SEED}

    publish = args.publish_dir.resolve() if args.publish_dir else None
    if publish:
        publish.mkdir(parents=True, exist_ok=True)
        if any((publish / name).exists() for name in PUBLIC.values()):
            raise ValueError("previous scoped campaign mirrors exist")
    published = {}

    def mirror():
        if publish:
            for name, public_name in PUBLIC.items():
                path = out / name
                if not path.is_file():
                    continue
                stat = path.stat()
                stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                if stamp == published.get(name) or not stat.st_size:
                    continue
                data = json.loads(path.read_bytes())
                write(publish / public_name, data)
                published[name] = stamp

    def save():
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        write(out / "report.json", report)
        mirror()

    def stage(key, argv, env, cap):
        available = min(cap, deadline - time.monotonic())
        if available <= 0:
            raise TimeoutError(key)
        stages[key] = {"status": "running"}
        save()
        begin = time.monotonic()
        with (work / f"{key}.log").open("wb") as log:
            child = subprocess.Popen(argv, env=env, cwd=base, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                expires = begin + available
                while True:
                    remaining = expires - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(argv, available)
                    try:
                        rc = child.wait(timeout=min(remaining, 2) if publish else remaining)
                        break
                    except subprocess.TimeoutExpired:
                        mirror()
            except subprocess.TimeoutExpired:
                stop(child)
                stages[key]["status"] = "timeout"
                raise TimeoutError(key) from None
            except BaseException:
                stop(child)
                stages[key]["status"] = "interrupted"
                raise
            finally:
                stages[key]["elapsed_s"] = round(time.monotonic() - begin, 3)
        stages[key].update(status="completed" if rc == 0 else "failed", returncode=rc)
        if rc:
            raise RuntimeError(key)
        save()

    # Native calls and every isolated HTTP worker inherit the same graph, backend and thread settings.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("IRONMULE_", "GGML_"))}
    lib = str(base / "runtime/lib")
    env.update(GGML_CUDA_DISABLE_GRAPHS="1", IRONMULE_GGML_BINDINGS=str(base / "runtime/bindings"),
               LLAMA_CPP_LIB_PATH=lib, LD_LIBRARY_PATH=os.pathsep.join([lib, env.get("LD_LIBRARY_PATH", "")]),
               IRONMULE_GGML_GPU_LAYERS="999", IRONMULE_GGML_THREADS="2", IRONMULE_GGML_STREAMS="1",
               IRONMULE_GGML_PREFILL_BATCH="0", IRONMULE_KG="0", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(base), PIP_NO_INDEX="1")
    try:
        save()
        if gguf.stat().st_size != MODEL_BYTES:
            report["input_error"] = "wrong_model_size"
            raise ValueError("wrong model size")
        revision = source.parent / "revision.txt"
        if revision.is_file():
            report["source_revision"] = revision.read_text().strip()
        stage("prepare", [sys.executable, program, "prepare", base, product, gguf, source], env, 120)
        stage("device", [python, program, "device", out / "device.json"], env, 20)
        report.update(gpu=json.loads((out / "device.json").read_text()), gguf_sha256=MODEL,
                      stock_cuda_sha256=PINNED["runtime/lib/libggml-cuda.so"], scoped_plugin_sha256=R1)
        receipts = []
        for arm in ("stock", "scoped", "stock-closing"):
            receipt = out / f"native-{arm}.json"
            receipts.append(receipt)
            phase_env = {**env, "IRONMULE_PILOT": "0", "IRONMULE_GGML_SEQS": "32", "IRONMULE_GGML_SEQ_CTX": "768"}
            for label, var in (("kg4", "IRONMULE_KG4_COUNTER_DIR"), ("cublas", "IRONMULE_CUBLAS_COUNTER_DIR")):
                folder = work / f"{arm}-{label}"
                folder.mkdir()
                phase_env[var] = str(folder)
            if arm == "scoped":
                phase_env.update(IRONMULE_KG4_SCOPE="low-width8", IRONMULE_KG="1", IRONMULE_KG_LIB=str(base / "r1.so"))
            stage(arm, [python, "-m", "lab.kg4.scoped_native", "--capsule-dir", base, "--arm", arm, "--gguf", gguf,
                        "--seed", str(SEED), "--output", receipt], phase_env, 540)
            measured = json.loads(receipt.read_text())
            if (measured.get("arm") != arm or measured.get("status") != "measured"
                    or measured.get("gpu") != report["gpu"] or measured.get("gguf_sha256") != MODEL
                    or measured.get("plugin_sha256") != (R1 if arm == "scoped" else None)):
                raise ValueError("receipt differs from bound campaign")
            stages[arm]["receipt_sha256"] = sha(receipt)
            save()
        decision = out / "native-qualification.json"
        stage("judge", [python, program, "judge", decision, *receipts], env, 20)
        verdict = json.loads(decision.read_text())
        report["native_eligible"] = verdict["eligible"] is True
        if not report["native_eligible"]:
            report.update(status="retained_stock", reasons=verdict.get("reasons", []))
        elif args.native_only:
            report.update(status="native_only")
        else:
            http = out / "scoped-http.json"
            remaining = int(deadline - time.monotonic())
            stage("http", [python, "-m", "lab.kg4.scoped_http", "--gguf", gguf, "--tokenizer", tok,
                           "--receipt-a", receipts[0], "--receipt-b", receipts[1], "--receipt-c", receipts[2],
                           "--state-dir", work / "http", "--output", http, "--budget-seconds", str(max(1, remaining-5))],
                  {**env, "IRONMULE_KG4_PLACEMENT": "1"}, remaining)
            result = json.loads(http.read_text())
            if result.get("status") != "measured" or type(result.get("eligible")) is not bool:
                raise ValueError("incomplete HTTP qualification")
            report.update(status="measured", http_eligible=result["eligible"], reasons=result.get("reasons", []))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="incomplete", failure=type(exc).__name__)
        return 1
    finally:
        save()


if __name__ == "__main__":
    if sys.argv[1:2] == ["prepare"]:
        prepare(*(Path(s) for s in sys.argv[2:]))
    elif sys.argv[1:2] == ["device"]:
        from select_layout import gpu
        card = gpu()
        if card["name"] != "NVIDIA L40S":
            raise ValueError("the scoped experiment requires an L40S")
        write(Path(sys.argv[2]), card)
    elif sys.argv[1:2] == ["judge"]:
        from lab.kg4.scoped_native import qualify
        write(Path(sys.argv[2]), qualify(*(json.loads(Path(p).read_text()) for p in sys.argv[3:])))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for key in ("capsule", "gguf", "tokenizer", "out"):
            parser.add_argument("--" + key, type=Path, required=True)
        parser.add_argument("--budget-seconds", type=int, default=1500)
        parser.add_argument("--native-only", action="store_true")
        parser.add_argument("--publish-dir", type=Path, help="atomically mirror only numeric reports for a live collector")
        args = parser.parse_args()
        if not 1 <= args.budget_seconds <= 3600:
            parser.error("budget must be between one and 3600 seconds")
        def interrupted(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        raise SystemExit(run(args))
