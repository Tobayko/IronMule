"""Requalify, stress, withdraw and restart a local CPU Pilot on its original RunPod binding."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

COMMIT = "79ee889e679f38835df5f30e215215aa9d0c60d2"
SMOL = "2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d"
HEAD = "f66700d1593befeee7f01c9835606c644bebd3b2ca6aeb1bbb36c81b9157def0"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    tmp = path.with_suffix(".tmp")
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def filenames(args, prior):
    key = prior["binding"]["scratch_thread_head"]["learner_binding"]
    return args.state / "local", args.state / "local" / "pilot" / (key + ".candidate.npz")


def worker(args, prior):
    from ironmule import ggml_engine as ge
    from ironmule.ggml_binary import identity
    from ironmule.gguf_tokenizer import GgufTokenizer
    from ironmule.pilot_learning import ThreadLearning
    from ironmule_devices import verify_ggml
    from training import pilot_threads as pt
    from training import serving_trial as st

    binding = prior["binding"]
    for name, checksum in prior["runtime_source_sha256"].items():
        if sha(Path(ge.__file__).parent / name) != checksum:
            raise ValueError(f"unexpected runtime source {name}")
    if (sha(Path(st.__file__)) != prior["evaluator_sha256"]
            or sha(Path(pt.__file__)) != binding["scratch_thread_head"]["helper_sha256"]
            or sha(Path(ge.__file__).with_name("pilot_learning.py")) != binding["scratch_thread_head"]["learner_sha256"]):
        raise ValueError("serving or qualification implementation differs")
    local, published = filenames(args, prior)
    L = ge.bindings()
    L.llama_backend_init()
    verify_ggml("cpu")
    with contextlib.closing(ge.GgmlEngine(str(args.gguf), threads=2, seqs=4, seq_ctx=512, speculate=False)) as engine, \
            contextlib.closing(GgufTokenizer(str(args.gguf), L)) as tokenizer:
        env_hash = st.digest({k: v for k, v in os.environ.items() if k.startswith(
            ("IRONMULE_", "GGML_", "OPENBLAS_", "OMP_", "CUDA_"))})
        if (ge.machine_key() != binding["machine"] or sorted(os.sched_getaffinity(0)) != binding["affinity"]
                or identity()["sha256"] != binding["native"]["sha256"] or engine.gpu_layers or engine.settings != binding["engine_settings"]
                or not engine.pilot or sha(engine.pilot.weights_path) != binding["pilot_weights_sha256"]
                or engine.pilot.cores != 4 or engine.prefix_reuse != binding["prefix_reuse"] or env_hash != binding["env_sha256"]):
            raise ValueError("CPU/load geometry, binary, environment or Pilot differs from the earlier EPYC run")

        def fresh():
            return ThreadLearning(engine, local, SMOL, binding["pilot_weights_sha256"])

        def snapshot():
            learning = engine.pilot.learning
            return {"status": learning.status(), "pending_proposals": dict(learning.proposals), "pid": os.getpid(),
                    "head_available": learning.thread_candidate is not None}

        def record(name, value):
            save(args.state / f"{args.phase}-{name}.json", value)

        def waves(label):
            learning = engine.pilot.learning
            st.run_wave(engine, tokenizer, st.fixture(1241 ^ 0xbeef, 4, "short"), 3, {})
            # Exclude the warm-up from actual post-gate decision counts.
            old = dict(learning.candidate_status or {})
            rows = []
            for i in range(4):
                phase = ("short", "document")[i % 2]
                w = st.run_wave(engine, tokenizer, st.fixture(1241 + i, 4, phase), 3, {})
                rows.append({**w, "phase": phase, "thread_choice": learning.thread_decision()})
            digest = st.digest([w["output_sha256"] for w in rows])
            total, correct = sum(w["completed"] for w in rows), sum(w["correct"] for w in rows)
            status = learning.candidate_status or {}
            artifact = {"waves": rows, "completed": total, "correct": correct, "digest": digest,
                        "qualified_decisions": status.get("qualified_decisions", 0) - old.get("qualified_decisions", 0),
                        "reference_decisions": status.get("reference_decisions", 0) - old.get("reference_decisions", 0),
                        "state": snapshot()}
            record(label, artifact)
            if (total, correct, digest) != (16, 16, prior["output_sha256"]):
                raise ValueError(f"{label} changed copy completion, output or quality")
            return artifact

        engine.pilot.learning = fresh()
        if engine.pilot.learning.binding != binding["scratch_thread_head"]["learner_binding"]:
            raise ValueError("foreign thread binding")
        record("boot", snapshot())
        if engine.pilot.learning.weights or engine.pilot.learning.thread_candidate is None:
            raise ValueError("a fresh process must start with hints and no active offsets")
        if args.phase == "restart":
            before = waves("prequalification")
            if before["qualified_decisions"]:
                raise ValueError("restarted process applied a head before qualification")

        receipt = pt.qualify_candidate(engine, args.gguf, published, local, profile_clients=4)
        record("qualification", receipt)
        steady = waves("steady")
        if args.phase == "restart":
            record("result", {"fresh_qualification": receipt["binding"]["qualified_offsets"],
                              "served_qualified_decisions": steady["qualified_decisions"],
                              "outcome": "qualified choices served" if steady["qualified_decisions"] else "abstained"})
            return
        # The regular idle recheck may remove an existing offset, retain it, or see a different load bin.
        before = snapshot()
        ctx = multiprocessing.get_context("spawn")
        stop = ctx.Event()
        load = ctx.Process(target=pt.busy, args=(stop,), daemon=True)
        load.start()
        try:
            exercise = waves("under_load")
            ids = engine.tokenize("Continue the sequence by counting from one to ten.")
            prompt = (ids * ((97 + len(ids) - 1) // len(ids)))[:97]
            # As in the frozen qualifier, a timing fixture must leave its four slots free for idle replay.
            # The served waves retain their usual prefix policy.
            reuse = engine.prefix_reuse
            engine.prefix_reuse = False
            try:
                trace = pt.request(engine, prompt, clients=4)
                last_before = engine.pilot.learning.last
                engine.pilot.learning.idle(lambda: False)
                new_verdict = engine.pilot.learning.last is not last_before
            finally:
                engine.prefix_reuse = reuse
            after = snapshot()
            record("load_recheck", {"before": before, "after": after, "request": trace, "foreign_workers": 1,
                                    "fresh_idle_verdict": new_verdict, "served_qualified_decisions": exercise["qualified_decisions"]})
        finally:
            stop.set()
            load.join(timeout=3)
            if load.is_alive():
                load.terminate()
                load.join()
        # Only withdraw the private published copy. The source checkpoint remains immutable.
        published.replace(args.state / "withdrawn-head.npz")
        engine.pilot.learning.idle(lambda: False)
        withdrawn = waves("withdrawn")
        if withdrawn["state"]["head_available"] or withdrawn["qualified_decisions"]:
            raise ValueError("a withdrawn head is still making scratch choices")
        record("result", {"real_load_recheck": (after["status"].get("last") or {}).get("result") if new_verdict else "no new verdict",
                          "initially_served_qualified_decisions": steady["qualified_decisions"],
                          "checkpoint_withdrawal": "served verified outputs with head unavailable"})


def controller(args, prior):
    if not set(range(4)).issubset(os.sched_getaffinity(0)):
        raise ValueError("original CPU affinity 0,1,2,3 unavailable")
    os.sched_setaffinity(0, set(range(4)))
    os.nice(10)
    frozen = subprocess.check_output(["git", "-C", args.repo, "rev-parse", "HEAD"], text=True).strip()
    if frozen != COMMIT or prior["commit"] != COMMIT:
        raise ValueError("expected the frozen EPYC runtime and earlier serving receipt")
    if sha(args.gguf) != SMOL or sha(args.weights) != HEAD:
        raise ValueError("unexpected checkpoint or GGUF")
    if prior["binding"]["scratch_thread_head"]["weights_sha256"] != HEAD or prior["correct"] != 16:
        raise ValueError("wrong original serving receipt")
    args.state.mkdir(mode=0o700, parents=True, exist_ok=False)
    _, published = filenames(args, prior)
    published.parent.mkdir(parents=True)
    start = time.monotonic()
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "HOME", "LD_LIBRARY_PATH", "IRONMULE_GGML_BINDINGS", "LLAMA_CPP_LIB_PATH", "LANG")}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", IRONMULE_GGML_GPU_LAYERS="0",
               CUDA_VISIBLE_DEVICES="", IRONMULE_PILOT="1", PYTHONPATH=str(args.repo))
    def interrupted(signum, _frame):
        raise InterruptedError(f"stopped by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = {"schema": "ironmule.epyc-withdrawal.v1", "prior_sha256": sha(args.prior), "head_sha256": HEAD,
              "gguf_sha256": SMOL, "commit": frozen, "driver_sha256": sha(Path(__file__)),
              "budget_seconds": args.seconds, "new_bookings_usd": 0, "phases": []}
    try:
        for phase in ("live", "restart"):
            # Restart uses the same persistent policy store but the original candidate file is reinstated.
            shutil.copyfile(args.weights, published)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--phase", phase, "--repo", args.repo,
                   "--gguf", args.gguf, "--weights", args.weights, "--prior", args.prior, "--state", args.state]
            with (args.state / f"{phase}.log").open("wb") as log:
                child = subprocess.Popen(cmd, cwd=args.repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    rc = child.wait(timeout=max(0.1, args.seconds - (time.monotonic() - start)))
                finally:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
            result["phases"].append({"phase": phase, "exit": rc, "through_s": time.monotonic() - start})
            if rc:
                raise RuntimeError(f"phase {phase} failed; see its log")
        result["outcome"] = "completed"
    except BaseException as exc:
        result["outcome"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_s"] = time.monotonic() - start
        result["files"] = {str(p.relative_to(args.state)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                           for p in sorted(args.state.rglob("*")) if p.is_file()}
        save(args.state / "manifest.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("--repo", "--gguf", "--weights", "--prior", "--state"):
        parser.add_argument(name, type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument("--seconds", type=int, choices=range(1, 361), default=285, metavar="1..360")
    parser.add_argument("--phase", choices=("live", "restart"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    prior = json.loads(args.prior.read_text())
    if args.phase:
        worker(args, prior)
    else:
        controller(args, prior)


if __name__ == "__main__":
    main()
