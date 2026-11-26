"""Qualify and use the ordinary product bridge on a published, locally activated EPYC runtime."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

from lab.rp import cpu_code_trial as trial
from lab.rp.restricted_worker import open_worker
from training import scoped_release as release
from training.code_trial import judge, read, save
from training.serving_trial import COLORS, fixture

WORKER = Path(__file__).with_name("cpu_product_worker.py")
MATCHED = Path(__file__).with_name("cpu_matched_profile.py")
SUITE = list(range(9821, 9827))
COPY = re.compile(r"Record [0-9]{1,6}\. ((?:Notes for the copy task\. ){1,12})\nColor: red\nCopy: red\nColor: blue\nCopy: blue\nColor: (" + "|".join(COLORS) + r")\nCopy:")


def scope(prompts, limit):
    matches = [COPY.fullmatch(p) for p in prompts]
    return len(matches) == 4 and limit == 3 and all(m and m[1].count("Notes") in (1, 12) for m in matches) and len({m[1] for m in matches}) == 1


def checked_line(proc):
    raw = proc.stdout.readline(1 << 20)
    if not raw.endswith("\n"):
        raise ValueError("incomplete product worker message")
    return json.loads(raw)


def collect_reply(proc, start, lanes):
    first = [None] * lanes
    pieces = [None] * lanes
    for _ in range(lanes):
        packet = checked_line(proc)
        lane = packet.get("lane")
        if type(lane) is not int or not 0 <= lane < lanes or pieces[lane] is not None:
            raise ValueError("missing or duplicate first visible text")
        if packet.get("event") == "empty_answer":
            pieces[lane] = ""
        elif packet.get("event") == "first_text" and isinstance(packet.get("text"), str) and packet["text"]:
            first[lane] = time.monotonic() - start
            pieces[lane] = packet["text"]
        else:
            raise ValueError("invalid first-text event")
    row = checked_line(proc)
    row["broker_wall_s"] = time.monotonic() - start
    if any(bool(answer["text"]) != bool(piece) or not answer["text"].startswith(piece)
           for answer, piece in zip(row["answers"], pieces, strict=True)):
        raise ValueError("streamed text differs from the completed answer")
    row["worker_ttfts_s"], row["ttfts_s"] = row["ttfts_s"], first
    return row


@contextmanager
def arm(args, receipt, name):
    from lab.rp.cpu_code_release import author_order
    tree = release.projected_tree(receipt, args.reference, args.worker_root,
                                  args.candidate / release.SOURCE if name == "candidate" else None)
    order, _, _ = trial.inputs(args)
    if author_order(receipt) != order:
        raise ValueError("different product order")
    env = {k: os.environ[k] for k in ("IRONMULE_GGML_BINDINGS", "LD_LIBRARY_PATH", "LLAMA_CPP_LIB_PATH")}
    env.update(IRONMULE_GGML_GPU_LAYERS="0", IRONMULE_PILOT="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               IRONMULE_GGML_SEQ_CTX="512", IRONMULE_GGML_THREADS="2", IRONMULE_PILOT_LEARN="0",
               CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(tree))
    readonly = [tree, args.head, args.gguf, args.python.parent.parent, Path("/etc/machine-id"),
                Path(env["IRONMULE_GGML_BINDINGS"]), Path(env["LLAMA_CPP_LIB_PATH"]),
                *(Path(p) for p in env["LD_LIBRARY_PATH"].split(":") if p and not Path(p).resolve().is_relative_to("/usr"))]
    with tempfile.TemporaryDirectory(dir=args.worker_root) as raw:
        stage = Path(raw)
        stage.chmod(0o711)
        trusted = stage / "trusted"
        trusted.mkdir(mode=0o755)
        readonly.append(trusted)

        def stage_input(path, name):
            target = trusted / name
            target.write_bytes(path.read_bytes())
            target.chmod(0o444)
            return target

        worker_path = stage_input(WORKER, WORKER.name)
        order_path = stage_input(args.order, "order.json")
        if read(order_path) != order:
            raise ValueError("staged product order differs from the source release")
        out = stage / "out"
        out.mkdir(mode=0o700)
        os.chown(out, 60000, 60000)
        cmd = ["/usr/bin/env", "-u", "CUDA_CACHE_PATH", str(args.python), str(worker_path), "--gguf", str(args.gguf),
               "--head", str(args.head), "--order", str(order_path), "--arm", name]
        if receipt["adapter"] == "epyc-prefill-matched-v2":
            from lab.rp.cpu_matched_profile import verify_profile

            if not args.profile:
                raise ValueError("expected the published matching CPU calibration")
            profile_path = stage_input(args.profile, "profile.json")
            if trial.sha(profile_path) != receipt["order"]["paired_profile_sha256"]:
                raise ValueError("staged CPU calibration differs from the source release")
            profile = read(profile_path)
            verify_profile(profile, order)
            helper_path = stage_input(MATCHED, MATCHED.name)
            cmd.extend(["--profile", str(profile_path), "--matched-helper", str(helper_path)])
        with (args.state / "deployment" / f"product-{name}.log").open("ab") as log, open_worker(
                cmd, env=env, cwd=tree, output=out, readonly=readonly, timeout=args.timeout, stderr=log) as proc:
            started = checked_line(proc)
            if not started["ready"] or started["worker_sha256"] != trial.sha(WORKER):
                raise ValueError("unverified product worker start")
            if receipt["adapter"] == "epyc-prefill-matched-v2":
                from lab.rp.cpu_matched_profile import measured_margin

                if started["profile_sha256"] != receipt["order"]["paired_profile_sha256"] or started["costs"] != profile["costs"]:
                    raise ValueError("foreign product calibration")
                for key, offset in profile["weights"].items() if started["qualified"] else ():
                    checked = started["fresh"][key]
                    if measured_margin(key, offset, checked["probes"]) != checked["margin"]:
                        raise ValueError("unverified product admission")
            def request(prompts, limit):
                start = time.monotonic()
                proc.stdin.write(json.dumps({"prompts": prompts, "max_tokens": limit}) + "\n")
                proc.stdin.flush()
                row = collect_reply(proc, start, len(prompts))
                if (len(row["answers"]) != len(prompts) or len(row["ttfts_s"]) != len(prompts)
                        or any(len(a["tokens"]) > limit or a["finish_reason"] not in ("length", "stop") for a in row["answers"])
                        or not all(v is None or math.isfinite(v) and 0 < v <= row["broker_wall_s"] for v in row["ttfts_s"])):
                    raise ValueError("incomplete product replies")
                return row
            yield request, started


def grade(request, seed, binding):
    waves, expected = [], []
    for i, phase in enumerate(("short", "document", "short", "document")):
        jobs = fixture(seed + i, 4, phase)
        waves.append(request([t for t, _ in jobs], 3))
        expected.extend([target for _, target in jobs])
    answers = [answer for w in waves for answer in w["answers"]]
    ttfts = [t for w in waves for t in w["ttfts_s"]]
    if any(t is None for t in ttfts):
        raise ValueError("copy fixture did not deliver visible text to every client")
    ttfts.sort()
    correct = sum(bool(re.match(r"\s*" + word + r"\b", answer["text"], re.IGNORECASE)) for word, answer in zip(expected, answers, strict=True))
    return {"waves": waves, "completed": len(answers), "correct": correct, "tokens": sum(len(a["tokens"]) for a in answers),
            "scratch_uses": sum(w["scratch_uses"] for w in waves),
            "output_sha256": hashlib.sha256(json.dumps(answers, sort_keys=True).encode()).hexdigest(),
            "wall_s": sum(w["broker_wall_s"] for w in waves), "ttft_s": ttfts[math.ceil(.95 * len(ttfts)) - 1],
            "peak_rss_kib": max(w["peak_rss_kib"] for w in waves), "binding": binding, "profiled": False,
            "workload": {"seed": seed, "clients": 4, "waves": 4, "max_tokens": 3}, "evaluator_sha256": trial.sha(WORKER)}


def binding(args, receipt):
    return {"receipt_sha256": args.receipt_sha256, "worker_sha256": trial.sha(WORKER), "broker_sha256": trial.sha(__file__),
            "qualification_scope": receipt["qualification"]["binding"], "seeds": SUITE,
            "matched_profile_sha256": receipt["order"].get("paired_profile_sha256")}


def qualify_inner(args, receipt):
    if release.selected(args.state, args.receipt_sha256) != "candidate":
        raise ValueError("the source release has no active target qualification")
    pin = binding(args, receipt)
    save(args.state / "deployment" / "product-gate.json", {"passed": False, "reason": "requalifying", "binding": pin})
    with ExitStack() as stack:
        workers = {name: stack.enter_context(arm(args, receipt, name)) for name in ("reference", "candidate")}
        for request, ready in workers.values():
            if not ready["qualified"]:
                raise ValueError("thread head abstained in the product worker")
            request([text for text, _ in fixture(55491, 4, "short")], 3)
        pairs = []
        for i, seed in enumerate(SUITE):
            sequence = list(workers)
            pair = {name: grade(workers[name][0], seed, pin) for name in (reversed(sequence) if i % 2 else sequence)}
            if any(v["scratch_uses"] == 0 or v["tokens"] != 48 for v in pair.values()):
                raise ValueError("missing real scratch choice or incomplete product work")
            save(args.state / "deployment" / f"product-pair-{i}.json", pair)
            pairs.append((pair["reference"], pair["candidate"]))
    result = judge(pairs)
    result.update(binding=pin, starts={name: opened[1] for name, opened in workers.items()})
    limits = {}
    for offset, name in enumerate(("short", "document")):
        reference = [w for r, _ in pairs for w in r["waves"][offset::2]]
        limits[name] = {"wall_s": max(w["broker_wall_s"] for w in reference),
                        "ttft_s": max(max(w["ttfts_s"]) for w in reference)}
    result["reference_limits"] = limits
    save(args.state / "deployment" / "product-gate.json", result)
    return result


def qualify(args, receipt):
    with (args.state / "deployment" / ".product-qualify.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return qualify_inner(args, receipt)


@contextmanager
def service(args, receipt):
    """Warm and requalify before admitting requests. New work incurs no qualification in this loop."""
    pin = binding(args, receipt)
    gate_path = args.state / "deployment" / "product-gate.json"
    try:
        gate = read(gate_path)
    except (OSError, ValueError):
        gate = {}
    with ExitStack() as stack:
        reference, _ = stack.enter_context(arm(args, receipt, "reference"))
        candidate = None
        if (release.selected(args.state, args.receipt_sha256) == "candidate"
                and gate.get("passed") is True and gate.get("binding") == pin):
            try:
                candidate, _ = stack.enter_context(arm(args, receipt, "candidate"))
            except (ValueError, OSError, RuntimeError) as exc:
                release.rollback(args.state, f"product candidate could not start: {exc}"[:400])

        def dispatch(prompts, limit):
            start = time.monotonic()
            if not isinstance(prompts, list) or not 1 <= len(prompts) <= 4 or type(limit) is not int or not 1 <= limit <= 64:
                raise ValueError("product batch outside configured capacity")
            if not all(isinstance(t, str) and 0 < len(t) <= 4096 for t in prompts):
                raise ValueError("invalid prompt")
            try:
                current = candidate and scope(prompts, limit) and read(gate_path) == gate
            except (OSError, ValueError):
                current = False
            if current and release.selected(args.state, args.receipt_sha256) == "candidate":
                try:
                    result = candidate(prompts, limit)
                    for prompt, answer in zip(prompts, result["answers"], strict=True):
                        if not re.match(r"\s*" + COPY.fullmatch(prompt)[2] + r"\b", answer["text"], re.IGNORECASE):
                            raise ValueError("incorrect product copy")
                    phase = "document" if prompts[0].count("Notes") == 12 else "short"
                    bound = gate["reference_limits"][phase]
                    elapsed = time.monotonic() - start
                    if elapsed > bound["wall_s"] or max(result["ttfts_s"]) > bound["ttft_s"]:
                        release.rollback(args.state, "exceeded the reference's product timing envelope")
                    return {"arm": "candidate", "work": result, "total_dispatch_s": time.monotonic() - start}
                except (OSError, RuntimeError, ValueError) as exc:
                    release.rollback(args.state, f"product candidate failed: {exc}"[:400])
            result = reference(prompts, limit)
            return {"arm": "reference", "work": result, "total_dispatch_s": time.monotonic() - start}

        yield dispatch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("qualify", "run", "serve"))
    for key in ("state", "order", "reference", "candidate", "gguf", "head", "worker-root", "python"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--receipt-sha256", required=True)
    p.add_argument("--timeout", type=int, choices=range(1, 901), default=300)
    p.add_argument("--requests", type=Path, help="JSON object with prompts and max_tokens")
    p.add_argument("--profile", type=Path, help="immutable CPU calibration for matched-v2")
    args = p.parse_args()
    receipt = release.pinned(args.state, args.receipt_sha256)
    if args.operation == "qualify":
        report = qualify(args, receipt)
        print(json.dumps(report, allow_nan=False))
        if not report["passed"]:
            raise SystemExit(1)
    else:
        with service(args, receipt) as dispatch:
            if args.operation == "serve":
                print(json.dumps({"ready": True}), flush=True)
                for line in sys.stdin:
                    try:
                        request = json.loads(line)
                        result = dispatch(request["prompts"], request["max_tokens"])
                    except (ValueError, KeyError, TypeError) as exc:
                        result = {"error": str(exc)}
                    print(json.dumps(result, allow_nan=False), flush=True)
            else:
                if args.requests is None:
                    p.error("run requires --requests")
                request = read(args.requests)
                print(json.dumps(dispatch(request["prompts"], request["max_tokens"]), allow_nan=False))


if __name__ == "__main__":
    main()
