"""Independent cold-restart quality, latency and serving gate for an attributed Python source change."""
from __future__ import annotations

import argparse
import fcntl
import math
import os
import shutil
import time
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from lab.pk1 import native_trial as native
from lab.pk1.restart_order import PATH, SCHEMA, order_digest, read
from lab.portable_gpu.bench import save
from training.code_trial import PAIRS

HARNESS = Path(__file__).with_name("restart_arm.py")
SEED = 92741


def arm(args, tree, seed, out, receipt=None):
    env = {"IRONMULE_" + k: v for k, v in native.FLAGS.items()}
    env.update(IRONMULE_KG_LIB="/kernel/active.so", IRONMULE_W4_TABLE="/kernel/dispatch.table")
    mounts = [(args.reference_lib, env["IRONMULE_KG_LIB"]), (args.table, env["IRONMULE_W4_TABLE"]),
              (HARNESS, "/kernel/restart_arm.py")]
    if receipt:
        env.update(IRONMULE_KG_RESTART_RECEIPT="/kernel/restart-receipt.json", IRONMULE_KG_RESTART_SHA256=native.checksum(receipt))
        mounts.append((receipt, env["IRONMULE_KG_RESTART_RECEIPT"]))
    started = time.perf_counter()
    run = SimpleNamespace(**{**vars(args), "checkout": tree})
    with TemporaryDirectory(dir=out.parent) as tmp:
        product = Path(tmp) / out.name
        native.launch(run, product, ["/kernel/restart_arm.py", "--seed", str(seed), "--gguf", "/models/" + args.gguf.name,
                                     "--output", "/output/" + out.name], mounts, env)
        row = read(product)
    row["broker_elapsed_s"] = time.perf_counter() - started
    save(out, row)
    return row


def validate(row, seed, *, expected_source=None):
    from ironmule.factual_gate import cases, correct
    from training.serving_trial import digest

    b = row["binding"]
    if (row["schema"] != "ironmule.qwen-restart.arm.v1" or row["seed"] != seed
            or row["harness_sha256"] != native.checksum(HARNESS) or row["pilot_decisions"] < 1
            or (expected_source is not None and row["source_sha256"] != expected_source)
            or b["gpu_layers"] != 999 or b["seqs"] != 32 or b["seq_ctx"] != 2048 or not b["speculate"]
            or b["flags"] != {"IRONMULE_" + k: v for k, v in native.FLAGS.items()}
            or b["settings"] != {"flash_attn": True, "kv_unified": False, "kv_f32": False}
            or not b["native"].get("sha256") or len(b["fp"]["devices"]) != 1
            or row["mask_bits"] != native.mask_bits(row["mask"]) or row["facts"] != [True] * 18
            or len(row["device_used_bytes"]) != 4 or any(type(v) is not int or v <= 0 for v in row["device_used_bytes"])
            or type(row["reused"]) is not bool):
        raise ValueError("incomplete source, hardware, execution profile or factual coverage")
    if len(row["factual"]) != 18 or not all(correct(q, answer["answer"]) and answer["output_ids"]
            for q, answer in zip(cases(seed), row["factual"], strict=True)):
        raise ValueError("failed independent factual grading")
    for name in ("cold", "warm"):
        wave = row[name]
        timings = wave["ttfts_s"]
        answers = wave["answers"]
        if (len(answers) != 32 or wave["completed"] != 32 or len(timings) != 32
                or not all(math.isfinite(t) and 0 < t <= wave["wall_s"] for t in timings)
                or any(not 0 < len(a["output_ids"]) <= 32 or a["stop"] not in ("eos", "length")
                       or (a["stop"] == "length" and len(a["output_ids"]) != 32) for a in answers)
                or wave["tokens"] != sum(len(a["output_ids"]) for a in answers) or wave["output_sha256"] != digest(answers)):
            raise ValueError("incomplete timed deliveries")
    if (not all(math.isfinite(row[k]) and row[k] > 0 for k in ("cold_s", "initialized_s", "broker_elapsed_s", "peak_rss_kib"))
            or row["initialized_s"] + row["cold"]["wall_s"] > row["cold_s"] + .02
            or row["cold_s"] + row["warm"]["wall_s"] > row["broker_elapsed_s"] + .05):
        raise ValueError("impossible process or wave timings")


def compare_pair(ref, cand, seed, binding=None, *, require_hit=True):
    for row in (ref, cand):
        validate(row, seed)
    if (ref["reused"] or cand["reused"] != require_hit or ref["binding"] != cand["binding"]
            or (binding is not None and ref["binding"] != binding)
            or ref["pilot_sha256"] != cand["pilot_sha256"] or ref["mask_bits"] != cand["mask_bits"]
            or ref["factual"] != cand["factual"] or any(any(ref[name][field] != cand[name][field] for field in ("output_sha256", "geometry"))
                                                       for name in ("cold", "warm"))):
        raise ValueError("different execution conditions, work, masks or replay status")
    return {"cold_speedup": ref["cold_s"] / cand["cold_s"],
            "warm_ratio": cand["warm"]["wall_s"] / ref["warm"]["wall_s"],
            "warm_p95_ratio": sorted(cand["warm"]["ttfts_s"])[30] / sorted(ref["warm"]["ttfts_s"])[30],
            "rss_ratio": cand["peak_rss_kib"] / ref["peak_rss_kib"],
            "device_ratio": max(cand["device_used_bytes"]) / max(ref["device_used_bytes"])}


def fresh_pass(ref, cand, seed):
    r = compare_pair(ref, cand, seed)
    return {**r, "passed": r["cold_speedup"] >= 1.02 and r["warm_ratio"] <= 1.02 and r["warm_p95_ratio"] <= 1.02
            and max(r["rss_ratio"], r["device_ratio"]) <= 1.05}


def judge(pairs, aa):
    if len(pairs) != PAIRS or len(aa) != 2:
        raise ValueError("need A/A plus six alternating pairs")
    compare_pair(aa[0], aa[1], SEED, require_hit=False)
    noise = abs(aa[0]["cold_s"] - aa[1]["cold_s"]) / min(aa[0]["cold_s"], aa[1]["cold_s"])
    scored = [compare_pair(r, c, SEED + i, aa[0]["binding"]) for i, (r, c) in enumerate(pairs)]
    cold = [r["cold_speedup"] for r in scored]
    passed = (min(cold) > 1 and median(cold) >= 1.02 and median(cold) - 1 > noise
              and median(r["warm_ratio"] for r in scored) <= 1.02 and median(r["warm_p95_ratio"] for r in scored) <= 1.02
              and max(max(r["rss_ratio"], r["device_ratio"]) for r in scored) <= 1.05)
    return {"passed": passed, "aa_relative_spread": noise, "median_cold_speedup": median(cold), "pairs": scored}


def _tree(base, source, root):
    dest = root / ("restart-candidate-" + native.checksum(source))
    if not dest.exists():
        with TemporaryDirectory(dir=root) as scratch:
            tree = Path(scratch) / "tree"
            shutil.copytree(base, tree, symlinks=True, copy_function=os.link, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
            replace = tree / PATH
            replace.unlink()  # This path is a fresh hardlink, never edit the reference inode.
            shutil.copyfile(source, replace)
            replace.chmod(0o444)
            compile(replace.read_text(), str(replace), "exec")
            tree.rename(dest)
    if native.checksum(dest / PATH) != native.checksum(source):
        raise ValueError("candidate checkout differs")
    return dest


def verify_order(order, args):
    if (order["schema"] != SCHEMA or order_digest(order) != order["order_id"]
            or native.checksum(args.checkout / PATH) != order["source_sha256"] or order["allowed_paths"] != [PATH]):
        raise ValueError("unexpected proposal binding")


def verify_binding(row, order):
    a, b = row["binding"], order["binding"]
    if (a["model_sha256"] != b["gguf"] or a["fp"]["devices"] != b["fp"]["devices"]
            or any(a["fp"][k] != b["fp"][k] for k in ("driver", "cc", "generator", "kg"))
            or a["native"] != b["native"] or a["table_sha256"] != b["table_sha256"] or a["w4_sha256"] != b["w4_sha256"]
            or a["settings"] != b["engine"] or row["mask_bits"] != native.mask_bits(b["mask"])):
        raise ValueError("restart binding differs from calibrated target")


def receipt_fields(row):
    return {"schema": "ironmule.w4-restart-qualified.v1", "binding": row["binding"], "mask": row["mask"],
            "quality": {"completed": 32, "facts": 18, "output_sha256": row["cold"]["output_sha256"]}}


def proof(args, directory):
    order = read(args.order)
    verify_order(order, args)
    refsha, candsha = order["source_sha256"], native.checksum(args.candidate_source)
    aa = [read(directory / f"aa-{i}.json") for i in range(2)]
    for row in aa:
        validate(row, SEED, expected_source=refsha)
        verify_binding(row, order)
    receipt = read(directory / "receipt.json")
    if {k: v for k, v in receipt.items() if k != "created_s"} != receipt_fields(aa[0]):
        raise ValueError("serving receipt is not derived from graded incumbent")
    candidate = read(directory / "context.json")
    if candidate != context(args, order):
        raise ValueError("source or harness has changed")
    pairs = []
    for i in range(PAIRS):
        ref, cand = (read(directory / f"{tag}-{i}.json") for tag in ("ref", "cand"))
        for row, source in ((ref, refsha), (cand, candsha)):
            validate(row, SEED + i, expected_source=source)
        if cand["mask"] != receipt["mask"]:
            raise ValueError("reused mask differs from the supplied reference")
        pairs.append((ref, cand))
    for name in ("missing", "stale"):
        cold = read(directory / f"{name}.json")
        validate(cold, SEED, expected_source=candsha)
        compare_pair(aa[0], cold, SEED, require_hit=False)
    wrong = read(directory / "stale-receipt.json")
    if wrong != {**receipt, "created_s": receipt["created_s"] - 172800}:
        raise ValueError("stale fixture was changed")
    scored = judge(pairs, aa)
    return {"passed": scored["passed"], "reference_sha256": refsha, "candidate_sha256": candsha,
            "binding": aa[0]["binding"], "evaluator_sha256": native.checksum(HARNESS), "metrics": scored}


def context(args, order):
    return {"order": order["order_id"], "base_commit": order["base_commit"], "source": order["source_sha256"],
            "candidate": native.checksum(args.candidate_source), "harness": native.checksum(HARNESS),
            "evaluator": native.checksum(__file__), "model": order["binding"]["gguf"], "w4": order["binding"]["w4_sha256"]}


def receipts(directory):
    names = ["context.json", "receipt.json", "stale-receipt.json", "stale.json", "missing.json", "aa-0.json", "aa-1.json",
             *(f"{tag}-{i}.json" for i in range(PAIRS) for tag in ("ref", "cand"))]
    return {n: native.checksum(directory / n) for n in names}


def qualification(args):
    result = read(args.trial / "result.json")
    verified = proof(args, args.trial)
    if (not verified["passed"] or result["summary"] != verified or result["context"] != context(args, read(args.order))
            or result["measurements_sha256"] != receipts(args.trial)):
        raise ValueError("no successful source-bound restart gate")
    return verified


def qualify(args):
    from training.code_author import write_once

    order = read(args.order)
    verify_order(order, args)
    state = args.trial
    state.mkdir(parents=True, exist_ok=True)
    with (state / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        setup = context(args, order)
        write_once(state / "context.json", setup)
        try:
            base = args.checkout.resolve()
            cand = _tree(base, args.candidate_source, args.worker_root)
            def run(name, tree, seed, receipt=None):
                dest = state / (name + ".json")
                if not dest.exists():
                    arm(args, tree, seed, dest, receipt)
                row = read(dest)
                validate(row, seed, expected_source=native.checksum(tree / PATH))
                verify_binding(row, order)
                return row

            aa = [run(f"aa-{i}", base, SEED) for i in range(2)]
            compare_pair(*aa, SEED, require_hit=False)
            doc = state / "receipt.json"
            if not doc.exists():
                save(doc, {**receipt_fields(aa[0]), "created_s": time.time()})
            receipt = read(doc)
            stale = state / "stale-receipt.json"
            write_once(stale, {**receipt, "created_s": receipt["created_s"] - 172800})
            for name, pointer in (("stale", stale), ("missing", None)):
                compare_pair(aa[0], run(name, cand, SEED, pointer), SEED, require_hit=False)
            for i in range(PAIRS):
                rows = {}
                for label in (("ref", "cand") if i % 2 == 0 else ("cand", "ref")):
                    rows[label] = run(f"{label}-{i}", cand if label == "cand" else base, SEED + i,
                                      doc if label == "cand" else None)
                compare_pair(rows["ref"], rows["cand"], SEED + i)
            result = proof(args, state)
            save(state / "result.json", {"schema": SCHEMA, "passed": result["passed"], "summary": result, "context": setup,
                                        "measurements_sha256": receipts(state), "elapsed_s": time.monotonic() - start})
        except Exception as exc:
            save(state / "result.json", {"schema": SCHEMA, "passed": False, "context": setup,
                                         "reason": str(exc)[:1200], "elapsed_s": time.monotonic() - start})
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("trial", "gguf", "checkout", "reference-lib", "table", "worker-root", "order", "candidate-source"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--worker-uid", type=int, default=60000)
    qualify(p.parse_args())


if __name__ == "__main__":
    main()
