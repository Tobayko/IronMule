"""Run a single attributed program mutation against the active cached PR60 serving profile."""

from __future__ import annotations

import argparse
import ast
import fcntl
import math
import time
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from ironmule.factual_gate import cases, correct
from lab.pk1 import followup_contract as rules
from lab.pk1 import restart_trial as prior
from lab.pk1.restart_order import PATH, order_digest
from training import scoped_release as release
from training.code_author import digest, write_once
from training.code_trial import read
from training.pilot_restart_profiles import select

contract = rules.contract
WORKER = Path(__file__).with_name("followup_arm.py")


def validate_order(order):
    target = order["target"]
    if (
        order["schema"] != rules.SCHEMA
        or order_digest(order) != order["order_id"]
        or order["base_commit"] != rules.BASE
        or order["parent_order"] != rules.PARENT
        or (order["control_receipt_sha256"], order["weights_sha256"]) != (rules.CONTROL, rules.HEAD)
        or order["contract"] != rules.contract_digest()
        or order["allowed_paths"] != [PATH]
        or target["path"] != PATH
        or target["class"] != "GgmlEngine"
        or not target["function"].isidentifier()
        or target["function"] == "measure_kg"
    ):
        raise ValueError("different observed program, release or prospective contract")


def evaluator_sha():
    return digest(
        [
            prior.native.checksum(f)
            for f in (
                __file__,
                rules.__file__,
                WORKER,
                prior.__file__,
                prior.HARNESS,
                prior.native.__file__,
                select.__code__.co_filename,
            )
        ]
    )


def source_check(src, candidate, order):
    def split(tree):
        text = tree.read_text()
        lines = text.splitlines(keepends=True)
        (klass,) = [x for x in ast.parse(text).body if isinstance(x, ast.ClassDef) and x.name == "GgmlEngine"]
        (method,) = [f for f in klass.body if isinstance(f, ast.FunctionDef) and f.name == order["target"]["function"]]
        if method.decorator_list:
            raise ValueError("decorated method outside experiment")
        return method.lineno, lines[: method.lineno - 1], lines[method.end_lineno :], ast.dump(method.args)

    first = split(src)
    second = split(candidate)
    if (
        first != second
        or first[0] != order["target"]["line"]
        or prior.native.checksum(src) != order["source_sha256"]
        or prior.native.checksum(candidate) == order["source_sha256"]
    ):
        raise ValueError("changed code outside the observed method or unchanged proposal")


def parent(args):
    order = read(args.order)
    validate_order(order)
    if prior.native.checksum(args.control) != rules.CONTROL or prior.native.checksum(args.weights) != rules.HEAD:
        raise ValueError("foreign release or scratch Pilot")
    receipt = read(args.control)
    q = receipt["qualification"]
    published = receipt["publication"]
    if (
        q["candidate_sha256"] != order["source_sha256"]
        or receipt["order"]["order_id"] != rules.PARENT
        or published["candidate"] != rules.BASE
        or q["passed"] is not True
    ):
        raise ValueError("incompatible parent PR or source")
    source_check(args.checkout / PATH, args.candidate_source, order)
    return order, receipt


def context(args):
    order, receipt = parent(args)
    return {
        "order": order["order_id"],
        "parent_order": rules.PARENT,
        "base_commit": rules.BASE,
        "control": rules.CONTROL,
        "source": order["source_sha256"],
        "candidate": prior.native.checksum(args.candidate_source),
        "weights": rules.HEAD,
        "cache": receipt["qualification"]["binding"]["cache_receipt_sha256"],
        "contract": rules.contract_digest(),
        "evaluator": evaluator_sha(),
    }


def live(args, label, seed, out, *, factual_first=False):
    """Time a fully verified, already admitted parent startup and one physical fresh reference/candidate worker."""
    order, receipt = parent(args)
    expected = receipt["qualification"]["binding"]
    tree = (
        args.checkout if label == "reference" else prior._tree(args.checkout, args.candidate_source, args.worker_root)
    )
    if label not in ("reference", "candidate"):
        raise ValueError("unknown source")
    began = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    active = release.selected(args.parent_state, rules.CONTROL) == "candidate"
    cache_ready = (
        prior.native.checksum(args.cache) == expected["cache_receipt_sha256"]
        and 0 <= time.time() - read(args.cache)["created_s"] <= 86400
    )
    family = "factual-first-18" if factual_first else "mixed-first-32"
    choice = select(
        args.weights, receipt, family=family, cache_ready=cache_ready, active=active, expected_sha256=rules.HEAD
    )
    if choice["profile"] != "candidate" or choice["reason"] != "qualified model choice":
        raise ValueError("bound cached PR60 unavailable; retain preceding deployment")
    env = {"IRONMULE_" + k: v for k, v in prior.native.FLAGS.items()}
    env.update(
        IRONMULE_KG_LIB="/kernel/active.so",
        IRONMULE_W4_TABLE="/kernel/dispatch.table",
        IRONMULE_KG_RESTART_RECEIPT="/kernel/receipt.json",
        IRONMULE_KG_RESTART_SHA256=expected["cache_receipt_sha256"],
    )
    mounts = [
        (args.reference_lib, env["IRONMULE_KG_LIB"]),
        (args.table, env["IRONMULE_W4_TABLE"]),
        (args.cache, env["IRONMULE_KG_RESTART_RECEIPT"]),
        (prior.HARNESS, "/kernel/restart_arm.py"),
        (WORKER, "/kernel/followup_arm.py"),
    ]
    actual = SimpleNamespace(**{**vars(args), "checkout": tree})
    with TemporaryDirectory(dir=out.parent) as scratch:
        dest = Path(scratch) / "observed.json"
        prior.native.launch(
            actual,
            dest,
            [
                "/kernel/followup_arm.py",
                "--broker-start-ns",
                str(began),
                "--output",
                "/output/observed.json",
                "--gguf",
                "/models/" + args.gguf.name,
                "--seed",
                str(seed),
                *(["--factual-first"] if factual_first else []),
            ],
            mounts,
            env,
        )
        row = read(dest)
    row.update(
        broker_elapsed_s=(time.clock_gettime_ns(time.CLOCK_BOOTTIME) - began) / 1e9,
        source_label=label,
        startup_selection=choice,
        followup_order=order["order_id"],
        cache_sha256=expected["cache_receipt_sha256"],
        weights_sha256=rules.HEAD,
    )
    write_once(out, row)
    return row


def validate(row, label, seed, ctx, receipt, factual=False):
    expected = receipt["qualification"]["binding"]
    p = receipt["order"]
    sha = ctx["source" if label == "reference" else "candidate"]
    prior.validate(row, seed, expected_source=sha)
    prior.verify_binding(row, p)
    choice = row["startup_selection"]
    if (
        row["binding"] != expected["runtime"]
        or row["mask_bits"] != expected["mask_bits"]
        or row["cuda_vendor"] != expected["vendor"]
        or row["pilot_sha256"] != expected["pilot_sha256"]
        or not row["reused"]
        or row["weights_sha256"] != rules.HEAD
        or row["followup_order"] != ctx["order"]
        or row["source_label"] != label
        or row["followup_harness_sha256"] != prior.native.checksum(WORKER)
        or row["cache_sha256"] != expected["cache_receipt_sha256"]
        or choice["weights_sha256"] != rules.HEAD
        or choice["reason"] != "qualified model choice"
        or choice["profile"] != "candidate"
        or choice["previous"]
        or choice["ablate_weights"]
        or choice["family"] != ("factual-first-18" if factual else "mixed-first-32")
        or not 0 < choice["cpu_ms"] <= 1000
        or not math.isfinite(row["broker_to_process_s"])
        or not 0 <= row["broker_to_process_s"] <= 600
        or row["cold_s"] + row["broker_to_process_s"] + row["warm"]["wall_s"] > row["broker_elapsed_s"] + 0.05
        or len(row["factual_first"]) != (2 if factual else 0)
    ):
        raise ValueError("source, profile, work or controller changed")
    last = row["initialized_s"]
    for offset, batch in enumerate(row["factual_first"]):
        correct_seed = seed if offset == 0 else seed ^ 419
        prior.fact_timing(batch)
        if (
            batch["seed"] != correct_seed
            or batch["correct"] != [True] * 18
            or len(batch["answers"]) != 18
            or batch["completed_s"] < last + batch["wall_s"] - 0.02
            or batch["completed_s"] > row["cold_s"]
            or not all(
                correct(q, a["answer"]) and a["output_ids"] and a["stop"] in ("length", "eos")
                for q, a in zip(cases(correct_seed), batch["answers"], strict=True)
            )
        ):
            raise ValueError("factual-first workload differs")
        last = batch["completed_s"]


def scores(ref, cand):
    if (
        ref["factual"] != cand["factual"]
        or any(
            ref[k]["output_sha256"] != cand[k]["output_sha256"] or ref[k]["geometry"] != cand[k]["geometry"]
            for k in ("cold", "warm")
        )
        or ref["mask_bits"] != cand["mask_bits"]
        or ref["binding"] != cand["binding"]
        or ref["cuda_vendor"] != cand["cuda_vendor"]
    ):
        raise ValueError("different delivered work or effective serving profile")

    def charged(row, batch=None):
        return row["broker_to_process_s"] + (batch["completed_s"] if batch else row["cold_s"])

    result = {
        "cold_speedup": charged(ref) / charged(cand),
        "warm_wall": cand["warm"]["wall_s"] / ref["warm"]["wall_s"],
        "warm_p95": sorted(cand["warm"]["ttfts_s"])[30] / sorted(ref["warm"]["ttfts_s"])[30],
        "warm_fact": cand["factual_timing"]["wall_s"] / ref["factual_timing"]["wall_s"],
        "warm_fact_p95": prior.fact_timing(cand["factual_timing"]) / prior.fact_timing(ref["factual_timing"]),
        "rss": cand["peak_rss_kib"] / ref["peak_rss_kib"],
        "device": max(cand["device_used_bytes"]) / max(ref["device_used_bytes"]),
    }
    if ref["factual_first"]:
        off, on = ref["factual_first"], cand["factual_first"]
        if any(a["answers"] != b["answers"] for a, b in zip(off, on, strict=True)):
            raise ValueError("changed factual deliveries")

        def ttft(row, b):
            return charged(row, b) - b["wall_s"] + prior.fact_timing(b)

        result.update(
            factual_cold=charged(ref, off[0]) / charged(cand, on[0]),
            factual_cold_p95=ttft(cand, on[0]) / ttft(ref, off[0]),
            factual_warm=on[1]["wall_s"] / off[1]["wall_s"],
            factual_warm_p95=prior.fact_timing(on[1]) / prior.fact_timing(off[1]),
        )
    return result


def select_source(expected, *, actual, admitted, parent_available):
    if not parent_available:
        return {"source": "preceding-deployment", "reason": "matching active parent unavailable"}
    if not admitted or expected != actual:
        return {"source": "reference", "reason": "withdrawn follow-up" if not admitted else "unqualified source digest"}
    return {"source": "candidate", "reason": "bound verified source"}


def gate(mixed, factual, noise):
    def rate(rows, k):
        return median(r[k] for r in rows)

    warm = ("warm_wall", "warm_p95", "warm_fact", "warm_fact_p95")
    return (
        all(r["cold_speedup"] > 1 for r in mixed)
        and rate(mixed, "cold_speedup") >= 1.02
        and rate(mixed, "cold_speedup") - 1 > noise
        and all(rate(mixed, k) <= 1.02 for k in warm)
        and all(r["factual_cold"] > 1 for r in factual)
        and rate(factual, "factual_cold") >= 1.02
        and all(rate(factual, k) <= 1.02 for k in ("factual_cold_p95", "factual_warm", "factual_warm_p95"))
        and all(max(r["rss"], r["device"]) <= 1.05 for r in [*mixed, *factual])
    )


def file_names():
    return [
        "context.json",
        "aa-0.json",
        "aa-1.json",
        "invalid-followup.json",
        "withdrawn-followup.json",
        *(
            f"{group}-{tag}-{seed}.json"
            for group, seeds in (("mixed", rules.MIXED), ("facts", rules.FACTS))
            for seed in seeds
            for tag in ("reference", "candidate")
        ),
    ]


def proof(args):
    order, receipt = parent(args)
    ctx = context(args)
    p = args.trial
    if read(p / "context.json") != ctx:
        raise ValueError("changed follow-up apparatus or source")

    def row(name, tag, seed, first=False):
        item = read(p / (name + ".json"))
        validate(item, tag, seed, ctx, receipt, first)
        return item

    aa = [row(f"aa-{i}", "reference", rules.AA_SEED) for i in range(2)]
    a_times = [a["cold_s"] + a["broker_to_process_s"] for a in aa]
    noise = max(a_times) / min(a_times) - 1
    scores(*aa)
    for name, actual, eligible in [
        ("invalid-followup", ctx["source"], True),
        ("withdrawn-followup", ctx["candidate"], False),
    ]:
        record = row(name, "reference", rules.AA_SEED)
        intended = select_source(ctx["candidate"], actual=actual, admitted=eligible, parent_available=True)
        if record["followup_route"] != intended or intended["source"] != "reference":
            raise ValueError("could not restore prior release")
        scores(aa[0], record)
    groups = []
    for name, seeds in (("mixed", rules.MIXED), ("facts", rules.FACTS)):
        outcome = []
        for seed in seeds:
            ref, cand = (row(f"{name}-{t}-{seed}", t, seed, name == "facts") for t in ("reference", "candidate"))
            outcome.append(scores(ref, cand))
        groups.append(outcome)
    mixed, factual = groups
    return {
        "passed": gate(mixed, factual, noise),
        "reference_sha256": ctx["source"],
        "candidate_sha256": ctx["candidate"],
        "evaluator_sha256": ctx["evaluator"],
        "binding": {
            "parent_control": ctx["control"],
            "parent_order": rules.PARENT,
            "profile": receipt["qualification"]["binding"],
            "cache_sha256": ctx["cache"],
            "weights_sha256": ctx["weights"],
            "order": order["order_id"],
            "contract": ctx["contract"],
        },
        "metrics": {
            "aa_spread": noise,
            "mixed": mixed,
            "factual_first": factual,
            "median_charged_cold_speedup": median(r["cold_speedup"] for r in mixed),
            "median_charged_factual_speedup": median(r["factual_cold"] for r in factual),
        },
    }


def qualification(args):
    observation = proof(args)
    stored = read(args.trial / "result.json")
    if (
        not observation["passed"]
        or observation != stored.get("summary")
        or stored["artifacts"] != {n: prior.native.checksum(args.trial / n) for n in file_names()}
    ):
        raise ValueError("no qualifying source-bound comparison over current release")
    return observation


def qualify(args):
    order, receipt = parent(args)
    ctx = context(args)
    p = args.trial
    p.mkdir(parents=True, exist_ok=True)
    with (p / ".lock").open("a") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(p / "context.json", ctx)
        if (p / "result.json").is_file():
            return read(p / "result.json")
        started = time.monotonic()

        def run(name, tag, seed, *, first=False, route=None):
            dest = p / (name + ".json")
            if not dest.is_file():
                if route is None:
                    live(args, tag, seed, dest, factual_first=first)
                else:
                    with TemporaryDirectory(dir=p) as temp:
                        obs = live(args, tag, seed, Path(temp) / "raw.json")
                        write_once(dest, {**obs, "followup_route": route})
            row = read(dest)
            validate(row, tag, seed, ctx, receipt, first)
            return row

        try:
            aa = [run(f"aa-{i}", "reference", rules.AA_SEED) for i in range(2)]
            scores(*aa)
            for name, actual, admitted in [
                ("invalid-followup", ctx["source"], True),
                ("withdrawn-followup", ctx["candidate"], False),
            ]:
                route = select_source(ctx["candidate"], actual=actual, admitted=admitted, parent_available=True)
                scores(aa[0], run(name, route["source"], rules.AA_SEED, route=route))
            for name, seeds in (("mixed", rules.MIXED), ("facts", rules.FACTS)):
                for i, seed in enumerate(seeds):
                    rows = {}
                    for tag in ("reference", "candidate") if i % 2 == 0 else ("candidate", "reference"):
                        rows[tag] = run(f"{name}-{tag}-{seed}", tag, seed, first=name == "facts")
                    scores(rows["reference"], rows["candidate"])
            summary = proof(args)
            outcome = {
                "schema": "ironmule.followup-trial.v1",
                "passed": summary["passed"],
                "summary": summary,
                "artifacts": {n: prior.native.checksum(p / n) for n in file_names()},
            }
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            outcome = {
                "schema": "ironmule.followup-trial.v1",
                "passed": False,
                "reason": f"{type(exc).__name__}: {exc}"[:1200],
                "artifacts": {n: prior.native.checksum(p / n) for n in file_names() if (p / n).is_file()},
            }
        write_once(p / "result.json", {**outcome, "context": ctx, "elapsed_s": time.monotonic() - started})
        return read(p / "result.json")


def main():
    import json

    p = argparse.ArgumentParser(description=__doc__)
    for k in (
        "order",
        "checkout",
        "candidate-source",
        "trial",
        "control",
        "weights",
        "parent-state",
        "cache",
        "worker-root",
        "reference-lib",
        "table",
        "gguf",
    ):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--worker-uid", type=int, default=60000)
    p.add_argument("--timeout", type=int, default=420)
    print(json.dumps(qualify(p.parse_args())))


if __name__ == "__main__":
    main()
