#!/usr/bin/env python3
"""The profile-driven service-mode choice: migration, user path, and its own cost.

Three parts, one process, one model load.

**Migration** happens on a working copy only. The tool refuses to run against the
default store, so the production profile database is never touched. A profile without
the record must keep its behaviour; a profile with it must still do nothing until a
caller opts in.

**User path** walks what a local user does: load without the opt-in, load with it, serve
one request, a pair and four, read the status, present an unknown configuration, and
switch back. Correctness is checked against `InteractiveMode` on the same prompts, and
the bit-level logit/KV identity comes from `B51`; it is not re-derived here.

**Cost** asks the only timing question the change can raise: does choosing the strategy
automatically cost anything against naming the same strategy by hand? Thresholds are
preregistered in a separate file that this run reads and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import mlx.core as mx
import mlx_lm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ironmule import service_strategy as ss  # noqa: E402
from ironmule.hw import STORE, fingerprint  # noqa: E402
from ironmule.plans import StrictOneShotPlan  # noqa: E402
from ironmule.runtime import BASELINE  # noqa: E402
from ironmule.service import (  # noqa: E402
    AutomaticMode,
    InteractiveMode,
    PairedThroughputMode,
    Request,
    Runtime,
    ThroughputMode,
    paired_status,
)
from ironmule.tune import (  # noqa: E402
    conditions,
    gpu_busy,
    load_profile,
    resolve_local_model,
    save_profile,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = Path.home() / ".ironmule"
# The files whose content decides what this run measures. Hashed into the record so a
# result always names the code it came from, which a git revision cannot do while the
# tree is uncommitted.
MEASURED_SOURCES = (
    "ironmule/service.py", "ironmule/service_strategy.py", "ironmule/paired_research.py",
    "ironmule/qmv_shared.py", "ironmule/qmv_k3840.py", "ironmule/executor.py",
    "ironmule/runtime.py", "ironmule/tune.py", "tools/b52_automatic_selection.py",
)


def source_binding() -> dict[str, str]:
    """One digest per measured file, plus one over all of them in order."""

    digests = {}
    running = hashlib.sha256()
    for name in MEASURED_SOURCES:
        data = (PROJECT_ROOT / name).read_bytes()
        digests[name] = hashlib.sha256(data).hexdigest()
        running.update(name.encode())
        running.update(data)
    return {"files": digests, "combined_sha256": running.hexdigest()}


def write_once(path: Path, payload: str) -> None:
    """Raw evidence is written once, atomically, and never overwritten."""

    if path.exists():
        raise SystemExit(f"refusing to overwrite existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)

# Preregistered before the timing run; see `--preregister`.
PREREGISTRATION = {
    "question": (
        "does choosing the qualified strategy from the profile cost anything against "
        "naming the same strategy by hand"
    ),
    "arms": ["M_manual_paired", "M_manual_paired_aa", "A_automatic"],
    "blocks": 8,
    "workload": "two ready requests, 24 requested tokens each, one model, one process",
    "statistic": "per-block ratio, median, 95 per cent bootstrap over 10000 resamples",
    "equivalence_margin": 0.02,
    "decision_rule": (
        "EQUIVALENT only if the whole 95 per cent interval of A_automatic over "
        "M_manual_paired lies inside 0.98 to 1.02, and the A/A control interval "
        "contains 1.0. An interval that merely contains 1.0 is not equivalence."
    ),
    "blocking": "any swapout, any fallback, or a failed A/A control blocks the verdict",
    "not_measured": (
        "the paired path's own gain: that is B46 and B50 and is not re-derived here"
    ),
    "user_path_cases": [
        "one request, two ready requests, four ready requests, all decided on their own",
        "two requests where the partner has not arrived yet: the established mode, "
        "because readiness is counted rather than group size",
        "a record naming a foreign library build: refused",
        "assigning an explicit mode again: back to the established behaviour",
    ],
}

PROMPTS = (
    "Explain in three sentences why unified memory changes how a laptop runs a language model.",
    "Describe how a key value cache grows during autoregressive decoding.",
    "Summarise what a tokenizer does before a model sees any text.",
    "State one reason a quantised model reads fewer bytes per token.",
)
EVIDENCE_RUN_IDS = [
    "B45_shared_weight_verdict_20260909",
    "B46_paired_product_verdict_20260909",
    "B47U_opt_in_release_20260909",
    "B50_paired_deployment_guide_20260909",
    "B51_identity_gap_20260909",
]
CORRECTNESS_CONTRACT = (
    "per request, logit bit patterns per step, the full KV state, tokens, stop reason "
    "and step count are identical to an independent library run; cancellation mid-flight "
    "is not covered because the served interface carries no cancel handle"
)


def _vm() -> dict:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        match = re.match(r'"?([A-Za-z ,\-]+)"?:\s+(\d+)', line.strip())
        if match:
            values[match.group(1).strip()] = int(match.group(2))
    return {k: values[k] for k in ("Pageins", "Pageouts", "Swapins", "Swapouts") if k in values}


def _shape(results) -> list[dict]:
    return [{"tokens": list(r.tokens), "stop_reason": r.stop_reason,
             "text_sha256": hashlib.sha256(r.text.encode()).hexdigest()} for r in results]


def _requests(runtime, count: int, max_tokens: int,
              arrivals: tuple[float, ...] = ()) -> list[Request]:
    stamps = arrivals or (0.0,) * count
    return [Request(prompt_ids=runtime.encode(p), max_tokens=max_tokens,
                    plan=StrictOneShotPlan(), arrival_ms=arrival)
            for p, arrival in zip(PROMPTS[:count], stamps)]


# -- part one: migration on a working copy ------------------------------------

def migrate(model_id: str) -> dict:
    """Write the record into a working-copy profile through the authorised path."""

    resolved = resolve_local_model(model_id)
    identity = resolved.identity
    steps: dict = {"store": str(STORE), "default_store_exists": DEFAULT_STORE.exists()}

    # A profile carrying only what a profile must carry: this machine, this model, the
    # baseline knobs. No timing metric is invented to fill a field.
    base = {
        "conditions": conditions(model_id, 64, 24, model_identity=identity),
        "fingerprint": fingerprint(),
        "model_id": identity.model_id,
        "model_identity": identity.to_dict(),
        "knobs": BASELINE.as_dict(),
    }
    save_profile(base)
    before = load_profile(model_id, model_identity=identity)
    steps["profile_without_record_loads"] = before is not None
    steps["record_absent_before"] = ss.read_record(before) is None
    steps["selects_nothing_before"] = ss.select(
        before, opt_in=True, ready_requests=2,
        identity_sha256=identity.identity_sha256, fingerprint=fingerprint(),
        mlx=mx.__version__, mlx_lm=mlx_lm.__version__)["strategy"] == ss.STRATEGY_THROUGHPUT

    record = ss.build_record(
        strategy=ss.STRATEGY_PAIRED, min_ready=2, max_ready=4,
        identity_sha256=identity.identity_sha256, fingerprint=fingerprint(),
        mlx=mx.__version__, mlx_lm=mlx_lm.__version__,
        correctness_contract=CORRECTNESS_CONTRACT, evidence_run_ids=EVIDENCE_RUN_IDS,
    )
    save_profile(ss.attach(before, record))
    after = load_profile(model_id, model_identity=identity)
    steps["profile_with_record_loads"] = after is not None
    steps["record_present_after"] = ss.read_record(after) == record
    steps["knobs_unchanged"] = after["knobs"] == before["knobs"]
    steps["no_opt_in_still_selects_nothing"] = ss.select(
        after, opt_in=False, ready_requests=2,
        identity_sha256=identity.identity_sha256, fingerprint=fingerprint(),
        mlx=mx.__version__, mlx_lm=mlx_lm.__version__)["strategy"] == ss.STRATEGY_THROUGHPUT
    steps["default_store_untouched"] = DEFAULT_STORE.exists() == steps["default_store_exists"]
    steps["record"] = record
    steps["clean"] = all(steps[key] for key in (
        "profile_without_record_loads", "record_absent_before", "selects_nothing_before",
        "profile_with_record_loads", "record_present_after", "knobs_unchanged",
        "no_opt_in_still_selects_nothing", "default_store_untouched"))
    return steps


# -- part two: the user path ---------------------------------------------------

def user_path(model_id: str, max_tokens: int) -> tuple[dict, Runtime]:
    walk: dict = {}

    plain = Runtime.load(model_id, use_tuned_profile=True)
    try:
        walk["load_without_opt_in_mode"] = plain.mode.name
        walk["load_without_opt_in_status"] = paired_status(plain.mode)
        walk["without_opt_in_is_off"] = paired_status(plain.mode)["enabled"] is False
    finally:
        plain.close()

    runtime = Runtime.load(model_id, automatic_service_mode=True)
    mode = runtime.mode
    walk["load_with_opt_in_mode"] = mode.name
    walk["status_before_serving"] = mode.status()

    truth = {}
    for count in (1, 2, 4):
        runtime.mode = InteractiveMode()
        truth[count] = _shape(runtime.serve(_requests(runtime, count, max_tokens)))

    served = {}
    for count in (1, 2, 4):
        runtime.mode = mode
        results = runtime.serve(_requests(runtime, count, max_tokens))
        served[count] = {
            "identical": [a == b for a, b in zip(truth[count], _shape(results))],
            "status": mode.status(),
        }
    walk["served"] = served
    walk["single_stays_on_the_established_mode"] = served[1]["status"]["strategy"] == "throughput"
    walk["pair_selects_paired"] = served[2]["status"]["strategy"] == ss.STRATEGY_PAIRED
    walk["quad_selects_paired"] = served[4]["status"]["strategy"] == ss.STRATEGY_PAIRED
    walk["pair_actually_shared_steps"] = served[2]["status"]["paired_steps"] > 0
    walk["status_names_strategy_reason_and_evidence"] = all(
        served[2]["status"].get(key) for key in ("strategy", "reason", "evidence_run_ids"))
    walk["outputs_identical"] = all(all(row["identical"]) for row in served.values())
    walk["decisions_equal_serves"] = mode.decisions_made == 3

    # A partner that has not arrived yet is not a partner. The group holds two
    # requests, only one of them is ready, and the established mode is the answer.
    runtime.mode = mode
    staggered = runtime.serve(_requests(runtime, 2, max_tokens, arrivals=(0.0, 120.0)))
    walk["staggered_status"] = mode.status()
    walk["staggered_counts_readiness_not_group_size"] = (
        mode.status()["strategy"] == "throughput"
        and mode.status()["ready_requests"] == 1
        and mode.status()["group_requests"] == 2
    )
    walk["staggered_still_correct"] = _shape(staggered) == truth[2]

    # An unknown configuration: the same record, one library version away from here.
    unknown = json.loads(json.dumps(mode.profile))
    unknown["service_strategy"]["admitted_range"]["mlx"] = "0.0.0-not-this-build"
    refused = AutomaticMode(unknown, opt_in=True,
                            identity_sha256=runtime.model_identity.identity_sha256,
                            fingerprint=fingerprint(),
                            mlx=mx.__version__, mlx_lm=mlx_lm.__version__)
    runtime.mode = refused
    refused_results = _shape(runtime.serve(_requests(runtime, 2, max_tokens)))
    walk["unknown_configuration_refused"] = refused.status()["strategy"] == "throughput"
    walk["unknown_configuration_reason"] = refused.status()["reason"]
    walk["unknown_configuration_still_correct"] = refused_results == truth[2]

    # Back to the established mode, by naming it.
    runtime.mode = ThroughputMode()
    walk["after_switching_back"] = paired_status(runtime.mode)
    walk["switching_back_is_off"] = paired_status(runtime.mode)["enabled"] is False

    walk["clean"] = all(walk[key] for key in (
        "without_opt_in_is_off", "single_stays_on_the_established_mode",
        "pair_selects_paired", "quad_selects_paired", "pair_actually_shared_steps",
        "status_names_strategy_reason_and_evidence", "outputs_identical",
        "decisions_equal_serves", "unknown_configuration_refused",
        "unknown_configuration_still_correct", "switching_back_is_off",
        "staggered_counts_readiness_not_group_size", "staggered_still_correct"))
    return walk, runtime


# -- part three: what the automatic choice costs -------------------------------

def cost(runtime: Runtime, profile: dict, blocks: int, max_tokens: int) -> dict:
    identity = runtime.model_identity
    modes = {
        "M_manual_paired": PairedThroughputMode(),
        "M_manual_paired_aa": PairedThroughputMode(),
        "A_automatic": AutomaticMode(profile, opt_in=True,
                                     identity_sha256=identity.identity_sha256,
                                     fingerprint=fingerprint(),
                                     mlx=mx.__version__, mlx_lm=mlx_lm.__version__),
    }
    arms = list(modes)

    def one_round(name):
        runtime.mode = modes[name]
        requests = _requests(runtime, 2, max_tokens)
        began = time.perf_counter_ns()
        results = runtime.serve(requests)
        return (time.perf_counter_ns() - began,
                any(r.metrics["fell_back"] for r in results))

    for name in arms:                                   # warm every arm
        one_round(name)

    before = _vm()
    samples = {name: [] for name in arms}
    fell_back = False
    for block in range(blocks):
        rotation = arms[block % len(arms):] + arms[: block % len(arms)]
        for name in rotation:
            elapsed, fallback = one_round(name)
            samples[name].append(elapsed)
            fell_back = fell_back or fallback
    after = _vm()
    paging = {key: after.get(key, 0) - value for key, value in before.items()}

    def interval(numerator, denominator):
        ratios = [a / b for a, b in zip(samples[numerator], samples[denominator])]
        rng = random.Random(20260909)
        draws = sorted(median([ratios[rng.randrange(len(ratios))] for _ in ratios])
                       for _ in range(10000))
        return {"median": median(ratios), "min": min(ratios), "max": max(ratios),
                "ci95_low": draws[int(0.025 * len(draws))],
                "ci95_high": draws[int(0.975 * len(draws)) - 1]}

    margin = PREREGISTRATION["equivalence_margin"]
    automatic = interval("A_automatic", "M_manual_paired")
    control = interval("M_manual_paired_aa", "M_manual_paired")
    return {
        "samples_ns": samples,
        "automatic_over_manual": automatic,
        "aa_control": control,
        "paging": paging,
        "fell_back": fell_back,
        "aa_control_passes": control["ci95_low"] <= 1.0 <= control["ci95_high"],
        "equivalent": (1 - margin) <= automatic["ci95_low"]
        and automatic["ci95_high"] <= (1 + margin),
        "decisions_made": modes["A_automatic"].decisions_made,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=PREREGISTRATION["blocks"])
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregister", action="store_true",
                        help="write the preregistration and exit, before any measurement")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.preregister:
        write_once(args.preregistration,
                   json.dumps({"schema": "ironmule.automatic_selection_preregistration.v1",
                               "written_at": datetime.now(timezone.utc).isoformat(),
                               **PREREGISTRATION}, indent=2, sort_keys=True))
        print(f"preregistered: {args.preregistration}")
        return 0

    if args.out is None:
        parser.error("--out is required for a measuring run")
    if not args.preregistration.is_file():
        parser.error("the preregistration must be written before the run")
    if args.out.exists():
        parser.error(f"{args.out} already exists; every run gets its own id")
    if STORE == DEFAULT_STORE:
        parser.error("run against a working copy: set IRONMULE_HOME to a temporary store")
    busy = gpu_busy()
    if busy:
        parser.error(f"another process is on the GPU: {busy}")

    prereg_bytes = args.preregistration.read_bytes()
    migration = migrate(args.model)
    if not migration["clean"]:
        write_once(args.out, json.dumps({"state": "BLOCKED", "migration": migration},
                                        indent=2, sort_keys=True))
        print(json.dumps({"state": "BLOCKED", "stage": "migration"}, sort_keys=True))
        return 1

    load_began = time.perf_counter_ns()
    walk, runtime = user_path(args.model, args.max_tokens)
    load_and_walk_ns = time.perf_counter_ns() - load_began
    try:
        print(json.dumps({"migration_clean": migration["clean"],
                          "user_path_clean": walk["clean"]}, sort_keys=True))
        timing = cost(runtime, load_profile(args.model), args.blocks, args.max_tokens)
    finally:
        runtime.close()

    blocked = bool(timing["paging"].get("Swapouts", 0)) or timing["fell_back"]
    if blocked:
        verdict = "BLOCKED"
    elif not (migration["clean"] and walk["clean"]):
        verdict = "NO-GO"
    elif not timing["aa_control_passes"]:
        verdict = "UNCLEAR"
    elif not timing["equivalent"]:
        verdict = "NO-GO"
    else:
        verdict = "READY"

    record = {
        "schema": "ironmule.automatic_service_selection.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "product_release_check",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "preregistration": json.loads(prereg_bytes),
        "preregistration_sha256": hashlib.sha256(prereg_bytes).hexdigest(),
        "source_binding": source_binding(),
        "working_copy_store": str(STORE),
        "migration": migration,
        "user_path": walk,
        "cost": timing,
        "verdict": verdict,
        "default_off": "a load without automatic_service_mode never reaches the record",
        "load_and_user_path_ns": load_and_walk_ns,
        "mlx_version": mx.__version__,
        "mlx_lm_version": mlx_lm.__version__,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"verdict": verdict,
                      "automatic_over_manual": timing["automatic_over_manual"],
                      "aa_control": timing["aa_control"],
                      "paging": timing["paging"]}, indent=2, sort_keys=True))
    return 0 if verdict == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
