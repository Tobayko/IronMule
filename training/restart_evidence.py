"""Capture and verify portable receipts from one source-bound Qwen restart experiment."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import textwrap
import time
from types import SimpleNamespace

from lab.pk1 import restart_order, restart_trial
from training.code_author import read_request, read_response, write_once
from training.code_trial import PAIRS, read, save
from training.native_evidence import COST_JOURNAL, ROOT, bounded, patch, persist_assets, sha, source_at, verify_manifest

SCHEMA = "ironmule.portable-qwen-restart.v1"
TRIAL = ("context.json", "aa-0.json", "aa-1.json", "stale.json", "missing.json", "receipt.json", "stale-receipt.json",
         *(f"{tag}-{i}.json" for i in range(PAIRS) for tag in ("ref", "cand")),
         *(f"facts-{tag}-{i}.json" for i in range(2) for tag in ("ref", "cand")), "result.json")
RELEASE = ("release-activate.json", "release-run.json", "release-exercise.json", "target-check.json", "rollback-exercise.json", "automatic-withdrawal.json")
INCIDENT = ("previous-selection.json", "selection.json", "result.json", "broker.log", "restarted-reference.json", "withdrawn-reference-arm.json")
PILOT_ARMS = ("pilot-initial.json", "pilot-trained.json")
RECOVERY = ("coordinator-restart-intent.json", "coordinator-restart.json")
# Store benchmark and HTTP receipts under separate names; both protocols use decision/gate files.
HTTP_ARMS = tuple(f"{arm}-{seed}.json" for seed in range(95111, 95117) for arm in ("reference", "candidate"))
HTTP_FILES = (*HTTP_ARMS, "gate-reference.json", "gate-candidate.json", "gate-devices-reference.json", "gate-devices-candidate.json", "trained-head.json", "geometry.json", "decision.json", "withdrawn.json",
              "product-qualify.json", "product-exercise.json", "product-serve.json", "product-client.json", "product-listener.json", "live-ready.json")
COST_PHASES = ("during-qualification", "after-verdict", "after-release", "after-product")
COST_FILES = tuple(f"cost-{p}.json" for p in COST_PHASES)
PLANS = ("replan-after-product.json", "replan-after-product-declined.json")
OBS = [f"observation-{i}{suffix}.json" for i in range(2) for suffix in ("", ".profile")]
REQUIRED = {"order.json", "reference-source.txt", *OBS}
NAMES = {"product-tokenizer-diagnostic.json", *REQUIRED, *COST_FILES, *RECOVERY, "startup-fit.json", "startup-head.npz", "request.json", "response.json", "authorship.json", "candidate.diff", "campaign.json",
         "qualified.json", "control.json", *PLANS, *("incident-" + n for n in INCIDENT), "pilot-comparison.json", *PILOT_ARMS, *TRIAL, *RELEASE, *("http-" + n for n in HTTP_FILES)}


def manifest(path):
    return verify_manifest(path, schema=SCHEMA, names=NAMES, required=REQUIRED)


def candidate_bytes(source, request, order, proposal):
    node = restart_order.site(source)
    lines = source.splitlines(keepends=True)
    original = textwrap.dedent("".join(lines[node.lineno - 1:node.end_lineno]))
    if request["source"] != original or request["order"] != order or order["target"]["line"] != node.lineno:
        raise ValueError("worker request differs from the bound startup function")
    edited = textwrap.indent(textwrap.dedent(proposal["definition"]).strip(), " " * node.col_offset)
    return ("".join(lines[:node.lineno - 1]) + edited + "\n" + "".join(lines[node.end_lineno:])).encode()


def observe_cost(state, phase, *, product=None):
    from lab.rp import rp

    path = (product or state) / f"cost-{phase}.json"
    if path.is_file():
        return
    campaign = read(state / "campaign.json")
    if campaign["config_sha256"] != sha(bounded(state / "target.json")):
        raise ValueError("cost snapshot requires the exact target configuration")
    target = read(state / "target.json")
    if (phase == "after-verdict" and not (state / "trial/result.json").is_file()
            or phase == "after-release" and not (state / "release/release-run.json").is_file()
            or phase == "after-product" and not ((product or state / "product") / "product-qualify.json").is_file()):
        raise ValueError("requested campaign cost phase has not occurred")
    with rp.locked(rp.STATE):
        ledger = rp.load()
        pod = ledger.get("pod")
        if not pod or pod["id"] != target["pod_id"]:
            raise ValueError("startup campaign no longer owns the observed lease")
        user = rp.account()
        matches = [item for item in user["pods"] if (item["id"], item["name"]) == (pod["id"], pod["name"])]
        if len(matches) != 1:
            raise ValueError("provider does not confirm the registered lease")
        now = time.time()
        elapsed = max(0, now - pod["started"])
        hourly = rp.money(matches[0]["costPerHr"]) + rp.money(pod["disk_rate"])
        arms = {}
        for label in TRIAL:
            p = state / "trial" / label
            if p.is_file():
                row = read(p)
                if "broker_elapsed_s" in row:
                    arms[label] = {"sha256": sha(bounded(p)), "broker_elapsed_s": row["broker_elapsed_s"]}
        write_once(path, {"order_id": campaign["order"], "phase": phase, "observed_unix_s": now,
                   "lease_elapsed_s": elapsed, "quoted_compute_usd_per_h": matches[0]["costPerHr"],
                   "disk_rate_estimate_usd_per_h": pod["disk_rate"], "estimated_lease_usd_before_tail": elapsed * hourly / 3600,
                   "campaign_reserved_usd": rp.held(ledger), "retained_tail_seconds": rp.CLEANUP_SECONDS,
                   "provider_account_delta_since_lease_start_usd": pod["balance_before"] - user["clientBalance"],
                   "completed_worker_arms": arms, "cost_journal": COST_JOURNAL, **({"product_attempt": product.name} if product else {}),
                   "basis": "A100 lease-to-date covers all work, waiting and concurrent CPU experiments. "
                            "Provider account delta can include other use and lag charges. Per-arm times are measured durations."})


def checkpoint_assets(state, order, verified, assets):
    key = order["order_id"]
    from training.pilot_restart_profiles import binding as head_binding
    weights = bounded(state / "startup-head.npz")
    fit = read(state / "startup-fit.json")
    def times(prefix, i):
        rows = [json.loads(assets[f"{prefix}{tag}-{i}.json"]) for tag in ("ref", "cand")]
        return [r["factual_first"][0]["completed_s"] if prefix else r["cold_s"] for r in rows]
    if (len(weights) > 1 << 20 or fit["schema"] != "ironmule.restart-pilot.v1" or fit["weights_sha256"] != sha(weights)
            or fit["binding"] != head_binding(order, verified) or fit["order"] != key
            or fit["trial_result_sha256"] != sha(assets["result.json"])
            or fit["training_pairs_s"] != [times("", i) for i in range(PAIRS)]
            or fit["holdout_pairs_s"] != [times("facts-", i) for i in range(2)]):
        raise ValueError("startup head is not bound to the measured training and held-out families")
    assets["startup-fit.json"] = bounded(state / "startup-fit.json")
    assets["startup-head.npz"] = weights


def tokenizer_diagnosis(state, order, assets):
    source = state / "product-nested-diagnostic.json"
    if not source.is_file():
        return
    data = bounded(source)
    report = json.loads(data)
    prior = json.loads(assets["http-gate-reference.json"])
    child = report["nested_diagnostic"]["tokenizer_child"]
    if (report["order_id"] != order["order_id"] or report["source_sha256"] != prior["source_sha256"]
            or report["gguf_sha256"] != order["binding"]["gguf"] or report["gate_diagnostic"] != prior["diagnostic"]
            or report["ids_count"] != 0 or child["returncode"] == 0 or child["ids_count"] != 0
            or "import mlx.core as mx" not in child["stderr"] or "ImportError: cudaMallocManaged(" not in child["stderr"]):
        raise ValueError("nested reference failure differs from the initially diagnosed tokenizer gate")
    assets["product-tokenizer-diagnostic.json"] = data


def capture_incident(state, assets, order, verified):
    folder = state / "release/automatic-withdrawal-incident"
    if not (folder / "result.json").is_file():
        return
    raw = {n: bounded(folder / n) for n in INCIDENT if (folder / n).is_file()}
    verify_incident(raw, sha(assets["control.json"]), order, verified)
    assets.update({"incident-" + n: data for n, data in raw.items()})


def verify_incident(raw, control_digest, order, verified):
    prior, withdrawn, failure = (json.loads(raw[n]) for n in INCIDENT[:3])
    if (prior["receipt_sha256"] != control_digest or prior["arm"] != "candidate" or prior["withdrawn"]
            or failure["original_order"] != order["order_id"] or failure["passed"] is not False or failure["incumbent_returned_work"] is not False
            or withdrawn != failure["automatic_withdrawal"] or withdrawn["arm"] != "reference" or withdrawn["withdrawn"] is not True
            or failure["observed_error"].encode() not in raw["broker.log"]):
        raise ValueError("incompatible cache incident differs from the bound release or observed failure")
    if "restarted-reference.json" in raw and "withdrawn-reference-arm.json" in raw:
        recovered, row = (json.loads(raw[n]) for n in INCIDENT[-2:])
        work = recovered["result"]
        restart_trial.validate(row, work["seed"], expected_source=verified["reference_sha256"])
        restart_trial.verify_binding(row, order)
        from training.code_author import digest
        actual = digest([row["cold"]["output_sha256"], row["warm"]["output_sha256"], row["factual"]])
        if (recovered["passed"] is not True or recovered["selection_after"] != withdrawn or recovered["receipt_sha256"] != prior["receipt_sha256"]
                or recovered["frozen_evaluator_sha256"] != sha(bounded(Path(restart_trial.__file__))) or row["reused"] or work["arm"] != "reference"
                or work["cache_used"] or work["work_sha256"] != actual or work["source_sha256"] != row["source_sha256"]):
            raise ValueError("restarted incumbent recovery has different source, cache, output or selection")
        if (work["graded_correct"], work["cold_performance_completions"], work["warm_performance_completions"],
                work["pilot_decisions"]) != (sum(row["facts"]), row["cold"]["completed"], row["warm"]["completed"], row["pilot_decisions"]):
            raise ValueError("recovered work counts differ from raw worker data")
        return work
    return None


def verify_withdrawal(assets, order, verified):
    row = json.loads(assets["automatic-withdrawal.json"])
    first, second = row["incompatible"], row["restarted_with_fresh_cache"]
    actual = [r["result"] for r in (first, second)]
    if (row["schema"] != "ironmule.qwen-automatic-withdrawal.v1" or row["passed"] is not True or row["order"] != order["order_id"]
            or row["receipt_sha256"] != sha(assets["control.json"]) or row["valid_cache_sha256"] != sha(assets["receipt.json"])
            or row["prior"]["arm"] != "candidate" or row["withdrawal"]["arm"] != "reference" or not row["withdrawal"]["withdrawn"]
            or row["main_selection"]["arm"] != "candidate" or row["main_selection"]["withdrawn"]
            or first["pid"] == second["pid"] or actual[0]["work_sha256"] != actual[1]["work_sha256"]
            or any(r["arm"] != "reference" or r["source_sha256"] != verified["reference_sha256"] or r["cache_used"]
                   or (r["graded_correct"], r["cold_performance_completions"], r["warm_performance_completions"]) != (18, 32, 32)
                   for r in actual)):
        raise ValueError("automatic withdrawal evidence differs from bound source, scope or completed work")


def pilot_serving_assets(state, verified, assets):
    from lab.rp import restart_pilot_campaign
    from training import scoped_release

    pin = sha(assets["control.json"])
    receipt = scoped_release.pinned(state / "release", pin)
    if receipt["qualification"] != verified:
        raise ValueError("pilot serving source differs from verified evaluation")
    folder = state / "native-pilot"
    for n in PILOT_ARMS:
        if (folder / n).is_file():
            assets[n] = bounded(folder / n)
    if (folder / "comparison.json").is_file():
        expected = restart_pilot_campaign.assess([json.loads(assets[n]) for n in PILOT_ARMS], receipt, pin, sha(assets["startup-head.npz"]))
        comparison = bounded(folder / "comparison.json")
        if expected != json.loads(comparison):
            raise ValueError("trained pilot serving comparison differs from the source-bound work")
        assets["pilot-comparison.json"] = comparison


def product_assets(state, order, verified, assets, *, attempt=None, checkout=None):
    """Bind held-out product measurements to the independently approved source and deployed scratch head."""
    from training import scoped_release

    work = attempt or state / "product"
    raw = {n: bounded(work / n) for n in HTTP_FILES if (work / n).is_file()}
    if not raw:
        return
    control = scoped_release.pinned(state / "release", sha(assets["control.json"]))
    if control["order"] != order or control["qualification"] != verified:
        raise ValueError("HTTP evidence belongs to a different release")
    if attempt is not None and raw:
        from lab.rp.restart_campaign import tokenizer_retry

        proof = work / "tokenizer-preflight.json"
        tokenizer_retry(state, SimpleNamespace(checkout=checkout), control, proof)
        data = bounded(proof)
        if "geometry.json" in raw and json.loads(raw["geometry.json"]).get("runtime_preflight_sha256") != sha(data):
            raise ValueError("retry geometry differs from the validated tokenizer environment")
        assets["tokenizer-preflight.json"] = data
    private = work / "guarded-child-raw.local.jsonl"
    if private.is_file():
        original = bounded(private)
        logs = [json.loads(line) for line in original.splitlines()]
        row, = [r for r in logs if r["stage"] == "reference" and r["status"] == -6]
        trace = row["stderr"]
        devices = json.loads(raw["gate-devices-reference.json"])
        held = next(d for d in devices if d["stage"] == "GATE_CANDIDATE")
        if (held["child_status"] != -6 or held["provider"] != "cuda" or "offloaded 65/65 layers to GPU" not in trace
                or "CUDA error: invalid configuration argument" not in trace or "ggml_cuda_mul_mat_batched_cublas_impl" not in trace
                or "ggml-cuda.cu:2077" not in trace or json.loads(raw["gate-reference.json"])["passes"] is not False):
            raise ValueError("restricted stock CUDA failure differs from source-bound gate evidence")
        public = {"schema": "ironmule.stock-product-reference-failure.v1", "private_raw_sha256": sha(original),
                  "reference_sha256": verified["reference_sha256"], "gate_device_receipt_sha256": sha(raw["gate-devices-reference.json"]),
                  "stage": held["stage"], "device": held["device"], "requested_gpu_layers": held["gpu_layers"],
                  "actual_offloaded_layers": 65, "model_layers": 65, "exit_status": row["status"],
                  "error": "CUDA error: invalid configuration argument", "function": "ggml_cuda_mul_mat_batched_cublas_impl",
                  "location": "ggml-cuda.cu:2077"}
        assets["stock-reference-diagnostic.json"] = json.dumps(public, sort_keys=True, indent=2).encode() + b"\n"
    head = sha(assets["startup-head.npz"])
    for name in HTTP_ARMS:
        if name not in raw:
            continue
        row = json.loads(raw[name])
        arm, number = name.removesuffix(".json").split("-")
        if (row["seed"] != int(number) or row["source_sha256"] != verified[arm + "_sha256"]
                or row["startup_decision"]["weights_sha256"] != head):
            raise ValueError("unbound ordinary product arm")
    for n in ("decision.json", "product-qualify.json"):
        if n not in raw:
            continue
        decision = json.loads(raw[n])
        if type(decision.get("passed")) is not bool:
            raise ValueError("incomplete product verdict")
        if decision["passed"] is True or "integrity" in decision:
            from lab.pk1 import restart_http
            if restart_http.assess(work, control) != decision:
                raise ValueError("product decision differs from independently graded raw HTTP pairs")
    if "product-qualify.json" in raw and json.loads(raw["product-qualify.json"]) != json.loads(raw["decision.json"]):
        raise ValueError("product qualification differs from target decision")
    if "product-serve.json" in raw:
        from lab.pk1 import restart_live

        listener, witness, ready = (json.loads(raw[n]) for n in ("product-listener.json", "product-client.json", "live-ready.json"))
        if (ready["receipt_sha256"] != sha(assets["control.json"]) or witness["listener_pid"] != ready["pid"]
                or witness["nonce"] != ready["nonce"] or ready["geometry_sha256"] != sha(raw["geometry.json"])
                or ready["ready_ns"] > witness["started_ns"] or restart_live.verify(listener, witness) != json.loads(raw["product-serve.json"])):
            raise ValueError("independent client or passive listener differs from the claimed delivered work")
    for name in ("product-serve.json", "product-exercise.json"):
        if name not in raw:
            continue
        stage = json.loads(raw[name])
        rows = stage["completions"] if name == "product-serve.json" else [stage["before"], stage["after"]]
        for row in rows:
            arm = row["arm"]
            if (arm not in ("reference", "candidate") or row["source_sha256"] != verified[arm + "_sha256"]
                    or row["startup_decision"]["weights_sha256"] != head or row["startup_decision"]["profile"] != arm):
                raise ValueError("unbound product action")
        if name == "product-exercise.json" and stage["passed"] and not (
                rows[0]["arm"] == "candidate" and rows[1]["arm"] == "reference"
                and rows[0]["work_sha256"] == rows[1]["work_sha256"] and all(row["graded_correct"] == 18 for row in rows)):
            raise ValueError("actual product withdrawal or work differs")
    assets.update({"http-" + name: data for name, data in raw.items()})
    for name in PLANS:
        if ((attempt or state) / name).is_file():
            assets[name] = bound_plan((attempt or state) / name, order, assets)


def bound_plan(path, order, assets):
    from training.code_author import digest
    from training.restart_replan import past_attempts, completed_phase_probe

    data = bounded(path)
    plan = json.loads(data)
    receipts = {"source_receipt": "control.json", "cost_observation_sha256": "cost-after-verdict.json"}
    known = past_attempts(ROOT / "lab/autonomous/results", order)
    if path.name == PLANS[0]:
        receipts.update({"cpu_head": "startup-head.npz", "grade_sha256": "http-product-qualify.json",
                         "served_sha256": "http-product-serve.json", "rollback_sha256": "http-product-exercise.json",
                         "counterbalanced_native_sha256": "result.json"})
        match = plan["schema"] == "ironmule.restart-replan.v1" and plan["negative_history"] == known
        if "completed_phase_probe" in plan:
            probe = completed_phase_probe(ROOT / "lab/autonomous/results", order, json.loads(assets["control.json"]),
                                          sha(assets["startup-head.npz"]), sha(assets["control.json"]))
            match = match and probe is not None and plan["completed_phase_probe"] == probe
    else:
        receipts.update({"native_result_sha256": "result.json", "product_verdict_sha256": "http-product-qualify.json"})
        observed = {n: sha(assets["http-" + n]) for n in HTTP_ARMS if "http-" + n in assets}
        match = (plan["schema"] == "ironmule.restart-replan-declined.v1" and plan["retained_measured_history"] == known
                 and plan["completed_raw_files"] == observed and json.loads(assets["http-product-qualify.json"])["passed"] is False
                 and plan["new_serving_improvement"] is None)
    if (not match or plan["order"] != order["order_id"] or plan["plan_id"] != digest({k: v for k, v in plan.items() if k != "plan_id"})
            or any(plan[k] != sha(assets[name]) for k, name in receipts.items())):
        raise ValueError("post-product forecast differs from measured deployment and negative history")
    return data


def export(args):
    state = args.state
    raw_order = bounded(state / "order.json")
    order = json.loads(raw_order)
    key = order["order_id"]
    if (order["schema"] != restart_order.SCHEMA or key != restart_order.order_digest(order)
            or order["allowed_paths"] != [restart_order.PATH] or not re.fullmatch("[a-f0-9]{40}", order["base_commit"])):
        raise ValueError("invalid startup source order")
    folder = args.destination / key
    original = source_at(order["base_commit"], restart_order.PATH)
    if original is None:
        path = folder / "reference-source.txt"
        original = bounded(args.reference or (path if path.is_file() else ROOT / restart_order.PATH))
    if sha(original) != order["source_sha256"]:
        raise ValueError("startup baseline source mismatch")
    evidence = order["evidence"]
    if evidence["historical_artifact"] != restart_order.ARCHIVE or sha(bounded(ROOT / restart_order.ARCHIVE)) != evidence["historical_sha256"]:
        raise ValueError("historical startup traces changed")
    assets = {"order.json": raw_order, "reference-source.txt": original}
    for i, expected in enumerate(evidence["local_profile_ids"]):
        path = args.observations[i] if args.observations else folder / f"observation-{i}.json"
        raw = bounded(path)
        profile = bounded(path.with_suffix(".profile.json"))
        observed = json.loads(raw)
        if (restart_order.order_digest(observed) != expected or observed["order_id"] != expected
                or observed["profile_sha256"] != sha(profile) or json.loads(profile)["binding"] != observed["binding"]
                or observed["binding"]["mask"] != evidence["local_mask_measurements"][i]):
            raise ValueError("local mask observations do not match the startup order")
        assets.update({f"observation-{i}.json": raw, f"observation-{i}.profile.json": profile})
    candidatesrc = None
    author = state / "authoring"
    inbox = author / "author"
    if (inbox / "request.json").is_file():
        assets["request.json"] = bounded(inbox / "request.json")
        request_id, request = read_request(inbox)
        if request["order"] != order:
            raise ValueError("startup worker request belongs to another order")
        if (inbox / "response.json").is_file():
            assets["response.json"] = bounded(inbox / "response.json")
            reply = read_response(inbox, request_id)
            candidatesrc = candidate_bytes(original.decode(), request, order, reply)
            compile(candidatesrc, restart_order.PATH, "exec")
        if (author / "candidate.py").exists():
            assets["authorship.json"] = bounded(author / "authorship.json")
            worker = json.loads(assets["authorship.json"])
            if (candidatesrc is None or bounded(author / "candidate.py") != candidatesrc
                    or worker["source_sha256"] != sha(candidatesrc) or worker["order"] != key
                    or worker["rationale"] != reply["rationale"]):
                raise ValueError("actual startup candidate differs from the returned function")
            assets["candidate.diff"] = patch(original, candidatesrc, restart_order.PATH)
    cost_phase = getattr(args, "cost_phase", None)
    selected_attempt = getattr(args, "product_attempt", None)
    if selected_attempt and not re.fullmatch(r"product-v[2-9][0-9]*|product-v1[0-9]+", selected_attempt):
        raise ValueError("invalid product attempt")
    if cost_phase:
        if selected_attempt and cost_phase != "after-product":
            raise ValueError("product attempt cost needs its product verdict")
        observe_cost(state, cost_phase, product=state / selected_attempt if selected_attempt else None)
    paths = {**{n: state / n for n in (*COST_FILES, *RECOVERY)}, **{n: state / "trial" / n for n in TRIAL}, **{n: state / "release" / n for n in RELEASE},
             "control.json": state / "release/deployment/control.json", "qualified.json": state / "qualified.json",
             "campaign.json": state / "campaign.json"}
    assets.update({n: bounded(p) for n, p in paths.items() if p.is_file()})
    if RECOVERY[1] in assets:
        before, after = (json.loads(assets[n]) for n in RECOVERY)
        hashes = before["receipts"]
        if (before["remote_job"] != "restart-" + key[:18] or before["remote_running"] is not True
                or after["intent_sha256"] != sha(assets[RECOVERY[0]]) or before["remote_job"] != after["remote_job"]
                or after["old_control_exited"] is not True or after["same_live_job"] is not True
                or after["preserved_receipts"] != len(hashes) or not hashes or after["resumed_s"] < before["at_s"]
                or not all(n in TRIAL and n in assets and sha(assets[n]) == value for n, value in hashes.items())):
            raise ValueError("startup restart witness or preserved trial receipts differ")
    if any(json.loads(assets[n])["order_id"] != key for n in COST_FILES if n in assets):
        raise ValueError("cost observation is bound to a different startup order")
    context = json.loads(assets["context.json"]) if "context.json" in assets else None
    if context and (candidatesrc is None or context["order"] != key or context["candidate"] != sha(candidatesrc)
                    or context["source"] != sha(original) or context["model"] != order["binding"]["gguf"]):
        raise ValueError("startup qualification context differs from the measured source or model")
    if "result.json" in assets and (context is None or json.loads(assets["result.json"])["context"] != context):
        raise ValueError("result belongs to another startup attempt")
    extra_attempts = {}
    qualified = None
    if "result.json" in assets:
        verdict = json.loads(assets["result.json"])
        if type(verdict["passed"]) is not bool:
            raise ValueError("invalid startup verdict")
        qualified = False
        if verdict["passed"]:
            # Independent replay on the control host verifies both baselines, the negative cases, all six pairs and grades.
            with TemporaryDirectory() as tmp:
                base = Path(tmp) / restart_order.PATH
                base.parent.mkdir(parents=True)
                base.write_bytes(original)
                check = SimpleNamespace(order=state / "order.json", trial=state / "trial", checkout=Path(tmp),
                                        candidate_source=author / "candidate.py")
                if (state / "startup-fit.json").is_file():
                    from lab.pk1 import restart_release
                    verified = restart_release.approved(check)
                    checkpoint_assets(state, order, verified, assets)
                    capture_incident(state, assets, order, verified)
                    if "automatic-withdrawal.json" in assets:
                        verify_withdrawal(assets, order, verified)
                    if (state / "native-pilot").is_dir():
                        pilot_serving_assets(state, verified, assets)
                    for work in sorted(p for p in state.glob("product-v*") if p.is_dir()):
                        if not re.fullmatch(r"product-v(?:[2-9]|[1-9][0-9]+)", work.name) or work.is_symlink() or not work.is_dir():
                            raise ValueError("invalid isolated product attempt directory")
                        shared = dict(assets)
                        product_assets(state, order, verified, shared, attempt=work, checkout=check.checkout)
                        additions = {n: b for n, b in shared.items() if n.startswith("http-") or n in PLANS or n in ("tokenizer-preflight.json", "stock-reference-diagnostic.json")}
                        for n in ("cost-after-product.json",):
                            if (work / n).is_file():
                                additions[n] = bounded(work / n)
                        if additions:
                            extra_attempts[work.name] = additions
                    if (state / "product").is_dir():
                        product_assets(state, order, verified, assets)
                        tokenizer_diagnosis(state, order, assets)
                else:
                    restart_trial.qualification(check)
            qualified = True
    folder.mkdir(parents=True, exist_ok=True)
    if folder.is_symlink():
        raise ValueError("symlinked startup export directory")
    fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        previous = manifest(folder) if (folder / "manifest.json").exists() else {}
        files = persist_assets(folder, previous.get("files", {}), assets)
        if qualified is None:
            qualified = previous.get("qualified")
        for attempt, entries in extra_attempts.items():
            out = folder / attempt
            out.mkdir(exist_ok=True)
            if out.is_symlink():
                raise ValueError("symlinked product attempt export")
            old = read(out / "manifest.json") if (out / "manifest.json").is_file() else {}
            attempt_files = persist_assets(out, old.get("files", {}), entries)
            save(out / "manifest.json", {"schema": "ironmule.portable-product-attempt.v1", "order_id": key,
                 "attempt": attempt, "control_sha256": sha(assets["control.json"]), "head_sha256": sha(assets["startup-head.npz"]),
                 "files": attempt_files})
        candidate_sha = sha(candidatesrc) if candidatesrc else previous.get("candidate_sha256")
        save(folder / "manifest.json", {"schema": SCHEMA, "order_id": key, "files": files, "base_commit": order["base_commit"],
             "cost_journal": COST_JOURNAL, "qualified": qualified, "candidate_sha256": candidate_sha,
             "rebuild": {"source_path": restart_order.PATH, "reference_sha256": sha(original), "w4_binary_sha256": order["binding"]["w4_sha256"],
                         "instructions": "Check out base_commit and apply candidate.diff to reconstruct this Python candidate. "
                                         "Restore the native library, table and GGUF by their bound hashes and run lab.pk1.restart_trial."}})
    finally:
        os.close(fd)
    return folder / "manifest.json"



def product_attempts(folder):
    """Enumerate distinct first and corrective product runs under one qualified native order."""
    base = manifest(folder)["files"]
    if "http-product-qualify.json" in base:
        yield folder
    for path in sorted(folder.glob("product-v*/manifest.json"), key=lambda p: int(p.parent.name[9:])):
        out = path.parent
        index = read(path)
        if (index["schema"] != "ironmule.portable-product-attempt.v1" or index["attempt"] != out.name or index["order_id"] != folder.name
                or index["control_sha256"] != base["control.json"]["sha256"] or index["head_sha256"] != base["startup-head.npz"]["sha256"]
                or not set(index["files"]) <= {*("http-" + n for n in HTTP_FILES), *PLANS, "cost-after-product.json", "tokenizer-preflight.json", "stock-reference-diagnostic.json"}
                or any(sha(bounded(out / n)) != row["sha256"] for n, row in index["files"].items())):
            raise ValueError("product attempt scope or captured bytes differ")
        yield out


def selected_product(folder):
    """Return the newest independently accepted attempt that actually served and exercised withdrawal."""
    selected = None
    for attempt in product_attempts(folder):
        try:
            grade, exercise, work = (read(attempt / ("http-product-" + stage + ".json")) for stage in ("qualify", "exercise", "serve"))
        except FileNotFoundError:
            continue
        if grade["passed"] is not True or exercise["passed"] is not True or work["arm"] != "candidate" or work["completions"][-1]["graded_correct"] != 18:
            continue
        from lab.pk1 import restart_http, restart_live

        if attempt != folder:
            proof = bounded(attempt / "tokenizer-preflight.json")
            if read(attempt / "http-geometry.json").get("runtime_preflight_sha256") != sha(proof):
                raise ValueError("retry was not performed with the verified tokenizer recovery")
        control = read(folder / "control.json")
        if (control["order"]["order_id"] != folder.name or control["qualification"]["passed"] is not True
                or exercise["before"]["arm"] != "candidate" or exercise["after"]["arm"] != "reference"
                or exercise["before"]["work_sha256"] != exercise["after"]["work_sha256"]):
            raise ValueError("product activation differs from the qualified native release")
        with TemporaryDirectory() as tmp:
            staging = Path(tmp)
            for n in HTTP_FILES:
                if (attempt / ("http-" + n)).is_file():
                    (staging / n).write_bytes(bounded(attempt / ("http-" + n)))
            if (restart_http.assess(staging, control) != grade
                    or restart_live.verify(read(staging / "product-listener.json"), read(staging / "product-client.json")) != work):
                raise ValueError("product replay differs from actual serving and qualification")
        selected = attempt
    return selected


def product_rows(root, folder, files, key, context):
    from training.experience_archive import record

    def emit(kind, name, **values):
        return record(kind, str((folder / name).relative_to(root)), files[name]["sha256"], kind, key, **context,
                      attempt=folder.name if folder.name.startswith("product-v") else "product-v1", **values)

    for name in PLANS:
        if name in files:
            plan = read(folder / name)
            yield emit("restart_post_product_plan", name, provenance="simulated", reason=plan["provenance"],
                       metrics=[plan.get("observed", plan.get("raw_medians", {}))], plan=plan, authorship="IronMule experiment planner",
                       uncertainty="The next experiment is prioritized from real serving history. Gains and costs are forecasts.")
    for name in HTTP_ARMS:
        public = "http-" + name
        if public not in files:
            continue
        row = read(folder / public)
        yield emit("restart_product_arm", public, reason="matched_first_http_requests", arm=name.split("-")[0],
                   seed=row["seed"], source_sha256=row["source_sha256"], startup_decision=row["startup_decision"],
                   cold_admission=row["cold_admission"],
                   metrics=[{"cold_completed_s": row["cold_completed_s"], "worker_ready_s": row["startup_s"],
                             **{phase: {k: row[phase][k] for k in ("correct", "wall_s", "ttft_p95_s")}
                                for phase in ("cold", "warm")},
                             **{k: row[k] for k in ("perf_wall_s", "perf_ttft_s", "perf_tokens", "peak_rss", "device_bytes")}}],
                   authorship="Ordinary target ProductService and independent bounded workload",
                   uncertainty="Paired HTTP qualification combines all six independent seeds.")
    if "stock-reference-diagnostic.json" in files:
        diag = read(folder / "stock-reference-diagnostic.json")
        yield emit("restart_stock_gpu_reference_failure", "stock-reference-diagnostic.json", reason=diag["error"], observed=diag,
                   decision=False, authorship="Restricted A100 gate broker", uncertainty="No comparable product performance arms were run.")
    for name, kind in (("product-qualify.json", "restart_product_qualification"),
                       ("product-exercise.json", "restart_product_rollback"), ("product-serve.json", "restart_product_serving")):
        public = "http-" + name
        if public in files:
            stage = read(folder / public)
            yield emit(kind, public, reason=stage.get("reason", name), observed=stage,
                       authorship="ProductService on the source-bound target", uncertainty="Only the admitted target workload was observed.")
    if folder.name.startswith("product-v") and "cost-after-product.json" in files:
        cost = read(folder / "cost-after-product.json")
        yield emit("restart_cost_observation", "cost-after-product.json", provenance="reported", reason=cost["phase"],
                   metrics=[cost], authorship="Registered RunPod operator", uncertainty=cost["basis"])


def experiences(root, source):
    from training.experience_archive import record

    folder = (root / source).parent
    summary = manifest(folder)
    files = summary["files"]
    order = read(folder / "order.json")
    key = order["order_id"]
    if key != summary["order_id"] or key != restart_order.order_digest(order):
        raise ValueError("startup order identity mismatch")
    binding = order["binding"]
    context = dict(hardware=binding["fp"], base_commit=order["base_commit"], model=binding["gguf"],
                   quantization="Q4_K_M", software={"native": binding["native"], "serving": binding["serving_sources"]},
                   cost_reference=summary["cost_journal"], title="GgmlEngine.__init__ W4 startup calibration",
                   workload={"cold": order["contract"]["cold"], "warm": order["contract"]["warm"]})

    def emit(kind, name, **values):
        return record(kind, str((folder / name).relative_to(root)), files[name]["sha256"], kind, key, **context, **values)

    yield emit("restart_experiment_plan", "order.json", provenance="simulated", hypothesis=order["hypothesis"],
               reason="reuse_recent_device_bound_mask_on_repeat_start", observations=order["evidence"],
               qualification_contract=order["contract"], authorship="IronMule selected the observed startup phase",
               uncertainty="Historical phases and local mask observations guide selection. Their timings do not establish a gain.")
    if "startup-fit.json" in files:
        fit = read(folder / "startup-fit.json")
        yield emit("restart_pilot_heldout", "startup-fit.json", provenance="measured_replay",
                   reason="heldout_factual_first_family", weights_sha256=fit["weights_sha256"],
                   metrics=[{k: fit[k] for k in ("seed", "parameters_per_head", "train_family", "holdout_family",
                       "training_cpu_wall_s", "train_complete_pairs", "holdout_complete_pairs", "holdout_decision",
                       "holdout_correct_profile", "holdout_seconds", "matched_eligibility_ablation", "train_log_margin_error")}],
                   initialization="Three independently seeded heads trained from random weights on CPU",
                   authorship="IronMule fitted and replayed the CPU startup controller",
                   uncertainty=fit["uncertainty"])
    if "product-tokenizer-diagnostic.json" in files:
        report = read(folder / "product-tokenizer-diagnostic.json")
        child = report["nested_diagnostic"]["tokenizer_child"]
        yield emit("restart_product_gate_correction", "product-tokenizer-diagnostic.json", provenance="corrected_interpretation",
                   reason="nested_mlx_managed_allocation_import_failure", previous_report=report["gate_diagnostic"],
                   observed_error=child["stderr"].strip().splitlines()[-1], packages=report["packages"],
                   quality={"returned_ids": child["ids_count"], "factual_fixture": child["facts_sha256"]},
                   note="The tokenization wrapper classified the ImportError as missing_dependency. The nested process imported MLX and CUDA rejected its managed-memory allocation.",
                   authorship="Restricted physical reference diagnostic", uncertainty="This identifies the failing call. A successful repair requires a separate full tokenizer preflight.")
    if "incident-result.json" in files:
        failure = read(folder / "incident-result.json")
        yield emit("restart_automatic_withdrawal_incident", "incident-result.json", reason=failure["observed_error"],
                   decision=False, observed=failure, authorship="Physical shadow release",
                   uncertainty="The first incumbent attempt failed its frozen calibration check.")
    if "incident-restarted-reference.json" in files and "incident-withdrawn-reference-arm.json" in files:
        recovered = read(folder / "incident-restarted-reference.json")
        yield emit("restart_withdrawn_reference_recovery", "incident-restarted-reference.json", reason="persisted_withdrawal_fresh_process",
                   decision=True, observed=recovered, authorship="Physical shadow release",
                   uncertainty="Subsequent fresh reference completed the fixed workload and retained the withdrawal.")
    if "automatic-withdrawal.json" in files:
        withdrawal = read(folder / "automatic-withdrawal.json")
        yield emit("restart_automatic_withdrawal", "automatic-withdrawal.json", reason=withdrawal["withdrawal"]["reason"],
                   decision=withdrawal["passed"], observed=withdrawal,
                   authorship="Target release runner in two fresh processes under a shadow selection",
                   uncertainty="Tests an expired cache receipt on this source and device.")
    if "pilot-comparison.json" in files:
        comparison = read(folder / "pilot-comparison.json")
        arms = [read(folder / n) for n in PILOT_ARMS]
        yield emit("restart_pilot_actual_serving", "pilot-comparison.json", reason="matched_admission_initial_and_trained",
                   decision=comparison["same_work_policy_changed_source"], actual_work=[r["work"] for r in arms],
                   metrics=[comparison], actions=[r["selection"] for r in arms], authorship="IronMule CPU pilot on the scoped A100 release",
                   uncertainty="These two process runs distinguish the policy action. Relative timing is judged in the separate counterbalanced gate.")
    for attempt in product_attempts(folder):
        yield from product_rows(root, attempt, files if attempt == folder else read(attempt / "manifest.json")["files"], key, context)
    for name in COST_FILES:
        if name in files:
            cost = read(folder / name)
            yield emit("restart_cost_observation", name, provenance="reported", reason=cost["phase"],
                       metrics=[{k: v for k, v in cost.items() if k not in ("order_id", "basis", "cost_journal")}],
                       authorship="Registered RunPod campaign operator", uncertainty=cost["basis"])
    if all(n in files for n in RECOVERY):
        before, after = (read(folder / n) for n in RECOVERY)
        yield emit("startup_coordinator_recovery", RECOVERY[1], reason="running_job_and_receipts_preserved",
                   metrics=[{"interruption_s": after["resumed_s"] - before["at_s"],
                             "immutable_receipts_verified": after["preserved_receipts"]}],
                   note=before["reason"], authorship="Campaign coordinator",
                   uncertainty="One observed restart during target qualification.")
    samples = [n for n in TRIAL if n.startswith(("ref-", "cand-", "aa-", "facts-")) or n in ("missing.json", "stale.json")]
    for name in samples:
        if name not in files:
            continue
        row = read(folder / name)
        label = name.split(".", 1)[0]
        hit = label.startswith(("cand-", "facts-cand-"))
        candidate = hit or name in ("missing.json", "stale.json")
        try:
            seed = restart_trial.SEED
            if label.startswith("facts-"):
                seed += 100 + int(label.rsplit("-", 1)[1])
            elif label.startswith(("ref-", "cand-")):
                seed += int(label.rsplit("-", 1)[1])
            expected = summary["candidate_sha256"] if candidate else order["source_sha256"]
            restart_trial.validate(row, seed, expected_source=expected)
            restart_trial.verify_binding(row, order)
            trusted = row["reused"] == hit
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            trusted = False
        timings = []
        for phase in ("cold", "warm"):
            wave = row.get(phase, {})
            times = sorted(wave.get("ttfts_s", []))
            timings.append({"phase": phase, "ttft_p95_s": times[30] if len(times) == 32 else None,
                            **{k: wave.get(k) for k in ("wall_s", "completed", "tokens", "output_sha256")}})
        yield emit("restart_serving_arm", name, arm=label, reason="bound_observation" if trusted else "unverified_binding_or_work",
                   metrics=timings, cold_to_first_batch_s=row.get("cold_s"), initialized_s=row.get("initialized_s"),
                   broker_elapsed_s=row.get("broker_elapsed_s"), peak_rss_kib=row.get("peak_rss_kib"),
                   device_used_bytes=row.get("device_used_bytes"), profile_reused=row.get("reused"),
                   quality={"graded_correct": sum(row.get("facts", [])),
                            "factual_first_batches": [{"wall_s": b.get("wall_s"), "completed_s": b.get("completed_s"),
                                                       "ttft_p95_s": max(b.get("ttfts_s") or [0]) or None,
                                                       "correct": sum(b.get("correct", []))} for b in row.get("factual_first", [])]},
                   serving_factual_timing=row.get("factual_timing"), mapped_cuda_libraries=row.get("cuda_vendor"),
                   startup_prices=row.get("startup_prices"), authorship="Isolated target evaluator",
                   uncertainty="A single arm cannot establish a causal startup or serving advantage.")
    result = read(folder / "result.json") if "result.json" in files else {}
    reply = read(folder / "response.json") if "response.json" in files else {}
    passed = result.get("passed")
    decision = passed if passed is False or (passed is True and summary["qualified"] is True) else None
    yield emit("restart_code_decision", "result.json" if result else "order.json", provenance="reported",
               decision=decision, reported_qualification=passed, metrics=[result["summary"]["metrics"]] if result.get("summary") else [],
               elapsed_s=result.get("elapsed_s"), reason=result.get("reason") or
               ("qualification_pending" if decision is None else "paired_cold_and_warm_gate"),
               note=reply.get("proposal", {}).get("rationale"), candidate_sha256=summary["candidate_sha256"],
               authorship={"selector": "IronMule", "code_author": reply.get("model")},
               references={n: item["sha256"] for n, item in files.items()},
               deployment=read(folder / "release-run.json") if "release-run.json" in files else None,
               rollback=read(folder / "release-exercise.json") if "release-exercise.json" in files else None,
               uncertainty="Only the measured repeat-start scope is eligible for a separate target activation.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    p.add_argument("--product-attempt", help="place the after-product cost observation alongside the independently retried protocol")
    p.add_argument("--cost-phase", choices=COST_PHASES, help="take a redacted, read-only observation of the live registered lease")
    p.add_argument("--reference", type=Path, help="exact original ggml_engine.py for a shallow checkout")
    p.add_argument("--observations", nargs=2, type=Path, metavar=("PREVIOUS", "CANONICAL"))
    print(export(p.parse_args()))


if __name__ == "__main__":
    main()
