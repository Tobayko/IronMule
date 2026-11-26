"""Build and query an offline, source-addressed index of IronMule experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
from collections import Counter
from collections.abc import Iterator

from friday_evidence.canonical import canonical_json, canonical_sha256
from ironmule_product.placement_training import check_trace

ROOT = Path(__file__).resolve().parents[1]
PILOT_TRIALS = "evidence/runtime/cpu-pilot-*-trials.json"
# The trace schema records an opaque configuration hash and affinity. These five
# file hashes are the specific runs attributed to the EPYC host in docs/LAB.md.
EPYC_PLACEMENT_REPORTS = {
    "104d1387456c77db5f8108f8c9cb5bf5173fdb7e1a879b9f2f94131f45ac443a",
    "50b630e1d5e84588e4f9853fbc1194d2815748623873c57e6213b0ef3c12a945",
    "4cb1637f4a0773c4b812b164e8149f452bf7bc2aa930508c382d2c6c7da7ec6f",
    "9b5b1d2eb263aa4632987466650538cf5bed7854f7c8b2e3223b0e6cba38969d",
    "13f9a940162dc46e61ff122e066f34fa23af94556694d39f3e07bf26a7440acc",
}
ARCHIVES = {
    36: "lab/pk1/results/pod36/ps1b_all.tgz",
    37: "lab/pk1/results/pod37/pk6_all.tgz",
}


def record(kind, source, digest, locator, group, **data):
    """The locator addresses the exact row within the content-addressed source."""
    return {
        "schema": "ironmule.experience.v1",
        "id": canonical_sha256([kind, source, digest, locator]),
        "kind": kind, "source": source, "source_sha256": digest, "locator": locator, "group": group,
        "provenance": "measured", "label_eligible": False, "hardware": None, "software": None,
        "model": None, "quantization": None, "base_commit": None, "workload": {}, "profiles": [],
        "metrics": [], "quality": None, "memory_mib": None, "cost_usd": None, "reason": None,
        "uncertainty": None, "authorship": None, **data,
    }


def placement(source, digest, doc):
    check_trace(doc)
    binding = doc["binding"]
    context = {
        "hardware": {"cpu": "AMD EPYC 9V74" if digest in EPYC_PLACEMENT_REPORTS else None,
                     "affinity": binding["affinity"], "cpu_quota": binding["cpu_budget"], "devices": binding["devices"]},
        "software": {"protocol": doc["schema"], "trainer_sha256": binding["trainer"],
                     "configuration_sha256": binding["configuration"]},
        "model": binding["revision"], "quantization": "Q4_K_M" if digest in EPYC_PLACEMENT_REPORTS else None,
        "profiles": binding["gates"],
        "authorship": "IronMule placement calibration; maintainer specified worker profiles",
    }
    for i, trial in enumerate(doc["trials"]):
        chars, limit, queues = trial["case"]
        outputs = trial.get("outputs", [None] * len(queues))
        metrics = [{"elapsed_ms": ms, "ttft_ms": first, "generated_tokens": n,
                    "tokens_per_second": n * 1000 / ms, "output_sha256": h}
                   for (ms, first, n), h in zip(trial["arms"], outputs, strict=True)]
        reason = None
        if len(metrics) < 2:
            reason = "single_worker"
        elif None in outputs:
            reason = "response_digests_unavailable"
        elif len(set(outputs)) > 1 or len({m["generated_tokens"] for m in metrics}) > 1:
            reason = "different_completions"
        # Hold the fixture family together across queues, repeats and v1/v2/v3 reports.
        group = f"placement/{binding['revision']}/{chars}chars"
        yield record("placement_pair", source, digest, f"trials/{i}", group, **context,
                     workload={"prompt_chars": chars, "max_tokens": limit, "queued": queues},
                     metrics=metrics, quality={"response_and_stop_digests": outputs},
                     label_eligible=reason is None, reason=reason,
                     uncertainty="Same-host route comparison only. Background queues are part of the workload.")
    # Retain superseded, rejected and accepted campaign verdicts alongside the timing cells.
    for i, verdict in enumerate([*doc.get("history", []), doc["comparison"]]):
        yield record("placement_verdict", source, digest, f"verdict/{i}", f"placement/{doc['identity']}",
                     **context, metrics=verdict.get("holdout", []), reason=verdict.get("reason"),
                     decision=verdict.get("qualified"), error=verdict.get("failure"),
                     uncertainty="Campaign-level verdict, possibly from an earlier workload.")


def notebook(source, digest, blob):
    lines = blob.decode().splitlines()
    headings = [i for i, line in enumerate(lines) if re.match(r"^#{1,3} ", line)]
    for i, end in zip(headings, [*headings[1:], len(lines)]):
        body = "\n".join(lines[i + 1:end]).strip()
        if not body:
            continue
        yield record("lab_note", source, digest, f"lines/{i + 1}-{end}", f"lab/{i + 1}",
                     provenance="reported", title=lines[i].lstrip("# "), note=body,
                     uncertainty="Notebook text includes hypotheses and commentary. Consult the linked raw source.")


def a100(root, pod, source, digest):
    protocol = "lab/ps1/a100_ps1b.sh" if pod == 36 else "lab/pk6/a100_pk6.sh"
    with tarfile.open(root / source, "r:gz") as archive:
        for member in archive:
            # Read only known numeric event streams, never unpack a tar path to disk.
            if not member.isfile() or not re.fullmatch(
                r"\./(?:http_(?:[RSK][12]|vllm)q?|paired_(?:D|D3))_(?:32|8)\.jsonl", member.name
            ):
                continue
            if member.size > 1024 * 1024:
                raise ValueError(f"unexpected wave stream size: {member.name}")
            stream = archive.extractfile(member)
            assert stream is not None
            for i, line in enumerate(stream):
                wave = json.loads(line)
                if "wall_s" not in wave:
                    continue
                label = wave.get("arm") or wave["label"]
                external = label.startswith("vllm")
                yield record("http_wave", source, digest, f"{member.name}:{i + 1}", f"a100/pod{pod}",
                             hardware={"gpu": "NVIDIA A100 SXM4 80GB", "vcpus": 32},
                             model="Qwen3-32B", quantization="AWQ" if external else "Q4_K_M",
                             base_commit=None if external else ("02b285b" if pod == 37 else "cdd45ad"),
                             software={"engine": "vLLM 0.11.0" if external else "IronMule GGUF",
                                       "candidate_build": None if external else ("0d6d69d" if pod == 37 else "605bfef")},
                             profiles=[{"arm": label, "protocol": protocol, "pod": pod}],
                             workload={k: wave[k] for k in ("round", "repeat", "seed", "width", "pos", "prompt_tokens")
                                       if k in wave},
                             metrics=[{k: wave[k] for k in ("wall_s", "tokens", "tok_s", "ttft_p50_ms",
                                                           "ttft_p95_ms", "ttft_max_ms") if k in wave}],
                             quality=wave.get("quality"), output_digest=wave.get("texts"),
                             authorship="Claude (maintainer session)",
                             reason="different_quantization" if external else "descriptive_wave_only",
                             uncertainty="Pooled run. Cache and graph state were not captured per HTTP row.")


def traffic_replay(source, digest, doc):
    names = doc["features"]
    scope = {"hardware": {"cpu_arch": doc["cpu_arch"], "threads": doc["threads"]},
             "model": doc["gguf_sha256"], "software": {"native_sha256": doc["native_llama_sha256"]},
             "authorship": "IronMule traffic world model"}
    for block in doc["blocks"]:
        i = block["block"]
        workload = {"phase": block["phase"], "context": doc["context"], "max_tokens": doc["max_new_tokens"]}
        yield record("traffic_block", source, digest, f"blocks/{i}/actual", source, **scope,
                     workload=workload, profiles=[block["serving"]], quality={"correct": block["correct"],
                     "requests": block["requests"]}, metrics=[dict(zip(names, block["actual"], strict=True))],
                     reason="only_one_serving_arm")
        if "forecast" in block:
            yield record("traffic_forecast", source, digest, f"blocks/{i}/forecast", source, **scope,
                         provenance="simulated", workload=workload,
                         predictions={model: dict(zip(names, values, strict=True))
                                      for model, values in block["forecast"].items()},
                         error=block.get("error"), decision=block.get("decision"), reason="simulated_forecast")


def pilot_trials(source, digest, doc):
    from training.pilot_threads import examples

    # Reuse the training contract: every supported thread count needs two matching arms.
    groups = {id(group): set(choices) for group, _, choices in examples(doc)}
    profile = doc["profile"]
    for i, sample in enumerate(doc["groups"]):
        obs = sample["observations"]
        choices = groups.get(id(sample), set())
        arms = Counter((r["threads"], r["repetition"]) for r in obs)
        valid = (len(choices) > 1 and arms == Counter((n, rep) for n in choices for rep in (0, 1))
                 and all(re.fullmatch(r"[0-9a-f]{64}", r["tokens_digest"] or "") for r in obs))
        resolved = []
        if valid:
            by_thread = {n: [r["seconds"] for r in obs if r["threads"] == n] for n in choices}
            reference = by_thread[sample["base"]]
            resolved = sorted(n for n, times in by_thread.items()
                              if n != sample["base"] and (max(times) < min(reference) or min(times) > max(reference)))
        reason = None if resolved else ("overlapping_repeats" if valid else "unverified_comparison")
        yield record("pilot_thread_trial", source, digest, f"groups/{i}", str(sample["group"]),
                     hardware=profile["machine"], software={"platform": profile["platform"], "numpy": profile["numpy"]},
                     model=profile["gguf"], base_commit=profile["commit"], new_resource_bookings_usd=profile["cost_usd_incremental"],
                     profiles=[profile], binding=doc["binding"], metrics=obs,
                     workload={k: v for k, v in sample.items() if k not in ("group", "observations")},
                     label_eligible=bool(resolved), labeled_alternative_threads=resolved, reason=reason,
                     uncertainty="Per-step measurement on the recorded binding.")


def code_experiment(root, source, result_hash):
    """Index a published qualification in place. Each pair stays tied to its order and both Git trees."""
    folder = (root / source).parent
    result = json.loads((root / source).read_bytes())
    metadata = {}
    refs = {}
    for name in ("order.json", "evaluation.json", "proposal.json", "candidate.diff"):
        blob = (folder / name).read_bytes()
        refs[str((folder / name).relative_to(root))] = canonical_sha256(blob)
        if name.endswith(".json"):
            metadata[name] = json.loads(blob)
    order, signature, proposal = (metadata[k] for k in ("order.json", "evaluation.json", "proposal.json"))
    if signature["order_id"] != order["order_id"]:
        raise ValueError("reported candidate order differs from its evaluation manifest")
    group = order["order_id"]
    binding = order["binding"]
    context = {"base_commit": order["base_commit"], "model": binding["model_name"], "hardware": binding["machine"],
               "software": {"native": binding["native"], "python": binding["python"]}, "profiles": [signature],
               "title": order["target"]["function"],
               "authorship": "IronMule selected the target and evaluated the outcome. Codex authored the code."}
    metrics = []
    for path in sorted(folder.glob("pair-*.json")):
        blob = path.read_bytes()
        pair = json.loads(blob)
        if any(pair.get(k) != value for k, value in signature.items()):
            raise ValueError("published pair differs from its evaluation manifest")
        arms = [pair["reference"], pair["candidate"]]
        summaries = [{k: row.get(k) for k in ("wall_s", "ttft_s", "peak_rss_kib", "completed", "tokens",
                                               "correct", "output_sha256", "device_free_bytes_after_waves")}
                     for row in arms]
        metric = {"pair": path.stem, "reference": summaries[0], "candidate": summaries[1]}
        comparable = (all(arms[0][k] == arms[1][k] for k in ("workload", "binding", "output_sha256", "tokens", "completed"))
                      and all(r["correct"] == r["completed"] and not r["profiled"] for r in arms))
        if comparable and summaries[1]["wall_s"] > 0:
            metric["rate_ratio"] = summaries[0]["wall_s"] / summaries[1]["wall_s"]
        raw = str(path.relative_to(root))
        refs[raw] = canonical_sha256(blob)
        metrics.append(metric)
        yield record("code_trial_pair", raw, refs[raw], "pair", group, **context,
                     workload=arms[0]["workload"], binding=arms[0]["binding"], metrics=[metric],
                     reason=None if comparable else "mismatched_work_or_conditions",
                     uncertainty="One paired observation; admission uses the complete campaign.")
    faster = sum(m.get("rate_ratio", 0) > 1 for m in metrics)
    outcome = "Passed" if result["passed"] else "Rejected"
    yield record("code_trial_decision", source, result_hash, "decision", group, **context,
                 provenance="reported", decision=result["passed"], reason=f"{outcome}: {result['reason']}",
                 note=f"{outcome}. {faster}/{len(metrics)} pairs faster. {proposal['rationale']}",
                 hypothesis=order["hypothesis"], metrics=[result], references=refs,
                 uncertainty="The decision covers only the measured fixture and runtime. Activation is recorded separately.")


def scratch_qualification(source, digest, run):
    qualification, binding = run["scratch_qualification"], run["binding"]
    state = qualification["status"]["scratch_candidate"]
    head = qualification["binding"]
    used = run["scratch_measured_decisions"]
    qualified = bool(head["qualified_offsets"])
    return record("pilot_head_qualification", source, digest, "scratch_qualification", head["learner_binding"],
                  model=binding["model_name"], base_commit=run["commit"],
                  hardware=binding["machine"], binding=binding, workload=run["workload"],
                  software={"evaluator_sha256": run["evaluator_sha256"], "runtime": run["runtime_source_sha256"]},
                  authorship="The from-scratch CPU Pilot proposed the thread choices. Its local gate decided admission.",
                  decision=qualified, reason="locally_qualified" if qualified else "no_locally_qualified_thread_choice",
                  profiles=[head], quality={"completed": run["completed"], "correct": run["correct"],
                                            "output_sha256": run["output_sha256"]},
                  metrics=[{"head_proposals": state["trials"], "search_and_warmup_s": qualification["search_and_warmup_s"],
                            "scratch_measured_decisions": used, "wall_s": run["wall_s"], "ttft_p95_s": run["ttft_s"],
                            "output_tokens": run["tokens"]}],
                  note="The reference thread policy served all measured decisions." if used == 0 else
                       "Local admission and measured uses are separate counts.",
                  uncertainty="One run. No independent serving comparator for a throughput or latency reward.")


def cpu_experiment(root, source, digest):
    from training.code_author import digest as proposal_digest
    from training.experiment_planner import digest as wave_digest

    path = root / source
    doc = json.loads(path.read_bytes())
    if source.endswith("-replay.json"):
        trials = path.with_name(path.name.replace("-replay.json", "-trials.json"))
        weights = path.with_name(path.name.replace("-replay.json", "-candidate.npz"))
        raw = trials.read_bytes()
        inputs = json.loads(raw)
        # The SQLite fitter serializes compatible cells newest-first within each band.
        sorted_cells = sorted(reversed(inputs["groups"]), key=lambda g: g["context"] // 64)
        fit_input = json.dumps({**inputs, "groups": sorted_cells}, sort_keys=True, allow_nan=False).encode()
        if (doc["trials_sha256"] not in (canonical_sha256(raw), canonical_sha256(fit_input))
                or doc["weights_sha256"] != canonical_sha256(weights.read_bytes())
                or set(doc["train_groups"]) & {row["group"] for row in doc["heldout"]}):
            raise ValueError("CPU replay input, checkpoint or grouped split changed")
        profile = inputs["profile"]
        yield record("pilot_heldout_replay", source, digest, "heldout", inputs["binding"],
                     hardware=profile["machine"], model=profile["gguf"], software=doc["backend"],
                     base_commit=profile["commit"], profiles=[profile], provenance="measured_replay",
                     workload={"train_groups": doc["train_groups"], "new_groups": doc["new_experience_groups"]},
                     metrics=doc["heldout"], changed_training_decisions=doc["training_decisions_changed"],
                     weights_sha256=doc["weights_sha256"], seed=doc["seed"], parameters=doc["parameters"],
                     fitting_seconds=doc["both_fits_seconds"], note=doc["interpretation"],
                     reason="heldout_decode_replay", uncertainty="Held-out step times are from already indexed trials.")
    elif doc.get("schema") == "ironmule.runpod.cpu-pilot.v1":
        provenance = json.loads((path.parent / "provenance.json").read_bytes())
        if provenance["base_commit"] != doc["commit"] or provenance["assigned_affinity"] != doc["cpus"]:
            raise ValueError("RunPod CPU provenance does not match the campaign")
        context = dict(hardware=provenance["physical_cpu"], base_commit=doc["commit"], model=doc["model_sha256"],
                       quantization="Q4_K_M", workload={"affinity": doc["cpus"], "quota_cores": provenance["cgroup_quota_cores"]},
                       cost_reference="lab/rp/results/campaign-costs.json", authorship="IronMule CPU Pilot")
        for i, attempt in enumerate(provenance["attempts"]):
            prior = str((path.parent / "provenance.json").relative_to(root))
            yield record("cpu_campaign_attempt", prior, canonical_sha256((root / prior).read_bytes()),
                         f"attempts/{i}", digest, **context, software=attempt["runtime"], reason=attempt["result"],
                         note=attempt.get("reason"), metrics=[{"elapsed_s": attempt["elapsed_s"]}],
                         uncertainty="An environment startup failure provides no timing comparison.")
        yield record("cpu_thread_campaign", source, digest, "steps", digest, **context, reason=doc["result"],
                     software=provenance["runtime"], metrics=doc["steps"], elapsed_s=doc["elapsed_s"],
                     new_resource_bookings_usd=doc["new_resource_bookings_usd"], note=doc["cost_source"],
                     uncertainty="The campaign occupied an existing paid lease.")
    elif doc.get("schema") == "ironmule.deferred-code-experiment.v1":
        bind = doc["binding"]
        waves = []
        for identity in doc["completed_observations"]:
            if not re.fullmatch("[a-f0-9]{64}", identity):
                raise ValueError("invalid CPU wave digest")
            row = json.loads((path.parent / "waves" / f"wave-{identity}.json").read_bytes())
            if (wave_digest(row) != identity or row["base_commit"] != doc["base_commit"] or row["binding"] != bind
                    or row.get("correct") is not True):
                raise ValueError("deferred CPU plan belongs to another wave or machine")
            waves.append({"workload": row["workload"], "wall_s": row["wall_s"], "requests": len(row["requests"])})
        yield record("cpu_deferred_experiment", source, digest, "plan", canonical_sha256(doc["completed_observations"]),
                     hardware=bind["machine"], model=bind["model_name"], base_commit=doc["base_commit"], software=bind,
                     provenance="simulated", decision=False, reason=doc["reason"], metrics=waves,
                     measured_waves=doc["completed_observations"], strongest_site=doc.get("strongest_observed_site"),
                     simulation=doc["simulation"], authorship="IronMule experiment planner",
                     uncertainty=doc["simulation"]["assumption"])
    else:
        order = json.loads((path.parent / "prior/order.json").read_bytes())
        response = json.loads((path.parent / "author-response.json").read_bytes())
        if (order["order_id"] != doc["order_id"] or doc["author_request_id"] != response["request_id"]
                or response["proposal_sha256"] != proposal_digest(response["proposal"]) or doc["queue_consumed"] is not True
                or doc["evaluation_state"] != "complete" or doc["decision"] != {"passed": False, "reason": "no source change"}):
            raise ValueError("unbound CPU no-change receipt")
        yield record("cpu_serving_author_no_change", source, digest, "decision", doc["order_id"],
                     hardware=doc["machine"], model=doc["model"], software=order["binding"]["native"],
                     base_commit=order["base_commit"], workload=doc["producer_workload"], decision=False,
                     reason=doc["decision"]["reason"], provenance="reported", title=doc["target"]["function"],
                     quality={"producer_correct": doc["producer_correct"], "producer_answers": doc["producer_answers"],
                              "discovery_correct": doc["offline_discovery"]["correct"]},
                     metrics=[{"optimization_s": doc["total_recorded_optimization_s"]}], note=response["proposal"]["rationale"],
                     new_resource_bookings_usd=doc["additional_bookings_usd"], api_charges_usd=doc["api_charges_usd"],
                     authorship={"selector": "IronMule", "worker": response["model"]},
                     uncertainty="The worker retained the reference using the archived rejection. No new code was timed.")


def native_experiment(root, source, digest, calibration_seen):
    """Import native diagnostics, code provenance and raw trial outcomes as separate record types."""
    from training.native_evidence import verify_manifest

    folder = (root / source).parent
    manifest = verify_manifest(folder)
    files = manifest["files"]

    def doc(name):
        return json.loads((folder / name).read_bytes()) if name in files else {}

    def emit(kind, name, group, **fields):
        path = (folder / name).relative_to(root).as_posix()
        return record(kind, path, files[name]["sha256"], kind, group, **context, **fields)

    order, profile = doc("order.json"), doc("order.profile.json")
    binding = order["binding"]
    group = order["order_id"]
    if (manifest["order_id"] != group or profile["binding"] != binding
            or manifest["base_commit"] != order["base_commit"] or profile["base_commit"] != order["base_commit"]
            or order["profile_sha256"] != files["order.profile.json"]["original_sha256"]):
        raise ValueError("native order and measured device attribution disagree")
    site = order["target"]
    context = {"base_commit": manifest["base_commit"], "hardware": binding["fp"], "model": binding["gguf"],
               "quantization": "Q4_K_M" if binding["gguf"] ==
                               "efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689" else None,
               "software": {"native": binding["native"], "runtime_sources": binding.get("serving_sources"),
                            "kernel_source": binding["source_sha256"], "tuner": binding.get("tuner")},
               "title": site["function"] + " " + site["specialization"], "profiles": [binding["engine"], binding["mask"]],
               "cost_reference": manifest["cost_journal"]}
    yield emit("native_device_profile", "order.profile.json", group, workload={
               k: profile[k] for k in ("width", "decode_steps", "prompt_tokens_per_client", "reference_flags")},
               metrics=[{"wall_seconds_per_window": profile["wall_seconds_per_window"],
                         "kernel_seconds_per_step": site["kernel_seconds_per_step"], "calls_per_step": site["calls_per_step"]}],
               authorship="IronMule attributed warmed device events to the compiled specialization",
               reason="diagnostic_only", uncertainty="A decode trace measures a subset of complete serving time.")
    yield emit("native_dream", "order.json", group, provenance="simulated", workload=order["workload"],
               predictions=order["dreams"], hypothesis=order["hypothesis"], reason="hypothetical_shorter_kernel",
               uncertainty=order["simulation"]["uncertainty"], authorship="IronMule native experiment planner")
    tuning = doc("measurement.json")
    if tuning and binding["tuning_receipt_sha256"] not in calibration_seen:
        calibration_seen.add(binding["tuning_receipt_sha256"])
        data = tuning["measurement"]
        rejects = Counter(row.get("rejection") or "accepted" for row in data["measured"])
        yield emit("native_calibration", "measurement.json", binding["tuning_receipt_sha256"],
                   metrics=[{"tuning_s": data["tune_s"], "measured_variants": len(data["measured"]),
                             "counts_by_outcome": dict(rejects), "dispatch_entries": len(data["entries"])}],
                   quality={"reference_relative_error_bound": data["error_bound"]}, reason=tuning["outcome"],
                   authorship="Maintainer specified kernels. The on-device tuner measured dispatch choices.")
    failure = doc("prior-failure.json")
    if failure and failure["source_log_sha256"] not in calibration_seen:
        calibration_seen.add(failure["source_log_sha256"])
        path = "prior-failure.json"
        yield record("native_failed_attribution", (folder / path).relative_to(root).as_posix(), files[path]["sha256"],
                     "prior_run", failure["source_log_sha256"], hardware={"device": failure["device"], "cc": failure["cc"]},
                     model=failure["model_filename"], decision=False, reason=failure["final_error"],
                     note="Rejected diagnostic; the source site could not be attributed.", source_log_sha256=failure["source_log_sha256"],
                     uncertainty="Model and device names are from the earlier log. Its other bindings were not verified.")
    repair = doc("startup-retry.json")
    if repair:
        yield emit("native_startup_repair", "startup-retry.json", group, reason=repair["incident"],
                   note=repair["repair"], provenance="reported", authorship="Campaign coordinator",
                   uncertainty="The failed launch reached the framework import before evaluating the candidate.")
    restart = doc("restart-witness.json")
    if restart:
        yield emit("native_coordinator_restart", "restart-witness.json", group, reason="same_target_job_and_evaluator_context",
                   metrics=[{"interruption_s": restart["interruption_s"]}], note=restart["method"],
                   candidate=restart["candidate"], context_sha256=restart["context_sha256"],
                   before_sha256=files["before-restart.json"]["sha256"], authorship="Campaign coordinator",
                   uncertainty="One witnessed local coordinator restart while the isolated build continued on the target.")
    raw_pairs = []
    result = doc("result.json")
    build, setup, screen = (doc(k) for k in ("build.json", "context.json", "screen.json"))
    for index in range(6):
        raw = [doc(f"{arm}-{index}.json") for arm in ("ref", "cand")]
        if not any(raw):
            continue
        ref, cand = raw
        trusted = bool(ref and cand and build and setup and screen
                       and setup.get("order") == group and setup.get("candidate_source") == build.get("source")
                       and not any(r.get("profiled") for r in raw)
                       and ref.get("binding", {}).get("model_sha256") == binding["gguf"]
                       and all(r.get("evaluator_sha256") == setup.get("harness") for r in raw)
                       and ref.get("workload", {}).get("seed") == 92741 + index
                       and all(ref.get(k) == cand.get(k) for k in ("workload", "binding", "completed",
                                                                    "tokens", "output_sha256", "fact_digest"))
                       and all(r.get("completed") == 32 and r.get("facts") == [True] * 18 for r in raw)
                       and [r.get("treatment") for r in raw] == [
                           {"binary": binary, "table": binding["table_sha256"]}
                           for binary in (binding["w4_sha256"], build["binary"])])
        if trusted:
            raw_pairs.append((ref, cand))
        details = []
        for label, row in zip(("reference", "candidate"), raw):
            if row:
                wall = row.get("wall_s")
                throughput = {k + "_per_s": row[k] / wall for k in ("completed", "tokens")
                              if isinstance(wall, (int, float)) and wall > 0 and isinstance(row.get(k), (int, float))}
                details.append({"arm": label, **throughput, **{k: row.get(k) for k in ("wall_s", "ttft_s", "tokens", "completed",
                   "total_elapsed_s", "pilot_decisions", "peak_rss_kib", "device_used_bytes_after_phases", "treatment", "output_sha256")},
                                "facts_passed": sum(row.get("facts", []))})
        name = f"{'ref' if ref else 'cand'}-{index}.json"
        yield emit("native_code_pair", name, group, workload=(ref or cand).get("workload"), metrics=details,
                   reason="paired_measurement" if trusted else "unpaired_or_confounded",
                   quality={"answers_and_factual_grades": [f for f in (f"ref-{index}.json", f"cand-{index}.json") if f in files]},
                   authorship="IronMule evaluated a Codex-proposed source change",
                   uncertainty="One observation of the native workload; HTTP and vLLM were outside this trial.")
    response, outcome = doc("response.json"), doc("outcome.json")
    if result or response or outcome:
        reported = result.get("passed")
        actual = reported if type(reported) is bool else None
        reason = result.get("reason") or outcome.get("stage") or "qualification_pending"
        if actual is True:
            from lab.pk1.native_release import check_row, score_screen
            from lab.pk1.native_trial import judge_native, validate_calibration
            signed = result.get("measurements_sha256", {})
            names = ["build.json", "screen.json", "calibration.json", *(f"{arm}-{i}.json" for i in range(6) for arm in ("ref", "cand"))]
            try:
                if len(raw_pairs) != 6:
                    raise ValueError("incomplete paired outcomes")
                score_screen(screen, order, build["binary"], folder / "dispatch.table")
                frozen = validate_calibration(doc("calibration.json"), binding)
                for index, pair in enumerate(raw_pairs):
                    for row, binary in zip(pair, (binding["w4_sha256"], build["binary"])):
                        check_row(row, order, binary, 92741 + index)
                        if row["binding"] != raw_pairs[0][0]["binding"] or row["binding"]["profile"] != frozen:
                            raise ValueError("target changed between observations")
                checks_passed = (all(k in files and signed.get(k) == files[k]["original_sha256"] for k in names)
                                 and screen.get("all_elements_identical") is True
                                 and screen.get("candidate") == build.get("binary") and screen.get("reference") == binding["w4_sha256"]
                                 and raw_pairs[0][0]["binding"].get("profile") == doc("calibration.json").get("profile")
                                 and result.get("context") == setup and judge_native(raw_pairs)["passed"])
            except (FileNotFoundError, IndexError, KeyError, ValueError, TypeError, ZeroDivisionError):
                checks_passed = False
            if not checks_passed:
                actual, reason = None, "confounded_or_unverified_reported_pass"
        note = response.get("proposal", {}).get("rationale")
        word = {True: "Qualified", False: "Rejected", None: "Unresolved"}[actual]
        yield record("native_code_decision", source, digest, "decision", group, **context, provenance="reported",
                     decision=actual, reported_qualification=reported, deployment=outcome.get("stage"), reason=reason,
                     workload=raw_pairs[0][0]["workload"] if raw_pairs else order["workload"],
                     coverage={"candidate_built": bool(build), "numerical_screen": bool(screen),
                               "frozen_calibration": "calibration.json" in files, "comparable_serving_pairs": len(raw_pairs)},
                     metrics=[{k: v for k, v in result.items() if k.endswith("_ratios") or k in
                              ("elapsed_s", "gpu_memory_ratios_at_phase_boundaries")}],
                     cost_window=outcome.get("cost_window"), hypothesis=order["hypothesis"],
                     note=f"{word}: {reason}. {note or 'No author reply in the export.'}",
                     references={k: v["sha256"] for k, v in files.items()},
                     authorship={"selector": "IronMule", "code_author": response.get("model"),
                                 "worker_receipt": response.get("worker_receipt")},
                     uncertainty="Archive of a source experiment. Serving requires its separate target-bound release.")


def build(root=ROOT) -> Iterator[dict]:
    root = Path(root)
    # Limit imports to repository evidence. Keep credentials, private trials and generated indexes out.
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "--", ":(glob)lab/**/results/**", "evidence/runtime", "docs/LAB.md"], cwd=root,
    ).decode().split("\0")
    checkpoints = {str(path.relative_to(root)) for path in root.glob(PILOT_TRIALS) if path.is_file()}
    files.extend(sorted(checkpoints - set(files)))
    calibration_seen = set()
    for source in sorted(filter(None, files)):
        path = root / source
        if source in checkpoints:
            blob = path.read_bytes()
            yield from pilot_trials(source, canonical_sha256(blob), json.loads(blob))
            continue
        with path.open("rb") as f:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        if source == "docs/LAB.md":
            yield from notebook(source, digest, path.read_bytes())
            # Same letter denoted two different executable configurations in these reports.
            yield record("correction", source, digest, "pod37/cross-pod-reference", "a100/pod37",
                         provenance="corrected_interpretation", reason="cross_pod_reference_changed",
                         note="Pod 36 R disabled PS1. Pod 37 R enabled PS1. Their ratios cannot establish pod spread. "
                              "Pod 36 S was the comparable previous default. See the correction in the pod 37 entry.",
                         references=list(ARCHIVES.values()))
            for pod, cost in [(36, 3.18), (37, 1.36)]:
                yield record("historical_pod_cost", source, digest, f"pod{pod}/session-cost", f"a100/pod{pod}",
                             provenance="reported", cost_usd=cost, reason="prior_campaign_total",
                             uncertainty="Approximate whole-session cost reported by the maintainer.",
                             references=[ARCHIVES[pod]])
        elif source == "evidence/runtime/smollm2-135m-cpu-dream-replay.json":
            yield from traffic_replay(source, digest, json.loads(path.read_bytes()))
        elif source.startswith("evidence/runtime/cpu-pilot-") and source.endswith("-discovery.json"):
            run = json.loads(path.read_bytes())
            if run.get("schema") == "ironmule.serving_trial.v1" and run.get("scratch_qualification"):
                yield scratch_qualification(source, digest, run)
            else:
                yield record("source", source, digest, "file", source, provenance="unclassified",
                             reason="no_scratch_qualification_reported")
        elif (source.startswith("evidence/runtime/cpu-pilot-") and source.endswith("-replay.json")
              or source in {"lab/autonomous/results/runpod-epyc-pilot/campaign.json",
                            "lab/autonomous/results/runpod-epyc-pilot/planner-decision.json",
                            "lab/autonomous/results/cpu-serving-demand/followup-plan.json",
                            "lab/autonomous/results/cpu-serving-demand/summary.json"}):
            yield from cpu_experiment(root, source, digest)
        elif source.startswith("lab/placement/results/") and path.suffix == ".json":
            doc = json.loads(path.read_bytes())
            if "trials" in doc:
                yield from placement(source, digest, doc)
            else:
                yield record("idle_campaign", source, digest, "summary", source,
                             hardware=doc["hardware"], model=doc["model"], metrics=doc["holdout"],
                             workload={"requests": len(doc["requests"])}, decision=doc["policy_active"],
                             reason="campaign_summary")
        elif source == "lab/autonomous/results/a100-w4-startup-head/fit.json":
            fit = json.loads(path.read_bytes())
            weights = path.with_name("startup.npz")
            if fit["schema"] != "ironmule.w4-profile-fit.v1" or canonical_sha256(weights.read_bytes()) != fit["weights_sha256"]:
                raise ValueError("startup W4 profile result does not match its checkpoint")
            yield record("pilot_w4_profile_holdout", source, digest, "holdout", fit["order"],
                         hardware=fit["gpu"], model=fit["gguf_sha256"], base_commit=fit["native_base"],
                         software={"weights_sha256": fit["weights_sha256"], "profile_sha256": fit["profile_sha256"],
                                   "tuner_receipt_sha256": fit["tuner_receipt_sha256"], "seed": fit["seed"]},
                         workload={"train_widths": fit["train_widths"], "new_widths": fit["new_widths"],
                                   "calibration_widths": fit["margin_calibration_widths"], "geometry": fit["geometry"], "held_out_widths": fit["holdout_widths"]},
                         metrics=[{k: fit[k] for k in ("step_ms_by_rows", "cpu_train_s", "measured_holdout_ms",
                                   "shipped_pilot_with_local_mask_ms", "fixed_stock_ms", "kernel_attempts_diagnostic_only",
                                   "kernel_rejections")}], profiles=[fit["holdout"]], decision=fit["holdout"]["profile"], reason=fit["holdout"]["reason"],
                         authorship="IronMule fitted a CPU profile head against the measured GPU startup calibration.",
                         uncertainty=fit["uncertainty"], references=[str(weights.relative_to(root))])
        elif source.startswith("lab/autonomous/results/a100-native/") and path.name == "manifest.json":
            yield from native_experiment(root, source, digest, calibration_seen)
        elif (source.startswith("lab/autonomous/results/") and path.name == "result.json"
              and all((path.parent / name).is_file() for name in
                      ("order.json", "evaluation.json", "proposal.json", "candidate.diff"))):
            yield from code_experiment(root, source, digest)
        elif source in ARCHIVES.values():
            pod = next(p for p, name in ARCHIVES.items() if name == source)
            yield from a100(root, pod, source, digest)
        else:
            yield record("source", source, digest, "file", source, provenance="unclassified",
                         reason="raw_source_requires_protocol_specific_interpretation")


def write_archive(target, rows):
    """Publish a complete snapshot atomically; an interrupted rebuild keeps the previous one."""
    target = Path(target)
    counts = Counter()
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=target.parent, delete=False) as f:
        try:
            for row in rows:
                counts[row["kind"]] += 1
                f.write(canonical_json(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
            f.close()
            os.replace(f.name, target)
        finally:
            Path(f.name).unlink(missing_ok=True)
    return dict(counts)


def search(archive, query, limit):
    terms = query.casefold().split()
    with Path(archive).open(encoding="utf8") as f:
        for line in f:
            if all(term in line.casefold() for term in terms):
                yield json.loads(line)
                limit -= 1
                if limit == 0:
                    break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_command = commands.add_parser("build")
    build_command.add_argument("--output", required=True, type=Path)
    find = commands.add_parser("search")
    find.add_argument("--archive", required=True, type=Path)
    find.add_argument("--query", required=True)
    find.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if args.command == "build":
        print(canonical_json(write_archive(args.output, build())))
    else:
        if args.limit < 1:
            parser.error("--limit must be positive")
        for row in search(args.archive, args.query, args.limit):
            print(canonical_json(row))


if __name__ == "__main__":
    main()
