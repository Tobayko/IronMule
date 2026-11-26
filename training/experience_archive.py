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


def post_restart_observations(root, source, digest):
    """Read attributed serving time and estimated editable headroom without promoting either to a policy label."""
    from lab.pk1 import post_restart_verify as check, restart_trial
    from training import restart_evidence

    folder = (root / source).parent
    def read(path):
        return json.loads(path.read_bytes())
    plan = read(root / source)
    proof = read(folder / "replayed-observations.json")
    original = folder.parent / "a100-restart" / plan["parent_order"]
    bound = restart_evidence.manifest(original)
    receipt = read(original / "control.json")
    contract = root / "training/post_restart_contract.json"
    limits = read(contract)
    if (plan["schema"] != "ironmule.post-restart-deferred.v1" or plan["selected"] is not None or any(c["eligible"] for c in plan["candidates"])
            or plan["verification"] != proof or plan["observations"] != proof["observation_sha256"]
            or plan["base_commit"] != receipt["publication"]["candidate"] or plan["source_sha256"] != bound["candidate_sha256"]
            or proof["seeds"] != limits["seeds"] or proof["contract_sha256"] != check.sha(contract)
            or any(plan[k] != proof[k] for k in ("source_sha256", "base_commit", "parent_order", "control_receipt_sha256", "weights_sha256"))
            or plan["weights_sha256"] != bound["files"]["startup-head.npz"]["sha256"] or plan["control_receipt_sha256"] != bound["files"]["control.json"]["sha256"]):
        raise ValueError("follow-up price or measured A100 discovery belongs to an unrelated plan")
    common = dict(hardware=receipt["order"]["binding"]["fp"], base_commit=plan["base_commit"], model=receipt["order"]["binding"]["gguf"],
                  quantization="Q4_K_M", software={"source_sha256": plan["source_sha256"], "weights_sha256": plan["weights_sha256"]},
                  cost_reference="lab/rp/results/campaign-costs.json")
    for seed, expected in zip(proof["seeds"], proof["observation_sha256"], strict=True):
        path = folder / f"{seed}.json"
        row = read(path)
        if check.sha(path) != expected or row["selection_before"] != "candidate" or row["selection_after"] != "candidate":
            raise ValueError("profiled work changed since independent verification")
        check.check_selection(row["selection"], original / "startup-head.npz", receipt, limits)
        restart_trial.validate(row["arm"], seed, expected_source=plan["source_sha256"])
        restart_trial.verify_binding(row["arm"], receipt["order"])
        yield record("post_release_profiled_target", str(path.relative_to(root)), expected, str(seed), plan["parent_order"], **common,
                     reason="instrumented_measured_exclusive_times", workload={"seed": seed, "phases": limits["phases"]},
                     metrics=row["phases"], broker_elapsed_s=row["broker_elapsed_s"], selection=row["selection"],
                     authorship="IronMule measured the merged source on the original A100", uncertainty="Profiling costs prevent reuse as serving speed labels.")
    yield record("post_release_source_plan", source, digest, "defer", plan["parent_order"], **common, provenance="simulated",
                 reason=plan["reason"], decision=False, metrics=plan["candidates"], next_experiment=plan["next_experiment"],
                 required_cold_saving_s=plan["required_cold_saving_s"], measured_discovery_s=plan["measured_discovery_s"],
                 estimated_qualification_usd=plan["estimated_qualification_usd"], authorship="IronMule experiment planner",
                 uncertainty=plan["scope"] + " This historical price includes 18 processes. The follow-up protocol requires 20"
                 " (2 A/A, 2 negative controls, 12 mixed, 4 factual). The extra cost cannot make a deferred option profitable.")


def confounded_native_inventory(root, source, digest):
    from lab.pk1 import restart_trial
    from training import restart_evidence

    folder = (root / source).parent
    plan = json.loads((root / source).read_bytes())
    proof = plan["verification"]
    start = folder.parent / "a100-restart" / plan["parent_order"]
    manifest = restart_evidence.manifest(start)
    control = json.loads((start / "control.json").read_bytes())
    incident = plan["confounder"]
    if (plan["schema"] != "ironmule.post-restart-native-deferred.v1" or plan["selected"] is not None
            or any(plan[k] is not False for k in ("timing_eligible", "training_eligible", "authoring_eligible"))
            or plan["control_receipt_sha256"] != manifest["files"]["control.json"]["sha256"] or plan["weights_sha256"] != manifest["files"]["startup-head.npz"]["sha256"]
            or plan["observations"] != proof["observation_sha256"] or proof["seeds"] != [97151, 97152]
            or plan["source_sha256"] != manifest["candidate_sha256"] or plan["base_commit"] != control["publication"]["candidate"]
            or incident["evidence"] != "product-nested-diagnostic.json" or canonical_sha256((folder / incident["evidence"]).read_bytes()) != incident["sha256"]
            or incident["sha256"] != manifest["files"]["product-tokenizer-diagnostic.json"]["original_sha256"]):
        raise ValueError("unbound native inventory, GPU overlap, or timing eligibility")
    for seed, sha in zip(proof["seeds"], proof["observation_sha256"], strict=True):
        path = folder / f"{seed}.json"
        row = json.loads(path.read_bytes())
        if canonical_sha256(path.read_bytes()) != sha or row["control_receipt_sha256"] != plan["control_receipt_sha256"] or row["seed"] != seed:
            raise ValueError("native inventory or device selection changed")
        restart_trial.validate(row["arm"], seed, expected_source=plan["source_sha256"])
        restart_trial.verify_binding(row["arm"], control["order"])
        kernels = sorted({ev["symbol"] for cell in row["events"] for ev in cell["activity"]["events"]})
        yield record("native_structural_inventory", str(path.relative_to(root)), sha, str(seed), plan["parent_order"],
                     provenance="measured_confounded", hardware=control["order"]["binding"]["fp"], base_commit=plan["base_commit"],
                     model=control["order"]["binding"]["gguf"], quantization="Q4_K_M", cost_reference="lab/rp/results/campaign-costs.json",
                     workload={"seed": seed, "controlled_profile": row["selection"]["profile"]},
                     software={"source_sha256": plan["source_sha256"], "abi_images": row["abi_images"]},
                     abi_symbols=row["abi_symbols"], observed_kernel_symbols=kernels, reason="overlapping_reference_diagnostic",
                     timing_eligible=False, training_eligible=False, authoring_eligible=False, references=[source, incident["evidence"]],
                     uncertainty=incident["reason"], authorship="Two dependent physical native-activity traces")


def stopped_epyc(root, source, digest):
    from lab.rp import cpu_code_trial, cpu_matched_profile
    from training.code_author import digest as proposal_digest
    from training.code_trial import verify_deliveries

    folder = (root / source).parent
    def doc(name):
        return json.loads((folder / name).read_bytes())
    hashes = doc("sha256.json")
    stop = doc("qualification-stopped.json")
    context, profile, build = (doc(n) for n in ("context.json", "profile.json", "build.json"))
    req, reply, order = (doc(n) for n in ("../request-binding.json", "../response.json", "../../prefill-order.json"))
    if (stop["schema"] != "ironmule.epyc-matched-code-stopped.v1" or stop["passed"] is not False or stop["training_label"] is not False
            or stop["order_id"] != order["order_id"] or any(stop[k] != context[k] for k in context if k != "schema")
            or any(Path(n).name != n or canonical_sha256((folder / n).read_bytes()) != h for n, h in hashes.items())
            or hashes["qualification-stopped.json"] != digest or hashes["profile.json"] != stop["profile_sha256"]
            or hashes[stop["failed_arm"] + ".log"] != stop["failed_log_sha256"] or stop["failed_arm"] + ".json" in hashes
            or not build["passed"] or build["source_sha256"] != stop["candidate_sha256"] or build["log_sha256"] != hashes["build-1.log"]
            or req["request_id"] != reply["request_id"] or req["proposal_sha256"] != proposal_digest(reply["proposal"])
            or req["reference_source_sha256"] != stop["reference_sha256"] or req["candidate_source_sha256"] != stop["candidate_sha256"]
            or req["patch_sha256"] != canonical_sha256((folder.parent / "candidate.diff").read_bytes())):
        raise ValueError("unbound EPYC stopped verdict, source, profile or failure log")
    modes = ("reference", "candidate", "rule")
    expected = {f"{mode}-{seed}.json" for mode in modes for seed in range(18351, 18357)}
    names = stop["completed_arms"]
    triples = sum(all(f"{mode}-{seed}.json" in names for mode in modes) for seed in range(18351, 18357))
    if (len(names) != len(set(names)) or set(names) != expected.intersection(hashes)
            or triples != stop["complete_triplets"] or triples >= 6):
        raise ValueError("EPYC partial arm coverage differs")
    cpu_matched_profile.verify_profile(profile, order)
    common = dict(base_commit=order["base_commit"], hardware=profile["binding"]["machine"], model=profile["model_sha256"],
                  quantization="Q4_K_M", profiles=[profile["binding"]], cost_reference="lab/rp/results/campaign-costs.json")
    for name in names:
        mode, seed = name.removesuffix(".json").split("-")
        seed = int(seed)
        row = doc(name)
        cpu_matched_profile.verify_arm(row, profile, stop["profile_sha256"], mode, order)
        verify_deliveries(row, seed)
        sources = {**profile["runtime_source_sha256"], "ggml_engine.py": stop["candidate_sha256" if mode == "candidate" else "reference_sha256"]}
        if (row["binding"] != profile["binding"] or row["runtime_source_sha256"] != sources or row["commit"] != order["base_commit"]
                or row["workload"] != {"seed": seed, "clients": 4, "waves": 4, "max_tokens": 3, "warmup_waves": 1,
                                       "context": 512, "phases": ["short", "document"]}
                or row["profiled"] or row["completed"] != 16 or row["correct"] != 16 or len(row["waves"]) != 4
                or any(w["completed"] != 4 or w["correct"] != 4 for w in row["waves"])):
            raise ValueError("EPYC source, workload or hardware mismatch")
        if mode == "rule":
            fixed = row["simple_thread_rule"]
            if fixed["actual_threads"] != [4] or fixed["overrides"] < 1 or fixed["source_sha256"] != cpu_code_trial.sha(cpu_code_trial.RULE):
                raise ValueError("unverified fixed CPU rule")
        yield record("cpu_matched_partial_arm", str((folder / name).relative_to(root)), hashes[name], name, stop["order_id"],
                     **common, software=sources, workload=row["workload"], mode=mode, reason="partial_trial",
                     metrics=[{k: row[k] for k in ("completed", "correct", "tokens", "wall_s", "ttft_s", "total_elapsed_s", "peak_rss_kib")}],
                     quality={"verified_deliveries": row["correct"], "output_sha256": row["output_sha256"]},
                     authorship="Independent CPU evaluator", uncertainty="The six-triplet protocol ended early.")
    yield record("cpu_matched_code_stopped", source, digest, "qualification-stopped", stop["order_id"], **common,
                 software={"evaluator_sha256": stop["evaluator_sha256"], "profile_sha256": stop["profile_sha256"]},
                 reason=stop["reason"], decision=False, failed_arm=stop["failed_arm"], complete_triplets=triples,
                 completed_arms=names, references=hashes, note=reply["proposal"]["rationale"],
                 authorship={"selector": "IronMule", "code_author": reply["model"]},
                 uncertainty="Fresh native qualification failed during the third triplet. No comparative release score was issued.")


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
    elif doc.get("schema") == "ironmule.confounded-cpu-pair.v1":
        folder = path.parent
        receipts = json.loads((folder / "sha256.json").read_bytes())
        if any(Path(n).name != n or canonical_sha256((folder / n).read_bytes()) != h for n, h in receipts.items()):
            raise ValueError("EPYC paired observation changed")
        rows = {tag: json.loads((folder / f"{tag}-{int(doc['seed'])}.json").read_bytes()) for tag in ("reference", "candidate")}
        ref, cand = rows.values()
        observed = {k: r["binding"]["scratch_thread_head"]["qualified_offsets"] for k, r in rows.items()}
        if (doc["passed"] is not False or doc["training_label"] is not False or doc["observed_profiles"] != observed
                or observed["reference"] == observed["candidate"] or "comparison-rejected.json" not in receipts
                or doc["order"] != json.loads((folder / "context.json").read_bytes())["order_id"]):
            raise ValueError("unbound EPYC mismatch report")
        response = json.loads((folder.parent / "response.json").read_bytes())
        request = json.loads((folder.parent / "request-binding.json").read_bytes())
        if response["request_id"] != request["request_id"] or response["proposal_sha256"] != proposal_digest(response["proposal"]):
            raise ValueError("EPYC source reply mismatch")
        yield record("cpu_confounded_code_pair", source, digest, "rejected_profile_mismatch", doc["order"],
                     base_commit=ref["commit"], model=ref["binding"]["model_name"], hardware=ref["binding"]["machine"],
                     quantization="Q4_K_M", title="GgmlEngine._prefill", software=ref["binding"]["native"],
                     workload=ref["workload"], profiles=observed, decision=False, reason=doc["reason"],
                     metrics=[{"arm": k, **{v: r[v] for v in ("wall_s", "ttft_s", "peak_rss_kib", "tokens", "completed",
                              "correct", "output_sha256", "scratch_measured_decisions")}} for k, r in rows.items()],
                     quality={"outputs_match": ref["output_sha256"] == cand["output_sha256"]}, note=response["proposal"]["rationale"],
                     authorship={"selector": "IronMule", "code_author": response["model"]},
                     cost_reference="lab/rp/results/campaign-costs.json", uncertainty=doc["quality_scope"])
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


def prefill_experiment(root, source, digest):
    from lab.pk1.prefill_verdict import decide

    folder = (root / source).parent
    plan, receipt, decision = (json.loads((folder / f).read_bytes()) for f in
                               ("plan.json", "measurement.json", "decision.json"))
    if not re.fullmatch("[a-f0-9]{64}", plan["parent_order"]):
        raise ValueError("invalid parent for traced prefill")
    parent = folder.parent / "a100-native" / plan["parent_order"]
    origin = json.loads((parent / "order.json").read_bytes())
    if (decide(plan, receipt) != decision or plan["binding"] != origin["binding"]
            or plan["base_commit"] != origin["base_commit"] or plan["profile_sha256"] != origin["profile_sha256"]
            or plan["trace_sha256"] != canonical_sha256((folder / "profile.log").read_bytes())):
        raise ValueError("prefill microprobe receipts, trace or planning verdict disagree")
    binding = plan["binding"]
    context = dict(hardware=binding["fp"], base_commit=plan["base_commit"], model=binding["gguf"],
                   quantization="Q4_K_M", cost_reference="lab/rp/results/campaign-costs.json",
                   authorship="IronMule selected the information experiment. The device benchmark tested cuBLASLt choices.",
                   references={"parent_order": plan["parent_order"], "trace_sha256": plan["trace_sha256"],
                               "probe_sha256": plan["source_sha256"], "binary_sha256": receipt["binary_sha256"]})
    for shape, target in enumerate(receipt["targets"]):
        info, *events = target["events"]
        dimensions = {k: target["shape"][k] for k in ("family", "m", "n", "k")}
        for choice in range(info["heuristics"]):
            rows = [r for r in events if r["index"] == choice]
            paired = rows[0]["status"] == "paired"
            yield record("prefill_gemm_microprobe", source, digest, f"targets/{shape}/heuristics/{choice}",
                         receipt["request_id"], **context, software={"cublas_version": info["cublas_version"]},
                         workload={**dimensions, "inputs": "synthetic deterministic matrices", "contract": plan["contract"]},
                         metrics=rows, reason="paired_bitwise_equivalent" if paired else rows[0]["status"],
                         quality={"bitwise_equal_to_reference": True if paired else
                                  (False if rows[0]["status"] == "changed_output" else None),
                                  "mismatched_elements": rows[0].get("mismatches")},
                         uncertainty="These are isolated matrix operations. No serving responses were generated.")
    verdict = folder / "decision.json"
    yield record("prefill_gemm_planning", str(verdict.relative_to(root)), canonical_sha256(verdict.read_bytes()),
                 "simulation", receipt["request_id"], **context, provenance="simulated", reason=decision["outcome"],
                 metrics=[{k: decision[k] for k in ("baseline_prefill_s", "throughput_requirement",
                         "minimum_ms_for_2pct_if_decode_were_free", "optimistic_best_sum_ms", "per_shape")}],
                 uncertainty=decision["uncertainty"])


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
                            "lab/autonomous/results/runpod-epyc-pilot/prefill-code/qualification-v1/comparison-rejected.json",
                            "lab/autonomous/results/cpu-serving-demand/followup-plan.json",
                            "lab/autonomous/results/cpu-serving-demand/summary.json"}):
            yield from cpu_experiment(root, source, digest)
        elif source == "lab/autonomous/results/runpod-epyc-pilot/prefill-code/qualification-v2/qualification-stopped.json":
            yield from stopped_epyc(root, source, digest)
        elif source == "lab/autonomous/results/runpod-epyc-pilot/withdrawal/outcome.json":
            from training.pilot_withdrawal_replay import replay

            report = json.loads(path.read_bytes())
            prior_path = root / "evidence/runtime/cpu-pilot-epyc-discovery.json"
            prior = json.loads(prior_path.read_bytes())
            checked = replay(path.parent, prior_path)
            if (report["prior_sha256"] != canonical_sha256(prior_path.read_bytes()) or report["elapsed_s"] != checked["elapsed_s"]
                    or report["head_sha256"] != prior["binding"]["scratch_thread_head"]["weights_sha256"]
                    or any(report["stages"][name][k] != v for name, stage in checked["stages"].items() for k, v in stage.items())):
                raise ValueError("EPYC withdrawal differs from the verified worker samples")
            yield record("cpu_checkpoint_withdrawal", source, digest, "restart", report["prior_sha256"],
                         hardware=prior["binding"]["machine"], base_commit=report["commit"], model=report["gguf_sha256"],
                         quantization="Q4_K_M", workload=prior["workload"], metrics=[checked], quality={"correct": checked["verified_correct"]},
                         reason="head_missing_then_requalified_after_process_restart", weights_sha256=report["head_sha256"],
                         note=report["scope"], uncertainty=report["timing_scope"], authorship="Bound CPU target evaluator",
                         cost_reference="lab/rp/results/campaign-costs.json", new_resource_bookings_usd=report["new_resource_bookings_usd"])
        elif source == "lab/autonomous/results/a100-post-restart-native-60/decision.json":
            yield from confounded_native_inventory(root, source, digest)
        elif source == "lab/autonomous/results/a100-post-restart-60/decision.json":
            yield from post_restart_observations(root, source, digest)
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
        elif source == "lab/autonomous/results/a100-prefill-lt/measurement.json":
            yield from prefill_experiment(root, source, digest)
        elif source.startswith("lab/autonomous/results/a100-restart/") and path.name == "manifest.json" and path.parent.parent.name == "a100-restart":
            from training.restart_evidence import experiences
            yield from experiences(root, source)
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
