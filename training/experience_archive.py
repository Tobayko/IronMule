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


def build(root=ROOT) -> Iterator[dict]:
    root = Path(root)
    # Limit imports to repository evidence. Keep credentials, private trials and generated indexes out.
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "--", ":(glob)lab/**/results/**", "evidence/runtime", "docs/LAB.md"], cwd=root,
    ).decode().split("\0")
    checkpoints = {str(path.relative_to(root)) for path in root.glob(PILOT_TRIALS) if path.is_file()}
    files.extend(sorted(checkpoints - set(files)))
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
        elif source.startswith("lab/placement/results/") and path.suffix == ".json":
            doc = json.loads(path.read_bytes())
            if "trials" in doc:
                yield from placement(source, digest, doc)
            else:
                yield record("idle_campaign", source, digest, "summary", source,
                             hardware=doc["hardware"], model=doc["model"], metrics=doc["holdout"],
                             workload={"requests": len(doc["requests"])}, decision=doc["policy_active"],
                             reason="campaign_summary")
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
