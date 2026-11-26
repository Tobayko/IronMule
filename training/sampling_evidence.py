"""Export or replay one attributed sampling trial against the previously activated PR60 source."""
from __future__ import annotations

import argparse
import ast
import fcntl
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import textwrap
import time
from types import SimpleNamespace

from lab.pk1 import followup_contract as rules, followup_release, followup_trial as trial
from training import restart_evidence as prior
from training.code_author import read_request, read_response, write_once
from training.code_trial import read, save
from training.native_evidence import COST_JOURNAL, ROOT, bounded, patch, persist_assets, sha
from training.trial_release import require

SCHEMA = "ironmule.portable-sampling-trial.v1"
TRIAL = (*trial.file_names({"sampling_plan": True}), "result.json")
FRESH = tuple(f"fresh-{tag}-{seed}{suffix}.json" for seed in (97051, 97052)
              for tag in ("reference", "candidate") for suffix in ("", "-dispatch"))
RELEASE = ("activate.json", "run.json", "exercise.json", "target-check.json")
COST_PHASES = ("author-request", "after-verdict", "after-release")
COST_FILES = tuple(f"cost-{phase}.json" for phase in COST_PHASES)
NAMES = {"runtime-framework.json", *COST_FILES, "order.json", "decision.json", "request.json", "response.json", "candidate.diff", "outcome.json",
         "control.json", *TRIAL, *FRESH, *RELEASE}


def manifest(folder):
    index = read(folder / "manifest.json")
    require(index["schema"] == SCHEMA and folder.name == "trial-" + index["order_id"] and set(index["files"]) <= NAMES
            and {"decision.json", "order.json"} <= set(index["files"])
            and all(sha(bounded(folder / n)) == f["sha256"] for n, f in index["files"].items()), "invalid sampling evidence manifest")
    return index


def parent_source(data):
    folder = data / "a100-restart" / rules.PARENT
    index = prior.manifest(folder)
    content = prior.candidate_bytes((folder / "reference-source.txt").read_text(), read(folder / "request.json"), read(folder / "order.json"),
                                    read(folder / "response.json")["proposal"])
    require(index["candidate_sha256"] == sha(content) == followup_release.PR60_SOURCE, "different deployed PR60 source")
    return folder, content


def author_bytes(order, source, request, definition=None):
    lines = source.decode().splitlines(keepends=True)
    klass, = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == order["target"]["class"]]
    func, = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == order["target"]["function"]]
    frag = textwrap.dedent("".join(lines[func.lineno-1:func.end_lineno]))
    require(not func.decorator_list and func.lineno == order["target"]["line"] and request["source"] == frag and request["order"] == order,
            "different author request or function")
    if definition is None:
        return None
    content = ("".join(lines[:func.lineno-1]) + textwrap.indent(textwrap.dedent(definition).strip(), " " * func.col_offset) + "\n"
               + "".join(lines[func.end_lineno:])).encode()
    compile(content, trial.PATH, "exec")
    return content


def verify(order, evidence, data, local, *, inbox=None):
    parent, reference = parent_source(data)
    private = local / "checkout" / trial.PATH
    private.parent.mkdir(parents=True)
    private.write_bytes(reference)
    args = SimpleNamespace(order=local / "order.json", trial=local / "trial", checkout=local / "checkout", candidate_source=local / "candidate.py",
                           weights=parent / "startup-head.npz", control=parent / "control.json", sampling_evidence=data)
    order_bytes = evidence["order.json"]
    args.order.write_bytes(order_bytes)
    require(json.loads(evidence["decision.json"]) == order and followup_release.verified_sampling(args, order) == order["order_id"],
            "sampling order does not reproduce from its prior and withheld measurements")
    source = None
    rationale = None
    if "request.json" in evidence:
        if inbox is None:
            inbox = local / "author"
            inbox.mkdir()
            for n in ("request.json", "response.json"):
                if n in evidence:
                    (inbox / n).write_bytes(evidence[n])
        key, request = read_request(inbox)
        author_bytes(order, reference, request)
        if "response.json" in evidence:
            proposal = read_response(inbox, key)
            source = author_bytes(order, reference, request, proposal["definition"])
            rationale = proposal["rationale"]
            args.candidate_source.write_bytes(source)
            if source != reference:
                trial.source_check(private, args.candidate_source, order)
            if "candidate.diff" in evidence:
                require(evidence["candidate.diff"] == patch(reference, source, trial.PATH), "different authored source patch")
    args.trial.mkdir()
    for n in TRIAL:
        if n in evidence:
            (args.trial / n).write_bytes(evidence[n])
    if "context.json" in evidence:
        ctx = read(args.trial / "context.json")
        require(source is not None and ctx["candidate"] == sha(source) and ctx["source"] == sha(reference)
                and ctx["order"] == order["order_id"], "different sampling runtime or code")
        if "sampling-profile.json" in evidence:
            require(ctx == trial.context(args), "different sampling profile or evaluator")
    for name in COST_FILES:
        if name in evidence:
            require(json.loads(evidence[name])["order_id"] == order["order_id"], "cost for a different order")
    verdict = None
    if "result.json" in evidence:
        result = read(args.trial / "result.json")
        require(type(result["passed"]) is bool and source is not None, "unbound terminal outcome")
        if result["passed"]:
            trial.qualification(args)
        elif "summary" in result:
            require(trial.proof(args) == result["summary"] and result["summary"]["passed"] is False,
                    "negative threshold comparison differs from the raw pairs")
        verdict = result["passed"]
    if "control.json" in evidence:
        ctl = json.loads(evidence["control.json"])
        require(verdict is True and ctl["order"] == order and ctl["qualification"]["candidate_sha256"] == sha(source)
                and ctl["qualification"]["reference_sha256"] == sha(reference), "unqualified or different sampling publication")
    return reference, source, verdict, rationale


def export(state, data):
    order = read(state / "order.json")
    folder = data / "a100-calibration-sampling" / ("trial-" + order["order_id"])
    entries = {"order.json": bounded(state / "order.json"), "decision.json": bounded(state / "decision.json")}
    paths = {**{n: state / n for n in COST_FILES}, "request.json": state / "authoring/author/request.json", "response.json": state / "authoring/author/response.json",
             "control.json": state / "release/deployment/control.json", "outcome.json": state / "outcome.json",
             **{n: state / "trial" / n for n in TRIAL}, **{n: state / "release" / n for n in RELEASE},
             **{n: state / "release/fresh" / n for n in FRESH}}
    entries.update({n: bounded(path) for n, path in paths.items() if path.is_file()})
    with TemporaryDirectory() as tmp:
        ref, source, verdict, rationale = verify(order, entries, data, Path(tmp))
    candidate = state / "authoring/candidate.py"
    if (state / "campaign.json").is_file():
        raw = bounded(state / "campaign.json")
        framework = json.loads(raw)
        require(source is not None and framework["source"] == sha(source) and framework["order"] == order["order_id"]
                and all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in framework["framework"]), "different target framework scope")
        entries["runtime-framework.json"] = json.dumps({"campaign_sha256": sha(raw), "source": sha(source), "order": order["order_id"],
                              "framework": framework["framework"]}, ensure_ascii=False, indent=2).encode() + b"\n"
    if source is not None:
        if candidate.is_file():
            require(bounded(candidate) == source, "worker actually received different candidate bytes")
        entries["candidate.diff"] = patch(ref, source, trial.PATH)
    folder.mkdir(exist_ok=True, parents=True)
    fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        old = manifest(folder) if (folder / "manifest.json").is_file() else {}
        files = persist_assets(folder, old.get("files", {}), entries)
        save(folder / "manifest.json", {"schema": SCHEMA, "order_id": order["order_id"], "files": files, "qualified": verdict,
             "parent_order": rules.PARENT, "pr60_source_sha256": sha(ref), "base_commit": order["base_commit"],
             "candidate_sha256": sha(source) if source else None, "rationale": rationale, "cost_journal": COST_JOURNAL,
             "rebuild": "Apply candidate.diff to the bound PR60 source, or reconstruct it from the immutable parent and author response."})
    finally:
        os.close(fd)
    return folder / "manifest.json"


def replay(folder, data):
    index = manifest(folder)
    inputs = {n: bounded(folder / n) for n in index["files"]}
    with TemporaryDirectory() as tmp:
        _, source, verdict, _ = verify(json.loads(inputs["order.json"]), inputs, data, Path(tmp))
    require((sha(source) if source else None) == index["candidate_sha256"] and verdict is index["qualified"], "different sampling replay")
    return {"order_id": index["order_id"], "candidate_sha256": index["candidate_sha256"], "qualified": verdict, "checked_files": len(inputs)}


def experiences(root, source):
    from training.experience_archive import record

    folder = (root / source).parent
    data = folder.parent.parent
    index = manifest(folder)
    files = index["files"]
    order = read(folder / "order.json")
    parent = read(data / "a100-restart" / rules.PARENT / "control.json")
    key = order["order_id"]
    def entry(kind, name, **fields):
        return record(kind, str((folder / name).relative_to(root)), files[name]["sha256"], name, key, **fields,
                      base_commit=order["base_commit"], hardware=parent["order"]["binding"]["fp"], model=parent["order"]["binding"]["gguf"],
                      quantization="Q4_K_M", cost_reference=COST_JOURNAL, title=order["target"]["class"] + "." + order["target"]["function"])

    yield entry("calibration_sampling_order", "order.json", provenance="simulated", reason="prospective_heldout_observations_admitted_experiment",
                decision=None, measured_holdouts=order["observations"], selected_plan=order["selected_plan"], hypothesis=order["hypothesis"],
                metrics=order["blind_results"], authorship="IronMule sampling planner", uncertainty="Simulated retained-sample costs do not qualify a program.")
    if "response.json" in files:
        answer = read(folder / "response.json")
        yield entry("calibration_sampling_candidate", "response.json", provenance="reported", reason="actual_source_returned_to_planner",
                    candidate_sha256=index["candidate_sha256"], request_id=answer["request_id"], authorship=answer["model"],
                    note=index["rationale"], uncertainty="Return of a proposed source change does not measure a speedup.")
    if "context.json" in files:
        ctx = read(folder / "context.json")
        for name in TRIAL:
            if name not in files:
                continue
            row = read(folder / name)
            if "source_label" not in row or "cold" not in row:
                continue
            label = row["source_label"]
            try:
                trial.validate(row, label, row["seed"], ctx, parent, name.startswith("facts-"), name.startswith("screen-"))
                valid, error = True, None
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                valid, error = False, str(exc)[:260]
            yield entry("calibration_sampling_target_arm", name, provenance="measured", reason="raw_arm" if valid else "failed_or_unverified_arm",
                        software={"source": row["source_sha256"], "evaluator": ctx["evaluator"]}, workload={"seed": row["seed"], "arm": label},
                        metrics=[{"cold_s": row["cold_s"], "warm_s": row["warm"]["wall_s"], "broker_elapsed_s": row["broker_elapsed_s"],
                                  "warm_ttft_p95_s": sorted(row["warm"]["ttfts_s"])[30] if len(row["warm"]["ttfts_s"]) == 32 else None,
                                  "peak_rss_kib": row["peak_rss_kib"], "sampling": row.get("sampling")}],
                        quality={"graded_correct": sum(row["facts"]), "cold_completed": row["cold"]["completed"], "warm_completed": row["warm"]["completed"]},
                        error=error, authorship="Independent target evaluator", uncertainty="Single GPU arm; see the complete source gate for a causal score.")
    for name in COST_FILES:
        if name in files:
            cost = read(folder / name)
            yield entry("calibration_sampling_cost_observation", name, provenance="reported", reason=cost["phase"], metrics=[cost],
                        authorship="Registered provider lease", uncertainty=cost["basis"])
    if "result.json" in files:
        checked = replay(folder, data)
        verdict = read(folder / "result.json")
        yield entry("calibration_sampling_code_verdict", "result.json", provenance="reported", reason=verdict.get("reason", "complete prospective gate"),
                    decision=checked["qualified"], metrics=[verdict.get("summary", {}).get("metrics", {})],
                    references=files, authorship="IronMule with independent frozen target evaluator",
                    uncertainty="Only a separate merged, bound and freshly activated release may serve.")


def observe_cost(state, target, phase):
    from lab.rp import rp

    output = state / f"cost-{phase}.json"
    if output.is_file():
        return
    witness = {"author-request": "authoring/author/request.json", "after-verdict": "trial/result.json", "after-release": "release/exercise.json"}[phase]
    require((state / witness).is_file(), "sampling phase has not yet occurred")
    with rp.locked(rp.STATE):
        ledger = rp.load()
        pod = ledger["pod"]
        require(pod and pod["id"] == read(target)["pod_id"], "different registered target")
        account = rp.account()
        matches = [p for p in account["pods"] if (p["name"], p["id"]) == (pod["name"], pod["id"])]
        require(len(matches) == 1, "identical active lease unconfirmed")
        now = time.time()
        elapsed = max(0., now - pod["started"])
        write_once(output, {"order_id": read(state / "order.json")["order_id"], "phase": phase, "observed_unix_s": now,
                   "witness_sha256": sha(bounded(state / witness)), "lease_elapsed_s": elapsed, "provider_hourly_usd": matches[0]["costPerHr"],
                   "disk_hourly_upper_usd": pod["disk_rate"], "estimated_lease_usd_before_tail": elapsed * pod["cost"] / 3600,
                   "account_delta_since_initial_reservation_usd": pod["balance_before"] - account["clientBalance"],
                   "actual_reserved_including_tail_usd": rp.held(ledger), "budget_usd": rp.limit(ledger), "tail_s": rp.CLEANUP_SECONDS,
                   "basis": "Whole lease to date across all contributing experiments. This snapshot is not an incremental charge."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op", choices=("export", "replay"))
    parser.add_argument("location", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "lab/autonomous/results")
    parser.add_argument("--cost-phase", choices=COST_PHASES, help="capture an authenticated observation from the active shared lease")
    parser.add_argument("--target", type=Path, help="private registration for that lease")
    args = parser.parse_args()
    if args.cost_phase:
        if args.op != "export" or not args.target:
            parser.error("cost snapshot requires export and --target")
        observe_cost(args.location, args.target, args.cost_phase)
    print(export(args.location, args.data) if args.op == "export" else json.dumps(replay(args.location, args.data)))


if __name__ == "__main__":
    main()
