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
from types import SimpleNamespace

from lab.pk1 import restart_order, restart_trial
from training.code_author import read_request, read_response
from training.code_trial import PAIRS, read, save
from training.native_evidence import COST_JOURNAL, ROOT, bounded, patch, persist_assets, sha, source_at, verify_manifest

SCHEMA = "ironmule.portable-qwen-restart.v1"
TRIAL = ("context.json", "aa-0.json", "aa-1.json", "stale.json", "missing.json", "receipt.json", "stale-receipt.json",
         *(f"{tag}-{i}.json" for i in range(PAIRS) for tag in ("ref", "cand")), "result.json")
RELEASE = ("release-activate.json", "release-run.json", "release-exercise.json", "target-check.json", "rollback-exercise.json")
OBS = [f"observation-{i}{suffix}.json" for i in range(2) for suffix in ("", ".profile")]
REQUIRED = {"order.json", "reference-source.txt", *OBS}
NAMES = {*REQUIRED, "request.json", "response.json", "authorship.json", "candidate.diff", "campaign.json",
         "qualified.json", "control.json", *TRIAL, *RELEASE}


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
    paths = {**{n: state / "trial" / n for n in TRIAL}, **{n: state / "release" / n for n in RELEASE},
             "control.json": state / "release/deployment/control.json", "qualified.json": state / "qualified.json",
             "campaign.json": state / "campaign.json"}
    assets.update({n: bounded(p) for n, p in paths.items() if p.is_file()})
    context = json.loads(assets["context.json"]) if "context.json" in assets else None
    if context and (candidatesrc is None or context["order"] != key or context["candidate"] != sha(candidatesrc)
                    or context["source"] != sha(original) or context["model"] != order["binding"]["gguf"]):
        raise ValueError("startup qualification context differs from the measured source or model")
    if "result.json" in assets and (context is None or json.loads(assets["result.json"])["context"] != context):
        raise ValueError("result belongs to another startup attempt")
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
        candidate_sha = sha(candidatesrc) if candidatesrc else previous.get("candidate_sha256")
        save(folder / "manifest.json", {"schema": SCHEMA, "order_id": key, "files": files, "base_commit": order["base_commit"],
             "cost_journal": COST_JOURNAL, "qualified": qualified, "candidate_sha256": candidate_sha,
             "rebuild": {"source_path": restart_order.PATH, "reference_sha256": sha(original), "w4_binary_sha256": order["binding"]["w4_sha256"],
                         "instructions": "Check out base_commit and apply candidate.diff to reconstruct this Python candidate. "
                                         "Restore the native library, table and GGUF by their bound hashes and run lab.pk1.restart_trial."}})
    finally:
        os.close(fd)
    return folder / "manifest.json"



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
    samples = [n for n in TRIAL if n.startswith(("ref-", "cand-", "aa-")) or n in ("missing.json", "stale.json")]
    for name in samples:
        if name not in files:
            continue
        row = read(folder / name)
        label = name.split(".", 1)[0]
        candidate = label.startswith("cand-") or name in ("missing.json", "stale.json")
        try:
            seed = restart_trial.SEED if name.startswith("aa-") or name in ("missing.json", "stale.json") else restart_trial.SEED + int(label.split("-")[1])
            expected = summary["candidate_sha256"] if candidate else order["source_sha256"]
            restart_trial.validate(row, seed, expected_source=expected)
            restart_trial.verify_binding(row, order)
            trusted = row["reused"] == label.startswith("cand-")
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
                   quality={"graded_correct": sum(row.get("facts", []))}, authorship="Isolated target evaluator",
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
    p.add_argument("--reference", type=Path, help="exact original ggml_engine.py for a shallow checkout")
    p.add_argument("--observations", nargs=2, type=Path, metavar=("PREVIOUS", "CANONICAL"))
    print(export(p.parse_args()))


if __name__ == "__main__":
    main()
