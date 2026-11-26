"""Export a source-reproducible native order and its measured evidence to a private staging directory or Git."""
from __future__ import annotations

import argparse
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from lab.pk1.native_order import SOURCE
from training.code_author import read_request, read_response
from training.code_trial import PAIRS, ROOT, read, save

SCHEMA = "ironmule.portable-native-trial.v1"
COST_JOURNAL = "lab/rp/results/campaign-costs.json"
TRIAL_FILES = ["context.json", "calibration.json", "build.json", "screen.json", "result.json",
               *(f"{label}-{i}.json" for i in range(PAIRS) for label in ("ref", "cand"))]
NAMES = {"order.json", "order.profile.json", "request.json", "response.json", "authorship.json", "candidate.diff",
         "prior-failure.json", "control.json", "dispatch.table", "measurement.json", "tuner.diff", "outcome.json", "activate.json", "run.json", *TRIAL_FILES}
PRIVATE = re.compile(rb'(?i)"(?:password|api_key|authorization|pod_id|secret|access_token|refresh_token)"\s*:'
                     rb'|\b(?:sk-proj-|ghp_|github_pat_|rpa_)[A-Za-z0-9_-]{12,}')
HOST_PATH = re.compile(rb'/(?:home|root|workspace|tmp|work)/[^\s"\\]+')
LIMIT = 16 * (1 << 20)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bounded(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > LIMIT:
        raise ValueError(f"unexpected file type or size: {path.name}")
    return path.read_bytes()


def source_at(base, path):
    return subprocess.check_output(["git", "show", f"{base}:{path}"], cwd=ROOT)


def patch(original, actual, name):
    return b"".join(difflib.diff_bytes(difflib.unified_diff, original.splitlines(True), actual.splitlines(True),
                                       fromfile=b"a/" + name.encode(), tofile=b"b/" + name.encode()))


def export(args):
    order_data = bounded(args.order)
    order = json.loads(order_data)
    key = order["order_id"]
    original_id = sha(json.dumps({k: v for k, v in order.items() if k != "order_id"}, sort_keys=True).encode())
    if (key != original_id or not re.fullmatch("[a-f0-9]{40}", order["base_commit"])
            or order["target"]["path"] != SOURCE or order["schema"] != "ironmule.native-code-experiment.v1"):
        raise ValueError("invalid native order")
    binding = order["binding"]
    original = source_at(order["base_commit"], SOURCE)
    profile_data = bounded(args.order.with_suffix(".profile.json"))
    profile = json.loads(profile_data)
    if (sha(original) != binding["source_sha256"] or sha(profile_data) != order["profile_sha256"]
            or profile["base_commit"] != order["base_commit"] or profile["binding"] != binding):
        raise ValueError("source or device attribution changed")
    assets = {"order.json": order_data, "order.profile.json": profile_data}
    state = args.state
    author = state / "authoring"
    request = author / "author/request.json"
    if request.exists():
        request_id, details = read_request(request.parent)
        if details["source"].encode() != original or details["order"] != order:
            raise ValueError("author request not bound to measured order")
        assets["request.json"] = bounded(request)
        reply = request.with_name("response.json")
        if reply.exists():
            assets["response.json"] = bounded(reply)
            read_response(request.parent, request_id)
    candidate = author / "candidate.cu"
    binary_hash = None
    if candidate.exists():
        candidate_bytes = bounded(candidate)
        authored = read(author / "authorship.json")
        if (authored["order"] != key or authored["candidate_sha256"] != sha(candidate_bytes)
                or "response.json" not in assets):
            raise ValueError("candidate does not match worker handoff")
        reply = json.loads(assets["response.json"])["proposal"]
        # Independently reproduce the sole changed definition from the immutable, attributed source span.
        lines = original.splitlines(keepends=True)
        start = order["target"]["line"] - 1
        if not 0 <= start < len(lines) or (order["target"]["function"] + "(").encode() not in lines[start]:
            raise ValueError("authoring source location changed")
        end = next(i + 1 for i in range(start + 1, len(lines)) if lines[i].rstrip() == b"}")
        definition = reply["definition"].strip().encode()
        reconstructed = b"".join(lines[:start]) + definition + b"\n" + b"".join(lines[end:])
        if (candidate_bytes != reconstructed or authored["rationale"] != reply["rationale"]
                or definition.split(b"{", 1)[0].strip() != b"".join(lines[start:end]).split(b"{", 1)[0].strip()):
            raise ValueError("compiled source differs from the worker's replacement definition")
        assets.update({"authorship.json": bounded(author / "authorship.json"),
                       "candidate.diff": patch(original, candidate_bytes, SOURCE)})
    for name in TRIAL_FILES:
        path = state / "trial" / name
        if path.exists():
            assets[name] = bounded(path)
    if "context.json" in assets and json.loads(assets["context.json"])["order"] != key:
        raise ValueError("capture belongs to a different trial")
    if "build.json" in assets:
        build = json.loads(assets["build.json"])
        if ("authorship.json" not in assets or json.loads(assets["authorship.json"])["candidate_sha256"] != build["source"]
                or not re.fullmatch("[0-9a-f]{64}", build["binary"])):
            raise ValueError("unbound build receipt")
        binary_hash = build["binary"]
        compiled = state / "trial/candidate.so"
        if compiled.is_file():
            with compiled.open("rb") as f:
                checksum = hashlib.sha256()
                for block in iter(lambda: f.read(1 << 20), b""):
                    checksum.update(block)
                if checksum.hexdigest() != binary_hash:
                    raise ValueError("candidate binary changed since build")
    for name in ("outcome.json", "activate.json", "run.json"):
        if (state / name).is_file():
            assets[name] = bounded(state / name)
    if "outcome.json" in assets and json.loads(assets["outcome.json"])["order"] != key:
        raise ValueError("campaign outcome belongs to a different order")
    control = state / "release/deployment/control.json"
    if control.is_file():
        receipt = bounded(control)
        if json.loads(receipt)["order"] != order:
            raise ValueError("release receipt belongs to a different order")
        assets["control.json"] = receipt
    for name, path, pin in (("measurement.json", args.tuning, "tuning_receipt_sha256"),
                            ("dispatch.table", args.table, "table_sha256")):
        if path:
            data = bounded(path)
            if sha(data) != binding[pin]:
                raise ValueError(f"different calibration: {name}")
            assets[name] = data
    tuner_name = "ironmule/w4_kernels.py"
    base_tuner = source_at(order["base_commit"], tuner_name)
    if args.tuner:
        tuner = bounded(args.tuner)
        if sha(tuner) != binding["tuner"]:
            raise ValueError("different calibration source")
        if tuner != base_tuner:
            assets["tuner.diff"] = patch(base_tuner, tuner, tuner_name)
    if args.prior_failure_log:
        log = bounded(args.prior_failure_log)
        errors = re.findall(rb"^(?:ValueError|RuntimeError|OSError): (.{1,600})$", log, re.MULTILINE)
        if not errors:
            raise ValueError("no final bounded error in prior log")
        gpu = re.search(rb"Device 0: ([^,\r\n]+), compute capability ([\d.]+)", log)
        model = re.search(rb"from (?:[^\s]*/)?([^/\s]+\.gguf) ", log)
        previous_failure = {"source_log_sha256": sha(log), "final_error": HOST_PATH.sub(b"<private-path>", errors[-1]).decode(),
                            "device": gpu[1].decode() if gpu else None, "cc": gpu[2].decode() if gpu else None,
                            "model_filename": model[1].decode() if model else None}
        assets["prior-failure.json"] = json.dumps(previous_failure, indent=2).encode() + b"\n"
    directory = args.destination / key
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("refusing a symlinked output directory")
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        return publish(directory, key, order, original, binding, base_tuner, binary_hash, assets)
    finally:
        os.close(fd)


def publish(directory, key, order, original, binding, base_tuner, binary_hash, assets):
    manifest = directory / "manifest.json"
    previous = verify_manifest(directory)["files"] if manifest.exists() else {}
    files = dict(previous)
    secrets = [os.fsencode(v) for k, v in os.environ.items() if len(v) >= 16
               and k in ("RUNPOD_API_KEY", "OPENAI_API_KEY", "GH_TOKEN", "GITHUB_TOKEN", "AGENT_TOKEN")]
    for name, data in assets.items():
        if PRIVATE.search(data) or any(secret in data for secret in secrets):
            raise ValueError(f"private fields in {name}; review locally")
        public = HOST_PATH.sub(b"<private-path>", data) if name in ("result.json", "outcome.json") else data
        if name == "response.json" and HOST_PATH.search(public):
            response = json.loads(public)
            receipt = response["worker_receipt"]
            response["worker_receipt"] = HOST_PATH.sub(b"<worker-location>", receipt.encode()).decode()
            public = json.dumps(response, ensure_ascii=False, indent=2).encode() + b"\n"
        if HOST_PATH.search(public):
            raise ValueError(f"private filesystem path in {name}; review locally")
        info = {"sha256": sha(public), "original_sha256": sha(data)}
        dest = directory / name
        if (name in previous and previous[name] != info) or (dest.exists() and bounded(dest) != public):
            raise ValueError(f"conflicting previously exported {name}")
        if not dest.exists():
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                stream.write(public)
                stream.flush()
                os.fsync(stream.fileno())
                temp = Path(stream.name)
            try:
                os.link(temp, dest)
            except FileExistsError:
                if bounded(dest) != public:
                    raise ValueError(f"concurrent export changed {name}") from None
            finally:
                temp.unlink(missing_ok=True)
        files[name] = info
    evaluator_hash = read(directory / "context.json").get("harness") if "context.json" in files else None
    evaluator_commit = None
    if evaluator_hash:
        evaluator = "lab/pk1/native_trial.py"
        for revision in subprocess.check_output(["git", "log", "-32", "--format=%H", "--", evaluator], cwd=ROOT).decode().splitlines():
            if sha(source_at(revision, evaluator)) == evaluator_hash:
                evaluator_commit = revision
                break
    document = {"schema": SCHEMA, "order_id": key, "base_commit": order["base_commit"], "files": files,
                "cost_journal": COST_JOURNAL, "reference_binary_sha256": binding["w4_sha256"],
                "candidate_binary_sha256": binary_hash or (read(directory / "build.json")["binary"]
                                                            if "build.json" in files else None),
                "rebuild": {"source": SOURCE, "base_kernel_sha256": sha(original), "compiler": binding["fp"]["nvcc"],
                            "evaluator_sha256": evaluator_hash, "evaluator_commit": evaluator_commit,
                            "target_cc": binding["fp"]["cc"], "tuner_sha256": binding["tuner"],
                            "tuner_source_available": sha(base_tuner) == binding["tuner"] or "tuner.diff" in files,
                            "instructions": "Check out base_commit. Apply candidate.diff to a separate copy. "
                                            "Use native_trial --mode build on the original compute capability. "
                                            "Restore dispatch.table. Compare the build receipt and requalify."}}
    save(manifest, document)
    return manifest


def verify_manifest(folder):
    manifest = read(folder / "manifest.json")
    if (manifest["schema"] != SCHEMA or manifest["order_id"] != folder.name or not set(manifest["files"]) <= NAMES
            or not {"order.json", "order.profile.json"} <= set(manifest["files"])):
        raise ValueError("invalid portable native manifest")
    for name, receipt in manifest["files"].items():
        if sha(bounded(folder / name)) != receipt["sha256"]:
            raise ValueError(f"changed portable evidence: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("order", "state", "destination"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("tuning", "table", "tuner", "prior-failure-log"):
        parser.add_argument("--" + name, type=Path)
    print(export(parser.parse_args()))


if __name__ == "__main__":
    main()
