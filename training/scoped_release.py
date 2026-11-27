"""Publish a qualified single-file change and select it after a fresh, bound target check.

The evaluator adapters own measurement and scoring. Only the trusted control host
talks to GitHub. The target receives the control receipt and digest separately.
"""
from __future__ import annotations

import ast
import hashlib
import os
from tempfile import NamedTemporaryFile
import json
from pathlib import Path
import re
import shutil
import tempfile

from training.code_author import write_once
from training.code_trial import git, prepare, read, save, submit_candidate
from training.experiment_planner import digest
from training.trial_release import deployment, merged_source, require, switch_lock, sync_dir

SCHEMA = "ironmule.scoped-code-release.v1"
SOURCE = "ironmule/ggml_engine.py"
ADAPTERS = {
    "epyc-prefill": ("79ee889e679f38835df5f30e215215aa9d0c60d2", "GgmlEngine._prefill"),
    "epyc-prefill-matched-v2": ("79ee889e679f38835df5f30e215215aa9d0c60d2", "GgmlEngine._prefill"),
    "qwen-restart": ("33dc7bf544954f6c79a8a15536c0d6f5724b26a2", "GgmlEngine.__init__"),
}
BOUND = ("reference_sha256", "candidate_sha256", "binding", "evaluator_sha256")


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_order(adapter, order):
    base, method = ADAPTERS[adapter]
    target = order["target"]
    name = target["function"]
    if "." not in name and target.get("class"):
        name = target["class"] + "." + name
    require(target["path"] == SOURCE and name == method
            and order["allowed_paths"] == [SOURCE] and order["base_commit"] == base
            and order["order_id"] == digest({k: v for k, v in order.items() if k != "order_id"}),
            "unbound order or scope")
    if adapter == "epyc-prefill-matched-v2":
        require(order == matched_order(order["author_order"], order["paired_profile_sha256"]),
                "altered parent order, frozen profile or matched protocol")


def matched_order(parent, profile_sha):
    """The author order remains immutable. A new paired evaluation receives its own PR and target state."""
    validate_order("epyc-prefill", parent)
    require(re.fullmatch("[a-f0-9]{64}", profile_sha), "invalid frozen CPU profile")
    result = {k: parent[k] for k in ("target", "allowed_paths", "base_commit", "hypothesis")}
    result.update(schema="ironmule.epyc-prefill-matched-order.v2", evaluator="ironmule.epyc-prefill-matched.v2",
                  author_order=parent, parent_order_id=parent["order_id"], paired_profile_sha256=profile_sha,
                  first_seed=18351, pairs=6)
    return {**result, "order_id": digest(result)}


def check_method(adapter, reference, candidate):
    method = ADAPTERS[adapter][1].split(".")[1]
    def unchanged(path):
        text = Path(path).read_text()
        klass, = (n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "GgmlEngine")
        function, = (n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == method)
        require(not function.decorator_list, "unexpected target decorators")
        lines = text.splitlines(keepends=True)
        return lines[:function.lineno - 1], ast.dump(function.args), lines[function.end_lineno:]
    require(unchanged(reference) == unchanged(candidate), "proposal changed more than the ordered method")


def validate(observation):
    require(observation["passed"] is True and isinstance(observation["binding"], dict) and observation["binding"]
            and isinstance(observation["metrics"], dict) and observation["metrics"]
            and all(re.fullmatch("[a-f0-9]{64}", observation[k]) for k in BOUND if k != "binding")
            and observation["reference_sha256"] != observation["candidate_sha256"], "incomplete qualification")


# These fixed roots are exported with the engine to both target arms. Each remaining
# file in the reviewed Git tree is identical by the single-file merge check.
RUNTIME = ("ironmule", "ironmule_product", "ironmule_devices.py", "ironmule_inventory.py",
           "lab", "training", "tests", "pytest.ini", "friday_evidence", "pyproject.toml")


def runtime_manifest(tree):
    tracked = git("ls-tree", "-r", "--name-only", "-z", "HEAD", "--", *RUNTIME, cwd=tree).rstrip("\x00")
    return {name: checksum(tree / name) for name in tracked.split("\x00")}


def check_tree(receipt, tree, candidate):
    """Verify the exported tracked scope and reject importable additions, including orphaned bytecode."""
    expected = dict(receipt["runtime_manifest"])
    expected[SOURCE] = receipt["qualification"]["candidate_sha256" if candidate else "reference_sha256"]
    for name, sha in expected.items():
        path = tree / name
        require(not path.is_symlink() and checksum(path) == sha, f"changed release input: {name}")
    for root in RUNTIME:
        for file in (tree / root).rglob("*") if (tree / root).is_dir() else ():
            if file.suffix in (".py", ".so", ".pyc") and str(file.relative_to(tree)) not in expected:
                raise ValueError(f"untracked import in release tree: {file.relative_to(tree)}")


def projected_tree(receipt, base, worker_root, candidate_source=None):
    """Build an importable, read-only projection of the reviewed tree, omitting local bytecode and unrelated files."""
    candidate = candidate_source is not None
    identity = receipt["publication"]["candidate_tree" if candidate else "reference_tree"]
    require(re.fullmatch("[a-f0-9]{40}", identity), "invalid release tree")
    parent = worker_root / "release-inputs"
    parent.mkdir(mode=0o755, exist_ok=True)
    parent.chmod(0o755)
    dest = parent / identity
    if not dest.exists():
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            stage = Path(tmp) / "tree"
            stage.mkdir(mode=0o755)
            for name in receipt["runtime_manifest"]:
                source = Path(candidate_source) if candidate and name == SOURCE else base / name
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                target.chmod(0o555 if source.stat().st_mode & 0o111 else 0o444)
            for folder in stage.rglob("*"):
                if folder.is_dir():
                    folder.chmod(0o755)
            stage.chmod(0o755)
            check_tree(receipt, stage, candidate)
            try:
                stage.rename(dest)
            except OSError:
                if not dest.is_dir():
                    raise
    check_tree(receipt, dest, candidate)
    return dest


def exercise(state, expected_sha, run):
    """Exercise candidate and persisted withdrawal in a separate target state with identical real input."""
    require(selected(state, expected_sha) == "candidate", "there is no verified active release")
    try:
        with tempfile.TemporaryDirectory(dir=state) as tmp:
            shadow = Path(tmp)
            dest = shadow / "deployment"
            dest.mkdir(mode=0o700)
            original = state / "deployment"
            with switch_lock(original):
                for name in ("control.json", "selection.json", "target-check.json"):
                    shutil.copyfile(original / name, dest / name)
            before = run(shadow)
            require(before["arm"] == "candidate" and selected(shadow, expected_sha) == "candidate", "candidate unavailable")
            rollback(shadow, "target rollback exercise")
            after = run(shadow)
            require(after["arm"] == "reference" and selected(shadow, expected_sha) == "reference"
                    and before["work_sha256"] == after["work_sha256"]
                    and before["source_sha256"] != after["source_sha256"], "rollback did not restore equivalent real work")
            result = {"passed": True, "before": before, "after": after, "receipt_sha256": expected_sha}
            save(original / "rollback-exercise.json", result)
            return result
    except Exception as exc:
        rollback(state, f"rollback exercise failed: {type(exc).__name__}: {exc}"[:400])
        raise


def publish(state, order, candidate_source, *, adapter, verify, rationale, submit=False):
    """verify() must reconstruct all measured pairs, scoring, provenance, and binding from raw evidence."""
    validate_order(adapter, order)
    observed = verify()
    validate(observed)
    state.mkdir(parents=True, exist_ok=True)
    with deployment(state) as d:
        tree = prepare(d, order["base_commit"], "proposal")
        original = prepare(d, order["base_commit"], "reference")
        require(not git("status", "--porcelain", cwd=original) and checksum(original / SOURCE) == observed["reference_sha256"]
                and checksum(candidate_source) == observed["candidate_sha256"], "different evaluated source bytes")
        check_method(adapter, original / SOURCE, candidate_source)
        (tree / SOURCE).write_bytes(Path(candidate_source).read_bytes())
        git("add", "--", SOURCE, cwd=tree)
        require(git("diff", "--cached", "--name-only", "HEAD", cwd=tree) == SOURCE and not git("diff", cwd=tree)
                and not git("ls-files", "--others", "--exclude-standard", cwd=tree), "unexpected proposal contents")
        tree_id = git("write-tree", cwd=tree)
        if not submit:
            return {"status": "qualified, awaiting publication", "tree": tree_id, "metrics": observed["metrics"]}
        body = ("## Problem\n\n" + order["hypothesis"] + "\n\n## Change\n\n" + rationale
                + "\n\n## Qualification\n\n\u0060\u0060\u0060json\n" + json.dumps(observed["metrics"], indent=2)
                + "\n\u0060\u0060\u0060\n\n## Target and rollback\n\nScope is limited to the measured "
                + adapter + " profile. The target rechecks the pair before selecting this source. "
                "Withdraw the local selection using the release adapter to restore the reference.")
        number = submit_candidate(d, order, tree, body)
        merged = merged_source(d, number, order, allowed_targets={SOURCE})
        require(merged["candidate_tree"] == tree_id, "merged tree differs from measured source")
        receipt = {"schema": SCHEMA, "adapter": adapter, "order": order, "publication": merged, "qualification": observed,
                   "runtime_manifest": runtime_manifest(original)}
        write_once(d / "control.json", receipt)
        return {"pr": number, "receipt_sha256": checksum(d / "control.json"), "receipt": str(d / "control.json")}


def pinned(state, expected_sha):
    path = state / "deployment/control.json"
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == expected_sha, "incorrect control receipt digest")
    receipt = json.loads(content)
    validate_order(receipt["adapter"], receipt["order"])
    publication = receipt["publication"]
    require(receipt["schema"] == SCHEMA and publication["order_id"] == receipt["order"]["order_id"]
            and publication["base"] == receipt["order"]["base_commit"] and type(publication["pr"]) is int
            and publication["pr"] > 0 and publication["branch"] == "qualification/" + publication["order_id"][:16],
            "incomplete merged publication")
    validate(receipt["qualification"])
    return receipt


def activate(state, receipt_path, expected_sha, verify):
    """Pin control bytes, execute fresh target measurements, and atomically select only a verified arm."""
    state.mkdir(parents=True, exist_ok=True)
    content = Path(receipt_path).read_bytes()
    require(hashlib.sha256(content).hexdigest() == expected_sha, "incorrect transferred receipt digest")
    with deployment(state) as d:
        dest = d / "control.json"
        if not dest.exists():
            with NamedTemporaryFile(dir=d, delete=False) as tmp:
                tmp.write(content)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp.name, dest)
            sync_dir(d)
        receipt = pinned(state, expected_sha)
        with switch_lock(d):
            if (d / "selection.json").exists() and read(d / "selection.json")["withdrawn"]:
                return read(d / "selection.json")
        try:
            checked = verify(receipt)
            validate(checked)
            require(all(checked[k] == receipt["qualification"][k] for k in BOUND), "deployment conditions or source differ")
            save(d / "target-check.json", checked)
        except Exception as exc:
            return rollback(state, f"target verification failed: {type(exc).__name__}: {exc}"[:400])
        with switch_lock(d):
            current = read(d / "selection.json") if (d / "selection.json").exists() else {}
            if current.get("withdrawn"):
                return current
            current = {"arm": "candidate", "withdrawn": False, "receipt_sha256": expected_sha,
                       "target_check_sha256": checksum(d / "target-check.json"), "pr": receipt["publication"]["pr"]}
            save(d / "selection.json", current)
            return current


def rollback(state, reason="explicit target withdrawal"):
    d = state / "deployment"
    d.mkdir(parents=True, exist_ok=True)
    with switch_lock(d):
        current = {"arm": "reference", "withdrawn": True, "reason": reason}
        save(d / "selection.json", current)
        return current


def selected(state, expected_sha):
    """Read a durable decision; the caller checks the local binding before launching its candidate."""
    try:
        d = state / "deployment"
        current = read(d / "selection.json")
        if current.get("withdrawn") is not False or current.get("arm") != "candidate":
            return "reference"
        receipt = pinned(state, expected_sha)
        checked = read(d / "target-check.json")
        validate(checked)
        if (current["receipt_sha256"] == expected_sha and current["pr"] == receipt["publication"]["pr"]
                and all(checked[k] == receipt["qualification"][k] for k in BOUND)
                and current["target_check_sha256"] == checksum(d / "target-check.json")):
            return "candidate"
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return "reference"
