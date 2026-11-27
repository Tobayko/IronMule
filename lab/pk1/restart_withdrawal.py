"""On the activated Qwen target verify automatic withdrawal on an incompatible pinned cache."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
import subprocess
import sys

from lab.pk1 import restart_release as runner
from training import scoped_release as release
from training.code_author import write_once
from training.code_trial import read
from training.trial_release import require

SEED = 94842


def exercise(args):
    state, shadow = args.state, args.shadow
    require(
        not shadow.resolve().is_relative_to(state.resolve()) and not state.resolve().is_relative_to(shadow.resolve()),
        "withdrawal exercise requires independent target state",
    )
    live = state / "deployment"
    backed, rollback = read(live / "control.json"), read(live / "rollback-exercise.json")
    pin = args.receipt_sha256
    require(
        release.pinned(state, pin) == backed
        and release.selected(state, pin) == "candidate"
        and rollback["passed"] is True
        and rollback["receipt_sha256"] == pin,
        "need verified, deployed source and successful explicit rollback",
    )
    dest = shadow / "deployment"
    shadow.mkdir(parents=True, exist_ok=True)
    original = shadow / "previous-selection.json"
    if not original.exists():
        with release.switch_lock(live):
            if not dest.exists():
                with tempfile.TemporaryDirectory(dir=shadow) as scratch:
                    copy = Path(scratch) / "deployment"
                    copy.mkdir(mode=0o700)
                    for name in ("control.json", "target-check.json", "selection.json"):
                        shutil.copyfile(live / name, copy / name)
                    copy.rename(dest)
            require(release.selected(shadow, pin) == "candidate", "shadow did not select activated source")
            write_once(original, read(dest / "selection.json"))
    cached = read(args.trial / "receipt.json")
    stale = read(args.trial / "stale-receipt.json")
    require(stale == {**cached, "created_s": cached["created_s"] - 172800}, "missing independently tested stale input")
    changed = shadow / "expired"
    changed.mkdir(exist_ok=True)
    write_once(changed / "receipt.json", stale)

    def run(name, inputs):
        path = shadow / (name + ".json")
        if not path.exists():
            argv = [
                sys.executable,
                "-m",
                "lab.pk1.restart_release",
                "run",
                "--state",
                str(shadow),
                "--trial",
                str(inputs),
                "--order",
                str(args.order),
                "--candidate-source",
                str(args.candidate_source),
                "--checkout",
                str(args.checkout),
                "--reference-lib",
                str(args.reference_lib),
                "--table",
                str(args.table),
                "--gguf",
                str(args.gguf),
                "--worker-root",
                str(args.worker_root),
                "--receipt-sha256",
                pin,
                "--seed",
                str(SEED),
            ]
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, text=True)
            try:
                output, _ = proc.communicate(timeout=600)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise
            if proc.returncode:
                raise RuntimeError(f"fresh release process returned {proc.returncode}")
            write_once(path, {"pid": proc.pid, "result": json.loads(output)})
        return read(path)

    first = run("incompatible", changed)
    decision = read(dest / "selection.json")
    require(
        decision["withdrawn"]
        and decision["arm"] == "reference"
        and "qualified startup receipt changed or expired" in decision["reason"]
        and release.selected(shadow, pin) == "reference",
        "candidate did not withdraw automatically",
    )
    second = run("restart", args.trial)
    ref = backed["qualification"]["reference_sha256"]
    for log in (first, second):
        completed = log["result"]
        require(
            completed["arm"] == "reference"
            and completed["source_sha256"] == ref
            and completed["cache_used"] is False
            and completed["cold_performance_completions"] == 32
            and completed["warm_performance_completions"] == 32
            and completed["graded_correct"] == 18,
            "fallback did not serve the full reference workload",
        )
    require(
        first["pid"] != second["pid"]
        and first["result"]["work_sha256"] == second["result"]["work_sha256"]
        and read(dest / "selection.json") == decision
        and release.selected(state, pin) == "candidate",
        "withdrawal did not persist or affected the active deployment",
    )
    report = {
        "schema": "ironmule.qwen-automatic-withdrawal.v1",
        "passed": True,
        "receipt_sha256": pin,
        "order": backed["order"]["order_id"],
        "release_sha256": release.checksum(runner.__file__),
        "prior": read(original),
        "incompatible": first,
        "restarted_with_fresh_cache": second,
        "withdrawal": decision,
        "stale_cache_sha256": release.checksum(changed / "receipt.json"),
        "valid_cache_sha256": release.checksum(args.trial / "receipt.json"),
        "main_selection": read(live / "selection.json"),
    }
    write_once(args.output, report)
    return report


if __name__ == "__main__":
    cmd = argparse.ArgumentParser(description=__doc__)
    for field in (
        "state",
        "shadow",
        "trial",
        "order",
        "candidate-source",
        "checkout",
        "reference-lib",
        "table",
        "gguf",
        "worker-root",
        "output",
    ):
        cmd.add_argument("--" + field, required=True, type=Path)
    cmd.add_argument("--receipt-sha256", required=True)
    exercise(cmd.parse_args())
