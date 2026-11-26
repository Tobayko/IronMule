"""Retain the unmodified physical fallback arm before the immutable release verifier returns."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace
from lab.pk1 import restart_release as runner
from training.code_author import write_once
from training.code_trial import read


def main():
    cfg = json.loads(sys.argv[1])
    args = SimpleNamespace(
        **{k: Path(v) if k not in ("receipt_sha256", "seed", "timeout", "worker_uid") else v for k, v in cfg.items()}
    )
    assert runner.release.selected(args.state, args.receipt_sha256) == "reference"
    original = runner.trial.arm

    def retain(*a, **kw):
        row = original(*a, **kw)
        write_once(args.out.parent / "withdrawn-reference-arm.json", row)
        return row

    runner.trial.arm = retain
    try:
        output = runner.run(args, args.state, args.seed)
        passed = (
            output["arm"] == "reference"
            and output["source_sha256"]
            == runner.release.pinned(args.state, args.receipt_sha256)["qualification"]["reference_sha256"]
            and output["graded_correct"] == 18
        )
        outcome = {"passed": passed, "result": output}
    except Exception as exc:
        outcome = {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}
    write_once(
        args.out,
        {
            **outcome,
            "receipt_sha256": args.receipt_sha256,
            "selection_after": read(args.state / "deployment/selection.json"),
            "frozen_evaluator_sha256": runner.release.checksum(runner.trial.__file__),
        },
    )


if __name__ == "__main__":
    main()
