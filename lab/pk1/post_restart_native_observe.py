"""Follow measured caller cells in an independently booted, selected real Qwen worker."""
from __future__ import annotations

import tempfile
from pathlib import Path

from lab.pk1 import native_trial as native
from lab.pk1 import post_restart_native as trace
from lab.pk1 import post_restart_observe as original
from training.code_author import write_once
from training.code_trial import read

CONTRACT = original.CONTRACT.with_name("post_restart_native_contract.json")
WORKER = Path(trace.__file__)
ACTIVITY = WORKER.with_name("post_restart_activity.py")


def observe(args):
    protocol = read(CONTRACT)
    seeds = protocol["seeds"]
    if args.seed not in seeds or (args.first is None) != (args.seed == seeds[0]):
        raise ValueError("unregistered dependency or seed")
    mounts = [(ACTIVITY, "/kernel/post_restart_activity.py")]
    env = {}
    previous = None
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temp:
        if args.first is not None:
            previous = native.checksum(args.first)
            first = read(args.first)
            if (first["schema"] != protocol["observation_schema"] or first["seed"] != seeds[0]
                    or first["contract_sha256"] != native.checksum(CONTRACT)
                    or first["profiler_sha256"] != native.checksum(WORKER)
                    or first["control_receipt_sha256"] != protocol["control_receipt_sha256"]
                    or first["selection_before"] != "candidate" or first["selection_after"] != "candidate"):
                raise ValueError("unbound prior observation")
            cells = trace.select_cells(first["records"])
            if len(cells) != 2 or any(r["aggregate_abi_s"] <= 0 for r in cells):
                raise ValueError("insufficient measured ABI work for a diagnostic")
            staged = Path(temp) / "cells.json"
            write_once(staged, cells)
            mounts.append((staged, "/kernel/cells.json"))
            env["IRONMULE_NATIVE_ACTIVITY_CELLS"] = "/kernel/cells.json"
        observation = original.observe(args, contract_path=CONTRACT, worker=WORKER, extra_mounts=mounts, extra_env=env)
    return {**observation, "prior_observation_sha256": previous, "activity_helper_sha256": native.checksum(ACTIVITY)}


def main():
    p = original.argument_parser()
    p.add_argument("--first", type=Path)
    args = p.parse_args()
    if args.out.exists():
        p.error("this observation already exists")
    write_once(args.out, observe(args))


if __name__ == "__main__":
    main()
