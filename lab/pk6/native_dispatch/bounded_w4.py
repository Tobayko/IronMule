"""Rebuild the original W4 table on this board and rerun the previously interrupted memory-bounded experiment."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from ironmule.kernel_gen import q4k_tensors
from ironmule.w4_kernels import ensure
from lab.pk6.mixed_progress import footprint, grade
from lab.pk6.mixed_progress.trial import GGUF_NAME, board
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import GGUF, save


def screen(primary, state, out, model):
    if (
        checksum(model) != GGUF
        or board() != json.loads((primary / "target.json").read_bytes())["hardware"]
    ):
        raise ValueError("foreign target")
    samples = grade.telemetry_samples(
        primary / "device-load.csv"
        if (primary / "device-load.csv").exists()
        else Path("/work/telemetry/primary.csv")
    )
    rows = grade.lines(primary / "waves-R1.jsonl")
    if len(rows) != 10 or not (primary / "end-R1.json").is_file():
        raise ValueError("incomplete stock process")
    peak = grade.sampled_resources(samples, rows)["peak_gpu_mib"]
    library, table = ensure(str(model), state / "kernels/w4", log=print)
    report = json.loads(table.with_name("measurement.json").read_bytes())
    geometry = []
    for kind, code in (("Q4_K", 12), ("Q6_K", 14)):
        for (k, n), item in q4k_tensors(str(model), kind=kind).items():
            widths = sorted(
                r["B"]
                for r in report["measurement"]["entries"]
                if (r["type"], r["K"], r["N"]) == (code, k, n)
            )
            if widths:
                geometry.append(
                    {
                        "kind": kind,
                        "code": code,
                        "K": k,
                        "N": n,
                        "count": item["count"],
                        "total_weight_bytes": int(item["data"].nbytes) * item["count"],
                        "widths": widths,
                    }
                )
    out.mkdir(exist_ok=False)
    inventory = out / "inventory.json"
    save(
        inventory,
        {
            "measured_shapes": geometry,
            "table_sha256": checksum(table),
            "library_sha256": checksum(library),
            "measurement_sha256": checksum(table.with_name("measurement.json")),
            "model": GGUF,
            "hardware": board(),
        },
    )
    footprint.run(
        SimpleNamespace(
            primary=primary,
            stock_peak_mib=peak,
            inventory=inventory,
            table=table,
            out=out / "screen",
            gguf=model,
        )
    )


if __name__ == "__main__":
    import fcntl

    p = argparse.ArgumentParser(description=__doc__)
    for key, arg in (
        ("primary", "/work/primary"),
        ("state", "/work/state"),
        ("out", "/work/bounded-w4"),
        ("model", "/models/" + GGUF_NAME),
    ):
        p.add_argument("--" + key, type=Path, default=Path(arg))
    args = p.parse_args()
    with Path(
        os.environ.get("IRONMULE_LAB_GPU_LOCK", "/work/native-dispatch.lock")
    ).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        screen(args.primary, args.state, args.out, args.model)
