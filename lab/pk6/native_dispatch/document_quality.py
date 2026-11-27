"""Check exact retrieval under real KV pressure and subsequent short factual requests."""

import argparse
import gzip
import random
from pathlib import Path
from unittest.mock import patch

from ironmule.factual_gate import FactCase
from lab.pk6.mixed_progress import grade, trial
from lab.pk6.native_dispatch import run
from lab.pk6.native_dispatch.score import read, waves
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import save

SEEDS = (762941, 762967)


def documents(seed):
    rng = random.Random(seed)
    words = ("alder", "birch", "cedar", "elm", "fir", "oak", "pine", "willow")
    common = rng.choices(words, k=900)
    codes = rng.sample(range(100000, 999999), 96)
    result = []
    for index in range(32):
        tokens = common.copy() if index % 3 == 0 else rng.choices(words, k=900)
        wanted, other, last = codes[3 * index : 3 * index + 3]
        positions = [10, 450, 890]
        records = [("keel", wanted), ("mast", other), ("sail", last)]
        for offset, position in enumerate(positions):
            lot, code = records[(offset + index) % 3]
            tokens[position] = f"\nThe {lot} shipment seal is {code}.\n"
        prompt = (
            "Shipping log.\n"
            + " ".join(tokens)
            + "\nWhat is the keel shipment seal? Reply only with its six digits."
        )
        result.append(
            FactCase(f"{seed}/{index}/keel", "extraction", prompt, str(wanted))
        )
    return result


def main():
    original = trial.wave

    def evaluate(
        folder, engine, server, arm, seed, repeat, width, prompts, facts, tok, deadline
    ):
        if repeat != 3:
            return
        for index, probe_seed in enumerate(SEEDS):
            cases = documents(probe_seed)
            original(
                folder,
                engine,
                server,
                arm,
                probe_seed,
                index,
                32,
                [c.prompt for c in cases],
                cases,
                tok,
                deadline,
            )
        original(
            folder, engine, server, arm, seed, 3, 32, prompts, facts, tok, deadline
        )

    with patch.object(trial, "wave", evaluate):
        run.main()


def assess(root):
    result = {}
    by_arm = {}
    for arm in ("R2", "H2"):
        files = list(root.rglob(f"end-{arm}.json"))
        if not files:
            continue
        if len(files) != 1:
            raise ValueError("ambiguous retrieval process")
        folder = files[0].parent
        target = read(folder / "target.json")
        start, end = read(folder / f"start-{arm}.json"), read(files[0])
        rows = waves(folder, arm)
        if (
            start["hardware"] != target["hardware"]
            or end["hardware"] != target["hardware"]
            or [r["seed"] for r in rows] != [*SEEDS, run.SEEDS[1]]
            or [r["repeat"] for r in rows] != [0, 1, 3]
        ):
            raise ValueError("unmatched target or quality workload")
        gate = read(folder / f"ready-{arm}.json")["health"]["gate"]
        if gate.get("passes") is not True:
            raise ValueError("model quality gate did not pass")
        if len({q["server_id"] for row in rows for q in row["requests"]}) != 96:
            raise ValueError("missing or reused request ID")
        verified = grade.bindings(folder, arm, rows)
        counts = []
        for index, w in enumerate(rows):
            grade.measure(w)
            expected = (
                [c.key for c in documents(w["seed"])]
                if index < 2
                else [c.key for c, _ in trial.quality_cases(w["seed"])]
            )
            requests = w["requests"]
            if (
                w["arm"] != arm
                or w["width"] != 32
                or [q["case"] for q in requests] != expected
                or not start["unix"] <= w["unix_start"] < w["unix_end"] <= end["unix"]
                or any(type(q["correct"]) is not bool for q in requests)
                or index < 2
                and not all(900 < q["prompt_tokens"] <= 2016 for q in requests)
            ):
                raise ValueError("invalid document retrieval evidence")
            counts.append(sum(q["correct"] for q in requests))
        if arm == "H2":
            trace = folder / "trace-H2.txt"
            raw = (
                trace.read_bytes()
                if trace.is_file()
                else gzip.decompress(trace.with_suffix(".txt.gz").read_bytes())
            )
            if b"physical_pool cells=16384 allocated=16384 cap=32" not in raw:
                raise ValueError("missing physical pool")
        result[arm] = {
            "hardware": target["hardware"],
            "model": target["gguf_sha256"],
            "correct_by_wave": counts,
            "correct_case_ids": [
                [q["case"] for q in w["requests"] if q["correct"]] for w in rows
            ],
            "requests_per_wave": 32,
            "input_digests": [r["input_digest"] for r in rows],
            "trace_sha256": verified["trace_sha256"],
            "gate": gate,
            "source": target["source"],
            "engine_sha256": target["engine_sha256"],
            "waves_sha256": verified["waves_sha256"],
            "target_sha256": checksum(folder / "target.json"),
        }
        by_arm[arm] = rows
    if len(by_arm) == 2:
        if (
            result["H2"]["hardware"] != result["R2"]["hardware"]
            or result["H2"]["source"] != result["R2"]["source"]
            or result["H2"]["engine_sha256"] != result["R2"]["engine_sha256"]
            or result["H2"]["model"] != result["R2"]["model"]
            or result["H2"]["input_digests"] != result["R2"]["input_digests"]
        ):
            raise ValueError("foreign retrieval comparator")
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "assess":
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument("command", choices=["assess"])
        p.add_argument("root", type=Path)
        p.add_argument("out", type=Path)
        args = p.parse_args()
        save(args.out, assess(args.root))
    else:
        import fcntl
        import os

        with Path(
            os.environ.get("IRONMULE_LAB_GPU_LOCK", "/work/native-dispatch.lock")
        ).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            main()
