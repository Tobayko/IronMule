"""Update one local Pilot checkpoint offline from its measured experience archive and new paired trials."""
from __future__ import annotations

import argparse
from contextlib import closing
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace

PER_BAND = 16
MAX_ROW_BYTES = 2 << 20
FITTER_SEED = 20261126


def digest(data):
    return hashlib.sha256(data).hexdigest()


def entries(archive, incoming):
    from training.experience_archive import PILOT_TRIALS, ROOT, pilot_trials

    if archive:
        if not archive.is_file() or archive.stat().st_size > 256 << 20:
            raise ValueError("supply a regular archive segment of at most 256 MiB")
        with archive.open("rb") as f:
            while line := f.readline(MAX_ROW_BYTES + 1):
                if len(line) > MAX_ROW_BYTES:
                    raise ValueError("archive row exceeds reader limit")
                yield json.loads(line)
    else:
        # Read the same content-addressed pilot corpus as the combined repository archive.
        for path in sorted(ROOT.glob(PILOT_TRIALS)):
            blob = path.read_bytes()
            yield from pilot_trials(str(path.relative_to(ROOT)), digest(blob), json.loads(blob))
    if incoming:
        if not incoming.is_file() or incoming.stat().st_size > 32 << 20:
            raise ValueError("supply a regular trial file of at most 32 MiB")
        data = incoming.read_bytes()
        yield from pilot_trials(incoming.name, digest(data), json.loads(data))


def update(state: Path, binding: str, *, archive: Path | None = None, incoming: Path | None = None):
    from training.experience_archive import pilot_trials
    from training.pilot_threads import fit, save

    if not re.fullmatch("[0-9a-f]{64}", binding):
        raise ValueError("expected the SHA-256 thread-learning binding")
    folder = state / "pilot"
    history = folder / (binding + ".training")
    history.mkdir(mode=0o700, parents=True, exist_ok=True)
    active = folder / (binding + ".candidate.npz")
    with (history / "update.lock").open("a") as lock, closing(sqlite3.connect(history / "trials.db")) as db:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db.execute("CREATE TABLE IF NOT EXISTS samples (id TEXT PRIMARY KEY, band INTEGER, eligible INTEGER,"
                   " sample TEXT NOT NULL, profile TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS sources (sample_id TEXT, source TEXT, sha256 TEXT, locator TEXT,"
                   " PRIMARY KEY(sample_id,source,sha256,locator))")
        added = 0
        with db:
            for item in entries(archive, incoming):
                if (item.get("schema") != "ironmule.experience.v1" or item.get("kind") != "pilot_thread_trial"
                        or item.get("binding") != binding or item.get("provenance") != "measured"):
                    continue
                group = {**item["workload"], "group": item["group"], "observations": item["metrics"]}
                band = group["context"] // 64
                if band not in range(1, 8) or group["group"] != f"ctx-bin-{band}":
                    raise ValueError("ambiguous context family")
                profile, = item["profiles"]
                if len(group["features"]) != 20 or len(group["observations"]) > 16:
                    raise ValueError("invalid Pilot sample")
                raw = json.dumps(group, sort_keys=True, allow_nan=False, separators=(",", ":"))
                checked = next(pilot_trials(item["source"], item["source_sha256"],
                                            {"binding": binding, "profile": profile, "groups": [group]}))
                eligible = item["label_eligible"] is True and checked["label_eligible"]
                identity = digest(raw.encode())
                added += db.execute("INSERT OR IGNORE INTO samples VALUES (?,?,?,?,?)",
                                    (identity, band, eligible, raw, json.dumps(profile))).rowcount
                db.execute("INSERT OR IGNORE INTO sources VALUES (?,?,?,?)",
                           (identity, item["source"], item["source_sha256"], item["locator"]))
        rows = []
        profile = None
        sample_ids = []
        for band in range(1, 8):
            # Older measurements and failures remain available in the journal. The fixed-size
            # fit retains the latest compatible cells within each disjoint context family.
            for identity, sample, details in db.execute(
                    "SELECT id,sample,profile FROM samples WHERE band=? AND (eligible=1 OR band>=6) "
                    "ORDER BY rowid DESC LIMIT ?", (band, PER_BAND)):
                profile = json.loads(details)
                rows.append(json.loads(sample))
                sample_ids.append(identity)
        receipt = {"binding": binding, "new_rows": added,
                   "stored_rows": db.execute("SELECT count(*) FROM samples").fetchone()[0], "fit_rows": len(rows)}
        if not profile:
            return {**receipt, "state": "awaiting comparable measurements"}
        record = {"schema": 1, "binding": binding, "profile": profile, "groups": rows}
        from training import pilot_threads

        examples = list(pilot_threads.examples(record))
        bands = {g["context"] // 64 for g, _, _ in examples}
        if not (bands & {1, 2, 3} and bands & {4, 5} and bands & {6, 7}):
            return {**receipt, "state": "awaiting complete grouped comparisons"}
        content = json.dumps(record, sort_keys=True, allow_nan=False).encode()
        revision = digest(content + Path(pilot_threads.__file__).read_bytes() + str(FITTER_SEED).encode())
        save(history / f"{revision}.inputs.json", {"sample_ids": sample_ids, "seed": FITTER_SEED,
                                                  "trainer_sha256": digest(Path(pilot_threads.__file__).read_bytes())})
        report = history / f"{revision}.json"
        checkpoint = history / f"{revision}.npz"
        with TemporaryDirectory(dir=history) as tmp:
            tmp = Path(tmp)
            if not report.exists() or not checkpoint.exists():
                trials = tmp / "trials.json"
                trials.write_bytes(content)
                fit(SimpleNamespace(trials=trials, weights=tmp / "model.npz", report=tmp / "report.json",
                                    seed=FITTER_SEED))
                (tmp / "model.npz").replace(checkpoint)
                (tmp / "report.json").replace(report)
            sha = digest(checkpoint.read_bytes())
            if sha != json.loads(report.read_text())["weights_sha256"]:
                raise ValueError("training report does not match checkpoint")
            changed = not active.exists() or digest(active.read_bytes()) != sha
            if changed:
                shutil.copyfile(checkpoint, tmp / "candidate.npz")
                (tmp / "candidate.npz").replace(active)
        outcome = {**receipt, "state": "candidate published" if changed else "checkpoint unchanged",
                   "weights_sha256": sha, "fit_revision": revision, "checkpoint": active.name}
        save(history / "last-update.json", outcome)
        return outcome


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, help="source-addressed IronMule JSONL experience archive")
    p.add_argument("--incoming", type=Path, help="fresh paired measurements from the CPU collector")
    p.add_argument("--binding", help="local learner binding (defaults to the incoming trial binding)")
    p.add_argument("--state", type=Path, required=True)
    args = p.parse_args()
    if not args.archive and not args.incoming:
        p.error("supply an experience archive or incoming trial file")
    if not args.binding and not args.incoming:
        p.error("supply --binding for archive-only updates")
    binding = args.binding or json.loads(args.incoming.read_text())["binding"]
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    os.nice(10)
    print(json.dumps(update(args.state, binding, archive=args.archive, incoming=args.incoming), sort_keys=True))


if __name__ == "__main__":
    main()
