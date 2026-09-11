#!/usr/bin/env python3
"""State what B52 evidence exists, what is lost, and which code each run describes.

The first B52 execution wrote its record to the same path as the second, which
overwrote it. This tool does not reconstruct the lost run. It records that the loss
happened, what was searched for it, and the checksums and code binding of what is
actually on disk, so a reader can tell a retained measurement from a missing one.

The tool only reads. It writes one record, once, and refuses to overwrite it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b52_automatic_selection import source_binding, write_once  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "research" / "raw"


def _digest(path: Path) -> dict | None:
    path = path.resolve()
    if not path.is_file():
        return None
    data = path.read_bytes()
    return {"path": str(path.relative_to(PROJECT_ROOT)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime,
                                                tz=timezone.utc).isoformat()}


def _local_snapshots() -> list[str]:
    out = subprocess.run(["tmutil", "listlocalsnapshots", "/"],
                         capture_output=True, text=True, check=False).stdout
    return [line.strip() for line in out.splitlines() if line.strip().startswith("com.apple")]


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--retained", type=Path, required=True,
                        help="the surviving measuring record")
    parser.add_argument("--preregistration", type=Path, required=True)
    args = parser.parse_args()

    retained = _digest(args.retained)
    if retained is None:
        raise SystemExit(f"the retained record is not there: {args.retained}")
    record_body = json.loads(args.retained.read_text())

    searched = {
        "same_path_second_write": "the second run wrote the same file name; no copy was kept",
        "evidence_archive": sorted(p.name for p in
                                   (PROJECT_ROOT / ".friday-data" /
                                    "ironmule-evidence-archive").rglob("*B52*")),
        "research_raw_siblings": sorted(p.name for p in RAW.glob("B52*")),
        "apfs_local_snapshots": _local_snapshots(),
        "ssot_index": "rebuilt and atomically replaced after the second run; "
                      "the earlier index is gone with it",
    }
    record = {
        "schema": "ironmule.b52_evidence_provenance.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "evidence_provenance",
        "status": "recorded",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "runs": {
            "B52_automatic_selection_20260909_attempt1": {
                "state": "EVIDENCE_LOST",
                "raw_record": None,
                "attested_by": "the working session that executed it, not by a retained "
                               "artifact",
                "reason": "the second execution wrote the same output path and replaced it",
                "recoverable": False,
                "values_not_restated": "no number from this run is carried into any "
                                       "release claim; it counts as not measured",
            },
            "B52_automatic_selection_20260909": {
                "state": "RETAINED",
                "raw_record": retained,
                "observed_at": record_body.get("observed_at"),
                "verdict": record_body.get("verdict"),
                "git_revision": record_body.get("git_revision"),
                "code_binding": "none recorded; the tool gained `source_binding` after "
                                "this run, so this record names only a git revision of "
                                "an uncommitted tree",
                "superseded_by": "the release run measured after the repair work",
            },
        },
        "recovery_attempted": searched,
        "preregistration": _digest(args.preregistration),
        "source_binding_now": source_binding(),
        "prevention": [
            "every run writes its own output id; the tool refuses an existing path",
            "raw records are written to a .partial file and moved into place atomically",
            "each measuring record carries a source binding over the files it measures",
        ],
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"written": str(args.out),
                      "lost_runs": 1, "retained_runs": 1}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
