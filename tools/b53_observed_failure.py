#!/usr/bin/env python3
"""Turn the one JUnit file that recorded the B53 failure into an indexed record.

The failure was seen once, in a full-suite run, and the only artefact is that run's
JUnit XML. This reads it, extracts the three failing cases with the byte prefixes
pytest printed, decodes those prefixes as bfloat16 so the size of the disagreement is
on the record, and writes the result once. Nothing here re-runs anything.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import struct
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b52_automatic_selection import write_once  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "test_both_kernels_match_the_library_byte_for_byte"


def _bfloat16(prefix: bytes) -> list[float]:
    """Decode whole little-endian bfloat16 values out of a byte prefix."""

    values = []
    for index in range(0, len(prefix) - 1, 2):
        raw = struct.unpack("<H", prefix[index:index + 2])[0]
        values.append(struct.unpack("<f", struct.pack("<I", raw << 16))[0])
    return values


def _leading_bytes(literal: str) -> bytes:
    """The real leading bytes of a byte literal pytest truncated with an ellipsis.

    The elision can cut an escape in half, so the literal is shortened one character at
    a time until it parses. Only the part before the ellipsis is real data.
    """

    body = literal[2:-1].split("...", 1)[0]
    while body:
        try:
            return ast.literal_eval("b'" + body + "'")
        except (SyntaxError, ValueError):
            body = body[:-1]
    return b""


def _prefixes(message: str) -> dict:
    """The two byte literals pytest put in the assertion message, as bytes."""

    match = re.search(r"assert (b'.*?') == (b'.*?')\n", message + "\n")
    if match is None:
        return {}
    got, expected = (_leading_bytes(part) for part in match.groups())
    return {"observed_prefix_hex": got.hex(), "expected_prefix_hex": expected.hex(),
            "observed_bfloat16": _bfloat16(got), "expected_bfloat16": _bfloat16(expected)}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = args.junit.read_bytes()
    root = ET.fromstring(data)
    suite = root if root.tag == "testsuite" else root[0]

    cases = {}
    for case in suite:
        if TARGET not in (case.get("name") or ""):
            continue
        failure = case.find("failure")
        if failure is None:
            continue
        message = failure.get("message") or ""
        cases[case.get("name")] = {
            "classname": case.get("classname"),
            "seconds": float(case.get("time") or 0.0),
            "message": message,
            "first_differing_index": 0 if "At index 0 diff" in message else None,
            **_prefixes(message),
        }

    record = {
        "schema": "ironmule.b53_observed_failure.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "observed_failure",
        "status": "recorded",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "source_junit": {
            "path": str(args.junit.resolve().relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        },
        "run": {
            "started": suite.get("timestamp"),
            "hostname": suite.get("hostname"),
            "seconds": float(suite.get("time") or 0.0),
            "tests": int(suite.get("tests") or 0),
            "failures": int(suite.get("failures") or 0),
            "skipped": int(suite.get("skipped") or 0),
            "invocation": 'pytest tests -m "not integration"; pytest.ini supplies '
                          "-n auto --dist loadfile",
        },
        "code_version": {
            "git_head": "df15e59c4403da0495faf1695c421a3c85d2a5d5",
            "worktree": "uncommitted; tracked changes and the file list are kept beside "
                        "this record as tracked.diff and status.txt",
            "test_at_the_time": "inputs were unseeded mx.random.normal draws; the seeding "
                                "and the input digest in the message came afterwards",
        },
        "cases": cases,
        "what_the_bytes_say": (
            "the observed values are ordinary bfloat16 numbers of the same order as the "
            "expected ones, not uninitialised memory; the disagreement is a few per cent, "
            "which is far above one unit in the last place"
        ),
        "not_established": [
            "which of the three arms is wrong: the port, the specialisation or the "
            "library reference; the run compared each kernel against the library only",
            "whether the inputs the two arms saw were the same, because they were not "
            "captured",
        ],
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"cases": sorted(cases), "written": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
