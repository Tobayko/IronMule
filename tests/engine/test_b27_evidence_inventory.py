import json
from pathlib import Path

import pytest

from research.b27_evidence_inventory import inventory, markdown


def _source(root: Path, *, duplicate: bytes | None = None) -> None:
    raw = root / "research" / "raw"
    raw.mkdir(parents=True)
    result = {
        "schema": "example.v1",
        "status": "QUALIFIED",
        "activation_allowed": False,
        "environment": {"hardware": "test"},
        "workload": {"requests": 2},
        "baseline": {"samples": [2.0, 2.1]},
        "candidate": {"samples": [1.0, 1.1]},
        "correctness": {"token_identity": True},
        "resources": {"peak_memory": 10, "swap": 0},
        "provenance": {"commit": "abc", "revision": "model"},
    }
    (raw / "E1.json").write_text(json.dumps(result))
    (raw / "E1_preregistration.md").write_text("# prereg\n")
    if duplicate is not None:
        (raw / "copy.json").write_bytes(duplicate)


def test_inventory_separates_sources_deduplicates_and_reports_coverage(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _source(first)
    duplicate = (first / "research" / "raw" / "E1.json").read_bytes()
    _source(second, duplicate=duplicate)

    report = inventory([("branch", first), ("local", second)],
                       generated_at="2026-08-28T00:00:00+00:00")

    assert report["schema"] == "ironmule.evidence_inventory.v1"
    assert report["summary"]["artifacts"] == 5
    assert report["summary"]["unique_artifacts"] == 2
    assert report["summary"]["duplicate_groups"] == 2
    assert report["coverage"]["correctness"] == {"present": 3, "eligible_json": 3}
    assert all("/Users/" not in json.dumps(row) for row in report["records"])
    assert "read-only corpus audit" in markdown(report)


def test_inventory_rejects_missing_or_ambiguous_sources(tmp_path):
    with pytest.raises(ValueError, match="no research/raw"):
        inventory([("missing", tmp_path)])

    first = tmp_path / "first"
    second = tmp_path / "second"
    _source(first)
    _source(second)
    with pytest.raises(ValueError, match="unique"):
        inventory([("same", first), ("same", second)])
