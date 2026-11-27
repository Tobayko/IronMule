"""Protect the distinction between measured wall time, profiled calls and changed work."""

import json
from pathlib import Path

import pytest

from lab.pk6.native_dispatch import bottleneck

RESULTS = Path(__file__).resolve().parents[2] / "lab/autonomous/results"
PROBE = RESULTS / "l40s-mixed-bottleneck"


def test_counted_receipts_replay():
    result = bottleneck.replay(RESULTS / "l40s-native-dispatch", PROBE)
    assert json.loads(json.dumps(result)) == json.loads(
        (PROBE / "summary.json").read_bytes()
    )


def test_span_witness_required(tmp_path):
    for file in (PROBE / "span").glob("*.json"):
        (tmp_path / file.name).symlink_to(file)
    receipt = tmp_path / "intervention-180341.json"
    rows = json.loads(receipt.read_bytes())
    rows[1]["effective"] = rows[1]["natural"]
    receipt.unlink()
    receipt.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="span or order mismatch"):
        bottleneck.intervention(tmp_path)


def test_rejects_initial_comparison_with_different_kernel_settings(tmp_path):
    (tmp_path / "R1").symlink_to(PROBE / "http/R1", target_is_directory=True)
    (tmp_path / "H1").symlink_to(PROBE / "exploratory/H1", target_is_directory=True)
    target = json.loads((PROBE / "span/target.json").read_bytes())
    with pytest.raises(ValueError, match="unmatched kernel profile"):
        bottleneck.served(tmp_path, RESULTS / "l40s-native-dispatch", target)
