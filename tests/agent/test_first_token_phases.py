import json
from pathlib import Path

import pytest

from lab.pk6.first_token import phases

RESULTS = phases.RESULTS / "a100-warm-ttft"


def test_historical_replay_binds_all_six_packed_calls_and_nine_client_waves():
    report = json.loads((RESULTS / "phases.json").read_bytes())
    for label, expected in report.items():
        records = phases.clients(phases.HISTORICAL, label)
        src = RESULTS / f"native-{label}.json"
        native = json.loads(src.read_bytes()) if src.exists() else None
        assert phases.summarize(records, native) == expected
        if native:
            old = phases.RESULTS / f"a100-dynamic-calibration/init-{label}.json"
            assert native["trace_sha256"] == json.loads(old.read_bytes())["trace_sha256"]
            assert [w["prefill_calls"][0]["width"] for w in expected] == [32] * 3


@pytest.mark.parametrize("error", ["foreign_id", "clock", "sequence", "unaccounted_call"])
def test_tampered_join_is_rejected(error):
    rows = phases.clients(phases.HISTORICAL, "K2")
    native = json.loads((RESULTS / "native-K2.json").read_bytes())
    job = native["jobs"][rows[0]["server_id"]]
    if error == "foreign_id":
        native["jobs"]["foreign"] = native["jobs"].pop(rows[0]["server_id"])
    elif error == "clock":
        job["seen"] = job["prefill_end"]
    elif error == "sequence":
        job["seq"] = -1
    else:
        native["calls"].append(dict(native["calls"][0], start=1.0))
    with pytest.raises(ValueError):
        phases.summarize(rows, native)


def test_extraction_rejects_duplicate_native_completion(tmp_path):
    records = phases.clients(phases.HISTORICAL, "K2")
    row = records[0]
    trace = Path(tmp_path) / "native.trace"
    line = (f"# job seq=1 tag={row['server_id']} prompt=33 reused=2 tokens=32 reason=length "
            "seen=1 prefill_start=2 prefill_end=3 first=4 end=5\n")
    trace.write_text(line * 2)
    with pytest.raises(ValueError, match="duplicate"):
        phases.project(trace, records)
