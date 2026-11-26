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


def test_faster_median_cannot_override_throughput_tail_or_missing_leading_group():
    from lab.pk6.first_token.grade import paired

    def arm(candidate):
        rate, first, tail = (99.0, 500., 2100.) if candidate else (100., 1600., 2000.)
        stats = [{"p50_ms": first, "p95_ms": tail, "max_ms": tail + 5,
                  "prefill_calls": [{"width": 32, "short": 16}]} for _ in range(3)]
        return {"profile": {"pilot": True, "pk2_min_rows": 128}, "phases": stats, "waves": [
            {"repeat": i, "tok_s": rate, "ttft_p50_ms": first, "ttft_p95_ms": tail, "tokens": 1024,
             "prompt_tokens": 7632, "texts": "same"} for i in range(3)],
             "quality": [{"quality": "18/18", "tokens": 256, "prompt_tokens": 1908} for _ in range(2)]}

    checks = paired(arm(False), arm(True))["checks"]
    assert checks["p50"] and checks["quality"] and checks["paired_work"]
    assert not any(checks[k] for k in ("throughput", "p95", "prefill_groups"))


def test_measured_trial_replays_and_archives_the_rejection_as_one_experiment_group():
    import hashlib

    from lab.pk6.first_token import grade
    from training.experience_archive import first_token_trial

    folder = phases.RESULTS / "a100-first-token"
    path = folder / "result.json"
    result = grade.grade(folder)
    assert result == json.loads(path.read_bytes())
    assert result["decision"] == "retain_reference"
    assert all(p["checks"]["p50"] and not p["checks"]["p95"] for p in result["pairs"].values())
    rows = list(first_token_trial(phases.ROOT, str(path.relative_to(phases.ROOT)), hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(rows) == 2 and {r["group"] for r in rows} == {folder.name}
    assert all(r["reason"] == "retain_reference" and not r["label_eligible"] for r in rows)
