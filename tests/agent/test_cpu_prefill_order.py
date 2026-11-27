import json
from pathlib import Path

import pytest

from training.cpu_prefill_order import plan
from training.experiment_planner import DEFERRED, ORDER, digest

ROOT = Path(__file__).resolve().parents[2]
CPU = ROOT / "lab/autonomous/results/runpod-epyc-pilot"


def test_native_prefill_order_requires_real_qualified_serving_and_bound_native_time(tmp_path):
    prior_path = ROOT / "evidence/runtime/cpu-pilot-epyc-discovery.json"
    trace = json.loads((CPU / "prefill-attribution.json").read_text())
    a = tmp_path / "trace.json"
    a.write_text(json.dumps(trace))
    args = (prior_path, CPU / "waves", a, CPU / "planner-decision.json", None, ROOT)
    order = plan(*args)
    assert order["schema"] == ORDER and order["target"]["function"] == "GgmlEngine._prefill"
    assert digest({k: v for k, v in order.items() if k != "order_id"}) == order["order_id"]
    trace["measured_choices"] = 0
    a.write_text(json.dumps(trace))
    assert plan(*args)["schema"] == DEFERRED
    trace["windows"][0]["output_sha256"] = "wrong"
    a.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="changed fixed work"):
        plan(*args)


def test_archived_prefill_source_must_match_measured_bytes_without_git_history(tmp_path):
    import gzip
    import subprocess

    trace = CPU / "prefill-attribution.json"
    host = tmp_path / "planner-decision.json"
    host.write_bytes((CPU / host.name).read_bytes())
    measured = json.loads(trace.read_text())
    empty = tmp_path / "empty"
    subprocess.run(["git", "init", "-q", empty], check=True)
    snapshot = tmp_path / "sources" / f"{measured['base_commit']}-ggml_engine.py.gz"
    snapshot.parent.mkdir()
    snapshot.write_bytes((CPU / "sources" / snapshot.name).read_bytes())
    args = (ROOT / "evidence/runtime/cpu-pilot-epyc-discovery.json", CPU / "waves", trace, host, None)
    replay = plan(*args, empty)
    assert replay == plan(*args, ROOT)
    assert replay["base_commit"] == measured["base_commit"]
    snapshot.write_bytes(gzip.compress(b"unrelated source"))
    with pytest.raises(ValueError, match="source does not match"):
        plan(*args, empty)
