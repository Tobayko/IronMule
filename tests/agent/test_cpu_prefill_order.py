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
