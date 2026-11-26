import json
import time
from types import SimpleNamespace

from lab.pk1 import restart_pilot_run as entry
from training import pilot_restart_profiles as policy


def test_per_workload_reference_preserves_durable_eligibility(tmp_path, monkeypatch):
    state = tmp_path / "state"
    (state / "deployment").mkdir(parents=True)
    (state / "deployment/control.json").write_text("{}")
    trial = tmp_path / "trial"
    trial.mkdir()
    cache = trial / "receipt.json"
    cache.write_text(json.dumps({"created_s": time.time()}))
    receipt = {"order": {"order_id": "order"}, "publication": {"pr": 1},
               "qualification": {"reference_sha256": "reference-source", "candidate_sha256": "candidate-source",
                                 "binding": {"cache_receipt_sha256": entry.scoped_release.checksum(cache)}}}
    args = SimpleNamespace(trial=trial, state=state, receipt_sha256="control", weights=tmp_path / "weights",
                           weights_sha256="weights", seed=19, ablate_weights=True)
    monkeypatch.setattr(entry.scoped_release, "pinned", lambda *a: receipt)
    monkeypatch.setattr(entry.scoped_release, "selected", lambda *a: "candidate")
    def select(_weights, _control, *, family, cache_ready, active, ablate_weights, expected_sha256):
        assert active and cache_ready and family == "mixed-first-32" and expected_sha256 == "weights"
        return {"profile": "reference" if ablate_weights else "candidate"}
    def run(_args, route, seed):
        assert (route / "deployment/control.json").read_bytes() == b"{}" and seed == 19
        arm = "candidate" if route == state else "reference"
        return {"arm": arm, "source_sha256": arm + "-source", "cache_used": arm == "candidate",
                "graded_correct": 18, "pilot_decisions": 1, "warm_performance_completions": 32,
                "cold_performance_completions": 32}
    monkeypatch.setattr(policy, "select", select)
    monkeypatch.setattr(entry.restart_release, "run", run)
    assert entry.serve(args)["work"]["arm"] == "reference"
    args.ablate_weights = False
    assert entry.serve(args)["work"]["arm"] == "candidate"
    assert list((state / "deployment").iterdir()) == [state / "deployment/control.json"]
