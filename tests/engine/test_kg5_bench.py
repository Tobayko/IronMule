"""KG5's bounded experiment can refuse a candidate before another GPU stage is spent."""
import json
import subprocess
import sys
import time

import numpy as np
import pytest

from lab.kg5 import bench, worker


def test_paired_gate_requires_a_measured_dispatch_numeric_agreement_and_ten_percent(tmp_path):
    ref, test = tmp_path / "off.npy", tmp_path / "on.npy"
    np.save(ref, np.zeros((4, 3)))
    np.save(test, np.ones((4, 3)))  # softmax is invariant under a common offset
    off = {"hits": 0, "step_ms": [10] * 12, "logits": ref}
    on = {"hits": 32, "step_ms": [9] * 12, "logits": test}
    assert bench.evaluate_steps(off, on)["passed"]
    assert not bench.evaluate_steps(off, {**on, "step_ms": [9.1] * 12})["passed"]
    with pytest.raises(bench.Rejected):
        bench.evaluate_steps(off, {**on, "hits": 0})
    np.save(test, np.tile([1, 2, 3], (4, 1)))
    assert not bench.evaluate_steps(off, on)["passed"]
    np.save(test, np.full((4, 3), np.nan))
    with pytest.raises(bench.Rejected):
        bench.evaluate_steps(off, on)


def test_http_confirmation_tracks_each_fact_and_tail_latency():
    facts = {q: True for q, _ in worker.questions()}
    off = {"hits": 0, "tok_s": 100, "ttft_median_s": 2, "ttft_p95_s": 3, "facts": facts, "fact_hits": [0, 0]}
    on = {**off, "hits": 100, "tok_s": 120, "fact_hits": [32, 32]}
    assert bench.evaluate_http(off, on)["passed"]
    assert not bench.evaluate_http(off, {**on, "ttft_p95_s": 4})["passed"]
    with pytest.raises(bench.Rejected):
        bench.evaluate_http(off, {**on, "fact_hits": [32, 0]})
    changed = {**facts, next(iter(facts)): False}
    assert not bench.evaluate_http(off, {**on, "facts": changed})["passed"]


def test_budget_terminates_subprocess_and_persists_timeout(tmp_path, monkeypatch):
    report = tmp_path / "case.json"
    def slow(args, record, temp, deadline, save):
        bench.process([sys.executable, "-c", "import time; time.sleep(15)"], deadline)
    monkeypatch.setattr(bench, "run", slow)
    assert bench.main(["--stage", "smoke", "--budget-s", "1", "--output", str(report)]) == 124
    assert json.loads(report.read_text())["status"] == "timeout"


def test_counter_snapshot_reads_the_isolated_serving_worker(tmp_path, monkeypatch):
    product = tmp_path / "ironmule_product"
    product.mkdir()
    (product / "worker.py").write_text("import time\ntime.sleep(30)\n")
    fake_counter = tmp_path / "worker.py"
    fake_counter.write_text("def counter(): return lambda: 42\n")
    monkeypatch.setattr(worker, "__file__", str(fake_counter))
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    folder = tmp_path / "instrumented"
    folder.mkdir()
    env = worker.instrumented_server_environment(folder)
    proc = subprocess.Popen([sys.executable, "-I", folder / "ironmule_product/worker.py"], env=env)
    try:
        for _ in range(50):
            if (folder / "pid").exists() or proc.poll() is not None:
                break
            time.sleep(0.05)
        assert worker.serving_count(folder) == 42
    finally:
        proc.terminate()
        proc.wait(timeout=5)
