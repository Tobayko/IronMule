"""Exercise the product completion and idle hooks, privacy boundary, and offline-worker exclusion."""
import json
import threading
import time
from types import SimpleNamespace

from ironmule.ggml_agent import GgmlAgent
from ironmule_product.experiment_queue import ExperimentQueue, WINDOW
from ironmule_product.ggml_bridge import GgmlBridge
from training.demand_worker import consume


def test_completed_work_queues_one_aggregate_and_offline_consumer_defers(tmp_path):
    queue = tmp_path / "queue"
    agent = GgmlAgent(None, {}, {"only": {"seqs": 4, "step_one": .01, "step_full": .03}}, "only")
    bridge = GgmlBridge.__new__(GgmlBridge)
    bridge._agent, bridge._admission = agent, None
    bridge._experiment_queue = ExperimentQueue(queue, 0)
    bridge._gguf_sha256 = "a" * 64
    bridge._demand_lock = bridge._run_lock = threading.Lock()
    bridge._eos = ()
    bridge._tokenizer = SimpleNamespace(decode=lambda ids: "ok")
    bridge.metadata = bridge.counters = dict

    def run(jobs, **unused):
        for job in jobs:
            job.tokens = [98765]
            job.reason = "length"
        return jobs
    bridge._engine = SimpleNamespace(run=run, close=lambda: None)
    results = bridge.generate_many([[1234567] * (20 if i % 2 else 200) for i in range(WINDOW)], 1)
    assert len(results) == agent.requests == WINDOW
    assert not (queue / "pending.json").exists()
    # Run one idle iteration through the actual bridge hook without an artificial busy wait.
    calls = iter((False, True))
    bridge._stop = SimpleNamespace(wait=lambda t: next(calls), set=lambda: None, is_set=lambda: False)
    bridge._last_run = time.monotonic() - 10
    bridge._idle_loop()
    ticket = json.loads((queue / "pending.json").read_text())
    assert (ticket["short"], ticket["document"]) == (16, 16)
    assert sorted(ticket) == sorted(("version", "id", "producer", "model_sha256", "short", "document"))
    assert consume(SimpleNamespace(queue=queue, state=tmp_path / "evaluation", wait=0)).startswith("deferred")
    bridge.close()
    assert (queue / "pending.json").is_file()


def test_consumer_can_complete_a_subscription_order_and_request_gated_publication(tmp_path, monkeypatch):
    import json
    from training import demand_worker

    q = tmp_path / "queue"
    q.mkdir()
    model = tmp_path / "model"
    model.write_bytes(b"test")
    ticket = {"version": 1, "id": "9" * 32, "producer": "0" * 64, "model_sha256": "f" * 64,
              "short": 32, "document": 0}
    (q / "pending.json").write_text(json.dumps(ticket))
    state = tmp_path / "eval"
    run = state / "runs" / ("demand-" + ticket["id"])
    (run / "trial/author").mkdir(parents=True)
    (run / "trial/author/request.json").write_text("{}")
    def start(argv, **kwargs):
        assert "--publish" in argv and "--demand-summary" in argv
        assert kwargs["env"]["IRONMULE_CODE_AUTHOR"] == "cli"
        (run / "complete.json").write_text("{}")
        return SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr(demand_worker.subprocess, "Popen", start)
    args = SimpleNamespace(queue=q, state=state, gguf=model, wait=0, seconds=60, cores=1, provider="cpu",
                           thread_candidate=None, publish=True, author="cli")
    assert demand_worker.consume(args) == "complete"
    assert not (q / "pending.json").exists()
