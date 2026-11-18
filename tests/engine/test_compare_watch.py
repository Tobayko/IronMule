import json

import pytest

from ironmule_product import compare_watch as watch
from ironmule_product.state import _atomic_write


def test_watch_rechecks_promotions_archives_failures_and_caps_after_restart(tmp_path, monkeypatch):
    output = tmp_path / "progress.json"
    argv = ["--total-seconds", "2", "--", "--gguf", "model", "--tokenizer", "tokenizer",
            "--llama-server", "stock", "--vllm", "vllm", "--output", str(output), "--seed", "7"]
    received = []
    monkeypatch.setattr(watch, "_fingerprint", lambda *_: "gpu-a")

    def run(options):
        args = watch.compare.arguments(options)
        received.append((args.seed, args.budget_seconds, args.agent_dir))
        status = "complete" if len(received) == 1 else "error"
        _atomic_write(output, {"run_id": str(args.seed), "status": status, "protocol": {"seed": args.seed},
                               "learning": {"status": "qualified" if len(received) == 1 else "deferred"}})
        return int(status == "error")

    monkeypatch.setattr(watch.compare, "main", run)
    monkeypatch.setattr(watch.time, "sleep", lambda *_: (_ for _ in ()).throw(AssertionError("unexpected wait")))
    assert watch.main(argv) == watch.main(argv) == 0
    history = tmp_path / "progress.json.watch"
    assert received == [(7, 2, history / "agent"), (8, 1, history / "agent")]
    assert json.loads((history / "cycle-000002.json").read_text())["status"] == "error"


def test_watch_charges_crashed_run_and_uses_changed_weights(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.write_bytes(b"a")
    output = tmp_path / "progress.json"
    options = ["--gguf", str(model), "--tokenizer", str(tmp_path / "tokenizer"), "--llama-server", "stock",
               "--vllm", "vllm", "--output", str(output)]
    args = watch.compare.arguments(options)
    fingerprint = watch._fingerprint(args, options)
    model.write_bytes(b"ab")
    assert watch._fingerprint(args, options) != fingerprint

    def crash(args):
        raise OSError("worker lost")

    monkeypatch.setattr(watch.compare, "main", crash)
    argv = ["--total-seconds", "3", "--", *options]
    with pytest.raises(OSError, match="worker lost"):
        watch.main(argv)
    assert watch.main(argv) == 0
    state = json.loads((output.parent / (output.name + ".watch") / "watch.json").read_text())
    assert state["cycles"] == 1 and state["charged_seconds"] == 3
