"""The optional product release requires independent native and full HTTP qualification."""
import copy
from types import SimpleNamespace

import pytest

from ironmule.factual_gate import cases
from ironmule_product.errors import InvalidRequest
from ironmule_product.types import GenerationRequest
from lab.pk1 import native_http as http


def pairs():
    data = []
    for seed in http.SEEDS:
        cells = {str(w): {"wall_s": 1., "ttft_s": .2, "output_sha256": "fixed", "completed": w, "tokens": w * 32}
                 for w in http.GENERATIONS}
        facts = http.graded_wave(seed, [[c.expected, "stop", 16, 3] for c in cases(seed)], 2., [100.] * 18)
        row = {**cells["32"], "workload": {"seed": seed, "widths": list(http.GENERATIONS), "max_tokens": 32},
               "binding": "hardware", "evaluator_sha256": "same", "factual": facts,
               "peak_rss_kib": 123, "device_used_bytes": 1000, "cells": cells}
        cand = copy.deepcopy(row)
        cand["wall_s"] = cand["cells"]["32"]["wall_s"] = .95
        data.append((row, cand))
    return data


@pytest.mark.parametrize("change", ["facts", "work", "slow-one", "memory", "nan", "lost-pair",
                                   "graded-rate", "graded-ttft", "false-score", "foreign-binding", "wrong-seed"])
def test_http_pair_gate_requires_correct_work_across_widths_and_six_wins(change):
    rows = pairs()
    decision = http.verdict(rows)
    assert decision["passed"]
    wave = decision["waves"][0]
    assert wave["performance_completed_per_arm"] == 32 and wave["performance_outputs_equivalent"]
    assert wave["graded"]["cand"]["correct"] == 18 and wave["graded"]["cand"]["correct_completions_per_s"] == 9.
    assert decision["factual_ttft_ratios"] == [1.] * 6
    if change == "facts":
        for row in rows[0]:
            fact = row["factual"]
            fact["responses"][0][0] = "an incorrect factual answer"
            row["factual"] = http.graded_wave(row["workload"]["seed"], fact["responses"], fact["wall_s"], fact["first_token_ms"])
    elif change == "work":
        rows[0][1]["cells"]["8"]["tokens"] -= 1
    elif change == "slow-one":
        for _, cand in rows:
            cand["cells"]["1"]["ttft_s"] *= 2
    elif change == "memory":
        rows[0][1]["device_used_bytes"] *= 2
    elif change == "nan":
        rows[0][1]["cells"]["8"]["wall_s"] = float("nan")
    elif change in ("graded-rate", "graded-ttft"):
        for _, cand in rows:
            fact = cand["factual"]
            wall = fact["wall_s"] * (1.01 if change == "graded-rate" else 1.)
            ttft = [v * (1.01 if change == "graded-ttft" else 1.) for v in fact["first_token_ms"]]
            cand["factual"] = http.graded_wave(cand["workload"]["seed"], fact["responses"], wall, ttft)
    elif change == "false-score":
        rows[0][1]["factual"]["correct_completions_per_s"] = 99
    elif change == "foreign-binding":
        rows[2][1]["binding"] = "other"
    elif change == "wrong-seed":
        rows[-1] = rows[0]
    else:
        rows[0][1]["cells"]["32"]["wall_s"] = rows[0][1]["wall_s"] = rows[0][0]["wall_s"] * 1.01
    assert not http.verdict(rows)["passed"]


def test_measure_keeps_length_capped_completions_and_factual_rate_separate(monkeypatch):
    import os
    import ironmule_devices

    seed = http.SEEDS[0]
    def batch(engine, address, prompts, width, tokens, deadline):
        answers = [q.expected for q in cases(seed)] if width == 18 else ["partial answer"] * width
        return ([{"text": a, "finish_reason": "length" if width != 18 else "stop", "prompt_tokens": 52,
                  "tokens": 32 if width != 18 else 3, "ttft_ms": 125.} for a in answers], 3.)
    monkeypatch.setattr(http.compare, "batch", batch)
    monkeypatch.setattr(ironmule_devices, "ggml_devices", lambda: [{"provider": "cuda", "total_memory": 20, "free_memory": 8}])
    service = SimpleNamespace(live_samples=lambda: [])
    client = SimpleNamespace(_process=SimpleNamespace(pid=os.getpid()))
    row = http.measure({}, service, client, seed, "test-binding", 0)
    assert "correct" not in row and "correct" not in row["cells"]["32"]
    assert row["completed"] == 32 and row["tokens"] == 1024
    assert row["factual"]["correct"] == 18 and row["factual"]["correct_completions_per_s"] == 6.
    assert row["factual"]["first_token_ms"] == [125.] * 18 and row["factual"]["ttft_p95_s"] == .125


def test_live_bounds_use_matched_requests_and_measured_totals():
    sample = {"queue_ms": 23., "ttft_ms": 120., "prompt_tokens": 49, "completion_tokens": 32, "tokens_per_second": 64}
    assert http.live_bounds([sample] * 5) == {}
    bounds = http.live_bounds([sample] * 6)
    assert http.missed(sample, bounds) is False
    assert http.missed({**sample, "tokens_per_second": 24}, bounds) is True
    assert http.missed({**sample, "prompt_tokens": 200}, bounds) is None


def test_scope_checks_and_public_stream_use_the_product_service(monkeypatch):
    sentinel = object()
    seen = []
    def actual_stream(self, req):
        seen.append(req)
        return sentinel
    monkeypatch.setattr(http.ProductService, "stream", actual_stream)
    service = http.ScopedService.__new__(http.ScopedService)
    args = {"model": "gguf/fixture", "messages": (("user", "Describe a bridge"),), "max_tokens": 32}
    for bad in ({"max_tokens": 1}, {"temperature": .3}, {"messages": (("user", "x" * 1537),)},
                {"messages": (("assistant", "scope"),)}, {"stop": ("custom",)}):
        with pytest.raises(InvalidRequest):
            service.stream(GenerationRequest(**{**args, **bad}))
    real = GenerationRequest(**args)
    assert service.stream(real) is sentinel and seen == [real]


def test_release_client_does_not_forward_server_secrets(monkeypatch):
    launches = []
    from contextlib import ExitStack, nullcontext
    def isolated(layout, mode, args, timeout):
        launches.append((layout, mode, args, timeout))
        return nullcontext("isolated")
    monkeypatch.setattr(http, "launch", isolated)
    client = http.ReleaseClient.__new__(http.ReleaseClient)
    client.layout = SimpleNamespace(args=SimpleNamespace(timeout=9))
    with ExitStack() as client.resources:
        proc = client._launch_worker(["python", "-I", "worker.py", "--spec", "public"], {"GH_TOKEN": "dummy"})
    assert proc == "isolated" and launches == [(client.layout, "serve", ["--spec", "public"], 9)]


def test_native_and_http_rollback_remain_withdrawn_across_restart(tmp_path):
    args = SimpleNamespace(state=tmp_path)
    with http.deployment(tmp_path) as d:
        http.save(d / "selection.json", {"arm": "cand", "withdrawn": False})
    http.update_selection(args, {"arm": "cand", "withdrawn": False}, promote=True)
    http.main(["rollback", "--state", str(tmp_path)])
    with pytest.raises(ValueError, match="withdrawn"):
        http.update_selection(args, {"arm": "cand", "withdrawn": False}, promote=True)
    assert http.read(d / "http-selection.json")["arm"] == "ref"


def test_serve_reverts_failed_persistent_worker_and_starts_reference_on_same_port(monkeypatch, tmp_path):
    from contextlib import contextmanager
    evidence = tmp_path / "http-binding"
    evidence.mkdir()
    receipt = {"publication": {"pr": 4}}
    decision = {"decision": {"passed": True}, "binding": "binding", "control_receipt": "pin", "reference_bounds": {}}
    http.save(evidence / "decision.json", decision)
    for arm in ("ref", "cand"):
        http.save(evidence / f"gate-{arm}.json", {})
    http.save(tmp_path / "http-selection.json", {"arm": "cand", "withdrawn": False, "binding": "binding", "control_receipt": "pin"})
    monkeypatch.setattr(http, "assess", lambda *_: decision)
    monkeypatch.setattr(http, "native_selected", lambda *_: True)
    def withdraw(args, reason):
        http.save(tmp_path / "http-selection.json", {"arm": "ref", "withdrawn": True, "reason": reason})
    monkeypatch.setattr(http, "withdraw", withdraw)
    starts = []
    @contextmanager
    def server(args, receipt, cand, gate, port):
        starts.append((cand, port))
        if cand:
            raise ValueError("worker died")
        yield ({"base_url": "http://127.0.0.1"}, SimpleNamespace(ready=lambda: True, instance={}, health=lambda: {"completed_requests": 2}), SimpleNamespace(ready=True))
    monkeypatch.setattr(http, "server", server)
    http.live(SimpleNamespace(state=tmp_path, port=8230, lifetime=.05), tmp_path, receipt, "pin", "binding")
    assert starts == [(True, 8230), (False, 8230)]
    assert http.read(tmp_path / "http-selection.json")["withdrawn"]


def test_http_redirect_for_unmatched_public_input_checks_existing_reference(monkeypatch):
    import json
    import threading
    from contextlib import contextmanager
    from http.client import HTTPConnection

    from ironmule_product.http_server import create_server

    @contextmanager
    def frontend(service):
        address = create_server(service, "127.0.0.1", 0)
        thread = threading.Thread(target=address.serve_forever, daemon=True)
        thread.start()
        try:
            yield address.server_address[1]
        finally:
            address.shutdown()
            address.server_close()
            thread.join()

    class Existing:
        def ready(self):
            return True
        def health(self):
            return {"ready": True, "ggml": {"gate": {"passes": True}}}
        def models(self):
            return [{"id": "gguf/Qwen", "revision": "sha256:" + http.native_release.QWEN32, "loaded": True}]
    service = http.ScopedService.__new__(http.ScopedService)
    monkeypatch.setattr(service, "note", lambda *a, **kw: None)
    service.spec = SimpleNamespace(model_id="gguf/Qwen")
    with frontend(Existing()) as ref, frontend(service) as port:
        service.reference_url = f"http://127.0.0.1:{ref}"
        service.listen_port = port
        req = {"model": "gguf/Qwen", "messages": [{"role": "user", "content": "A longer output"}], "max_tokens": 160}
        client = HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            client.request("POST", "/v1/chat/completions", json.dumps(req), headers={"Content-Type": "application/json"})
            result = client.getresponse()
            assert result.status == 307 and result.getheader("Location") == service.reference_url + "/v1/chat/completions"
        finally:
            client.close()
        with pytest.raises(ValueError):
            http.reference_endpoint(service.reference_url, "gguf/other", port)
