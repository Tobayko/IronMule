"""Real startup gains require equal graded service from both released product sources."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ironmule.factual_gate import cases
from lab.pk1 import native_http
from lab.pk1 import restart_http as product
from training import scoped_release as release


def pair(seed):
    def factual(s):
        return native_http.graded_wave(s, [[q.expected, "stop", 45, 2] for q in cases(s)], 2., [120.] * 18)
    base = {"seed": seed, "source_sha256": "ref", "startup_s": 5., "cold_completed_s": 7.1,
            "cold": factual(seed), "warm": factual(seed ^ 733), "perf_completed": 32, "perf_output_sha256": "fixed",
            "perf_tokens": 960, "perf_wall_s": 2.8, "perf_ttft_s": .2, "peak_rss": 100, "device_bytes": 50,
            "startup_decision": {"profile": "reference", "previous": True, "cpu_ms": .2, "family": "factual-first-18"}}
    candidate = deepcopy(base)
    candidate.update(source_sha256="new", startup_s=3.5, cold_completed_s=5.6)
    candidate["startup_decision"].update(profile="candidate", previous=False)
    for row, ablate in ((base, True), (candidate, False)):
        row["startup_decision"].update(previous=False, ablate_weights=ablate)
        row["cold"]["wall_s"] += row["startup_s"]
        row["cold"]["correct_completions_per_s"] = 18 / row["cold"]["wall_s"]
        row["cold_admission"] = {"request_received_monotonic_ns": 5, "worker_ready_monotonic_ns": 70, "joined_requests": 18,
                                 "policy": row["startup_decision"]}
    return base, candidate


@pytest.mark.parametrize("problem", ["latency", "long_output", "facts", "changed_source", "untrained", "memory", "one_loss"])
def test_cold_gains_preserve_warm_facts_memory_and_32_client_completions(problem):
    pairs = [pair(i) for i in product.SEEDS]
    q = {"candidate_sha256": "new", "reference_sha256": "ref"}
    assert product.score(pairs, q)["passed"]
    for r, c in pairs:
        if problem == "latency":
            c["perf_ttft_s"] *= 1.03
        elif problem == "long_output":
            c["perf_tokens"] += 4
        elif problem == "facts":
            c["warm"]["responses"][0][0] = "incorrect"
        elif problem == "changed_source":
            c["source_sha256"] = r["source_sha256"]
        elif problem == "untrained":
            c["startup_decision"]["previous"] = True
        elif problem == "memory":
            c["device_bytes"] *= 1.1
        else:
            c["cold_completed_s"] = r["cold_completed_s"]
            break
    if problem in ("long_output", "facts", "changed_source", "untrained"):
        with pytest.raises(ValueError):
            product.score(pairs, q)
    else:
        assert not product.score(pairs, q)["passed"]


def test_live_http_withdrawal_starts_reference_with_same_graded_input(tmp_path, monkeypatch):
    from contextlib import contextmanager

    raw = {"passed": True, "reference_bounds": {}}
    native_http.save(tmp_path / "decision.json", raw)
    for name in ("reference", "candidate"):
        release.save(tmp_path / f"gate-{name}.json", {})
    monkeypatch.setattr(product, "assess", lambda *_: raw)
    import time
    starts = []
    def choose(args, receipt, *, withdrawn, previous=False, ablate=False, family=None):
        name = "reference" if previous or withdrawn or ablate or family != "factual-first-18" else "candidate"
        return {"profile": name, "previous": previous, "cpu_ms": .1}
    monkeypatch.setattr(product, "choose", choose)
    answers = {q.prompt: q.expected for q in cases(44)}
    class Existing:
        def ready(self):
            return True
        def health(self):
            return {"completed_requests": 18}
        def outstanding(self):
            return 0
        def stream(self, req):
            return iter([{"type": "token", "text": answers[req.messages[0][1]]}, {"type": "done", "finish_reason": "stop",
                         "prompt_tokens": 49, "completion_tokens": 2}])
    @contextmanager
    def server(args, receipt, cand, *a, **kw):
        starts.append(cand)
        assert kw["listen"] is False
        yield {}, Existing(), SimpleNamespace(_process=SimpleNamespace(pid=42), ready=True)
    monkeypatch.setattr(native_http, "server", server)
    receipt = {"qualification": {"candidate_sha256": "new", "reference_sha256": "ref", "binding": {"runtime": {"settings": {}}}}}
    shape = product.byte_shape([len(q.prompt.encode()) for q in cases(44)])
    args = SimpleNamespace(state=tmp_path, gguf=tmp_path / "model.gguf", port=0, deadline=time.monotonic() + 15,
                           geometry_sha256="fixture", geometry_record={"count": 18, "minimum": shape, "maximum": shape})
    withdraw = tmp_path / "withdrawn.json"
    before = product.serve(args, receipt, tmp_path, withdraw, seed=44, lifetime=0)
    release.save(withdraw, {"withdrawn": True})
    after = product.serve(args, receipt, tmp_path, withdraw, seed=44, lifetime=0)
    assert starts == [True, False] and before["arm"] == "candidate" and after["arm"] == "reference"
    for result in [before, after]:
        c = result["completions"][0]
        assert c["graded_correct"] == 18 and c["cold_admission"]["first_request_id"]
        assert c["cold_admission"]["joined_requests"] == 18
        assert c["cold_admission"]["worker_ready_monotonic_ns"] > c["cold_admission"]["request_received_monotonic_ns"]
    assert before["completions"][0]["work_sha256"] == after["completions"][0]["work_sha256"]


@pytest.mark.parametrize("count,mixed,chosen", [(18, False, "candidate"), (17, False, "reference"),
                                                (19, False, "reference"), (18, True, "reference")])
def test_unseen_inputs_drive_actual_cold_http_admission_without_prompt_matching(tmp_path, monkeypatch, count, mixed, chosen):
    import time
    from contextlib import contextmanager
    # Many callers may send the same text. Only independently observed sizes and arrival geometry matter.
    short = "Please summarize the attached maintenance note."
    long = "Please review this maintenance note and summarize the recorded delay in plain language for the evening team."
    prompts = [short] * (count - 6) + [long] * 6
    base = product.byte_shape([len(short.encode())] * 12 + [len(long.encode())] * 6)
    if mixed:
        prompts[-1] = long * 4
    args = SimpleNamespace(state=tmp_path, gguf=tmp_path / "fixture.gguf", deadline=time.monotonic() + 10,
                           geometry_sha256="fixture", geometry_record={"count": 18, "minimum": base, "maximum": base})
    receipt = {"qualification": {"binding": {"runtime": {"settings": {}}}}}
    def select(_, __, *, withdrawn, previous=False, ablate=False, family=None):
        return {"profile": "candidate" if family == "factual-first-18" and not previous else "reference"}
    monkeypatch.setattr(product, "choose", select)
    class Service:
        def ready(self):
            return True
        def outstanding(self):
            return 0
        def stream(self, req):
            return iter([{"type": "token", "text": "Delivered"}, {"type": "done", "finish_reason": "stop",
                         "prompt_tokens": 15, "completion_tokens": 1}])
    @contextmanager
    def worker(*a, **kw):
        yield {}, Service(), SimpleNamespace(_process=SimpleNamespace(pid=12))
    monkeypatch.setattr(native_http, "server", worker)
    with product.cold_endpoint(args, receipt, {"candidate": {}, "reference": {}}, withdrawn=lambda: False) as (addr, front, status):
        assert front.backend is None and not status
        rows, _ = product.compare.batch("ironmule", addr, prompts, count, 32, args.deadline)
        assert status["arm"] == chosen and len(rows) == count
