"""SP2: speculation from the context with a window learned online (`ironmule/speculation.py`) on the CPU stacked plan.

The policy's decision and learning on synthetic numbers; and through the Runtime on tiny Llama and Qwen 3 models,
every answer with speculation equal token for token to the same plan without it, alone and stacked."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ironmule import speculation as sp

ROOT = Path(__file__).resolve().parents[2]


def test_a_draft_is_what_followed_the_most_recent_repeat_and_names_its_match_length():
    assert sp.draft([5, 6, 7, 8, 1, 5, 6, 7], 3) == ([8, 1, 5], 3)
    assert sp.draft([1, 2, 9, 1, 2, 4, 2], 2) == ([4, 2], 1)
    assert sp.draft([1, 2, 3], 4) == ([], 0) and sp.draft([4, 4], 0) == ([], 0)
    assert sp.accept([3, 4, 5], [3, 4, 9, 7]) == [3, 4, 9] and sp.accept([], [6]) == [6]


def test_nothing_to_draft_or_no_free_row_means_a_plain_step():
    policy = sp.WindowPolicy(rows=8)
    assert policy.choose([([], 0), ([], 0)]) == [0, 0]
    assert sp.WindowPolicy(rows=2).choose([([1, 2], 3), ([4], 3)]) == [0, 0]
    assert policy.choose([]) == []


def test_it_drafts_where_acceptance_pays_for_the_rows_and_stops_where_it_does_not():
    policy = sp.WindowPolicy(rows=8)
    for _ in range(50):  # bucket 3 almost always right, bucket 1 almost never
        policy.observe_drafts(3, 4, 4)
        policy.observe_drafts(1, 4, 0)
    ks = policy.choose([(list(range(7)), 3), (list(range(7)), 1)])
    assert ks[0] >= 4 and ks[1] == 0 and sum(ks) + 2 <= 8
    policy.observe_step(1, 0.010, False)
    policy.observe_step(8, 0.080, True)  # rows cost as much as running them one by one: speculation never pays
    assert policy.choose([(list(range(7)), 3)]) == [0]


def test_prices_are_learned_from_steps_and_unmeasured_rows_lie_on_their_line():
    policy = sp.WindowPolicy(rows=8)
    assert policy.price(8) == pytest.approx(1.42)
    policy.observe_step(1, 0.050, False)
    assert policy.price(8) == pytest.approx(0.050 * 1.42)
    policy.observe_step(3, 0.070, True)
    assert policy.price(5) == pytest.approx(0.090)
    policy.observe_step(1, 0.100, False)
    assert policy.price(1) == pytest.approx(0.8 * 0.050 + 0.2 * 0.100)
    out = policy.summary()
    assert out["speculative_steps"] == 1 and out["plain_steps"] == 2 and out["step_ms_by_rows"]["3"] == 70.0


def test_acceptance_counts_the_first_refused_draft_and_starts_from_its_prior():
    policy = sp.WindowPolicy(rows=8)
    assert policy.acceptance(2) == pytest.approx(0.3)
    policy.observe_drafts(2, 5, 2)  # two taken, the third refused, the last two never tried
    assert policy.acceptance(2) == pytest.approx((2 + 1.2) / (3 + 4))
    assert policy.summary()["drafted"] == 5 and policy.summary()["accepted"] == 2


def _tiny(name: str):
    helper = importlib.util.spec_from_file_location("dense_cpu_tests", ROOT / "tests" / "engine" / "test_dense_cpu.py")
    cpu = importlib.util.module_from_spec(helper)
    helper.loader.exec_module(cpu)
    return cpu.tiny(getattr(cpu, name), seed=2)


class _Tokenizer:
    eos_token_id = 299
    eos_token_ids = {299}

    def encode(self, text):
        return [ord(c) % 250 + 1 for c in text]

    def decode(self, ids):
        return "".join(chr(32 + i % 90) for i in ids)


def _runtime(model, speculate: bool):
    from ironmule import dense_cpu
    from ironmule.runtime import Engine, Knobs
    from ironmule.service import DenseStackedMode, Runtime

    plan, record = dense_cpu.install(model, slots=8, capacity=128)
    assert record["verify_windows_identical"] is True
    return Runtime(Engine(model, _Tokenizer(), Knobs()), _Tokenizer(),
                   mode=DenseStackedMode(plan, max_width=8, speculate=speculate))


@pytest.mark.parametrize("family", ["LLAMA", "QWEN3"])
def test_speculation_changes_no_answer_alone_or_stacked_and_learns_while_it_serves(family):
    pytest.importorskip("mlx.core")
    from ironmule.service import Request

    model = _tiny(family)
    plain, spec = _runtime(model, False), _runtime(model, True)
    prompts = [[7, 3, 9, 7, 3, 9, 7, 3, 9, 7, 3, 11], [5, 6, 7, 8, 5, 6, 7, 8, 5, 6], [40, 41, 42, 43, 44],
               [12, 13, 12, 13, 12, 13, 12]]
    for ids in prompts:  # one at a time: the policy has free rows
        want = plain.serve([Request(prompt_ids=ids, max_tokens=24)])[0].tokens
        assert spec.serve([Request(prompt_ids=ids, max_tokens=24)])[0].tokens == want
    want = [r.tokens for r in plain.serve([Request(prompt_ids=ids, max_tokens=24) for ids in prompts])]
    assert [r.tokens for r in spec.serve([Request(prompt_ids=ids, max_tokens=24) for ids in prompts])] == want
    policy = spec.backend.policy
    assert policy is not None and plain.backend.policy is None
    assert policy.steps > 0 and policy.drafted > 0 and policy.cost  # it drafted, verified and priced its steps
    assert spec.backend.free_slots() == 8


def test_a_plan_whose_window_is_not_verified_never_speculates():
    from ironmule.service import DenseStackedMode

    class Plan:
        rows = slots = 8
        verifies_windows = False

    assert DenseStackedMode(Plan(), speculate=True).speculate is False


def test_through_the_product_speculation_answers_as_without_it_and_health_shows_what_it_learned(tmp_path, monkeypatch):
    """The service, the client, the worker process and the bridge's own `install_dense` (a tiny Llama in
    `dense_cpu_worker.py`), with and without `speculate`: the same answers, and /health shows the policy."""
    import subprocess
    import sys

    import ironmule_product.backend as backend_module
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore
    from ironmule_product.types import GenerationRequest, ModelSpec
    from tests.engine import dense_cpu_worker

    real_popen = subprocess.Popen

    def popen(args, *rest, **kwargs):
        if len(args) > 3 and str(args[3]).endswith("worker.py"):
            args = [sys.executable, "-u", dense_cpu_worker.__file__, *args[4:]]
        return real_popen(args, *rest, **kwargs)

    monkeypatch.setattr(backend_module.subprocess, "Popen", popen)
    monkeypatch.setenv("DENSE_CPU_FAMILY", "llama")
    monkeypatch.delenv("DENSE_CPU_LEDGER", raising=False)
    monkeypatch.delenv("DENSE_CPU_FAULTS", raising=False)
    store = ProductStore(tmp_path / "state")
    store.setup("server")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    spec = ModelSpec("local/test", "revision", str(snapshot.resolve()), 1)
    store.register_model(spec)
    texts = [("repeat after me: " + f"alpha beta {i} " * 6) for i in range(4)]

    def run(speculate: bool) -> tuple[list[str], dict]:
        backend = MLXWorkerClient(spec, execution_variant="current_engine", engine_configuration="dense_stacked",
                                  speculate=speculate)
        ready = backend.start()
        assert ready["engine"]["dense_plan"]["speculation"] == ("context" if speculate else "off")
        service = ProductService(store, backend=backend, spec=spec)
        try:
            answers = []
            for i, text in enumerate(texts):
                request = GenerationRequest("local/test", (("user", text),), max_tokens=24, request_id=f"r{i}",
                                            stream=True)
                answers.append("".join(e.get("text", "") for e in service.stream(request) if isinstance(e, dict)))
            return answers, service.health()
        finally:
            service.close()

    plain, health_plain = run(False)
    spec_answers, health = run(True)
    assert spec_answers == plain and all(plain)
    assert "speculation" not in health_plain
    learned = health["speculation"]["learned"]
    assert health["speculation"]["serving"] == "context" and learned["drafted"] > 0 and learned["speculative_steps"] > 0
    assert health["failed_requests"] == 0
    with pytest.raises(ValueError, match="dense_stacked"):
        MLXWorkerClient(spec, speculate=True)
