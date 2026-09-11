from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

ROOT = Path(__file__).resolve().parents[1]

from ironmule import paired_research as pr  # noqa: E402
from ironmule.executor import AsyncGroupedB1Executor  # noqa: E402
from ironmule.qmv_k3840 import K3840Unsupported  # noqa: E402
from ironmule.service import (  # noqa: E402
    InteractiveMode,
    PairedThroughputMode,
    ThroughputMode,
    paired_status,
)


def test_the_paired_mode_is_never_a_default() -> None:
    """Both shipped modes must report disabled through the same status shape."""

    for mode in (InteractiveMode(), ThroughputMode()):
        status = paired_status(mode)
        assert status["enabled"] is False
        assert status["admitted"] is False
        assert status["paired_steps"] == 0


def test_status_distinguishes_enabled_from_admitted() -> None:
    """Naming the mode is not the same as being allowed to run it."""

    status = paired_status(PairedThroughputMode())
    assert status["enabled"] is True
    assert status["admitted"] is False
    assert status["solo_steps_no_partner"] == 0


def test_the_shared_kernel_is_the_qualified_one() -> None:
    studied = (ROOT / "tools" / "b45_shared_weight_kernel.py").read_text()
    shipped = (ROOT / "ironmule" / "qmv_shared.py").read_text()
    assert studied[studied.index("SIMD_SIZE = 32"):] in shipped


def test_admission_refuses_without_a_model_identity() -> None:
    with pytest.raises(K3840Unsupported):
        pr.admit(nn.Module(), None)


def test_the_group_hook_leaves_the_shipped_executor_unchanged() -> None:
    """The hook exists so a subclass can pair; the base must still step one by one."""

    class Backend:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def step(self, state, token, capacity):
            self.calls.append(token)
            return (token, state)

    class Session:
        def __init__(self, token: int) -> None:
            self.state = {"n": token}
            self.tokens = [token]

    backend = Backend()
    executor = AsyncGroupedB1Executor.__new__(AsyncGroupedB1Executor)
    executor.backend = backend

    handles = executor._step_group([Session(7), Session(9)], capacity=64)

    assert backend.calls == [7, 9]
    assert len(handles) == 2


def test_a_lone_request_is_counted_as_solo_not_paired() -> None:
    """The contract says a single request never waits for a partner."""

    class Backend:
        engine = None

        def step(self, state, token, capacity):
            return (token, state)

    class Session:
        def __init__(self) -> None:
            self.state = {}
            self.tokens = [1]

    executor = pr.PairedGroupedExecutor.__new__(pr.PairedGroupedExecutor)
    executor.backend = pr.PairedBackend(Backend(), share=True)
    executor.paired_steps = 0
    executor.solo_steps = 0

    executor._step_group([Session()], capacity=64)

    assert executor.paired_steps == 0
    assert executor.solo_steps == 1


class _FakeBackend:
    """A backend that counts steps. No model, no GPU: this is a flow test only."""

    eos_ids = ()
    engine = None

    def __init__(self) -> None:
        self.pair_calls: list[int] = []
        self.solo_calls = 0

    def step(self, state, token, capacity):
        self.solo_calls += 1
        return ("solo", token + 1, state)

    def complete(self, handles):
        return None

    def read(self, handle):
        return handle[1], handle[2]

    def reset_state(self, base_state, offset):
        return base_state


class _PairCountingBackend(pr.PairedBackend):
    def step_pair(self, states, tokens, capacity):
        self._inner.pair_calls.append(len(states))
        return [("solo", token + 1, state) for state, token in zip(states, tokens)]


def _session(rid: str, arrival_ms: float, max_tokens: int):
    from ironmule.executor import RequestMetrics, Session

    metrics = RequestMetrics(rid=rid, arrival_ns=0, prompt_tokens=1)
    return Session(rid=rid, prompt_ids=[1], max_tokens=max_tokens, plan=None,
                   arrival_ms=arrival_ms, base_state={}, state={}, tokens=[1],
                   metrics=metrics)


def _paired_executor(inner):
    from ironmule.telemetry import Telemetry

    executor = pr.PairedGroupedExecutor.__new__(pr.PairedGroupedExecutor)
    backend = _PairCountingBackend(inner, share=True)
    backend._inner = inner
    AsyncGroupedB1Executor.__init__(executor, backend, Telemetry(mode="test"), max_width=2)
    executor.paired_steps = 0
    executor.solo_steps = 0
    return executor


def _stepping_clock(monkeypatch, step_ns: int = 1_000_000):
    """A monotonic clock that advances one millisecond per read.

    The executor decides admission from elapsed wall time. A real sleep would make the
    test slow and flaky; a counted clock makes the arrival boundary exact.
    """

    from ironmule import executor as executor_module

    ticks = {"now": 0}

    def fake_now() -> int:
        ticks["now"] += step_ns
        return ticks["now"]

    monkeypatch.setattr(executor_module, "now", fake_now)


def test_a_late_partner_is_joined_only_at_a_step_boundary(monkeypatch) -> None:
    """The early request runs alone until the partner arrives, then they share steps."""

    _stepping_clock(monkeypatch)
    inner = _FakeBackend()
    executor = _paired_executor(inner)
    sessions = [_session("a", 0.0, 40), _session("b", 12.0, 40)]

    executor.run(sessions, capacity=64)

    assert executor.solo_steps > 0, "the early request must not wait for a partner"
    assert executor.paired_steps > 0, "the partner must be picked up once it arrives"
    assert all(width == 2 for width in inner.pair_calls)
    assert all(session.done for session in sessions)


def test_only_requests_the_scheduler_admitted_are_ever_paired(monkeypatch) -> None:
    """A request still waiting for its arrival is never part of a shared step."""

    _stepping_clock(monkeypatch)
    inner = _FakeBackend()
    executor = _paired_executor(inner)
    # The third request arrives only after the other two have finished.
    sessions = [_session("a", 0.0, 4), _session("b", 0.0, 4), _session("c", 400.0, 2)]

    executor.run(sessions, capacity=64)

    assert inner.pair_calls, "the two ready requests should have shared steps"
    assert all(width == 2 for width in inner.pair_calls)
    assert sessions[2].done
