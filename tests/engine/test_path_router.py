"""AG4 step 2: the agent's router between two serving paths on one card (ironmule_product/path_router.py)."""

from __future__ import annotations

import time

from ironmule_product.device_pool import DevicePool
from ironmule_product.path_router import PathCost, PathRouter

# AG4-Q run 2's measurement on a T4 with Qwen 3 1.7B: float16 alone and the dense engine at 1 and 8 rows
SETUP = {"stacked": {"speed": {name: {"base_step_s": 0.0173, "base_first_s": 0.06, "plan_first_s": 0.36,
                                      "plan_step_s": {"1": 0.0212, "8": 0.0282}} for name in ("short", "long")}}}


def test_a_quiet_server_uses_the_single_path_and_a_queue_moves_requests_to_the_stacked_engine():
    router = PathRouter.from_setup(SETUP)
    assert router.choose([0, 0], 256) == 0  # alone: float16 is faster and its first token earlier
    assert router.choose([1, 0], 256) == 1  # one ahead on the single path: the stacked engine finishes first
    assert router.choose([1, 7], 256) == 1  # eight sharing still beats waiting a whole answer
    assert router.choose([1, 8], 256) == 0  # the stacked engine full: its queue is longer than one answer
    status = router.status()
    assert status["decisions"] == {"single": 2, "stacked": 2} and status["decision_us"] < 1000


def test_decisions_take_microseconds():
    router = PathRouter.from_setup(SETUP)
    started = time.perf_counter()
    for i in range(10_000):
        router.choose([i % 3, i % 9], 256)
    assert (time.perf_counter() - started) / 10_000 < 100e-6  # well under 0.1 ms each, in pure Python


def test_it_learns_a_path_that_is_slower_than_measured():
    router = PathRouter.from_setup(SETUP)
    for _ in range(200):  # the stacked engine turns out twice as slow per token at four sharing
        router.observe(1, 4, 200, 199 * 0.05, None)
    assert router.costs[1].step(4) > 0.04
    assert router.choose([1, 3], 256) == 0  # now waiting for the single path is the better bet
    assert router.status()["paths"]["stacked"]["requests"] == 200


def test_watch_learns_from_a_finished_stream():
    router = PathRouter([ "a", "b"], [PathCost(1, 0.02, 0.0, 0.1), PathCost(8, 0.03, 0.001, 0.3)])
    events = [{"type": "token", "text": "x"}, {"type": "token", "text": "y"},
              {"type": "done", "completion_tokens": 2}]
    assert list(router.watch(0, 1, iter(events))) == events
    assert router.costs[0].seen == 1


class _Shard:
    def __init__(self, outstanding):
        self.spec, self.store, self._outstanding, self.streamed = None, object(), outstanding, []

    def accepting(self):
        return True

    def outstanding(self):
        return self._outstanding

    def holds(self, request_id):
        return False

    def health(self):
        return {"ready": True}

    def stream(self, request):
        self.streamed.append(request.request_id)
        return iter([{"type": "token", "text": "a"}, {"type": "done", "completion_tokens": 1}])


def test_the_pool_places_by_the_router_and_reports_it():
    from ironmule_product.types import GenerationRequest

    single, stacked = _Shard(1), _Shard(0)
    pool = DevicePool([single, stacked], ["0:single", "0:stacked"])
    pool.router = PathRouter.from_setup(SETUP)
    request = GenerationRequest(model="m", messages=(("user", "hi"),), max_tokens=64, request_id="r1")
    list(pool.stream(request))
    assert stacked.streamed == ["r1"] and single.streamed == []
    assert pool.router.status()["decisions"]["stacked"] == 1


class _NoRoom(_Shard):
    """A path whose slot cannot hold the request: it refuses before any token, as the stacked engine does."""

    def __init__(self, outstanding, after_token=False):
        super().__init__(outstanding)
        self.after_token = after_token

    def stream(self, request):
        from ironmule_product.errors import EngineLimitExceeded

        self.streamed.append(request.request_id)

        def events():
            if self.after_token:
                yield {"type": "token", "text": "a"}
            raise EngineLimitExceeded("the prompt and max_tokens exceed the 1024-token limit of this server's engine")
        return events()


def test_a_request_the_chosen_path_cannot_hold_is_served_by_the_other_path():
    """AG4-R run 1: the router sent over-long requests to the stacked engine, which refused them. The pool now
    serves them on the next path, counts it, and the router learns nothing from the refusal."""
    from ironmule_product.types import GenerationRequest

    single, stacked = _Shard(1), _NoRoom(0)
    pool = DevicePool([single, stacked], ["0:single", "0:stacked"])
    pool.router = PathRouter.from_setup(SETUP)
    seen = pool.router.costs[1].seen
    request = GenerationRequest(model="m", messages=(("user", "hi"),), max_tokens=64, request_id="r1")
    events = list(pool.stream(request))
    assert [e["type"] for e in events] == ["token", "done"]
    assert stacked.streamed == ["r1"] and single.streamed == ["r1"]
    assert pool.path_fallbacks == 1 and pool.health()["paths"]["fallbacks"] == 1
    assert pool.router.costs[1].seen == seen, "the refusal teaches the router nothing"


def test_a_refusal_after_a_token_or_a_context_limit_is_not_rerouted():
    import pytest

    from ironmule_product.errors import EngineLimitExceeded, InvalidRequest
    from ironmule_product.types import GenerationRequest

    late = DevicePool([_Shard(1), _NoRoom(0, after_token=True)], ["0:single", "0:stacked"])
    late.router = PathRouter.from_setup(SETUP)
    request = GenerationRequest(model="m", messages=(("user", "hi"),), max_tokens=64, request_id="r2")
    with pytest.raises(EngineLimitExceeded):
        list(late.stream(request))
    assert late.shards[0].streamed == [] and late.path_fallbacks == 0

    class ContextLimit(_Shard):
        def stream(self, request):
            def events():
                raise InvalidRequest("the prompt and max_tokens exceed the 4096-token context limit")
                yield  # pragma: no cover
            return events()

    context = DevicePool([_Shard(1), ContextLimit(0)], ["0:single", "0:stacked"])
    context.router = PathRouter.from_setup(SETUP)
    with pytest.raises(InvalidRequest):
        list(context.stream(GenerationRequest(model="m", messages=(("user", "hi"),), max_tokens=64,
                                              request_id="r3")))
    assert context.shards[0].streamed == [] and context.path_fallbacks == 0
