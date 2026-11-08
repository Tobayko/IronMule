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


# ---------------------------------------------------------------- AG6: replay, dream, PolicyRouter

def _costs(stacked_step=0.034, stacked_one=0.0229, first=0.1):
    from ironmule_product.path_router import PathCost

    return [PathCost(1, 0.0174, 0.0, first, tokens=200.0),
            PathCost(8, stacked_one, (stacked_step - stacked_one) / 7, first, tokens=200.0)]


def test_the_single_path_replays_one_request_at_a_time():
    import pytest

    from ironmule_product.path_router import PathCost, replay

    costs = [PathCost(1, 0.01, 0.0, 0.1, tokens=3.0), PathCost(8, 0.01, 0.0, 0.1, tokens=3.0)]
    result = replay(costs, [(0.0, 3), (0.0, 3)], "single")
    # first: 0.1 then 2 steps of 0.01 -> 0.12; second starts at 0.12 -> first 0.22, ends 0.24
    assert result["tokens_per_s"] == pytest.approx(6 / 0.24)
    assert result["ttft_mean_s"] == pytest.approx((0.1 + 0.22) / 2)


def test_the_stacked_engine_replays_rows_together_and_newcomers_stall_them():
    import pytest

    from ironmule_product.path_router import PathCost, replay

    costs = [PathCost(1, 0.01, 0.0, 0.1, tokens=3.0), PathCost(8, 0.01, 0.002, 0.1, tokens=3.0)]
    result = replay(costs, [(0.0, 3), (0.0, 3)], "stacked")
    # both admitted in turn (0.1 each), then 2 steps of two rows at 0.012 -> 0.224
    assert result["tokens_per_s"] == pytest.approx(6 / 0.224)
    assert result["ttft_mean_s"] == pytest.approx((0.1 + 0.2) / 2)


def test_both_paths_share_one_card_so_routing_a_spike_is_slower_than_stacking_it():
    """AG4-R run 1: in a spike the router made 131 tokens/s and the stacked engine alone 231; replayed as two free
    servers the router looked like 236. On one shared card the replay gives the measured order."""
    from ironmule_product.path_router import replay

    spike = [(0.001 * i, 200) for i in range(32)]
    routed, stacked, single = (replay(_costs(), spike, mode)["tokens_per_s"] for mode in ("route", "stacked", "single"))
    assert stacked > 1.5 * routed > 1.5 * single


def test_the_dream_moves_to_stacking_for_a_spike_and_keeps_its_mode_for_a_trickle():
    from ironmule_product.path_router import dream

    spike = [(0.001 * i, 200) for i in range(32)]
    moved = dream(_costs(), spike, "route")
    assert moved["mode"] == "stacked" and set(moved["replayed"]) == {"route", "stacked", "single"}
    trickle = [(30.0 * i, 200) for i in range(32)]  # one request at a time: every mode serves it alike
    assert dream(_costs(), trickle, "route")["mode"] == "route", "no move without a 5 % gain"


def test_the_policy_router_dreams_every_sixteen_finished_requests_and_serves_its_mode():
    from ironmule_product.path_router import DREAM_MIN, PolicyRouter

    router = PolicyRouter(["single", "stacked"], _costs())
    for i in range(DREAM_MIN - 1):
        router.finished(100.0 + 0.001 * i, 200)
    assert router.dreams == 0 and router.mode == "route", "no dream before 32 logged requests"
    router.finished(100.0 + 0.001 * DREAM_MIN, 200)
    assert router.dreams == 1 and router.mode == "stacked" and router.switches == [
        {"at_request": DREAM_MIN, "from": "route", "to": "stacked"}]
    assert router.choose([5, 0], 256) == 1 and router.choose([0, 9], 256) == 1, "a lookup in its mode"
    status = router.status()
    assert status["mode"] == "stacked" and status["last_dream"]["requests"] == DREAM_MIN


# ---------------------------------------------------------------- stability: a router fault never fails a request

def test_a_decision_that_fails_serves_on_the_first_path_and_is_counted():
    router = PathRouter.from_setup(SETUP)

    def broken(outstanding, tokens):
        raise ZeroDivisionError("no price")

    router.costs[1].expected = broken
    assert router.choose([0, 0], 64) == 0
    status = router.status()
    assert status["errors"] == 1 and status["last_error"].startswith("choose: ZeroDivisionError")


def test_learning_that_fails_never_breaks_the_stream_it_watches():
    router = PathRouter.from_setup(SETUP)

    def broken(*args):
        raise ValueError("bad sample")

    router.observe = broken
    events = [{"type": "token", "text": "a"}, {"type": "token", "text": "b"}, {"type": "done", "completion_tokens": 2}]
    assert list(router.watch(1, 1, iter(events))) == events
    assert router.status()["errors"] == 1


def test_a_dream_that_fails_keeps_the_mode_and_the_answer_goes_out(monkeypatch):
    from ironmule_product import path_router
    from ironmule_product.path_router import DREAM_MIN, PolicyRouter

    def broken(*args):
        raise RuntimeError("the replay diverged")

    monkeypatch.setattr(path_router, "dream", broken)
    router = PolicyRouter(["single", "stacked"], _costs())
    for i in range(DREAM_MIN - 1):
        router.finished(100.0 + 0.001 * i, 200)
    events = [{"type": "token", "text": "a"}, {"type": "done", "completion_tokens": 1}]
    assert list(router.watch(1, 1, iter(events))) == events  # the 32nd request finishes and the dream fails
    status = router.status()
    assert status["mode"] == "route" and status["dreams"] == 0 and status["errors"] == 1
    assert status["last_error"].startswith("dream: RuntimeError")


def test_the_pool_serves_a_request_when_the_router_itself_fails():
    from ironmule_product.types import GenerationRequest

    class Broken:
        def choose(self, outstanding, max_tokens):
            raise KeyError("costs")

        def watch(self, index, sharing, events):
            return events

        def status(self):
            raise RuntimeError("no status")

    single, stacked = _Shard(1), _Shard(0)
    pool = DevicePool([single, stacked], ["0:single", "0:stacked"])
    pool.router = Broken()
    request = GenerationRequest(model="m", messages=(("user", "hi"),), max_tokens=64, request_id="r1")
    assert [e["type"] for e in pool.stream(request)] == ["token", "done"]
    assert stacked.streamed == ["r1"], "placement order (least outstanding first) served it"
    paths = pool.health()["paths"]
    assert paths["router_faults"] == 1 and paths["status_error"].startswith("RuntimeError")


def test_a_rerouted_request_is_priced_at_the_load_its_new_path_has_now():
    """AG6-H: a request the stacked engine refused was taught to the router at the single path's load when the
    request first came (often none), though it waited behind earlier refusals there. The single path's learned
    first token rose to 7.3 s, which no request alone ever took."""
    from ironmule_product.types import GenerationRequest

    single, stacked = _Shard(0), _NoRoom(0)
    pool = DevicePool([single, stacked], ["0:single", "0:stacked"])
    pool.router = PathRouter.from_setup(SETUP)
    seen = []
    watch = pool.router.watch
    pool.router.watch = lambda index, sharing, events: seen.append((index, sharing)) or watch(index, sharing, events)
    pool.router.choose = lambda outstanding, max_tokens: 1
    single._outstanding = 0
    request = GenerationRequest(model="m", messages=(("user", "hi"),), max_tokens=64, request_id="r1")
    stream = pool.stream(request)
    single._outstanding = 3  # three refusals got there first
    assert [e["type"] for e in stream] == ["token", "done"]
    assert seen == [(1, 1), (0, 4)]


# ---------------------------------------------------------------- review of 2026-11-08: AG6 on quiet traffic

def _quiet_costs():
    from ironmule_product.path_router import PathCost

    return [PathCost(capacity=1, step_s=0.010, step_s_per_row=0.0, first_s=0.010),
            PathCost(capacity=8, step_s=0.012, step_s_per_row=0.003, first_s=0.100)]


def test_on_quiet_traffic_the_dream_leaves_a_mode_whose_first_tokens_come_ten_times_later():
    """Every mode's tokens per second is the arrival rate on quiet traffic, so the 5 % rule never moved the agent
    off `stacked`, whose first tokens came 10x later. At the same throughput the earliest first token now wins."""
    from ironmule_product.path_router import dream

    quiet = [(10.0 * i, 32) for i in range(64)]
    out = dream(_quiet_costs(), quiet, "stacked")
    assert out["mode"] == "route" and out["rule"] == "first token"
    assert out["replayed"]["stacked"]["overlapped"] == 0
    assert dream(_quiet_costs(), quiet, "route")["mode"] == "route", "and it stays: nothing serves sooner"


def test_under_load_the_first_token_rule_stays_out_and_the_two_rules_do_not_take_turns():
    """The replay's first tokens are unreliable when requests overlap (AG6-H bursts at load 0.8: it priced the
    router 25 % sooner, the card measured stacking 5x sooner), so the first-token rule waits for quiet traffic."""
    from ironmule_product.path_router import DREAM_QUIET, dream

    spike = [(0.01 * i, 200) for i in range(64)]
    out = dream(_costs(), spike, "stacked")
    assert out["replayed"]["stacked"]["overlapped"] > DREAM_QUIET and out["rule"] != "first token"
    moved = dream(_costs(), spike, "route")["mode"]
    assert dream(_costs(), spike, moved)["mode"] == moved, "a move for throughput is not undone for latency"
