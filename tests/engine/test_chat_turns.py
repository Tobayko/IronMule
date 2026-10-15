"""SERVE7: chat-turn boundaries, the plan's engine settings, and the turn cache. No model."""

from __future__ import annotations

import mlx.core as mx

from ironmule.plans import ChatTurnsPlan, ReusableSessionPlan, StrictOneShotPlan
from ironmule.runtime import TurnCache, _padded_state
from ironmule_product.worker import _turn_boundaries


class _Template:
    """Renders each message as [role id, content length] and a generation prompt as [9]."""

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        ids = []
        for message in messages:
            ids += [{"system": 1, "user": 2, "assistant": 3}[message["role"]], len(message["content"])]
        return ids + ([9] if add_generation_prompt else [])


def test_boundaries_sit_where_each_assistant_message_begins():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "more"}]
    tokenizer = _Template()
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    # The generation prompt [9] is not a prefix of the rendered assistant message [3, 5]: no boundary.
    assert _turn_boundaries(tokenizer, messages, prompt) == []

    class Prefixing(_Template):  # a template whose generation prompt opens the assistant message
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            ids = []
            for message in messages:
                ids += [{"user": 2, "assistant": 3}[message["role"]], len(message["content"])]
            return ids + ([3] if add_generation_prompt else [])

    prefixing = Prefixing()
    prompt = prefixing.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    assert _turn_boundaries(prefixing, messages, prompt) == [3]


def test_only_the_chat_plan_sets_boundaries_and_every_other_plan_clears_them():
    class Engine:
        prefix_cache = None
        turn_boundaries = None

    engine = Engine()
    ChatTurnsPlan([5, 0, 3, 5]).apply(engine)
    assert engine.turn_boundaries == (3, 5) and engine.prefix_cache is None
    StrictOneShotPlan().apply(engine)
    assert engine.turn_boundaries is None
    ChatTurnsPlan([2]).apply(engine)
    ReusableSessionPlan([1, 2]).apply(engine)
    assert engine.turn_boundaries is None and engine.prefix_cache is not None
    ChatTurnsPlan([2]).release(engine)
    assert engine.turn_boundaries is None and engine.prefix_cache is None


def test_the_turn_cache_stores_unpadded_states_and_evicts_the_oldest():
    with mx.stream(mx.cpu):
        layer = {"keys": mx.arange(2 * 8 * 3).reshape(1, 2, 8, 3).astype(mx.float32)}
        layer["values"] = layer["keys"] + 100
        cache = TurnCache(max_entries=2)
        cache.put([1, 2, 3], {"layers": [layer]})
        stored = cache.get([1, 2, 3])
        assert stored[0]["keys"].shape == (1, 2, 3, 3)
        padded = _padded_state(stored, 3, 16)
        assert padded["layers"][0]["keys"].shape == (1, 2, 16, 3)
        assert mx.array_equal(padded["layers"][0]["keys"][..., :3, :], layer["keys"][..., :3, :]).item()
        assert padded["position"]["offset"].item() == 3
        cache.put([4], {"layers": [layer]})
        cache.get([1, 2, 3])  # used most recently
        cache.put([5], {"layers": [layer]})
        assert cache.get([4]) is None and cache.get([1, 2, 3]) is not None and cache.get([5]) is not None
