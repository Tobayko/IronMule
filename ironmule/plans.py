"""Execution plans. The caller chooses one; nothing in the runtime may switch it.

Two plans exist, and they are not interchangeable. E9 measured them producing
byte-identical output *within* a plan and up to 4.31 logits apart *between* plans;
E13 then measured the quality difference between them as bounded by 1.14 accuracy
points on extractive question answering, which is a bound for that evaluation set
and not a licence to treat them as equivalent.

Because the plans differ in output, choosing one is a caller decision with visible
consequences. `Runtime` records every attempt to substitute one for the other, and
that counter is expected to stay at zero forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .runtime import PrefixCache

RUNTIME_VERSION = "0.1.0"


class ExecutionPlan:
    """Base class. A plan owns how a prompt's prefill is arranged, nothing else."""

    kind: str = "abstract"

    def apply(self, engine) -> None:            # pragma: no cover - overridden
        raise NotImplementedError

    def release(self, engine) -> None:
        engine.prefix_cache = None

    def describe(self) -> dict:                 # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True)
class StrictOneShotPlan(ExecutionPlan):
    """One forward over the whole prompt. No reuse of any kind.

    Use when a request must be answered exactly as an untuned single-shot path
    would answer it, or when prompts share no prefix worth caching.
    """

    kind: str = "strict_one_shot"

    def apply(self, engine) -> None:
        engine.prefix_cache = None

    def describe(self) -> dict:
        return {"kind": self.kind}


class ReusableSessionPlan(ExecutionPlan):
    """Prefill chunked at a declared prefix, whose KV state is reused across requests.

    Within this plan a cache hit is bit exact: E9 measured `max |delta| = 0` across
    twelve requests and every decode step, and E12 reproduced that across prefix
    lengths from 276 to 2048 tokens spanning Gemma 3's 1024 sliding-window boundary,
    over 756 requests and 14,369 decode steps in five processes.

    It does **not** agree with `StrictOneShotPlan`. That is the trade, and it is the
    caller's to make.
    """

    kind = "reusable_session"

    def __init__(self, prefix_ids: Sequence[int], name: str = "session"):
        if not prefix_ids:
            raise ValueError("a reusable session needs a non-empty prefix")
        self.prefix_ids = tuple(prefix_ids)
        self.name = name
        self.cache = PrefixCache(self.prefix_ids)

    def matches(self, prompt_ids: Sequence[int]) -> bool:
        return self.cache.matches(list(prompt_ids))

    def apply(self, engine) -> None:
        engine.prefix_cache = self.cache

    def describe(self) -> dict:
        return {"kind": self.kind, "name": self.name,
                "prefix_tokens": len(self.prefix_ids),
                "hits": self.cache.hits, "misses": self.cache.misses}


def plan_kind(plan: ExecutionPlan) -> str:
    return getattr(plan, "kind", "unknown")


def _self_check() -> None:
    strict = StrictOneShotPlan()
    assert strict.kind == "strict_one_shot"
    assert strict.describe() == {"kind": "strict_one_shot"}

    session = ReusableSessionPlan([5, 6, 7], name="docs")
    assert session.matches([5, 6, 7, 8]) and not session.matches([5, 6, 9])
    assert session.describe()["prefix_tokens"] == 3

    try:
        ReusableSessionPlan([])
    except ValueError:
        pass
    else:                                        # pragma: no cover
        raise AssertionError("empty prefix must be rejected")

    class FakeEngine:
        prefix_cache = "dirty"

    engine = FakeEngine()
    strict.apply(engine)
    assert engine.prefix_cache is None, "strict must clear any session cache"
    session.apply(engine)
    assert engine.prefix_cache is session.cache
    session.release(engine)
    assert engine.prefix_cache is None
    print("plans self-check ok")


if __name__ == "__main__":
    _self_check()
