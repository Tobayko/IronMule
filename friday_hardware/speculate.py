"""Draft the next few tokens from the context, verify them in one pass.

Measured here at 1.695x on the 1B rewriting a function and 1.133x on an agent-shaped
context, with output identical to greedy decoding token for token. The technique is
described in docs/PROMPT_LOOKUP_2026-08-23.md; this is the part meant to be called
rather than benchmarked.

The two pure functions carry the whole correctness argument and live apart from any
model so they can be tested without one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .profile import HardwareProfile, ProfileError


def find_continuation(tokens: list[int], ngram: int, draft_length: int) -> list[int]:
    """Longest-recent match: what followed the last `ngram` tokens the last time.

    Searched backwards from the end, because the most recent occurrence is the one
    most likely to continue the same way -- an identifier repeated three lines ago
    beats the same identifier in a different function forty lines up.

    Returns fewer than `draft_length` tokens near the end of the context, and none at
    all when nothing matches. Proposing nothing is an ordinary decode step, which is
    why a miss is cheap -- though not free at every window length: a one-token window
    matched often enough and wrongly enough to measure slower than not drafting.
    """

    if ngram < 1 or draft_length < 0:
        raise ValueError("ngram must be positive and draft length non-negative")
    if draft_length == 0 or len(tokens) <= ngram:
        return []
    needle = tokens[-ngram:]
    # Stop before the trailing occurrence, which is the needle itself.
    for start in range(len(tokens) - ngram - 1, -1, -1):
        if tokens[start : start + ngram] == needle:
            proposal = tokens[start + ngram : start + ngram + draft_length]
            if proposal:
                return list(proposal)
    return []


def accepted_prefix(drafted: list[int], produced: list[int]) -> int:
    """How many drafted tokens the model would have produced anyway.

    Acceptance stops at the first disagreement. Keeping a later match after an earlier
    miss would splice a continuation onto a prefix that never existed and silently
    change the answer, which is the one thing this must not do.
    """

    count = 0
    for want, got in zip(drafted, produced):
        if want != got:
            break
        count += 1
    return count


@dataclass(frozen=True)
class Generation:
    """Tokens, plus enough accounting to tell whether drafting earned its place."""

    tokens: list[int]
    seconds: float
    steps: int
    drafted: int
    accepted: int
    ngram: int
    draft_length: int
    # Steps that could not draft because the cache had stopped being rewindable.
    unrewindable_steps: int = 0

    @property
    def acceptance(self) -> float | None:
        return self.accepted / self.drafted if self.drafted else None

    @property
    def tokens_per_step(self) -> float:
        return len(self.tokens) / self.steps if self.steps else 0.0


def speculative_generate(
    model,
    sampler,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    profile: HardwareProfile | None = None,
    ngram: int | None = None,
    draft_length: int | None = None,
) -> Generation:
    """Greedy decode, drafting from the context where the profile says it pays.

    Passing a profile is the intended use: the window and draft depth are both
    model-specific, measured rather than assumed, and getting them from the wrong
    model costs real speed in both directions. Explicit values override it, which is
    what the measuring tool does when sweeping.

    With `draft_length` zero this is ordinary greedy decoding, which makes it the
    baseline its own speedup is measured against.
    """

    import mlx.core as mx
    from mlx_lm.models.cache import (
        can_trim_prompt_cache,
        make_prompt_cache,
        trim_prompt_cache,
    )

    if max_tokens < 1:
        raise ProfileError("generation needs at least one token")
    if ngram is None or draft_length is None:
        if profile is None:
            raise ProfileError("pass a profile, or both ngram and draft_length")
        ngram = profile.lookup_ngram if ngram is None else ngram
        draft_length = profile.lookup_draft if draft_length is None else draft_length
    if ngram < 1 or draft_length < 0:
        raise ProfileError("ngram must be positive and draft length non-negative")

    cache = make_prompt_cache(model)
    context = list(prompt_ids)

    started = time.perf_counter()
    logits = model(mx.array([context]), cache=cache)
    first = int(sampler(logits[:, -1, :].astype(mx.float32))[0])
    mx.eval(first)
    context.append(first)
    generated = [first]

    steps = drafted_total = accepted_total = unrewindable = 0
    while len(generated) < max_tokens:
        # Speculation is only safe where the rejected tokens can be taken back out
        # of the cache. Gemma 3 keeps most layers in a rotating cache that stops
        # being rewindable once the context passes its window -- 512 tokens on the
        # 1B, 1024 on the 4B -- and trim_prompt_cache reports that by returning
        # zero rather than raising. Drafting anyway leaves rejected tokens in the
        # cache and quietly changes every later token, which is exactly what it did
        # before this check existed.
        rewindable = can_trim_prompt_cache(cache)
        drafted = (
            find_continuation(context, ngram, draft_length) if rewindable else []
        )
        if not rewindable:
            unrewindable += 1
        window = [context[-1]] + drafted
        logits = model(mx.array([window]), cache=cache)
        picks = sampler(logits[0].astype(mx.float32)).tolist()
        mx.eval(logits)

        keep = accepted_prefix(drafted, picks[:-1]) if drafted else 0
        # picks[i] is what the model produces after position i, so the accepted run
        # plus one bonus token is exactly picks[:keep + 1].
        emitted = picks[: keep + 1]
        steps += 1
        drafted_total += len(drafted)
        accepted_total += keep

        # The pass wrote len(window) positions into the cache. Everything past the
        # accepted run and its bonus token never happened and must be rolled back,
        # or the next step would attend to tokens the model rejected.
        surplus = len(window) - (keep + 1)
        if surplus > 0:
            removed = trim_prompt_cache(cache, surplus)
            if removed != surplus:
                # The pre-check said this was rewindable, so a short trim means the
                # cache changed its mind mid-step. Continuing would silently corrupt
                # the answer; there is no safe way to carry on from here.
                raise ProfileError(
                    f"cache rollback removed {removed} of {surplus} tokens; "
                    "the generated text would no longer match greedy decoding"
                )

        for token in emitted:
            context.append(int(token))
            generated.append(int(token))
            if len(generated) >= max_tokens:
                break

    mx.eval(mx.array(generated))
    mx.synchronize()
    return Generation(
        tokens=generated[:max_tokens],
        seconds=time.perf_counter() - started,
        steps=steps,
        drafted=drafted_total,
        accepted=accepted_total,
        ngram=ngram,
        draft_length=draft_length,
        unrewindable_steps=unrewindable,
    )
