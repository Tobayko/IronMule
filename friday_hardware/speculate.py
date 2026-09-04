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


def find_match(
    tokens: list[int], ngram: int, draft_length: int, max_extend: int = 40
) -> tuple[int, list[int]]:
    """Most recent match, how far back it agrees, and what followed it.

    The window used to *find* a match and the confidence that match carries are
    different questions. A short window finds more matches; how far the agreement
    extends backwards says how much to trust one. Measured on real project text, a
    three-to-eight token agreement had its continuation accepted 53.6% of the time
    while a nine-or-longer one was accepted 48 times out of 48 -- so the same search
    can feed a cautious draft and a confident one.

    Extension is capped because the answer only has to distinguish "short" from
    "long"; walking a five-hundred-token agreement to its end would cost more than
    the distinction is worth.
    """

    if ngram < 1 or draft_length < 0 or max_extend < ngram:
        raise ValueError("ngram positive, draft non-negative, max_extend at least ngram")
    if draft_length == 0 or len(tokens) <= ngram:
        return 0, []
    needle = tokens[-ngram:]
    for start in range(len(tokens) - ngram - 1, -1, -1):
        if tokens[start : start + ngram] != needle:
            continue
        length = ngram
        while (
            length < max_extend
            and start - (length - ngram) - 1 >= 0
            and length + 1 <= len(tokens)
            and tokens[start - (length - ngram) - 1] == tokens[-length - 1]
        ):
            length += 1
        return length, list(tokens[start + ngram : start + ngram + draft_length])
    return 0, []


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
    # Steps where the running acceptance had fallen below every break-even, so the
    # generator chose a plain step instead.
    declined_steps: int = 0

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
    adapt: bool = True,
    by_match_length: bool = True,
    warmup_drafts: int = 4,
    memory: float = 0.7,
) -> Generation:
    """Greedy decode, drafting from the context where the profile says it pays.

    Passing a profile is the intended use: the window and draft depth are both
    model-specific, measured rather than assumed, and getting them from the wrong
    model costs real speed in both directions. Explicit values override it, which is
    what the measuring tool does when sweeping.

    With `draft_length` zero this is ordinary greedy decoding, which makes it the
    baseline its own speedup is measured against.

    With `by_match_length` and a profile, the search window and the draft depth are
    decoupled: a short window finds candidates, and how far the agreement extends
    backwards decides how many of them to risk.

    With `adapt` set and a profile given, the draft depth follows the acceptance the
    run is actually seeing rather than the one the profile was measured at. That
    matters because acceptance is a property of the text, not of the model: the same
    4B accepted every drafted token while rewriting a function and 44% of them while
    summarising prose, where a fixed depth measured 0.980x. Depth is only spent while
    it is being repaid.
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

    can_adapt = adapt and profile is not None
    running = 1.0            # optimistic until the run has evidence of its own
    seen = 0
    steps = drafted_total = accepted_total = unrewindable = declined = 0
    while len(generated) < max_tokens:
        depth = draft_length
        if can_adapt and seen >= warmup_drafts:
            depth = profile.draft_length_for(running, limit=draft_length)
        # Speculation is only safe where the rejected tokens can be taken back out
        # of the cache. Gemma 3 keeps most layers in a rotating cache that stops
        # being rewindable once the context passes its window -- 512 tokens on the
        # 1B, 1024 on the 4B -- and trim_prompt_cache reports that by returning
        # zero rather than raising. Drafting anyway leaves rejected tokens in the
        # cache and quietly changes every later token, which is exactly what it did
        # before this check existed.
        rewindable = can_trim_prompt_cache(cache)
        if not rewindable:
            unrewindable += 1
            drafted = []
        elif depth < 1:
            declined += 1
            drafted = []
        elif by_match_length and profile is not None:
            # Search with the short window, spend depth according to how far the
            # agreement actually reaches. One search, two decisions.
            match_length, candidate = find_match(context, ngram, depth)
            allowed = profile.depth_for_match(match_length, limit=depth)
            drafted = candidate[:allowed]
            if not drafted and candidate:
                declined += 1
        else:
            drafted = find_continuation(context, ngram, depth)
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
        if drafted:
            observed = keep / len(drafted)
            # Exponential memory: acceptance drifts as the text moves between
            # quoting and inventing, and a plain average would still be reacting to
            # the opening paragraph a hundred tokens later. The horizon is short on
            # purpose -- at 0.9 the estimate needed about thirty drafted steps to
            # fall from its optimistic start, which is most of a short answer.
            running = memory * running + (1.0 - memory) * observed
            seen += len(drafted)

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
        declined_steps=declined,
    )
