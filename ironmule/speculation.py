"""SP2: speculation from the request's own context, with a window learned while the server runs (BACKLOG SP2).

Drafts are copied from the request's prompt and answer so far (prompt lookup, SP1-0): the longest suffix of 3 down
to 1 tokens that occurred before, its most recent earlier occurrence, and up to `MAX_DRAFT` tokens after it. The
stacked plan verifies a request's drafts as rows of its own slot at consecutive positions, in the same forward as
the other requests' rows; a request takes its drafts up to the first that differs from the plan's own argmax, plus
that argmax, so its tokens are exactly those of one row at a time.

How many drafts each request gets is `WindowPolicy`'s decision, Strata's draft policy
(`src/spec/draft_policy.cpp`, BACKLOG HQ1) learned online on this machine:

* the time of a step by its rows, an exponential average of measured steps (plain and speculative alike);
  an unmeasured row count is priced by a line through the measured ones;
* per draft bucket (its match length), the chance that a draft is accepted when the ones before it were, from every
  verified draft, with a prior of 0.3 worth 4 drafts;
* each step it hands out drafts greedily where they add the most expected tokens while the step's expected tokens
  per second rises, and speculates only where that beats the plain step of the same requests by `MARGIN`.

The policy never changes a token: it only decides how many positions are checked at once.
"""

from __future__ import annotations

import threading
from typing import Any, Sequence

MAX_DRAFT = 7
MAX_NGRAM = 3
PRIOR_ACCEPT, PRIOR_WEIGHT = 0.3, 4.0
COST_WEIGHT = 0.2
ROW_SLOPE = 0.06  # CPU6-A2: eight stacked rows cost 1.42x one row on this container's CPU
MARGIN = 1.02


def draft(context: Sequence[int], k: int, max_ngram: int = MAX_NGRAM) -> tuple[list[int], int]:
    """Up to `k` tokens after the most recent earlier occurrence of the context's longest repeated suffix, and that
    suffix's length (0 when nothing repeats)."""
    if k < 1:
        return [], 0
    n_ctx = len(context)
    for n in range(min(max_ngram, n_ctx - 1), 0, -1):
        suffix = list(context[n_ctx - n:])
        for start in range(n_ctx - n - 1, -1, -1):
            if list(context[start:start + n]) == suffix:
                tokens = list(context[start + n:start + n + k])
                if tokens:
                    return tokens, n
    return [], 0


def accept(drafts: Sequence[int], argmax: Sequence[int]) -> list[int]:
    """The drafts the plan agrees with, then its own next token."""
    n = 0
    while n < len(drafts) and drafts[n] == argmax[n]:
        n += 1
    return list(drafts[:n]) + [int(argmax[n])]


class WindowPolicy:
    """How many drafts each request of a step gets, learned from this machine's steps and this traffic's drafts."""

    def __init__(self, rows: int):
        if rows < 1:
            raise ValueError("a step needs at least one row")
        self.rows = rows
        self.cost: dict[int, float] = {}
        self.cost_samples: dict[int, int] = {}
        self.tried: dict[int, float] = {}
        self.taken: dict[int, float] = {}
        self.steps = self.plain_steps = self.drafted = self.accepted = 0
        self._lock = threading.Lock()

    # --- prices -------------------------------------------------------------------------------------------------
    def price(self, rows: int) -> float:
        """Seconds for one step of `rows` rows: measured, else a line through the measured row counts, else the
        prior slope (a relative price; only ratios between row counts matter)."""
        with self._lock:
            if rows in self.cost:
                return self.cost[rows]
            known = sorted(self.cost.items())
        if not known:
            return 1.0 + ROW_SLOPE * (rows - 1)
        if len(known) == 1:
            r0, t0 = known[0]
            return t0 * (1.0 + ROW_SLOPE * (rows - 1)) / (1.0 + ROW_SLOPE * (r0 - 1))
        n = len(known)
        mean_r = sum(r for r, _ in known) / n
        mean_t = sum(t for _, t in known) / n
        spread = sum((r - mean_r) ** 2 for r, _ in known)
        slope = sum((r - mean_r) * (t - mean_t) for r, t in known) / spread if spread else 0.0
        slope = max(slope, 0.0)  # a step of more rows never costs less than the line says of fewer
        return max(1e-6, mean_t + slope * (rows - mean_r))

    def acceptance(self, bucket: int) -> float:
        with self._lock:
            tried, taken = self.tried.get(bucket, 0.0), self.taken.get(bucket, 0.0)
        return (taken + PRIOR_ACCEPT * PRIOR_WEIGHT) / (tried + PRIOR_WEIGHT)

    # --- the decision ---------------------------------------------------------------------------------------------
    def choose(self, offers: Sequence[tuple[Sequence[int], int]]) -> list[int]:
        """Drafts per request for one step. `offers` holds each request's available drafts and their bucket; the
        answer never exceeds them or the plan's rows. All zeros means a plain step."""
        n = len(offers)
        ks = [0] * n
        budget = self.rows - n
        if n == 0 or budget <= 0 or not any(tokens for tokens, _ in offers):
            return ks
        rates = [self.acceptance(bucket) for _, bucket in offers]
        reach = [1.0] * n  # the chance that request i's next draft is reached and accepted, before multiplying
        tokens, rows = float(n), n
        plain = n / self.price(n)
        rate = plain
        while budget > 0:
            best, gain = None, 0.0
            for i, (drafts, _) in enumerate(offers):
                if ks[i] < len(drafts):
                    add = reach[i] * rates[i]
                    if add > gain:
                        best, gain = i, add
            if best is None:
                break
            new_rate = (tokens + gain) / self.price(rows + 1)
            if new_rate <= rate:
                break
            ks[best] += 1
            reach[best] *= rates[best]
            tokens, rows, rate, budget = tokens + gain, rows + 1, new_rate, budget - 1
        if rate < MARGIN * plain:
            return [0] * n
        return ks

    # --- learning ---------------------------------------------------------------------------------------------------
    def observe_step(self, rows: int, seconds: float, speculative: bool) -> None:
        if not (seconds > 0) or rows < 1:
            return
        with self._lock:
            old = self.cost.get(rows)
            self.cost[rows] = seconds if old is None else (1 - COST_WEIGHT) * old + COST_WEIGHT * seconds
            self.cost_samples[rows] = self.cost_samples.get(rows, 0) + 1
            if speculative:
                self.steps += 1
            else:
                self.plain_steps += 1

    def observe_drafts(self, bucket: int, drafted: int, accepted: int) -> None:
        """`accepted` of `drafted` drafts were taken; the first refused one was tried too, later ones never were."""
        if drafted < 1:
            return
        with self._lock:
            self.tried[bucket] = self.tried.get(bucket, 0.0) + accepted + (1 if accepted < drafted else 0)
            self.taken[bucket] = self.taken.get(bucket, 0.0) + accepted
            self.drafted += drafted
            self.accepted += accepted

    def summary(self) -> dict[str, Any]:
        with self._lock:
            cost = {str(r): round(t * 1000, 3) for r, t in sorted(self.cost.items())}
            buckets = sorted(set(self.tried) | set(self.taken))
            counts = (self.steps, self.plain_steps, self.drafted, self.accepted)
        return {"policy": "context", "speculative_steps": counts[0], "plain_steps": counts[1],
                "drafted": counts[2], "accepted": counts[3],
                "acceptance": {str(b): round(self.acceptance(b), 4) for b in buckets},
                "step_ms_by_rows": cost}


__all__ = ["MARGIN", "MAX_DRAFT", "WindowPolicy", "accept", "draft"]
