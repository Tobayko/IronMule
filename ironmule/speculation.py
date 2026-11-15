"""SP2: speculation from the request's own context, with a window learned while the server runs (BACKLOG SP2).

Drafts are copied from the request's prompt and answer so far (prompt lookup, SP1-0): the longest suffix of 3 down
to 1 tokens that occurred before, its most recent earlier occurrence, and up to `MAX_DRAFT` tokens after it. The
stacked plan verifies a request's drafts as rows of its own slot at consecutive positions, in the same forward as
the other requests' rows; a request takes its drafts up to the first that differs from the plan's own argmax, plus
that argmax, so its tokens are exactly those of one row at a time.

How many drafts each request gets is `WindowPolicy`'s decision, Strata's draft policy
(`src/spec/draft_policy.cpp`, BACKLOG HQ1) learned online on this machine:

* a plain step's time by its rows, an exponential average of its chained steps; a speculative step's time by a line
  through every speculative step measured so far (policy version 2, SP3: the two are priced apart, since a
  speculative step reads the host every step and a plain one every fourth);
* per draft bucket (its match length, at least `MIN_NGRAM`), the chance that a draft is accepted when the ones before
  it were, from every verified draft, with a prior of 0.3 worth 4 drafts;
* each step it hands out drafts greedily where they add the most expected tokens while the speculative step's expected
  tokens per second rises, and speculates only where that beats the plain step of the same requests by `MARGIN`.

The policy never changes a token: it only decides how many positions are checked at once.
"""

from __future__ import annotations

import threading
from typing import Any, Sequence

POLICY_VERSION = 2  # 2 (SP3): no 1-token matches; plain and speculative steps priced apart
MAX_DRAFT = 7
MAX_NGRAM = 3
MIN_NGRAM = 2  # SP3: in SP2-H, drafts from 1-token matches were right 0.07-0.35 of the time
LINE_MEMORY = 50  # SP3: a new speculative step weighs as much as the last 50 together
PRIOR_ACCEPT, PRIOR_WEIGHT = 0.3, 4.0
COST_WEIGHT = 0.2
ROW_SLOPE = 0.06  # CPU6-A2: eight stacked rows cost 1.42x one row on this container's CPU
MARGIN = 1.02


def draft(context: Sequence[int], k: int, max_ngram: int = MAX_NGRAM,
          min_ngram: int = MIN_NGRAM) -> tuple[list[int], int]:
    """Up to `k` tokens after the most recent earlier occurrence of the context's longest repeated suffix of
    `min_ngram` to `max_ngram` tokens, and that suffix's length (0 when nothing repeats)."""
    if k < 1:
        return [], 0
    n_ctx = len(context)
    for n in range(min(max_ngram, n_ctx - 1), min_ngram - 1, -1):
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
        self.plain: dict[int, float] = {}  # rows -> seconds, an exponential average of plain steps
        self._line = [0.0] * 5  # decayed sums of w, w*r, w*r*r, w*t, w*r*t over speculative steps
        self._line_rows: set[int] = set()
        self.tried: dict[int, float] = {}
        self.taken: dict[int, float] = {}
        self.steps = self.plain_steps = self.drafted = self.accepted = 0
        self._lock = threading.Lock()

    # --- prices -------------------------------------------------------------------------------------------------
    def plain_price(self, rows: int) -> float:
        """Seconds for a plain step of `rows` rows: measured, else scaled from the nearest measured row count by the
        prior slope, else the prior slope alone (a relative price)."""
        with self._lock:
            if rows in self.plain:
                return self.plain[rows]
            known = sorted(self.plain.items(), key=lambda item: abs(item[0] - rows))
        if not known:
            return 1.0 + ROW_SLOPE * (rows - 1)
        r0, t0 = known[0]
        return t0 * (1.0 + ROW_SLOPE * (rows - 1)) / (1.0 + ROW_SLOPE * (r0 - 1))

    def speculative_price(self, rows: int, requests: int) -> float:
        """Seconds for a speculative step of `rows` rows serving `requests` requests: the line through measured
        speculative steps once it has two row counts, else the plain price of the requests plus the prior slope."""
        with self._lock:
            w, wr, wrr, wt, wrt = self._line
            fitted = len(self._line_rows) >= 2
        if fitted:
            spread = w * wrr - wr * wr
            slope = max(0.0, (w * wrt - wr * wt) / spread) if spread > 0 else 0.0
            return max(1e-6, (wt - slope * wr) / w + slope * rows)
        return self.plain_price(requests) * (1.0 + ROW_SLOPE * (rows - requests))

    def price(self, rows: int) -> float:
        """A plain step's price (kept for callers of policy version 1)."""
        return self.plain_price(rows)

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
        rate = n / self.speculative_price(n, n)
        while budget > 0:
            best, gain = None, 0.0
            for i, (drafts, _) in enumerate(offers):
                if ks[i] < len(drafts):
                    add = reach[i] * rates[i]
                    if add > gain:
                        best, gain = i, add
            if best is None:
                break
            new_rate = (tokens + gain) / self.speculative_price(rows + 1, n)
            if new_rate <= rate:
                break
            ks[best] += 1
            reach[best] *= rates[best]
            tokens, rows, rate, budget = tokens + gain, rows + 1, new_rate, budget - 1
        if not any(ks) or rate < MARGIN * n / self.plain_price(n):
            return [0] * n
        return ks

    # --- learning ---------------------------------------------------------------------------------------------------
    def observe_step(self, rows: int, seconds: float, speculative: bool) -> None:
        if not (seconds > 0) or rows < 1:
            return
        with self._lock:
            if speculative:
                keep = 1.0 - 1.0 / LINE_MEMORY
                self._line = [value * keep for value in self._line]
                for i, add in enumerate((1.0, rows, rows * rows, seconds, rows * seconds)):
                    self._line[i] += add
                self._line_rows.add(rows)
                self.steps += 1
            else:
                old = self.plain.get(rows)
                self.plain[rows] = seconds if old is None else (1 - COST_WEIGHT) * old + COST_WEIGHT * seconds
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
            plain = {str(r): round(t * 1000, 3) for r, t in sorted(self.plain.items())}
            buckets = sorted(set(self.tried) | set(self.taken))
            counts = (self.steps, self.plain_steps, self.drafted, self.accepted)
            fitted = len(self._line_rows) >= 2
        line = ({str(r): round(self.speculative_price(r, 1) * 1000, 3) for r in range(1, self.rows + 1)}
                if fitted else {})
        return {"policy": "context", "policy_version": POLICY_VERSION, "speculative_steps": counts[0],
                "plain_steps": counts[1], "drafted": counts[2], "accepted": counts[3],
                "acceptance": {str(b): round(self.acceptance(b), 4) for b in buckets},
                "plain_step_ms_by_rows": plain, "speculative_step_ms_by_rows": line}


class StepPolicy:
    """GG1, policy version 3: drafts per request for an engine whose every step is one call priced by its rows alone
    (ggml: plain and speculative steps are the same decode call). A row count's price is the median of its last
    `WINDOW` measured steps once it has `PROBES` of them; before that it is priced from the nearest measured row count
    by the prior slope, so a size is tried a few times before a guessed price may veto it (Strata's probing, which
    SP3 lacked)."""

    WINDOW, PROBES, VERSION = 16, 3, 3

    def __init__(self, rows: int):
        if rows < 1:
            raise ValueError("a step needs at least one row")
        self.rows = rows
        self.times: dict[int, list[float]] = {}
        self.tried: dict[int, float] = {}
        self.taken: dict[int, float] = {}
        self.steps = self.plain_steps = self.drafted = self.accepted = 0
        self._lock = threading.Lock()

    def price(self, rows: int) -> float:
        with self._lock:
            measured = {r: sorted(t)[len(t) // 2] for r, t in self.times.items() if len(t) >= self.PROBES}
        if rows in measured:
            return measured[rows]
        if not measured:
            return 1.0 + ROW_SLOPE * (rows - 1)
        r0 = min(measured, key=lambda r: (abs(r - rows), r))
        return measured[r0] * (1.0 + ROW_SLOPE * (rows - 1)) / (1.0 + ROW_SLOPE * (r0 - 1))

    def acceptance(self, bucket: int) -> float:
        with self._lock:
            tried, taken = self.tried.get(bucket, 0.0), self.taken.get(bucket, 0.0)
        return (taken + PRIOR_ACCEPT * PRIOR_WEIGHT) / (tried + PRIOR_WEIGHT)

    def choose(self, offers: Sequence[tuple[Sequence[int], int]]) -> list[int]:
        n = len(offers)
        ks = [0] * n
        budget = self.rows - n
        if n == 0 or budget <= 0 or not any(tokens for tokens, _ in offers):
            return ks
        rates = [self.acceptance(bucket) for _, bucket in offers]
        reach = [1.0] * n
        tokens, rows = float(n), n
        plain = n / self.price(n)
        rate = plain
        while budget > 0:
            best, gain = None, 0.0
            for i, (drafts, _) in enumerate(offers):
                if ks[i] < len(drafts) and reach[i] * rates[i] > gain:
                    best, gain = i, reach[i] * rates[i]
            if best is None:
                break
            new_rate = (tokens + gain) / self.price(rows + 1)
            if new_rate <= rate:
                break
            ks[best] += 1
            reach[best] *= rates[best]
            tokens, rows, rate, budget = tokens + gain, rows + 1, new_rate, budget - 1
        return ks if any(ks) and rate >= MARGIN * plain else [0] * n

    def observe_step(self, rows: int, seconds: float, speculative: bool) -> None:
        if not (seconds > 0) or rows < 1:
            return
        with self._lock:
            times = self.times.setdefault(rows, [])
            times.append(seconds)
            del times[:-self.WINDOW]
            if speculative:
                self.steps += 1
            else:
                self.plain_steps += 1

    def observe_drafts(self, bucket: int, drafted: int, accepted: int) -> None:
        if drafted < 1:
            return
        with self._lock:
            self.tried[bucket] = self.tried.get(bucket, 0.0) + accepted + (1 if accepted < drafted else 0)
            self.taken[bucket] = self.taken.get(bucket, 0.0) + accepted
            self.drafted += drafted
            self.accepted += accepted

    def summary(self) -> dict[str, Any]:
        with self._lock:
            counts = (self.steps, self.plain_steps, self.drafted, self.accepted)
            buckets = sorted(set(self.tried) | set(self.taken))
            sizes = sorted(self.times)
        return {"policy": "context", "policy_version": self.VERSION, "speculative_steps": counts[0],
                "plain_steps": counts[1], "drafted": counts[2], "accepted": counts[3],
                "acceptance": {str(b): round(self.acceptance(b), 4) for b in buckets},
                "step_ms_by_rows": {str(r): round(self.price(r) * 1000, 3) for r in sizes}}


__all__ = ["MARGIN", "MAX_DRAFT", "MIN_NGRAM", "POLICY_VERSION", "StepPolicy", "WindowPolicy", "accept", "draft"]


COLLECTIVE_BUCKET = 100  # drafts from other requests' answers are learned apart: bucket 100 + match length
COLLECTIVE_NGRAM = 4


class CollectiveIndex:
    """SP4 (2026-11-13): drafts from what other requests answered, not only from the request's own context.

    Prompt lookup (llama.cpp's `--lookup`, vLLM's n-gram speculator, SP1-0 here) copies drafts from the request itself.
    On a server many users ask alike things, and an answer's next tokens often repeat what the model answered someone
    else. This index maps every `COLLECTIVE_NGRAM` .. 2 tokens of recent answers to the tokens that followed them, most
    recent first; a request without a long own match drafts from it. The drafts are verified like any others: a token
    is kept only where it is the model's own next token, so nothing of another user's answer reaches a reply unless
    the model would have written it anyway. Their acceptance is learned in its own bucket, so the step policy stops
    drafting from it where it does not pay. IRONMULE_GGML_COLLECTIVE=0 turns it off."""

    def __init__(self, keys: int = 200_000, ngram: int = COLLECTIVE_NGRAM):
        from collections import OrderedDict

        self.keys, self.ngram = keys, ngram
        self.table: OrderedDict[tuple[int, ...], tuple[int, ...]] = OrderedDict()
        self._lock = threading.Lock()
        self.added = 0

    def add(self, tokens: Sequence[int]) -> None:
        """Index an answer: every n-gram (n = ngram .. 2) and up to MAX_DRAFT tokens after it."""
        tokens = list(tokens)
        with self._lock:
            for n in range(self.ngram, 1, -1):
                for i in range(len(tokens) - n):
                    key = tuple(tokens[i:i + n])
                    self.table[key] = tuple(tokens[i + n:i + n + MAX_DRAFT])
                    self.table.move_to_end(key)
            while len(self.table) > self.keys:
                self.table.popitem(last=False)
            self.added += 1

    def draft(self, context: Sequence[int], k: int) -> tuple[list[int], int]:
        """Up to `k` tokens that followed the context's last n tokens in some answer (longest n first), and the
        bucket (COLLECTIVE_BUCKET + n), or ([], 0)."""
        if k < 1:
            return [], 0
        with self._lock:
            for n in range(min(self.ngram, len(context)), 1, -1):
                found = self.table.get(tuple(context[len(context) - n:]))
                if found:
                    return list(found[:k]), COLLECTIVE_BUCKET + n
        return [], 0
