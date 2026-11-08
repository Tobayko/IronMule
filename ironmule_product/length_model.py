"""AG2: an online model of how long an answer will be, for the agent's "shortest predicted first" order.

It learns from every finished live request (label: the completion tokens it made) and predicts, once per
request when the request is first seen, the number of tokens its answer will take. Features are hashed word
unigrams and bigrams of the last message (its first `WORDS` words), the log of the message's length and the
log of the output limit. Learning is SGD on the log length; the model keeps weights only and no text.

A prediction changes the order requests are served in, never an answer (BACKLOG AG2).
"""

from __future__ import annotations

from collections import deque
import math
import re
import threading
from typing import Any
import zlib

SCHEMA = "ironmule.agent.lengths.v1"
BUCKETS = 1 << 14          # hashed features
WORDS = 64                 # words of the last message that are read
RATE = 0.05                # SGD step for the hashed features (scaled by 1/sqrt(features))
SLOW_RATE = 0.01           # SGD step for the bias and the two length features
QUALITY_WINDOW = 512       # recent (prediction, truth) pairs kept for the status's correlation
_WORD = re.compile(r"[a-z0-9']+")


def features(request: Any) -> tuple[tuple[int, ...], float, float]:
    """(hashed feature indices, log message length, log output limit) of a request; reads its last message."""
    messages = getattr(request, "messages", ()) or ()
    text = messages[-1][1] if messages else ""
    words = _WORD.findall(text.lower())[:WORDS]
    found = {zlib.crc32(word.encode()) % BUCKETS for word in words}
    found |= {zlib.crc32(f"{a} {b}".encode()) % BUCKETS for a, b in zip(words, words[1:])}
    return tuple(sorted(found)), math.log1p(len(text)), math.log1p(max(1, int(request.max_tokens)))


def _rank(values: list[float]) -> list[float]:
    """Ranks with ties sharing their average rank: consecutive ranks gave a constant prediction a perfect
    correlation with any truth (review of 2026-11-08). A constant input then has no spread, and `spearman` is
    undefined (None), as in SciPy."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2
        start = end + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    sa = math.sqrt(sum((x - ma) ** 2 for x in ra))
    sb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if sa == 0 or sb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sa * sb)


class LengthModel:
    def __init__(self) -> None:
        self.weights: dict[int, float] = {}
        self.bias = math.log(64.0)
        self.w_chars = 0.0
        self.w_limit = 0.0
        self.seen = 0
        self.recent: deque[tuple[float, float]] = deque(maxlen=QUALITY_WINDOW)
        self._lock = threading.Lock()  # several agents (one per GPU, AG2) may share one model
    def _score(self, feats: tuple[tuple[int, ...], float, float]) -> float:
        indices, chars, limit = feats
        scale = 1.0 / math.sqrt(len(indices) or 1)
        return (self.bias + self.w_chars * chars + self.w_limit * limit
                + scale * sum(self.weights.get(i, 0.0) for i in indices))

    def predict(self, feats: tuple[tuple[int, ...], float, float], limit: int) -> float:
        """Predicted answer tokens, at least 1 and at most the request's output limit."""
        return max(1.0, min(float(limit), math.expm1(max(0.0, min(20.0, self._score(feats))))))

    def learn(self, feats: tuple[tuple[int, ...], float, float], tokens: int, predicted: float | None = None) -> None:
        """One finished request: its features and the tokens its answer took. `predicted` is the prediction
        made when the request was first seen (before this lesson), recorded for the status's correlation."""
        with self._lock:
            self._learn(feats, tokens, predicted)

    def _learn(self, feats: tuple[tuple[int, ...], float, float], tokens: int, predicted: float | None) -> None:
        indices, chars, limit = feats
        target = math.log1p(max(1, int(tokens)))
        error = target - self._score(feats)
        step = RATE / math.sqrt(len(indices) or 1)
        for i in indices:
            self.weights[i] = self.weights.get(i, 0.0) + step * error
        self.bias += SLOW_RATE * error
        self.w_chars += SLOW_RATE * error * chars / 10.0
        self.w_limit += SLOW_RATE * error * limit / 10.0
        self.seen += 1
        if predicted is not None:
            self.recent.append((math.log1p(predicted), target))

    def quality(self) -> float | None:
        """Spearman correlation of the recent predictions (made before each lesson) with the truth."""
        pairs = list(self.recent)
        return spearman([p for p, _ in pairs], [t for _, t in pairs])

    def to_json(self) -> dict[str, Any]:
        with self._lock:
            return self._json()

    def _json(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "weights": {str(k): round(v, 6) for k, v in self.weights.items() if v},
                "bias": self.bias, "w_chars": self.w_chars, "w_limit": self.w_limit, "seen": self.seen,
                "recent": [list(pair) for pair in self.recent]}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "LengthModel":
        if data.get("schema") != SCHEMA:
            raise ValueError("unknown length model schema")
        model = cls()
        model.weights = {int(k): float(v) for k, v in data["weights"].items() if 0 <= int(k) < BUCKETS}
        model.bias, model.w_chars, model.w_limit = float(data["bias"]), float(data["w_chars"]), float(data["w_limit"])
        model.seen = int(data["seen"])
        model.recent.extend((float(p), float(t)) for p, t in data.get("recent", ()))
        return model
