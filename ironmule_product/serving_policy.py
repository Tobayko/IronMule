"""RSI4: how the service groups chats under load, learned from its own traffic.

Three decisions shape serving under load, none of them hardware-specific: how many streams a group
may hold (``max_width``), whether a chat that arrives while a group streams joins it at the next
decode round (``join``, SERVE8), and how long the scheduler waits for company before it starts a
group (``gather_ms``; 0 never waits). The service logs every completed request (arrival, dispatch,
first token, end, token counts, group width); ``dream`` fits a cost model to that log — prefill
seconds per prompt token and decode seconds per round at each width, as measured — replays the
logged arrivals under every candidate policy in a discrete-event simulation, and keeps the one with
the best 95th-percentile time to first token that holds aggregate throughput within 3 % of the
current policy's. A dreamed policy changes scheduling only; every answer stays the request's own
greedy answer (grouped batch-1 is exact, SERVE4/SERVE8). It serves after hardware qualification:
the RSI4 run compares it with fixed SERVE8 on held-out traffic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import threading
from typing import Any, Iterable

POLICY_FILE = "serving-policy.json"
LOG_FILE = "serving-log.jsonl"
WIDTHS = (1, 2, 4, 8)
GATHERS_MS = (0, 50, 150, 400)


@dataclass(frozen=True)
class ServingPolicy:
    max_width: int = 32
    join: bool = False  # SERVE8 killed: joining cost 1B 14 % aggregate tok/s with 4 chats (SERVE8c)
    gather_ms: int = 0
    source: str = "default"

    def __post_init__(self) -> None:
        if not (type(self.max_width) is int and 1 <= self.max_width <= 32):
            raise ValueError("max_width must be an int in 1..32")
        if type(self.join) is not bool:
            raise ValueError("join must be a bool")
        if not (type(self.gather_ms) is int and 0 <= self.gather_ms <= 2000):
            raise ValueError("gather_ms must be an int in 0..2000")


def load_policy(root: Path) -> ServingPolicy:
    """The stored policy, or the default (widest groups, no joining, no wait: SERVE4's queueing)
    if none is stored. A malformed file is ignored rather than trusted: scheduling falls back to it."""
    path = Path(root) / POLICY_FILE
    try:
        data = json.loads(path.read_text())
        return ServingPolicy(**{k: data[k] for k in ("max_width", "join", "gather_ms", "source")})
    except (OSError, ValueError, KeyError, TypeError):
        return ServingPolicy()


def save_policy(root: Path, policy: ServingPolicy, evidence: dict[str, Any] | None = None) -> Path:
    path = Path(root) / POLICY_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**asdict(policy), "evidence": evidence or {}}, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


class ServingLog:
    """JSON lines, one per completed request; times in seconds on one monotonic clock. Capped at
    ``max_bytes`` (about 5000 requests at the default): when the next line would pass it, the older
    half is dropped, since the dreamer needs recent traffic, not history."""

    def __init__(self, root: Path, max_bytes: int = 1 << 20, name: str = LOG_FILE):
        self.path = Path(root) / name
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True) + "\n"
        with self._lock:
            try:
                if self.path.exists() and self.path.stat().st_size + len(line) > self.max_bytes:
                    lines = self.path.read_text().splitlines(keepends=True)
                    tmp = self.path.with_suffix(".tmp")
                    tmp.write_text("".join(lines[len(lines) // 2:]))
                    tmp.replace(self.path)
                with self.path.open("a") as stream:
                    stream.write(line)
            except OSError:
                pass  # logging never fails a request

    def read(self) -> list[dict[str, Any]]:
        try:
            lines = self.path.read_text().splitlines()
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if all(isinstance(record.get(k), (int, float)) for k in ("arrive", "start", "first", "end",
                                                                       "prompt_tokens", "new_tokens", "width")):
                out.append(record)
        return out


@dataclass(frozen=True)
class CostModel:
    """Seconds per prompt token of prefill, and seconds per decode round at each group width."""
    prefill_s_per_token: float
    round_s: dict[int, float]

    def step(self, width: int) -> float:
        known = sorted(self.round_s)
        if width in self.round_s:
            return self.round_s[width]
        below = [w for w in known if w < width]
        above = [w for w in known if w > width]
        if below and above:
            lo, hi = below[-1], above[0]
            return self.round_s[lo] + (self.round_s[hi] - self.round_s[lo]) * (width - lo) / (hi - lo)
        edge = below[-1] if below else above[0]
        return self.round_s[edge] * (width / edge if width > edge else 1.0)


def fit(records: Iterable[dict[str, Any]]) -> CostModel | None:
    """Prefill cost from requests that ran alone (first token minus start over prompt tokens),
    round cost per width from every request's decode (end minus first over its new tokens)."""
    records = list(records)
    alone = [(r["first"] - r["start"]) / r["prompt_tokens"] for r in records
             if r["width"] == 1 and r["prompt_tokens"] > 0 and r["first"] >= r["start"]]
    if not alone:
        alone = [(r["first"] - r["start"]) / r["prompt_tokens"] for r in records
                 if r["prompt_tokens"] > 0 and r["first"] >= r["start"]]
    rounds: dict[int, list[float]] = {}
    for r in records:
        if r["new_tokens"] > 1 and r["end"] > r["first"]:
            rounds.setdefault(int(r["width"]), []).append((r["end"] - r["first"]) / (r["new_tokens"] - 1))
    if not alone or not rounds:
        return None
    return CostModel(statistics.median(alone), {w: statistics.median(v) for w, v in rounds.items()})


def simulate(arrivals: list[tuple[float, int, int]], policy: ServingPolicy, cost: CostModel,
             capacity: int = 8) -> dict[str, float]:
    """Replay (arrival s, prompt tokens, new tokens) through one model under ``policy``.

    A free server takes the oldest waiting chats up to the group width (waiting up to gather_ms
    for more when fewer are there), prefills them one after another, then decodes in rounds whose
    cost depends on how many still stream; with ``join`` a chat that arrives while the group runs
    is prefilled at the next round boundary and streams from there while the width allows.
    Returns p95 and max time to first token and aggregate tokens per second."""
    width = max(1, min(policy.max_width, capacity))
    pending = sorted(arrivals)
    i, now, ttft, total_tokens = 0, 0.0, [], 0
    first_arrival = pending[0][0] if pending else 0.0
    queue: list[tuple[float, int, int]] = []

    def admit(until: float) -> None:
        nonlocal i
        while i < len(pending) and pending[i][0] <= until:
            queue.append(pending[i])
            i += 1

    while i < len(pending) or queue:
        admit(now)
        if not queue:
            now = pending[i][0]
            continue
        if width > 1 and len(queue) < width and policy.gather_ms > 0:
            now += policy.gather_ms / 1000.0
            admit(now)
        group = queue[:width]
        del queue[:width]
        active: list[list[float]] = []  # [remaining tokens]
        for arrive, prompt, new in group:
            now += prompt * cost.prefill_s_per_token
            ttft.append(now - arrive)
            total_tokens += new
            active.append([new - 1])
        while active:
            now += cost.step(len(active))
            for entry in active:
                entry[0] -= 1
            active = [e for e in active if e[0] > 0]
            if policy.join and width > 1:
                admit(now)
                while queue and len(active) < width:
                    arrive, prompt, new = queue.pop(0)
                    now += prompt * cost.prefill_s_per_token
                    ttft.append(now - arrive)
                    total_tokens += new
                    active.append([new - 1])
    ttft.sort()
    span = max(now - first_arrival, 1e-9)
    p95 = ttft[min(len(ttft) - 1, int(round(0.95 * (len(ttft) - 1))))] if ttft else 0.0
    return {"p95_ttft_s": p95, "max_ttft_s": ttft[-1] if ttft else 0.0, "tok_s": total_tokens / span,
            "requests": len(ttft)}


def dream(records: list[dict[str, Any]], current: ServingPolicy, capacity: int = 8) -> tuple[ServingPolicy, dict]:
    """Every candidate policy replayed on the logged arrivals; the best p95 TTFT among those that
    keep simulated throughput within 3 % of the current policy's. Without enough log, the current
    policy stays."""
    cost = fit(records)
    if cost is None or len(records) < 8:
        return current, {"reason": "not enough log", "records": len(records)}
    t0 = min(r["arrive"] for r in records)
    arrivals = [(r["arrive"] - t0, int(r["prompt_tokens"]), int(r["new_tokens"])) for r in records]
    base = simulate(arrivals, current, cost, capacity)
    table = []
    for width in WIDTHS:
        for join in (True, False):
            for gather in GATHERS_MS:
                candidate = ServingPolicy(width, join, gather, "dream")
                table.append((candidate, simulate(arrivals, candidate, cost, capacity)))
    feasible = [(p, m) for p, m in table if m["tok_s"] >= 0.97 * base["tok_s"]]
    best, metrics = min(feasible or [(current, base)], key=lambda pm: (pm[1]["p95_ttft_s"], -pm[1]["tok_s"]))
    evidence = {"records": len(records), "current": {**asdict(current), **base}, "chosen": metrics,
                "cost": {"prefill_s_per_token": cost.prefill_s_per_token, "round_s": cost.round_s},
                "candidates": len(table)}
    return best, evidence
