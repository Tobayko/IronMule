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
import hashlib
import json
import os
from pathlib import Path
import statistics
import threading
import weakref
from typing import Any, Iterable

POLICY_FILE = "serving-policy.json"
LOG_FILE = "serving-log.jsonl"
WIDTHS = (1, 2, 4, 8)
GATHERS_MS = (0, 50, 150, 400)
ORDERS = ("fifo", "sjf")
SCHEDULE_SCHEMA = "ironmule.schedule_evidence.v1"


@dataclass(frozen=True)
class ServingPolicy:
    max_width: int = 32
    join: bool = False  # SERVE8 killed: joining cost 1B 14 % aggregate tok/s with 4 chats (SERVE8c)
    gather_ms: int = 0
    source: str = "default"
    order: str = "fifo"  # RL6: "sjf" starts the waiting request with the smallest output limit first

    def __post_init__(self) -> None:
        if not (type(self.max_width) is int and 1 <= self.max_width <= 32):
            raise ValueError("max_width must be an int in 1..32")
        if type(self.join) is not bool:
            raise ValueError("join must be a bool")
        if not (type(self.gather_ms) is int and 0 <= self.gather_ms <= 2000):
            raise ValueError("gather_ms must be an int in 0..2000")
        if self.order not in ORDERS:
            raise ValueError("order must be fifo or sjf")


def load_policy(root: Path) -> ServingPolicy:
    """The stored policy, or the default (widest groups, no joining, no wait: SERVE4's queueing)
    if none is stored. A malformed file is ignored rather than trusted: scheduling falls back to it."""
    path = Path(root) / POLICY_FILE
    try:
        data = json.loads(path.read_text())
        return ServingPolicy(**{k: data[k] for k in ("max_width", "join", "gather_ms", "source")},
                             order=data.get("order", "fifo"))
    except (OSError, ValueError, KeyError, TypeError):
        return ServingPolicy()


def stored_policy_exists(root: Path | None) -> bool:
    return root is not None and (Path(root) / POLICY_FILE).exists()


def qualified_schedule(binding: dict[str, Any], folder: Path | None = None) -> tuple[ServingPolicy, dict] | None:
    """RL6/LD1-V: a fixed schedule shipped with the hardware comparison that qualified it, for exactly the
    model, revision, GPU name, GPU count and engine it was measured on. Anything else, or a record whose digest,
    schema or verdict does not hold, gives None and the default schedule serves."""
    folder = Path(__file__).with_name("dispatch") if folder is None else Path(folder)
    for path in sorted(folder.glob("*/schedule.json")):
        try:
            record = json.loads(path.read_text())
            body = {k: v for k, v in record.items() if k != "sha256"}
            if (hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != record.get("sha256")
                    or body.get("schema") != SCHEDULE_SCHEMA or body.get("verdict") != "pass"):
                continue
            wanted = body.get("binding") or {}
            if any(wanted.get(key) != binding.get(key) for key in {"model_id", "revision", "gpu", "devices", *wanted}):
                continue
            policy = body["policy"]
            return (ServingPolicy(max_width=policy["max_width"], gather_ms=policy["gather_ms"],
                                  order=policy["order"], source=f"qualified:{body['experiment']}"), body)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def schedule_record(experiment: str, binding: dict[str, Any], policy: ServingPolicy, domain: list[str],
                    rival: str, gain: float, gain_ci95: list[float], noise: float) -> dict[str, Any]:
    """The record `qualified_schedule` accepts, from a passed comparison; the verdict is the bound's."""
    body = {"schema": SCHEDULE_SCHEMA, "experiment": experiment, "binding": binding,
            "policy": {"max_width": policy.max_width, "gather_ms": policy.gather_ms, "order": policy.order},
            "domain": list(domain), "rival": rival, "gain": gain, "gain_ci95": list(gain_ci95), "noise": noise,
            "verdict": "pass" if gain_ci95[0] > noise else "fail"}
    return {**body, "sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}


def save_policy(root: Path, policy: ServingPolicy, evidence: dict[str, Any] | None = None) -> Path:
    path = Path(root) / POLICY_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**asdict(policy), "evidence": evidence or {}}, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


# ADM1 (2026-11-25): see ironmule_product/http_server.py; "0" restores pathlib's appends
_ADM1 = os.environ.get("IRONMULE_ADM1", "1") != "0"
# ADM2 (2026-11-25): the service's logs (stability and serving log) are written by one thread per log, which writes
# every line waiting in one call. A 32-request burst takes about six lines per request (admit, join, member_done,
# http, end, the serving log), each an open, write and close on the Python path that the burst's HTTP threads, the
# worker's reader and the scheduler share; on the A100 pods the 32 done frames of a wave's last step were read over
# 50-85 ms after the engine had finished them in 10-13 ms. The lines and their order are those `append` writes; a
# reader in this process sees them after `flush()` (`read()` and `close()` flush), another process within
# milliseconds. A100 pod 34, paired waves on top of ADM1 (with BF1): 32-client wave -33.1 ms (8 of 8 rounds). "0" writes
# every line on its caller's thread, as before.
_ADM2 = os.environ.get("IRONMULE_ADM2", "1") != "0"
_DEFERRED: "weakref.WeakSet[ServingLog]" = weakref.WeakSet()


def flush_logs() -> None:
    """ADM2: write every deferred log line of this process now (tests and diagnostics that read the files)."""
    for log in list(_DEFERRED):
        log.flush()


class ServingLog:
    """JSON lines, one per completed request; times in seconds on one monotonic clock. Capped at
    ``max_bytes`` (about 5000 requests at the default): when the next line would pass it, the older
    half is dropped, since the dreamer needs recent traffic, not history."""

    def __init__(self, root: Path, max_bytes: int = 1 << 20, name: str = LOG_FILE, deferred: bool = False):
        self.path = Path(root) / name
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self.deferred = bool(deferred)  # ADM2, while _ADM2 holds
        self._waiting: list[bytes] = []
        self._writing = False
        self._ready = threading.Condition()
        self._thread: threading.Thread | None = None
        if self.deferred:
            _DEFERRED.add(self)

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True) + "\n"
        if self.deferred and _ADM2:  # ADM2: the writer thread takes it
            with self._ready:
                self._waiting.append(line.encode("utf-8"))
                if self._thread is None:
                    self._thread = threading.Thread(target=self._write_behind, name="ironmule-log-writer",
                                                    daemon=True)
                    self._thread.start()
                self._ready.notify_all()
            return
        if self.deferred:
            self.flush()  # lines written behind before the switch turned it off come first
        if _ADM1:
            self._append_fd([line.encode("utf-8")])
            return
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

    def flush(self) -> None:
        """ADM2: return once every line appended so far is in the file."""
        if not self.deferred:
            return
        with self._ready:
            while self._waiting or self._writing:
                self._ready.wait(0.05)

    def _write_behind(self) -> None:
        while True:
            with self._ready:
                while not self._waiting:
                    self._ready.wait()
                lines, self._waiting = self._waiting, []
                self._writing = True
            try:
                self._append_fd(lines)
            finally:
                with self._ready:
                    self._writing = False
                    self._ready.notify_all()

    def _append_fd(self, lines: list[bytes]) -> None:
        """ADM1: the same bytes and cap as `append` line by line, with one open, fstat and write on a descriptor
        instead of pathlib's exists, stat and open for each line; ADM2 hands it every line waiting at once."""
        with self._lock:
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
                try:
                    size = os.fstat(fd).st_size
                    chunk: list[bytes] = []
                    for data in lines:
                        if size and size + len(data) > self.max_bytes:  # the older half goes, as in `append`
                            if chunk:
                                os.write(fd, b"".join(chunk))
                                chunk = []
                            os.close(fd)
                            fd = -1
                            kept = self.path.read_text().splitlines(keepends=True)
                            tmp = self.path.with_suffix(".tmp")
                            tmp.write_text("".join(kept[len(kept) // 2:]))
                            tmp.replace(self.path)
                            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
                            size = os.fstat(fd).st_size
                        chunk.append(data)
                        size += len(data)
                    if chunk:
                        os.write(fd, b"".join(chunk))
                finally:
                    if fd >= 0:
                        os.close(fd)
            except OSError:
                pass  # logging never fails a request

    def read(self) -> list[dict[str, Any]]:
        self.flush()
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
