"""DM1: speculation with a small draft model, stopped by an optimal rule learned on the machine (docs/LAB.md, DM1).

On a CPU with the ROW1/FA1 kernels a further row of a decode step costs a few per cent of the step (RL-MATH: 2.4 ms
against 99 ms for Qwen3-4B), so verifying drafts is nearly free and one request alone, which sits at the memory bound,
can only gain by getting several tokens out of one read of the weights. A small model of the same family (the same
tokenizer) drafts the next tokens; the target verifies them in one step and keeps them up to the first that is not its
own next token, so answers are the target's.

How many tokens to draft is decided token by token by the optimal myopic rule: one more draft costs a draft call and a
verification row and adds q * p expected tokens (q: the chance that every draft so far is accepted); draft on while
that marginal rate beats the step's rate so far. The chance of a draft comes from the draft model's own top-1
probability through a calibration learned online (accepted or not, by confidence bin); the costs are running means of
this machine's draft calls and steps. Under load the same rule stops drafting by itself: with many requests in a step
a draft call buys too little.

`Drafter` keeps the draft model's KV in step with each request it drafts for (a few slots; a slot's cells are kept
after a request for the next one that shares its prefix).
"""

from __future__ import annotations

from typing import Any, Sequence

BINS = 10
CHUNK = 256  # catch-up tokens a draft call carries at most


class DraftStop:
    """Calibration and costs; `start` and `more` are the rule. Pure arithmetic, no model.

    Two layers. The model: a step's seconds are a line in its rows fitted to the measured steps, a draft call's seconds
    a running mean per rows in the call. And a check on the model: per number of running requests, the tokens per second
    that steps with drafts really made against what plain steps make; where drafting measured slower it stops, and is
    tried again once in `PROBE` steps (the load or the text may have changed)."""

    PROBE, TRUST = 200, 8

    def __init__(self) -> None:
        self.tried = [2.0] * BINS
        self.taken = [2.0 * (i + 0.5) / BINS for i in range(BINS)]
        self.tried[-1], self.taken[-1] = 20.0, 16.0  # most drafts are confident ones: the mean starts near 0.7
        self.call: dict[int, float] = {}  # rows of a draft call -> seconds
        self.by_rows: dict[int, float] = {}  # rows of a step -> seconds
        self.rate: dict[int, list[float]] = {}  # running requests -> [tokens/s of steps with drafts, how many]
        self.asked: dict[int, int] = {}
        self.drafted = self.accepted = self.calls = self.steps = 0

    # --- what the machine costs ---------------------------------------------------------------------------------------
    def _line(self) -> tuple[float, float]:
        """(seconds of a step's first row, seconds per further row) through the measured steps."""
        points = list(self.by_rows.items())
        if len(points) < 2:
            base = points[0][1] - 0.0025 * (points[0][0] - 1) if points else 0.085
            return max(1e-4, base), 0.0025
        n = len(points)
        mx, my = sum(r for r, _ in points) / n, sum(t for _, t in points) / n
        var = sum((r - mx) ** 2 for r, _ in points)
        slope = max(0.0005, sum((r - mx) * (t - my) for r, t in points) / var) if var > 0 else 0.0025
        return max(1e-4, my - slope * (mx - 1)), slope

    @property
    def c_row(self) -> float:
        return self._line()[1]

    def step_seconds(self, rows: int) -> float:
        if rows in self.by_rows:
            return self.by_rows[rows]
        first, slope = self._line()
        return first + slope * (rows - 1)

    def call_seconds(self, rows: int) -> float:
        if rows in self.call:
            return self.call[rows]
        if not self.call:
            return 0.015 + 0.001 * (rows - 1)
        nearest = min(self.call, key=lambda n: abs(n - rows))
        return max(1e-4, self.call[nearest] + 0.001 * (rows - nearest))

    # --- the calibration ----------------------------------------------------------------------------------------------
    def p(self, confidence: float) -> float:
        b = min(BINS - 1, max(0, int(confidence * BINS)))
        return self.taken[b] / self.tried[b]

    def mean(self) -> float:
        return sum(self.taken) / sum(self.tried)

    # --- the rule -----------------------------------------------------------------------------------------------------
    def start(self, running: int, sharers: int = 1) -> bool:
        """Draft at all in this step? One draft per sharing request must beat the plain step in the model, and steps
        with drafts must not have measured slower than plain ones at this load."""
        self.asked[running] = self.asked.get(running, 0) + 1
        plain = running / self.step_seconds(running)
        measured = self.rate.get(running)
        if measured is not None and measured[1] >= self.TRUST and measured[0] < plain:
            if self.asked[running] % self.PROBE:
                return False
        cost = self.call_seconds(sharers) + self.c_row * sharers
        return (running + sharers * self.mean()) / (self.step_seconds(running) + cost) > plain

    def more(self, q: float, expected: float, spent: float, running: int, sharers: int) -> bool:
        """Compute one more draft for a request whose drafts so far are all accepted with chance `q`? `expected` is the
        step's expected draft tokens so far, `spent` the seconds the drafts so far cost, `sharers` the requests that
        would share the draft call. The next draft's confidence is not known before it is paid for, so its chance is
        the mean."""
        cost = self.call_seconds(sharers) / max(1, sharers) + self.c_row
        return q * self.mean() / cost > (running + expected) / (self.step_seconds(running) + spent)

    # --- learning -----------------------------------------------------------------------------------------------------
    def learn(self, confidences: Sequence[float], accepted: int) -> None:
        for i, confidence in enumerate(confidences[:accepted + 1]):  # every draft that was reached
            b = min(BINS - 1, max(0, int(confidence * BINS)))
            self.tried[b] += 1
            self.taken[b] += i < accepted
        self.drafted += len(confidences)
        self.accepted += accepted

    def observe_call(self, seconds: float, rows: int = 1) -> None:
        old = self.call.get(rows)
        self.call[rows] = seconds if old is None else old + 0.1 * (seconds - old)
        self.calls += 1

    def observe_step(self, seconds: float, running: int, draft_rows: int, tokens: int | None = None,
                     draft_seconds: float = 0.0) -> None:
        rows = running + draft_rows
        old = self.by_rows.get(rows)
        self.by_rows[rows] = seconds if old is None else old + 0.1 * (seconds - old)
        if draft_seconds > 0 and tokens is not None:  # what a step with drafts really made
            made = tokens / (seconds + draft_seconds)
            rate = self.rate.setdefault(running, [made, 0])
            rate[0] += 0.1 * (made - rate[0])
            rate[1] += 1
        self.steps += 1

    def summary(self) -> dict[str, Any]:
        first, slope = self._line()
        return {"drafted": self.drafted, "accepted": self.accepted, "draft_calls": self.calls,
                "acceptance_by_confidence": [round(a / b, 3) for a, b in zip(self.taken, self.tried)],
                "draft_call_ms_by_rows": {str(n): round(1000 * t, 1) for n, t in sorted(self.call.items())},
                "step_ms": round(1000 * first, 1), "row_ms": round(1000 * slope, 2),
                "tok_s_with_drafts_by_requests": {str(n): round(r[0], 1) for n, r in sorted(self.rate.items())}}


def _common(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class Drafter:
    """The draft model and its slots. `engine` is a `GgmlEngine` on the draft GGUF (or a stand-in with `_decode_conf`,
    `eog`, `L`, `memory` and `seqs`)."""

    def __init__(self, engine: Any, max_running: int = 2):
        self.e = engine
        # DM1 on a 4-vCPU Xeon (LAB.md): 1.68x for one request, about 1.3x for two, nothing from three on (the draft
        # calls then cost what the drafts bring); beyond this many running requests the draft model rests
        self.max_running = max_running
        self.stop = DraftStop()
        self.held: dict[int, list[int]] = {slot: [] for slot in range(engine.seqs)}  # slot -> tokens its KV holds
        self.slot_of: dict[int, int] = {}  # the target's sequence -> slot
        self._pending: dict[int, tuple[int, list[float]]] = {}  # sequence -> (context length, confidences) of its drafts
        self.step_seconds = 0.0  # seconds the draft model took for the step being built

    def _assign(self, key: int, context: Sequence[int]) -> int | None:
        free = [slot for slot in self.held if slot not in self.slot_of.values()]
        if not free:
            return None
        slot = max(free, key=lambda s: _common(self.held[s], context))
        self.slot_of[key] = slot
        return slot

    def release(self, key: int) -> None:
        """The request ended; its slot's cells stay for a later request with the same prefix."""
        self.slot_of.pop(key, None)
        self._pending.pop(key, None)

    def _call(self, rows: list[tuple[int, int, int, bool]]) -> list[tuple[int, float]]:
        import time

        started = time.perf_counter()
        out = self.e._decode_conf(rows)
        seconds = time.perf_counter() - started
        self.step_seconds += seconds
        if len(out) == len(rows):  # a call of draft rows only (a catch-up carries more tokens than drafts)
            self.stop.observe_call(seconds, len(rows))
        return out

    def draft(self, wants: dict[int, tuple[Sequence[int], int]], running: int) -> dict[int, list[int]]:
        """Drafts for the requests in `wants` (sequence -> (its tokens so far, how many drafts it has room for)), as
        many as the rule gives; {} where it gives none. `running` is the number of requests in the step."""
        self.step_seconds = 0.0
        if running > self.max_running:
            return {}
        keys = [k for k in wants if k in self.slot_of or self._assign(k, wants[k][0]) is not None]
        if not keys or not self.stop.start(running, len(keys)):
            return {}
        rows: list[tuple[int, int, int, bool]] = []
        for key in keys:
            context, slot = list(wants[key][0]), self.slot_of[key]
            held = self.held[slot]
            shared = min(_common(held, context), len(context) - 1)  # the last token is always computed, for logits
            if len(held) > shared:
                if shared == 0 or not self.e.L.llama_memory_seq_rm(self.e.memory, slot, shared, -1):
                    self.e.L.llama_memory_seq_rm(self.e.memory, slot, -1, -1)
                    shared = 0
                del held[shared:]
            feed = context[shared:]
            while len(feed) > CHUNK:  # a long catch-up (a new prompt) goes first on its own, without logits
                part, feed = feed[:CHUNK], feed[CHUNK:]
                self._call([(t, len(held) + j, slot, False) for j, t in enumerate(part)])
                held.extend(part)
            rows += [(t, len(held) + j, slot, j == len(feed) - 1) for j, t in enumerate(feed)]
            held.extend(feed)
        picks = self._call(rows)
        state = {key: {"next": picks[i], "drafts": [], "conf": [], "q": 1.0} for i, key in enumerate(keys)}
        expected, spent, going = 0.0, self.step_seconds, list(keys)
        while going:
            again = []
            for key in going:
                s = state[key]
                token, confidence = s["next"]
                s["drafts"].append(token)
                s["conf"].append(confidence)
                s["q"] *= self.stop.p(confidence)
                expected += s["q"]
                if len(s["drafts"]) < wants[key][1] and not self.e.eog(token):
                    again.append(key)
            going = [key for key in again
                     if self.stop.more(state[key]["q"], expected, spent, running, len(again))]
            if not going:
                break
            rows = []
            for key in going:
                slot = self.slot_of[key]
                rows.append((state[key]["drafts"][-1], len(self.held[slot]), slot, True))
                self.held[slot].append(state[key]["drafts"][-1])
            picks = self._call(rows)
            spent = self.step_seconds + self.stop.c_row * sum(len(state[k]["drafts"]) for k in keys)
            for key, pick in zip(going, picks):
                state[key]["next"] = pick
        out = {}
        for key in keys:
            self._pending[key] = (len(wants[key][0]), state[key]["conf"])
            out[key] = state[key]["drafts"]
        return out

    def settle(self, key: int, used: int, accepted: int) -> None:
        """The target verified the first `used` of the request's drafts and accepted `accepted` of them."""
        pending = self._pending.pop(key, None)
        slot = self.slot_of.get(key)
        if pending is None or slot is None:
            return
        before, confidences = pending
        self.stop.learn(confidences[:used], accepted)
        held = self.held[slot]
        keep = before + min(accepted, max(0, len(held) - before))  # the cells that still match the text
        if len(held) > keep:
            self.e.L.llama_memory_seq_rm(self.e.memory, slot, keep, -1)
            del held[keep:]

    def end_step(self, seconds: float, running: int, draft_rows: int, tokens: int) -> None:
        """The step the drafts were for took `seconds` and made `tokens` tokens."""
        self.stop.observe_step(seconds, running, draft_rows, tokens, self.step_seconds if draft_rows else 0.0)
        self.step_seconds = 0.0

    def summary(self) -> dict[str, Any]:
        return {"slots": len(self.held), "max_running": self.max_running, **self.stop.summary()}

    def close(self) -> None:
        close = getattr(self.e, "close", None)
        if close is not None:
            close()


__all__ = ["DraftStop", "Drafter"]
