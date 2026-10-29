"""LD1: a learned dispatch policy, opt-in and bound to the evidence that qualified it.

The policy is the one `tools/rl4_agent.py` trains (BACKLOG RL4, RL4-S, RL6): per request a softmax
over GPUs from that request's output limit and prompt length and each GPU's admitted work, per
group start a logistic choice of shortest-job-first, and in a v2 policy a softmax over group
widths. The server applies it greedily. It changes only where and in which order requests run,
never what a request answers: each request still runs alone on one worker at its own capacity.

It activates only with evidence that names this policy's digest, a passed verdict and the binding
this server runs: model, revision, GPU name and number of GPUs. Missing, corrupt, foreign or
mismatched evidence leaves the reference dispatch serving, with the reason in `/health`. A failure
while deciding falls back to the reference for that decision. After `MAX_ERRORS` of them the
policy is withdrawn for the life of the process.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Any

LONG_TOKENS = 128          # an output limit at least this long is `long` (RL3)
LONG_PROMPT_CHARS = 1000   # a request whose last message is longer counts as a long prompt
SJF_AGE_S = 10.0           # a request waiting longer starts by arrival under shortest-job-first
PLACEMENT_FEATURES = 10
ORDER_FEATURES = 4
WIDTH_CHOICES = (1, 2, 4, 8)
MAX_ERRORS = 3
POLICY_SCHEMAS = ("ironmule.rl4.policy.v1", "ironmule.rl4.policy.v2")
EVIDENCE_SCHEMA = "ironmule.dispatch_evidence.v1"


@dataclass(frozen=True)
class View:
    """What the policy reads of one request: its output limit and whether its prompt is long."""
    max_tokens: int
    long_prompt: bool

    @classmethod
    def of(cls, request: Any) -> "View":
        messages = getattr(request, "messages", ()) or ()
        last = messages[-1][1] if messages else ""
        return cls(int(request.max_tokens), len(last) > LONG_PROMPT_CHARS)


def _is_long(view: View) -> bool:
    return view.max_tokens >= LONG_TOKENS


def placement_features(request: View, cards: list[list[View]], card: int) -> list[float]:
    """The same features as `tools/rl2_headroom.placement_features`, for one candidate GPU."""
    outstanding = cards[card]
    load = len(outstanding) / 8.0
    longs = sum(_is_long(r) for r in outstanding) / 8.0
    shorts = load - longs
    long_request = 1.0 if _is_long(request) else 0.0
    last = 1.0 if card == len(cards) - 1 else 0.0
    idle = 1.0 if not outstanding else 0.0
    work = sum(r.max_tokens for r in outstanding) / 1024.0
    return [load, long_request * longs, (1 - long_request) * longs, long_request * shorts,
            (1 - long_request) * shorts, long_request * last, (1 - long_request) * last,
            float(request.long_prompt) * load, idle, work]


def order_features(waiting: list[View]) -> list[float]:
    longs = sum(_is_long(r) for r in waiting)
    shorts = len(waiting) - longs
    return [1.0, len(waiting) / 8.0, float(longs > 0 and shorts > 0), longs / max(1, len(waiting))]


def _softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    return [e / sum(exps) for e in exps]


def _digest(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Policy:
    placement: tuple[float, ...]
    order: tuple[float, ...]
    width: tuple[float, ...] | None
    sha256: str

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Policy":
        body = {k: v for k, v in data.items() if k != "sha256"}
        if _digest(body) != data.get("sha256"):
            raise ValueError("the policy's digest does not match its weights")
        if body.get("schema") not in POLICY_SCHEMAS:
            raise ValueError("unknown policy schema")
        placement, order = [float(x) for x in body["placement"]], [float(x) for x in body["order"]]
        width = body.get("width") if body["schema"] == "ironmule.rl4.policy.v2" else None
        if (len(placement) != PLACEMENT_FEATURES or len(order) != ORDER_FEATURES
                or (width is not None and (body.get("widths") != list(WIDTH_CHOICES)
                                           or len(width) != len(WIDTH_CHOICES) * ORDER_FEATURES))):
            raise ValueError("the policy's shape does not match this server's features")
        return cls(tuple(placement), tuple(order), None if width is None else tuple(float(x) for x in width),
                   data["sha256"])

    def best_card(self, request: View, cards: list[list[View]]) -> int:
        scores = [sum(w * x for w, x in zip(self.placement, placement_features(request, cards, c)))
                  for c in range(len(cards))]
        p = _softmax(scores)
        return max(range(len(p)), key=lambda i: (p[i], -i))

    def uses_sjf(self, waiting: list[View]) -> bool:
        z = sum(w * x for w, x in zip(self.order, order_features(waiting)))
        return 1 / (1 + math.exp(-max(-30.0, min(30.0, z)))) > 0.5

    def best_width(self, waiting: list[View]) -> int | None:
        if self.width is None:
            return None
        f = order_features(waiting)
        p = _softmax([sum(w * x for w, x in zip(self.width[c * ORDER_FEATURES:(c + 1) * ORDER_FEATURES], f))
                      for c in range(len(WIDTH_CHOICES))])
        return WIDTH_CHOICES[max(range(len(p)), key=lambda i: (p[i], -i))]


class LearnedDispatch:
    """The active policy and its state; `None` decisions mean "the reference decides"."""

    def __init__(self, policy: Policy | None, evidence: dict[str, Any] | None, reason: str | None):
        self.policy, self.evidence, self.reason = policy, evidence, reason
        self.errors = 0
        self.decisions = {"placement": 0, "order": 0, "width": 0}
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.policy is not None

    @classmethod
    def load(cls, policy_path: str | Path, evidence_path: str | Path,
             binding: dict[str, Any]) -> "LearnedDispatch":
        """Fail closed: anything that does not prove this policy for this binding keeps the reference."""
        try:
            policy = Policy.from_json(json.loads(Path(policy_path).read_text()))
            evidence = json.loads(Path(evidence_path).read_text())
            body = {k: v for k, v in evidence.items() if k != "sha256"}
            if _digest(body) != evidence.get("sha256"):
                raise ValueError("the evidence's digest does not match its content")
            if body.get("schema") != EVIDENCE_SCHEMA:
                raise ValueError("unknown evidence schema")
            if body.get("policy_sha256") != policy.sha256:
                raise ValueError("the evidence qualifies a different policy")
            if body.get("verdict") != "pass":
                raise ValueError("the evidence does not report a passed comparison")
            wanted = body.get("binding") or {}
            mismatched = sorted(key for key in ("model_id", "revision", "gpu", "devices")
                                if wanted.get(key) != binding.get(key))
            if mismatched:
                raise ValueError("the evidence was measured on another setup: " + ", ".join(mismatched))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return cls(None, None, f"learned dispatch is off: {exc}; the reference dispatch serves")
        return cls(policy, body, None)

    def _decide(self, kind: str, decide: Any) -> Any:
        if self.policy is None:
            return None
        try:
            value = decide(self.policy)
        except Exception as exc:  # noqa: BLE001 - a policy fault never stops serving
            with self._lock:
                self.errors += 1
                if self.errors >= MAX_ERRORS:
                    self.policy, self.reason = None, f"withdrawn after {self.errors} errors (last: {exc!r})"
            return None
        with self._lock:
            self.decisions[kind] += 1
        return value

    def card(self, request: Any, cards: list[list[Any]]) -> int | None:
        """The GPU for `request` given each GPU's admitted requests, or None for the reference."""
        return self._decide("placement", lambda p: p.best_card(View.of(request),
                                                               [[View.of(r) for r in card] for card in cards]))

    def sjf(self, waiting: list[Any]) -> bool | None:
        """Whether this group starts shortest-job-first, or None for the reference (arrival order)."""
        return self._decide("order", lambda p: p.uses_sjf([View.of(r) for r in waiting]))

    def width(self, waiting: list[Any]) -> int | None:
        """This group's width, or None for the reference (the serving policy's)."""
        if self.policy is None or self.policy.width is None:
            return None
        return self._decide("width", lambda p: p.best_width([View.of(r) for r in waiting]))

    def status(self) -> dict[str, Any]:
        evidence = self.evidence or {}
        return {"active": self.active, "reason": self.reason,
                "policy_sha256": None if self.policy is None else self.policy.sha256,
                "experiment": evidence.get("experiment"), "domain": evidence.get("domain"),
                "gain": evidence.get("gain"), "gain_ci95": evidence.get("gain_ci95"),
                "decisions": dict(self.decisions), "errors": self.errors}


def evidence_for(policy_sha256: str, experiment: str, binding: dict[str, Any], domain: list[str],
                 learned: dict[str, Any], noise: float) -> dict[str, Any]:
    """The evidence record `LearnedDispatch.load` accepts, from a passed comparison's analysis."""
    passed = learned["gain_ci95"][0] > noise
    body = {"schema": EVIDENCE_SCHEMA, "experiment": experiment, "policy_sha256": policy_sha256,
            "verdict": "pass" if passed else "fail", "binding": binding, "domain": list(domain),
            "rival": learned["rival"], "gain": learned["gain"], "gain_ci95": list(learned["gain_ci95"]),
            "noise": noise}
    return {**body, "sha256": _digest(body)}
