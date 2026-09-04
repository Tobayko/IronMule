"""Adaptive draft width: a bandit whose job is to switch speculation *off*.

The measured spread on real prompts runs from `0.974` (a loss) to `1.096` (a
gain) and the best draft width moves between `2`, `3` and `4`
(``experiments/prompt_lookup/``). A single fixed width is therefore wrong on
some workload no matter which one is picked, and the first naive adaptive
attempt lost to fixed (`adaptive.json`: `1.148` against `1.153`, and `0.859`
against `1.013` on 1B/code) because it tried to tune the width rather than to
abandon it.

So the objective here is asymmetric on purpose: **the gain is in switching off,
not in fine-tuning.** Turning speculation off reliably on unfavourable workloads
converts `-2.6 %` into `0 %` while keeping `+9.6 %` where it exists, and that is
worth more than picking `3` over `4` where both win.

Why this is safe to decide without asking: speculative decoding verifies every
drafted token against the model itself, so its output is identical *by
construction*, not by measurement. Every one of the 14 recorded runs reports
``all_identical_to_greedy=true``, and it could not report otherwise. This is the
one lever in the project that needs no per-device promotion gate — which is
exactly why it is the one the bandit is allowed to touch.

What is **not** learned: promotion decisions, thresholds, tolerances, kernels,
or anything that crosses devices. The action space is five draft widths.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: Draft widths the bandit may choose. ``0`` is "no speculation" and is a first
#: class action, not a fallback: on `journal` and `tests` prompts it wins.
#:
#: The plan named ``{off, 2, 3, 4, 8}``. That set conflates two different knobs:
#: ``tools/measure_prompt_lookup.py:56`` sweeps ``DRAFT_LENGTHS = (0, 1, 2, 3, 4)``
#: and the ``_n8`` in the run filenames is the *n-gram* length, not a draft
#: width. No draft width above `4` has ever been measured here, so the action
#: space is the one the 14 runs actually cover.
ACTIONS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: Prompt-lookup speculation pays off when the answer repeats prompt n-grams.
#: The repetition of the prompt itself is the cheapest pre-generation proxy for
#: that, and unlike acceptance rate it is available *before* the first token.
#:
#: Unigram, not 3-gram: on the three recorded prompt families the 3-gram rate is
#: `0.0` for all of them (they are 22-94 words long) and separates nothing,
#: while the unigram rate orders them `prose 0.09 < agent 0.28 < code_edit 0.43`
#: — the same order as their measured speedups.
#:
#: The two edges are a *prior* read off those three prompts, not a validated
#: threshold. If they are wrong the bandit degrades to a single-class bandit,
#: which is still bounded by the same gate; it does not become unsafe.
NGRAM = 1
CLASS_EDGES = (0.15, 0.35)
CLASS_NAMES = ("sparse", "mixed", "repetitive")

#: Half-width of the speedup range mapped onto the posterior. A request `SPAN`
#: faster than greedy scores `1.0`, one `SPAN` slower scores `0.0`, and greedy
#: itself lands exactly on `0.5`.
REWARD_SPAN = 0.5

DEFAULT_PROPENSITY_DRAWS = 512


class BanditError(ValueError):
    """A bandit input is outside the closed action space or malformed."""


def repetition_rate(token_ids: Sequence[int], *, n: int = NGRAM) -> float:
    """Share of prompt n-grams that occur more than once. Cheap and deterministic."""

    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise BanditError("token_ids must be a sequence")
    if len(token_ids) <= n:
        return 0.0
    counts: dict[tuple[int, ...], int] = {}
    for index in range(len(token_ids) - n + 1):
        key = tuple(int(value) for value in token_ids[index : index + n])
        counts[key] = counts.get(key, 0) + 1
    repeated = sum(count for count in counts.values() if count > 1)
    total = sum(counts.values())
    return repeated / total if total else 0.0


def workload_class(token_ids: Sequence[int]) -> str:
    """Bucket a prompt before generating anything. Three classes, fixed edges."""

    rate = repetition_rate(token_ids)
    if rate < CLASS_EDGES[0]:
        return CLASS_NAMES[0]
    if rate < CLASS_EDGES[1]:
        return CLASS_NAMES[1]
    return CLASS_NAMES[2]


@dataclass
class Posterior:
    """Beta posterior over "this action beats not speculating, on this class"."""

    alpha: float = 1.0
    beta: float = 1.0

    def update(self, reward: float) -> None:
        """Fractional Beta update. ``reward`` is in ``[0, 1]``; `0.5` is neutral.

        A hard success/failure threshold cannot express "off was exactly right":
        greedy decoding is neither a win nor a loss, it is the reference. A
        fractional update puts it at `0.5` and lets every other width earn or
        lose ground against it.
        """

        value = min(max(float(reward), 0.0), 1.0)
        self.alpha += value
        self.beta += 1.0 - value

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def as_dict(self) -> dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta, "mean": self.mean}


@dataclass
class SpeculationBandit:
    """Thompson sampling over five draft widths, one posterior per class.

    The reward is observable per request and needs no counterfactual: a draft
    width succeeds when the realised decode rate beats this class's reference
    rate, and the reference is what action ``0`` — plain greedy decoding —
    actually achieves. Action ``0`` therefore scores around `0.5` by
    construction, and any width has to earn its place against it.
    """

    prior_strength: float = 2.0
    reference: dict[str, float] = field(default_factory=dict)
    posteriors: dict[tuple[str, int], Posterior] = field(default_factory=dict)
    reference_counts: dict[str, int] = field(default_factory=dict)

    # -- seeding --------------------------------------------------------------
    @classmethod
    def seeded(cls, width_curve: Mapping[int, float] | None, **kwargs) -> "SpeculationBandit":
        """Start from the calibration width curve instead of from ignorance.

        The curve is one device's measured decode rate per width. It is a prior,
        not evidence: it enters as ``prior_strength`` pseudo-observations, so a
        handful of real requests can and does overrule it.
        """

        bandit = cls(**kwargs)
        if not width_curve:
            return bandit
        baseline = width_curve.get(0)
        if not baseline or baseline <= 0:
            return bandit
        for name in CLASS_NAMES:
            bandit.reference[name] = float(baseline)
            for action in ACTIONS:
                rate = width_curve.get(action)
                if rate is None or rate <= 0:
                    continue
                share = min(max(0.5 + (rate / baseline - 1.0) / (2.0 * REWARD_SPAN), 0.01), 0.99)
                posterior = bandit._posterior(name, action)
                posterior.alpha = 1.0 + bandit.prior_strength * share
                posterior.beta = 1.0 + bandit.prior_strength * (1.0 - share)
        return bandit

    def _posterior(self, name: str, action: int) -> Posterior:
        if action not in ACTIONS:
            raise BanditError(f"action outside the closed space: {action!r}")
        key = (name, action)
        posterior = self.posteriors.get(key)
        if posterior is None:
            posterior = self.posteriors[key] = Posterior()
        return posterior

    # -- selection ------------------------------------------------------------
    def _draw(self, name: str, rng: random.Random) -> int:
        best_action, best_value = ACTIONS[0], -1.0
        for action in ACTIONS:
            posterior = self._posterior(name, action)
            value = rng.betavariate(posterior.alpha, posterior.beta)
            if value > best_value:
                best_action, best_value = action, value
        return best_action

    def select(self, name: str, *, seed: int) -> tuple[int, float]:
        """Return ``(draft_width, propensity)``.

        Thompson sampling has no closed-form propensity, so it is estimated by
        redrawing from the same posteriors. Estimated is enough for the
        importance-sampling estimators as long as it is the *actual* sampling
        distribution, which it is — and unlike a hand-set epsilon it cannot
        drift away from what the policy really does.
        """

        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise BanditError("select requires a non-negative integer seed")
        action = self._draw(name, random.Random(seed))
        return action, self.propensity(name, action, seed=seed)

    def propensity(
        self, name: str, action: int, *, seed: int, draws: int = DEFAULT_PROPENSITY_DRAWS
    ) -> float:
        if action not in ACTIONS:
            raise BanditError(f"action outside the closed space: {action!r}")
        rng = random.Random(seed ^ 0x5EED)
        hits = sum(1 for _ in range(draws) if self._draw(name, rng) == action)
        # Never report zero: an unreachable action would make an IPS weight
        # infinite, which is a silent way to fabricate confidence.
        return max(hits / draws, 1.0 / (draws + 1))

    def distribution(self, name: str, *, seed: int = 0, draws: int = DEFAULT_PROPENSITY_DRAWS):
        rng = random.Random(seed ^ 0x5EED)
        counts = {action: 0 for action in ACTIONS}
        for _ in range(draws):
            counts[self._draw(name, rng)] += 1
        return {action: count / draws for action, count in counts.items()}

    # -- learning -------------------------------------------------------------
    def observe(self, name: str, action: int, decode_tps: float) -> float:
        """Record one realised request. Returns the reward it produced."""

        if isinstance(decode_tps, bool) or not isinstance(decode_tps, (int, float)):
            raise BanditError("decode_tps must be a number")
        rate = float(decode_tps)
        if not math.isfinite(rate) or rate <= 0.0:
            raise BanditError("decode_tps must be finite and positive")
        if action == 0:
            # Greedy decoding *is* the reference; a running mean tracks drift
            # (thermal state, other load) without a separate calibration.
            seen = self.reference_counts.get(name, 0)
            current = self.reference.get(name, rate)
            self.reference[name] = (current * seen + rate) / (seen + 1)
            self.reference_counts[name] = seen + 1
        reference = self.reference.get(name)
        if reference is None or reference <= 0.0:
            # Nothing to compare against yet: neutral, so an unmeasured class
            # neither rewards nor punishes the width that happened to run first.
            reward = 0.5
        else:
            reward = 0.5 + (rate / reference - 1.0) / (2.0 * REWARD_SPAN)
        reward = min(max(reward, 0.0), 1.0)
        self._posterior(name, action).update(reward)
        return reward

    def as_dict(self) -> dict[str, Any]:
        return {
            "actions": list(ACTIONS),
            "classes": list(CLASS_NAMES),
            "reference": dict(self.reference),
            "posteriors": {
                f"{name}:{action}": posterior.as_dict()
                for (name, action), posterior in sorted(self.posteriors.items())
            },
        }


def best_fixed_action(rows: Iterable[Mapping[str, Any]]) -> tuple[int, float]:
    """The strongest single width across all classes — the bar the bandit must clear."""

    totals: dict[int, list[float]] = {}
    for row in rows:
        action = int(row["action"])
        totals.setdefault(action, []).append(float(row["speedup"]))
    if not totals:
        raise BanditError("no rows to compare against")
    scored = {
        action: sum(values) / len(values) for action, values in totals.items()
    }
    action = max(scored, key=lambda key: scored[key])
    return action, scored[action]


__all__ = [
    "ACTIONS",
    "CLASS_EDGES",
    "CLASS_NAMES",
    "REWARD_SPAN",
    "BanditError",
    "Posterior",
    "SpeculationBandit",
    "best_fixed_action",
    "repetition_rate",
    "workload_class",
]
