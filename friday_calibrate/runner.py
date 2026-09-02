"""The gated calibration run: one process, one model, one profile at the end.

Structure follows ``experiments/f1_integration/measure_f1.py`` deliberately —
paired arms, alternating AB/BA order, token identity checked per pair, a single
mismatch ends the run. What differs is the purpose: F1 asked whether two knobs
together beat a threshold *here*; calibration asks, knob by knob, whether each
one is admissible *on this device at all*.

The measurement core (:func:`paired_arms`, :func:`verdict_for`) takes a plain
``run(knobs) -> Sample`` callable, so the whole decision logic runs offline
against a fake engine. Only :func:`calibrate` touches hardware.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from friday_optimizer.evaluator import MetricSample
from friday_optimizer.integration import evaluate_integration

from .plan import (
    DEFAULT_PAIRS,
    EXPECTED_IRONMULE_HEAD,
    KNOB_TO_ENGINE,
    MODEL_ID,
    OUTPUT_TOKENS,
    PREFILL_STEP_SIZE_NOTE,
    PROMPT_TOKENS,
)
from .profile import CALIBRATED_KNOBS, DeviceProfile, KnobVerdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IRONMULE = PROJECT_ROOT / ".worktrees" / "friday-optimizer-ironmule"

#: The action space the speculative bandit will choose from. The curve measured
#: here is its prior, so the two must be the same set.
DRAFT_WIDTHS = (0, 2, 3, 4, 8)


class CalibrationError(RuntimeError):
    """The calibration run cannot produce an honest profile."""


@dataclass(frozen=True)
class Sample:
    """One request under one knob setting."""

    ttft_seconds: float
    decode_tps: float
    tokens: int
    token_sha256: str
    #: The tokens themselves, so an identity break can be *shown* rather than
    #: only reported. H1.0 hit one at 4B/128/width 2 and threw both sequences
    #: away with the process; the file held the message and no evidence. Empty
    #: by default, and never read by any decision here.
    token_ids: tuple[int, ...] = ()


def _metric(sample: Sample, *, arm: str, index: int, order: str, workload: str) -> MetricSample:
    # session_id and pair_id are shared by both arms of a pair: the evaluator
    # pairs on (pair_id, session_id), so an arm-specific id silently unpairs
    # every sample and the run comes back "rejected" for the wrong reason.
    return MetricSample(
        session_id=f"{workload}-s{index}",
        pair_id=f"p{index}",
        arm=arm,
        order=order,
        fingerprint=workload,
        workload=workload,
        ttft_seconds=sample.ttft_seconds,
        decode_tps=sample.decode_tps,
        tokens=sample.tokens,
        status="ok",
    )


def paired_arms(
    run: Callable[[Mapping[str, Any]], Sample],
    candidate_knobs: Mapping[str, Any],
    *,
    pairs: int = DEFAULT_PAIRS,
    workload: str = "sealed-897-32",
    baseline_knobs: Mapping[str, Any] | None = None,
    on_break: Callable[[int, Sample, Sample], None] | None = None,
) -> tuple[list[MetricSample], list[MetricSample], tuple[str, ...]]:
    """Run ``pairs`` alternating AB/BA pairs and report any identity break.

    Order alternates so neither arm systematically runs on the warmer cache —
    W1 measured that warm-up, not context growth, dominates a short run, which
    makes a fixed order the single most effective way to fake a gain.

    ``baseline_knobs`` defaults to the bare engine. H1.0 needs the other case:
    the switch point is a step *from the serving default*, so its baseline is
    the combined knob set, not an engine nobody ships.

    ``on_break`` receives ``(pair_index, baseline_sample, candidate_sample)``
    when token identity breaks, so the two sequences can be recorded instead of
    dying with the process. It cannot influence the outcome: it runs after the
    break is decided and its return value is ignored.
    """

    base = dict(baseline_knobs or {})
    baseline: list[MetricSample] = []
    candidate: list[MetricSample] = []
    breaks: list[str] = []
    for index in range(pairs):
        order = "AB" if index % 2 == 0 else "BA"
        if order == "AB":
            left, right = run(base), run(candidate_knobs)
        else:
            right, left = run(candidate_knobs), run(base)
        if left.token_sha256 != right.token_sha256:
            breaks.append(f"token_identity_broken:pair_{index}")
            if on_break is not None:
                on_break(index, left, right)
            break
        baseline.append(_metric(left, arm="baseline", index=index, order=order, workload=workload))
        candidate.append(
            _metric(right, arm="candidate", index=index, order=order, workload=workload)
        )
    return baseline, candidate, tuple(breaks)


def verdict_for(
    knob: str,
    run: Callable[[Mapping[str, Any]], Sample],
    *,
    pairs: int = DEFAULT_PAIRS,
    mde: float,
) -> KnobVerdict:
    """Decide one knob on this device. Anything short of proof is not ``verified``."""

    engine_knobs = KNOB_TO_ENGINE.get(knob)
    if engine_knobs is None:
        return KnobVerdict(
            knob=knob, verdict="not_applicable", reason=PREFILL_STEP_SIZE_NOTE
        )
    baseline, candidate, breaks = paired_arms(run, engine_knobs, pairs=pairs)
    if breaks:
        return KnobVerdict(knob=knob, verdict="failed", pairs=len(baseline), reason=breaks[0])
    result = evaluate_integration(
        baseline, candidate, arm="warm", min_gain=0.0, mde=mde, min_pairs=max(1, pairs // 2)
    )
    ratio = result.ratio_median
    low, high = (result.ci or (None, None))
    identical = True
    # `qualified` means the interval clears the threshold; with min_gain 0.0 that
    # is exactly "measurably faster than baseline, beyond this device's noise".
    if result.qualified and ratio is not None and high is not None and high < 1.0:
        return KnobVerdict(
            knob=knob,
            verdict="verified",
            pairs=result.pairs,
            ratio=ratio,
            ci_low=low,
            ci_high=high,
            token_identical=identical,
        )
    return KnobVerdict(
        knob=knob,
        verdict="failed",
        pairs=result.pairs,
        ratio=ratio,
        ci_low=low,
        ci_high=high,
        token_identical=identical,
        reason=result.status if not result.reasons else ";".join(result.reasons),
    )


def noise_mde(
    run: Callable[[Mapping[str, Any]], Sample],
    *,
    pairs: int = DEFAULT_PAIRS,
    knobs: Mapping[str, Any] | None = None,
) -> tuple[float, float]:
    """A/A: the same arm against itself. Returns ``(aa_spread, mde)``.

    The spread of an A/A run is the floor under every later claim: an effect
    smaller than the noise of measuring nothing is not an effect. ``knobs``
    picks *which* arm is measured against itself; the noise of the combined
    path is not the noise of the bare engine, and neither is transferable.
    """

    arm = dict(knobs or {})
    baseline, candidate, breaks = paired_arms(
        run, arm, pairs=pairs, workload="aa", baseline_knobs=arm
    )
    if breaks:
        raise CalibrationError("A/A run broke token identity; the harness is wrong, not the knob")
    result = evaluate_integration(
        baseline, candidate, arm="warm", min_gain=0.0, mde=0.0, min_pairs=max(1, pairs // 2)
    )
    if result.ci is None or result.ratio_median is None:
        raise CalibrationError(f"A/A run produced no interval: {result.status}")
    low, high = result.ci
    spread = max(abs(1.0 - low), abs(1.0 - high))
    # The MDE is the A/A spread; a knob has to beat the device's own noise.
    return spread, spread


def draft_width_curve(
    run: Callable[[Mapping[str, Any]], Sample], *, widths: Sequence[int] = DRAFT_WIDTHS
) -> dict[int, float]:
    """Tokens per second at each draft width — the bandit's prior on this device.

    The plan named the prefill-width sweep here. That curve is about prefill
    chunking; the bandit chooses a *draft* width, so the prior has to be
    measured over the same action space the bandit will act on.
    """

    curve: dict[int, float] = {}
    for width in widths:
        knobs = {} if width == 0 else {"speculate_k": width}
        sample = run(knobs)
        curve[width] = sample.decode_tps
    return curve


# -- the hardware path --------------------------------------------------------
def _ironmule_head() -> str:
    import subprocess

    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(IRONMULE), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        raise CalibrationError("bound IronMule checkout is unreadable")
    return completed.stdout.strip()


#: The local snapshots calibration may use. Both sizes matter: speculation
#: carries far more on the smaller model (`code_edit_1b.json` 1.6946 against the
#: 4B's 1.2148), and the community target covers both.
MODELS = {
    "4b": "mlx-community/gemma-3-4b-it-4bit",
    "1b": "mlx-community/gemma-3-1b-it-4bit",
}


def build_runner(
    pairs: int = DEFAULT_PAIRS,
    *,
    model_id: str = MODEL_ID,
    output_tokens: int = OUTPUT_TOKENS,
    prompt_tokens: int = PROMPT_TOKENS,
):
    """Load the model once and return ``(run, identity, guard)``. Touches the GPU."""

    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from _bench import (  # noqa: E402
        BudgetGuard,
        check_prompt_length,
        enforce_offline,
        require_ac_power,
        resolve_local_model_snapshot,
    )

    head = _ironmule_head()
    if head != EXPECTED_IRONMULE_HEAD:
        raise CalibrationError(
            f"IronMule checkout is at {head}, expected {EXPECTED_IRONMULE_HEAD}"
        )
    require_ac_power()
    guard = BudgetGuard()
    enforce_offline()

    sys.path.insert(0, str(IRONMULE))
    import hashlib
    import json

    from ironmule import BASELINE, Engine, Knobs  # noqa: E402
    from mlx_lm import load  # noqa: E402

    snapshot = resolve_local_model_snapshot(model_id)
    model, tokenizer = load(str(snapshot.path))

    # The sealed prompt, verbatim from experiments/persistent_process/worker.py.
    filler = (
        "You are a careful engineering assistant working in a Python repository. "
        "Follow the existing style and explain your reasoning briefly. "
    ) * 40
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": filler + "\n\n" + "Why is false sharing slow?"}],
        add_generation_prompt=True,
    )
    ids = list(templated if isinstance(templated, list) else tokenizer.encode(templated))
    # The sealed prompt is 897 tokens under the 4B tokenizer. A different model
    # tokenises it differently, so the count is checked only where it is the
    # sealed workload and merely recorded otherwise.
    if prompt_tokens:
        check_prompt_length(ids, prompt_tokens)
    eos_ids = tuple(
        sorted(
            {
                int(value)
                for value in (
                    getattr(tokenizer, "eos_token_id", None),
                    *(getattr(tokenizer, "eos_token_ids", None) or ()),
                )
                if isinstance(value, int)
            }
        )
    )
    if not eos_ids:
        raise CalibrationError("tokenizer exposes no end-of-sequence id")

    debt = 0.0

    def charge(seconds: float) -> None:
        nonlocal debt
        guard.record_gpu(seconds)
        debt += seconds * (1 - 0.15) / 0.15
        while debt >= 4.0:
            guard.required_break()
            debt -= 4.0

    engines: dict[str, Any] = {}

    def run(overrides: Mapping[str, Any]) -> Sample:
        knobs = BASELINE if not overrides else Knobs(**dict(overrides))
        key = knobs.key()
        engine = engines.get(key)
        if engine is None:
            engine = engines[key] = Engine(model, tokenizer, knobs)
        at = time.perf_counter()
        out = engine.generate(ids, output_tokens, eos_ids)
        charge(time.perf_counter() - at)
        tokens = list(out["logical_tokens"])
        decode_seconds = out["decode_ns"] / 1e9
        return Sample(
            ttft_seconds=out["prefill_ns"] / 1e9,
            decode_tps=(len(tokens) - 1) / decode_seconds
            if len(tokens) > 1 and decode_seconds > 0
            else 0.0,
            tokens=len(tokens),
            token_sha256=hashlib.sha256(
                json.dumps(tokens, separators=(",", ":")).encode()
            ).hexdigest(),
            token_ids=tuple(tokens),
        )

    identity = {
        "model_id": model_id,
        "model_revision": snapshot.revision,
        "ironmule_head": head,
        "prompt_tokens": len(ids),
        "output_tokens": output_tokens,
    }
    return run, identity, guard


def calibrate(
    run: Callable[[Mapping[str, Any]], Sample],
    identity: Mapping[str, Any],
    *,
    hardware_sha256: str,
    environment_sha256: str,
    pairs: int = DEFAULT_PAIRS,
    profile_id: str | None = None,
    roofline: Mapping[str, Any] | None = None,
) -> DeviceProfile:
    """Run every calibration step against ``run`` and return the device profile."""

    spread, mde = noise_mde(run, pairs=pairs)
    verdicts = tuple(
        verdict_for(knob, run, pairs=pairs, mde=mde) for knob in CALIBRATED_KNOBS
    )
    curve = draft_width_curve(run)
    return DeviceProfile(
        profile_id=profile_id or f"device-{time.strftime('%Y%m%d-%H%M%S')}",
        model_id=str(identity.get("model_id", MODEL_ID)),
        model_revision=str(identity.get("model_revision", "")),
        hardware_sha256=hardware_sha256,
        environment_sha256=environment_sha256,
        mde=mde,
        knobs=verdicts,
        width_curve=curve,
        roofline=dict(roofline or {}),
        aa_noise=spread,
    )


__all__ = [
    "CalibrationError",
    "DRAFT_WIDTHS",
    "Sample",
    "build_runner",
    "calibrate",
    "draft_width_curve",
    "noise_mde",
    "paired_arms",
    "verdict_for",
]
