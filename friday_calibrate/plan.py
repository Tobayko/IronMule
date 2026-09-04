"""What a calibration run does, written down before it runs.

The run is orchestration, not new measurement science: every step below already
exists somewhere in this repository and is reused rather than rewritten. The
plan is a separate module from the runner so it can be printed, reviewed and
budget-checked without touching the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .profile import CALIBRATED_KNOBS, KNOB_PHASE

#: The bound IronMule checkout. F1 measured through this engine, so calibration
#: measures through it too; a different commit is a different engine and the
#: run refuses to start.
EXPECTED_IRONMULE_HEAD = "03e884cb28a05d090d20844460fc3afc8e738a91"

MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"

#: The sealed workload. Two of the three confirmed gains were measured on it and
#: F1's 0.860057 is stated for it; calibrating on anything else would produce a
#: profile that cannot be compared against the evidence the project already has.
PROMPT_TOKENS = 897
OUTPUT_TOKENS = 32

#: Six pairs is what F1 used and what the A/A noise on this device supports.
DEFAULT_PAIRS = 6

#: How each calibrated knob maps onto the engine that actually implements it.
#: ``prefill_step_size`` has no entry: see ``PREFILL_STEP_SIZE_NOTE``.
KNOB_TO_ENGINE = {
    "head_skip": {"head_skip_prefill": True},
    "fixed_compiled": {"compiled_fixed_cache": True},
    "bundled_readback": {"readback_every": 8},
}

PREFILL_STEP_SIZE_NOTE = (
    "The engine that produced F1's number prefills the whole prompt in one "
    "forward (ironmule/runtime.py:_prefill, prefix_cache is None), which is "
    "already the fastest point of the measured step-size curve. There is no "
    "smaller-step baseline left to improve on, so this knob is calibrated as "
    "not_applicable unless a chunked plan is explicitly in use."
)


@dataclass(frozen=True)
class Step:
    name: str
    purpose: str
    reuses: str
    gpu_seconds_estimate: float


def steps(pairs: int = DEFAULT_PAIRS) -> tuple[Step, ...]:
    """The ordered calibration steps and what each one reuses."""

    knob_steps = tuple(
        Step(
            name=f"knob:{knob}",
            purpose=(
                f"verify {knob} ({KNOB_PHASE[knob]} phase) is token-identical and "
                "faster on this device"
            ),
            reuses="friday_optimizer.integration.evaluate_integration",
            gpu_seconds_estimate=pairs * 2 * 2.3,
        )
        for knob in CALIBRATED_KNOBS
        if knob in KNOB_TO_ENGINE
    )
    return (
        Step(
            name="aa_noise",
            purpose="this device's own minimum detectable effect, baseline against itself",
            reuses="friday_optimizer.integration.evaluate_integration",
            gpu_seconds_estimate=pairs * 2 * 2.3,
        ),
        *knob_steps,
        Step(
            name="width_curve",
            purpose="seed the speculative bandit with this device's draft-width curve",
            reuses="experiments/decode_width/measure_prefill.py",
            gpu_seconds_estimate=16.0,
        ),
        Step(
            name="roofline",
            purpose="prefill compute utilisation and decode bandwidth of this device",
            reuses="experiments/roofline/phase_roofline.py",
            gpu_seconds_estimate=20.0,
        ),
    )


def as_dict(
    pairs: int = DEFAULT_PAIRS,
    *,
    model_id: str = MODEL_ID,
    prompt_tokens: int = PROMPT_TOKENS,
    output_tokens: int = OUTPUT_TOKENS,
) -> dict[str, Any]:
    """The run, written down. Defaults are the sealed 4B workload, unchanged.

    Another model is calibrated the same way against its own tokenisation of the
    same prompt: ``prompt_tokens`` is the exact count only for the sealed
    workload, and ``0`` elsewhere, where the observed count is recorded instead
    of asserted (``runner.build_runner``).
    """

    ordered = steps(pairs)
    return {
        "model_id": model_id,
        "expected_ironmule_head": EXPECTED_IRONMULE_HEAD,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "pairs": pairs,
        "steps": [
            {
                "name": step.name,
                "purpose": step.purpose,
                "reuses": step.reuses,
                "gpu_seconds_estimate": step.gpu_seconds_estimate,
            }
            for step in ordered
        ],
        "gpu_seconds_estimate": sum(step.gpu_seconds_estimate for step in ordered),
        "not_calibrated": {
            "prefill_step_size": PREFILL_STEP_SIZE_NOTE,
        },
        "formal_claim": False,
    }


__all__ = [
    "DEFAULT_PAIRS",
    "EXPECTED_IRONMULE_HEAD",
    "KNOB_TO_ENGINE",
    "MODEL_ID",
    "OUTPUT_TOKENS",
    "PREFILL_STEP_SIZE_NOTE",
    "PROMPT_TOKENS",
    "Step",
    "as_dict",
    "steps",
]
