"""Closed calibration protocol shared by execution and independent evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from friday_evidence.canonical import canonical_sha256


PROMPT = "Write a short sentence about apples."
VARIANTS = ("reference", "bounded_prefetch")
LIMIT_ORDERS = ((1, 8, 32), (8, 32, 1), (32, 1, 8))
CELL_CALLS = (
    ("warmup", "reference", True),
    ("warmup", "reference", False),
    ("warmup", "bounded_prefetch", True),
    ("warmup", "bounded_prefetch", False),
    ("aa", "reference", False),
    ("aa", "reference", False),
    ("ab", "reference", False),
    ("ab", "bounded_prefetch", False),
    ("ba", "bounded_prefetch", False),
    ("ba", "reference", False),
)


@dataclass(frozen=True)
class EvaluationPolicy:
    schema: int = 1
    protocol: str = "prod3-exact-greedy-http-v1"
    min_effect_fraction: float = 0.02
    aa_noise_multiplier: float = 3.0
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = 20260907
    swap_delta_limit_bytes: int = 256 * 1024**2
    peak_memory_fraction: float = 0.6
    # Deployment confirmation is deliberately a different authority/phase.
    activation_allowed: bool = False


POLICY = EvaluationPolicy()


def specification() -> dict:
    return {
        "policy": asdict(POLICY),
        "prompt_sha256": canonical_sha256(PROMPT),
        "limit_orders": [list(order) for order in LIMIT_ORDERS],
        "cell_calls": [list(call) for call in CELL_CALLS],
        "expected_calls": len(LIMIT_ORDERS) * 3 * len(CELL_CALLS),
        "metric": "full_http_completion_wall_seconds",
        "cache": "fresh_private_per_request",
        "decoding": "greedy",
    }


def plan_id() -> str:
    return canonical_sha256(specification())


def schedule() -> list[dict]:
    result = []
    for worker_index, order in enumerate(LIMIT_ORDERS):
        for limit in order:
            for phase, variant, trace in CELL_CALLS:
                result.append({"sample_index": len(result), "worker_index": worker_index,
                               "limit": limit, "phase": phase, "variant": variant,
                               "trace_forwards": trace})
    return result
