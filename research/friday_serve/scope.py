"""Stage B of the scope check: what the request *is*, never what it claims to be.

The sealed runtimes derive scope from the actual tensors and tokens
(``friday_head_skip_runtime/policy.py:562-567``,
``friday_runtime_n10/executor.py:47-79``) rather than from a label the caller
passes in. That property is the reason an unattended dispatch can be safe, so it
is carried over here unchanged: :func:`observe` reads the encoded prompt and the
loaded model, and a request it cannot fully characterise returns ``None``, which
means baseline.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

#: The calibration workload is greedy, batch 1, fixed horizon. A request outside
#: that shape was never verified for token identity, so it does not get a knob.
CALIBRATED_TEMPERATURE = 0.0
CALIBRATED_BATCH = 1


@functools.lru_cache(maxsize=1)
def live_machine_sha256() -> str:
    """This host's stable identity digest. Constant for the process; the
    ``sysctl`` calls behind it must not run on every request."""
    from friday_runtime_core.provenance import machine_sha256

    return machine_sha256()


@functools.lru_cache(maxsize=1)
def live_ironmule_head() -> str | None:
    """The bound engine checkout's commit, or ``None`` when unreadable.

    Unreadable means the profile's engine binding cannot be checked, which is
    the state every profile was in before the field existed; it is not treated
    as a mismatch.
    """

    try:
        from .ironmule_backend import ironmule_head

        return ironmule_head()
    except Exception:
        return None


@dataclass(frozen=True)
class RequestScope:
    model_id: str
    model_revision: str
    prompt_sha256: str
    prompt_tokens: int
    output_tokens: int
    temperature: float
    batch: int


def observe(
    *,
    model_id: Any,
    model_revision: Any,
    token_ids: Any,
    output_tokens: Any,
    temperature: Any = CALIBRATED_TEMPERATURE,
    batch: Any = CALIBRATED_BATCH,
) -> RequestScope | None:
    """Derive the scope of one request, or ``None`` when it cannot be derived."""

    if (
        not isinstance(model_id, str)
        or not model_id
        or not isinstance(model_revision, str)
        or not model_revision
        or isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens <= 0
        or isinstance(batch, bool)
        or not isinstance(batch, int)
        or batch <= 0
        or isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or isinstance(token_ids, (str, bytes))
        or not isinstance(token_ids, Sequence)
        or not token_ids
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in token_ids
        )
    ):
        return None
    digest = hashlib.sha256(
        b",".join(str(int(value)).encode("ascii") for value in token_ids)
    ).hexdigest()
    return RequestScope(
        model_id=model_id,
        model_revision=model_revision,
        prompt_sha256=digest,
        prompt_tokens=len(token_ids),
        output_tokens=output_tokens,
        temperature=float(temperature),
        batch=int(batch),
    )


def in_calibrated_scope(scope: RequestScope | None, profile) -> tuple[bool, str]:
    """Is this request inside what the device profile actually verified?

    Prompt content is deliberately *not* part of the answer. The knobs were
    verified as token-identical, which is a property of the computation and not
    of the prompt; binding the profile to one prompt hash would reproduce the
    over-narrow scope that made the sealed runtimes unusable.
    """

    if scope is None:
        return False, "scope_underivable"
    if scope.model_id != profile.model_id:
        return False, "model_mismatch"
    if scope.model_revision != profile.model_revision:
        return False, "model_revision_mismatch"
    if scope.batch != CALIBRATED_BATCH:
        return False, "batch_out_of_scope"
    if scope.temperature != CALIBRATED_TEMPERATURE:
        return False, "sampling_out_of_scope"
    # A profile carries the stable host identity it was measured on. If it was
    # copied from another Mac, its knobs were never verified here.
    # ponytail: only the stable subset (CPU/model/memory/arch, not the macOS
    # version) is compared, so a routine OS update does not invalidate it.
    expected = getattr(profile, "machine_sha256", None)
    if expected is not None and live_machine_sha256() != expected:
        return False, "machine_mismatch"
    # A knob was verified against one engine. A different IronMule commit is a
    # different computation, so the verdicts do not carry over to it.
    engine = getattr(profile, "ironmule_head", None)
    if engine is not None and live_ironmule_head() != engine:
        return False, "engine_mismatch"
    return True, "device_profile_verified"


__all__ = [
    "CALIBRATED_BATCH",
    "CALIBRATED_TEMPERATURE",
    "RequestScope",
    "in_calibrated_scope",
    "live_machine_sha256",
    "observe",
]
