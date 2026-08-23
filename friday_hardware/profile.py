"""What a device costs, measured, and what follows from it.

Eleven rounds of measurement produced numbers that only help if something reads
them. This is that something: a profile carrying what was measured on one device
for one model at one quantisation, plus the decisions those measurements license.

Three findings shape the design, and each of them is a refusal rather than a
feature:

  * **A profile is not portable.** The 4B regresses at widths 6 through 9 and gains
    6.16x from batching; the 1B regresses only at 48 and gains 15.03x. Applying one
    model's policy to the other would avoid widths that are fine and leave most of
    the gain unclaimed. So a profile records what it was measured on and refuses to
    answer for anything else.
  * **Measured widths only.** The cost curve is a step function with plateaus and
    cliffs, so interpolating between measured points would invent structure that was
    never observed. Only measured widths are offered.
  * **The bottleneck changes sides between devices.** On this machine the 1B is 73%
    dispatch; at a tenth of the bandwidth the same model is 72% bandwidth. A
    projection therefore needs both device parameters, and asking for only one is an
    error rather than a default.

Nothing here measures anything. The tools under `tools/` do that; this reads their
results and decides.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


class ProfileError(RuntimeError):
    """A profile was missing, malformed, or asked about something it never saw."""


@dataclass(frozen=True)
class Plan:
    """One scheduling decision, with the reasoning that produced it."""

    width: int
    requested_width: int
    steps_per_segment: int
    segments: int
    estimated_seconds: float
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "requested_width": self.requested_width,
            "steps_per_segment": self.steps_per_segment,
            "segments": self.segments,
            "estimated_seconds": round(self.estimated_seconds, 4),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HardwareProfile:
    """Measurements for one (device, model, quantisation), and nothing beyond it."""

    device: str
    model_id: str
    bits: int
    group_size: int
    layers: int
    weight_gb: float
    # Two-term single-token model: layers * per_layer_ms + weight_gb * ms_per_gigabyte
    per_layer_ms: float
    ms_per_gigabyte: float
    # Measured forward-pass cost per width. Keys are the widths actually measured.
    width_ms: Mapping[int, float]
    # Widths where going wider made cost per position worse than a narrower width
    # already achieved. Never schedule onto one of these.
    regression_widths: tuple[int, ...] = ()
    prefill_ms_per_position: float | None = None
    measured_at: str = ""
    notes: str = ""
    # Safety headroom when sizing a segment against a continuous-load limit. Step
    # cost drifts with temperature and cache length; a segment sized to exactly fill
    # the limit fails on the first slower run, and a failed run is not a slow run.
    segment_safety: float = 0.75

    def __post_init__(self) -> None:
        if self.layers < 1 or self.weight_gb <= 0:
            raise ProfileError("a profile needs a positive depth and weight")
        if self.per_layer_ms <= 0 or self.ms_per_gigabyte <= 0:
            raise ProfileError("both cost terms must be positive")
        if not self.width_ms:
            raise ProfileError("a profile without measured widths cannot schedule")
        if any(w < 1 or ms <= 0 for w, ms in self.width_ms.items()):
            raise ProfileError("measured widths and times must be positive")
        if not 0.0 < self.segment_safety <= 1.0:
            raise ProfileError("segment safety must lie in (0, 1]")
        unknown = set(self.regression_widths) - set(self.width_ms)
        if unknown:
            raise ProfileError(f"regression widths were never measured: {sorted(unknown)}")

    # -- identity -----------------------------------------------------------

    def applies_to(self, model_id: str, bits: int, group_size: int) -> bool:
        return (self.model_id, self.bits, self.group_size) == (model_id, bits, group_size)

    def require(self, model_id: str, bits: int, group_size: int) -> None:
        """Refuse to answer for a configuration this profile never saw.

        The width curve moved between two models on the same machine and between two
        group sizes for the same model. Silently answering anyway is the failure
        mode this whole exercise kept running into: a number that held at one scale
        and quietly stopped holding at another.
        """

        if not self.applies_to(model_id, bits, group_size):
            raise ProfileError(
                f"profile is for {self.model_id} at {self.bits} bit group "
                f"{self.group_size}; asked about {model_id} at {bits} bit group "
                f"{group_size}. Measure a profile for that configuration."
            )

    # -- prediction ---------------------------------------------------------

    def single_token_ms(self) -> float:
        """Predicted cost of one token at width one, from the two-term model."""

        return self.layers * self.per_layer_ms + self.weight_gb * self.ms_per_gigabyte

    def cost_shares(self) -> dict[str, float]:
        """How the single-token cost splits between dispatch and weight reading."""

        fixed = self.layers * self.per_layer_ms
        stream = self.weight_gb * self.ms_per_gigabyte
        total = fixed + stream
        return {
            "dispatch_ms": fixed,
            "bandwidth_ms": stream,
            "dispatch_share": fixed / total,
            "bandwidth_share": stream / total,
        }

    def project(self, *, bandwidth_gb_s: float, per_layer_ms: float) -> dict[str, float]:
        """Project this model's single-token cost onto another device.

        Both parameters are required. Reusing this machine's dispatch cost with
        another device's bandwidth assumes the two schedulers are equally fast,
        which is exactly the assumption that makes a projection look like a
        measurement.
        """

        if bandwidth_gb_s <= 0 or per_layer_ms <= 0:
            raise ProfileError("device parameters must both be positive")
        fixed = self.layers * per_layer_ms
        stream = self.weight_gb / bandwidth_gb_s * 1000.0
        total = fixed + stream
        return {
            "ms_per_token": total,
            "tokens_per_second": 1000.0 / total,
            "dispatch_share": fixed / total,
            "bandwidth_share": stream / total,
        }

    # -- scheduling ---------------------------------------------------------

    def usable_widths(self) -> list[int]:
        """Measured widths that are not regressions, cheapest per position first."""

        candidates = [w for w in self.width_ms if w not in self.regression_widths]
        if not candidates:
            raise ProfileError("every measured width is a regression")
        return sorted(candidates, key=lambda w: (self.width_ms[w] / w, w))

    def choose_width(self, requested: int) -> tuple[int, str]:
        """Pick a width that serves `requested` items without landing on a cliff.

        Never returns less than requested: dropping work to hit a nicer number would
        change what was asked for. It may return more, when a wider measured width
        costs no more in absolute terms -- those extra positions are free.
        """

        if requested < 1:
            raise ProfileError("cannot schedule fewer than one item")
        wide_enough = [w for w in self.usable_widths() if w >= requested]
        if not wide_enough:
            widest = max(self.width_ms)
            return widest, (
                f"{requested} exceeds the widest measured width {widest}; "
                "schedule in several passes"
            )
        exact = min(wide_enough)
        budget = self.width_ms[exact]
        free = [w for w in wide_enough if self.width_ms[w] <= budget]
        best = max(free) if free else exact
        if best != requested:
            if best > exact:
                return best, (
                    f"width {exact} costs {self.width_ms[exact]:.1f} ms and width "
                    f"{best} costs {self.width_ms[best]:.1f} ms; the extra positions "
                    "are free"
                )
            return best, f"width {requested} is a regression; nearest usable is {best}"
        return best, "requested width is measured and not a regression"

    def steps_per_segment(self, width: int, continuous_limit_s: float) -> int:
        """Decode steps that fit in one continuous block at this width.

        Generation has to be cut into segments when the efficient width would
        otherwise overrun a load limit. Splitting is free numerically -- a KV cache
        is state, and pausing between steps changes nothing about the next token --
        but only if the segment is sized from the measured step cost rather than a
        constant, which stops being safe at a different width.
        """

        if continuous_limit_s <= 0:
            raise ProfileError("continuous limit must be positive")
        if width not in self.width_ms:
            raise ProfileError(f"width {width} was never measured")
        budget_ms = continuous_limit_s * self.segment_safety * 1000.0
        return max(1, int(budget_ms / self.width_ms[width]))

    def plan(
        self, *, items: int, max_new_tokens: int, continuous_limit_s: float
    ) -> Plan:
        """Decide width and segmentation for one batch of work."""

        if max_new_tokens < 1:
            raise ProfileError("a plan needs at least one token to generate")
        width, reason = self.choose_width(items)
        per_segment = self.steps_per_segment(width, continuous_limit_s)
        segments = math.ceil(max_new_tokens / per_segment)
        seconds = max_new_tokens * self.width_ms[width] / 1000.0
        return Plan(
            width=width,
            requested_width=items,
            steps_per_segment=per_segment,
            segments=segments,
            estimated_seconds=seconds,
            reason=reason,
        )

    # -- speculation --------------------------------------------------------

    def speculation_break_even(self, draft_length: int) -> float:
        """Acceptance a free draft needs before speculating `draft_length` pays.

        Verifying k drafted tokens costs one pass at width k+1 and yields at most
        k+1 tokens, so the question is whether the expected yield beats that cost.
        The answer comes from the measured width curve rather than from theory,
        because the curve is a step function: the same k is a bargain on one model
        and a loss on another. On this device three drafted tokens need about 0.72
        acceptance on the 4B and far less on the flatter 1B.

        Returns 0.0 when even a never-accepted draft would pay, and 1.0 when no
        acceptance rate could.
        """

        if draft_length < 1:
            raise ProfileError("speculation needs at least one drafted token")
        width = draft_length + 1
        if width not in self.width_ms:
            raise ProfileError(f"width {width} was never measured")
        cost = self.width_ms[width] / self.width_ms[min(self.width_ms)]
        if cost <= 1.0:
            return 0.0
        if cost > draft_length + 1:
            return 1.0
        low, high = 0.0, 1.0
        for _ in range(60):
            mid = (low + high) / 2.0
            if sum(mid**i for i in range(draft_length + 1)) < cost:
                low = mid
            else:
                high = mid
        return high

    def speculation_speedup(self, draft_length: int, acceptance: float) -> float:
        """Expected speedup from a free draft at a given acceptance rate."""

        if not 0.0 <= acceptance <= 1.0:
            raise ProfileError("acceptance must lie in [0, 1]")
        if draft_length < 1:
            return 1.0
        width = draft_length + 1
        if width not in self.width_ms:
            raise ProfileError(f"width {width} was never measured")
        yielded = sum(acceptance**i for i in range(draft_length + 1))
        return yielded * self.width_ms[min(self.width_ms)] / self.width_ms[width]

    # -- persistence --------------------------------------------------------

    def as_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "model_id": self.model_id,
            "bits": self.bits,
            "group_size": self.group_size,
            "layers": self.layers,
            "weight_gb": self.weight_gb,
            "per_layer_ms": self.per_layer_ms,
            "ms_per_gigabyte": self.ms_per_gigabyte,
            "width_ms": {str(k): v for k, v in sorted(self.width_ms.items())},
            "regression_widths": list(self.regression_widths),
            "prefill_ms_per_position": self.prefill_ms_per_position,
            "measured_at": self.measured_at,
            "notes": self.notes,
            "segment_safety": self.segment_safety,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "HardwareProfile":
        try:
            widths = {int(k): float(v) for k, v in dict(raw["width_ms"]).items()}
            return cls(
                device=str(raw["device"]),
                model_id=str(raw["model_id"]),
                bits=int(raw["bits"]),
                group_size=int(raw["group_size"]),
                layers=int(raw["layers"]),
                weight_gb=float(raw["weight_gb"]),
                per_layer_ms=float(raw["per_layer_ms"]),
                ms_per_gigabyte=float(raw["ms_per_gigabyte"]),
                width_ms=widths,
                regression_widths=tuple(int(w) for w in raw.get("regression_widths", ())),
                prefill_ms_per_position=(
                    None if raw.get("prefill_ms_per_position") is None
                    else float(raw["prefill_ms_per_position"])
                ),
                measured_at=str(raw.get("measured_at", "")),
                notes=str(raw.get("notes", "")),
                segment_safety=float(raw.get("segment_safety", 0.75)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProfileError(f"profile is malformed: {exc}") from exc

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "HardwareProfile":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileError(f"profile at {path} is unreadable: {exc}") from exc
        return cls.from_dict(raw)


# Sample counts, from the accuracy measurements rather than from the hardware.
# Reported here because spending a sample budget is a scheduling decision, but the
# evidence is a different kind: 1B accuracy rose 27.1% to 65.6% at 32 samples
# (p=0.00063), while 8 samples on the 4B was significantly *worse* than one
# (-31.2 points, p=0.025). Coverage saturates near four samples and follows retry
# statistics; majority voting needs about thirty-two.
_HARMFUL_SAMPLES = range(5, 16)


def sample_budget(*, want_accuracy: bool) -> int:
    """How many samples to draw, avoiding the range that measured worse than one.

    There is no middle setting. A handful of samples is enough to stop a model
    degenerating mid-answer but not enough for a majority to be reliable, and it has
    already given up the single most likely path that greedy decoding takes. Measured,
    that combination lands below drawing one sample at all.
    """

    budget = 32 if want_accuracy else 1
    if budget in _HARMFUL_SAMPLES:  # pragma: no cover - defensive, both values are outside
        raise ProfileError(f"{budget} samples measured worse than one")
    return budget
