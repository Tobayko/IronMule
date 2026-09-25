"""Which opt-in numeric plan is qualified for this model on this device, and which is not.

A numeric plan — `compute_dtype="float32"` or `"float16"` — computes the checkpoint's own
weights in a different floating type. It changes the output, so IronMule never selects one
for a caller. That much was already true. What PORT2 measured is that "never selects" is
not enough guidance, because the answer is neither per device nor per plan but per
*architecture*, and the three possible answers are far apart:

* Qwen 3 in `float16` runs at 0.31 of stock wall time and its perplexity ratio against its
  own bfloat16 is 0.997689 with the whole interval below 1. Faster and no worse.
* Llama 3.1 in `float32` runs at 1.52 — half again slower — while passing its quality gate
  comfortably. Correct, and a waste.
* Gemma 3 in `float16` runs at 0.31 and takes perplexity from 102.54 to 209.57. Fast and
  ruined.

`ironmule doctor` used to advise `float32` on any NVIDIA GPU below compute capability 8,
which is right for Gemma 3 and Qwen 3, wrong for Llama 3.1, and silent about `float16`
being both the best and the worst option in the set. This module replaces that with the
measurements, so a recommendation exists only where a measurement does.

Every row cites the committed run it comes from, and `tests/test_numeric_plans.py`
re-derives each number from that file, so a re-measurement moves the table or fails the
suite. A plan is only ever *recommended* when it is both faster and inside the quality
bound; a plan measured to break the bound is *refused*, because a numeric plan the caller
cannot evaluate is exactly the thing this project exists not to ship.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The quality bound the project uses everywhere: the upper end of the bootstrap interval
#: on the perplexity ratio against the checkpoint's own dtype.
QUALITY_BOUND = 1.005

#: Devices are grouped by what actually changes the answer, which is the arithmetic the
#: hardware has rather than the model of card. Turing and Volta emulate bfloat16.
CUDA_PRE_AMPERE = "cuda_pre_ampere"


@dataclass(frozen=True)
class PlanMeasurement:
    """One architecture, one plan, one device class, and what was measured."""

    architecture: str  # the mlx-lm module a block comes from, as `fast.py` identifies it
    plan: str
    device: str
    wall_ratio: float
    wall_evidence: str
    wall_arm: str
    models: tuple[str, ...]
    quality_ratio: float | None = None
    quality_interval: tuple[float, float] | None = None
    quality_evidence: tuple[str, str] | None = None
    #: Set when a gate ran but its result cannot be used. "not measured" and "measured and
    #: unusable" are different facts, and printing them the same way invites someone to
    #: paste the unusable number in as if it counted.
    quality_note: str | None = None

    @property
    def label(self) -> str:
        """The short name a reader recognises, e.g. `qwen3` for `mlx_lm.models.qwen3`."""
        return self.architecture.rsplit(".", 1)[-1]

    @property
    def faster(self) -> bool:
        return self.wall_ratio < 1.0

    @property
    def quality_known(self) -> bool:
        return self.quality_interval is not None

    @property
    def quality_passes(self) -> bool:
        """The whole interval is inside the bound."""
        return self.quality_known and self.quality_interval[1] < QUALITY_BOUND

    @property
    def quality_degrades(self) -> bool:
        """Even the optimistic end is outside the bound, so the loss is not noise.

        This is deliberately not `not quality_passes`. Gemma 3 in float32 measured
        1.017846 [0.990878; 1.051591] — an interval that contains 1, on 16 chunks of a text
        whose perplexity is 100. That is inconclusive, and PORT1 qualified exactly that plan
        by a different route, matching Apple Silicon's own float32 to 1e-5 nats. Treating it
        as a failure would refuse the plan this project already ships at +91%. Gemma 3 in
        float16 measured [1.873506; 2.244196], which no amount of interval width explains.
        """
        return self.quality_known and self.quality_interval[0] > QUALITY_BOUND

    @property
    def speedup_percent(self) -> float:
        return (1.0 / self.wall_ratio - 1.0) * 100.0

    def verdict(self) -> str:
        if not self.faster:
            return "slower"
        if self.quality_passes:
            return "recommended"
        if self.quality_degrades:
            return "refused"
        return "unqualified"


_R = "experiments/kaggle_compat/results"

#: Measured on a Kaggle 2 x Tesla T4 cell (compute capability 7.5), MLX 0.32.2 and
#: mlx-lm 0.31.3, 4-bit mlx-community checkpoints at pinned revisions. `research/LEDGER.md`
#: entry PORT2 carries the protocol; these are its cells.
MEASUREMENTS: tuple[PlanMeasurement, ...] = (
    PlanMeasurement(
        architecture="mlx_lm.models.gemma3_text", plan="float32", device=CUDA_PRE_AMPERE,
        wall_ratio=0.5224738508013052,
        wall_evidence=f"{_R}/port2-run6-59ce8efc/cross-fp16-gemma3-4b.json",
        wall_arm="ironmule_fp32", models=("mlx-community/gemma-3-4b-it-4bit",),
        quality_ratio=1.0178461034960435,
        quality_interval=(0.990878377714877, 1.05159097747294),
        quality_evidence=(f"{_R}/port2-run2-74fe1a6d/quality-gemma3-4b.json",
                          "ppl_ratio_fp32_over_bf16"),
    ),
    PlanMeasurement(
        architecture="mlx_lm.models.gemma3_text", plan="float16", device=CUDA_PRE_AMPERE,
        wall_ratio=0.3112612606949747,
        wall_evidence=f"{_R}/port2-run6-59ce8efc/cross-fp16-gemma3-4b.json",
        wall_arm="ironmule_fp16", models=("mlx-community/gemma-3-4b-it-4bit",),
        quality_ratio=2.0437919258411332,
        quality_interval=(1.8735056823875418, 2.2441964383761954),
        quality_evidence=(f"{_R}/port2-run6-59ce8efc/quality16-gemma3-4b-float16.json",
                          "paired-with-bf16"),
    ),
    PlanMeasurement(
        architecture="mlx_lm.models.llama", plan="float32", device=CUDA_PRE_AMPERE,
        wall_ratio=1.5181699264420379,
        wall_evidence=f"{_R}/port2-run4-c86664a3/cross-fused-llama31-8b.json",
        wall_arm="ironmule_fused_fp32", models=("mlx-community/Llama-3.1-8B-Instruct-4bit",),
        quality_ratio=1.0001260817054374,
        quality_interval=(0.9999932944999448, 1.0002600292622594),
        quality_evidence=(f"{_R}/port2-run2-74fe1a6d/quality-llama31-8b.json",
                          "ppl_ratio_fp32_over_bf16"),
    ),
    PlanMeasurement(
        architecture="mlx_lm.models.qwen3", plan="float32", device=CUDA_PRE_AMPERE,
        wall_ratio=0.5346285865587584,
        wall_evidence=f"{_R}/port2-run6-59ce8efc/cross-fp16-qwen3-8b.json",
        wall_arm="ironmule_fp32", models=("mlx-community/Qwen3-8B-4bit",),
        quality_ratio=0.9978173586287342,
        quality_interval=(0.9961770917809496, 0.9995836091934635),
        quality_evidence=(f"{_R}/port2-run2-74fe1a6d/quality-qwen3-8b.json",
                          "ppl_ratio_fp32_over_bf16"),
    ),
    PlanMeasurement(
        architecture="mlx_lm.models.qwen3", plan="float16", device=CUDA_PRE_AMPERE,
        wall_ratio=0.3099993176898163,
        wall_evidence=f"{_R}/port2-run6-59ce8efc/cross-fp16-qwen3-8b.json",
        wall_arm="ironmule_fp16",
        models=("mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen3-14B-4bit"),
        quality_ratio=0.9976891942506944,
        quality_interval=(0.996051171482173, 0.9994539470776317),
        quality_evidence=(f"{_R}/port2-run6-59ce8efc/quality16-qwen3-8b-float16.json",
                          "paired-with-bf16"),
    ),
    PlanMeasurement(
        architecture="mlx_lm.models.gpt_oss", plan="float32", device=CUDA_PRE_AMPERE,
        wall_ratio=0.2818421251530665,
        wall_evidence=f"{_R}/port2-run6-59ce8efc/cross-fp16-gptoss-20b.json",
        wall_arm="ironmule_fp32", models=("mlx-community/gpt-oss-20b-MXFP4-Q4",),
        quality_ratio=1.0036731004672461,
        quality_interval=(0.9399962188361414, 1.0647064625141354),
        quality_evidence=(f"{_R}/port2-run7-cddae1f9/quality-gptoss-20b-float32-24.json",
                          "paired-with-bf16"),
    ),
    PlanMeasurement(
        architecture="mlx_lm.models.gpt_oss", plan="float16", device=CUDA_PRE_AMPERE,
        wall_ratio=0.19907076432448156,
        wall_evidence=f"{_R}/port2-run6-59ce8efc/cross-fp16-gptoss-20b.json",
        wall_arm="ironmule_fp16", models=("mlx-community/gpt-oss-20b-MXFP4-Q4",),
        quality_ratio=1.0087523082914356,
        quality_interval=(0.94863413628636, 1.0665546660770577),
        quality_evidence=(f"{_R}/port2-run7-cddae1f9/quality-gptoss-20b-float16-16.json",
                          "paired-with-bf16"),
    ),
    PlanMeasurement(
        architecture="mlx_lm.models.ministral3", plan="float32", device=CUDA_PRE_AMPERE,
        wall_ratio=0.548738270413917,
        wall_evidence=f"{_R}/port2-run3-281b971a/cross-mistral-lean.json",
        wall_arm="ironmule_lean_fp32",
        models=("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",),
        # 512 and 256-token chunks ran out of memory at 13.26 GB of weights; 128 fits, so
        # the largest checkpoint that runs on one card also has a gate, and passes it.
        quality_ratio=0.9980194835285368,
        quality_interval=(0.9960571974819411, 0.9999098222387136),
        quality_evidence=(f"{_R}/port2-run8-da1a6469/quality-mistral-24b-float32-8.json",
                          "paired-with-bf16"),
    ),
    # `native` is not a dtype: the bf16 checkpoint stays, and IronMule's own CUDA kernels do the
    # 4-bit matmuls (`ironmule/cuda_native.py`, PERF1). Speed is the product path itself
    # (`compute_dtype="native"` through `cross.py`, run 7). The plan changes two paths, so four
    # gates ran, against stock bf16 on the same path: run 5 decode on 8B and prefill on 8B and
    # 14B, BACKLOG2 decode on 14B; the row carries the one with the highest upper bound. Run 5
    # measured the kernel with free rather than pinned float32 rounding — the same sums in a
    # possibly different order, below the final bf16 rounding the output goes through; BACKLOG2
    # measured it pinned, as the product runs it.
    PlanMeasurement(
        architecture="mlx_lm.models.qwen3", plan="native", device=CUDA_PRE_AMPERE,
        wall_ratio=0.201317682316364,
        wall_evidence=f"{_R}/perf1-run7-080bfab7/cross-native-qwen3-8b.json",
        wall_arm="ironmule_native",
        models=("mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen3-14B-4bit"),
        quality_ratio=1.0005260152936564,
        quality_interval=(0.9990003312687394, 1.0020184600683042),
        quality_evidence=(f"{_R}/backlog2-run1-17b2ca39/gate-qwen3-14b-kernel-decode.json",
                          "worst-of-four-gates"),
    ),
    # Gemma 3 under `native` (PERF1-Y): the same two paths gated against stock bf16, with BOS on
    # every chunk (without it Gemma 3 moves by up to 0.34 nats per chunk from rounding alone),
    # BACKLOG8 on 12B; the row carries the decode path, the higher upper bound. Speed is the
    # product path against stock in PERF1 run 18 (`cross.py`, `compute_dtype="native"`).
    PlanMeasurement(
        architecture="mlx_lm.models.gemma3_text", plan="native", device=CUDA_PRE_AMPERE,
        wall_ratio=0.19992481489806835,
        wall_evidence=f"{_R}/perf1-run18-863237d6/cross-gemma3-12b.json",
        wall_arm="ironmule_native",
        models=("mlx-community/gemma-3-12b-it-4bit",),
        quality_ratio=1.0005780846983543,
        quality_interval=(0.9979509901997468, 1.0031967314836343),
        quality_evidence=(f"{_R}/backlog8-run1-1665f2ae/gate-gemma3-12b-kernel-decode.json",
                          "worst-of-two-gates"),
    ),
    # Gemma 4 deliberately carries no quality interval, and the reason has to be read
    # before anyone "completes" these rows. The gate ran and produced
    # 0.974548 [0.945572; 1.003527] for float32 — an upper bound inside the bound, which
    # would make it `recommended`. It is not usable: the bfloat16 reference it is measured
    # against has a perplexity of 22 212 on WikiText-2, where Gemma 3 4B scores 100.5 and
    # Qwen 3 8B 14.9 on the same text through the same harness. A ratio between two numbers
    # that mean nothing is not evidence of anything, so the speed stands on its own and the
    # verdict stays `unqualified` until the reference itself is explained. `PORT2-I` in the
    # backlog carries that; chat decoding is fine and token-identical to stock.
    PlanMeasurement(
        architecture="mlx_lm.models.gemma4_text", plan="float32", device=CUDA_PRE_AMPERE,
        wall_ratio=0.41069269598397673,
        wall_evidence=f"{_R}/port2-run9b-e8751c84/cross-gemma4-e2b.json",
        wall_arm="ironmule_fp32",
        models=("mlx-community/gemma-4-e2b-it-4bit", "mlx-community/gemma-4-e4b-it-4bit",
                "mlx-community/gemma-4-E4B-it-qat-4bit"),
        quality_note="gate ran; its bfloat16 reference scores perplexity 22212, so unusable",
    ),
    PlanMeasurement(
        architecture="mlx_lm.models.gemma4_text", plan="float16", device=CUDA_PRE_AMPERE,
        wall_ratio=0.253552451835916,
        wall_evidence=f"{_R}/port2-run9b-e8751c84/cross-gemma4-e2b.json",
        wall_arm="ironmule_fp16",
        models=("mlx-community/gemma-4-e2b-it-4bit", "mlx-community/gemma-4-e4b-it-4bit",
                "mlx-community/gemma-4-E4B-it-qat-4bit"),
        quality_note="gate ran; its bfloat16 reference scores perplexity 22212, so unusable",
    ),
)


def architecture_of(model: Any) -> str | None:
    """The mlx-lm module a model's transformer blocks come from.

    The same key `fast.py` fuses by, for the same reason: a checkpoint's `model_type` and
    its `architectures` entry are what the publisher wrote, while the module is what mlx-lm
    actually runs. `mistral3` resolves to `ministral3` or `llama` depending on its config,
    and those are different measurements.
    """
    from .runtime import _trunk

    for candidate in (lambda: _trunk(model).layers, lambda: model.layers):
        # Both spellings, because mlx-lm exposes the blocks on the top-level `Model` and on
        # the inner one, and a helper that returns None for a model plainly holding layers
        # would silently skip the guard.
        try:
            layers = candidate()
            return type(layers[0].self_attn).__module__
        except (AttributeError, IndexError, TypeError):
            continue
    return None


def device_class(info: dict[str, Any] | None) -> str | None:
    """Name the arithmetic this device has, from an `mx.device_info()`-shaped mapping."""
    if not info:
        return None
    major = info.get("compute_capability_major")
    if isinstance(major, bool) or not isinstance(major, int):
        return None
    return CUDA_PRE_AMPERE if major < 8 else None


def measurements_for(architecture: str, device: str | None) -> tuple[PlanMeasurement, ...]:
    if device is None:
        return ()
    return tuple(row for row in MEASUREMENTS
                 if row.architecture == architecture and row.device == device)


def recommend(architecture: str, device: str | None) -> tuple[str | None, str]:
    """The fastest plan that is both faster and inside the quality bound, and why.

    Returns `(plan, reason)`; `plan` is None when nothing is recommended, and the reason
    says which of the three cases applies, because "no recommendation" for an unmeasured
    architecture and "no recommendation" for one that was measured and lost are different
    facts and a caller deserves to know which one they have.
    """
    rows = measurements_for(architecture, device)
    if not rows:
        return None, (f"no numeric plan has been measured for {architecture!r} on this device; "
                      "the checkpoint's own dtype is the only qualified path here")
    recommended = sorted((row for row in rows if row.verdict() == "recommended"),
                         key=lambda row: row.wall_ratio)
    if recommended:
        best = recommended[0]
        return best.plan, (
            f"--compute-dtype {best.plan} ran {best.label} at {best.wall_ratio:.4f} of stock "
            f"(+{best.speedup_percent:.0f}%) with a perplexity ratio of {best.quality_ratio:.6f} "
            f"[{best.quality_interval[0]:.6f}; {best.quality_interval[1]:.6f}], inside the "
            f"{QUALITY_BOUND} bound")
    refused = [row for row in rows if row.verdict() == "refused"]
    unqualified = [row for row in rows if row.verdict() == "unqualified"]
    slower = [row for row in rows if row.verdict() == "slower"]
    parts = []
    for row in refused:
        parts.append(f"{row.plan} is refused ({row.quality_ratio:.4f} perplexity ratio)")
    for row in unqualified:
        if row.quality_known:
            detail = (f"its quality interval [{row.quality_interval[0]:.4f}; "
                      f"{row.quality_interval[1]:.4f}] is too wide to qualify")
        else:
            detail = row.quality_note or "no quality gate could be run"
        parts.append(f"{row.plan} is faster (+{row.speedup_percent:.0f}%) but {detail}")
    for row in slower:
        parts.append(f"{row.plan} measured slower ({row.wall_ratio:.4f} of stock)")
    return None, f"no plan is recommended for {rows[0].label}: " + "; ".join(parts)


class PlanRefused(ValueError):
    """The requested plan was measured to break the quality bound on this device."""


def check(architecture: str, plan: str | None, device: str | None) -> None:
    """Refuse a plan this project measured to ruin the output on this device class.

    Only a measured failure refuses. An architecture nobody measured, or a plan whose
    interval is merely too wide, passes through — the plan is opt-in and the caller may
    have their own evidence. What they cannot have is IronMule quietly letting through the
    one combination it has watched double a model's perplexity.
    """
    if plan is None:
        return
    for row in measurements_for(architecture, device):
        if row.plan == plan and row.verdict() == "refused":
            raise PlanRefused(
                f"compute_dtype={plan!r} is refused for {row.label} on this device: "
                f"measured perplexity ratio {row.quality_ratio:.6f} "
                f"[{row.quality_interval[0]:.6f}; {row.quality_interval[1]:.6f}] against the "
                f"checkpoint's own dtype, far outside the {QUALITY_BOUND} bound "
                f"(evidence: {row.quality_evidence[0]}). The speed is real "
                f"(+{row.speedup_percent:.0f}%) and the output is not."
            )


__all__ = ["CUDA_PRE_AMPERE", "MEASUREMENTS", "PlanMeasurement", "PlanRefused", "QUALITY_BOUND",
           "check", "device_class", "measurements_for", "recommend"]
