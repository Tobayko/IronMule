"""Recompute the expansion plan's target table from the evidence it cites.

The plan multiplies four ratios: F1 `0.860057` x prefill step size `0.9288` x
decode environment `0.8906` x speculative `0.9124`. Three things are wrong with
that product, and they push in the same direction.

1. **Phase ratios do not multiply.** ``friday_optimizer/integration.py:99`` says
   so in the repository's own words: a prefill gain and a decode gain act on
   different parts of the same request, so the composition is the time-weighted
   mean, not the product. That module exists because F1 was built to falsify
   exactly this kind of arithmetic.

2. **Two of the factors are already inside F1.** F1's candidate arm is
   ``Knobs(compiled_fixed_cache=True, head_skip_prefill=True)``
   (``experiments/f1_integration/measure_f1.py``). The decode-environment factor
   `0.8906` is `fixed_compiled 0.9296 x bundled_readback 0.9581`, and
   `fixed_compiled` **is** `compiled_fixed_cache`. Stacking it on F1 counts it
   twice. Only `bundled_readback` is genuinely new.

3. **The prefill step-size factor has no baseline left to improve.** It comes
   from ``experiments/decode_width/prefill.json``, which chunks a prompt through
   ``mlx_lm`` at 256/512/1024/2048. The engine F1 measured through prefills the
   whole prompt in a single forward (``ironmule/runtime.py:_prefill``,
   ``prefix_cache`` is ``None``), which already *is* the fast end of that curve.
   That measurement also has n=1 per chunk size, ascending, in one process — the
   exact shape W1 showed warm-up produces on its own.

Run: ``python experiments/f1_projection/recompute.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from friday_optimizer.integration import project_request_ratio  # noqa: E402

#: F1's own six pairs, docs/ARBEITSJOURNAL.md "2026-09-02 - F1 warmer Arm
#: gemessen". Medians of the measured baseline arm, not the preregistration's
#: planning values: this is what the machine actually did.
BASELINE_TTFT = 1.7287
BASELINE_DECODE_TPS = 67.39
SHORT_TOKENS = 32

F1_RATIO = 0.860057  # measured, 6 pairs, CI [0.853444, 0.873056]
#: Measured phase ratios from the same run. Using these instead of re-deriving
#: them from BACKLOG percentages keeps the composition anchored to evidence.
F1_TTFT_RATIO = 0.849479
F1_DECODE_TPS_RATIO = 1.094084

#: New on top of F1 - but only under a bar the project has not yet adopted.
#:
#: BACKLOG.md P1 stacks the two decode knobs as 0.9296 x 0.9581. The first is
#: F1's compiled_fixed_cache and is already inside F1. The second was measured
#: in Zyklus 17 and **rejected**: ratio median 0.9581074518 against a
#: preregistered median_ratio_max of 0.95, verdict
#: ``no_clear_speedup_baseline_retained`` (PROJECT_STATUS.md:36).
#:
#: It was rejected for missing a *size* threshold, not for being absent: its
#: bootstrap CI is [0.9534714913689875, 0.9598849359131666], wholly below 1.0.
#: So it is a real ~4.2 % decode gain that did not clear a 5 % promotion bar.
#:
#: Whether serving may use it is a decision, not a measurement. A study
#: promotion asserts a size; a serving knob only has to be real and identical.
#: The device profile's bar (interval wholly below 1.0, tokens identical) is
#: deliberately the weaker one - and adopting it is exactly what has to be said
#: out loud rather than assumed by quietly reusing the number.
BUNDLED_READBACK = 0.9581
BUNDLED_READBACK_STATUS = (
    "measured, CI [0.95347, 0.95989] wholly below 1.0, but rejected at its own "
    "preregistered 5 % promotion threshold (0.9581 vs 0.95). Counting it here "
    "assumes serving adopts a weaker bar than study promotion."
)
#: The speculative numbers cannot be used as decode ratios.
#: ``friday_hardware/speculate.py:182`` starts the timer *before* the prefill
#: forward on line 183, so ``experiments/prompt_lookup/real/results.json``'s
#: `1.162` (code, 4B) is a whole-request speedup at 859 prompt / 64 generated
#: tokens. Turning it into a decode-only ratio needs the prefill share of that
#: run, which was not recorded. Re-projecting it onto 897/512 would be inventing
#: the missing half.
#:
#: So the long regime is reported as a *sensitivity*, not a projection: what the
#: end-to-end gain would be if the speculative decode-only ratio lay anywhere in
#: this band. The band is deliberately wide.
SPECULATIVE_DECODE_BAND = (0.70, 0.95)
SPECULATIVE_UNFAVOURABLE = 1.0  # the bandit's job is to make this exactly 1.0

#: W1: 32 tokens 63.83 tok/s, 256 tokens 72.36 tok/s, still rising inside the
#: run (68.34 -> 77.23). 512 is an extrapolation, not a measurement, so the long
#: regime is reported as a range over a plausible band.
LONG_TOKENS = 512
LONG_TPS_BAND = (72.0, 80.0)


def compose(*, tokens: int, decode_tps: float, ttft_ratio: float, decode_ratios: list[float]):
    """Time-weighted composition, the way integration.py defines it."""

    decode_ratio = 1.0
    for value in decode_ratios:
        decode_ratio *= value
    return project_request_ratio(
        ttft_seconds=BASELINE_TTFT,
        tokens=tokens,
        decode_tps=decode_tps,
        ttft_ratio=ttft_ratio,
        # project_request_ratio takes a *throughput* ratio, so a decode-time
        # ratio below 1 is a throughput ratio above 1.
        decode_tps_ratio=1.0 / decode_ratio,
    )


def f1_phase_ratios() -> tuple[float, float]:
    """F1's measured phase ratios, as decode *time* rather than throughput."""

    return F1_TTFT_RATIO, 1.0 / F1_DECODE_TPS_RATIO


def main() -> int:
    ttft_ratio, decode_ratio = f1_phase_ratios()
    check = compose(
        tokens=SHORT_TOKENS,
        decode_tps=BASELINE_DECODE_TPS,
        ttft_ratio=ttft_ratio,
        decode_ratios=[decode_ratio],
    )

    rows = []

    # Short: 897 / 32, the sealed workload F1 measured.
    short_plus_readback = compose(
        tokens=SHORT_TOKENS,
        decode_tps=BASELINE_DECODE_TPS,
        ttft_ratio=ttft_ratio,
        decode_ratios=[decode_ratio, BUNDLED_READBACK],
    )
    rows.append(
        {
            "regime": "short 897/32",
            "plan_claim_percent": 18.7,
            "measured_percent": round((1.0 - F1_RATIO) * 100, 2),
            "conditional_percent": round((1.0 - short_plus_readback) * 100, 2),
            "condition": BUNDLED_READBACK_STATUS,
            "levers_beyond_f1": ["bundled_readback (conditional)"],
            "note": (
                "Without adopting the weaker serving bar there is no lever left "
                "beyond F1 on this workload at all, and the honest number is F1's "
                "measured 13.99 %. Prefill step size is gone either way: no chunked "
                "baseline left to improve."
            ),
        }
    )

    for label, speculatives, claim in (
        ("long favourable 897/512 (code)", SPECULATIVE_DECODE_BAND, 19.2),
        ("long unfavourable 897/512 (prose)", (SPECULATIVE_UNFAVOURABLE,), 13.0),
    ):
        band = [
            compose(
                tokens=LONG_TOKENS,
                decode_tps=tps,
                ttft_ratio=ttft_ratio,
                decode_ratios=[decode_ratio, BUNDLED_READBACK, speculative],
            )
            for tps in LONG_TPS_BAND
            for speculative in speculatives
        ]
        rows.append(
            {
                "regime": label,
                "plan_claim_percent": claim,
                "sensitivity_percent_range": [
                    round((1.0 - max(band)) * 100, 2),
                    round((1.0 - min(band)) * 100, 2),
                ],
                "levers_beyond_f1": ["bundled_readback", "speculative"],
                "status": "not_projectable",
                "note": (
                    "Two inputs are missing, not uncertain: the decode rate at 512 "
                    "tokens (W1 measured 32 and 256 and the rate was still rising) "
                    "and the speculative decode-only ratio (only a whole-request "
                    "speedup at a different token count was recorded). This row is "
                    "a sensitivity over those two, not a projection."
                ),
            }
        )

    report = {
        "schema": "friday.f1-projection.recompute.v1",
        "f1_measured_ratio": F1_RATIO,
        "f1_phase_ratios": {"ttft": round(ttft_ratio, 6), "decode": round(decode_ratio, 6)},
        "f1_recomposed_ratio": round(check, 6),
        "f1_recomposition_error": round(abs(check - F1_RATIO), 6),
        "rows": rows,
        "removed_from_the_plan": {
            "prefill_step_size_0.9288": (
                "not available: the F1 engine already prefills unchunked, and the "
                "0.9288 comes from a single unrepeated measurement per chunk size, "
                "taken in ascending order in one process"
            ),
            "decode_environment_0.8906": (
                "double counted: 0.9296 of it is fixed_compiled, which is already "
                "F1's compiled_fixed_cache; only bundled_readback 0.9581 is new"
            ),
            "multiplication_of_phase_ratios": (
                "phase gains compose as a time-weighted mean, not a product "
                "(friday_optimizer/integration.py:99)"
            ),
        },
        "conclusion": (
            "On the sealed short workload the plan's 18.7 % does not survive. Two "
            "of its four factors were already inside F1 or had no baseline left to "
            "improve, and the third was rejected at its own promotion threshold. "
            "Without a decision to adopt a weaker serving bar the number is F1's "
            "measured 13.99 %; with that decision it is 14.5 %. Either way the "
            "20 % target is out of reach on this workload under strict identity. "
            "The long regime is not projectable from what has been measured."
        ),
        "what_would_make_the_long_regime_projectable": [
            "one gated run at 897/512 recording prefill_ns and decode_ns separately "
            "(the F1 worker already does; measure_prompt_lookup does not)",
            "the same run with speculation on and off, so the speculative ratio is "
            "decode-only and measured at the token count it is claimed for",
        ],
        "formal_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
