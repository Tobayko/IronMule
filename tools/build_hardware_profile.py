#!/usr/bin/env python3
"""Assemble a hardware profile from measurements that already exist.

This does not measure. It reads what `measure_decode_width.py` and
`measure_device_model.py` wrote and packs it into the form
`friday_hardware.HardwareProfile` consumes, so the numbers a scheduler acts on can
be traced back to the run that produced them.

Kept separate from the measuring tools on purpose: a profile that could be written
by hand, or assembled from mismatched runs, is a profile that will eventually
describe a machine nobody has. Every field here comes from a named report file.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from friday_hardware import HardwareProfile, ProfileError  # noqa: E402

MODEL_IDS = {
    "4b": "mlx-community/gemma-3-4b-it-4bit",
    "1b": "mlx-community/gemma-3-1b-it-4bit",
}


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read {path}: {exc}") from exc


def build(width_report: Path, device_report: Path, model_key: str, *,
          prefill_report: Path | None = None) -> HardwareProfile:
    width = _load(width_report)
    device = _load(device_report)

    reported = width.get("model", "4b")
    if reported != model_key:
        raise ProfileError(
            f"{width_report} was measured for '{reported}', not '{model_key}'"
        )

    forward = width["forward_pass"]
    width_ms = {int(row["width"]): float(row["ms"]) for row in forward["widths"]}
    regressions = tuple(int(w) for w in forward["policy"]["regression_widths"])

    # Depth and weight come from the device-model run, which recorded them per model
    # while fitting; taking them from anywhere else would risk describing one model
    # with another's shape.
    shape = next(
        (p for p in device["fit_points"] if p["model"] == model_key), None
    )
    if shape is None:
        raise ProfileError(f"{device_report} has no fit point for '{model_key}'")

    fit = device["device_model"]
    prefill = None
    if prefill_report is not None:
        rows = _load(prefill_report)
        best = min(
            (v for k, v in rows.items() if not k.startswith("_")),
            key=lambda v: float(v["ms_per_position"]),
        )
        prefill = float(best["ms_per_position"])

    return HardwareProfile(
        device=f"{platform.machine()} / {platform.system()} {platform.release()}",
        model_id=MODEL_IDS[model_key],
        bits=4,
        group_size=64,
        layers=int(shape["layers"]),
        weight_gb=float(shape["weight_gb"]),
        per_layer_ms=float(fit["per_layer_ms"]),
        ms_per_gigabyte=float(fit["ms_per_gigabyte"]),
        width_ms=width_ms,
        regression_widths=regressions,
        prefill_ms_per_position=prefill,
        measured_at=str(width.get("model_revision", ""))[:12],
        notes=(
            "Assembled from measurement reports; see docs/DECODE_WIDTH_BEFUND and "
            "docs/GERAETEMODELL. Applies to this device, model and quantisation only."
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_IDS), required=True)
    parser.add_argument("--width-report", type=Path, required=True)
    parser.add_argument("--device-report", type=Path, required=True)
    parser.add_argument("--prefill-report", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    built = build(
        args.width_report, args.device_report, args.model,
        prefill_report=args.prefill_report,
    )
    built.save(args.out)
    plan = built.plan(items=8, max_new_tokens=240, continuous_limit_s=6.0)
    shares = built.cost_shares()
    print(json.dumps({
        "written": str(args.out),
        "model": built.model_id,
        "layers": built.layers,
        "weight_gb": built.weight_gb,
        "single_token_ms": round(built.single_token_ms(), 3),
        "dispatch_share": round(shares["dispatch_share"], 4),
        "regression_widths": list(built.regression_widths),
        "example_plan_for_8_items": plan.as_dict(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
