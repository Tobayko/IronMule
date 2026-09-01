"""Derive F1's preregistered end-to-end thresholds from measured evidence.

Offline analysis only: it reads sealed experiment results already on disk and
prints a projection.  No model, no hardware, no network, no writes outside
stdout.  The projection is the number F1 exists to falsify.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from friday_optimizer.integration import (  # noqa: E402
    CONFIRMED_RATIOS,
    prefill_share,
    project_request_ratio,
)

PERSISTENT = ROOT / "experiments" / "persistent_process" / "results.json"


def warm_profile() -> dict[str, float]:
    """Median warm request profile of the sealed persistent-process study."""

    data = json.loads(PERSISTENT.read_text())
    pairs = data["characterization"]["pairs"] + data["validation"]["pairs"]
    warm = [pair["warm"] for pair in pairs]
    cold = [pair["cold"] for pair in pairs]
    tokens = {len(item["tokens"]) for item in warm}
    if len(tokens) != 1:
        raise SystemExit("token count is not constant across pairs")
    count = tokens.pop()
    ttft = statistics.median(item["ttft_ns"] for item in warm) / 1e9
    total = statistics.median(item["total_wall_ns"] for item in warm) / 1e9
    cold_total = statistics.median(item["total_wall_ns"] for item in cold) / 1e9
    # The first token arrives with the prefill, so only the remaining steps
    # are paid at the decode rate.
    return {
        "ttft_seconds": ttft,
        "tokens": count,
        "decode_seconds": total - ttft,
        "decode_tps": (count - 1) / (total - ttft),
        "warm_total": total,
        "cold_total": cold_total,
        "prompt_tokens": warm[0]["prompt_tokens"],
    }


def main() -> int:
    profile = warm_profile()
    ttft_ratio = CONFIRMED_RATIOS["head_skip_prefill"]
    decode_tps_ratio = 1.0 / CONFIRMED_RATIOS["fixed_compiled_cache"]
    shared = {
        "ttft_seconds": profile["ttft_seconds"],
        "tokens": profile["tokens"],
        "decode_tps": profile["decode_tps"],
    }
    share = prefill_share(**shared)
    warm_ratio = project_request_ratio(**shared, ttft_ratio=ttft_ratio, decode_tps_ratio=decode_tps_ratio)
    head_only = project_request_ratio(**shared, ttft_ratio=ttft_ratio)
    decode_only = project_request_ratio(**shared, decode_tps_ratio=decode_tps_ratio)
    persistent = CONFIRMED_RATIOS["persistent_process"]
    cold_ratio = persistent * warm_ratio

    print(f"workload           prompt={profile['prompt_tokens']} tokens, generated={profile['tokens']}")
    print(f"warm baseline      ttft={profile['ttft_seconds']:.4f}s decode={profile['decode_seconds']:.4f}s"
          f" ({profile['decode_tps']:.2f} tok/s) total={profile['warm_total']:.4f}s")
    print(f"cold baseline      total={profile['cold_total']:.4f}s")
    print(f"prefill share      {share * 100:.2f} % of the warm request")
    print()
    print(f"head_skip only     ratio={head_only:.6f}  gain={100 * (1 - head_only):.2f} %")
    print(f"fixed_compiled     ratio={decode_only:.6f}  gain={100 * (1 - decode_only):.2f} %")
    print(f"both, warm arm     ratio={warm_ratio:.6f}  gain={100 * (1 - warm_ratio):.2f} %")
    print(f"both + persistent  ratio={cold_ratio:.6f}  gain={100 * (1 - cold_ratio):.2f} %  (vs cold baseline)")
    print()
    naive = ttft_ratio * CONFIRMED_RATIOS["fixed_compiled_cache"]
    print(f"naive product      ratio={naive:.6f}  gain={100 * (1 - naive):.2f} %  <- wrong, phases do not multiply")
    print()
    print("sensitivity — generated tokens vs composed warm gain:")
    for count in (8, 16, 32, 64, 128, 256, 512):
        ratio = project_request_ratio(
            ttft_seconds=profile["ttft_seconds"], tokens=count, decode_tps=profile["decode_tps"],
            ttft_ratio=ttft_ratio, decode_tps_ratio=decode_tps_ratio,
        )
        current = prefill_share(ttft_seconds=profile["ttft_seconds"], tokens=count, decode_tps=profile["decode_tps"])
        print(f"  {count:4d} tokens  prefill_share={current * 100:5.2f} %  ratio={ratio:.6f}  gain={100 * (1 - ratio):5.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
