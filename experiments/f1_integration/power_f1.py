"""Can F1 actually detect what it claims to detect?

F1's warm arm expects 11.93 % on the workload its harness runs, against a
preregistered threshold of 10 %. That is 1.93 points of margin. This script
asks the real decision function, under realistic noise, how often it reaches
`qualified` — and how often it wrongly qualifies when the truth is below the
threshold.

Offline simulation. Synthetic samples exercise the decision rule; no
performance claim follows. Noise levels are taken from sealed evidence:

  persistent process, six pairs   relative sd 0.734 %
  head skip, per session          sd_log_ratio 0.0066 to 0.0083
  head skip calibration           session_ratio_sd 0.4526 %
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from friday_optimizer.evaluator import MetricSample  # noqa: E402
from friday_optimizer.integration import evaluate_integration  # noqa: E402

#: Warm baseline of the sealed persistent-process study.
BASE_TTFT = 1.7851
BASE_TPS = 70.99
TOKENS = 32
TRIALS = 400
MIN_GAIN = 0.10
MDE = 0.05


def pair_samples(true_ratio: float, noise: float, pairs: int, rng: random.Random):
    """Build one paired A/B set whose request-time ratio is *true_ratio*."""

    baseline, candidate = [], []
    for index in range(pairs):
        order = "AB" if index % 2 == 0 else "BA"
        # Session-level drift affects both arms of a pair equally and cancels
        # in the ratio, exactly as the paired design intends.
        drift = rng.gauss(1.0, noise)
        base_request = (BASE_TTFT + TOKENS / BASE_TPS) * drift
        observed = true_ratio * rng.gauss(1.0, noise)
        candidate_request = base_request * observed
        # Split each request back into ttft and decode so the sample carries
        # the same fields a real run records.
        share = BASE_TTFT / (BASE_TTFT + TOKENS / BASE_TPS)
        baseline.append(MetricSample(
            session_id=f"s{index}", pair_id=f"p{index}", arm="baseline", order=order,
            ttft_seconds=base_request * share, tokens=TOKENS,
            decode_tps=TOKENS / (base_request * (1 - share)),
        ))
        candidate.append(MetricSample(
            session_id=f"s{index}", pair_id=f"p{index}", arm="candidate", order=order,
            ttft_seconds=candidate_request * share, tokens=TOKENS,
            decode_tps=TOKENS / (candidate_request * (1 - share)),
        ))
    return baseline, candidate


def rate(true_gain: float, noise: float, pairs: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    counts: dict[str, int] = {}
    for trial in range(TRIALS):
        baseline, candidate = pair_samples(1.0 - true_gain, noise, pairs, rng)
        result = evaluate_integration(baseline, candidate, arm="warm", min_gain=MIN_GAIN,
                                      mde=MDE, min_pairs=pairs, seed=trial, resamples=200)
        counts[result.status] = counts.get(result.status, 0) + 1
    return {key: value / TRIALS for key, value in counts.items()}


def main() -> int:
    print(f"threshold {MIN_GAIN*100:.0f} %, mde {MDE*100:.0f} %, {TRIALS} trials per cell,"
          f" {TOKENS} generated tokens")
    print()
    print("A. power: true gain 11.93 % (the 322-token expectation), how often does it qualify?")
    print(f"{'pairs':>6} " + " ".join(f"{n*100:>9.1f} %" for n in (0.005, 0.010, 0.020, 0.030, 0.050)))
    for pairs in (6, 12, 20, 30):
        row = []
        for noise in (0.005, 0.010, 0.020, 0.030, 0.050):
            row.append(rate(0.1193, noise, pairs, seed=1000 + pairs).get("qualified", 0.0))
        print(f"{pairs:6d} " + " ".join(f"{value*100:9.1f} %" for value in row))
    print("        (columns are per-pair relative noise; sealed evidence sits at 0.5 to 0.8 %)")
    print()
    print("B. false qualification: true gain 8 %, below the threshold. Should be near zero.")
    print(f"{'pairs':>6} " + " ".join(f"{n*100:>9.1f} %" for n in (0.005, 0.010, 0.020, 0.030, 0.050)))
    for pairs in (6, 12, 20, 30):
        row = []
        for noise in (0.005, 0.010, 0.020, 0.030, 0.050):
            row.append(rate(0.08, noise, pairs, seed=2000 + pairs).get("qualified", 0.0))
        print(f"{pairs:6d} " + " ".join(f"{value*100:9.1f} %" for value in row))
    print()
    print("C. the 897-token expectation, 13.68 %, for comparison at six pairs")
    for noise in (0.005, 0.010, 0.020, 0.030, 0.050):
        outcome = rate(0.1368, noise, 6, seed=3000)
        print(f"  noise {noise*100:4.1f} %: " + ", ".join(f"{k} {v*100:.1f} %" for k, v in sorted(outcome.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
