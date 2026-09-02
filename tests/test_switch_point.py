"""H1.0's decision logic, decided offline against a fake engine."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from friday_calibrate.runner import Sample, noise_mde, paired_arms  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "switch_point_measure", ROOT / "experiments" / "switch_point" / "measure.py"
)
measure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measure)


def _engine(times):
    """A fake engine: knob key -> request seconds. Tokens are always identical."""

    def run(knobs):
        key = tuple(sorted(knobs.items()))
        seconds = times[key]
        return Sample(ttft_seconds=seconds / 2, decode_tps=32 / (seconds / 2),
                      tokens=32, token_sha256="a" * 64)

    return run


def test_pair_rule_is_even_bounded_and_scales_with_noise():
    assert measure.pairs_for(0.0) == 6
    assert measure.pairs_for(0.03) == 6
    assert measure.pairs_for(0.0369) == 10
    assert measure.pairs_for(0.1425) == 24
    assert all(measure.pairs_for(s) % 2 == 0 for s in (0.0, 0.02, 0.05, 0.09, 0.2))


def test_baseline_arm_is_the_combined_path_not_the_bare_engine():
    """Without `baseline_knobs` the baseline would be the engine nobody ships."""

    combined = tuple(sorted(measure.COMBINED.items()))
    candidate = dict(measure.COMBINED, speculate_k=1, speculate_ngram=3)
    seen = []

    def run(knobs):
        seen.append(tuple(sorted(knobs.items())))
        return Sample(1.0, 32.0, 32, "a" * 64)

    paired_arms(run, candidate, pairs=2, baseline_knobs=measure.COMBINED)
    assert set(seen) == {combined, tuple(sorted(candidate.items()))}
    assert () not in seen


def test_a_faster_candidate_beyond_the_noise_wins():
    combined = tuple(sorted(measure.COMBINED.items()))
    candidate = dict(measure.COMBINED, speculate_k=1, speculate_ngram=3)
    run = _engine({combined: 1.0, tuple(sorted(candidate.items())): 0.85})
    baseline, cand, breaks = paired_arms(run, candidate, pairs=6,
                                         baseline_knobs=measure.COMBINED)
    assert not breaks
    from friday_optimizer.integration import evaluate_integration

    result = evaluate_integration(baseline, cand, arm="warm", min_gain=0.03,
                                  mde=0.03, min_pairs=3)
    assert measure.verdict_for(result.status) == "wins"


def test_a_slower_candidate_loses_and_a_wash_ties():
    combined = tuple(sorted(measure.COMBINED.items()))
    from friday_optimizer.integration import evaluate_integration

    for factor, expected in ((1.15, "loses"), (1.0, "tie")):
        candidate = dict(measure.COMBINED, speculate_k=2, speculate_ngram=3)
        run = _engine({combined: 1.0, tuple(sorted(candidate.items())): factor})
        baseline, cand, _ = paired_arms(run, candidate, pairs=6,
                                        baseline_knobs=measure.COMBINED)
        result = evaluate_integration(baseline, cand, arm="warm", min_gain=0.03,
                                      mde=0.03, min_pairs=3)
        assert measure.verdict_for(result.status) == expected, (factor, result.status)


def test_token_identity_break_ends_the_arm():
    calls = {"n": 0}

    def run(knobs):
        calls["n"] += 1
        digest = "a" * 64 if calls["n"] < 2 else "b" * 64
        return Sample(1.0, 32.0, 32, digest)

    baseline, _, breaks = paired_arms(run, {"speculate_k": 1}, pairs=6,
                                      baseline_knobs=measure.COMBINED)
    assert breaks and breaks[0].startswith("token_identity_broken")
    assert baseline == []


def test_aa_noise_is_measured_on_the_arm_it_describes():
    """A/A must run the combined path against itself, not the bare engine."""

    seen = []

    def run(knobs):
        seen.append(tuple(sorted(knobs.items())))
        return Sample(1.0 + 0.01 * len(seen), 32.0, 32, "a" * 64)

    noise_mde(run, pairs=6, knobs=measure.COMBINED)
    assert set(seen) == {tuple(sorted(measure.COMBINED.items()))}
