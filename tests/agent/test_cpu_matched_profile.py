import pytest

from lab.rp.cpu_matched_profile import measured_margin


def receipt():
    return [{"context": 68+13*i, "base": 1, "threads": t, "seed": i+1,
             "seconds": 0.25 if t == 4 else 1., "tokens_sha256": str(i)}
            for i in range(4) for t in ((1, 4, 1) if i % 2 == 0 else (4, 1, 1))]


def test_regrades_real_native_work_in_both_orders_and_rejects_wrong_tokens_or_noise():
    rows = receipt()
    got = measured_margin("1:1:4:4", 3, rows)
    assert got == {"windows": 4, "candidate_threads": 4, "max_ratio": .25, "max_aa_fraction": 0.,
                   "min_gain_after_noise_us_per_step": 93750.}
    rows[-1]["tokens_sha256"] = "bad"
    with pytest.raises(ValueError):
        measured_margin("1:1:4:4", 3, rows)
    rows = receipt()
    rows[2]["seconds"] = .1
    with pytest.raises(ValueError):
        measured_margin("1:1:4:4", 3, rows)
