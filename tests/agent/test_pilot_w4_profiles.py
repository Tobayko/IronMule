import numpy as np
import pytest

from ironmule.pilot import Pilot
from training.pilot_w4_profiles import NAMES, features, fit_head


def test_whole_profile_requires_consensus_support_and_local_receipt():
    zero = np.zeros
    head = [zero((1, 2)), zero(1), zero((1, 1)), zero(1), zero((3, 1)), np.array([.1, .8, .2])]
    different = [*head[:-1], np.array([.1, .2, .8])]

    def choose(heads, eligible=NAMES, support=True, error=.1):
        return Pilot.startup_profile(heads, features(16), NAMES, eligible, "q4_q6", in_support=support, margin_error=error)

    assert choose([head])["profile"] == "q4"
    assert choose([head], ["q4_q6"])["reason"] == "whole profile lacks local qualification"
    assert choose([head, different])["reason"] == "unresolved model margin"
    assert choose([head], error=1)["profile"] == "q4_q6"
    unknown = choose([head], support=False)
    assert unknown["profile"] == "q4_q6" and unknown["scores"] == []
    with pytest.raises(ValueError):
        choose([head], eligible=[])


def test_fit_is_reproducible_and_uses_only_training_groups():
    times = {1: [40, 30, 27], 2: [48, 33, 27], 32: [40, 30, 27]}
    first = fit_head(times, [1, 2], 7)
    other = fit_head({**times, 32: [1, 500, 900]}, [1, 2], 7)
    assert all(np.array_equal(a, b) for a, b in zip(first, other, strict=True))
