import copy

import pytest

from lab.pk1 import sampling_eval as evaluation


def test_observer_rejects_stale_prices_and_unresolved_interval():
    state = {
        "prices": {"1": 0.02, "2": 0.025},
        "dispatch": {},
        "calls": {"1": 4, "2": 4},
        "native_seconds": {
            "1": [0.026, 0.022, 0.02, 0.02],
            "2": [0.024, 0.021, 0.025, 0.025],
        },
    }
    evaluation.actual_samples(state, 2, adaptive=True)
    wrong = copy.deepcopy(state)
    wrong["prices"]["1"] = 0.022
    with pytest.raises(ValueError, match="median"):
        evaluation.actual_samples(wrong, 2, adaptive=True)
    wrong = copy.deepcopy(state)
    wrong["prices"]["2"] = 0.021
    wrong["native_seconds"]["2"][-2:] = [0.018, 0.024]
    with pytest.raises(ValueError, match="uncertain"):
        evaluation.actual_samples(wrong, 2, adaptive=True)


def test_discrete_schedule_alone_cannot_qualify_measured_price_error():
    def arm(prices):
        return {
            "sampling": {"observed": {"startup": {"prices": prices, "dispatch": {}}}}
        }

    a = arm({"1": 0.02, "2": 0.025})
    assert evaluation.compare(a, a) == 0
    with pytest.raises(ValueError, match="two percent"):
        evaluation.compare(a, arm({"1": 0.023, "2": 0.025}))
