import pytest

from lab.pk1.calibration_stability_replay import measured_return


def test_all_padding_prices_must_generate_the_claimed_runtime_table():
    arm = {"binding": {"max_rows": 3}, "startup_prices": {"padding": {"1": 3, "2": 3}}}
    raw = {
        "return_value": {"1": 3, "2": 3},
        "direct_field_values": {"pad_prices": {"1": 1.0, "2": 1.0, "3": 0.91}},
    }
    assert measured_return("padding", raw, arm) == arm["startup_prices"]["padding"]
    raw["direct_field_values"]["pad_prices"]["3"] = 1.0
    with pytest.raises(ValueError, match="different padding"):
        measured_return("padding", raw, arm)
    del raw["direct_field_values"]["pad_prices"]["2"]
    with pytest.raises(ValueError, match="missing complete"):
        measured_return("padding", raw, arm)


def test_pass_costs_include_the_complete_real_runtime_object():
    points = [[1, 0.01], [8, 0.02]]
    arm = {"startup_prices": {"passes": points}}
    raw = {"return_value": {"object_type": "PassCosts", "fields": {"points": points}}}
    assert measured_return("passes", raw, arm) == points
    raw["return_value"]["fields"]["new_unpriced_state"] = 10
    with pytest.raises(ValueError, match="incomplete"):
        measured_return("passes", raw, arm)
