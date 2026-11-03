"""AG4 step 2's simulation (training/ag4_routing.py): three policies on the same arrivals, priced by a measurement."""

from __future__ import annotations

from training import ag4_routing as sim


def test_the_router_is_at_least_as_good_as_the_better_path_under_load():
    result = sim.run([3.0], traces=6, lengths=[60, 120, 200])
    row = result["3.0"]
    assert row["router"]["tokens_per_s"] >= 0.98 * max(row["single"]["tokens_per_s"], row["stacked"]["tokens_per_s"])
    assert row["router"]["ttft_mean_s"] <= row["single"]["ttft_mean_s"]
    assert 0 < row["router"]["stacked_share"] < 1
    assert row["router_status"]["decision_us"] < 1000


def test_one_request_alone_takes_the_single_path():
    router = sim.PathRouter.from_setup(sim.setup_of(sim.MEASURED))
    done = sim.simulate([{"arrive": 0.0, "tokens": 100}], "router", sim.MEASURED, router)
    assert done[0]["path"] == 0
    assert done[0]["end"] - done[0]["arrive"] < 100 * sim.MEASURED["stacked_step_s"][1]
