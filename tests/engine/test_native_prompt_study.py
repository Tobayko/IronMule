import json
from pathlib import Path

import pytest

from lab.pk6.native_dispatch.live_policy import choose
from lab.pk6.native_dispatch.score import exclusion, wave_energy
from lab.pk6.native_dispatch.tile_order import order
from training.code_author import read_request


def test_native_order_is_reproducible_from_only_counted_requests(tmp_path):
    root = (
        Path(__file__).resolve().parents[2]
        / "lab/autonomous/results/l40s-native-dispatch"
    )
    result = order(root / "http/primary", tmp_path)
    archived_id, original = read_request(root / "tile-order/author")
    request_id, replay = read_request(tmp_path / "author")
    assert result["status"] == "awaiting_codex" and archived_id == request_id
    assert replay == original and replay["order"]["target"]["ggml_type"] == 12


def test_power_estimate_requires_two_bracketing_samples_and_rejects_an_outage():
    wave = {"unix_start": 1.25, "unix_end": 1.75}
    assert wave_energy([(1.0, 100.0, 20.0), (2.0, 100.0, 40.0)], wave) == pytest.approx(
        15.0
    )
    assert wave_energy([(1.5, 100.0, 20.0), (2.0, 100.0, 40.0)], wave) is None
    assert wave_energy([(1.0, 100.0, 20.0), (5.0, 100.0, 40.0)], wave) is None


def test_observed_kernel_fallback_is_ineligible_even_with_matching_outputs():
    assert exclusion(
        "F1",
        {
            "prompt_plugin_dispatches_in_http_by_width": {"32": 100},
            "prompt_plugin_dispatches_in_http_quality": 0,
        },
    )
    assert exclusion("N1", {"shared_pool": True, "streams_pass": False})
    assert exclusion("P2", {})


def test_foreign_pilot_scope_falls_back_before_loading_any_weights(tmp_path):
    from lab.pk6.warm_repeat import GGUF

    (tmp_path / "fit.json").write_text(
        json.dumps(
            {
                "target": {
                    "hardware": {"name": "original"},
                    "gguf": GGUF,
                    "ggml_cuda": "one",
                }
            }
        )
    )
    assert choose(tmp_path, {"name": "other"}, "one")["profile"] == "stock"


def test_grid_keeps_only_cells_confirmed_on_an_independent_seed(tmp_path):
    from lab.pk6.native_dispatch.grid_screen import MODES, SEEDS, WIDTHS, select

    protocol = {
        "hardware": "local",
        "reference_cuda": "native",
        "gguf": "weights",
        "backend": "isolated",
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    for n in range(1, 6):
        for seed in SEEDS:
            cells = {}
            for b in WIDTHS:
                entries = {
                    m: {"us": 12, "graphs": 8, "dispatches": 8, "relative_rms": 0}
                    for m in MODES
                }
                entries["stock"]["us"] = 10
                entries["xy"]["us"] = 6 if (n == 1 or seed == SEEDS[0]) else 11
                if b != 448:
                    entries["xy"]["dispatches"] = 0
                cells[str(b)] = entries
            (tmp_path / f"shape-{n}-{seed}.json").write_text(
                json.dumps({"n": n, "k": 32, "seed": seed, "cells": cells})
            )
    assert select(tmp_path)["ranges"] == [[1, 32, 448, 448, 1]]
    (tmp_path / f"shape-1-{SEEDS[1]}.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        select(tmp_path)


def test_compact_capacity_is_checked_against_actual_worker_allocation():
    assert exclusion("H1", {"verified_compact_pool": False})
    assert exclusion("H1", {"verified_compact_pool": True}) is None
