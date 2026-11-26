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
