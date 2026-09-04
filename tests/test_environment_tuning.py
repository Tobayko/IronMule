"""Environment tuning must never abort server startup.

The call site in ``friday_serve.cli._prewarm_hardware`` runs before the model
loads; a missing MLX device API or a wrong memory constant used to raise
``UnboundLocalError`` and take the whole ``serve`` command down with it.
"""

from __future__ import annotations

import mlx.core as mx

from friday_serve import environment_tuning


def test_returns_a_dict_when_the_device_api_is_absent(monkeypatch) -> None:
    for name in ("set_cache_limit", "set_wired_limit", "device_info", "metal"):
        monkeypatch.delattr(mx, name, raising=False)

    info = environment_tuning.tune_runtime_environment()

    assert isinstance(info, dict)
    assert info["uma_gb"] > 0
    # wired limit never exceeds 70% of detected memory
    assert info["metal_wired_limit_gb"] <= 0.7 * info["uma_gb"] + 1e-6


def test_detect_uma_gb_is_positive() -> None:
    assert environment_tuning.detect_uma_gb() > 0
