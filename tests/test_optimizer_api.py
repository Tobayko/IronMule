from __future__ import annotations

import friday_optimizer as api


def test_public_api_exports_new_control_plane_types() -> None:
    expected = {
        "DoctorReport",
        "HistoryReader",
        "HistoryWriter",
        "ImportReport",
        "OptimizerConfig",
        "OptimizerOrchestrator",
        "OptimizerStatus",
        "SessionEvent",
        "ShadowRequest",
    }
    assert expected.issubset(set(api.__all__))
    for name in expected:
        assert getattr(api, name)
