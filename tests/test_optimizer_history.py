from __future__ import annotations

from pathlib import Path

import pytest

from friday_optimizer.history import HISTORY_KIND_PHASE, HistoryReader, HistoryWriter, SessionEvent
from friday_optimizer.memory import OptimizationMemoryV2


def test_history_is_immutable_idempotent_and_redacted(tmp_path: Path) -> None:
    memory = OptimizationMemoryV2(tmp_path / "memory.sqlite3")
    event = SessionEvent(
        kind="system",
        session_id="s1",
        state="started",
        payload={"prompt": "never persist", "log": "never persist", "ok": True},
    )
    writer = HistoryWriter(memory)
    writer.append(event)
    writer.append(event)
    assert memory.integrity().rows == 1
    loaded = HistoryReader(memory).recent()
    assert len(loaded) == 1
    assert "prompt" not in loaded[0].payload
    assert "log" not in loaded[0].payload
    with pytest.raises(TypeError):
        loaded[0].payload["x"] = 1  # type: ignore[index]


def test_history_rejects_invalid_hash_and_payload_bounds() -> None:
    with pytest.raises(ValueError):
        SessionEvent(kind="benchmark", evidence_hash="not-a-hash")
    with pytest.raises(ValueError):
        SessionEvent(kind="benchmark", payload={"data": "x" * (256 * 1024)})


def test_history_kind_phase_map_is_explicit_and_immutable() -> None:
    assert HISTORY_KIND_PHASE["benchmark"].value == "label"
    with pytest.raises(TypeError):
        HISTORY_KIND_PHASE["system"] = "label"  # type: ignore[index]
