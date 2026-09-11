"""Pure control-path coverage for the DATA1 local supervisor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from friday_evidence.portable.supervisor import SupervisorError, run_local_worker


def test_unknown_module_is_rejected_before_readiness_or_child_start(tmp_path: Path) -> None:
    output = tmp_path / "output"
    state = tmp_path / "state"
    before = set(sys.modules)

    result = run_local_worker("not.an.approved.module", [], output, timeout_seconds=1, state_dir=state)

    assert result["status"] == "rejected"
    assert result["error_code"] == "module_not_allowed"
    persisted = json.loads((output / "supervisor.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "rejected"
    assert not {"mlx.core", "torch", "jax"} & (set(sys.modules) - before)


def test_unapproved_portable_name_is_rejected_without_worker(tmp_path: Path) -> None:
    result = run_local_worker(
        "friday_evidence.portable.not_a_worker", [], tmp_path / "output",
        timeout_seconds=1, state_dir=tmp_path / "state",
    )

    assert result["status"] == "rejected"
    assert result["error_code"] == "module_not_allowed"


def test_invalid_argv_is_refused_before_creating_control_files(tmp_path: Path) -> None:
    with pytest.raises(SupervisorError, match="argv_invalid"):
        run_local_worker(
            "friday_evidence.portable.runner", ["\x00"], tmp_path / "output",
            timeout_seconds=1, state_dir=tmp_path / "state",
        )

    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "state").exists()
