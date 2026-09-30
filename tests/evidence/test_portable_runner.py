"""Control-plane tests for DATA1 portable execution (no accelerator execution)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from friday_evidence.portable.contracts import make_spec
from friday_evidence.portable.backends import _stable_mlx_hardware
from friday_evidence.portable.runner import _DirectEventSink, RunnerError, _drain_events, _load_operands, main, run_experiment


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path: Path) -> dict[str, object]:
    left = tmp_path / "left.npy"
    right = tmp_path / "right.npy"
    np.save(left, np.eye(2, dtype=np.float32), allow_pickle=False)
    np.save(right, np.eye(2, dtype=np.float32), allow_pickle=False)
    return {
        "case_id": "case-1", "a_file": left.name, "b_file": right.name,
        "a_sha256": _digest(left), "b_sha256": _digest(right), "shape": [2, 2, 2],
        "dtype": "float32", "lineage_id": "lineage-1", "weight_sha256": "1" * 64,
        "prompt_family": "prompt-1", "shape_family": "tiny-2", "capture_session": "capture-1",
        "model_id": "org/model", "model_sha256": "2" * 64,
        "source_kind": "real_model_capture", "partition": "train",
    }


def test_runner_import_does_not_import_accelerator_frameworks() -> None:
    # Importing the control API must not make MLX, CUDA, or TPU work possible.
    before = set(sys.modules)
    __import__("friday_evidence.portable.runner")
    introduced = set(sys.modules) - before
    assert "mlx.core" not in introduced
    assert "torch" not in introduced
    assert "jax" not in introduced


def test_hash_checked_numpy_capture_is_loaded_readonly(tmp_path: Path) -> None:
    case = _case(tmp_path)
    left, right, setup_seconds = _load_operands(case, tmp_path)

    assert left.dtype == np.float32 and right.dtype == np.float32
    assert not left.flags.writeable and not right.flags.writeable
    assert setup_seconds >= 0


def test_hash_mismatch_is_rejected_before_accelerator_import(tmp_path: Path) -> None:
    case = _case(tmp_path)
    case["a_sha256"] = "f" * 64
    before = set(sys.modules)

    with pytest.raises(RunnerError, match="SHA-256"):
        _load_operands(case, tmp_path)

    introduced = set(sys.modules) - before
    assert not {"mlx.core", "torch", "jax"} & introduced


def test_sealed_code_mismatch_refuses_execution_without_creating_output(tmp_path: Path) -> None:
    spec = make_spec([_case(tmp_path)], "mlx")
    spec["code_sha256"] = "a" * 64
    output = tmp_path / "output"
    before = set(sys.modules)

    with pytest.raises(RunnerError, match="code_sha256"):
        run_experiment(spec, tmp_path, output)

    assert not output.exists()
    introduced = set(sys.modules) - before
    assert not {"mlx.core", "torch", "jax"} & introduced


def test_cli_rejects_invalid_spec_without_execution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec = tmp_path / "bad.json"
    spec.write_text('{"schema":"wrong"}', encoding="utf-8")

    code = main(["--spec", str(spec), "--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "output")])

    assert code == 2
    assert "refused input" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_stable_mlx_identity_excludes_live_allocation_counters() -> None:
    stable = _stable_mlx_hardware({
        "chip_name": "Apple M4 Max", "memory_size": 137_438_953_472,
        "active_memory_bytes": 11, "peak_memory_bytes": 29,
    })

    assert stable == {
        "backend": "mlx", "accelerator": "Apple M4 Max", "driver": stable["driver"],
        "memory": {"total_memory_bytes": 137_438_953_472},
    }
    assert "active_memory_bytes" not in json.dumps(stable, sort_keys=True)


def test_direct_event_sink_persists_each_event_without_drain_duplication(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    event_path.touch()
    sink = _DirectEventSink(event_path)

    sink.put({"kind": "trial", "trial": {"case_id": "case-1", "status": "censored"}})

    assert _drain_events(sink, event_path) == []
    rows = event_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["kind"] == "trial"
def test_aa_resolution_is_fixed_before_candidate_measurements():
    from friday_evidence.portable.runner import aa_noise_gate

    def summary(low, high):
        return {"statistics": {"paired": {"ci_low": low, "ci_high": high}}}

    assert aa_noise_gate(summary(0.999, 1.001), 12)["passed"] is True
    noisy = aa_noise_gate(summary(0.96, 1.04), 12)
    assert noisy["passed"] is False
    assert noisy["suggested_pairs"] == 31
    assert noisy["automatic_retry"] is False
