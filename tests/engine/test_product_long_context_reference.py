"""Pure control contracts for PROD8; no native model path is exercised."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import product_long_context_reference as screen  # noqa: E402


def _metadata() -> dict:
    return {
        "schema": "ironmule.prod8_stock_reference.v1", "prompt_tokens": 1024,
        "prompt_ids_sha256": "c" * 64, "state": "ready", "device": "Device(gpu, 0)",
        "mlx_active_bytes": 1, "mlx_peak_bytes": 1,
    }


def _complete_stock_result() -> dict:
    result = _metadata()
    result["samples"] = [{
        "output_sha256": "a" * 64, "text_sha256": "b" * 64,
        "finish_reason": "length", "prompt_tokens": 1024, "completion_tokens": 8,
        "phase": "warmup" if index == 0 else "recorded", "sample_index": index,
        "parent_elapsed_seconds": 1.0,
    } for index in range(screen.REPEATS + 1)]
    return result


def test_prompt_is_deterministic_and_has_a_known_public_end_marker():
    assert screen._message() == screen._message()  # noqa: SLF001
    assert screen._message()[-1][-1].endswith("END-OF-PUBLIC-ORCHARD-NOTE.")  # noqa: SLF001


def test_stock_metadata_requires_fixed_context_contract():
    assert screen._validate_stock_metadata(_metadata())["prompt_tokens"] == 1024  # noqa: SLF001
    invalid = _metadata()
    invalid["state"] = "other"
    with pytest.raises(screen.LongContextFailure, match="stock_reference_invalid"):
        screen._validate_stock_metadata(invalid)  # noqa: SLF001


def test_stock_metadata_rejects_context_range_drift_without_native_work():
    invalid = _metadata()
    invalid["prompt_tokens"] = screen.MAX_PROMPT_TOKENS + 1
    with pytest.raises(screen.LongContextFailure, match="stock_reference_invalid"):
        screen._validate_stock_metadata(invalid)  # noqa: SLF001


def test_stock_metadata_rejects_non_hex_digest_and_non_gpu_device():
    for key, value in (("prompt_ids_sha256", "z" * 64), ("device", "Device(cpu, 0)")):
        invalid = _metadata()
        invalid[key] = value
        with pytest.raises(screen.LongContextFailure, match="stock_reference_invalid"):
            screen._validate_stock_metadata(invalid)  # noqa: SLF001


def test_frozen_calibration_export_passes_gate_and_binds_hashes():
    proof = screen._calibration_gate()  # noqa: SLF001
    assert proof["run_id"] == screen.CALIBRATION_RUN_ID
    assert len(proof["artifact_sha256"]) == 64


def test_calibration_gate_recomputes_canonical_report_hash(tmp_path: Path):
    value = json.loads(screen.CALIBRATION.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(value)
    tampered["report"]["resource_valid"] = False
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(screen.LongContextFailure, match="calibration_prerequisite_invalid"):
        screen._calibration_gate(path)  # noqa: SLF001


def test_stock_samples_require_exact_phase_order_and_finite_parent_timing():
    screen._validate_stock_samples(_complete_stock_result())  # noqa: SLF001
    invalid = _complete_stock_result()
    invalid["samples"][1]["phase"] = "warmup"
    with pytest.raises(screen.LongContextFailure, match="stock_reference_invalid"):
        screen._validate_stock_samples(invalid)  # noqa: SLF001
    invalid = _complete_stock_result()
    invalid["samples"][2]["parent_elapsed_seconds"] = float("nan")
    with pytest.raises(screen.LongContextFailure, match="stock_reference_invalid"):
        screen._validate_stock_samples(invalid)  # noqa: SLF001


def test_http_comparator_requires_exact_hashed_text_usage_and_finish():
    expected = {"text_sha256": "b" * 64, "finish_reason": "length", "prompt_tokens": 1024, "completion_tokens": 8}
    value = {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
             "usage": {"prompt_tokens": 1024, "completion_tokens": 8}}
    expected["text_sha256"] = screen._sha256(b"x")  # noqa: SLF001
    screen._assert_http(value, expected)  # noqa: SLF001
    value["usage"]["completion_tokens"] = 7
    with pytest.raises(screen.LongContextFailure, match="http_output_mismatch"):
        screen._assert_http(value, expected)  # noqa: SLF001
