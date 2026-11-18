import json
import subprocess

import pytest

from ironmule import ggml_gate_diagnostics as diag


@pytest.mark.parametrize("last_line,code,exc", [
    ("MemoryError", "out_of_memory", "MemoryError"),
    ("torch.OutOfMemoryError: CUDA_ERROR_OUT_OF_MEMORY", "out_of_memory", "OutOfMemoryError"),
    ("OSError: libggml-cuda.so: cannot open shared object file", "missing_cuda_library", "OSError"),
    ("ImportError: libllama.so: cannot open shared object file", "missing_cpu_library", "ImportError"),
    ("ModuleNotFoundError: No module named 'secret'", "missing_dependency", "ModuleNotFoundError"),
    ("ValueError: CUDA error: invalid configuration argument", "invalid_argument", "ValueError"),
    ("RuntimeError: CUDA error: no kernel image is available", "cuda_kernel_error", "RuntimeError"),
])
def test_classifies_child_failures_without_serializing_native_output(last_line, code, exc):
    result = diag.failed_child("tuning", "/secret/model\n" + last_line, 1)
    assert (result["code"], result["exit_code"], result["exception_class"]) == (code, 1, exc)
    assert "secret" not in json.dumps(result)


def test_timeout_and_signal_are_distinct_and_do_not_guess_oom():
    assert diag.failed_child("reference", "", -9)["code"] == "child_signal"
    result = diag.failed_child("candidate", "", exception=subprocess.TimeoutExpired("secret", 1))
    assert result["code"] == "timeout" and result["exception_class"] == "TimeoutExpired"
    assert "secret" not in json.dumps(result)


def test_public_sanitizer_bounds_candidates_and_excludes_unexpected_content():
    row = {"settings": {"flash_attn": True, "secret": True}, "role": "option", "passes": False,
           "nll_upper": float("nan"), "bound": float("inf"), "kg5_launches": -1,
           "factual": {"secret": 2, "facts": {"total": 2, "reference_correct": 3,
                                              "repeat_correct": 0, "candidate_correct": 0}}}
    result = diag.sanitize({"schema": diag.SCHEMA, "stage": "acceptance", "code": "quality_rejected",
                            "exit_code": 512, "exception_class": "secret", "candidates": [row] * 20, "secret": True})
    assert len(result["candidates"]) == 12
    assert result["exception_class"] is None and result["exit_code"] is None
    assert all(c["factual"] == {} and c["nll_upper"] is None and c["bound"] is None for c in result["candidates"])
    assert "secret" not in json.dumps(result, allow_nan=False)
    assert diag.sanitize({**result, "stage": "/private"}) is None
