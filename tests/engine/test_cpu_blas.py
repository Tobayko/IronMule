"""CPU6 step 1: MLX's float32 products on an optimised BLAS through a preloaded shim (`ironmule/cpu_blas.py`).

The shim is built with this machine's compiler and checked in child processes: forwarded, it must serve the
optimised BLAS and agree with a float64 reference; with a target it cannot resolve, it must fall back to MLX's own
BLAS and keep working. The worker client and the service report what serves.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ironmule import cpu_blas

ROOT = Path(__file__).resolve().parents[2]


def _available(tmp_path):
    variables, record = cpu_blas.environment(root=tmp_path / "cache")
    if not variables:
        pytest.skip(f"OpenBLAS shim unavailable here: {record.get('reason')}")
    return variables, record


def test_the_shim_serves_openblas_and_agrees_with_a_float64_reference(tmp_path):
    variables, record = _available(tmp_path)
    assert record["available"] is True and record["check_rel_error"] <= cpu_blas.CHECK_TOLERANCE
    assert set(variables) >= {"LD_PRELOAD", "IRONMULE_CPU_BLAS_LIBRARY", "IRONMULE_CPU_BLAS_SYMBOL",
                              "IRONMULE_CPU_BLAS_ILP64", "IRONMULE_CPU_BLAS_FALLBACK"}
    checked = cpu_blas.check(variables)
    assert checked["ok"] and checked["serving"] == "openblas"


@pytest.mark.parametrize("broken", [{"IRONMULE_CPU_BLAS_LIBRARY": "/nonexistent/libopenblas.so"},
                                    {"IRONMULE_CPU_BLAS_SYMBOL": "no_such_symbol"}])
def test_a_target_it_cannot_resolve_falls_back_to_mlx_s_own_blas_and_keeps_working(tmp_path, broken):
    variables, _ = _available(tmp_path)
    checked = cpu_blas.check({**variables, **broken})
    assert checked["ok"] is False and checked["serving"] == "stock"
    assert checked["rel_error"] <= cpu_blas.CHECK_TOLERANCE  # the stock BLAS answered


def test_the_build_is_cached_by_its_source(tmp_path):
    first, reason = cpu_blas.build(root=tmp_path)
    if first is None:
        pytest.skip(reason)
    again, _ = cpu_blas.build(root=tmp_path)
    assert again == first and first.parent.name in __import__("hashlib").sha256(
        cpu_blas.SHIM_SOURCE.encode()).hexdigest()


def test_without_a_compiler_or_off_linux_x86_the_worker_keeps_the_stock_blas(tmp_path, monkeypatch):
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(cpu_blas.shutil, "which", lambda name: None)
    variables, record = cpu_blas.environment(root=tmp_path / "none")
    assert variables == {} and record["available"] is False and "no C compiler" in record["reason"]
    monkeypatch.setattr(cpu_blas.sys, "platform", "darwin")
    assert "Linux" in cpu_blas.supported()
    variables, record = cpu_blas.environment(root=tmp_path / "none")
    assert variables == {} and "Linux" in record["reason"]


def test_a_process_without_the_shim_reports_the_stock_blas():
    assert cpu_blas.active() == {"loaded": False, "active": False, "serving": "stock"}


def test_the_thread_cap_reaches_the_worker_and_must_be_positive(tmp_path):
    variables, record = _available(tmp_path)
    capped, record = cpu_blas.environment(threads=2, root=tmp_path / "cache")
    assert capped["OPENBLAS_NUM_THREADS"] == "2" and record["threads"] == 2
    with pytest.raises(ValueError):
        cpu_blas.environment(threads=0, root=tmp_path / "cache")


def test_the_worker_client_validates_its_blas_options():
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.types import ModelSpec

    spec = ModelSpec("local/test", "revision", "/tmp/none", 1)
    with pytest.raises(ValueError, match="cpu_blas must be"):
        MLXWorkerClient(spec, cpu_blas="mkl")
    with pytest.raises(ValueError, match="CPU worker"):
        MLXWorkerClient(spec, cpu_blas="openblas", cuda_device="0")
    with pytest.raises(ValueError, match="cpu_threads"):
        MLXWorkerClient(spec, cpu_threads=2)
    client = MLXWorkerClient(spec, cpu_blas="openblas", cpu_threads=2)
    assert (client.cpu_blas, client.cpu_threads, client.cpu_blas_record) == ("openblas", 2, None)


def test_the_worker_reports_the_serving_blas_only_when_the_shim_was_asked_for(monkeypatch):
    from ironmule_product import worker

    monkeypatch.delenv("IRONMULE_CPU_BLAS_LIBRARY", raising=False)
    assert worker._cpu_blas_status() == {}
    monkeypatch.setenv("IRONMULE_CPU_BLAS_LIBRARY", "/x/libopenblas.so")
    assert worker._cpu_blas_status()["cpu_blas"]["serving"] == "stock"  # this test process has no shim


def test_serve_refuses_the_blas_option_with_gpus(tmp_path, capsys):
    from ironmule_product import cli

    with pytest.raises(SystemExit):
        cli.serve(["--model", "local/test", "--gpus", "all", "--cpu-blas", "openblas", "--state-dir",
                   str(tmp_path / "state")])
    assert "--cpu-blas applies to CPU workers" in capsys.readouterr().err


def test_health_says_which_blas_serves_and_why_not(tmp_path):
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore

    class Backend:
        device, cpu_blas, ready = "cpu", "openblas", True
        cpu_blas_record = {"available": False, "reason": "no C compiler (cc or gcc) to build the CPU BLAS shim"}
        ready_payload = {"cpu_blas": {"serving": "stock"}}

        def close(self):
            pass

    store = ProductStore(tmp_path / "state")
    store.setup("server")
    service = ProductService(store, backend=Backend(), spec=None)
    try:
        assert service.health()["cpu_blas"] == {"requested": "openblas", "serving": "stock",
                                                "reason": "no C compiler (cc or gcc) to build the CPU BLAS shim"}
        Backend.cpu_blas, Backend.ready_payload = None, {}
        assert "cpu_blas" not in service.health()
    finally:
        service.close()
