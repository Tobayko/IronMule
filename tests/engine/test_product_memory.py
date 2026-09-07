"""Real-process memory guard tests; no model or GPU work is performed."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

import pytest

from ironmule_product.memory import (
    LoadMemoryGuard,
    MemoryGuardError,
    POLL_INTERVAL_SECONDS,
    RSS_LIMIT_FRACTION,
    SWAP_DELTA_LIMIT_BYTES,
)
from ironmule_product.readiness import _swap_used


def _ready(**changes):
    value = {
        "startup_wall_seconds": 0.5,
        "process_peak_rss_bytes": 100,
        "mlx_active_bytes": 200,
        "mlx_peak_bytes": 300,
        "mlx_cache_bytes": 100,
        "recommended_working_set_bytes": 400,
    }
    value.update(changes)
    return value


@pytest.mark.skipif(platform.system() != "Darwin", reason="native RSS/swap probe is macOS-specific")
def test_real_process_rss_sample_is_numeric_and_throttled():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
    samples = []
    try:
        baseline, error = _swap_used()
        if error is not None or type(baseline) is not int:
            pytest.skip("sandbox does not permit the real swap probe")
        guard = LoadMemoryGuard(
            swap_baseline_bytes=baseline,
            memory_total_bytes=16 * 1024**3,
            on_sample=samples.append,
        )
        try:
            first = guard(process.pid, force=True)
        except MemoryGuardError as exc:
            if exc.code == "rss_probe_failed":
                pytest.skip("sandbox does not permit the real /bin/ps probe")
            raise
        second = guard(process.pid)
        assert first["pid"] == process.pid and first["rss_bytes"] > 0
        assert first["finished_monotonic_ns"] >= first["started_monotonic_ns"]
        assert second == first
        assert len(samples) == 1
        time.sleep(POLL_INTERVAL_SECONDS)
        third = guard(process.pid)
        assert third["finished_unix_ns"] >= first["finished_unix_ns"]
        assert len(samples) == 2
    finally:
        process.terminate()
        process.wait(timeout=5)


@pytest.mark.skipif(platform.system() != "Darwin", reason="native RSS/swap probe is macOS-specific")
def test_rss_limit_records_observation_before_raising():
    samples = []
    baseline, error = _swap_used()
    if error is not None or type(baseline) is not int:
        pytest.skip("sandbox does not permit the real swap probe")
    guard = LoadMemoryGuard(swap_baseline_bytes=baseline, memory_total_bytes=1, on_sample=samples.append)
    try:
        guard(os.getpid(), force=True)
    except MemoryGuardError as exc:
        if exc.code == "rss_probe_failed":
            pytest.skip("sandbox does not permit the real /bin/ps probe")
        assert exc.code == "rss_limit_exceeded"
    else:
        pytest.fail("RSS limit should reject the current process")
    assert samples and samples[-1]["errors"] == ["rss_limit_exceeded"]
    assert samples[-1]["rss_bytes"] is not None


def test_invalid_pid_and_callback_contract_fail_closed():
    with pytest.raises(ValueError):
        LoadMemoryGuard(swap_baseline_bytes=-1, memory_total_bytes=1)
    with pytest.raises(ValueError):
        LoadMemoryGuard(swap_baseline_bytes=0, memory_total_bytes=0)
    guard = LoadMemoryGuard(swap_baseline_bytes=0, memory_total_bytes=1)
    for pid in (0, -1, True):
        with pytest.raises(MemoryGuardError, match="pid_invalid"):
            guard(pid)
    with pytest.raises(MemoryGuardError, match="force_invalid"):
        guard(os.getpid(), force=1)


@pytest.mark.skipif(platform.system() != "Darwin", reason="native RSS probe is macOS-specific")
def test_guard_binds_first_pid_and_latches_probe_failure():
    guard = LoadMemoryGuard(swap_baseline_bytes=0, memory_total_bytes=1)
    with pytest.raises(MemoryGuardError) as first:
        guard(os.getpid(), force=True)
    if first.value.code == "rss_probe_failed":
        pytest.skip("sandbox does not permit the real /bin/ps probe")
    with pytest.raises(MemoryGuardError, match="pid_mismatch"):
        guard(os.getpid() + 1, force=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("startup_wall_seconds", 0),
        ("startup_wall_seconds", True),
        ("process_peak_rss_bytes", 1.0),
        ("mlx_active_bytes", -1),
        ("mlx_peak_bytes", True),
        ("mlx_cache_bytes", None),
        ("recommended_working_set_bytes", "400"),
    ],
)
def test_ready_telemetry_is_flat_and_strict(field, value):
    guard = LoadMemoryGuard(swap_baseline_bytes=0, memory_total_bytes=1000)
    with pytest.raises(MemoryGuardError):
        guard.validate_ready(_ready(**{field: value}))


def test_ready_requires_exact_flat_schema():
    guard = LoadMemoryGuard(swap_baseline_bytes=0, memory_total_bytes=1000)
    with pytest.raises(MemoryGuardError):
        incomplete = _ready()
        incomplete.pop("mlx_cache_bytes")
        guard.validate_ready(incomplete)
    selected = LoadMemoryGuard(swap_baseline_bytes=0, memory_total_bytes=1000).validate_ready(
        {**_ready(), "transport_debug": "discarded"}
    )
    assert "transport_debug" not in selected


def test_ready_applies_separate_rss_and_mlx_gates_without_summing_overlap():
    guard = LoadMemoryGuard(swap_baseline_bytes=0, memory_total_bytes=1000)
    result = guard.validate_ready(_ready(process_peak_rss_bytes=500, mlx_peak_bytes=500, recommended_working_set_bytes=500))
    assert result["rss_limit_bytes"] == int(1000 * RSS_LIMIT_FRACTION)
    assert result["swap_delta_limit_bytes"] == SWAP_DELTA_LIMIT_BYTES
    with pytest.raises(MemoryGuardError, match="rss_peak_limit_exceeded"):
        guard.validate_ready(_ready(process_peak_rss_bytes=601))
    with pytest.raises(MemoryGuardError, match="mlx_peak_working_set_exceeded"):
        LoadMemoryGuard(swap_baseline_bytes=0, memory_total_bytes=1000).validate_ready(
            _ready(mlx_peak_bytes=401, recommended_working_set_bytes=400)
        )
    with pytest.raises(MemoryGuardError, match="mlx_peak_limit_exceeded"):
        LoadMemoryGuard(swap_baseline_bytes=0, memory_total_bytes=1000).validate_ready(
            _ready(mlx_peak_bytes=601, recommended_working_set_bytes=601)
        )
    with pytest.raises(MemoryGuardError, match="mlx_active_exceeds_peak"):
        LoadMemoryGuard(swap_baseline_bytes=0, memory_total_bytes=1000).validate_ready(
            _ready(mlx_active_bytes=301, mlx_peak_bytes=300)
        )


def test_ready_gate_failure_is_latched():
    guard = LoadMemoryGuard(swap_baseline_bytes=0, memory_total_bytes=1000)
    with pytest.raises(MemoryGuardError, match="rss_peak_limit_exceeded"):
        guard.validate_ready(_ready(process_peak_rss_bytes=601))
    with pytest.raises(MemoryGuardError, match="rss_peak_limit_exceeded"):
        guard.validate_ready(_ready())
