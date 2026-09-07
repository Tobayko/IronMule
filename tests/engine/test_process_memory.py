"""Pure ABI contracts plus an optional real self-process Darwin sample."""

from __future__ import annotations

import ctypes
import os
import platform

import pytest

from friday_evidence import process_memory


def test_rusage_v4_layout_is_pinned() -> None:
    assert ctypes.sizeof(process_memory._RusageInfoV4) == 296
    assert process_memory._RusageInfoV4.ri_resident_size.offset == 64
    assert process_memory._RusageInfoV4.ri_phys_footprint.offset == 72
    assert process_memory._RusageInfoV4.ri_lifetime_max_phys_footprint.offset == 240
    assert process_memory._RusageInfoV4.ri_interval_max_phys_footprint.offset == 280


@pytest.mark.parametrize("pid", [0, -1, 2**31, True, "1", None])
def test_invalid_pid_fails_closed(pid: object) -> None:
    with pytest.raises(process_memory.ProcessMemoryError, match="pid_invalid"):
        process_memory.sample_process_memory(pid)  # type: ignore[arg-type]


@pytest.mark.skipif(platform.system() != "Darwin", reason="Darwin proc_pid_rusage only")
def test_real_self_process_sample() -> None:
    try:
        sample = process_memory.sample_process_memory(os.getpid())
    except process_memory.ProcessMemoryError as exc:
        pytest.skip(f"libproc unavailable: {exc.code}")
    assert set(sample) == {
        "resident_bytes", "physical_footprint_bytes", "lifetime_peak_footprint_bytes",
        "interval_peak_footprint_bytes", "wired_bytes", "process_start_abstime",
        "observed_monotonic_ns", "duration_ns",
    }
    assert all(type(value) is int and value >= 0 for value in sample.values())
    assert sample["duration_ns"] >= 0
