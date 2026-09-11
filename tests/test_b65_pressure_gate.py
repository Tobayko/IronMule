"""The qualified resource gate must fire on every criterion it claims, and on missing data.

`B65` replaced "any system-wide swapout blocks" with a composite that describes the run
instead of the machine. These tests exercise the native readers and the gate's arithmetic;
the separation evidence lives in `research/raw/B65_gate_semantics_20260910_final.json`.
"""

from __future__ import annotations

import re
import subprocess

from pathlib import Path

from ironmule import hw


def test_native_counters_bracket_the_shell_reading():
    """Monotone counters move between two reads, so the shell value must land inside them."""
    before = hw.vm_counters()
    text = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    after = hw.vm_counters()
    assert before and after
    for name, pattern in (("swapouts", r"Swapouts:\s*([0-9]+)"),
                          ("swapins", r"Swapins:\s*([0-9]+)"),
                          ("pageouts", r"Pageouts:\s*([0-9]+)")):
        shell = int(re.search(pattern, text).group(1))
        assert before[name] <= shell <= after[name], name


def test_pressure_level_is_readable_and_named():
    level = hw.memory_pressure_level()
    assert level in hw.MEMORY_PRESSURE_NAMES
    assert hw.MEMORY_PRESSURE_NAMES[hw.MEMORY_PRESSURE_NORMAL] == "normal"


def test_the_native_readers_start_no_subprocess(monkeypatch):
    started = []
    real = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: (started.append(a), real(*a, **k))[1])
    hw.vm_counters()
    hw.memory_pressure_level()
    assert started == []


def _harness_gate():
    """The real thing from `tools/b57_stack_composition.py`, not a copy of it."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "b57_gate", root / "tools" / "b57_stack_composition.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_resource_gate


def _gate(samples, *, rss_limit=None, total=34_359_738_368):
    for row in samples:
        row.setdefault("swapouts", 26_626_778)
    return _harness_gate()(samples, gate="pressure", memory_total=total,
                           peak_rss=rss_limit if rss_limit is not None else 0)["passed"]


def _legacy(samples, *, total=34_359_738_368):
    return _harness_gate()(samples, gate="legacy", memory_total=total, peak_rss=0)["passed"]


def _sample(free=50.0, swap=10_000, level=hw.MEMORY_PRESSURE_NORMAL):
    return {"memory_free_percent": free, "swap_used_bytes": swap,
            "memory_pressure_level": level}


def test_a_quiet_run_passes():
    assert _gate([_sample(swap=10_000), _sample(swap=9_500), _sample(swap=9_000)])


def test_swapouts_alone_no_longer_block():
    """The system-wide counter is recorded, not decisive: this is the whole change."""
    samples = [_sample(swap=10_000), _sample(swap=9_000)]
    for row, count in zip(samples, (26_626_778, 26_675_466)):
        row["swapouts"] = count
    assert _gate(list(samples))
    assert not _legacy(list(samples)), "the legacy gate must still block, unchanged"


def test_growing_swap_blocks():
    assert not _gate([_sample(swap=9_000), _sample(swap=24_000), _sample(swap=24_000)])


def test_a_spike_that_recovers_still_blocks():
    """The budget is on the maximum, not on the endpoint: a spike is not undone by recovery."""
    assert not _gate([_sample(swap=9_000), _sample(swap=24_000), _sample(swap=8_000)])


def test_abnormal_pressure_blocks():
    assert not _gate([_sample(), _sample(level=2), _sample()])
    assert not _gate([_sample(), _sample(level=4), _sample()])


def test_an_unreadable_pressure_level_blocks():
    assert not _gate([_sample(), _sample(level=None)])


def test_low_free_memory_still_blocks():
    assert not _gate([_sample(free=50.0), _sample(free=8.0)])


def test_a_missing_swap_probe_blocks():
    assert not _gate([{"memory_free_percent": 50.0, "swap_used_bytes": None,
                       "memory_pressure_level": hw.MEMORY_PRESSURE_NORMAL}])


def test_high_rss_still_blocks():
    assert not _gate([_sample()], rss_limit=int(34_359_738_368 * 0.61))
