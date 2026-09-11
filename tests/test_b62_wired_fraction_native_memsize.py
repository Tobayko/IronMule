"""`wired_fraction` must reach the installed memory size without starting a process.

`B60` proved the defect: `Engine.__init__` read the size through `static_facts()`, which
shells out, and the Q3f child guard blocks `subprocess.Popen`, so the knob raised
`GuardViolation` in every confirmation child and could never enter a profile. `B62` reads
`hw.memsize` through `sysctlbyname` instead, exactly as `swap_used_bytes` has since `B55`.
"""

from __future__ import annotations

import subprocess

import pytest

from ironmule import hw
from ironmule.runtime import BASELINE, Engine, Knobs


@pytest.fixture(autouse=True)
def _restore_wired_owners():
    """These tests build engines they never close; the owner stack must not leak."""
    from ironmule import runtime

    saved = list(runtime._WIRED_LIMIT_OWNERS)
    yield
    runtime._WIRED_LIMIT_OWNERS[:] = saved


def test_native_memsize_equals_the_shell_reading():
    native = hw.installed_memory_bytes()
    shell = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    assert native == int(shell.stdout.strip())


def test_static_facts_still_reports_the_same_number():
    """The parent-side fingerprint path is untouched and must stay in agreement."""
    assert hw.installed_memory_bytes() == hw.static_facts()["memory_bytes"]


def test_the_wired_branch_starts_no_subprocess(monkeypatch):
    started = []
    real = subprocess.Popen

    def watched(*args, **kwargs):
        started.append(args[0] if args else kwargs.get("args"))
        return real(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", watched)
    applied = []
    monkeypatch.setattr("ironmule.runtime.mx.set_wired_limit",
                        lambda value: applied.append(value) or 0)
    Engine(object(), object(), Knobs(wired_fraction=0.6))
    assert started == [], f"the wired branch shelled out: {started}"
    assert applied == [int(hw.installed_memory_bytes() * 0.6)]


def test_zero_leaves_the_limit_alone(monkeypatch):
    touched = []
    monkeypatch.setattr("ironmule.runtime.mx.set_wired_limit",
                        lambda value: touched.append(value) or 0)
    engine = Engine(object(), object(), BASELINE)
    assert BASELINE.wired_fraction == 0.0
    assert touched == []
    engine.close()


def test_an_unavailable_size_fails_closed(monkeypatch):
    """No total means no limit and no silent zero: the engine refuses to be built."""
    touched = []
    monkeypatch.setattr("ironmule.runtime.mx.set_wired_limit",
                        lambda value: touched.append(value) or 0)
    monkeypatch.setattr(hw, "installed_memory_bytes", lambda: None)
    with pytest.raises(RuntimeError, match="installed memory size is unavailable"):
        Engine(object(), object(), Knobs(wired_fraction=0.6))
    assert touched == [], "a failed read must not change the process-global limit"


def test_a_native_error_is_reported_as_unavailable(monkeypatch):
    class Broken:
        def sysctlbyname(self, *args, **kwargs):
            raise OSError("no such name")

    monkeypatch.setattr(hw, "_libc", lambda: Broken())
    assert hw.installed_memory_bytes() is None


def test_the_meaning_of_the_knob_is_unchanged(monkeypatch):
    applied = []
    monkeypatch.setattr("ironmule.runtime.mx.set_wired_limit",
                        lambda value: applied.append(value) or 0)
    for fraction in (0.25, 0.6, 1.0):
        Engine(object(), object(), Knobs(wired_fraction=fraction))
    total = hw.installed_memory_bytes()
    assert applied == [int(total * 0.25), int(total * 0.6), int(total * 1.0)]
