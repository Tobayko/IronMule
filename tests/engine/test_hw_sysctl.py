"""P1: the hardware fingerprint reads its sysctl values without starting a process."""

from __future__ import annotations

import platform
import subprocess

import pytest

from ironmule import hw

KEYS = ("machdep.cpu.brand_string", "hw.logicalcpu", "hw.perflevel0.logicalcpu",
        "hw.perflevel1.logicalcpu", "hw.memsize")


@pytest.mark.skipif(platform.system() != "Darwin", reason="sysctlbyname is the macOS path")
def test_sysctl_values_match_the_command_and_start_no_process(monkeypatch):
    printed = {key: subprocess.run(["sysctl", "-n", key], capture_output=True,
                                   text=True).stdout.strip() or None for key in KEYS}

    def refuse(*_args, **_kwargs):
        raise AssertionError("_sysctl started a process")

    monkeypatch.setattr(hw.subprocess, "run", refuse)
    assert {key: hw._sysctl(key) for key in KEYS} == printed
    assert hw._sysctl("no.such.key") is None


def test_sysctl_is_silent_off_macos(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    assert hw._sysctl("hw.memsize") is None
