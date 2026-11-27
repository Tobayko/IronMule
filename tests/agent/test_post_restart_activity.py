from types import SimpleNamespace

import pytest

from lab.pk1 import post_restart_activity as activity


def test_unavailable_or_broken_profiler_never_reissues_call(monkeypatch):
    calls = []
    identities = []
    class BadExit:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            raise OSError("unavailable")

    def alternate(summary):
        identities.append(summary["sample"])
        if len(identities) == 1:
            raise ValueError("unsupported")
        return BadExit(), lambda: None

    monkeypatch.setattr(activity, "setup", alternate)
    probe = activity.Activity()
    for x in range(3):
        result, row = probe.measure(lambda i: calls.append(i) or i, x)
        assert result == x
        assert row["status"] == ("unsupported" if x < 2 else "sample_limit")
    assert calls == [0, 1, 2] and identities == [1, 2]

    def failed():
        raise RuntimeError("original model error")

    with pytest.raises(RuntimeError, match="original model error"):
        activity.Activity().measure(failed)


def test_measured_symbols_sum_only_device_events():
    def event(kind, name, us):
        return SimpleNamespace(device_type=SimpleNamespace(name=kind), name=name, device_time_total=us)
    prof = SimpleNamespace(events=lambda: [event("CPU", "dispatch", 30),event("CUDA", "kernel", 3),event("CUDA", "kernel", 5)])
    result = activity.aggregate(prof)
    assert result["device_events"] == 2 and result["summed_device_s"] == pytest.approx(8e-6)
    assert result["events"][0]["symbol"] == "kernel" and result["events"][0]["calls"] == 2
