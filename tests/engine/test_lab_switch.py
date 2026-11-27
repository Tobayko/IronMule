"""The lab A/B switch file sets the named switches of the modules loaded in this process (diagnostics only)."""
import json
import time

import ironmule_product.http_server as hs
import ironmule_product.lab_switch as ls
import ironmule_product.serving_policy as sp


def test_the_switch_file_sets_the_named_switches_and_nothing_without_the_variable(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "_ADM1", True)
    monkeypatch.setattr(sp, "_ADM1", True)
    monkeypatch.setattr(sp, "_ADM2", True)
    monkeypatch.setattr(ls, "_started", False)
    monkeypatch.delenv("IRONMULE_LAB_SWITCH", raising=False)
    ls.start()
    assert ls._started is False  # no variable, no watcher
    path = tmp_path / "arms.json"
    path.write_text(json.dumps({"ADM1": 0}))
    monkeypatch.setenv("IRONMULE_LAB_SWITCH", str(path))
    ls.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and hs._ADM1:
        time.sleep(0.01)
    assert hs._ADM1 is False and sp._ADM1 is False and sp._ADM2 is True  # ADM2 not named: unchanged
    path.write_text(json.dumps({"ADM1": 1, "ADM2": 0}))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and sp._ADM2:
        time.sleep(0.01)
    assert hs._ADM1 is True and sp._ADM2 is False
    path.write_text(json.dumps({"ADM1": 1, "ADM2": 1}))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not sp._ADM2:
        time.sleep(0.01)
    ls.stop()
    time.sleep(0.1)
    monkeypatch.setattr(sp, "_ADM2", False)
    time.sleep(0.1)
    assert sp._ADM2 is False  # stopped: it no longer sets anything


def test_ps1_and_pk6_reach_the_environment_the_native_library_reads(tmp_path, monkeypatch):
    import os

    import ironmule.ggml_engine as ge

    monkeypatch.setattr(ls, "_started", False)
    monkeypatch.setattr(ge, "PK6", False)
    monkeypatch.setattr(ge, "PS1", True)
    monkeypatch.setenv("IRONMULE_PK6", "0")
    monkeypatch.setenv("IRONMULE_PS1", "1")
    ls._apply({"PK6": 1, "PS1": 0})
    assert ge.PK6 is True and os.environ["IRONMULE_PK6"] == "1"
    assert ge.PS1 is False and os.environ["IRONMULE_PS1"] == "0"
    ls._apply({"PK6": 0})
    assert ge.PK6 is False and os.environ["IRONMULE_PK6"] == "0" and os.environ["IRONMULE_PS1"] == "0"
