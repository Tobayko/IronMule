"""A rejected cheap stage must never launch a later GPU workload."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def test_failed_smoke_stops_session(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "lab" / "kg5" / "kaggle_session.py"
    spec = importlib.util.spec_from_file_location("kg5_session_test", source)
    session = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(session)
    stages = []

    def reject(args):
        stages.append(args[1])
        return 2

    monkeypatch.setitem(sys.modules, "bench", SimpleNamespace(main=reject))
    monkeypatch.setattr(session.os, "environ", dict(session.os.environ))
    rc = session.run({"through": "steps", "commit": "a" * 40, "budget_s": 60}, work=tmp_path)
    summary = json.loads((tmp_path / "kg5-session.json").read_text())
    assert (rc, stages, summary["status"], summary["stopped_at"]) == (2, ["smoke"], "rejected", "smoke")
