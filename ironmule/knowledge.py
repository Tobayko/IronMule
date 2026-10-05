"""Tune knowledge: what each knob did on a class of device, across models and machines.

Only counts are kept: per device class (backend, device name, architecture, as the hardware
probe binds them) and per knob value, how often it was screened, kept and unsupported. A
packaged seed carries the aggregate outcomes measured so far; every tune adds its own locally.

With `IRONMULE_TUNE_KNOWLEDGE=on`, a candidate screened at least `MIN_TRIES` times on this
class and never kept, or never supported, is skipped (opt-in: PRED1 saved too little time). Every `EXPLORE_EVERY`-th tune of a class screens everything, so a
changed library or driver can overturn the record instead of being hidden by it. The paired
confirmation of the winner is untouched: knowledge saves screening time, never decides a gain.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .hw import STORE

SCHEMA = "ironmule.tune_knowledge.v1"
SEED = Path(__file__).with_name("tune_knowledge_seed.json")
MIN_TRIES = 8
EXPLORE_EVERY = 5


def local_path() -> Path:
    return STORE / "tune_knowledge.json"


def device_class(record: Mapping[str, Any]) -> str | None:
    backend = (record.get("binding") or {}).get("backend") or {}
    info = backend.get("device_info") or {}
    if not backend.get("kind") or not info.get("device_name"):
        return None
    return f"{backend['kind']}:{info['device_name']}:{info.get('architecture', '')}"


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value.get("classes", {}) if isinstance(value, dict) and value.get("schema") == SCHEMA else {}


def counts(cls: str) -> dict[str, Any]:
    """Seed plus local counts for one class."""
    merged: dict[str, Any] = {"tunes": 0, "knobs": {}}
    for source in (_read(SEED), _read(local_path())):
        entry = source.get(cls) or {}
        merged["tunes"] += int(entry.get("tunes", 0))
        for key, value in (entry.get("knobs") or {}).items():
            slot = merged["knobs"].setdefault(key, {"tried": 0, "kept": 0, "unsupported": 0})
            for field in slot:
                slot[field] += int(value.get(field, 0))
    return merged


def plan(record: Mapping[str, Any]) -> dict[str, Any]:
    """Which candidates this tune may skip, and whether it is an exploring tune."""
    cls = device_class(record)
    # Opt-in since PRED1's T4 run saved too little tune time to justify skipping by default;
    # every tune still records its outcomes so the record keeps growing.
    if cls is None or os.environ.get("IRONMULE_TUNE_KNOWLEDGE") != "on":
        return {"class": cls, "skip": [], "exploring": False, "enabled": False}
    known = counts(cls)
    exploring = known["tunes"] % EXPLORE_EVERY == EXPLORE_EVERY - 1
    skip = [] if exploring else sorted(
        key for key, slot in known["knobs"].items()
        if slot["kept"] == 0 and (slot["tried"] >= MIN_TRIES
                                  or (slot["tried"] == 0 and slot["unsupported"] >= MIN_TRIES)))
    return {"class": cls, "skip": skip, "exploring": exploring, "enabled": True}


def record_tune(cls: str | None, trials: list[Mapping[str, Any]]) -> None:
    """Add one tune's screening outcomes to the local record (skipped candidates add nothing)."""
    if cls is None:
        return
    path = local_path()
    try:
        data = json.loads(path.read_text())
        if data.get("schema") != SCHEMA:
            data = {}
    except (OSError, ValueError):
        data = {}
    entry = data.setdefault("classes", {}).setdefault(cls, {"tunes": 0, "knobs": {}})
    entry["tunes"] += 1
    for trial in trials:
        disposition = trial.get("disposition")
        if disposition == "skipped":
            continue
        slot = entry["knobs"].setdefault(f"{trial['knob']}={trial['value']}",
                                         {"tried": 0, "kept": 0, "unsupported": 0})
        if disposition == "unsupported":
            slot["unsupported"] += 1
        else:
            slot["tried"] += 1
            slot["kept"] += disposition == "accepted"
    data["schema"] = SCHEMA
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    temporary.replace(path)
