"""PL1: the Pilot's features, decisions, fallbacks and speed, with a stand-in engine and made-up weights."""

import time
import types
from pathlib import Path

import numpy as np

from ironmule import pilot as pl

COSTS = {"step_one": 0.1, "row": 0.003, "prefill_1k": 3.0, "kv_1k": 0.02}


def weights(tmp_path: Path, prefer=(7, 6, 3, 1)) -> Path:
    """Weights whose output prefers the given choice per head, whatever the input."""
    rng = np.random.default_rng(0)
    w1 = rng.normal(0, 0.01, (128, pl.OBS)).astype(np.float32)
    w2 = rng.normal(0, 0.01, (128, 128)).astype(np.float32)
    b3 = np.zeros(sum(pl.HEADS), np.float32)
    off = 0
    for n, pick in zip(pl.HEADS, prefer):
        b3[off + pick] = 5.0
        off += n
    path = tmp_path / "w.npz"
    np.savez(path, w1=w1, b1=np.zeros(128, np.float32), w2=w2, b2=np.zeros(128, np.float32),
             w3=np.zeros((sum(pl.HEADS), 128), np.float32), b3=b3, heads=np.array(pl.HEADS), obs=np.array(pl.OBS))
    return path


def engine():
    return types.SimpleNamespace(drafter=None)


def test_a_decision_follows_the_network_and_stays_in_range(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "usable_cores", lambda: 16)
    pilot = pl.Pilot(engine(), COSTS, max_cap=32, weights=weights(tmp_path, (3, 2, 1, 0)))
    d = pilot.decide(running=4, queued=10, waiting_prompt_tokens=3000, mean_ctx=500.0)
    assert d == {"threads": 8, "cap": 4, "budget": 1024, "draft": False}
    features = pilot.features(4, 10, 3000, 500.0)
    assert features.shape == (pl.OBS,) and np.all(np.isfinite(features)) and np.abs(features).max() < 4


def test_the_cap_never_exceeds_the_engine_and_threads_are_at_least_one(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "usable_cores", lambda: 2)
    pilot = pl.Pilot(engine(), COSTS, max_cap=8, weights=weights(tmp_path, (0, 6, 3, 1)))
    d = pilot.decide(0, 0, 0, 0.0)
    assert d["threads"] == 1 and d["cap"] == 8


def test_without_weights_or_turned_off_the_engine_keeps_its_defaults(tmp_path, monkeypatch):
    pilot = pl.Pilot(engine(), COSTS, max_cap=16, weights=tmp_path / "missing.npz")
    assert not pilot.active and "no weights" in pilot.reason
    monkeypatch.setenv("IRONMULE_PILOT", "0")
    off = pl.Pilot(engine(), COSTS, max_cap=16, weights=weights(tmp_path))
    assert not off.active and "IRONMULE_PILOT=0" in off.reason


def test_an_input_outside_what_it_was_trained_on_falls_back_to_the_defaults(tmp_path):
    pilot = pl.Pilot(engine(), COSTS, max_cap=32, weights=weights(tmp_path, (0, 0, 0, 0)))
    d = pilot.decide(running=4, queued=10 ** 12, waiting_prompt_tokens=10 ** 15, mean_ctx=1e12)
    assert pilot.out_of_range == 1 and d["threads"] == pilot.cores and d["cap"] == 32


def test_a_decision_takes_microseconds(tmp_path):
    pilot = pl.Pilot(engine(), COSTS, max_cap=32, weights=weights(tmp_path))
    for _ in range(200):
        pilot.decide(8, 2, 300, 400.0)
    started = time.perf_counter()
    for _ in range(2000):
        pilot.decide(8, 2, 300, 400.0)
    assert (time.perf_counter() - started) / 2000 < 1e-3  # well under a millisecond (about 20 us here)


def test_foreign_load_is_read_from_proc():
    load = pl.CpuLoad()
    time.sleep(0.06)
    assert load.update() >= 0.0 and pl.usable_cores() >= 1
