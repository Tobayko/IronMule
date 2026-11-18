"""PI1: the Pilot's features, decisions, fallbacks and speed, with a stand-in engine and made-up weights."""

import time
import types
from pathlib import Path

import numpy as np

from ironmule import pilot as pl

COSTS = {"step_one": 0.1, "row": 0.003, "prefill_1k": 3.0, "kv_1k": 0.02}


def weights(tmp_path: Path, prefer=(7, 6, 4, 1)) -> Path:
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
    assert d == {"threads": 8, "cap": 4, "budget": 128, "draft": False}
    features = pilot.features(4, 10, 3000, 500.0)
    assert features.shape == (pl.OBS,) and np.all(np.isfinite(features)) and np.abs(features).max() < 4


def test_the_cap_never_exceeds_the_engine_and_threads_are_at_least_one(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "usable_cores", lambda: 2)
    pilot = pl.Pilot(engine(), COSTS, max_cap=8, weights=weights(tmp_path, (0, 6, 4, 1)))
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



def test_the_load_rereads_open_files_and_close_releases_them():
    load = pl.CpuLoad()
    assert load._fds is not None
    first = load._last
    time.sleep(0.06)
    load.update()
    assert load._last[0] > first[0] and load.foreign >= 0.0
    load.close()
    assert load._fds is None and load._read() is None
    load.close()  # twice is fine


def test_weights_for_the_four_budgets_of_version_1_are_refused(tmp_path):
    path = weights(tmp_path)
    z = dict(np.load(path))
    z["heads"] = np.array((8, 7, 4, 2))
    old = tmp_path / "v1.npz"
    np.savez(old, **z)
    pilot = pl.Pilot(engine(), COSTS, max_cap=16, weights=old)
    assert not pilot.active and "another Pilot version" in pilot.reason


def test_an_engine_on_a_gpu_takes_the_gpu_weights_and_a_cpu_engine_the_cpu_weights(tmp_path, monkeypatch):
    monkeypatch.delenv("IRONMULE_PILOT_WEIGHTS", raising=False)
    monkeypatch.delenv("IRONMULE_PILOT", raising=False)
    (tmp_path / "g").mkdir()
    cpu, gpu = weights(tmp_path, (7, 6, 4, 1)), weights(tmp_path / "g", (7, 6, 1, 1)).rename(tmp_path / "gpu.npz")
    monkeypatch.setattr(pl, "WEIGHTS", cpu)
    monkeypatch.setattr(pl, "WEIGHTS_GPU", gpu)
    on_gpu = pl.Pilot(engine(), COSTS, max_cap=32, gpu=True)
    on_cpu = pl.Pilot(engine(), COSTS, max_cap=32, gpu=False)
    assert on_gpu.status()["weights"] == "gpu.npz" and on_cpu.status()["weights"] == cpu.name
    assert on_gpu.decide(4, 2, 300, 200.0)["budget"] == pl.BUDGETS[1]
    assert on_cpu.decide(4, 2, 300, 200.0)["budget"] == pl.BUDGETS[4]
    monkeypatch.setattr(pl, "WEIGHTS_GPU", tmp_path / "missing.npz")  # without a GPU file the CPU weights serve both
    assert pl.Pilot(engine(), COSTS, max_cap=32, gpu=True).status()["weights"] == cpu.name
    monkeypatch.setenv("IRONMULE_PILOT_WEIGHTS", str(gpu))  # the variable names one file for every engine
    assert pl.Pilot(engine(), COSTS, max_cap=32, gpu=False).status()["weights"] == "gpu.npz"


def test_both_shipped_weight_files_load_for_this_version():
    for path in (pl.WEIGHTS, pl.WEIGHTS_GPU):
        pilot = pl.Pilot(engine(), COSTS, max_cap=32, gpu=path is pl.WEIGHTS_GPU, weights=path)
        assert pilot.active, (path.name, pilot.reason)


def test_local_thread_head_qualifies_rechecks_and_reverts(tmp_path, monkeypatch):
    from ironmule.pilot_learning import MIN_STEPS, ThreadLearning

    e = types.SimpleNamespace(settings={}, seqs=2, seq_ctx=512, shared=False, cap=2,
                              pilot=types.SimpleNamespace(cores=4))
    learner = ThreadLearning(e, tmp_path, "model", "weights")
    def replay(context, base, threads, seed, interrupted):
        return (0.8 if threads == 2 else 1.0), (1, 2, 3)
    monkeypatch.setattr(learner, "_probe", replay)
    for _ in range(MIN_STEPS):
        assert learner.choose(90, 0.1, 3) == 3
    learner.idle(lambda: False)
    assert learner.choose(90, 0.1, 3) == 2
    assert learner.choose(90, 1.0, 3) == learner.choose(150, 0.1, 3) == 3
    assert learner.status()["last"]["result"] == "qualified"
    resumed = ThreadLearning(e, tmp_path, "model", "weights")
    assert resumed.choose(90, 0.1, 3) == 3 and resumed.proposals
    assert not ThreadLearning(e, tmp_path, "different-model", "weights").proposals
    for _ in range(MIN_STEPS):
        learner.choose(90, 0.1, 3)
    monkeypatch.setattr(learner, "_probe", lambda context, base, threads, seed, interrupted: (1.0, (threads,)))
    learner.idle(lambda: False)
    assert learner.choose(90, 0.1, 3) == 3 and learner.last["result"] == "reverted"


def test_local_thread_head_refuses_aa_noise_and_interruption(tmp_path, monkeypatch):
    from ironmule.pilot_learning import MIN_STEPS, ThreadLearning

    e = types.SimpleNamespace(settings={}, seqs=2, seq_ctx=512, shared=False, cap=2,
                              pilot=types.SimpleNamespace(cores=4))
    learner = ThreadLearning(e, tmp_path, "m", "w")
    controls = iter([1.0, 1.3, 1.0, 1.3, 1.0, 1.3, 1.0, 1.3, 1.0])
    def noisy(context, base, threads, seed, interrupted):
        return (next(controls) if threads == base else 0.96), (1,)
    monkeypatch.setattr(learner, "_probe", noisy)
    for _ in range(MIN_STEPS):
        learner.choose(300, 0.1, 4)
    learner.idle(lambda: True)
    assert learner.last is None
    learner.idle(lambda: False)
    assert not learner.weights and learner.last["result"] == "refused"
