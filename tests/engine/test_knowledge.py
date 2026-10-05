import importlib
import json
import types
from pathlib import Path

import pytest

from ironmule import knowledge

T4 = {"binding": {"backend": {"kind": "cuda", "device_info": {"device_name": "Tesla T4", "architecture": "sm_75"}}}}
M1 = {"binding": {"backend": {"kind": "metal", "device_info": {"device_name": "Apple M1 Max",
                                                               "architecture": "applegpu_g13s"}}}}


@pytest.fixture(autouse=True)
def private_store(monkeypatch, tmp_path):
    monkeypatch.setattr(knowledge, "local_path", lambda: tmp_path / "tune_knowledge.json")
    monkeypatch.setenv("IRONMULE_TUNE_KNOWLEDGE", "on")


def test_device_class_comes_from_the_probe_binding():
    assert knowledge.device_class(T4) == "cuda:Tesla T4:sm_75"
    assert knowledge.device_class({"fingerprint": "x"}) is None


def test_seed_skips_only_what_never_won_with_enough_evidence():
    plan = knowledge.plan(T4)
    assert plan["enabled"] and not plan["exploring"]
    assert plan["skip"] == ["fuse_projections=True", "wired_fraction=0.6"]
    assert knowledge.plan(M1)["skip"] == [], "three M1 tunes are not enough evidence"


def test_every_fifth_tune_of_a_class_screens_everything():
    tunes = knowledge.counts("cuda:Tesla T4:sm_75")["tunes"]
    for _ in range((knowledge.EXPLORE_EVERY - 1 - tunes) % knowledge.EXPLORE_EVERY):
        knowledge.record_tune("cuda:Tesla T4:sm_75", [])
    plan = knowledge.plan(T4)
    assert plan["exploring"] and plan["skip"] == []


def test_a_new_keep_lifts_the_skip_and_skipped_trials_add_nothing():
    cls = "cuda:Tesla T4:sm_75"
    before = knowledge.counts(cls)["knobs"]["fuse_projections=True"]
    knowledge.record_tune(cls, [{"knob": "fuse_projections", "value": True, "disposition": "accepted"},
                                {"knob": "wired_fraction", "value": 0.6, "disposition": "skipped"}])
    after = knowledge.counts(cls)["knobs"]
    assert after["fuse_projections=True"]["kept"] == before["kept"] + 1
    assert after["wired_fraction=0.6"]["unsupported"] == 23
    assert "fuse_projections=True" not in knowledge.plan(T4)["skip"]


def test_knowledge_skipping_is_opt_in(monkeypatch):
    monkeypatch.delenv("IRONMULE_TUNE_KNOWLEDGE")
    assert knowledge.plan(T4) == {"class": "cuda:Tesla T4:sm_75", "skip": [], "exploring": False,
                                  "enabled": False}


def test_tune_skips_known_losers_without_measuring_them(monkeypatch):
    tune = importlib.import_module("ironmule.tune")
    from ironmule.model_identity import ModelIdentity, canonical_json, canonical_sha256
    measured = []

    class FakeEngine:
        def __init__(self, knobs):
            self.knobs, self._compiled = knobs, None


        def generate(self, *_args):  # tune's speed probe (TUNE1)

            return None

        @staticmethod
        def needs_reload(old, new):
            return False

    quantisation = {"bits": 4, "group_size": 64}
    identity = ModelIdentity(model_id=tune.DEFAULT_MODEL, revision="r", model_manifest_sha256="a" * 64,
                             architecture="a", quantisation_json=canonical_json(quantisation),
                             quantisation_sha256=canonical_sha256(quantisation), tokenizer_sha256="b" * 64,
                             manifest_file_count=3, manifest_bytes=3, tokenizer_file_count=1)
    monkeypatch.setattr(tune, "Engine", FakeEngine)
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    monkeypatch.setattr(tune, "probe", lambda: dict(T4, fingerprint="t4"))
    monkeypatch.setattr(tune, "resolve_local_model",
                        lambda *a, **k: types.SimpleNamespace(path=Path("/m"), identity=identity))
    monkeypatch.setattr(tune, "load_engine", lambda _m, knobs, **k: (FakeEngine(knobs), object()))
    monkeypatch.setattr(tune, "prompt_ids", lambda *_: [1, 2])
    monkeypatch.setattr(tune, "_eos_ids", lambda _: (99,))
    monkeypatch.setattr(tune, "conditions", lambda *a, **k: {"prompt_tokens": 2, "max_tokens": 2})
    monkeypatch.setattr(tune, "save_profile", lambda _p: None)
    monkeypatch.setattr(tune, "_release_device_memory", lambda: None)
    monkeypatch.setattr(tune, "SEARCH", [("wired_fraction", [0.6]), ("readback_every", [2])])

    def fake_measure(engine, *_a, **_k):
        measured.append(engine.knobs)
        return {"total_ns": 10, "prefill_ns": 5, "decode_ns": 5, "logical_tokens": [7],
                "deterministic": True, "capacity": 2}

    monkeypatch.setattr(tune, "measure", fake_measure)
    profile = tune.tune(repeats=1, confirm_winner=False)
    assert [k.wired_fraction for k in measured] == [0.0, 0.0], "baseline and readback only; wired skipped"
    skipped = next(t for t in profile["trials"] if t["knob"] == "wired_fraction")
    assert skipped["disposition"] == "skipped" and profile["knowledge"]["skip"]
    record = json.loads(knowledge.local_path().read_text())["classes"]["cuda:Tesla T4:sm_75"]
    assert record["tunes"] == 1 and "wired_fraction=0.6" not in record["knobs"]


def test_tune_gives_up_where_a_tiny_probe_takes_minutes(monkeypatch):
    tune = importlib.import_module("ironmule.tune")
    clock = [0.0]

    class SlowEngine:
        knobs = None

        def generate(self, *_args):  # a CPU's 4-bit kernels: 20 tokens in a minute (CPU1)
            clock[0] += 60.0

    measured = []
    monkeypatch.setattr(tune.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(tune, "resolve_local_model", lambda *a, **k: types.SimpleNamespace(path=Path("/m")))
    monkeypatch.setattr(tune, "load_engine", lambda *a, **k: (SlowEngine(), object()))
    monkeypatch.setattr(tune, "prompt_ids", lambda *_: list(range(40)))
    monkeypatch.setattr(tune, "_eos_ids", lambda _: (99,))
    monkeypatch.setattr(tune, "measure", lambda *a, **k: measured.append(a))
    monkeypatch.setattr(tune, "_release_device_memory", lambda: None)
    with pytest.raises(RuntimeError, match="TUNE1"):
        tune._screen("m", "p", 32, 5, None, {"skip": [], "class": None})
    assert measured == [], "no baseline is measured for hours"


def test_a_cold_first_probe_is_repeated_warm_before_giving_up(monkeypatch):
    # RSI1, T4: a first call that compiles kernels crossed the budget on small models.
    tune = importlib.import_module("ironmule.tune")
    clock, costs = [0.0], iter([30.0, 0.5])

    class ColdEngine:
        def generate(self, *_args):
            clock[0] += next(costs)

    monkeypatch.setattr(tune.time, "perf_counter", lambda: clock[0])
    assert tune.probe_too_slow(ColdEngine(), list(range(40)), (99,)) is None


def test_a_too_slow_screening_child_keeps_its_reason(monkeypatch):
    import subprocess
    tune = importlib.import_module("ironmule.tune")

    class Child:
        returncode = 3
        stdout = iter(["probe ...\n"])

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: Child())
    with pytest.raises(RuntimeError, match="TUNE1"):
        tune._screen_in_child("m", "p", 32, 5, None, {"skip": [], "class": None})
