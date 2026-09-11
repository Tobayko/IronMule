"""The profile may name a service strategy; nothing else may turn one on.

Every test here is about a gate, not about speed: which conditions let the automatic
choice reach the paired path, and which ones must leave the established mode in place.
"""

from __future__ import annotations

import pytest

from ironmule import service_strategy as ss
from ironmule.service import AutomaticMode, Runtime, ThroughputMode

FINGERPRINT = "dc652d66f24ac207"
IDENTITY = "2b5b13a3" * 8
RECORD = ss.build_record(
    strategy=ss.STRATEGY_PAIRED, min_ready=2, max_ready=4,
    identity_sha256=IDENTITY, fingerprint=FINGERPRINT, mlx="0.32.0", mlx_lm="0.31.3",
    correctness_contract="logits, KV, tokens, stop reason and step count identical",
    evidence_run_ids=["B51_identity_gap_20260909"],
)
HERE = {"identity_sha256": IDENTITY, "fingerprint": FINGERPRINT,
        "mlx": "0.32.0", "mlx_lm": "0.31.3"}


def _decide(profile, *, opt_in=True, ready=2, **overrides):
    return ss.select(profile, opt_in=opt_in, ready_requests=ready, **{**HERE, **overrides})


# -- the record itself ---------------------------------------------------------

def test_a_record_needs_the_runs_that_evidence_it() -> None:
    with pytest.raises(ss.StrategyRefused):
        ss.build_record(strategy=ss.STRATEGY_PAIRED, min_ready=2, max_ready=4,
                        identity_sha256=IDENTITY, fingerprint=FINGERPRINT,
                        mlx="0.32.0", mlx_lm="0.31.3",
                        correctness_contract="whatever", evidence_run_ids=[])


def test_an_unknown_strategy_is_refused_at_build_time() -> None:
    with pytest.raises(ss.StrategyRefused):
        ss.build_record(strategy="magic", min_ready=2, max_ready=4,
                        identity_sha256=IDENTITY, fingerprint=FINGERPRINT,
                        mlx="0.32.0", mlx_lm="0.31.3",
                        correctness_contract="c", evidence_run_ids=["r"])


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("correctness_contract"),
    lambda r: r.pop("evidence_run_ids"),
    lambda r: r.pop("admitted_range"),
    lambda r: r.update(schema="ironmule.tuned_profile.service_strategy.v0"),
    lambda r: r.update(extra=1),
    lambda r: r["admitted_range"].pop("mlx"),
    lambda r: r["admitted_range"].update(extra=1),
])
def test_an_incomplete_or_foreign_record_reads_back_as_absent(mutate) -> None:
    """A record missing a field is refused, never partially trusted."""

    import copy
    record = copy.deepcopy(RECORD)
    mutate(record)
    assert ss.read_record({"service_strategy": record}) is None


# -- selection -----------------------------------------------------------------

def test_an_older_profile_without_a_record_keeps_its_behaviour() -> None:
    decision = _decide({"knobs": {}})
    assert decision["strategy"] == ss.STRATEGY_THROUGHPUT
    assert decision["record_present"] is False


def test_a_missing_field_can_never_activate_the_paired_strategy() -> None:
    import copy
    broken = copy.deepcopy(RECORD)
    del broken["admitted_range"]["hardware_fingerprint"]
    assert _decide({"service_strategy": broken})["strategy"] == ss.STRATEGY_THROUGHPUT


def test_without_the_opt_in_a_valid_record_changes_nothing() -> None:
    decision = _decide({"service_strategy": RECORD}, opt_in=False)
    assert decision["strategy"] == ss.STRATEGY_THROUGHPUT
    assert decision["record_present"] is False


def test_inside_the_admitted_range_the_paired_path_is_chosen() -> None:
    decision = _decide({"service_strategy": RECORD}, ready=2)
    assert decision["strategy"] == ss.STRATEGY_PAIRED
    assert decision["evidence_run_ids"] == ["B51_identity_gap_20260909"]
    assert decision["correctness_contract"]


@pytest.mark.parametrize("ready", [1, 5])
def test_outside_the_ready_range_the_established_mode_stays(ready) -> None:
    decision = _decide({"service_strategy": RECORD}, ready=ready)
    assert decision["strategy"] == ss.STRATEGY_THROUGHPUT
    assert "ready_requests" in decision["reason"]


@pytest.mark.parametrize("field", ["identity_sha256", "fingerprint", "mlx", "mlx_lm"])
def test_an_unknown_configuration_is_refused(field) -> None:
    decision = _decide({"service_strategy": RECORD}, **{field: "something else"})
    assert decision["strategy"] == ss.STRATEGY_THROUGHPUT
    assert "outside the admitted range" in decision["reason"]


def test_the_response_length_is_not_a_feature() -> None:
    """Only facts known at decision time; `select` cannot even be told a length."""

    import inspect
    parameters = set(inspect.signature(ss.select).parameters)
    assert not parameters & {"max_tokens", "expected_tokens", "output_length"}


# -- attaching to a profile ----------------------------------------------------

def test_attach_refuses_a_record_it_would_not_read_back() -> None:
    import copy
    broken = copy.deepcopy(RECORD)
    del broken["correctness_contract"]
    with pytest.raises(ss.StrategyRefused):
        ss.attach({"knobs": {}}, broken)


def test_attach_leaves_the_original_profile_untouched() -> None:
    original = {"knobs": {}}
    migrated = ss.attach(original, RECORD)
    assert "service_strategy" not in original
    assert ss.read_record(migrated) == RECORD


# -- the mode ------------------------------------------------------------------

class _Telemetry:
    pass


class _Backend:
    engine = None


class _Session:
    """Only what the decision may look at: when it arrives and whether it is finished."""

    def __init__(self, arrival_ms: float = 0.0, done: bool = False) -> None:
        self.arrival_ms = arrival_ms
        self.done = done


def _ready(count: int) -> list[_Session]:
    return [_Session() for _ in range(count)]


class _Paired:
    name = "paired_throughput"

    def executor(self, backend, telemetry):
        return "paired-executor"

    def status(self):
        return {"mode": self.name, "enabled": True, "admitted": True}


def _admitted_mode() -> AutomaticMode:
    mode = AutomaticMode({"service_strategy": RECORD}, opt_in=True, **HERE)
    mode._paired = _Paired()
    return mode


def test_the_automatic_mode_defaults_to_the_established_executor() -> None:
    mode = AutomaticMode({"service_strategy": RECORD}, opt_in=True, **HERE)
    delegate = mode.delegate_for(_ready(1), _Backend(), _Telemetry())
    assert delegate.__class__ is ThroughputMode().executor(_Backend(), _Telemetry()).__class__
    assert mode.status()["strategy"] == "throughput"


def test_a_model_that_refuses_admission_falls_back_not_through() -> None:
    """The profile can say yes while the loaded model says no. That is not a crash."""

    mode = AutomaticMode({"service_strategy": RECORD}, opt_in=True, **HERE)
    mode.delegate_for(_ready(2), _Backend(), _Telemetry())
    status = mode.status()
    assert status["strategy"] == "throughput"
    assert "the model refused it" in status["reason"]


def test_an_admitted_pair_reaches_the_paired_executor() -> None:
    mode = _admitted_mode()
    assert mode.delegate_for(_ready(2), _Backend(), _Telemetry()) == "paired-executor"
    status = mode.status()
    assert status["strategy"] == ss.STRATEGY_PAIRED
    assert status["evidence_run_ids"] == ["B51_identity_gap_20260909"]
    assert status["enabled"] is True


def test_a_partner_that_has_not_arrived_yet_is_not_counted_as_ready() -> None:
    """Two requests submitted is not two requests ready."""

    mode = _admitted_mode()
    sessions = [_Session(), _Session(arrival_ms=50.0)]

    delegate = mode.delegate_for(sessions, _Backend(), _Telemetry())

    assert delegate != "paired-executor"
    status = mode.status()
    assert status["strategy"] == "throughput"
    assert (status["ready_requests"], status["group_requests"]) == (1, 2)


def test_a_request_finished_during_prefill_is_not_counted_as_ready() -> None:
    """A request that stopped on its first token never takes a step, so it is no partner."""

    mode = _admitted_mode()
    sessions = [_Session(), _Session(done=True)]

    delegate = mode.delegate_for(sessions, _Backend(), _Telemetry())

    assert delegate != "paired-executor"
    assert mode.status()["ready_requests"] == 1


def test_four_ready_requests_stay_inside_the_admitted_range() -> None:
    mode = _admitted_mode()

    assert mode.delegate_for(_ready(4), _Backend(), _Telemetry()) == "paired-executor"
    assert mode.status()["ready_requests"] == 4


def test_five_ready_requests_leave_the_admitted_range() -> None:
    mode = _admitted_mode()

    assert mode.delegate_for(_ready(5), _Backend(), _Telemetry()) != "paired-executor"
    assert mode.status()["strategy"] == "throughput"


def test_automatic_and_an_explicit_mode_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError):
        Runtime.load("mlx-community/gemma-3-4b-it-4bit", mode=ThroughputMode(),
                     automatic_service_mode=True)


# -- loading --------------------------------------------------------------------

class _FakeEngine:
    model_identity = None

    def close(self) -> None:
        return None


def _fake_load(monkeypatch, profile):
    """Load a runtime without a model, so the mode choice can be checked on its own."""

    import importlib
    import types

    from ironmule import service as service_module

    # `ironmule.tune` the name is the tuning function; the module is behind it.
    tune = importlib.import_module("ironmule.tune")

    from ironmule.model_identity import ModelIdentity, canonical_json, canonical_sha256

    quantisation = {"bits": 4, "group_size": 64}
    identity = ModelIdentity(
        model_id="org/model", revision="revision", model_manifest_sha256="a" * 64,
        architecture="gemma3", quantisation_json=canonical_json(quantisation),
        quantisation_sha256=canonical_sha256(quantisation), tokenizer_sha256="b" * 64,
        manifest_file_count=3, manifest_bytes=100, tokenizer_file_count=1,
    )
    seen = {"profile_reads": 0}

    def fake_load_profile(model_id, **_kwargs):
        seen["profile_reads"] += 1
        return profile

    monkeypatch.setattr(tune, "resolve_local_model",
                        lambda *_a, **_k: types.SimpleNamespace(identity=identity))
    monkeypatch.setattr(tune, "load_profile", fake_load_profile)
    monkeypatch.setattr(tune, "load_engine", lambda *_a, **_k: (_FakeEngine(), object()))
    monkeypatch.setattr(tune, "_eos_ids", lambda _tokenizer: ())
    monkeypatch.setattr(service_module, "MLXBackend", lambda *_a, **_k: object())
    monkeypatch.setattr(service_module.Runtime, "_automatic_mode",
                        staticmethod(lambda profile, _identity: AutomaticMode(
                            profile, opt_in=True, **HERE)))
    return seen


def test_loading_without_the_opt_in_leaves_the_established_default(monkeypatch) -> None:
    """A profile carrying the record changes nothing until a caller asks for it."""

    _fake_load(monkeypatch, {"knobs": {}, "service_strategy": RECORD})

    runtime = Runtime.load("org/model")

    assert runtime.mode.__class__.__name__ == "InteractiveMode"
    assert not hasattr(runtime.mode, "profile")


def test_loading_with_the_opt_in_hands_the_profile_to_the_mode(monkeypatch) -> None:
    _fake_load(monkeypatch, {"knobs": {}, "service_strategy": RECORD})

    runtime = Runtime.load("org/model", automatic_service_mode=True)

    assert runtime.mode.name == "automatic"
    assert ss.read_record(runtime.mode.profile) == RECORD


def test_the_opt_in_survives_a_missing_profile(monkeypatch) -> None:
    """No profile is not an error, it is the established mode."""

    _fake_load(monkeypatch, None)

    runtime = Runtime.load("org/model", automatic_service_mode=True)
    delegate = runtime.mode.delegate_for(_ready(2), _Backend(), _Telemetry())

    assert delegate.__class__ is ThroughputMode().executor(_Backend(), _Telemetry()).__class__
    assert runtime.mode.status()["record_present"] is False


@pytest.mark.parametrize("profile", [
    {"knobs": {}, "service_strategy": "not a record"},
    {"knobs": {}, "service_strategy": {"schema": ss.SERVICE_STRATEGY_SCHEMA}},
    {"knobs": {}},
])
def test_an_invalid_profile_never_raises_and_never_admits(monkeypatch, profile) -> None:
    _fake_load(monkeypatch, profile)

    runtime = Runtime.load("org/model", automatic_service_mode=True)
    runtime.mode.delegate_for(_ready(2), _Backend(), _Telemetry())

    assert runtime.mode.status()["strategy"] == "throughput"
