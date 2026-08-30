from dataclasses import FrozenInstanceError

import pytest

from friday_optimizer.candidates import CandidateError, CandidateRegistry
from friday_optimizer.fingerprint import (
    EnvironmentFingerprint,
    ExactFingerprint,
    FingerprintError,
    ModelFingerprint,
    WorkloadFingerprint,
)


def make_fingerprint(*, mode="interactive", complete=True):
    environment = EnvironmentFingerprint(
        chip="M1 Max", gpu="Apple GPU", ram_bytes=64 * 1024**3, cpu_cores=10,
        macos="14.5", mlx="0.32.0", mlx_lm="0.0.13", python="3.12.13", runtime_commit="a" * 64,
    )
    model = ModelFingerprint(
        model_id="google/gemma-3-4b-it", revision="r1", manifest="b" * 64,
        architecture="gemma", quant_bits=4, quant_group_size=128, tokenizer="tok-r1",
    )
    workload = WorkloadFingerprint(
        prompt_family="chat", tokenizer="tok-r1", generator="gen-r1", context_bucket="short",
        batch=1, concurrency=1, max_tokens=64, greedy=True, prompt_logprobs=False,
        power_mode="performance", mode=mode,
    )
    if not complete:
        model = ModelFingerprint(model_id=model.model_id)
    return ExactFingerprint(environment, model, workload)


def test_fingerprint_is_canonical_and_immutable():
    first = make_fingerprint()
    second = ExactFingerprint.from_mapping(first.as_dict())
    assert first.fingerprint_hash == second.fingerprint_hash
    assert first.canonical_bytes == second.canonical_bytes
    assert len(first.fingerprint_hash) == 64
    with pytest.raises(FrozenInstanceError):
        first.environment.chip = "other"


def test_type_changes_change_hash_and_bool_is_not_integer():
    first = make_fingerprint()
    altered = ModelFingerprint.from_mapping({**first.model.as_dict(), "quant_bits": 8})
    assert first.model != altered
    with pytest.raises(FingerprintError):
        EnvironmentFingerprint(ram_bytes=True)


def test_missing_identity_is_ood_and_cannot_match():
    incomplete = make_fingerprint(complete=False)
    assert incomplete.ood
    assert not incomplete.recommendation_allowed
    assert incomplete.ood_reason
    assert CandidateRegistry().ordered_ids(incomplete) == ("baseline",)


def test_allowlist_scopes_readback_and_throughput():
    registry = CandidateRegistry()
    interactive = make_fingerprint()
    throughput = make_fingerprint(mode="throughput")
    assert registry.is_allowed("readback_every_2", fingerprint=interactive)
    assert not registry.is_allowed("readback_every_2", fingerprint=throughput)
    assert registry.is_allowed("throughput_width_4", fingerprint=throughput)
    assert not registry.is_allowed("throughput_width_4", fingerprint=interactive)
    with pytest.raises(CandidateError):
        registry.resolve("throughput_width_4", {"width": 8})
    assert registry.ordered_ids(interactive, historical_hints=("readback_every_2",))[0] == "baseline"


def test_production_registry_rejects_spec_injection_and_exposes_stable_hash():
    registry = CandidateRegistry()
    assert len(registry.registry_hash) == 64
    with pytest.raises(CandidateError):
        CandidateRegistry(registry.specs)
