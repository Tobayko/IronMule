"""Focused release-blocker regressions for profile/tuning policy (R6/R7)."""

from __future__ import annotations

import json
import importlib
import os
import statistics
import subprocess
import sys
import types
from pathlib import Path

import pytest

from ironmule.runtime import BASELINE, Knobs
from ironmule._version import __version__ as CURRENT_VERSION
from ironmule.model_identity import (
    ModelIdentity, ModelIdentityError, canonical_json, canonical_sha256,
)

tune = importlib.import_module("ironmule.tune")


def _identity(model_id="model", revision="revision"):
    quantisation = {"bits": 4, "group_size": 64}
    return ModelIdentity(
        model_id=model_id,
        revision=revision,
        model_manifest_sha256="a" * 64,
        architecture="test-architecture",
        quantisation_json=canonical_json(quantisation),
        quantisation_sha256=canonical_sha256(quantisation),
        tokenizer_sha256="b" * 64,
        manifest_file_count=3,
        manifest_bytes=100,
        tokenizer_file_count=1,
    )


def test_import_does_not_mutate_huggingface_offline_environment():
    """A library import must preserve values chosen by the embedding process."""
    root = Path(__file__).resolve().parents[1]
    code = "import os; before=(os.environ.get('HF_HUB_OFFLINE'), os.environ.get('TRANSFORMERS_OFFLINE')); import ironmule; after=(os.environ.get('HF_HUB_OFFLINE'), os.environ.get('TRANSFORMERS_OFFLINE')); print(before == after)"
    env = {**os.environ, "HF_HUB_OFFLINE": "caller-value", "TRANSFORMERS_OFFLINE": "caller-value-2"}
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "True"


def _fake_load_engine(monkeypatch):
    seen = {}

    class FakeLoadedEngine:
        def __init__(self, model, tokenizer, knobs):
            self.model = model
            self.tokenizer = tokenizer
            self.knobs = knobs
            self.model_identity = None

    def fake_load(source):
        seen["source"] = source
        seen["environment"] = (os.environ.get("HF_HUB_OFFLINE"),
                                os.environ.get("TRANSFORMERS_OFFLINE"))
        return object(), object()

    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(load=fake_load))
    monkeypatch.setattr(tune, "Engine", FakeLoadedEngine)
    identity = _identity("org/model")

    def fake_resolve(model_id, revision=None):
        seen.setdefault("resolve_calls", []).append((model_id, revision))
        source = Path(model_id) if Path(model_id).is_dir() else Path("/cached/model")
        selected = _identity("local:model" if Path(model_id).is_dir() else "org/model",
                             revision or "revision")
        return types.SimpleNamespace(path=source, identity=selected)

    monkeypatch.setattr(tune, "resolve_local_model", fake_resolve)
    monkeypatch.setattr(tune, "verify_resolved_model", lambda _model, _resolved: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "caller-value")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "caller-value-2")
    seen["identity"] = identity
    return seen


def test_load_engine_offline_local_path_is_direct_and_preserves_environment(monkeypatch, tmp_path):
    seen = _fake_load_engine(monkeypatch)
    local_model = tmp_path / "model"
    local_model.mkdir()
    before = (os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"])

    engine, _ = tune.load_engine(str(local_model), BASELINE, offline=True)

    assert seen["source"] == str(local_model)
    assert seen["environment"] == before
    assert engine.model_identity.model_id == "local:model"
    assert (os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"]) == before


def test_load_engine_offline_hub_id_resolves_cached_snapshot(monkeypatch):
    seen = _fake_load_engine(monkeypatch)

    engine, _ = tune.load_engine("org/model", BASELINE, offline=True, revision="revision")

    assert seen["resolve_calls"] == [("org/model", "revision")]
    assert seen["source"] == "/cached/model"
    assert engine.model_identity.model_id == "org/model"


@pytest.mark.parametrize("offline", [False, None])
def test_load_engine_online_or_caller_mode_passes_hub_id_unchanged(monkeypatch, offline):
    seen = _fake_load_engine(monkeypatch)

    engine, _ = tune.load_engine("org/model", BASELINE, offline=offline)

    assert seen["source"] == "org/model"
    assert seen.get("resolve_calls") is None
    assert engine.model_identity is None


def test_load_engine_never_mutates_preimported_huggingface_state(monkeypatch, tmp_path):
    seen = _fake_load_engine(monkeypatch)
    constants = types.SimpleNamespace(HF_HUB_OFFLINE=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    local_model = tmp_path / "model"
    local_model.mkdir()

    tune.load_engine(str(local_model), BASELINE, offline=True)

    assert constants.HF_HUB_OFFLINE is False
    assert seen["environment"] == ("caller-value", "caller-value-2")


def test_load_engine_rejects_injected_model_or_revision_identity(monkeypatch):
    _fake_load_engine(monkeypatch)
    wrong_model = types.SimpleNamespace(
        path=Path("/cached/model"), identity=_identity("other/model")
    )
    with pytest.raises(ModelIdentityError, match="different model_id"):
        tune.load_engine("org/model", BASELINE, resolved_source=wrong_model)
    wrong_revision = types.SimpleNamespace(
        path=Path("/cached/model"), identity=_identity("org/model", "other")
    )
    with pytest.raises(ModelIdentityError, match="different revision"):
        tune.load_engine(
            "org/model", BASELINE, revision="wanted", resolved_source=wrong_revision
        )


def test_post_load_identity_check_detects_source_change(monkeypatch):
    exact = _identity("org/model", "revision")
    resolved = types.SimpleNamespace(path=Path("/cached/model"), identity=exact)
    monkeypatch.setattr(
        tune, "build_model_identity", lambda *_args, **_kwargs: _identity("org/model", "changed")
    )
    with pytest.raises(ModelIdentityError, match="changed during load"):
        tune.verify_resolved_model("org/model", resolved)


def test_revalidate_uses_tokenized_prompt_length_from_canary(monkeypatch):
    from ironmule import ab

    profile = {"conditions": {"prompt_tokens": 100, "max_tokens": 32},
               "knobs": BASELINE.as_dict(), "gain": 0.0}
    identity = _identity()
    monkeypatch.setattr(
        tune, "resolve_local_model",
        lambda _model, _revision=None: types.SimpleNamespace(
            path=Path("/cached/model"), identity=identity
        ),
    )
    monkeypatch.setattr(tune, "load_profile", lambda _model, **_kwargs: profile)
    observed = {}

    def fake_stale(_profile, _model, prompt_tokens, max_tokens, **_kwargs):
        observed["prompt_tokens"] = prompt_tokens
        observed["max_tokens"] = max_tokens
        return []

    monkeypatch.setattr(tune, "stale", fake_stale)
    monkeypatch.setattr(ab, "run", lambda *_args, **_kwargs: {
        "raw": [{"arms": {"baseline": {"prompt_tokens": 247}}}],
        "ratios": {"stored/baseline": {"total_ns": {"median_ratio": 0.5}}},
        "token_identity": True,
        "deterministic": True,
    })

    result = tune.revalidate(prompt="a materially different prompt", max_tokens=17)
    assert observed == {"prompt_tokens": 247, "max_tokens": 17}
    assert result["verdict"] == "still_valid"


def test_revalidate_requires_token_identity_and_determinism(monkeypatch):
    from ironmule import ab

    identity = _identity()
    profile = {"conditions": {"prompt_tokens": 2, "max_tokens": 17},
               "knobs": BASELINE.as_dict(), "gain": 0.0}
    monkeypatch.setattr(
        tune, "resolve_local_model",
        lambda _model, _revision=None: types.SimpleNamespace(
            path=Path("/cached/model"), identity=identity
        ),
    )
    monkeypatch.setattr(tune, "load_profile", lambda _model, **_kwargs: profile)
    monkeypatch.setattr(ab, "run", lambda *_args, **_kwargs: {
        "raw": [{"arms": {"baseline": {"prompt_tokens": 2}}}],
        "ratios": {"stored/baseline": {"total_ns": {"median_ratio": 0.5}}},
        "token_identity": True,
        "deterministic": False,
    })
    monkeypatch.setattr(tune, "stale", lambda *_args, **_kwargs: [])

    result = tune.revalidate(max_tokens=17)
    assert result["verdict"] == "retune_required"


def test_corrupt_or_incomplete_profile_is_not_reused(monkeypatch):
    monkeypatch.setattr(tune, "fingerprint", lambda: "fp")
    identity = _identity()
    monkeypatch.setattr(tune, "_all_profiles", lambda: {
        f"fp/{identity.identity_sha256}": {
            "model_id": "model", "conditions": {}, "knobs": {},
            "model_identity": identity.to_dict(),
        }
    })
    assert tune.load_profile("model", model_identity=identity) is None
    assert tune.load_profile(
        "model", require_compatible=False, model_identity=identity
    ) is None


def test_all_profiles_rejects_nonfinite_json_constants(monkeypatch, tmp_path):
    monkeypatch.setattr(tune, "PROFILES", tmp_path / "profiles.json")
    tune.PROFILES.write_text('{"value": NaN, "other": Infinity}')
    assert tune._all_profiles() == {}


def test_complete_freshly_shaped_profile_is_reusable(monkeypatch):
    monkeypatch.setattr(tune, "fingerprint", lambda *_args: "fp")
    from ironmule import bench, hw

    monkeypatch.setattr(hw, "static_facts", lambda: {
        "chip": "test-chip", "memory_bytes": 1, "gpu_cores": 1,
    })
    monkeypatch.setattr(bench, "environment", lambda: {
        "mlx": "0.32.0", "mlx_lm": "0.31.3", "power_source": "AC", "os": "test",
    })
    identity = _identity()
    conditions = tune.conditions(
        "model", prompt_tokens=2, max_tokens=2, model_identity=identity
    )
    assert set(conditions) == tune.PROFILE_CONDITION_FIELDS
    assert conditions["runtime_version"] == CURRENT_VERSION
    profile = {
        "model_id": "model",
        "model_identity": identity.to_dict(),
        "conditions": conditions,
        "knobs": BASELINE.as_dict(),
    }
    monkeypatch.setattr(
        tune, "_all_profiles", lambda: {f"fp/{identity.identity_sha256}": profile}
    )
    assert tune.load_profile("model", model_identity=identity) == profile

    invalid_knobs = [
        ("fuse_projections", 1), ("readback_every", 0),
        ("speculate_k", -1), ("speculate_ngram", 0),
        ("capacity_slack", -1), ("wired_fraction", float("nan")),
        ("wired_fraction", 1.1),
    ]
    for name, value in invalid_knobs:
        malformed = dict(profile, knobs=dict(profile["knobs"], **{name: value}))
        monkeypatch.setattr(
            tune, "_all_profiles", lambda malformed=malformed: {
                f"fp/{identity.identity_sha256}": malformed
            }
        )
        assert tune.load_profile("model", model_identity=identity) is None
    nonfinite_metric = dict(profile, baseline_ns=1e999)
    monkeypatch.setattr(
        tune, "_all_profiles", lambda: {f"fp/{identity.identity_sha256}": nonfinite_metric}
    )
    assert tune.load_profile("model", model_identity=identity) is None

    candidate = Knobs(readback_every=2)
    evidence = _confirmation()
    ratio = evidence["ratios"]["candidate/baseline"]
    compact = {
        "ratio": ratio, "token_identity": True, "token_count_identity": True,
        "stop_reason_identity": True, "deterministic": True,
        "accepted": True, "rejection_reason": None,
        "evidence_sha256": tune._confirmation_evidence_sha256(evidence),
    }
    accepted = dict(profile, knobs=candidate.as_dict(), confirmation=compact,
                    confirmation_candidate_knobs=candidate.as_dict(),
                    confirmation_evidence=evidence)
    monkeypatch.setattr(
        tune, "_all_profiles", lambda: {f"fp/{identity.identity_sha256}": accepted}
    )
    assert tune.load_profile("model", model_identity=identity) == accepted
    missing_binding = dict(accepted)
    missing_binding.pop("confirmation_candidate_knobs")
    monkeypatch.setattr(
        tune, "_all_profiles", lambda: {f"fp/{identity.identity_sha256}": missing_binding}
    )
    assert tune.load_profile("model", model_identity=identity) is None
    mismatched_binding = dict(accepted, confirmation_candidate_knobs=BASELINE.as_dict())
    monkeypatch.setattr(
        tune, "_all_profiles", lambda: {f"fp/{identity.identity_sha256}": mismatched_binding}
    )
    assert tune.load_profile("model", model_identity=identity) is None

    rejected_evidence = _confirmation(token_identity=False)
    rejected = dict(
        accepted,
        knobs=BASELINE.as_dict(),
        confirmation={
            "ratio": rejected_evidence["ratios"]["candidate/baseline"],
            "token_identity": False, "token_count_identity": True,
            "stop_reason_identity": True, "deterministic": True,
            "accepted": False, "rejection_reason": "token_identity",
            "evidence_sha256": tune._confirmation_evidence_sha256(rejected_evidence),
        },
        confirmation_evidence=rejected_evidence,
    )
    monkeypatch.setattr(
        tune, "_all_profiles", lambda: {f"fp/{identity.identity_sha256}": rejected}
    )
    assert tune.load_profile("model", model_identity=identity) == rejected

    unknown = dict(profile, conditions=dict(conditions, future_field=True))
    monkeypatch.setattr(
        tune, "_all_profiles", lambda: {f"fp/{identity.identity_sha256}": unknown}
    )
    assert tune.load_profile("model", model_identity=identity) is None
    malformed_confirmation = dict(profile, confirmation={"accepted": True, "ratio": {}})
    monkeypatch.setattr(
        tune, "_all_profiles",
        lambda: {f"fp/{identity.identity_sha256}": malformed_confirmation},
    )
    assert tune.load_profile("model", model_identity=identity) is None
    mismatched = dict(profile, conditions=dict(conditions, model_revision="tampered"))
    with pytest.raises(ModelIdentityError, match="do not match"):
        tune.save_profile(mismatched)
    wrong_schema = dict(
        profile,
        conditions=dict(conditions, conditions_schema="ironmule.tuned_profile.conditions.v1"),
    )
    with pytest.raises(ModelIdentityError, match="do not match"):
        tune.save_profile(wrong_schema)


@pytest.mark.parametrize("field,value", [("mlx_lm", "0.31.4"),
                                          ("runtime_version", "9.9.9"),
                                          ("os", "different-os")])
def test_profile_identity_drift_is_rejected_but_raw_revalidate_access_remains(monkeypatch, field, value):
    monkeypatch.setattr(tune, "fingerprint", lambda *_args: "fp")
    identity = _identity()
    base = {
        "conditions_schema": tune.PROFILE_CONDITIONS_SCHEMA,
        "fingerprint": "fp", "model_id": "model",
        "model_revision": identity.revision,
        "model_manifest_sha256": identity.model_manifest_sha256,
        "model_architecture": identity.architecture,
        "quantisation": identity.quantisation,
        "quantisation_sha256": identity.quantisation_sha256,
        "tokenizer_sha256": identity.tokenizer_sha256,
        "model_identity_sha256": identity.identity_sha256,
        "chip": "test-chip", "memory_bytes": 1, "gpu_cores": 1,
        "mlx": "0.32.0",
        "mlx_lm": "0.31.3", "runtime_version": CURRENT_VERSION, "os": "test",
        "power_source": "AC", "prompt_tokens": 2, "max_tokens": 2,
        "execution_plan": "single_shot",
    }
    profile = {
        "model_id": "model", "model_identity": identity.to_dict(),
        "conditions": base, "knobs": BASELINE.as_dict(),
    }
    monkeypatch.setattr(
        tune, "_all_profiles", lambda: {f"fp/{identity.identity_sha256}": profile}
    )
    monkeypatch.setattr(
        tune, "conditions", lambda *_args, **_kwargs: dict(base, **{field: value})
    )

    assert tune.load_profile("model", model_identity=identity) is None
    assert tune.load_profile(
        "model", require_compatible=False, model_identity=identity
    ) == profile


def test_profile_model_revision_change_is_not_reused_even_for_raw_revalidation(monkeypatch):
    monkeypatch.setattr(tune, "fingerprint", lambda *_args: "fp")
    stored = _identity(revision="old")
    current = _identity(revision="new")
    profile = {
        "model_id": stored.model_id,
        "model_identity": stored.to_dict(),
        "conditions": {
            "conditions_schema": tune.PROFILE_CONDITIONS_SCHEMA,
            "model_id": stored.model_id,
            "model_identity_sha256": stored.identity_sha256,
        },
        "knobs": BASELINE.as_dict(),
    }
    monkeypatch.setattr(
        tune, "_all_profiles", lambda: {f"fp/{stored.identity_sha256}": profile}
    )
    assert tune.load_profile("model", model_identity=current) is None
    assert tune.load_profile(
        "model", require_compatible=False, model_identity=current
    ) is None
    with pytest.raises(ModelIdentityError, match="model_id"):
        tune.load_profile("org/model", model_identity=current)


def test_static_facts_queries_system_profiler_once_per_process(monkeypatch):
    from ironmule import hw

    calls = []

    def fake_run(command, **_kwargs):
        if command[0] == "system_profiler":
            calls.append(command)
            return types.SimpleNamespace(stdout=json.dumps({
                "SPDisplaysDataType": [{"sppci_cores": "10"}],
            }))
        return types.SimpleNamespace(stdout="")

    fake_mlx = types.SimpleNamespace(__version__="0.32.0",
                                     metal=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(hw.subprocess, "run", fake_run)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx)
    hw._gpu_cores.cache_clear()
    try:
        first = hw.static_facts()
        second = hw.static_facts()
    finally:
        hw._gpu_cores.cache_clear()

    assert first["gpu_cores"] == second["gpu_cores"] == 10
    assert len(calls) == 1


def test_unsupported_candidate_is_typed_and_search_continues(monkeypatch):
    from ironmule.fast import FusionUnsupported

    class FakeEngine:
        def __init__(self, knobs):
            self.knobs = knobs
            self._compiled = None

        @staticmethod
        def needs_reload(old, new):
            return old.fuse_projections != new.fuse_projections

    monkeypatch.setattr(tune, "Engine", FakeEngine)
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    monkeypatch.setattr(tune, "probe", lambda: {"fingerprint": "test"})
    identity = _identity(tune.DEFAULT_MODEL)
    resolved = types.SimpleNamespace(path=Path("/cached/model"), identity=identity)
    monkeypatch.setattr(tune, "resolve_local_model", lambda *_args, **_kwargs: resolved)
    monkeypatch.setattr(
        tune, "load_engine",
        lambda _model, knobs, **_kwargs: (FakeEngine(knobs), object()),
    )
    monkeypatch.setattr(tune, "prompt_ids", lambda _tokenizer, _prompt: [1, 2])
    monkeypatch.setattr(tune, "_eos_ids", lambda _tokenizer: (99,))
    monkeypatch.setattr(
        tune, "conditions",
        lambda *_args, **_kwargs: {"prompt_tokens": 2, "max_tokens": 2},
    )
    monkeypatch.setattr(tune, "save_profile", lambda _profile: None)
    monkeypatch.setattr(tune, "SEARCH", [("fuse_projections", [True]), ("readback_every", [2])])

    def fake_measure(engine, _ids, _max_tokens, _eos, **_kwargs):
        if engine.knobs.fuse_projections:
            raise FusionUnsupported("fusion verified for mlx_lm 0.31.3")
        return {"total_ns": 10, "prefill_ns": 5, "decode_ns": 5,
                "logical_tokens": [7], "deterministic": True, "capacity": 2}

    monkeypatch.setattr(tune, "measure", fake_measure)
    profile = tune.tune(repeats=1, confirm_winner=False)
    unsupported = next(t for t in profile["trials"] if t["knob"] == "fuse_projections")
    continued = next(t for t in profile["trials"] if t["knob"] == "readback_every")
    assert unsupported["disposition"] == "unsupported"
    assert unsupported["verdict"] == "unsupported"
    assert continued["disposition"] in {"accepted", "rejected"}
    assert profile["model_identity"] == identity.to_dict()


def test_only_typed_or_explicitly_unsupported_candidate_errors_are_skippable():
    from ironmule.fast import FusionUnsupported

    assert tune._is_unsupported_candidate(FusionUnsupported("fusion verified for version"))
    assert tune._is_unsupported_candidate(TypeError("unsupported cache type"))
    assert tune._is_unsupported_candidate(ValueError("unsupported cache state"))
    assert not tune._is_unsupported_candidate(RuntimeError("unsupported wording is incidental"))
    assert not tune._is_unsupported_candidate(TypeError("invalid tensor shape"))


def test_metadata_has_qualified_dependency_intervals_and_dynamic_version():
    import tomllib

    metadata = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    project = metadata["project"]
    assert project["dynamic"] == ["version"]
    assert "mlx>=0.32,<0.33" in project["dependencies"]
    assert "mlx-lm>=0.31.3,<0.32" in project["dependencies"]
    assert metadata["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "ironmule._version.__version__"


def test_gpu_busy_does_not_report_a_foreign_command_line(monkeypatch):
    """Callers write this string into evidence records; other processes' args stay out."""
    # `ironmule/__init__.py` rebinds the name `tune` to the function, so the module
    # is only reachable through the import system, not as an attribute of the package.
    tune_module = importlib.import_module("ironmule.tune")

    # Must look like a loaded MLX job, or gpu_busy correctly ignores it.
    secret = "/usr/bin/python3 /Users/someone/private/mlx_train.py --token hunter2"
    line = f"4242 2000000 python {secret}"
    monkeypatch.setattr(
        tune_module.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(stdout=line + "\n", stderr="", returncode=0),
    )

    busy = tune_module.gpu_busy()
    assert busy is not None, "a 2 GB MLX python process must still be detected"
    assert "4242" in busy
    assert "hunter2" not in busy
    assert "private" not in busy
    assert "/Users/" not in busy


def test_stored_gain_comes_from_the_confirmation_not_the_screening():
    """The screening finds the candidate; the paired confirmation measures it.

    Q2 (2026-08-29) stored 0.1457 from a single-process screening while the six-process
    confirmation had measured 0.8568, i.e. 14.32%. Always the weaker of two numbers the
    tuner already had.
    """
    ironmule = importlib.import_module("ironmule")

    profile = {
        "gain": 1 - 0.8568,
        "baseline_ns": 936_890_000,
        "tuned_ns": 800_000_000,
        "confirmation": {"ratio": {"total_ns": {
            "median_ratio": 0.8568, "ci_low": 0.8549, "ci_high": 0.9402,
        }}, "token_identity": True},
    }
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(ironmule, "load_profile", lambda *a, **k: profile)
        monkey.setattr(
            importlib.import_module("ironmule.hw"), "static_facts",
            lambda: {"chip": "Apple M1 Max", "memory_bytes": 32 * 1024**3, "gpu_cores": 32},
        )
        line = ironmule.status()
    finally:
        monkey.undo()

    assert "legacy/unconfirmed profile" in line
    assert "14.32% faster" not in line
    assert "CI" not in line


def test_engine_close_restores_wired_limit_once(monkeypatch):
    from ironmule import hw, runtime

    current = 123
    calls = []

    def set_limit(value):
        nonlocal current
        previous, current = current, value
        calls.append((previous, value))
        return previous

    monkeypatch.setattr(runtime.mx, "set_wired_limit", set_limit)
    monkeypatch.setattr(hw, "static_facts", lambda: {"memory_bytes": 1_000})

    engine = runtime.Engine(object(), object(), Knobs(wired_fraction=0.5))
    assert calls == [(123, 500)]

    engine.close()
    assert calls == [(123, 500), (500, 123)]
    engine.close()
    assert calls == [(123, 500), (500, 123)], "close must be idempotent"


def test_nested_wired_engines_require_lifo_close(monkeypatch):
    from ironmule import hw, runtime

    current = 100
    calls = []

    def set_limit(value):
        nonlocal current
        previous, current = current, value
        calls.append((previous, value))
        return previous

    monkeypatch.setattr(runtime.mx, "set_wired_limit", set_limit)
    monkeypatch.setattr(hw, "static_facts", lambda: {"memory_bytes": 1_000})
    outer = runtime.Engine(object(), object(), Knobs(wired_fraction=0.5))
    inner = runtime.Engine(object(), object(), Knobs(wired_fraction=0.8))
    assert current == 800

    with pytest.raises(RuntimeError, match="LIFO"):
        outer.close()
    assert current == 800 and calls == [(100, 500), (500, 800)]

    inner.close()
    outer.close()
    assert current == 100
    assert calls[-2:] == [(800, 500), (500, 100)]


def test_wired_close_detects_external_mutation_after_restore(monkeypatch):
    from ironmule import hw, runtime

    current = 100

    def set_limit(value):
        nonlocal current
        previous, current = current, value
        return previous

    monkeypatch.setattr(runtime.mx, "set_wired_limit", set_limit)
    monkeypatch.setattr(hw, "static_facts", lambda: {"memory_bytes": 1_000})
    engine = runtime.Engine(object(), object(), Knobs(wired_fraction=0.5))
    runtime.mx.set_wired_limit(900)

    with pytest.raises(RuntimeError, match="changed externally"):
        engine.close()
    assert current == 900, "foreign external limit must be restored"
    engine.close()


def test_engine_context_and_closed_generate_guard():
    from ironmule import runtime

    engine = runtime.Engine(object(), object(), BASELINE)
    with engine as entered:
        assert entered is engine
    with pytest.raises(RuntimeError, match="engine is closed"):
        engine.generate([], 1, (1,))


def test_wired_registration_failure_restores_limit_without_owner_leak(monkeypatch):
    from ironmule import hw, runtime

    current = 100
    calls = []

    def set_limit(value):
        nonlocal current
        previous, current = current, value
        calls.append((previous, value))
        return previous

    monkeypatch.setattr(runtime.mx, "set_wired_limit", set_limit)
    monkeypatch.setattr(hw, "static_facts", lambda: {"memory_bytes": 1_000})
    before = list(runtime._WIRED_LIMIT_OWNERS)
    monkeypatch.setattr(runtime, "_register_wired_owner",
                        lambda *_args: (_ for _ in ()).throw(RuntimeError("register failed")))

    with pytest.raises(RuntimeError, match="register failed"):
        runtime.Engine(object(), object(), Knobs(wired_fraction=0.5))

    assert current == 100
    assert calls == [(100, 500), (500, 100)]
    assert runtime._WIRED_LIMIT_OWNERS == before


def test_tune_closes_each_reloaded_engine_and_on_final_exit(monkeypatch):
    events = []
    identity = _identity(tune.DEFAULT_MODEL)

    class FakeEngine:
        def __init__(self, knobs):
            self.knobs = knobs
            self._compiled = None
            self.closed = False

        @staticmethod
        def needs_reload(old, new):
            return (old.fuse_projections != new.fuse_projections
                    or old.wired_fraction != new.wired_fraction)

        def close(self):
            assert not self.closed
            self.closed = True
            events.append("close")

    def fake_load(_model, knobs, **_kwargs):
        events.append(("load", knobs.key()))
        return FakeEngine(knobs), object()

    monkeypatch.setattr(tune, "Engine", FakeEngine)
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    monkeypatch.setattr(tune, "probe", lambda: {"fingerprint": "test"})
    monkeypatch.setattr(tune, "resolve_local_model",
                        lambda *_args, **_kwargs: types.SimpleNamespace(
                            path=Path("/cached/model"), identity=identity))
    monkeypatch.setattr(tune, "load_engine", fake_load)
    monkeypatch.setattr(tune, "prompt_ids", lambda _tokenizer, _prompt: [1, 2])
    monkeypatch.setattr(tune, "_eos_ids", lambda _tokenizer: (99,))
    monkeypatch.setattr(tune, "conditions", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(tune, "save_profile", lambda _profile: None)
    monkeypatch.setattr(tune, "SEARCH", [
        ("wired_fraction", [0.6]), ("fuse_projections", [True]),
    ])

    def fake_measure(engine, _ids, _max_tokens, _eos, **_kwargs):
        total = 80 if engine.knobs.fuse_projections else 90 if engine.knobs.wired_fraction else 100
        return {"total_ns": total, "prefill_ns": total // 2, "decode_ns": total // 2,
                "logical_tokens": [7], "deterministic": True, "capacity": 2}

    monkeypatch.setattr(tune, "measure", fake_measure)
    profile = tune.tune(repeats=1, confirm_winner=False)

    assert events[0][0] == "load"
    assert events.count("close") == 3
    assert all(events[index] == "close" for index in (1, 3, 5))
    assert profile["knobs"]["fuse_projections"] is True


def test_tune_closes_engine_when_measurement_raises(monkeypatch):
    closed = []
    identity = _identity(tune.DEFAULT_MODEL)

    class FakeEngine:
        def __init__(self, knobs):
            self.knobs = knobs
            self._compiled = None

        @staticmethod
        def needs_reload(_old, _new):
            return False

        def close(self):
            closed.append(True)

    monkeypatch.setattr(tune, "Engine", FakeEngine)
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    monkeypatch.setattr(tune, "probe", lambda: {"fingerprint": "test"})
    monkeypatch.setattr(tune, "resolve_local_model",
                        lambda *_args, **_kwargs: types.SimpleNamespace(
                            path=Path("/cached/model"), identity=identity))
    monkeypatch.setattr(tune, "load_engine",
                        lambda _model, knobs, **_kwargs: (FakeEngine(knobs), object()))
    monkeypatch.setattr(tune, "prompt_ids", lambda _tokenizer, _prompt: [1, 2])
    monkeypatch.setattr(tune, "_eos_ids", lambda _tokenizer: (99,))
    monkeypatch.setattr(tune, "SEARCH", [("readback_every", [2])])

    calls = 0

    def failing_measure(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("measurement failed")
        return {"total_ns": 100, "prefill_ns": 50, "decode_ns": 50,
                "logical_tokens": [7], "deterministic": True, "capacity": 2}

    monkeypatch.setattr(tune, "measure", failing_measure)
    with pytest.raises(RuntimeError, match="measurement failed"):
        tune.tune(repeats=1, confirm_winner=False)
    assert closed == [True]


def test_rejected_confirmation_stores_baseline_without_gain(monkeypatch):
    identity = _identity(tune.DEFAULT_MODEL)
    captured = {}

    class FakeEngine:
        def __init__(self, knobs):
            self.knobs = knobs
            self._compiled = None

        @staticmethod
        def needs_reload(_old, _new):
            return False

        def close(self):
            pass

    monkeypatch.setattr(tune, "Engine", FakeEngine)
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    monkeypatch.setattr(tune, "probe", lambda: {"fingerprint": "test"})
    monkeypatch.setattr(tune, "resolve_local_model",
                        lambda *_args, **_kwargs: types.SimpleNamespace(
                            path=Path("/cached/model"), identity=identity))
    monkeypatch.setattr(tune, "load_engine",
                        lambda _model, knobs, **_kwargs: (FakeEngine(knobs), object()))
    monkeypatch.setattr(tune, "prompt_ids", lambda _tokenizer, _prompt: [1, 2])
    monkeypatch.setattr(tune, "_eos_ids", lambda _tokenizer: (99,))
    monkeypatch.setattr(tune, "conditions", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(tune, "save_profile", lambda profile: captured.update(profile))
    monkeypatch.setattr(tune, "SEARCH", [("readback_every", [2])])
    measurements = iter((100, 80))

    def measure_screening(*_args, **_kwargs):
        total = next(measurements)
        return {"total_ns": total, "prefill_ns": total // 2, "decode_ns": total // 2,
                "logical_tokens": [7], "deterministic": True, "capacity": 2}

    monkeypatch.setattr(tune, "measure", measure_screening)
    monkeypatch.setattr(tune, "confirm",
                        lambda *_args, **_kwargs: _confirmation(token_identity=False))

    result = tune.tune(repeats=1, confirm_winner=True)

    assert result["knobs"] == BASELINE.as_dict()
    assert result["tuned_ns"] == result["baseline_ns"]
    assert result["gain"] == 0.0
    assert result["confirmation"]["accepted"] is False
    assert result["confirmation"]["rejection_reason"] == "token_identity"
    assert result["confirmation_evidence"]
    assert result["confirmation"]["evidence_sha256"] == tune._confirmation_evidence_sha256(
        result["confirmation_evidence"]
    )
    assert result["confirmation_candidate_knobs"] == Knobs(readback_every=2).as_dict()
    assert captured["confirmation_candidate_knobs"] == Knobs(readback_every=2).as_dict()
    assert captured["confirmation"]["accepted"] is False


def test_accepted_confirmation_stores_candidate_gain(monkeypatch):
    identity = _identity(tune.DEFAULT_MODEL)

    class FakeEngine:
        def __init__(self, knobs):
            self.knobs = knobs
            self._compiled = None

        @staticmethod
        def needs_reload(_old, _new):
            return False

        def close(self):
            pass

    monkeypatch.setattr(tune, "Engine", FakeEngine)
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    monkeypatch.setattr(tune, "probe", lambda: {"fingerprint": "test"})
    monkeypatch.setattr(tune, "resolve_local_model",
                        lambda *_args, **_kwargs: types.SimpleNamespace(
                            path=Path("/cached/model"), identity=identity))
    monkeypatch.setattr(tune, "load_engine",
                        lambda _model, knobs, **_kwargs: (FakeEngine(knobs), object()))
    monkeypatch.setattr(tune, "prompt_ids", lambda _tokenizer, _prompt: [1, 2])
    monkeypatch.setattr(tune, "_eos_ids", lambda _tokenizer: (99,))
    monkeypatch.setattr(tune, "conditions", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(tune, "save_profile", lambda _profile: None)
    monkeypatch.setattr(tune, "SEARCH", [("readback_every", [2])])
    measurements = iter((100, 80))

    def measure_screening(*_args, **_kwargs):
        total = next(measurements)
        return {"total_ns": total, "prefill_ns": total // 2, "decode_ns": total // 2,
                "logical_tokens": [7], "deterministic": True, "capacity": 2}

    monkeypatch.setattr(tune, "measure", measure_screening)
    monkeypatch.setattr(tune, "confirm",
                        lambda *_args, **_kwargs: _confirmation())

    result = tune.tune(repeats=1, confirm_winner=True)

    assert result["knobs"]["readback_every"] == 2
    assert result["tuned_ns"] == 80
    assert result["gain"] == pytest.approx(0.2)
    assert result["confirmation"]["accepted"] is True
    assert result["confirmation_evidence"]
    assert result["confirmation"]["evidence_sha256"] == tune._confirmation_evidence_sha256(
        result["confirmation_evidence"]
    )
    assert result["confirmation_candidate_knobs"] == result["knobs"]


def test_confirmation_decision_binds_both_full_arm_knob_mappings():
    evidence = _confirmation()
    candidate = Knobs(readback_every=2)
    assert tune._confirmation_decision(
        evidence, expected_baseline=BASELINE, expected_candidate=candidate
    ) == (True, None)

    empty = dict(evidence, arms={})
    assert tune._confirmation_decision(
        empty, expected_baseline=BASELINE, expected_candidate=candidate
    ) == (False, "invalid_confirmation")

    swapped = dict(evidence, arms={
        "baseline": candidate.as_dict(), "candidate": BASELINE.as_dict(),
    })
    assert tune._confirmation_decision(
        swapped, expected_baseline=BASELINE, expected_candidate=candidate
    ) == (False, "invalid_confirmation")

    assert tune._confirmation_decision(
        evidence, expected_baseline=BASELINE,
        expected_candidate=Knobs(fused_argmax=True),
    ) == (False, "invalid_confirmation")


def _confirmation(*, token_identity=True, token_count_identity=True,
                  stop_reason_identity=True, deterministic=True, median=0.8,
                  ci_low=0.7, ci_high=0.9, pairs=None, extra=False):
    import importlib

    ab = importlib.import_module("ironmule.ab")

    def arm(name):
        logical = [[7] for _ in range(7)]
        physical = [[7] for _ in range(7)]
        counts = [{"logical": 1, "physical": 1} for _ in range(7)]
        stops = ["length" for _ in range(7)]
        if name == "candidate" and not token_identity:
            logical = [[8] for _ in range(7)]
        if name == "candidate" and not token_count_identity:
            physical = [[7, 8] for _ in range(7)]
            counts = [{"logical": 1, "physical": 2} for _ in range(7)]
        if name == "candidate" and not stop_reason_identity:
            stops = ["eos" for _ in range(7)]
        if name == "candidate" and not deterministic:
            logical[-1] = [8]
        total = [1.0 for _ in range(7)]
        if name == "candidate":
            total = [0.8 for _ in range(7)]
        return {
            "total_ns": total, "prefill_ns": [1.0 for _ in range(7)],
            "decode_ns": [1.0 for _ in range(7)],
            "logical_tokens": logical[0],
            "logical_tokens_per_repeat": logical,
            "physical_tokens_per_repeat": physical,
            "token_counts": counts, "stop_reasons": stops,
            "capacities": [64 for _ in range(7)],
            "deterministic": deterministic if name == "candidate" else True,
            "decode_steps": len(physical[0]) - 1,
            "prompt_tokens": 1, "mlx_peak_bytes": 10,
        }

    raw = []
    for index in range(6):
        raw.append({
            "pid": index + 1,
            "arms": {"baseline": arm("baseline"), "candidate": arm("candidate")},
            "order": ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"],
            "mlx_peak_bytes": 10,
            "guard": {"version": "ironmule.q3f_child_guard.v1", "installed": True, "events": []},
        })
    per_arm = {
        name: {
            metric: ab.summarise([
                statistics.median(child["arms"][name][metric]) for child in raw
            ])
            for metric in ("total_ns", "prefill_ns", "decode_ns")
        }
        for name in ("baseline", "candidate")
    }
    ratios = {
        metric: ab.paired_ratio(
            [statistics.median(child["arms"]["candidate"][metric]) for child in raw],
            [statistics.median(child["arms"]["baseline"][metric]) for child in raw],
        )
        for metric in ("total_ns", "prefill_ns", "decode_ns")
    }
    if any(value != default for value, default in ((median, 0.8), (ci_low, 0.7), (ci_high, 0.9))) or pairs is not None:
        total = ratios["total_ns"]
        total["median_ratio"] = median
        total["ci_low"] = ci_low
        total["ci_high"] = ci_high
        if pairs is not None:
            total["pairs"] = pairs
    if extra:
        ratios["total_ns"]["unexpected"] = 1
    return {
        "token_identity": token_identity,
        "token_count_identity": token_count_identity,
        "stop_reason_identity": stop_reason_identity,
        "deterministic": deterministic,
        "processes": 6, "repeats": 7, "warmup": 2,
        "raw": raw, "arms": {
            "baseline": BASELINE.as_dict(),
            "candidate": Knobs(readback_every=2).as_dict(),
        }, "per_arm": per_arm,
        "reference_tokens": [7], "ratios": {"candidate/baseline": ratios},
    }


def test_archived_q2_confirmation_full_ratio_shape_remains_legacy_valid():
    stored = {
        "ratio": _confirmation()["ratios"]["candidate/baseline"],
        "token_identity": True,
    }
    assert tune._legacy_confirmation_valid(stored)


def test_stored_accepted_confirmation_requires_ci_below_one_and_median_below_one():
    import copy

    evidence = _confirmation()
    ratio = evidence["ratios"]["candidate/baseline"]
    stored = {
        "ratio": ratio, "token_identity": True,
        "token_count_identity": True, "stop_reason_identity": True,
        "deterministic": True,
        "accepted": True, "rejection_reason": None,
        "evidence_sha256": tune._confirmation_evidence_sha256(evidence),
    }
    expected_candidate = Knobs(readback_every=2)
    assert tune._stored_confirmation_valid(
        stored, evidence, expected_candidate=expected_candidate
    )
    assert not tune._stored_confirmation_valid(
        stored, {"value": float("inf")}, expected_candidate=expected_candidate
    )

    boundary = copy.deepcopy(stored)
    boundary["ratio"]["total_ns"]["ci_high"] = 1.0
    assert not tune._stored_confirmation_valid(
        boundary, evidence, expected_candidate=expected_candidate
    )

    median = copy.deepcopy(stored)
    median["ratio"]["total_ns"]["median_ratio"] = 1.0
    median["ratio"]["total_ns"]["ci_high"] = 1.1
    assert not tune._stored_confirmation_valid(
        median, evidence, expected_candidate=expected_candidate
    )


@pytest.mark.parametrize(
    "confirmation,reason",
    [
        ({}, "invalid_confirmation"),
        (_confirmation(token_identity=False), "token_identity"),
        (_confirmation(ci_high=1.0), "invalid_confirmation"),
        (_confirmation(median=float("nan")), "invalid_confirmation"),
        (_confirmation(median=0.0), "invalid_confirmation"),
        (_confirmation(pairs=[]), "invalid_confirmation"),
        (_confirmation(pairs=[0.8, 0.0]), "invalid_confirmation"),
        (_confirmation(median=0.6, ci_low=0.7), "invalid_confirmation"),
        (_confirmation(extra=True), "invalid_confirmation"),
        (_confirmation(token_count_identity=False), "token_count_identity"),
        (_confirmation(stop_reason_identity=False), "stop_reason_identity"),
        (_confirmation(deterministic=False), "determinism"),
    ],
)
def test_confirmation_decision_rejects_invalid_or_boundary_evidence(confirmation, reason):
    assert tune._confirmation_decision(
        confirmation, expected_baseline=BASELINE,
        expected_candidate=Knobs(readback_every=2),
    ) == (False, reason)


def test_status_suppresses_rejected_confirmation_gain(monkeypatch):
    import ironmule
    from ironmule import hw

    monkeypatch.setattr(hw, "static_facts", lambda: {
        "chip": "Apple M1 Max", "memory_bytes": 32 * 1024**3, "gpu_cores": 32,
    })
    monkeypatch.setattr(ironmule, "load_profile", lambda *_args, **_kwargs: {
        "gain": 0.4, "baseline_ns": 100, "tuned_ns": 60,
        "confirmation": {
            "accepted": False, "rejection_reason": "token_identity",
            "ratio": {"total_ns": {"median_ratio": 0.6, "ci_low": 0.5, "ci_high": 0.7}},
        },
    })

    line = ironmule.status()
    assert "BASELINE retained" in line
    assert "confirmation rejected (token_identity)" in line
    assert "40.00% faster" not in line
    assert "CI" not in line


def test_status_does_not_claim_gain_for_screening_only_profile(monkeypatch):
    import ironmule
    from ironmule import hw

    monkeypatch.setattr(hw, "static_facts", lambda: {
        "chip": "Apple M1 Max", "memory_bytes": 32 * 1024**3, "gpu_cores": 32,
    })
    monkeypatch.setattr(ironmule, "load_profile", lambda *_args, **_kwargs: {
        "gain": 0.4, "baseline_ns": 100, "tuned_ns": 60, "confirmation": None,
    })

    line = ironmule.status()
    assert "screening-only" in line
    assert "no confirmed speedup" in line
    assert "40.00% faster" not in line


def test_status_accepted_confirmation_uses_paired_gain_only(monkeypatch):
    import ironmule
    from ironmule import hw

    monkeypatch.setattr(hw, "static_facts", lambda: {
        "chip": "Apple M1 Max", "memory_bytes": 32 * 1024**3, "gpu_cores": 32,
    })
    evidence = _confirmation()
    ratio = evidence["ratios"]["candidate/baseline"]
    monkeypatch.setattr(ironmule, "load_profile", lambda *_args, **_kwargs: {
        "gain": 0.4, "baseline_ns": 100, "tuned_ns": 60,
        "confirmation": {
            "accepted": True, "rejection_reason": None,
            "token_identity": True, "token_count_identity": True,
            "stop_reason_identity": True, "deterministic": True, "ratio": ratio,
            "evidence_sha256": tune._confirmation_evidence_sha256(evidence),
        },
        "confirmation_candidate_knobs": Knobs(readback_every=2).as_dict(),
        "confirmation_evidence": evidence,
    })

    line = ironmule.status()
    assert "20.00% faster in paired confirmation" in line
    assert "100.0 -> 60.0 ms" not in line
    assert "CI" in line


def test_status_rejects_malformed_accepted_confirmation(monkeypatch):
    import ironmule
    from ironmule import hw

    monkeypatch.setattr(hw, "static_facts", lambda: {
        "chip": "Apple M1 Max", "memory_bytes": 32 * 1024**3, "gpu_cores": 32,
    })
    monkeypatch.setattr(ironmule, "load_profile", lambda *_args, **_kwargs: {
        "gain": 0.4, "baseline_ns": 100, "tuned_ns": 60,
        "confirmation": {"accepted": True, "ratio": {}},
    })

    line = ironmule.status()
    assert "BASELINE retained; confirmation invalid" in line
    assert "40.00% faster" not in line


def test_runtime_close_and_context_forward_to_engine(monkeypatch):
    from ironmule.service import Runtime

    class FakeEngine:
        def __init__(self):
            self.closes = 0

        def close(self):
            self.closes += 1

    engine = FakeEngine()
    runtime = Runtime.__new__(Runtime)
    runtime.engine = engine
    assert runtime.__enter__() is runtime
    runtime.close()
    assert engine.closes == 1
    assert runtime.__exit__(None, None, None) is False
    assert engine.closes == 2


def test_runtime_init_closes_engine_on_identity_and_backend_failures(monkeypatch):
    from ironmule import service
    from ironmule.model_identity import ModelIdentityError

    class FakeEngine:
        def __init__(self, loaded_identity):
            self.model_identity = loaded_identity
            self.closes = 0

        def close(self):
            self.closes += 1

    tokenizer = types.SimpleNamespace(eos_token_ids=(1,), eos_token_id=None)
    loaded = _identity("org/model", "loaded")
    conflict = FakeEngine(loaded)
    with pytest.raises(ModelIdentityError, match="Engine identity"):
        service.Runtime(conflict, tokenizer, model_identity=_identity("org/model", "other"))
    assert conflict.closes == 1

    backend_failure = FakeEngine(loaded)
    monkeypatch.setattr(service, "MLXBackend",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("backend failed")))
    with pytest.raises(RuntimeError, match="backend failed"):
        service.Runtime(backend_failure, tokenizer)
    assert backend_failure.closes == 1


def test_runtime_init_preserves_cleanup_failure_as_exception_note(monkeypatch):
    from ironmule import service

    class BadEngine:
        model_identity = None

        def close(self):
            raise RuntimeError("close failed")

    monkeypatch.setattr(service, "MLXBackend",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("backend failed")))
    tokenizer = types.SimpleNamespace(eos_token_ids=(1,), eos_token_id=None)
    with pytest.raises(ValueError, match="backend failed") as error:
        service.Runtime(BadEngine(), tokenizer)
    assert any("cleanup failed" in note for note in error.value.__notes__)


def test_ab_child_closes_each_arm_in_order_and_on_exception(monkeypatch):
    import ironmule.ab as ab
    import mlx.core as mx

    events = []
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: events.append("reset"))
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 10)

    class FakeEngine:
        def __init__(self, name):
            self.name = name

        def generate(self, _ids, _max_tokens, _eos):
            events.append(("generate", self.name))
            if self.name == "bad":
                raise RuntimeError("generation failed")
            return {
                "total_ns": 1, "prefill_ns": 1, "decode_ns": 0,
                "logical_tokens": [7], "physical_tokens": [7], "capacity": 64,
            }

        def close(self):
            events.append(("close", self.name))

    tune_module = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(
        tune_module, "load_engine",
        lambda _model, knobs, **_kwargs: (
            FakeEngine("bad" if knobs.readback_every == 2 else "good"), object()
        ),
    )
    monkeypatch.setattr(tune_module, "prompt_ids", lambda _tok, _prompt: [1])
    monkeypatch.setattr(tune_module, "_eos_ids", lambda _tok: (99,))

    spec = {
        "order": ["good", "bad"],
        "arms": {
            "good": BASELINE.as_dict(),
            "bad": Knobs(readback_every=2).as_dict(),
        },
        "warmup": 1,
        "repeats": 1,
        "max_tokens": 2,
    }
    with pytest.raises(RuntimeError, match="generation failed"):
        ab._child(spec)

    assert events[:5] == [
        "reset", ("generate", "good"), ("generate", "good"),
        ("close", "good"), "reset",
    ]
    assert events[-1] == ("close", "bad")


def test_ab_child_preserves_generation_error_when_close_also_fails(monkeypatch):
    import ironmule.ab as ab
    import mlx.core as mx

    monkeypatch.setattr(mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 10)

    class BadEngine:
        def generate(self, _ids, _max_tokens, _eos):
            raise ValueError("generation failed")

        def close(self):
            raise RuntimeError("close failed")

    tune_module = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune_module, "load_engine", lambda *_args, **_kwargs: (BadEngine(), object()))
    monkeypatch.setattr(tune_module, "prompt_ids", lambda _tok, _prompt: [1])
    monkeypatch.setattr(tune_module, "_eos_ids", lambda _tok: (99,))

    spec = {
        "order": ["bad"], "arms": {"bad": BASELINE.as_dict()},
        "warmup": 1, "repeats": 1, "max_tokens": 2,
    }
    with pytest.raises(ValueError, match="generation failed") as error:
        ab._child(spec)
    assert any("cleanup failed" in note for note in error.value.__notes__)
