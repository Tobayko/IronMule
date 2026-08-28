import ast
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from ironmule.evidence import (
    ActorRole,
    ArtifactRef,
    ClosedInterval,
    CorrectnessEvidence,
    DomainMatchStatus,
    EvidenceQuality,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceValidationError,
    ExecutionStrategy,
    MetricEvidence,
    MetricSource,
    ResourceEvidence,
    StatisticsEvidence,
    TrustedExecutionProfile,
    ValidityDomain,
    canonical_json,
    canonical_sha256,
    domain_from_fingerprint,
    evidence_from_b27_public_cell,
    strategy_from_existing,
)


DIGESTS = {letter: letter * 64 for letter in "abcdef0123456789"}


def strategy(**updates):
    values = {
        "semantic_class": "exact",
        "prefill_policy": "strict_one_shot",
        "decode_policy": "greedy_fixed_cache",
        "cache_policy": "fixed_kv",
        "scheduling_policy": "ready_order_fair_rotation",
        "grouping_policy": "independent_batch1",
        "grouping_width": 4,
        "synchronization_policy": "group_completion",
        "memory_policy": "capacity_ceiling_8192",
        "compile_graph_policy": "baseline",
        "prefix_reuse_policy": "none",
        "plan_kind": "strict_one_shot",
        "service_mode": "throughput",
        "knobs_key": "compiled_fixed_cache=False",
        "implementation_revision": DIGESTS["a"],
    }
    values.update(updates)
    return ExecutionStrategy(**values)


def domain(**updates):
    values = {
        "apple_chip": "Apple M1 Max",
        "machine": "arm64",
        "ram_bytes": 32 * 1024**3,
        "gpu_cores": 32,
        "gpu_configuration": "32 integrated GPU cores",
        "macos_build": "26.5.2",
        "python_version": "3.12.13",
        "mlx_version": "0.32.0",
        "mlx_lm_version": "0.31.3",
        "runtime_version": "0.1.0",
        "model_id": "mlx-community/gemma-3-4b-it-4bit",
        "model_revision": "93724907d4ed1745d2fe50baadf3b0b01a65abf2",
        "model_manifest_sha256": DIGESTS["b"],
        "model_architecture": "gemma3",
        "tokenizer_sha256": DIGESTS["c"],
        "quantization_bits": 4,
        "quantization_group_size": 64,
        "quantization_format": "grouped_affine",
        "cache_family": "fixed_kv",
        "cache_layer_pattern": "all_kv",
        "capacity_policy": "longest_prompt_plus_output_le_8192",
        "plan_kind": "strict_one_shot",
        "prompt_bucket": ClosedInterval(1, 512),
        "context_bucket": ClosedInterval(1, 2048),
        "output_bucket": ClosedInterval(1, 128),
        "concurrency_bucket": ClosedInterval(1, 8),
        "arrival_pattern": "all_at_once",
        "workload_class": "concurrent_greedy_generation",
        "power_source": "AC",
        "low_power_mode": False,
        "thermal_state": "nominal",
        "swap_class": "zero_preflight",
    }
    values.update(updates)
    return ValidityDomain(**values)


def correctness(ok=True):
    return CorrectnessEvidence(
        comparison_performed=ok,
        token_identity=ok,
        stop_reason_identity=ok,
        token_count_identity=ok,
        deterministic=ok,
        state_identity=ok,
        quality_class="exact",
    )


def resources(complete=True):
    return ResourceEvidence(
        mlx_active_memory_bytes=100 if complete else None,
        mlx_peak_memory_bytes=200 if complete else None,
        rss_peak_bytes=300 if complete else None,
        swap_before_bytes=0 if complete else None,
        swap_after_bytes=0 if complete else None,
        timeout=False,
        crash_free=True,
        fallbacks=0,
        worker_status="complete",
        gates_passed=complete,
    )


def statistics(complete=True):
    return StatisticsEvidence(
        warmup_repeats=2 if complete else 0,
        measured_repeats=6 if complete else 0,
        raw_sample_count=12 if complete else 0,
        paired=True,
        arm_order=("AB", "BA") if complete else (),
        estimator="paired_median_ratio",
        uncertainty_method="bootstrap" if complete else "unavailable",
        confidence_level=0.95 if complete else None,
    )


def qualified_record(candidate=None, validity=None, **updates):
    candidate = candidate or strategy()
    validity = validity or domain()
    values = {
        "status": EvidenceStatus.QUALIFIED,
        "status_owner": ActorRole.EVALUATOR,
        "researcher_id": "researcher-1",
        "reviewer_id": "reviewer-1",
        "evaluator_id": "evaluator-1",
        "diagnostic_verdict": "QUALIFIED_BY_FROZEN_GATES",
        "baseline_strategy_id": DIGESTS["d"],
        "candidate_strategy_id": candidate.strategy_id,
        "validity_domain_id": validity.domain_id,
        "experiment_id": "B27-synthetic",
        "study_id": "B27",
        "preregistration_sha256": DIGESTS["e"],
        "reviewer_record_sha256": DIGESTS["f"],
        "code_sha256": DIGESTS["0"],
        "model_sha256": DIGESTS["1"],
        "environment_sha256": DIGESTS["2"],
        "workload_sha256": DIGESTS["3"],
        "raw_artifacts": (ArtifactRef(
            "research/raw/B27.json", DIGESTS["4"], EvidenceQuality.RAW_SAMPLES
        ),),
        "metrics": (
            MetricEvidence("outer_wall_ms", "ms", MetricSource.MEASURED,
                           (10.0, 9.8, 10.1), 10.0, 10.1, 9.7, 10.2),
            MetricEvidence("physical_tokens_per_second", "tokens/s", MetricSource.MEASURED,
                           (28.0, 28.2, 27.9), 28.0, 28.2, 27.8, 28.3),
        ),
        "correctness": correctness(),
        "resources": resources(),
        "statistics": statistics(),
        "evidence_quality": EvidenceQuality.RAW_SAMPLES,
        "recorded_at": "2026-08-28T12:00:00Z",
    }
    values.update(updates)
    return EvidenceRecord(**values)


def hypothesis_record(candidate=None, validity=None, **updates):
    candidate = candidate or strategy()
    validity = validity or domain()
    values = {
        "status": EvidenceStatus.HYPOTHESIS,
        "status_owner": ActorRole.RESEARCHER,
        "researcher_id": "researcher-1",
        "reviewer_id": "",
        "evaluator_id": "",
        "diagnostic_verdict": "UNMEASURED",
        "baseline_strategy_id": DIGESTS["d"],
        "candidate_strategy_id": candidate.strategy_id,
        "validity_domain_id": validity.domain_id,
        "experiment_id": "B27-hypothesis",
        "study_id": "B27",
        "preregistration_sha256": DIGESTS["e"],
        "reviewer_record_sha256": None,
        "code_sha256": DIGESTS["0"],
        "model_sha256": DIGESTS["1"],
        "environment_sha256": DIGESTS["2"],
        "workload_sha256": DIGESTS["3"],
        "raw_artifacts": (),
        "metrics": (),
        "correctness": correctness(False),
        "resources": resources(False),
        "statistics": statistics(False),
        "evidence_quality": EvidenceQuality.PARTIAL,
        "recorded_at": "2026-08-28T12:00:00+00:00",
    }
    values.update(updates)
    return EvidenceRecord(**values)


def test_status_vocabulary_is_closed_and_exact():
    assert [item.value for item in EvidenceStatus] == [
        "HYPOTHESIS", "QUALIFIED", "REJECTED", "INCONCLUSIVE", "INVALIDATED",
        "REVALIDATION_REQUIRED",
    ]


def test_canonical_json_is_stable_and_rejects_non_finite_values():
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert canonical_sha256({"value": True}) != canonical_sha256({"value": 1})
    with pytest.raises(EvidenceValidationError, match="NaN"):
        canonical_json({"value": float("nan")})
    with pytest.raises(EvidenceValidationError, match="keys must be strings"):
        canonical_json({"a": 1, 2: "b"})


def test_strategy_is_immutable_stable_strict_and_has_no_execution_surface():
    first = strategy()
    second = strategy()
    assert first.strategy_id == second.strategy_id
    assert ExecutionStrategy.from_dict(first.to_dict()) == first
    assert not hasattr(first, "run") and not hasattr(first, "select")
    with pytest.raises(FrozenInstanceError):
        first.plan_kind = "changed"
    changed = first.to_dict()
    changed["unknown"] = True
    with pytest.raises(EvidenceValidationError, match="unknown"):
        ExecutionStrategy.from_dict(changed)
    forged = first.to_dict()
    forged["strategy_id"] = DIGESTS["f"]
    with pytest.raises(EvidenceValidationError, match="strategy_id"):
        ExecutionStrategy.from_dict(forged)


def test_existing_strategy_adapter_preserves_caller_owned_ids():
    adapted = strategy_from_existing(
        plan_kind="reusable_session", service_mode="throughput",
        knobs_key="alpha=1|beta=False", implementation_revision=DIGESTS["a"],
        prefill_policy="chunked_declared_prefix", decode_policy="greedy_fixed_cache",
        cache_policy="fixed_kv", scheduling_policy="ready_order_fair_rotation",
        grouping_policy="independent_batch1", grouping_width=4,
        synchronization_policy="group_completion", memory_policy="ceiling_8192",
        compile_graph_policy="baseline", prefix_reuse_policy="declared_prefix",
    )
    assert adapted.plan_kind == "reusable_session"
    assert adapted.service_mode == "throughput"
    assert adapted.knobs_key == "alpha=1|beta=False"


def test_domain_exact_identity_and_closed_buckets_fail_closed():
    stored = domain()
    current = domain(
        prompt_bucket=ClosedInterval(322, 322),
        context_bucket=ClosedInterval(322, 322),
        output_bucket=ClosedInterval(48, 48),
        concurrency_bucket=ClosedInterval(6, 6),
    )
    assert stored.match(current).status is DomainMatchStatus.MATCH

    drifted = domain(
        prompt_bucket=ClosedInterval(900, 900),
        context_bucket=ClosedInterval(900, 900),
        output_bucket=ClosedInterval(48, 48),
        concurrency_bucket=ClosedInterval(6, 6),
        model_revision="different-revision",
    )
    match = stored.match(drifted)
    assert match.status is DomainMatchStatus.REVALIDATION_REQUIRED
    assert set(match.mismatches) == {"model_revision", "prompt_bucket"}


def test_domain_roundtrip_rejects_unknown_identity_and_forged_digest():
    original = domain()
    assert ValidityDomain.from_dict(original.to_dict()) == original
    with pytest.raises(EvidenceValidationError, match="must be known"):
        domain(thermal_state="unknown")
    forged = original.to_dict()
    forged["domain_id"] = DIGESTS["f"]
    with pytest.raises(EvidenceValidationError, match="domain_id"):
        ValidityDomain.from_dict(forged)


def test_fingerprint_adapter_requires_missing_identity_as_explicit_inputs():
    fingerprint = {
        "chip": "Apple M1 Max", "machine": "arm64", "memory_bytes": 32 * 1024**3,
        "gpu_cores": 32, "os": "Darwin 26.5.2", "mlx": "0.32.0",
        "mlx_lm": "0.31.3", "runtime_version": "0.1.0",
        "model_id": "mlx-community/gemma-3-4b-it-4bit",
        "quantisation": {"bits": 4, "group_size": 64},
        "execution_plan": "strict_one_shot", "power_source": "AC",
    }
    adapted = domain_from_fingerprint(
        fingerprint, model_revision="rev", model_manifest_sha256=DIGESTS["b"],
        model_architecture="gemma3", tokenizer_sha256=DIGESTS["c"],
        python_version="3.12.13", gpu_configuration="32 integrated GPU cores",
        quantization_format="grouped_affine", cache_family="fixed_kv",
        cache_layer_pattern="all_kv", capacity_policy="ceiling_8192",
        prompt_bucket=ClosedInterval(1, 512), context_bucket=ClosedInterval(1, 2048),
        output_bucket=ClosedInterval(1, 128), concurrency_bucket=ClosedInterval(1, 8),
        arrival_pattern="all_at_once", workload_class="greedy", low_power_mode=False,
        thermal_state="nominal", swap_class="zero_preflight",
    )
    assert adapted.model_revision == "rev"
    assert adapted.quantization_bits == 4
    broken = dict(fingerprint)
    del broken["model_id"]
    with pytest.raises(EvidenceValidationError, match="model_id"):
        domain_from_fingerprint(
            broken, model_revision="rev", model_manifest_sha256=DIGESTS["b"],
            model_architecture="gemma3", tokenizer_sha256=DIGESTS["c"],
            python_version="3.12.13", gpu_configuration="32 cores",
            quantization_format="grouped_affine", cache_family="fixed_kv",
            cache_layer_pattern="all_kv", capacity_policy="ceiling",
            prompt_bucket=ClosedInterval(1, 1), context_bucket=ClosedInterval(1, 1),
            output_bucket=ClosedInterval(1, 1), concurrency_bucket=ClosedInterval(1, 1),
            arrival_pattern="all", workload_class="greedy", low_power_mode=False,
            thermal_state="nominal", swap_class="zero",
        )


def test_artifact_reference_rejects_absolute_local_paths():
    with pytest.raises(EvidenceValidationError, match="absolute"):
        ArtifactRef("/Users/person/raw.json", DIGESTS["a"], EvidenceQuality.RAW_SAMPLES)
    relative = ArtifactRef("research/raw/result.json", DIGESTS["a"], EvidenceQuality.RAW_SAMPLES)
    assert ArtifactRef.from_dict(relative.to_dict()) == relative


def test_status_ownership_prevents_self_qualification():
    hypothesis = hypothesis_record()
    assert hypothesis.status_owner is ActorRole.RESEARCHER
    with pytest.raises(EvidenceValidationError, match="evaluator-owned"):
        qualified_record(status_owner=ActorRole.REVIEWER)
    with pytest.raises(EvidenceValidationError, match="distinct"):
        qualified_record(evaluator_id="researcher-1")
    with pytest.raises(EvidenceValidationError, match="without reviewer/evaluator"):
        hypothesis_record(reviewer_id="reviewer-1")


def test_qualified_evidence_requires_raw_samples_correctness_resources_and_uncertainty():
    record = qualified_record()
    assert EvidenceRecord.from_dict(record.to_dict()) == record
    with pytest.raises(EvidenceValidationError, match="RAW_SAMPLES"):
        qualified_record(evidence_quality=EvidenceQuality.SUMMARY_ONLY)
    with pytest.raises(EvidenceValidationError, match="correctness"):
        qualified_record(correctness=CorrectnessEvidence(
            True, True, True, True, True, False, "exact"
        ))
    with pytest.raises(EvidenceValidationError, match="resource"):
        qualified_record(resources=resources(False))
    with pytest.raises(EvidenceValidationError, match="uncertainty"):
        qualified_record(statistics=statistics(False))
    incomplete_metrics = (
        MetricEvidence("outer_wall_ms", "ms", MetricSource.MEASURED,
                       (10.0, 10.1), 10.0, None, 9.9, 10.2),
        MetricEvidence("physical_tokens_per_second", "tokens/s", MetricSource.MEASURED,
                       (28.0, 28.1), 28.0, 28.1, 27.9, 28.2),
    )
    with pytest.raises(EvidenceValidationError, match="median, p95"):
        qualified_record(metrics=incomplete_metrics)


def test_evidence_schema_is_strict_and_timestamp_must_be_utc():
    record = qualified_record()
    extra = record.to_dict()
    extra["later_field"] = True
    with pytest.raises(EvidenceValidationError, match="unknown"):
        EvidenceRecord.from_dict(extra)
    with pytest.raises(EvidenceValidationError, match="UTC"):
        qualified_record(recorded_at="2026-08-28T12:00:00+02:00")


def test_trusted_profile_accepts_only_qualified_matching_evaluator_evidence():
    selected = strategy()
    validity = domain()
    record = qualified_record(selected, validity)
    profile = TrustedExecutionProfile.from_qualified(
        selected, validity, [record], protected_baseline_evidence_id=DIGESTS["5"]
    )
    assert TrustedExecutionProfile.from_dict(profile.to_dict(), evidence=[record]) == profile
    assert profile.status is EvidenceStatus.QUALIFIED
    assert not hasattr(profile, "run") and not hasattr(profile, "select")
    assert profile.match(validity).status is DomainMatchStatus.MATCH
    changed = domain(model_revision="changed")
    assert profile.match(changed).status is DomainMatchStatus.REVALIDATION_REQUIRED

    with pytest.raises(EvidenceValidationError, match="must be built"):
        TrustedExecutionProfile(
            strategy=selected, validity_domain=validity,
            evidence_ids=(record.evidence_id,),
            protected_baseline_evidence_id=DIGESTS["5"],
            creation_evidence_id=record.evidence_id,
            last_revalidation_evidence_id=record.evidence_id,
            status=EvidenceStatus.QUALIFIED,
        )
    with pytest.raises(EvidenceValidationError, match="needs qualified evidence"):
        TrustedExecutionProfile.from_dict(profile.to_dict(), evidence=[])

    with pytest.raises(EvidenceValidationError, match="non-QUALIFIED"):
        TrustedExecutionProfile.from_qualified(
            selected, validity, [hypothesis_record(selected, validity)],
            protected_baseline_evidence_id=DIGESTS["5"],
        )
    with pytest.raises(EvidenceValidationError, match="domain/evidence mismatch"):
        TrustedExecutionProfile.from_qualified(
            selected, domain(model_revision="other"), [record],
            protected_baseline_evidence_id=DIGESTS["5"],
        )


def test_b27_public_adapter_is_explicitly_summary_only_and_cannot_create_profile():
    baseline = strategy(service_mode="interactive", grouping_width=1,
                        grouping_policy="none", synchronization_policy="per_step")
    candidate = strategy()
    validity = domain()
    cell = {
        "model": {"id": validity.model_id},
        "raw_sha256": DIGESTS["4"],
        "comparison": {
            "token_identity": True,
            "wall_ratio_throughput_over_interactive": {
                "median": 0.84, "ci_low": 0.83, "ci_high": 0.85,
            },
            "physical_rate_ratio_throughput_over_interactive": {
                "median": 1.18, "ci_low": 1.17, "ci_high": 1.19,
            },
        },
        "interactive": {"outer_wall_ms_median": 10},
        "throughput": {"outer_wall_ms_median": 8.4},
        "resources": {
            "mlx_peak_memory_bytes": 100, "swap_delta_bytes": 0,
            "fallbacks": 0, "correctness_errors": 0,
        },
    }
    record = evidence_from_b27_public_cell(
        cell, baseline_strategy_id=baseline.strategy_id,
        candidate_strategy_id=candidate.strategy_id, validity_domain_id=validity.domain_id,
        experiment_id="B27a2", study_id="B27", preregistration_sha256=DIGESTS["e"],
        reviewer_record_sha256=DIGESTS["f"], code_sha256=DIGESTS["0"],
        model_sha256=DIGESTS["1"], environment_sha256=DIGESTS["2"],
        workload_sha256=DIGESTS["3"], researcher_id="researcher-1",
        reviewer_id="reviewer-1", evaluator_id="evaluator-1",
        recorded_at="2026-08-28T12:00:00Z", warmup_repeats=2, measured_repeats=6,
    )
    assert record.status is EvidenceStatus.INCONCLUSIVE
    assert record.evidence_quality is EvidenceQuality.SUMMARY_ONLY
    assert not record.correctness.state_identity
    assert not record.resources.gates_passed and not record.resources.crash_free
    with pytest.raises(EvidenceValidationError, match="non-QUALIFIED"):
        TrustedExecutionProfile.from_qualified(
            candidate, validity, [record], protected_baseline_evidence_id=DIGESTS["5"]
        )


def test_module_and_runtime_import_boundaries_remain_one_way():
    root = Path(__file__).resolve().parents[1]
    evidence_path = root / "ironmule" / "evidence.py"
    tree = ast.parse(evidence_path.read_text())
    allowed = {
        "__future__", "hashlib", "json", "math", "re", "dataclasses", "datetime",
        "enum", "typing",
    }
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add((node.module or "").split(".")[0])
    assert imported <= allowed
    assert "mlx" not in imported and "mlx_lm" not in imported and "ironmule" not in imported

    runtime_files = [
        "__init__.py", "runtime.py", "service.py", "executor.py", "plans.py",
        "tune.py", "benchmark.py", "telemetry.py", "fingerprint.py",
    ]
    for name in runtime_files:
        source = (root / "ironmule" / name).read_text()
        assert ".evidence" not in source and "ironmule.evidence" not in source


def test_serialized_records_are_plain_json_data():
    profile_strategy = strategy()
    profile_domain = domain()
    record = qualified_record(profile_strategy, profile_domain)
    profile = TrustedExecutionProfile.from_qualified(
        profile_strategy, profile_domain, [record],
        protected_baseline_evidence_id=DIGESTS["5"],
    )
    payload = json.loads(canonical_json(profile))
    assert payload["schema"] == "ironmule.trusted_execution_profile.v1"
    assert payload["strategy"]["strategy_id"] == profile_strategy.strategy_id
