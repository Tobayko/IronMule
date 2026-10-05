"""Hardware-cache contracts using explicit scripted samples, without GPU work."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from friday_evidence.canonical import canonical_sha256
from ironmule import hw

_REAL_BACKEND_BINDING = hw._backend_binding


@pytest.fixture
def scripted(monkeypatch):
    facts = {
        "machine": "scripted", "system": "TestOS", "os_release": "1",
        "chip": "Scripted GPU", "cpu_logical": 4, "cpu_performance": 4,
        "cpu_efficiency": 0, "memory_bytes": 8 * 2**30, "gpu_cores": 8,
        "python": "test", "mlx": "test", "gpu_available": True,
    }
    backend = {
        "kind": "metal", "mlx": "test", "mlx_lm": "test",
        "device_info": {"device_name": "Scripted GPU", "total_memory": 8 * 2**30},
        "graph_environment": {"MLX_MAX_OPS_PER_BUFFER": None, "MLX_MAX_MB_PER_BUFFER": None},
    }
    samples = {
        "scalar_chain_s": [0.001, 0.0012, 0.0011, 0.001, 0.0013],
        "gemv_small_s": [0.005, 0.006, 0.0055, 0.005, 0.0053],
        "gemv_large_s": [0.015, 0.016, 0.0155, 0.015, 0.0153],
    }
    calls = []
    def measure():
        calls.append(1)
        return hw._measured_from_samples(copy.deepcopy(samples))
    monkeypatch.setattr(hw, "static_facts", lambda: copy.deepcopy(facts))
    def backend_binding(_, *, observation=None):
        if observation is not None:
            observation.update(observed_unix_ns=time.time_ns(), free_memory_bytes=None,
                               availability="unavailable", errors=["free_memory_unavailable"])
        return copy.deepcopy(backend)
    monkeypatch.setattr(hw, "_backend_binding", backend_binding)
    monkeypatch.setattr(hw, "apply_cuda_graph_defaults", lambda: {})
    monkeypatch.setattr(hw, "measure", measure)
    return facts, backend, samples, calls


def redigest(record):
    record["sha256"] = canonical_sha256({key: value for key, value in record.items() if key != "sha256"})
    return record


def test_probe_reuses_same_cache_and_exposes_bounded_raw_samples(scripted, tmp_path):
    facts, _, samples, calls = scripted
    record = hw.probe(cache_dir=tmp_path)
    again = hw.probe(cache_dir=tmp_path)
    assert again == record
    assert len(calls) == 1
    assert record["measured"]["raw_timing_samples"] == samples
    assert record["measured"]["timing_summary"]["scalar_chain_s"]["n"] == 5
    assert record["measured"]["timing_summary"]["scalar_chain_s"]["stdev"] > 0
    assert hw.validate_probe(record, facts) == (True, [])
    assert set(tmp_path.iterdir()) == {tmp_path / f"hw-{hw.fingerprint(facts)}.json",
                                      tmp_path / f".hw-{hw.fingerprint(facts)}.lock"}
    assert all((path.stat().st_mode & 0o777) == 0o600 for path in tmp_path.iterdir())
    assert record["binding"]["protocol"]["clock"] == "host_submit_eval_synchronize"
    assert str(tmp_path) not in json.dumps(record)


def test_force_refresh_changes_samples_metadata_without_changing_binding(scripted, tmp_path):
    _, _, samples, calls = scripted
    first = hw.probe(cache_dir=tmp_path)
    samples["scalar_chain_s"][0] *= 2
    second = hw.probe(True, cache_dir=tmp_path)
    assert len(calls) == 2
    assert first["binding"] == second["binding"]
    assert first["measured"] != second["measured"]
    assert first["sha256"] != second["sha256"]


def test_supplied_facts_avoid_another_os_inventory(scripted, tmp_path, monkeypatch):
    facts, _, _, _ = scripted
    record = hw.probe(cache_dir=tmp_path)
    def forbidden():
        raise AssertionError("a second OS inventory is forbidden")
    monkeypatch.setattr(hw, "static_facts", forbidden)
    assert hw.validate_probe(record, facts) == (True, [])


@pytest.mark.parametrize("case", ["backend", "environment", "os", "protocol", "source"])
def test_cached_diagnostic_cannot_cross_identity_or_protocol_changes(scripted, tmp_path, case, monkeypatch):
    facts, backend, _, calls = scripted
    record = hw.probe(cache_dir=tmp_path)
    if case == "backend":
        backend["device_info"]["total_memory"] += 1
    elif case == "environment":
        backend["graph_environment"]["MLX_MAX_OPS_PER_BUFFER"] = "400"
    elif case == "os":
        facts["os_release"] = "2"
    elif case == "protocol":
        record["binding"]["protocol"]["scalar_submitted_ops"] = 513
        redigest(record)
    else:
        monkeypatch.setattr(hw.statistics, "__file__", str(tmp_path / "replaced-source.py"))
    valid, reasons = hw.validate_probe(record, facts)
    assert valid is False
    if case == "source":
        assert reasons == ["probe_binding_unavailable"]
    else:
        assert "probe_binding_mismatch" in reasons
    assert len(calls) == 1  # Validation performs no benchmark.


@pytest.mark.parametrize("case,reason", [
    ("legacy", "probe_schema_invalid"), ("unknown", "probe_schema_invalid"),
    ("digest", "probe_digest_mismatch"), ("future", "probe_timestamp_invalid"),
    ("reversed", "probe_timestamp_invalid"), ("stale", "probe_stale"),
    ("shape", "probe_measurement_shape_invalid"), ("samples", "probe_timing_samples_invalid"),
    ("count", "probe_timing_samples_invalid"), ("version", "probe_timing_samples_invalid"),
    ("zero", "probe_timing_samples_invalid"), ("negative", "probe_timing_samples_invalid"),
    ("huge", "probe_timing_samples_invalid"), ("summary", "probe_summary_mismatch"),
    ("unavailable", "probe_unavailable_invalid"),
])
def test_invalid_cache_fields_fail_closed_even_when_redigested(scripted, tmp_path, case, reason):
    facts, _, _, _ = scripted
    record = hw.probe(cache_dir=tmp_path)
    now = record["finished_unix_ns"]
    if case == "legacy":
        record = {"fingerprint": hw.fingerprint(facts), "measured": {"probe_version": 2}}
    elif case == "unknown":
        record["unknown"] = True
    elif case == "digest":
        record["sha256"] = "0" * 64
    elif case == "future":
        record["finished_unix_ns"] = now + 1
    elif case == "reversed":
        record["observed_unix_ns"] = now + 1
    elif case == "stale":
        now += 2_000_000_000
    elif case == "shape":
        record["measured"]["unexpected"] = 0
    elif case == "samples":
        record["measured"]["raw_timing_samples"]["gemv_small_s"].pop()
    elif case == "count":
        record["measured"]["repeats"] = 33
    elif case == "version":
        record["measured"]["probe_version"] = float(hw.PROBE_VERSION)
    elif case in {"zero", "negative", "huge"}:
        record["measured"]["raw_timing_samples"]["gemv_small_s"][0] = {
            "zero": 0.0, "negative": -1.0, "huge": 1e300}[case]
    elif case == "summary":
        record["measured"]["dispatch_us"] *= 2
    else:
        record["availability"] = "unavailable"
    if case != "digest" and "sha256" in record:
        redigest(record)
    valid, reasons = hw.validate_probe(record, facts, max_age_s=1, now_unix_ns=now)
    assert valid is False
    assert reason in reasons


def test_nonfinite_raw_values_never_become_measurement_evidence(scripted, tmp_path):
    facts, _, _, _ = scripted
    record = hw.probe(cache_dir=tmp_path)
    record["measured"]["raw_timing_samples"]["gemv_small_s"][0] = float("nan")
    valid, reasons = hw.validate_probe(record, facts)
    assert valid is False
    assert "probe_data_invalid" in reasons
    assert "probe_timing_samples_invalid" in reasons


def test_legacy_private_cache_refreshes_without_changing_hardware_fingerprint(scripted, tmp_path):
    facts, _, _, calls = scripted
    ident = hw.fingerprint(facts)
    path = tmp_path / f"hw-{ident}.json"
    path.write_text(json.dumps({"fingerprint": ident, "measured": {"probe_version": 2}}))
    path.chmod(0o600)
    record = hw.probe(cache_dir=tmp_path)
    assert len(calls) == 1
    assert hw.fingerprint(facts) == ident == record["fingerprint"]
    assert record["schema"] == hw.PROBE_SCHEMA


def test_unavailable_hardware_remains_unavailable_and_executes_no_probe(scripted, tmp_path):
    facts, backend, _, calls = scripted
    facts["gpu_available"] = False
    backend["kind"] = "unavailable"
    backend["device_info"] = None
    record = hw.probe(cache_dir=tmp_path)
    assert calls == []
    assert record["availability"] == "unavailable"
    assert record["errors"]
    assert record["measured"] == {}
    assert hw.validate_probe(record, facts) == (True, [])


def test_symlinked_cache_is_replaced_without_modifying_its_target(scripted, tmp_path):
    facts, _, _, calls = scripted
    elsewhere = tmp_path / "source.json"
    elsewhere.write_text("private source remains unchanged")
    path = tmp_path / f"hw-{hw.fingerprint(facts)}.json"
    path.symlink_to(elsewhere)
    hw.probe(cache_dir=tmp_path)
    assert len(calls) == 1
    assert not path.is_symlink()
    assert elsewhere.read_text() == "private source remains unchanged"


def test_atomic_write_failure_preserves_previous_cache_and_removes_temporary(scripted, tmp_path, monkeypatch):
    facts, _, _, _ = scripted
    hw.probe(cache_dir=tmp_path)
    path = tmp_path / f"hw-{hw.fingerprint(facts)}.json"
    before = path.read_bytes()
    def fail_replace(*_):
        raise OSError("scripted replacement failure")
    monkeypatch.setattr(hw.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failure"):
        hw.probe(True, cache_dir=tmp_path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".hw-*.tmp"))
    assert len(list(tmp_path.iterdir())) == 2


def test_stable_binding_sources_include_shared_measurement_code(scripted, tmp_path):
    record = hw.probe(cache_dir=tmp_path)
    sources = record["binding"]["source_files"]
    assert sources["ironmule/hw.py"] == hashlib.sha256(Path(hw.__file__).read_bytes()).hexdigest()
    assert set(sources) == {"ironmule/hw.py", "friday_evidence/canonical.py",
                            "friday_evidence/statistics.py", "friday_evidence/portable/supervisor.py"}
    assert record["binding"]["source_sha256"] == canonical_sha256(sources)


@pytest.mark.parametrize("repeats", [0, -1, 33, True, 1.5])
def test_repeat_budget_rejects_invalid_values_before_backend_work(repeats):
    with pytest.raises(ValueError, match="repeats"):
        hw.measure(repeats)


@pytest.mark.parametrize("age", [0, -1, True, float("nan"), float("inf")])
def test_invalid_cache_age_is_configuration_error(age, tmp_path):
    with pytest.raises(ValueError, match="max_age_s"):
        hw.probe(cache_dir=tmp_path, max_age_s=age)


def test_cache_only_missing_or_stale_record_never_executes_or_overwrites(scripted, tmp_path):
    facts, _, _, calls = scripted
    with pytest.raises(hw.HardwareProbeUnavailable):
        hw.probe(cache_dir=tmp_path, allow_measure=False)
    assert calls == []
    assert list(tmp_path.iterdir()) == []
    record = hw.probe(cache_dir=tmp_path)
    path = tmp_path / f"hw-{hw.fingerprint(facts)}.json"
    before = path.read_bytes()
    assert hw.probe(cache_dir=tmp_path, allow_measure=False) == record
    with pytest.raises(hw.HardwareProbeUnavailable, match="probe_refresh_requested"):
        hw.probe(True, cache_dir=tmp_path, allow_measure=False)
    assert path.read_bytes() == before
    assert len(calls) == 1


def test_cache_only_denied_or_replaced_source_is_not_accepted(scripted, tmp_path):
    facts, _, _, calls = scripted
    hw.probe(cache_dir=tmp_path)
    path = tmp_path / f"hw-{hw.fingerprint(facts)}.json"
    path.chmod(0o644)
    with pytest.raises(hw.HardwareProbeUnavailable, match="unreadable"):
        hw.probe(cache_dir=tmp_path, allow_measure=False)
    assert len(calls) == 1
    assert (path.stat().st_mode & 0o777) == 0o644


@pytest.mark.parametrize("force", [False, True])
def test_concurrent_cold_and_force_calls_measure_only_once(scripted, tmp_path, monkeypatch, force):
    _, _, samples, _ = scripted
    entered, release = threading.Event(), threading.Event()
    calls, outputs, errors = [], [], []
    def held_measure():
        calls.append(1)
        entered.set()
        assert release.wait(3)
        return hw._measured_from_samples(copy.deepcopy(samples))
    monkeypatch.setattr(hw, "measure", held_measure)
    def producer():
        try:
            outputs.append(hw.probe(force, cache_dir=tmp_path))
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=producer)
    thread.start()
    try:
        assert entered.wait(3)
        with pytest.raises(hw.HardwareProbeUnavailable, match="probe_refresh_busy"):
            hw.probe(force, cache_dir=tmp_path)
        assert calls == [1]
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert errors == []
    assert len(outputs) == 1
    assert hw.probe(cache_dir=tmp_path, allow_measure=False) == outputs[0]


@pytest.mark.parametrize("force", [False, True])
def test_new_cache_is_rechecked_after_existing_lease_is_acquired(scripted, tmp_path, monkeypatch, force):
    facts, _, _, calls = scripted
    record = hw.probe(cache_dir=tmp_path)
    path = tmp_path / f"hw-{hw.fingerprint(facts)}.json"
    path.unlink()
    original = hw._lease
    @contextmanager
    def intervening_writer(lock_path):
        with original(lock_path):
            fresh = copy.deepcopy(record)
            fresh["observed_unix_ns"] = fresh["finished_unix_ns"] = time.time_ns()
            hw._write_probe(path, redigest(fresh))
            yield
    monkeypatch.setattr(hw, "_lease", intervening_writer)
    actual = hw.probe(force, cache_dir=tmp_path)
    assert len(calls) == 1
    assert actual["measured"] == record["measured"]


def test_gemv_uses_distinct_explicit_keys_without_advancing_global_rng(monkeypatch):
    # Only scripted array operations execute. No allocation or GPU timing occurs.
    keys, global_state = [], [17]
    class Array:
        def astype(self, _):
            return self
        def sum(self):
            return self
    def normal(shape, *, key=None):
        if key is None:
            global_state[0] += 1
        keys.append(key)
        return Array()
    fake = SimpleNamespace(
        random=SimpleNamespace(key=lambda seed: seed, normal=normal), bfloat16="bf16",
        quantize=lambda *_args, **_kwargs: (Array(), Array(), Array()),
        eval=lambda *_: None, clear_cache=lambda: None, sum=lambda value: value,
        stack=lambda values: Array(), quantized_matmul=lambda *_args, **_kwargs: Array(),
    )
    import mlx
    monkeypatch.setattr(mlx, "core", fake)
    monkeypatch.setitem(sys.modules, "mlx.core", fake)
    def time_script(fn, repeats, *, raw_samples=None):
        fn()
        if raw_samples is not None:
            raw_samples.extend([0.01] * repeats)
        return 0.01
    monkeypatch.setattr(hw, "_time", time_script)
    raw = []
    assert hw._gemv_gbps(1024, 512 * 2**20, 5, raw_samples=raw) > 0
    assert global_state == [17]
    assert len(keys) == len(set(keys)) == 49
    assert keys[0] == hw._PROTOCOL["random_seed"] + 1024
    assert keys[-1] == hw._PROTOCOL["random_seed"] + 1024 + 48
    assert raw == [0.01] * 5


def test_core_without_version_preserves_current_fingerprint_and_binds_real_package_version(monkeypatch):
    fake = SimpleNamespace(metal=SimpleNamespace(is_available=lambda: True),
                           cuda=SimpleNamespace(is_available=lambda: False),
                           device_info=lambda: {"device_name": "Scripted GPU", "memory_size": 32 * 2**30})
    import mlx
    monkeypatch.setattr(mlx, "core", fake)
    monkeypatch.setitem(sys.modules, "mlx.core", fake)
    versions = {"mlx": "0.32.0", "mlx-lm": "0.31.3"}
    monkeypatch.setattr(importlib.metadata, "version", lambda name: versions[name])
    values = {"machdep.cpu.brand_string": "Scripted CPU", "hw.logicalcpu": "10",
              "hw.perflevel0.logicalcpu": "8", "hw.perflevel1.logicalcpu": "2",
              "hw.memsize": str(32 * 2**30)}
    monkeypatch.setattr(hw, "_sysctl", lambda name: values[name])
    monkeypatch.setattr(hw, "_gpu_cores", lambda: 32)
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hw.platform, "machine", lambda: "arm64")
    facts = hw.static_facts()
    assert facts["mlx"] == "0.32.0"  # Current HEAD already uses this fallback.
    baseline = {"machine": "arm64", "chip": "Scripted CPU", "cpu_logical": 10,
                "memory_bytes": 32 * 2**30, "gpu_cores": 32, "mlx": "0.32.0"}
    expected = hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()[:16]
    assert hw.fingerprint(facts) == expected
    binding = hw._backend_binding(facts)
    assert binding["mlx"] == "0.32.0"
    assert binding["kind"] == "metal"
    # Supplying stale cached facts cannot conceal a changed backend installation.
    versions["mlx"] = "0.32.1"
    assert hw.fingerprint(facts) == expected
    assert hw._backend_binding(facts)["mlx"] == "0.32.1"


@pytest.fixture
def cuda_info(scripted, monkeypatch):
    facts, _, samples, calls = scripted
    info = {"device_name": "Scripted CUDA GPU", "architecture": "sm_75",
            "compute_capability_major": 7, "compute_capability_minor": 5,
            "total_memory": 15 * 2**30, "free_memory": 12 * 2**30,
            "pci_bus_id": "0000:00:04.0", "uuid": "GPU-scripted-first"}
    fake = SimpleNamespace(
        __version__="test", cuda=SimpleNamespace(is_available=lambda: True),
        metal=SimpleNamespace(is_available=lambda: False), device_info=lambda: dict(info),
    )
    import mlx
    monkeypatch.setattr(mlx, "core", fake)
    monkeypatch.setitem(sys.modules, "mlx.core", fake)
    monkeypatch.setattr(hw, "_backend_binding", _REAL_BACKEND_BINDING)
    return facts, info, samples, calls


def test_changing_cuda_free_memory_during_probe_and_cache_reuse_is_diagnostic(cuda_info, tmp_path, monkeypatch):
    facts, info, samples, calls = cuda_info
    def measured_allocation():
        calls.append(1)
        info["free_memory"] = 5 * 2**30
        return hw._measured_from_samples(copy.deepcopy(samples))
    monkeypatch.setattr(hw, "measure", measured_allocation)
    record = hw.probe(cache_dir=tmp_path)
    assert calls == [1]
    assert record["availability"] == "available"
    assert record["device_observations"]["before"]["free_memory_bytes"] == 12 * 2**30
    assert record["device_observations"]["after"]["free_memory_bytes"] == 5 * 2**30
    assert "free_memory" not in record["binding"]["backend"]["device_info"]
    assert record["binding"]["backend"]["device_info"]["uuid"] == info["uuid"]
    info["free_memory"] = 2 * 2**30
    assert hw.validate_probe(record, facts) == (True, [])
    assert hw.probe(cache_dir=tmp_path, allow_measure=False) == record
    assert calls == [1]


@pytest.mark.parametrize("field,value", [
    ("total_memory", 16 * 2**30), ("uuid", "GPU-scripted-next"),
    ("pci_bus_id", "0000:00:05.0"), ("architecture", "sm_80"),
    ("device_name", "Different GPU"), ("compute_capability_major", 8),
    ("compute_capability_minor", 6),
])
def test_real_device_changes_still_invalidate_cached_characterization(cuda_info, tmp_path, field, value):
    facts, info, _, calls = cuda_info
    record = hw.probe(cache_dir=tmp_path)
    ident = hw.fingerprint(facts)
    info[field] = value
    valid, reasons = hw.validate_probe(record, facts)
    assert not valid and "probe_binding_mismatch" in reasons
    with pytest.raises(hw.HardwareProbeUnavailable, match="probe_binding_mismatch"):
        hw.probe(cache_dir=tmp_path, allow_measure=False)
    assert hw.fingerprint(facts) == ident  # Existing legacy fingerprint is untouched.
    assert calls == [1]


def test_hardware_replacement_during_measurement_is_rejected_before_cache_write(cuda_info, tmp_path, monkeypatch):
    facts, info, samples, _ = cuda_info
    def changed_device():
        info["uuid"] = "GPU-scripted-replacement"
        return hw._measured_from_samples(copy.deepcopy(samples))
    monkeypatch.setattr(hw, "measure", changed_device)
    with pytest.raises(RuntimeError, match="binding changed during measurement"):
        hw.probe(cache_dir=tmp_path)
    assert not (tmp_path / f"hw-{hw.fingerprint(facts)}.json").exists()


@pytest.mark.parametrize("free", [None, -1, True, float("nan")])
def test_invalid_free_memory_is_unknown_observation_not_device_identity(cuda_info, tmp_path, free):
    facts, info, _, _ = cuda_info
    info["free_memory"] = free
    record = hw.probe(cache_dir=tmp_path)
    for observation in record["device_observations"].values():
        assert observation["availability"] == "unavailable"
        assert observation["free_memory_bytes"] is None
        assert observation["errors"] == ["free_memory_invalid"]
    assert hw.validate_probe(record, facts) == (True, [])


def test_stable_projection_drops_unknown_counters_without_mutating_input():
    raw = {"device_name": "Scripted GPU", "memory_size": 1024,
           "free_memory": 512, "allocation_count": object(), "utilization": float("nan")}
    result = hw.stable_device_info(raw)
    assert result == {"device_name": "Scripted GPU", "memory_size": 1024}
    assert raw["free_memory"] == 512
    result["memory_size"] = 2048
    assert raw["memory_size"] == 1024


@pytest.mark.parametrize("change", [{"device_name": ""}, {"total_memory": 0},
                                     {"total_memory": True}, {"compute_capability_minor": -1},
                                     {"uuid": [1, 2]}, {"architecture": "bad\nname"}])
def test_invalid_stable_device_fields_fail_closed(change):
    raw = {"device_name": "Scripted GPU", "total_memory": 1024, **change}
    with pytest.raises(ValueError):
        hw.stable_device_info(raw)
