"""Boundary tests for the read-only, fail-closed identity collector."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import pytest

from friday_optimizer.collector import (
    BoundedCommandRunner,
    Collector,
    CollectorError,
    OutputTruncated,
    WorkloadContract,
    WORKLOAD_SCHEMA,
    _canonical,
    _read_bounded_file,
)


class FakePlatform:
    def __init__(self, *, chip: str = "Apple M1 Max", macos: str = "26.5.2", machine: str = "arm64"):
        self._machine = machine
        self._macos = macos
        self._chip = chip

    def system(self):
        return "Darwin"

    def machine(self):
        return self._machine

    def mac_ver(self):
        return (self._macos, ("", "", ""), "")

    def python_version(self):
        return "3.12.13"


class Runner:
    def __init__(self, *, chip="Apple M1 Max", memory=34359738368, cores=10, gpu="Apple M1 Max", gpu_cores="32"):
        self.values = {
            "machdep.cpu.brand_string": chip + "\n",
            "hw.memsize": str(memory) + "\n",
            "hw.logicalcpu": str(cores) + "\n",
        }
        self.gpu = json.dumps({"SPDisplaysDataType": [{
            "spdisplays_vendor": "Apple",
            "sppci_model": gpu,
            "sppci_cores": gpu_cores,
        }]})
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((tuple(argv), kwargs))
        if argv[0].endswith("sysctl"):
            return Result(self.values.get(argv[-1], ""))
        if argv[0].endswith("system_profiler"):
            return Result(self.gpu)
        if argv[0].endswith("pmset"):
            return Result("Now drawing from 'AC Power'\n")
        if argv[0].endswith("sw_vers"):
            return Result("26.5.2\n")
        raise AssertionError(argv)


@dataclass
class Result:
    stdout: str
    returncode: int = 0


def metadata(name):
    return {"mlx": "0.32.0", "mlx-lm": "0.31.3"}[name]


def workload(**changes):
    value = {
        "schema": WORKLOAD_SCHEMA,
        "prompt_family": "interactive-v1",
        "tokenizer": "tok-sha256",
        "generator": "mlx-lm-0.31.3",
        "context_bucket": "short",
        "batch": 1,
        "concurrency": 1,
        "max_tokens": 32,
        "greedy": True,
        "prompt_logprobs": False,
        "power_mode": "AC",
        "mode": "interactive",
    }
    value.update(changes)
    return value


def model_identity(**changes):
    semantic = {
        "schema": "ironmule.model_identity.v1",
        "model_id": "mlx-community/gemma-3-4b-it-4bit",
        "revision": "9" * 40,
        "model_manifest_sha256": "a" * 64,
        "architecture": "gemma3",
        "quantisation": {"bits": 4, "group_size": 64},
        "quantisation_sha256": hashlib.sha256(_canonical({"bits": 4, "group_size": 64})).hexdigest(),
        "tokenizer_sha256": "b" * 64,
        "manifest_file_count": 12,
        "manifest_bytes": 100,
        "tokenizer_file_count": 6,
    }
    semantic.update(changes)
    semantic["identity_sha256"] = hashlib.sha256(_canonical({k: v for k, v in semantic.items() if k != "identity_sha256"})).hexdigest()
    return semantic


def q2_profile(**changes):
    identity = model_identity()
    conditions = {
        "chip": "Apple M1 Max", "execution_plan": "single_shot", "max_tokens": 32,
        "prompt_tokens": 322, "mlx": "0.32.0", "mlx_lm": "0.31.3", "os": "26.5.2",
        "power_source": "AC", "model_id": identity["model_id"], "model_revision": identity["revision"],
        "model_identity_sha256": identity["identity_sha256"], "model_manifest_sha256": identity["model_manifest_sha256"],
        "tokenizer_sha256": identity["tokenizer_sha256"], "quantisation": identity["quantisation"],
    }
    conditions.update(changes.pop("conditions", {}))
    return {
        "model_identity": identity, "conditions": conditions,
        "hardware": {"static": {"memory_bytes": 34359738368, "cpu_logical": 10, "gpu_cores": 32, "python": "3.12.13"}},
        "confirmation": {"ratio": {"decode_ns": {"median_ratio": 999}}},
        "knobs": {"compiled_fixed_cache": True},
    }


def collector(runner=None, platform=None):
    return Collector(runner=runner or Runner(), platform_module=platform or FakePlatform(), metadata_version=metadata)


def test_snapshot_supports_m1_m2_m3_and_never_imports_mlx():
    for chip in ("Apple M1 Max", "Apple M2 Pro", "Apple M3 Max"):
        run = Runner(chip=chip, gpu=chip)
        snap = collector(run, FakePlatform(chip=chip)).snapshot(runtime_commit="c" * 64)
        assert snap.environment.chip == chip
        assert snap.environment.runtime_commit == "c" * 64
        assert snap.power_mode == "AC"


def test_commands_are_absolute_and_shell_free():
    run = Runner()
    collector(run).snapshot(runtime_commit="c" * 64)
    assert all(call[1]["shell"] is False for call in run.calls)
    assert all(call[0][0].startswith("/") for call in run.calls)


@pytest.mark.parametrize("kind", ["pmset", "system_profiler", "sysctl"])
def test_truncated_or_ambiguous_public_fact_blocks(kind):
    class Broken(Runner):
        def __call__(self, argv, **kwargs):
            result = super().__call__(argv, **kwargs)
            if kind == "pmset" and argv[0].endswith("pmset"):
                result.stdout = "Now drawing from 'AC Power'\nNow drawing from 'Battery Power'\n"
            if kind == "system_profiler" and argv[0].endswith("system_profiler"):
                result.stdout = json.dumps({"SPDisplaysDataType": [{}, {}]})
            if kind == "sysctl" and argv[0].endswith("sysctl") and argv[-1] == "hw.memsize":
                result.stdout = "1\n2\n"
            return result

    with pytest.raises(CollectorError):
        collector(Broken()).snapshot(runtime_commit="c" * 64)


def test_truncation_fails_closed():
    class Long(Runner):
        def __call__(self, argv, **kwargs):
            result = super().__call__(argv, **kwargs)
            if argv[0].endswith("system_profiler"):
                result.stdout = "x" * 100
            return result

    with pytest.raises(CollectorError):
        collector(Long()).snapshot(runtime_commit="c" * 64)


def test_valid_q2_profile_binds_verified_identity_and_ignores_confirmation():
    report = collector().collect(runtime_commit="c" * 64, profile=q2_profile(), workload_contract=workload())
    assert report.ready
    assert report.recommendation_allowed
    assert report.fingerprint.model.model_id.endswith("4bit")
    assert report.fingerprint_hash == report.fingerprint.fingerprint_hash


def test_profile_confirmation_is_not_identity():
    a = q2_profile()
    b = q2_profile()
    b["confirmation"]["ratio"]["decode_ns"]["median_ratio"] = 0.1
    ra = collector().collect(runtime_commit="c" * 64, profile=a, workload_contract=workload())
    rb = collector().collect(runtime_commit="c" * 64, profile=b, workload_contract=workload())
    assert ra.ready and rb.ready
    assert ra.fingerprint_hash == rb.fingerprint_hash


def test_missing_or_stale_profile_blocks_without_coercion():
    missing = collector().collect(runtime_commit="c" * 64, profile={"x": {}}, workload_contract=workload())
    assert missing.blocked
    assert not missing.recommendation_allowed
    stale = q2_profile(conditions={"chip": "Apple M2 Max"})
    report = collector().collect(runtime_commit="c" * 64, profile=stale, workload_contract=workload())
    assert report.ready
    assert report.ood
    assert not report.recommendation_allowed
    assert "environment_mismatch:chip" in report.ood_reasons


def test_explicit_model_identity_and_source_hash():
    identity = model_identity()
    report = collector().collect(runtime_commit="c" * 64, model_identity=identity, workload_contract=workload())
    assert report.ready
    assert report.model_source_sha256 == hashlib.sha256(_canonical(identity)).hexdigest()


@pytest.mark.parametrize("field", ["model_id", "revision", "identity_sha256", "quantisation"])
def test_malformed_model_identity_blocks(field):
    identity = model_identity()
    if field == "quantisation":
        identity[field] = {"bits": 4}
    else:
        identity[field] = None
    report = collector().collect(runtime_commit="c" * 64, model_identity=identity, workload_contract=workload())
    assert report.blocked
    assert not report.recommendation_allowed


def test_malformed_workload_is_strict_and_no_prompt_content_is_accepted():
    contract = workload(extra="forbidden")
    with pytest.raises(CollectorError):
        WorkloadContract.from_mapping(contract)
    contract = workload()
    contract["prompt"] = "secret"
    report = collector().collect(runtime_commit="c" * 64, model_identity=model_identity(), workload_contract=contract)
    assert report.blocked


@pytest.mark.parametrize("change", [
    {"chip": "Apple M2 Max"}, {"ram_bytes": 64 * 1024**3}, {"cpu_cores": 8},
    {"macos": "27.0.0"}, {"mlx": "0.33.0"}, {"mlx_lm": "0.32.0"}, {"python": "3.13.0"},
])
def test_cross_mac_profile_environment_is_ood(change):
    profile = q2_profile()
    if "chip" in change:
        profile["conditions"]["chip"] = change["chip"]
    elif "ram_bytes" in change:
        profile["hardware"]["static"]["memory_bytes"] = change["ram_bytes"]
    elif "cpu_cores" in change:
        profile["hardware"]["static"]["cpu_logical"] = change["cpu_cores"]
    elif "macos" in change:
        profile["conditions"]["os"] = change["macos"]
    elif "mlx" in change:
        profile["conditions"]["mlx"] = change["mlx"]
    elif "mlx_lm" in change:
        profile["conditions"]["mlx_lm"] = change["mlx_lm"]
    elif "python" in change:
        profile["hardware"]["static"]["python"] = change["python"]
    report = collector().collect(runtime_commit="c" * 64, profile=profile, workload_contract=workload())
    assert report.ood
    assert not report.recommendation_allowed
    assert report.ood_reasons


def test_report_hash_is_deterministic_and_redacted():
    first = collector().collect(runtime_commit="c" * 64, model_identity=model_identity(), workload_contract=workload())
    second = collector().collect(runtime_commit="c" * 64, model_identity=model_identity(), workload_contract=workload())
    assert first.canonical_bytes == second.canonical_bytes
    assert first.canonical_bytes == first.canonical
    assert "/Users/" not in json.dumps(first.safe_redacted())


def test_local_identity_file_is_fd_bound_and_normal_read_works(tmp_path):
    path = tmp_path / "identity.json"
    path.write_bytes(_canonical(model_identity()))
    report = collector().collect(runtime_commit="c" * 64, model_identity=path, workload_contract=workload())
    assert report.ready
    assert report.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_symlink_identity_is_rejected(tmp_path):
    target = tmp_path / "target.json"
    target.write_bytes(_canonical(model_identity()))
    link = tmp_path / "identity.json"
    link.symlink_to(target)
    report = collector().collect(runtime_commit="c" * 64, model_identity=link, workload_contract=workload())
    assert report.blocked
    assert "source_open_failed" in report.errors


def test_oversized_file_is_rejected_before_any_read(tmp_path, monkeypatch):
    path = tmp_path / "large.json"
    path.write_bytes(b"x" * 128)
    reads = []
    original_read = os.read

    def forbidden(fd, size):
        reads.append(size)
        return original_read(fd, size)

    monkeypatch.setattr(os, "read", forbidden)
    with pytest.raises(OutputTruncated):
        _read_bounded_file(path, maximum=16)
    assert reads == []


def test_file_growth_during_bounded_read_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "growing.json"
    path.write_bytes(b"{}")
    original_read = os.read
    changed = False

    def grow(fd, size):
        nonlocal changed
        chunk = original_read(fd, size)
        if chunk and not changed:
            changed = True
            with path.open("ab") as handle:
                handle.write(b" ")
        return chunk

    monkeypatch.setattr(os, "read", grow)
    with pytest.raises(CollectorError, match="source_"):
        _read_bounded_file(path, maximum=64)


def test_streaming_runner_kills_stdout_bomb_without_capture_output():
    calls = []

    def popen(*args, **kwargs):
        assert "capture_output" not in kwargs
        calls.append(kwargs)
        return subprocess.Popen(*args, **kwargs)

    command = [sys.executable, "-c", "import sys; sys.stdout.write('x' * 10000000); sys.stdout.flush()"]
    with pytest.raises(OutputTruncated, match="stdout"):
        BoundedCommandRunner(max_stdout_bytes=1024, max_stderr_bytes=1024, popen=popen).run(command)
    assert calls and calls[0]["start_new_session"] is True


def test_streaming_runner_kills_stderr_bomb_and_timeout_cleans_up():
    stderr_command = [sys.executable, "-c", "import sys; sys.stderr.write('e' * 10000000); sys.stderr.flush()"]
    with pytest.raises(OutputTruncated, match="stderr"):
        BoundedCommandRunner(max_stdout_bytes=1024, max_stderr_bytes=1024).run(stderr_command)

    seen = []

    def record_popen(*args, **kwargs):
        process = subprocess.Popen(*args, **kwargs)
        seen.append(process)
        return process

    timeout_command = [sys.executable, "-c", "import time; time.sleep(5)"]
    started = time.monotonic()
    with pytest.raises(CollectorError, match="timeout"):
        BoundedCommandRunner(timeout_seconds=0.1, popen=record_popen).run(timeout_command)
    assert time.monotonic() - started < 2
    assert seen and seen[0].poll() is not None


def test_nonempty_stderr_is_strictly_rejected():
    command = [sys.executable, "-c", "import sys; sys.stderr.write('warning')"]
    with pytest.raises(CollectorError, match="stderr_not_empty"):
        BoundedCommandRunner().run(command)
