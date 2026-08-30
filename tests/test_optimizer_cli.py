"""Contract tests for the closed Friday optimizer CLI."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import io
from pathlib import Path

import pytest


PYTHON = Path(__file__).parents[1] / ".venv" / "bin" / "python"


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PYTHON), "-m", "friday_optimizer", *arguments],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )


def payload(result: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(result.stdout)


def test_help_and_unknown_flags_are_bounded() -> None:
    help_result = run_cli("--help")
    assert help_result.returncode == 0
    assert "activate" not in help_result.stdout
    unknown = run_cli("doctor", "--forbidden")
    assert unknown.returncode == 64
    assert "Traceback" not in unknown.stderr
    assert payload(unknown)["ok"] is False


def test_doctor_and_audit_do_not_change_files(tmp_path: Path) -> None:
    evidence = tmp_path / "experiments"
    evidence.mkdir()
    (evidence / "sample.json").write_text('{"model":"gemma-1b","timing":{"p50":1}}')
    before = {path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in tmp_path.rglob("*")}
    doctor = run_cli("doctor", "--root", str(tmp_path))
    audit = run_cli("audit", "--root", str(tmp_path))
    assert doctor.returncode == 0
    assert audit.returncode == 0
    assert payload(doctor)["source_metadata_unchanged"]
    assert payload(audit)["source_metadata_unchanged"]
    after = {path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in tmp_path.rglob("*")}
    assert before == after


def test_import_needs_execute_and_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "sample.json").write_text('{"latency_ms":1}')
    memory = tmp_path / "memory.sqlite3"
    dry = run_cli("import", "--root", str(tmp_path), "--memory", str(memory))
    assert dry.returncode == 78
    assert not memory.exists()
    first = run_cli("import", "--root", str(tmp_path), "--memory", str(memory), "--execute")
    second = run_cli("import", "--root", str(tmp_path), "--memory", str(memory), "--execute")
    assert first.returncode == second.returncode == 0
    assert payload(first)["records_written"] >= 1
    assert payload(second)["records_written"] == 0
    assert payload(second)["records_idempotent"] >= 1
    assert payload(first)["source_root"] == "."


def test_dataset_is_canonical_new_file_only(tmp_path: Path) -> None:
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "sample.json").write_text('{"latency_ms":1}')
    output = tmp_path / "dataset.json"
    dry = run_cli("dataset", "--root", str(tmp_path), "--out", str(output))
    assert dry.returncode == 78
    assert not output.exists()
    written = run_cli("dataset", "--root", str(tmp_path), "--out", str(output), "--execute")
    assert written.returncode == 0
    report = payload(written)
    assert report["written"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert __import__("hashlib").sha256(output.read_bytes()).hexdigest() == report["sha256"]
    refused = run_cli("dataset", "--root", str(tmp_path), "--out", str(output), "--execute")
    assert refused.returncode == 65
    assert payload(refused)["error"] == "output_exists"


def test_status_missing_is_read_only_unavailable(tmp_path: Path) -> None:
    memory = tmp_path / "missing.sqlite3"
    result = run_cli("status", "--memory", str(memory))
    assert result.returncode == 1
    assert not memory.exists()
    assert payload(result)["read_only"] is True


def _tampered_memory(path: Path) -> None:
    from friday_optimizer import DataPhase, OptimizationMemoryV2, OptimizationRecord, QualityClass, RecordKind
    import sqlite3

    with OptimizationMemoryV2(path) as memory:
        memory.append(OptimizationRecord("tamper", RecordKind.BENCHMARK, QualityClass.FORMAL, DataPhase.LABEL, {"value": 1}))
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER optimization_records_no_update")
        connection.execute("UPDATE optimization_records SET payload=?", (b'{"changed":true}',))


def _valid_shadow_request(path: Path) -> None:
    from friday_optimizer import EnvironmentFingerprint, ExactFingerprint, ModelFingerprint, WorkloadFingerprint

    fingerprint = ExactFingerprint(
        EnvironmentFingerprint("M1 Max", "Apple GPU", 32 * 1024**3, 10, "14.5", "0.32.0", "0.31.3", "3.12", "a" * 64),
        ModelFingerprint("google/gemma-3-4b-it", "r1", "b" * 64, "gemma", 4, 64, "tok"),
        WorkloadFingerprint("chat", "tok", "gen", "short", 1, 1, 32, True, False, "performance", "interactive"),
    )
    path.write_text(json.dumps({"fingerprint": fingerprint.as_dict(), "candidate_id": "baseline", "dataset_hash": "c" * 64, "code_hash": "d" * 64}))


def _file_state(path: Path) -> tuple[bytes, int, int, int]:
    info = path.stat()
    return path.read_bytes(), info.st_ino, stat.S_IMODE(info.st_mode), info.st_mtime_ns


def test_import_preflights_tampered_memory_without_mutation(tmp_path: Path) -> None:
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "sample.json").write_text('{"latency_ms":1}')
    memory = tmp_path / "memory.sqlite3"
    _tampered_memory(memory)
    before = _file_state(memory)
    result = run_cli("import", "--root", str(tmp_path), "--memory", str(memory), "--execute")
    assert result.returncode == 65
    assert payload(result)["error"] == "memory_preflight_failed"
    assert _file_state(memory) == before


def test_shadow_history_preflights_tampered_memory_without_mutation(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite3"
    _tampered_memory(memory)
    request = tmp_path / "request.json"
    _valid_shadow_request(request)
    before = _file_state(memory)
    result = run_cli("shadow", "--request", str(request), "--memory", str(memory), "--write-history", "--execute")
    assert result.returncode == 65
    assert payload(result)["error"] == "memory_preflight_failed"
    assert _file_state(memory) == before


def test_emit_raises_on_unrepresentable_output(monkeypatch: pytest.MonkeyPatch) -> None:
    import friday_optimizer.cli as cli

    monkeypatch.setattr(cli, "canonical_bytes", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("too large")))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    with pytest.raises(cli.CLIError) as raised:
        cli._emit({"not": "bounded"})
    assert raised.value.code == cli.ExitCode.INTERNAL
    assert json.loads(output.getvalue()) == {"error": "output_unavailable", "ok": False}


def test_shadow_rejects_unknown_request_fields_without_history(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"unexpected": True}))
    result = run_cli("shadow", "--request", str(request))
    assert result.returncode == 65
    assert payload(result)["error"] in {"unknown_request_field", "invalid_shadow_request"}
    assert not list(tmp_path.glob("*.sqlite3"))
