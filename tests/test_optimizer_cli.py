"""Contract tests for the closed Friday optimizer CLI."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import io
from types import SimpleNamespace
from pathlib import Path

import pytest

import friday_optimizer.cli as cli
from friday_optimizer.candidates import CandidateRegistry
from friday_optimizer.collector import CollectorReport, CurrentSnapshot
from friday_optimizer.fingerprint import EnvironmentFingerprint, ExactFingerprint, ModelFingerprint, WorkloadFingerprint
from friday_optimizer.canonical import canonical_bytes
from friday_optimizer.portfolio import MANIFEST_SCHEMA, MODEL_IDS
from friday_optimizer.real_session import FingerprintReport


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


def _portfolio_manifest() -> dict:
    digest = lambda seed: __import__("hashlib").sha256(seed.encode()).hexdigest()
    models = []
    for size in ("1b", "4b", "12b"):
        models.append({
            "size": size,
            "model_id": MODEL_IDS[size],
            "cache_status": "verified",
            "identity": {
                "revision": "rev-" + size,
                "manifest_sha256": digest("manifest-" + size),
                "tokenizer_sha256": digest("tokenizer-" + size),
                "architecture": "gemma3_text" if size == "1b" else "gemma3",
                "quant_bits": 4,
                "quant_group_size": 64,
                "identity_sha256": digest("identity-" + size),
                "identity_document_sha256": digest("identity-document-" + size),
            },
            "evidence": [],
            "preregistration": None,
            "readiness": "unknown",
        })
    models.append({"size": "27b", "model_id": MODEL_IDS["27b"], "cache_status": "missing", "identity": None, "evidence": [], "preregistration": None, "readiness": "unknown"})
    return {"schema": MANIFEST_SCHEMA, "version": 1, "models": models, "candidate_id": "combined_core_profile", "registry_hash": CandidateRegistry().registry_hash, "cache_inventory_sha256": digest("cache"), "evidence_inventory_sha256": digest("evidence"), "readiness_evidence_sha256": digest("readiness")}


def _fake_fingerprint_report(model_id: str) -> FingerprintReport:
    environment = EnvironmentFingerprint("Apple M1 Max", "Apple GPU", 32 * 1024**3, 10, "26.5.2", "0.32.0", "0.31.3", "3.12", "a" * 40)
    exact = ExactFingerprint(
        environment,
        ModelFingerprint(model_id, "rev", "b" * 64, "gemma", 4, 64, "c" * 64),
        WorkloadFingerprint("default", "d" * 64, "generator", "short", 1, 1, 32, True, False, "ac", "interactive"),
    )
    return FingerprintReport.from_collector(
        CollectorReport(CurrentSnapshot(environment, "AC", 32), exact, "e" * 64, "f" * 64, None)
    )


def test_help_and_unknown_flags_are_bounded() -> None:
    help_result = run_cli("--help")
    assert help_result.returncode == 0
    assert "activate" not in help_result.stdout
    unknown = run_cli("doctor", "--forbidden")
    assert unknown.returncode == 64
    assert "Traceback" not in unknown.stderr
    assert payload(unknown)["ok"] is False


def test_portfolio_cli_emits_canonical_path_free_snapshot(tmp_path: Path) -> None:
    manifest = tmp_path / "portfolio.json"
    manifest.write_bytes(canonical_bytes(_portfolio_manifest()))
    before = manifest.read_bytes()
    result = run_cli("portfolio", "--manifest", str(manifest))
    assert result.returncode == 0
    value = payload(result)
    assert value["command"] == "portfolio"
    assert value["ok"] is True
    assert value["schema"] == "friday.optimizer.portfolio.v1"
    assert "models" in value and len(value["models"]) == 4
    assert str(manifest) not in result.stdout
    assert result.stdout.encode().strip() == canonical_bytes(value)
    assert manifest.read_bytes() == before


def test_portfolio_cli_rejects_schema_and_execute_without_writing(tmp_path: Path) -> None:
    manifest = tmp_path / "portfolio.json"
    manifest.write_bytes(canonical_bytes(_portfolio_manifest()))
    invalid = run_cli("portfolio", "--manifest", str(manifest), "--execute")
    assert invalid.returncode == 64
    assert payload(invalid)["ok"] is False
    bad = tmp_path / "bad.json"
    bad.write_text("{\"schema\":\"wrong\"}")
    rejected = run_cli("portfolio", "--manifest", str(bad))
    assert rejected.returncode == 65
    assert "wrong" not in rejected.stdout
    assert str(bad) not in rejected.stdout
    missing = run_cli("portfolio", "--manifest", str(tmp_path / "missing.json"))
    assert missing.returncode == 65
    assert "missing.json" not in missing.stdout


def test_fingerprint_cli_and_file_projection_redact_local_model_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "model.json"
    workload = tmp_path / "workload.json"
    output = tmp_path / "fingerprint.json"
    model.write_text("{}")
    workload.write_text("{}")
    exact_report = _fake_fingerprint_report("/Users/alice/models/gemma-3-4b-it")
    monkeypatch.setattr(cli, "collect_fingerprint", lambda **_: exact_report)
    args = SimpleNamespace(
        model_identity=str(model), workload_contract=str(workload), runtime_commit="a" * 40,
        out=str(output), execute=True,
    )

    result, code = cli._fingerprint(args)

    assert int(code) == 0
    assert result["report"]["fingerprint"]["model"]["model_id"] == "<local-model>"
    assert result["fingerprint_hash"] == exact_report.fingerprint_hash
    assert result["report_hash"] == exact_report.report_hash
    written = json.loads(output.read_text())
    assert written["report"]["fingerprint"]["model"]["model_id"] == "<local-model>"
    assert "/Users/alice/models/gemma-3-4b-it" not in output.read_text()
    assert exact_report.as_dict()["report"]["fingerprint"]["model"]["model_id"] == "/Users/alice/models/gemma-3-4b-it"


def test_real_session_requires_explicit_execute_without_touching_targets(tmp_path: Path) -> None:
    result = run_cli(
        "session", "--checkout", str(tmp_path), "--expected-head", "a" * 40,
        "--interpreter", str(tmp_path / "python"), "--model-identity", str(tmp_path / "model.json"),
        "--workload-contract", str(tmp_path / "workload.json"), "--runtime-commit", "b" * 40,
        "--candidate", "combined_core_profile", "--duration", "5", "--prereg", str(tmp_path / "prereg.json"),
        "--memory", str(tmp_path / "memory.sqlite3"), "--result-out", str(tmp_path / "result.json"),
        "--session-id", "manual-1",
    )
    assert result.returncode == 78
    assert payload(result)["reason"] == "explicit_execute_required"
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / ".friday-data").exists()


def test_shadow_history_requires_explicit_execute(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}")
    result = run_cli("shadow", "--request", str(request), "--write-history")
    assert result.returncode == 78
    assert payload(result)["error"] == "explicit_execute_required"


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


def _fingerprint_document(tmp_path: Path) -> Path:
    from tests.test_optimizer_decisions import make_fingerprint

    target = tmp_path / "fingerprint.json"
    target.write_bytes(canonical_bytes(make_fingerprint().as_dict()))
    return target


def test_decide_requires_explicit_execute(tmp_path):
    document = _fingerprint_document(tmp_path)
    result = run_cli("decide", "--memory", str(tmp_path / "memory.sqlite3"), "--fingerprint", str(document))
    body = payload(result)
    assert body["ok"] is False and body["reason"] == "explicit_execute_required"
    assert body["written"] is False
    assert not (tmp_path / "memory.sqlite3").exists()


def test_decide_outcome_replay_round_trip(tmp_path):
    memory = str(tmp_path / "memory.sqlite3")
    document = _fingerprint_document(tmp_path)
    decision = payload(run_cli(
        "decide", "--memory", memory, "--fingerprint", str(document),
        "--hint", "head_skip_prefill", "--execute",
    ))
    assert decision["ok"] is True and decision["chosen"] == "head_skip_prefill"
    assert decision["propensity"] == 1.0 and decision["no_activation"] is True

    outcome = payload(run_cli(
        "outcome", "--memory", memory, "--decision", decision["decision_id"],
        "--censoring", "observed", "--reward", "0.846", "--execute",
    ))
    assert outcome["ok"] is True and outcome["reward"] == pytest.approx(0.846)

    replay = run_cli("replay", "--memory", memory)
    body = payload(replay)
    assert body["labelled_steps"] == 1
    assert body["learning_claim"] is False and body["conclusive"] is False
    # One measurement is never a result, whatever the estimate happens to be.
    assert all(item["status"] == "insufficient_data" for item in body["estimates"].values())
    assert replay.returncode == int(cli.ExitCode.UNAVAILABLE)


def test_replay_rejects_an_unknown_selection_rule(tmp_path):
    memory = str(tmp_path / "memory.sqlite3")
    document = _fingerprint_document(tmp_path)
    run_cli("decide", "--memory", memory, "--fingerprint", str(document), "--execute")
    result = run_cli("replay", "--memory", memory, "--rule", "policy_gradient")
    assert result.returncode == int(cli.ExitCode.USAGE)


def test_outcome_rejects_a_reward_on_a_censored_run(tmp_path):
    memory = str(tmp_path / "memory.sqlite3")
    document = _fingerprint_document(tmp_path)
    decision = payload(run_cli("decide", "--memory", memory, "--fingerprint", str(document), "--execute"))
    result = run_cli(
        "outcome", "--memory", memory, "--decision", decision["decision_id"],
        "--censoring", "censored_timeout", "--reward", "0.5", "--execute",
    )
    assert payload(result)["ok"] is False
    assert result.returncode == int(cli.ExitCode.DATA)


def test_campaign_plans_without_writing_and_reports_its_overlap(tmp_path):
    document = _fingerprint_document(tmp_path)
    result = run_cli(
        "campaign", "--memory", str(tmp_path / "memory.sqlite3"), "--fingerprint", str(document),
        "--campaign-id", "r2-v1", "--rule", "epsilon_greedy", "--epsilon", "0.5",
        "--policy-id", "log-v1", "--hint", "head_skip_prefill", "--seed-base", "42", "--required", "30",
    )
    body = payload(result)
    assert body["ok"] is True and body["written"] is False and body["reason"] == "planning_only"
    assert body["points"] == 50 and body["blocks"] == 5 and body["points_per_block"] == 10
    assert body["distinct_actions"] > 1, "a campaign without overlap is worthless"
    assert body["learning_claim"] is False
    assert not (tmp_path / "memory.sqlite3").exists()


def test_campaign_execute_writes_one_record_per_point(tmp_path):
    memory = str(tmp_path / "memory.sqlite3")
    document = _fingerprint_document(tmp_path)
    body = payload(run_cli(
        "campaign", "--memory", memory, "--fingerprint", str(document), "--campaign-id", "r2-v2",
        "--rule", "epsilon_greedy", "--epsilon", "0.5", "--policy-id", "log-v1",
        "--hint", "head_skip_prefill", "--seed-base", "7", "--points", "12", "--execute",
    ))
    assert body["written"] is True and body["records"] == 12
    decisions = payload(run_cli("replay", "--memory", memory))
    # Decisions are logged, outcomes are not: nothing is labelled yet.
    assert decisions["labelled_steps"] == 0
    assert all(item["status"] == "no_labels" for item in decisions["estimates"].values())


def test_campaign_requires_exactly_one_sizing_argument(tmp_path):
    document = _fingerprint_document(tmp_path)
    base = ["campaign", "--memory", str(tmp_path / "m.sqlite3"), "--fingerprint", str(document),
            "--campaign-id", "r2-v3", "--rule", "epsilon_greedy", "--epsilon", "0.5"]
    assert run_cli(*base).returncode == int(cli.ExitCode.USAGE)
    assert run_cli(*base, "--points", "10", "--required", "30").returncode == int(cli.ExitCode.USAGE)


def test_campaign_refuses_a_deterministic_rule(tmp_path):
    document = _fingerprint_document(tmp_path)
    result = run_cli(
        "campaign", "--memory", str(tmp_path / "m.sqlite3"), "--fingerprint", str(document),
        "--campaign-id", "r2-v4", "--points", "10",
    )
    assert result.returncode == int(cli.ExitCode.DATA)


def _session_result(tmp_path: Path, *, pairs=6, gain=0.12, name="result.json") -> Path:
    """A stage payload in exactly the shape the IronMule worker emits."""

    ttft, tokens, tps = 1.7851, 32, 70.99
    share = ttft / (ttft + tokens / tps)
    baseline, candidate = [], []
    for index in range(pairs):
        order = "AB" if index % 2 == 0 else "BA"
        base = ttft + tokens / tps
        cand = base * (1.0 - gain)
        for request, arm, sink in ((base, "baseline", baseline), (cand, "candidate", candidate)):
            sink.append({
                "session_id": f"s{index}", "pair_id": f"p{index}", "arm": arm, "order": order,
                "fingerprint": "f" * 64, "workload": "w",
                "ttft_seconds": request * share, "tokens": tokens,
                "decode_tps": tokens / (request * (1 - share)), "status": "ok", "error": "",
            })
    target = tmp_path / name
    target.write_bytes(canonical_bytes({
        "payload": {"schema": "friday.ironmule.result.v1", "stage": "test",
                    "baseline_samples": baseline, "candidate_samples": candidate,
                    "pair_count": pairs, "token_identity": True},
    }))
    return target


def test_integrate_turns_a_session_result_into_a_verdict(tmp_path):
    result = run_cli(
        "integrate", "--result", str(_session_result(tmp_path)), "--arm", "warm",
        "--min-gain", "0.10", "--mde", "0.05",
    )
    body = payload(result)
    assert body["status"] == "qualified" and body["ok"] is True
    assert body["pairs"] == 6
    assert body["gain_percent"] == pytest.approx(12.0, abs=0.5)
    assert body["formal_claim"] is False and body["no_activation"] is True


def test_integrate_reports_below_threshold_without_calling_it_a_failure(tmp_path):
    path = _session_result(tmp_path, gain=0.06, name="weak.json")
    result = run_cli("integrate", "--result", str(path), "--arm", "warm",
                     "--min-gain", "0.10", "--mde", "0.05")
    body = payload(result)
    assert body["status"] == "below_threshold" and body["qualified"] is False
    assert result.returncode == int(cli.ExitCode.UNAVAILABLE)


def test_integrate_combines_several_sessions(tmp_path):
    first = _session_result(tmp_path, pairs=3, name="a.json")
    second = _session_result(tmp_path, pairs=3, name="b.json")
    body = payload(run_cli("integrate", "--result", str(first), "--result", str(second),
                           "--arm", "warm", "--min-gain", "0.10", "--mde", "0.05"))
    # Both files use the same pair ids, so the evaluator's duplicate guard must
    # catch it rather than silently double-count the same evidence.
    assert body["status"] in ("rejected", "inconclusive")


def test_integrate_rejects_a_result_without_paired_samples(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_bytes(canonical_bytes({"payload": {"schema": "friday.ironmule.result.v1"}}))
    result = run_cli("integrate", "--result", str(empty), "--arm", "warm",
                     "--min-gain", "0.10", "--mde", "0.05")
    assert result.returncode == int(cli.ExitCode.DATA)
    assert payload(result)["error"] == "result_carries_no_paired_samples"
