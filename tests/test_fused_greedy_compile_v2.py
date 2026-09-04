from __future__ import annotations

from pathlib import Path


# Cycle 19 intentionally repeats the exact Cycle-18 experiment under new,
# independently sealed IDs.  Execute the established offline regression source
# against the v2 paths so every old gate stays mandatory without copy drift.
_BASE_TEST = Path(__file__).with_name("test_fused_greedy_compile.py")
_BASE_SOURCE = _BASE_TEST.read_text(encoding="utf-8")
_V2_SOURCE = _BASE_SOURCE.replace(
    'EXP = ROOT / "experiments" / "fused_greedy_compile"',
    'EXP = ROOT / "experiments" / "fused_greedy_compile_v2"',
).replace(
    'HARNESS_PATH = EXP / "measure_fused_greedy_compile.py"',
    'HARNESS_PATH = EXP / "measure_fused_greedy_compile_v2.py"',
)
assert _V2_SOURCE != _BASE_SOURCE
exec(compile(_V2_SOURCE, str(Path(__file__).resolve()), "exec"), globals())


def test_v2_environment_fingerprints_are_byte_identical_and_bind_removed_environment():
    assert worker.environment_fingerprint() == harness.environment_fingerprint()
    assert worker.OFFLINE_ENV == harness.OFFLINE_ENV
    assert worker.UNSAFE_ENV == harness.UNSAFE_ENV
    assert worker.environment_fingerprint() == worker._sha256_bytes(
        worker._canonical(
            {
                "offline": worker.OFFLINE_ENV,
                "removed": worker.UNSAFE_ENV,
                "python": str(Path(worker.sys.executable).resolve()),
                "machine": worker.platform.machine(),
            }
        )
    )


def test_v2_wrong_environment_hash_fails_before_model_load_and_preserves_error(
    monkeypatch: pytest.MonkeyPatch,
):
    emitted: list[dict[str, Any]] = []
    imports: list[str] = []
    original_import = __import__

    def guarded_import(name: str, *args: Any, **kwargs: Any):
        if name == "mlx" or name.startswith("mlx.") or name.startswith("mlx_lm"):
            imports.append(name)
            raise AssertionError("model/MLX import reached after bad environment binding")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    monkeypatch.setattr(worker, "_emit", lambda event: emitted.append(copy.deepcopy(event)))
    bindings = {
        "PARENT_PID": str(worker.os.getppid()),
        "RUN_ID": worker.RUN_ID,
        "MODEL_KEY": worker.MODEL_KEY,
        "NONCE": worker.AUTH_NONCE,
        "PROTOCOL_VERSION": str(worker.PROTOCOL_VERSION),
        "PROTOCOL_SHA256": worker.PROTOCOL_SHA256,
        "PREREG_SHA256": worker.FROZEN_PREREGISTRATION_SHA256,
        "PROMPT_SHA256": worker.EXPECTED_PROMPT_SHA256,
        "ENVIRONMENT_SHA256": "0" * 64,
    }
    for suffix, value in bindings.items():
        monkeypatch.setenv(worker.AUTH_PREFIX + suffix, value)
    for key, value in worker.OFFLINE_ENV.items():
        monkeypatch.setenv(key, value)

    assert worker.main(["--execute", "--model-key", worker.MODEL_KEY]) == 1
    assert imports == []
    assert len(emitted) == 1
    event = emitted[0]
    assert event["status"] == "error"
    assert event["load_count"] == 0
    assert event["error"] == {
        "type": "WorkerError",
        "message": "authorisation failed: ENVIRONMENT_SHA256",
    }
    assert event["environment_sha256"] == worker.environment_fingerprint()


def test_v2_valid_terminal_error_retains_original_worker_failure(
    monkeypatch: pytest.MonkeyPatch, synthetic_bindings: dict[str, Any]
):
    event = _terminal_event(synthetic_bindings, "error")
    event["error"] = {"type": "WorkerError", "message": "original worker failure"}
    assert _validate(copy.deepcopy(event))["error"] == event["error"]

    calls: list[int] = []

    def child(index: int, _order: tuple[str, str], _deadline: float, **_kwargs: Any):
        calls.append(index)
        return copy.deepcopy(event)

    _configure_fake_execute(monkeypatch, child)
    report = harness.execute()
    assert calls == [1]
    assert report["decision"] == "incomplete_evidence"
    assert report["partial_result"] is True
    assert report["error"] == event["error"]


def test_v2_real_terminal_provenance_mismatch_is_fail_closed_and_keeps_context(
    synthetic_bindings: dict[str, Any]
):
    event = _terminal_event(synthetic_bindings, "error")
    event["error"] = {"type": "WorkerError", "message": "original worker failure"}
    event["environment_sha256"] = "0" * 64
    with pytest.raises(harness.WorkerError) as raised:
        _validate(event)
    message = str(raised.value)
    assert "terminal provenance identity failed" in message
    assert "environment_sha256" in message
    assert "worker_error=original worker failure" in message


def test_v2_ids_paths_and_sealed_hashes_are_independent_and_exact():
    assert worker.STUDY_ID == harness.STUDY_ID == "fused-greedy-compile-20260825-02"
    assert worker.RUN_ID == harness.RUN_ID == "fused-greedy-compile-validation-20260825-02"
    assert worker.AUTH_NONCE == harness.AUTH_NONCE == "cycle19-fused-greedy-compile-v2"
    harness_source = HARNESS_PATH.read_text(encoding="utf-8")
    worker_source = WORKER_PATH.read_text(encoding="utf-8")
    assert 'ATTEMPT_DIR = PROJECT_ROOT / ".friday-data" / "fused-greedy-compile-v2"' in harness_source
    assert 'ATTEMPT_PATH = ATTEMPT_DIR / "attempt.json"' in harness_source
    assert 'RESULT_PATH = Path(__file__).with_name("results.json")' in harness_source
    assert '"fused-greedy-compile-v2" / "attempt.json"' in worker_source
    assert worker.HARNESS.name == "measure_fused_greedy_compile_v2.py"
    assert worker.PREREGISTRATION == harness.PREREGISTRATION
    prereg_hash = _sha_file(worker.PREREGISTRATION)
    assert prereg_hash == worker.FROZEN_PREREGISTRATION_SHA256 == harness.FROZEN_PREREGISTRATION_SHA256
    assert worker.PROTOCOL_SHA256 == _sha_bytes(_canonical(worker.protocol_contract()))


def test_v2_default_paths_and_prompt_exclude_irrelevant_model_or_markdown_behavior(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(worker, "_run_worker", lambda _key: pytest.fail("worker execution reached"))
    monkeypatch.setattr(harness, "execute", lambda: pytest.fail("harness execution reached"))
    assert worker.main([]) == 78
    assert harness.main([]) == 78
    assert all(json.loads(line)["formal_claim"] is False for line in capsys.readouterr().out.splitlines())
    assert worker.MODEL_ID == "mlx-community/gemma-3-4b-it-4bit"
    assert "gemma-3-1b" not in worker.PLANNER_PROMPT
    assert "Return only a JSON object" in worker.PLANNER_PROMPT
    assert "no prose, markdown, or explanation" in worker.PLANNER_PROMPT
