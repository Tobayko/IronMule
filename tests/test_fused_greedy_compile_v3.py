from __future__ import annotations

from pathlib import Path


# Cycle 20 keeps the Cycle-18 measurement contract and adds a fail-closed Git
# lifecycle.  Execute the established offline suite against the v3 files so all
# earlier correctness, resource, timeout, parser, and evidence gates remain
# mandatory without copying hundreds of lines of fixtures.
_BASE_TEST = Path(__file__).with_name("test_fused_greedy_compile.py")
_BASE_SOURCE = _BASE_TEST.read_text(encoding="utf-8")
_V3_SOURCE = _BASE_SOURCE.replace(
    'EXP = ROOT / "experiments" / "fused_greedy_compile"',
    'EXP = ROOT / "experiments" / "fused_greedy_compile_v3"',
).replace(
    'HARNESS_PATH = EXP / "measure_fused_greedy_compile.py"',
    'HARNESS_PATH = EXP / "measure_fused_greedy_compile_v3.py"',
)
assert _V3_SOURCE != _BASE_SOURCE
exec(compile(_V3_SOURCE, str(Path(__file__).resolve()), "exec"), globals())


# The v3 harness deliberately sees its own untracked result after the marker is
# created.  Adapt the inherited fake-execute fixture to that new, stricter
# lifecycle; the production validator itself is exercised against real Git
# repositories below.
_BASE_CONFIGURE_FAKE_EXECUTE = _configure_fake_execute


def _configure_fake_execute(
    monkeypatch: pytest.MonkeyPatch, child: Any, *, postflight_snapshot_mutates: bool = False
) -> list[dict[str, Any]]:
    checkpoints = _BASE_CONFIGURE_FAKE_EXECUTE(
        monkeypatch, child, postflight_snapshot_mutates=postflight_snapshot_mutates
    )

    def fake_git(*args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if args[:2] == ("status", "--porcelain=v1"):
            return f"?? {harness.RESULT_RELATIVE}"
        return ""

    monkeypatch.setattr(harness, "_git", fake_git)
    return checkpoints


def test_child_exception_stops_schedule_and_writes_fail_safe_final_result(
    monkeypatch: pytest.MonkeyPatch,
):
    """A generic orchestration failure is incomplete evidence, not a resource claim."""
    calls: list[int] = []

    def child(index: int, *_args: Any, **_kwargs: Any):
        calls.append(index)
        raise harness.WorkerError("synthetic child failure")

    checkpoints = _configure_fake_execute(monkeypatch, child)
    report = harness.execute()
    assert calls == [1]
    assert report["decision"] == "incomplete_evidence"
    assert report["partial_result"] is True
    assert report["error"] == {
        "type": "WorkerError",
        "message": "synthetic child failure",
    }
    assert checkpoints[0]["status"] == "running"
    assert checkpoints[-1]["decision"] == "incomplete_evidence"


def test_v3_environment_fingerprints_are_byte_identical_and_bind_removed_environment():
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


def test_v3_wrong_environment_hash_fails_before_model_load_and_preserves_error(
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


def test_v3_valid_terminal_error_retains_original_worker_failure(
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


def test_v3_real_terminal_provenance_mismatch_is_fail_closed_and_keeps_context(
    synthetic_bindings: dict[str, Any],
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


def test_v3_ids_paths_and_sealed_hashes_are_independent_and_exact():
    assert worker.STUDY_ID == harness.STUDY_ID == "fused-greedy-compile-20260825-03"
    assert worker.RUN_ID == harness.RUN_ID == "fused-greedy-compile-validation-20260825-03"
    assert worker.AUTH_NONCE == harness.AUTH_NONCE == "cycle20-fused-greedy-compile-v3"
    assert worker.RESULT_RELATIVE == harness.RESULT_RELATIVE == (
        "experiments/fused_greedy_compile_v3/results.json"
    )
    harness_source = HARNESS_PATH.read_text(encoding="utf-8")
    worker_source = WORKER_PATH.read_text(encoding="utf-8")
    assert 'ATTEMPT_DIR = PROJECT_ROOT / ".friday-data" / "fused-greedy-compile-v3"' in harness_source
    assert 'ATTEMPT_PATH = ATTEMPT_DIR / "attempt.json"' in harness_source
    assert 'RESULT_PATH = Path(__file__).with_name("results.json")' in harness_source
    assert '"fused-greedy-compile-v3" / "attempt.json"' in worker_source
    assert worker.HARNESS.name == "measure_fused_greedy_compile_v3.py"
    assert worker.PREREGISTRATION == harness.PREREGISTRATION
    prereg_hash = _sha_file(worker.PREREGISTRATION)
    assert prereg_hash == "c07ca8fbb7a6ef393d87541532b5732fe95c71e283aeaf2165e315ef2aff4009"
    assert prereg_hash == worker.FROZEN_PREREGISTRATION_SHA256
    assert prereg_hash == harness.FROZEN_PREREGISTRATION_SHA256
    assert worker.PROTOCOL_SHA256 == _sha_bytes(_canonical(worker.protocol_contract()))


def test_v3_default_paths_and_prompt_exclude_irrelevant_model_or_markdown_behavior(
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


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


@pytest.fixture
def v3_git_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "cycle20-tests@example.invalid")
    _git(repo, "config", "user.name", "Cycle 20 Tests")
    (repo / ".gitignore").write_text(".friday-data/\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("sealed\n", encoding="utf-8")
    experiment = repo / "experiments" / "fused_greedy_compile_v3"
    experiment.mkdir(parents=True)
    _git(repo, "add", ".gitignore", "tracked.txt")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "sealed baseline")

    marker_dir = repo / ".friday-data" / "fused-greedy-compile-v3"
    marker = marker_dir / "attempt.json"
    result = experiment / "results.json"
    monkeypatch.setattr(harness, "PROJECT_ROOT", repo)
    monkeypatch.setattr(harness, "ATTEMPT_DIR", marker_dir)
    monkeypatch.setattr(harness, "ATTEMPT_PATH", marker)
    monkeypatch.setattr(harness, "RESULT_PATH", result)
    monkeypatch.setattr(worker, "PROJECT_ROOT", repo)
    monkeypatch.setattr(worker, "RESULT_PATH", result)
    return {
        "repo": repo,
        "experiment": experiment,
        "marker_dir": marker_dir,
        "marker": marker,
        "result": result,
    }


def _create_private_marker(paths: dict[str, Path]) -> None:
    paths["marker_dir"].mkdir(parents=True, mode=0o700)
    os.chmod(paths["marker_dir"], 0o700)
    paths["marker"].write_bytes(b'{"formal_claim":false}\n')
    os.chmod(paths["marker"], 0o600)


def _prepare_exact_post_marker_state(paths: dict[str, Path]) -> None:
    _create_private_marker(paths)
    harness._atomic_result(
        {
            "study_id": harness.STUDY_ID,
            "run_id": harness.RUN_ID,
            "formal_claim": False,
            "status": "running",
        }
    )
    os.chmod(paths["result"], 0o644)


def _porcelain(paths: dict[str, Path]) -> list[str]:
    output = _git(
        paths["repo"],
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)ProjectAtlas",
    )
    return output.splitlines() if output else []


def test_real_git_lifecycle_starts_clean_and_accepts_only_parent_owned_result(
    v3_git_repo: dict[str, Path],
):
    revision, dirty = harness._clean_worktree()
    assert revision == _git(v3_git_repo["repo"], "rev-parse", "HEAD")
    assert dirty == ""
    assert _porcelain(v3_git_repo) == []

    _prepare_exact_post_marker_state(v3_git_repo)
    expected = [f"?? {harness.RESULT_RELATIVE}"]
    assert _porcelain(v3_git_repo) == expected
    assert harness._allowed_post_marker_status(expected) is True
    assert worker.allowed_post_marker_status(expected) is True
    harness._validate_post_marker_git_state()
    worker._validate_post_marker_git_state()


@pytest.mark.parametrize(
    "mutation",
    ["foreign_untracked", "tracked_modification", "staged_change", "rename", "temp_result"],
)
def test_real_git_lifecycle_rejects_every_change_beyond_own_result(
    mutation: str, v3_git_repo: dict[str, Path]
):
    _prepare_exact_post_marker_state(v3_git_repo)
    repo = v3_git_repo["repo"]
    if mutation == "foreign_untracked":
        (repo / "foreign.txt").write_text("foreign\n", encoding="utf-8")
    elif mutation == "tracked_modification":
        (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    elif mutation == "staged_change":
        (repo / "staged.txt").write_text("staged\n", encoding="utf-8")
        _git(repo, "add", "staged.txt")
    elif mutation == "rename":
        _git(repo, "mv", "tracked.txt", "renamed.txt")
    elif mutation == "temp_result":
        (v3_git_repo["experiment"] / ".results.json.123.tmp").write_text(
            "temporary\n", encoding="utf-8"
        )
    else:  # pragma: no cover - the parameter list is closed above.
        raise AssertionError(mutation)

    assert _porcelain(v3_git_repo) != [f"?? {harness.RESULT_RELATIVE}"]
    with pytest.raises((harness.StudyError, harness.WorkerError)):
        harness._validate_post_marker_git_state()
    with pytest.raises(worker.WorkerError):
        worker._validate_post_marker_git_state()


@pytest.mark.parametrize(
    "unsafe_state",
    ["result_symlink", "result_mode", "marker_symlink", "marker_mode", "marker_dir_mode"],
)
def test_post_marker_paths_reject_symlinks_and_unsafe_modes(
    unsafe_state: str, v3_git_repo: dict[str, Path]
):
    _prepare_exact_post_marker_state(v3_git_repo)
    if unsafe_state == "result_symlink":
        v3_git_repo["result"].unlink()
        target = v3_git_repo["experiment"] / "result-target"
        target.write_text("target\n", encoding="utf-8")
        v3_git_repo["result"].symlink_to(target)
    elif unsafe_state == "result_mode":
        os.chmod(v3_git_repo["result"], 0o600)
    elif unsafe_state == "marker_symlink":
        v3_git_repo["marker"].unlink()
        target = v3_git_repo["marker_dir"] / "marker-target"
        target.write_text("target\n", encoding="utf-8")
        os.chmod(target, 0o600)
        v3_git_repo["marker"].symlink_to(target)
    elif unsafe_state == "marker_mode":
        os.chmod(v3_git_repo["marker"], 0o644)
    elif unsafe_state == "marker_dir_mode":
        os.chmod(v3_git_repo["marker_dir"], 0o755)
    else:  # pragma: no cover
        raise AssertionError(unsafe_state)

    with pytest.raises((harness.StudyError, harness.WorkerError)):
        harness._validate_post_marker_git_state()
    with pytest.raises(worker.WorkerError):
        worker._validate_post_marker_git_state()


def test_post_marker_paths_reject_non_owner_even_when_modes_are_correct(
    monkeypatch: pytest.MonkeyPatch, v3_git_repo: dict[str, Path]
):
    _prepare_exact_post_marker_state(v3_git_repo)
    actual_uid = v3_git_repo["result"].lstat().st_uid
    monkeypatch.setattr(harness.os, "geteuid", lambda: actual_uid + 1)
    with pytest.raises(harness.StudyError, match="runner-owned|private"):
        harness._validate_post_marker_git_state()
    with pytest.raises(worker.WorkerError, match="runner-owned|private"):
        worker._validate_post_marker_git_state()


def test_post_marker_status_helpers_reject_clean_extra_modified_and_wrong_result_paths():
    exact = [f"?? {harness.RESULT_RELATIVE}"]
    rejected = [
        [],
        exact + ["?? foreign.txt"],
        [" M experiments/fused_greedy_compile_v3/worker.py", *exact],
        ["A  staged.txt", *exact],
        ["R  tracked.txt -> renamed.txt", *exact],
        ["?? experiments/fused_greedy_compile_v2/results.json"],
    ]
    assert harness._allowed_post_marker_status(exact)
    assert worker.allowed_post_marker_status(exact)
    for lines in rejected:
        assert not harness._allowed_post_marker_status(lines)
        assert not worker.allowed_post_marker_status(lines)


def test_preflight_existing_result_blocks_before_clean_target_snapshot_or_model_gate(
    monkeypatch: pytest.MonkeyPatch,
):
    harness.RESULT_PATH.write_bytes(b'{"already":"exists"}\n')
    os.chmod(harness.RESULT_PATH, 0o644)
    reached: list[str] = []
    monkeypatch.setattr(harness, "_clean_worktree", lambda: reached.append("git"))
    monkeypatch.setattr(harness, "_require_target", lambda: reached.append("target"))
    with pytest.raises(harness.StudyError, match="existing evidence"):
        harness._preflight()
    assert reached == []


def test_generic_terminal_error_is_incomplete_but_real_resource_terminal_is_resource_failed(
    monkeypatch: pytest.MonkeyPatch, synthetic_bindings: dict[str, Any]
):
    generic = _terminal_event(synthetic_bindings, "error")
    generic["error"] = {"type": "WorkerError", "message": "generic failure"}
    _configure_fake_execute(monkeypatch, lambda *_args, **_kwargs: copy.deepcopy(generic))
    generic_report = harness.execute()
    assert generic_report["decision"] == "incomplete_evidence"
    assert generic_report["gates"]["resource_pass"] is True
    assert generic_report["gates"]["budget_pass"] is True


def test_real_resource_exception_is_not_downgraded_to_incomplete(
    monkeypatch: pytest.MonkeyPatch,
):
    def child(*_args: Any, **_kwargs: Any):
        raise MemoryError("synthetic out of memory")

    _configure_fake_execute(monkeypatch, child)
    report = harness.execute()
    assert report["decision"] == "resource_or_budget_failed"
    assert report["partial_result"] is True
    assert report["gates"]["resource_pass"] is False
    assert report["gates"]["budget_pass"] is False
    assert report["error"]["type"] == "MemoryError"


def test_resource_terminal_stops_without_retry_and_retains_resource_decision(
    monkeypatch: pytest.MonkeyPatch, synthetic_bindings: dict[str, Any]
):
    calls: list[int] = []

    def child(index: int, *_args: Any, **_kwargs: Any):
        calls.append(index)
        return _terminal_event(synthetic_bindings, "resource_or_budget_failed", index=index)

    _configure_fake_execute(monkeypatch, child)
    report = harness.execute()
    assert calls == [1]
    assert report["decision"] == "resource_or_budget_failed"
    assert report["partial_result"] is True
    assert report["gates"]["resource_pass"] is False
    assert report["gates"]["budget_pass"] is False


def test_correctness_terminal_stops_without_retry_and_retains_correctness_decision(
    monkeypatch: pytest.MonkeyPatch, synthetic_bindings: dict[str, Any]
):
    calls: list[int] = []

    def child(index: int, *_args: Any, **_kwargs: Any):
        calls.append(index)
        return _terminal_event(synthetic_bindings, "correctness_failed", index=index)

    _configure_fake_execute(monkeypatch, child)
    report = harness.execute()
    assert calls == [1]
    assert report["decision"] == "correctness_failed"
    assert report["partial_result"] is True
    assert report["gates"]["correctness_pass"] is False


@pytest.mark.parametrize("later_failure", ["generic", "provenance", "snapshot"])
def test_resource_failure_is_never_downgraded_by_later_non_resource_postflight_failure(
    later_failure: str, monkeypatch: pytest.MonkeyPatch
):
    calls: list[int] = []

    def child(index: int, *_args: Any, **_kwargs: Any):
        calls.append(index)
        raise MemoryError("synthetic child out of memory")

    checkpoints = _configure_fake_execute(monkeypatch, child)
    if later_failure == "generic":
        target = harness._target_info()
        target_calls = 0

        def fail_target_postflight() -> dict[str, Any]:
            nonlocal target_calls
            target_calls += 1
            if target_calls == 1:
                return copy.deepcopy(target)
            raise RuntimeError("postflight verifier unavailable")

        monkeypatch.setattr(harness, "_target_info", fail_target_postflight)
    elif later_failure == "provenance":
        fingerprint_calls = 0

        def mutate_fingerprint_postflight() -> dict[str, str]:
            nonlocal fingerprint_calls
            fingerprint_calls += 1
            digest = "c" * 64 if fingerprint_calls <= 2 else "d" * 64
            return {"worker.py": digest}

        monkeypatch.setattr(harness, "code_fingerprints", mutate_fingerprint_postflight)
    elif later_failure == "snapshot":
        sys.modules["_bench"].resolve_local_model_snapshot = lambda _model: (
            _ for _ in ()
        ).throw(RuntimeError("snapshot verifier unavailable"))
    else:  # pragma: no cover - the parameter list is closed above.
        raise AssertionError(later_failure)

    report = harness.execute()
    assert calls == [1]
    assert report["decision"] == "resource_or_budget_failed"
    assert report["partial_result"] is True
    assert report["gates"]["resource_pass"] is False
    assert report["gates"]["budget_pass"] is False
    assert report["error"] == {
        "type": "MemoryError",
        "message": "synthetic child out of memory",
    }
    assert checkpoints[-1]["decision"] == "resource_or_budget_failed"


def test_genuine_later_postflight_resource_failure_upgrades_incomplete_evidence(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[int] = []

    def child(index: int, *_args: Any, **_kwargs: Any):
        calls.append(index)
        raise harness.WorkerError("synthetic generic child failure")

    checkpoints = _configure_fake_execute(monkeypatch, child)
    sys.modules["_bench"].resolve_local_model_snapshot = lambda _model: (
        _ for _ in ()
    ).throw(MemoryError("postflight snapshot out of memory"))

    report = harness.execute()
    assert calls == [1]
    assert report["decision"] == "resource_or_budget_failed"
    assert report["partial_result"] is True
    assert report["gates"]["resource_pass"] is False
    assert report["gates"]["budget_pass"] is False
    assert report["error"] == {
        "type": "WorkerError",
        "message": "synthetic generic child failure",
    }
    assert checkpoints[-1]["decision"] == "resource_or_budget_failed"
