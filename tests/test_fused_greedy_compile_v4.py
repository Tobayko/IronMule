from __future__ import annotations

from pathlib import Path


# Cycle 21 repeats the full Cycle-20 contract and adds independent parent/worker
# snapshot-identity parity.  Re-run every v3 regression against the v4 paths and
# sealed IDs, then add the parity-specific cases below.
_BASE_TEST = Path(__file__).with_name("test_fused_greedy_compile_v3.py")
_BASE_SOURCE = _BASE_TEST.read_text(encoding="utf-8")
_TRANSFORMS = (
    ("fused_greedy_compile_v3", "fused_greedy_compile_v4"),
    ("cycle20-fused-greedy-compile-v3", "cycle21-fused-greedy-compile-v4"),
    ("fused-greedy-compile-v3", "fused-greedy-compile-v4"),
    ("fused-greedy-compile-20260825-03", "fused-greedy-compile-20260825-04"),
    (
        "fused-greedy-compile-validation-20260825-03",
        "fused-greedy-compile-validation-20260825-04",
    ),
    (
        "c07ca8fbb7a6ef393d87541532b5732fe95c71e283aeaf2165e315ef2aff4009",
        "a734975191de7c77a4966c42c0225d8bdbe89d215e24ff63600affef0599dadf",
    ),
)
_V4_SOURCE = _BASE_SOURCE
for _old, _new in _TRANSFORMS:
    assert _old in _V4_SOURCE
    _V4_SOURCE = _V4_SOURCE.replace(_old, _new)
assert _V4_SOURCE != _BASE_SOURCE
exec(compile(_V4_SOURCE, str(Path(__file__).resolve()), "exec"), globals())


# The inherited Cycle-18 preflight fixture predates the v4 worker-builder parity
# call.  Add only that new public contract to its synthetic worker.
_BASE_CONFIGURE_PREFLIGHT = _configure_preflight


def _configure_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[SimpleNamespace, dict[str, Any]]:
    snapshot, identity = _BASE_CONFIGURE_PREFLIGHT(monkeypatch, tmp_path)
    module = harness._module()
    module._snapshot_identity = lambda _snapshot: (
        identity["snapshot_path"],
        copy.deepcopy(identity["execution_stat_manifest"]),
        copy.deepcopy(identity["snapshot_files_sha256"]),
        copy.deepcopy(identity["execution_files_sha256"]),
    )
    return snapshot, identity


def _fake_snapshot(
    tmp_path: Path,
    *,
    tokenizer_json: bool = True,
    tokenizer_model: bool = False,
) -> SimpleNamespace:
    repository = tmp_path / "models--local--snapshot"
    root = repository / "snapshots" / harness.MODEL_REVISION
    root.mkdir(parents=True)
    payloads = {
        "config.json": b'{"model_type":"synthetic"}\n',
        "tokenizer_config.json": b'{"synthetic":true}\n',
        "generation_config.json": b'{"eos_token_id":[1,106]}\n',
        "model.safetensors": b"synthetic weights",
    }
    if tokenizer_json:
        payloads["tokenizer.json"] = b'{"version":"1.0"}\n'
    if tokenizer_model:
        payloads["tokenizer.model"] = b"synthetic sentencepiece"
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    return SimpleNamespace(
        model_id=harness.MODEL_ID,
        revision=harness.MODEL_REVISION,
        path=str(root),
        weight_files=("model.safetensors",),
        weight_bytes=len(payloads["model.safetensors"]),
    )


def _assert_builder_parity(snapshot: SimpleNamespace) -> dict[str, Any]:
    parent = harness._snapshot_identity(snapshot)
    child = worker._snapshot_identity(snapshot)
    assert child == (
        parent["snapshot_path"],
        parent["execution_stat_manifest"],
        parent["snapshot_files_sha256"],
        parent["execution_files_sha256"],
    )
    return parent


@pytest.mark.parametrize(
    ("tokenizer_json", "tokenizer_model", "selected"),
    [(True, True, "tokenizer.json"), (False, True, "tokenizer.model")],
)
def test_fake_snapshot_builders_are_equal_and_select_exact_tokenizer_filename(
    tmp_path: Path, tokenizer_json: bool, tokenizer_model: bool, selected: str
):
    snapshot = _fake_snapshot(
        tmp_path, tokenizer_json=tokenizer_json, tokenizer_model=tokenizer_model
    )
    identity = _assert_builder_parity(snapshot)
    manifest = identity["execution_stat_manifest"]
    expected_names = {
        "config.json",
        "tokenizer_config.json",
        selected,
        "model.safetensors",
        "generation_config.json",
    }
    assert set(manifest) == expected_names
    assert set(identity["execution_files_sha256"]) == expected_names
    assert set(identity["snapshot_files_sha256"]) == expected_names - {
        "generation_config.json"
    }
    unselected = "tokenizer.model" if selected == "tokenizer.json" else "tokenizer.json"
    assert unselected not in manifest

    for name, item in manifest.items():
        metadata = Path(item["path"]).stat()
        assert set(item) == {"dev", "inode", "mtime_ns", "path", "size"}
        assert item == {
            "dev": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
            "mtime_ns": int(metadata.st_mtime_ns),
            "path": str(Path(snapshot.path, name).resolve()),
            "size": int(metadata.st_size),
        }
    assert identity["snapshot_sha256"] == harness._sha256_bytes(
        harness._canonical(identity["snapshot_files_sha256"])
    )


def test_real_local_resolver_parent_worker_identity_is_equal_without_model_or_mlx_load(
    monkeypatch: pytest.MonkeyPatch,
):
    imported: list[str] = []
    original_import = __import__

    def guarded_import(name: str, *args: Any, **kwargs: Any):
        if name == "mlx" or name.startswith("mlx.") or name.startswith("mlx_lm"):
            imported.append(name)
            raise AssertionError("MLX or model loading import reached in read-only parity test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    for key, value in worker.OFFLINE_ENV.items():
        monkeypatch.setenv(key, value)
    bench = _load("cycle21_readonly_snapshot_resolver", ROOT / "tools" / "_bench.py")
    snapshot = bench.resolve_local_model_snapshot(harness.MODEL_ID)
    assert snapshot.revision == harness.MODEL_REVISION
    identity = _assert_builder_parity(snapshot)
    assert identity["snapshot_sha256"] == harness.EXPECTED_SNAPSHOT_SHA256
    assert identity["weight_sha256"] == {
        "model.safetensors": harness.EXPECTED_WEIGHT_SHA256
    }
    assert identity["model_snapshot_weight_files"] == ["model.safetensors"]
    assert set(identity["execution_stat_manifest"]) == set(
        identity["execution_files_sha256"]
    )
    for item in identity["execution_stat_manifest"].values():
        assert set(item) == {"dev", "inode", "mtime_ns", "path", "size"}
        assert all(type(item[field]) is int for field in ("dev", "inode", "mtime_ns", "size"))
        assert type(item["path"]) is str and item["path"]
    assert imported == []


def _configure_snapshot_parity_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Any]:
    snapshot = _fake_snapshot(tmp_path)
    parent_identity = harness._snapshot_identity(snapshot)
    child_path, child_manifest, child_files, child_execution = worker._snapshot_identity(snapshot)
    calls: list[str] = []
    state: dict[str, Any] = {
        "snapshot": snapshot,
        "parent": copy.deepcopy(parent_identity),
        "child_path": child_path,
        "child_manifest": copy.deepcopy(child_manifest),
        "child_files": copy.deepcopy(child_files),
        "child_execution": copy.deepcopy(child_execution),
        "calls": calls,
    }
    module = SimpleNamespace(
        PROMPT_SHA256=harness.EXPECTED_PROMPT_SHA256,
        load_count=0,
        protocol_contract=lambda: {
            "study_id": harness.STUDY_ID,
            "run_id": harness.RUN_ID,
            "arms": list(harness.ARM_NAMES),
            "capacity": 512,
            "warmups": 8,
        },
    )

    def parent_builder(_snapshot: Any) -> dict[str, Any]:
        calls.append("parent")
        return copy.deepcopy(state["parent"])

    def child_builder(_snapshot: Any) -> tuple[Any, ...]:
        calls.append("worker")
        return (
            state["child_path"],
            copy.deepcopy(state["child_manifest"]),
            copy.deepcopy(state["child_files"]),
            copy.deepcopy(state["child_execution"]),
        )

    module._snapshot_identity = child_builder
    monkeypatch.setattr(harness, "_module", lambda: module)
    monkeypatch.setattr(harness, "_snapshot_identity", parent_builder)
    monkeypatch.setattr(harness, "_clean_worktree", lambda: ("a" * 40, ""))
    monkeypatch.setattr(harness, "_require_target", lambda: None)
    monkeypatch.setattr(harness, "_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(
        harness, "EXPECTED_SNAPSHOT_SHA256", parent_identity["snapshot_sha256"]
    )
    monkeypatch.setattr(
        harness,
        "EXPECTED_WEIGHT_SHA256",
        parent_identity["weight_sha256"]["model.safetensors"],
    )
    bench = types.ModuleType("_bench")
    bench.require_ac_power = lambda: "AC Power"
    bench.resolve_local_model_snapshot = lambda _model: snapshot
    monkeypatch.setitem(sys.modules, "_bench", bench)
    state["module"] = module
    return state


@pytest.mark.parametrize(
    "mutation",
    [
        "stat_dev",
        "stat_inode",
        "stat_mtime_ns",
        "stat_size",
        "stat_path",
        "snapshot_sha256",
        "snapshot_file_hash",
        "execution_hash",
        "generation_hash",
        "weight_hash",
        "weight_name",
        "revision",
        "snapshot_path",
    ],
)
def test_each_snapshot_identity_mutation_fails_before_marker_or_model_load(
    mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    state = _configure_snapshot_parity_preflight(monkeypatch, tmp_path)
    if mutation.startswith("stat_"):
        field = mutation.removeprefix("stat_")
        current = state["child_manifest"]["model.safetensors"][field]
        state["child_manifest"]["model.safetensors"][field] = (
            current + 1 if type(current) is int else current + ".mutated"
        )
    elif mutation == "snapshot_sha256":
        state["parent"]["snapshot_sha256"] = "0" * 64
    elif mutation == "snapshot_file_hash":
        state["child_files"]["config.json"] = "0" * 64
    elif mutation == "execution_hash":
        state["child_execution"]["config.json"] = "0" * 64
    elif mutation == "generation_hash":
        state["child_execution"]["generation_config.json"] = "0" * 64
    elif mutation == "weight_hash":
        state["parent"]["weight_sha256"]["model.safetensors"] = "0" * 64
    elif mutation == "weight_name":
        digest = state["parent"]["weight_sha256"].pop("model.safetensors")
        state["parent"]["weight_sha256"]["renamed.safetensors"] = digest
    elif mutation == "revision":
        state["snapshot"].revision = "0" * 40
    elif mutation == "snapshot_path":
        state["child_path"] += ".mutated"
    else:  # pragma: no cover - the parameter list is closed above.
        raise AssertionError(mutation)

    blocked_imports: list[str] = []
    original_import = __import__

    def guarded_import(name: str, *args: Any, **kwargs: Any):
        if name == "mlx" or name.startswith("mlx.") or name.startswith("mlx_lm"):
            blocked_imports.append(name)
            raise AssertionError("model/MLX import reached before snapshot parity gate")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    with pytest.raises(harness.StudyError):
        harness._preflight()
    assert state["module"].load_count == 0
    assert blocked_imports == []
    assert not harness.ATTEMPT_PATH.exists() and not harness.ATTEMPT_PATH.is_symlink()
    assert not harness.RESULT_PATH.exists() and not harness.RESULT_PATH.is_symlink()


def test_parent_worker_parity_completes_before_marker_or_result_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    state = _configure_snapshot_parity_preflight(monkeypatch, tmp_path)

    class MarkerReached(RuntimeError):
        pass

    def marker(*_args: Any, **_kwargs: Any) -> None:
        state["calls"].append("marker")
        raise MarkerReached

    monkeypatch.setattr(harness, "_write_exclusive", marker)
    with pytest.raises(MarkerReached):
        harness.execute()
    assert state["calls"] == ["parent", "worker", "marker"]
    assert state["module"].load_count == 0
    assert not harness.ATTEMPT_PATH.exists() and not harness.ATTEMPT_PATH.is_symlink()
    assert not harness.RESULT_PATH.exists() and not harness.RESULT_PATH.is_symlink()


@pytest.mark.parametrize("path", [WORKER_PATH, HARNESS_PATH])
def test_self_check_source_cannot_import_mlx_mlx_lm_or_call_model_load(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    self_check = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_self_check"
    )
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(self_check):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert not any(
        name == "mlx" or name.startswith("mlx.") or name.startswith("mlx_lm")
        for name in imported
    )
    assert "load" not in called
