import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import stat

import pytest

from ironmule_product.errors import ModelNotFound, StateError
from ironmule_product.state import ProductStore
from ironmule_product.types import ModelSpec


def _spec(tmp_path: Path, name: str = "org/model") -> ModelSpec:
    snapshot = tmp_path / name.replace("/", "--")
    snapshot.mkdir(exist_ok=True)
    return ModelSpec(name, "rev-1", str(snapshot), 123)


def test_setup_defaults_and_optimization_start_inactive(tmp_path):
    store = ProductStore(tmp_path / "state")

    with pytest.raises(StateError):
        store.settings()
    assert store.setup() == {
        "schema": 1,
        "mode": "desktop",
        "exact": True,
        "max_pending": 8,
        "request_timeout_s": 120,
        "optimization_paused": False,
    }
    assert store.optimization_status()["engine_started"] is False
    assert store.set_optimization_paused(True)["paused"] is True
    assert store.settings()["optimization_paused"] is True
    assert store.setup("server")["max_pending"] == 64
    assert store.settings()["optimization_paused"] is True


def test_model_registry_is_sorted_replaced_and_never_deletes_snapshot(tmp_path):
    store = ProductStore(tmp_path / "state")
    first = _spec(tmp_path, "z/model")
    second = _spec(tmp_path, "a/model")
    store.register_model(first)
    store.register_model(second)
    assert [item.model_id for item in store.models()] == ["a/model", "z/model"]
    replacement = ModelSpec(first.model_id, "rev-2", first.snapshot_path, 456)
    store.register_model(replacement)
    assert store.model(first.model_id) == replacement
    assert store.remove_model(first.model_id) is True
    assert Path(first.snapshot_path).is_dir()
    assert store.remove_model(first.model_id) is False
    with pytest.raises(ModelNotFound):
        store.model(first.model_id)


def test_state_files_are_private_atomic_json_and_symlinks_are_rejected(tmp_path):
    root = tmp_path / "state"
    store = ProductStore(root)
    snapshot = _spec(tmp_path).snapshot_path
    store.setup()
    store.register_model(_spec(tmp_path))
    assert json.loads((root / "settings.json").read_text())['schema'] == 1
    assert (root / "settings.json").stat().st_mode & 0o077 == 0
    assert (root / "models.json").stat().st_mode & 0o077 == 0
    assert (root / ".lock").stat().st_mode & 0o077 == 0
    link = tmp_path / "snapshot-link"
    link.symlink_to(snapshot, target_is_directory=True)
    with pytest.raises(StateError):
        store.register_model(ModelSpec("link/model", "r", str(link), 1))


def test_concurrent_model_registrations_do_not_lose_updates(tmp_path):
    store = ProductStore(tmp_path / "state")
    specs = [_spec(tmp_path, f"org/model-{index}") for index in range(12)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(store.register_model, specs))
    assert [item.model_id for item in store.models()] == sorted(item.model_id for item in specs)


def test_read_only_methods_do_not_create_absent_root_or_lock(tmp_path):
    root = tmp_path / "absent-state"
    store = ProductStore(root)

    with pytest.raises(StateError):
        store.settings()
    assert store.models() == []
    with pytest.raises(StateError):
        store.optimization_status()
    assert not root.exists()
    assert not (root / ".lock").exists()


def test_existing_public_root_is_rejected_without_changing_permissions(tmp_path):
    root = tmp_path / "public-state"
    root.mkdir(mode=0o755)
    os.chmod(root, 0o755)
    store = ProductStore(root)

    with pytest.raises(StateError):
        store.settings()
    with pytest.raises(StateError):
        store.models()
    with pytest.raises(StateError):
        store.setup()
    assert stat.S_IMODE(root.stat().st_mode) == 0o755
    assert not (root / ".lock").exists()


def test_setup_validates_existing_registry_before_writing_settings(tmp_path):
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    os.chmod(root, 0o700)
    settings = {
        "schema": 1,
        "mode": "desktop",
        "exact": True,
        "max_pending": 8,
        "request_timeout_s": 120,
        "optimization_paused": False,
    }
    (root / "settings.json").write_text(json.dumps(settings))
    (root / "models.json").write_text(json.dumps({"schema": 1, "models": [{"model_id": "bad"}]}))
    before = (root / "settings.json").read_bytes()

    with pytest.raises(StateError):
        ProductStore(root).setup("server")
    assert (root / "settings.json").read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.update(schema=True),
        lambda value: value.update(exact=1),
        lambda value: value.update(mode=1),
        lambda value: value.update(request_timeout_s=120.0),
    ],
)
def test_settings_schema_requires_strict_types(tmp_path, change):
    store = ProductStore(tmp_path / "state")
    store.setup()
    path = store.root / "settings.json"
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))
    os.chmod(path, 0o600)
    with pytest.raises(StateError):
        store.settings()


def test_malformed_model_metadata_is_rejected(tmp_path):
    store = ProductStore(tmp_path / "state")
    store.setup()
    path = store.root / "models.json"
    path.write_text(json.dumps({
        "schema": 1,
        "models": [{
            "model_id": "org/model",
            "revision": "r1",
            "snapshot_path": "/tmp/snapshot",
            "weight_bytes": True,
        }],
    }))
    os.chmod(path, 0o600)
    with pytest.raises(StateError):
        store.models()


def test_model_registry_rejects_unknown_metadata_fields(tmp_path):
    store = ProductStore(tmp_path / "state")
    store.setup()
    path = store.root / "models.json"
    path.write_text(json.dumps({"schema": 1, "models": [{
        "model_id": "org/model",
        "revision": "r1",
        "snapshot_path": "/tmp/snapshot",
        "weight_bytes": 1,
        "unexpected": "must not be ignored",
    }]}))
    os.chmod(path, 0o600)
    with pytest.raises(StateError):
        store.models()


def test_concurrent_pause_updates_are_serialized_on_filesystem(tmp_path):
    store = ProductStore(tmp_path / "state")
    store.setup()
    values = [index % 2 == 0 for index in range(20)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(store.set_optimization_paused, values))
    assert all(result["paused"] in (True, False) for result in results)
    assert type(store.settings()["optimization_paused"]) is bool


def test_hard_linked_state_file_is_rejected(tmp_path):
    store = ProductStore(tmp_path / "state")
    store.setup()
    source = tmp_path / "outside.json"
    source.write_text((store.root / "models.json").read_text())
    os.chmod(source, 0o600)
    (store.root / "models.json").unlink()
    os.link(source, store.root / "models.json")
    with pytest.raises(StateError):
        store.models()
