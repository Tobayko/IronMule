import json
from types import SimpleNamespace

import pytest

from ironmule.model_identity import (
    ModelIdentity,
    ModelIdentityError,
    build_model_identity,
    resolve_model_source,
    select_cached_snapshot,
)


def model_dir(root, *, content="weights", quantisation=None):
    root.mkdir(parents=True)
    (root / "model.safetensors").write_text(content)
    (root / "tokenizer.json").write_text('{"tokens": ["a"]}')
    (root / "tokenizer_config.json").write_text('{"version": 1}')
    (root / "config.json").write_text(json.dumps({
        "model_type": "test-architecture",
        "quantization": quantisation or {"bits": 4, "group_size": 64},
    }))
    return root


def test_identity_is_deterministic_path_free_and_round_trips(tmp_path):
    first = model_dir(tmp_path / "one")
    second = model_dir(tmp_path / "two")
    a = build_model_identity("org/model", first, "revision")
    b = build_model_identity("org/model", second, "revision")
    assert a == b
    assert ModelIdentity.from_dict(a.to_dict()) == a
    rendered = json.dumps(a.to_dict())
    assert str(tmp_path) not in rendered
    assert a.quantisation == {"bits": 4, "group_size": 64}


def test_manifest_tokenizer_and_quantisation_changes_move_identity(tmp_path):
    root = model_dir(tmp_path / "model")
    baseline = build_model_identity("org/model", root, "revision")
    (root / "model.safetensors").write_text("changed weights")
    weights = build_model_identity("org/model", root, "revision")
    assert weights.model_manifest_sha256 != baseline.model_manifest_sha256
    assert weights.identity_sha256 != baseline.identity_sha256

    (root / "tokenizer.json").write_text('{"tokens": ["b"]}')
    tokenizer = build_model_identity("org/model", root, "revision")
    assert tokenizer.tokenizer_sha256 != weights.tokenizer_sha256

    config = json.loads((root / "config.json").read_text())
    config["quantization"]["bits"] = 8
    (root / "config.json").write_text(json.dumps(config))
    quantized = build_model_identity("org/model", root, "revision")
    assert quantized.quantisation_sha256 != tokenizer.quantisation_sha256


def test_local_directory_uses_path_free_id_and_content_revision(tmp_path):
    root = model_dir(tmp_path / "private-model")
    identity = build_model_identity(str(root), root)
    assert identity.model_id == "local:private-model"
    assert identity.revision.startswith("local-")
    assert str(tmp_path) not in json.dumps(identity.to_dict())


def test_hf_snapshot_revision_is_recognized_and_mismatch_rejected(tmp_path):
    snapshot = model_dir(tmp_path / "snapshots" / "abc123")
    identity = build_model_identity("org/model", snapshot)
    assert identity.revision == "abc123"
    with pytest.raises(ModelIdentityError, match="does not match"):
        build_model_identity("org/model", snapshot, "other")


def test_local_symlink_cannot_escape_but_hf_blob_link_is_allowed(tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_text("secret")
    local = model_dir(tmp_path / "local")
    (local / "external.bin").symlink_to(outside)
    with pytest.raises(ModelIdentityError, match="escapes"):
        build_model_identity(str(local), local)

    repo = tmp_path / "models--org--model"
    blob = repo / "blobs" / "abc"
    blob.parent.mkdir(parents=True)
    blob.write_text("weights")
    snapshot = model_dir(repo / "snapshots" / "revision")
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").symlink_to(blob)
    assert build_model_identity("org/model", snapshot).revision == "revision"


def test_cached_resolution_requires_one_exact_revision(tmp_path):
    first = model_dir(tmp_path / "snapshots" / "first")
    second = model_dir(tmp_path / "snapshots" / "second")
    cache = SimpleNamespace(repos=[SimpleNamespace(
        repo_id="org/model",
        revisions=[
            SimpleNamespace(commit_hash="first", snapshot_path=first),
            SimpleNamespace(commit_hash="second", snapshot_path=second),
        ],
    )])
    with pytest.raises(ModelIdentityError, match="unique cached revision"):
        select_cached_snapshot(cache, "org/model")
    path, revision = select_cached_snapshot(cache, "org/model", "second")
    assert path == second.resolve() and revision == "second"
    resolved = resolve_model_source("org/model", revision="first", cache=cache)
    assert resolved.path == first.resolve()
    assert resolved.identity.revision == "first"
    # The model IS cached here, just not at that pin; the message must say which.
    with pytest.raises(ModelIdentityError, match="cached, but not at revision"):
        resolve_model_source("org/model", revision="missing", cache=cache)


@pytest.mark.parametrize("mutation,needle", [
    (lambda root: (root / "config.json").unlink(), "config.json"),
    (lambda root: (root / "tokenizer.json").unlink() or (root / "tokenizer_config.json").unlink(),
     "tokenizer"),
    (lambda root: (root / "config.json").write_text('{"model_type":"x"}'), "quantisation"),
    (lambda root: (root / "config.json").write_text(
        '{"model_type":"x","quantization":{"bits":0,"group_size":64}}'), "bits"),
])
def test_missing_or_invalid_identity_fields_fail_closed(tmp_path, mutation, needle):
    root = model_dir(tmp_path / "model")
    mutation(root)
    with pytest.raises(ModelIdentityError, match=needle):
        build_model_identity("org/model", root, "revision")
