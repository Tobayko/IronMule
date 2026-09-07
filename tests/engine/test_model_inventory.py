import json
from pathlib import Path

import ironmule_inventory

from ironmule_inventory import discover_models


def test_inventory_bounds_metadata_reads(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    repository, _ = _snapshot(hub)
    monkeypatch.setattr(ironmule_inventory, "_MAX_METADATA_BYTES", 64)
    (repository / "blobs" / "config").write_text(" " * 65, encoding="utf-8")
    row = discover_models([hub])[0]
    assert row["status"] == "error"
    assert any("size limit" in reason for reason in row["reasons"])


def test_inventory_reports_cyclic_snapshot_link(tmp_path):
    hub = tmp_path / "hub"
    snapshots = hub / "models--org--gemma-test" / "snapshots"
    snapshots.mkdir(parents=True)
    (snapshots / "cycle").symlink_to("cycle")
    row = discover_models([hub])[0]
    assert row["status"] == "error"
    assert row["revision"] == "cycle"


def _snapshot(hub: Path, *, model="google/gemma-3-1b-it", revision="abc123", shard=True):
    repository = hub / ("models--" + model.replace("/", "--"))
    snapshot = repository / "snapshots" / revision
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text(revision, encoding="ascii")
    config_blob = blobs / "config"
    config_blob.write_text(json.dumps({"model_type": "gemma3"}), encoding="utf-8")
    (snapshot / "config.json").symlink_to(Path("../../blobs/config"))
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    if shard:
        weight_blob = blobs / "weights"
        weight_blob.write_bytes(b"weight")
        (snapshot / "model.safetensors").symlink_to(Path("../../blobs/weights"))
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer": "model.safetensors"}}), encoding="utf-8"
        )
    return repository, snapshot


def test_inventory_reports_complete_snapshot_and_resolves_refs(tmp_path):
    hub = tmp_path / "hub"
    _, snapshot = _snapshot(hub)

    rows = discover_models([hub])

    assert len(rows) == 1
    assert rows[0] == {
        "model_id": "google/gemma-3-1b-it",
        "revision": "abc123",
        "snapshot_path": str(snapshot.resolve()),
        "cache_root": str(hub.resolve()),
        "family": "gemma3",
        "weight_bytes": 6,
        "weight_files": 1,
        "status": "available",
        "reasons": [],
        "warnings": [],
        "loader": "hf",
        "weight_selection": "hf_index",
    }
    json.dumps(rows)


def test_inventory_marks_missing_blob_and_unsafe_index_incomplete_or_error(tmp_path):
    hub = tmp_path / "hub"
    _, snapshot = _snapshot(hub)
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").symlink_to(Path("../../blobs/missing"))

    row = discover_models([hub])[0]

    assert row["status"] == "incomplete"
    assert row["weight_files"] == 0
    assert any("broken symlink" in reason for reason in row["reasons"])

    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "../escape.safetensors"}}), encoding="utf-8"
    )
    row = discover_models([hub])[0]
    assert row["status"] == "error"
    assert any("unsafe shard path" in reason for reason in row["reasons"])


def test_inventory_deduplicates_actual_snapshot_paths_and_requires_explicit_roots(
    tmp_path, monkeypatch
):
    hub = tmp_path / "hub"
    _, snapshot = _snapshot(hub, model="org/model")
    alias = tmp_path / "alias"
    alias.symlink_to(hub, target_is_directory=True)

    assert len(discover_models([hub, alias])) == 1
    monkeypatch.setenv("HOME", str(tmp_path))
    assert discover_models() == []
    assert len(discover_models([tmp_path / ".friday-data" / "models" / "hub"])) == 0
    assert snapshot.exists()


def test_inventory_family_filter_and_missing_root(tmp_path):
    hub = tmp_path / "hub"
    _snapshot(hub)

    assert discover_models([hub], family="GEMMA3")[0]["family"] == "gemma3"
    assert discover_models([hub], family="llama") == []
    assert discover_models([tmp_path / "does-not-exist"]) == []


def test_mlx_loader_uses_direct_weights_and_warns_about_upstream_index_mismatch(tmp_path):
    hub = tmp_path / "hub"
    _, snapshot = _snapshot(hub)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00005.safetensors"}}),
        encoding="utf-8",
    )

    row = discover_models([hub], loader="mlx_lm")[0]

    assert row["status"] == "available"
    assert row["weight_bytes"] == 6
    assert row["weight_files"] == 1
    assert row["weight_selection"] == "mlx_lm_direct_glob"
    assert row["warnings"]
    assert not row["reasons"]


def test_tokenizer_config_or_added_tokens_alone_is_incomplete(tmp_path):
    hub = tmp_path / "hub"
    _, snapshot = _snapshot(hub)
    (snapshot / "tokenizer.json").unlink()
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "added_tokens.json").write_text("[]", encoding="utf-8")

    row = discover_models([hub])[0]

    assert row["status"] == "incomplete"
    assert any("complete tokenizer" in reason for reason in row["reasons"])


def test_indexed_shard_through_directory_symlink_is_rejected(tmp_path):
    hub = tmp_path / "hub"
    _, snapshot = _snapshot(hub)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "weight.safetensors").write_bytes(b"outside")
    (snapshot / "sub").symlink_to(outside, target_is_directory=True)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "sub/weight.safetensors"}}),
        encoding="utf-8",
    )

    row = discover_models([hub])[0]

    assert row["status"] == "error"
    assert any("escapes model cache entry" in reason for reason in row["reasons"])
