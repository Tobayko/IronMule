"""Offline contract for project-local model resolution."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from _bench import LocalModelError, resolve_local_model_snapshot  # noqa: E402


class LocalModelSnapshotTests(unittest.TestCase):
    MODEL_ID = "example/model-4bit"
    REVISION = "a" * 40

    def make_cache(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        hub = Path(temporary.name) / "hub"
        repository = hub / "models--example--model-4bit"
        snapshot = repository / "snapshots" / self.REVISION
        (repository / "refs").mkdir(parents=True)
        snapshot.mkdir(parents=True)
        (repository / "refs" / "main").write_text(self.REVISION, encoding="ascii")
        (snapshot / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
        (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"weight-bytes")
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer": "model.safetensors"}}),
            encoding="utf-8",
        )
        return temporary, hub, snapshot

    def test_resolves_execution_complete_snapshot_without_repository_docs(self) -> None:
        temporary, hub, snapshot = self.make_cache()
        self.addCleanup(temporary.cleanup)

        resolved = resolve_local_model_snapshot(self.MODEL_ID, hub_root=hub)

        self.assertEqual(resolved.path, snapshot.resolve())
        self.assertEqual(resolved.revision, self.REVISION)
        self.assertEqual(resolved.weight_files, ("model.safetensors",))
        self.assertEqual(resolved.weight_bytes, len(b"weight-bytes"))
        self.assertEqual(
            resolved.report_identity()["model_source"],
            "validated_project_local_snapshot",
        )

    def test_rejects_identifier_path_traversal(self) -> None:
        with self.assertRaises(LocalModelError):
            resolve_local_model_snapshot("../model", hub_root=Path("unused"))

    def test_uses_mlx_monolith_even_when_upstream_index_names_absent_shards(self) -> None:
        temporary, hub, snapshot = self.make_cache()
        self.addCleanup(temporary.cleanup)
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer": "missing.safetensors"}}),
            encoding="utf-8",
        )

        resolved = resolve_local_model_snapshot(self.MODEL_ID, hub_root=hub)

        self.assertEqual(resolved.weight_files, ("model.safetensors",))

    def test_rejects_snapshot_without_mlx_weight_file(self) -> None:
        temporary, hub, snapshot = self.make_cache()
        self.addCleanup(temporary.cleanup)
        (snapshot / "model.safetensors").unlink()

        with self.assertRaises(LocalModelError):
            resolve_local_model_snapshot(self.MODEL_ID, hub_root=hub)

    def test_rejects_invalid_revision(self) -> None:
        temporary, hub, _ = self.make_cache()
        self.addCleanup(temporary.cleanup)
        reference = hub / "models--example--model-4bit" / "refs" / "main"
        reference.write_text("main", encoding="ascii")

        with self.assertRaises(LocalModelError):
            resolve_local_model_snapshot(self.MODEL_ID, hub_root=hub)

    def test_every_model_tool_loads_only_the_validated_local_path(self) -> None:
        for script in (
            "codegen_loop.py",
            "measure_fusion_layer.py",
            "measure_roofline.py",
            "model_loop.py",
        ):
            with self.subTest(script=script):
                source = (ROOT / "tools" / script).read_text(encoding="utf-8")
                self.assertIn("resolve_local_model_snapshot", source)
                self.assertIn("load(str(snapshot.path))", source)
                self.assertNotIn("load(model_id)", source)
                self.assertNotIn("load(MODEL_ID)", source)
                self.assertNotIn("snapshot_download", source)


if __name__ == "__main__":
    unittest.main()
