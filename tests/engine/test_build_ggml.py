"""tools/build_ggml.py: every patch set's files exist and its diffs are normal diffs against the named files."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location("build_ggml", ROOT / "tools" / "build_ggml.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_sets_are_complete_and_well_formed():
    tool = load()
    assert set(tool.PATCH_SETS) == {"fa1", "row1"}
    for name, (diffs, headers) in tool.PATCH_SETS.items():
        for target, diff in diffs:
            text = (tool.PATCHES / name / diff).read_text()
            assert target.startswith("ggml/src/ggml-cpu/")
            assert re.match(r"^\d+(,\d+)?[acd]\d+(,\d+)?$", text.splitlines()[0])  # a normal diff hunk header
        for header, target in headers:
            assert (tool.PATCHES / name / header).is_file() and target.endswith(header)


def test_fa1_is_taken_only_where_it_is_exact_in_kind():
    text = (ROOT / "native" / "ggml_patches" / "fa1" / "fa1_kern.h").read_text()
    for guard in ("max_bias != 0.0f) return false", "sinks", "GGML_TYPE_F16", "FA1"):
        assert guard in text
