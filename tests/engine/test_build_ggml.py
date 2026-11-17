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
    assert set(tool.PATCH_SETS) == {"fa1", "row1", "gr1", "kg1", "kg5", "gr2"}
    for name, (diffs, headers) in tool.PATCH_SETS.items():
        for target, diff in diffs:
            text = (tool.PATCHES / name / diff).read_text()
            assert target.startswith(("ggml/src/ggml-cpu/", "ggml/src/ggml-cuda/"))
            assert re.match(r"^\d+(,\d+)?[acd]\d+(,\d+)?$", text.splitlines()[0])  # a normal diff hunk header
        for header, target in headers:
            assert (tool.PATCHES / name / header).is_file() and target.endswith(header)


def test_fa1_is_taken_only_where_it_is_exact_in_kind():
    text = (ROOT / "native" / "ggml_patches" / "fa1" / "fa1_kern.h").read_text()
    for guard in ("max_bias != 0.0f) return false", "sinks", "GGML_TYPE_F16", "FA1"):
        assert guard in text


def test_kg5_hands_only_one_token_f16_decode_attention_to_the_library():
    text = (ROOT / "native" / "ggml_patches" / "kg5" / "fattn.cu.diff").read_text()
    for guard in ("Q->ne[1] != 1", "GGML_TYPE_F16", "max_bias != 0.0f", "dst->src[4] != nullptr", "ironmule_kg_flash_attn"):
        assert guard in text


def test_gr2_keeps_graphs_for_several_tokens_only_when_asked():
    text = (ROOT / "native" / "ggml_patches" / "gr2" / "ggml-cuda.cu.diff").read_text()
    assert '"IRONMULE_CUDA_GRAPHS"), "2"' in text and "ironmule_graphs_batched" in text
