import json
import subprocess
from types import SimpleNamespace

import pytest

from ironmule import factual_gate, ggml_engine, numeric_choice
from ironmule_product.cli import _qualify_ggml_kernels
from ironmule_product.errors import InvalidRequest


@pytest.mark.parametrize("mode", ["stock", "generated", "kg5"])
def test_uninformative_reference_skips_all_competing_shapes_but_requires_stock_identity(monkeypatch, tmp_path, mode):
    # The two measured Qwen3-0.6B CPU reference passes both answered these two cases correctly.
    # Replay their 16 NLL chunks and complete factual score vectors. No candidate returns a score.
    facts = {case.key: case.key in ("0:facts-3", "0:facts-5") for case in factual_gate.cases()}
    nll = [3.871598116233536, 3.688323066480487, 3.4716876100819265, 3.442983187758426,
           3.588649523051233, 3.4765137539285367, 3.4383472547297136, 3.9123646648437425,
           3.4395370835058605, 3.5346630853626637, 3.88431219122446, 3.4839486569057425,
           3.3838671373375324, 3.623345906051985, 3.4776923802583117, 3.563973235748329]
    monkeypatch.delenv("IRONMULE_KG_LIB", raising=False)
    monkeypatch.delenv("IRONMULE_KG5_SHAPE", raising=False)
    monkeypatch.setenv("IRONMULE_KG5", "1" if mode == "kg5" else "0")
    if mode == "generated":
        library = tmp_path / "kg.so"
        library.write_bytes(b"fixture")
        monkeypatch.setenv("IRONMULE_KG_LIB", str(library))
    references = []

    def run(args, **kwargs):
        if args[2] == ggml_engine.GATE_IDS:
            output = {"ids": [1] * (numeric_choice.CHUNKS * numeric_choice.CHUNK_TOKENS + 1),
                      "bos": None, "facts_sha256": "f" * 64}
        else:
            assert args[2] == ggml_engine.GATE_CANDIDATE
            request = json.loads(kwargs["input"])
            assert request["reference"] and not request["shared"] and request["settings"] == ggml_engine.DEFAULT_SETTINGS
            assert not kwargs["env"]["IRONMULE_KG_LIB"] and kwargs["env"]["IRONMULE_KG5"] == "0"
            references.append(True)
            output = {"nll": nll, "facts": facts, "repeat": facts}
        return SimpleNamespace(stdout=json.dumps(output), stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(ggml_engine, "gpu_layers", lambda: 999)
    monkeypatch.setattr(ggml_engine, "machine_key", lambda: "fixture")
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"model")
    if mode == "kg5":
        with pytest.raises(InvalidRequest, match="skipped_uninformative_reference"):
            _qualify_ggml_kernels(str(model), str(model), tmp_path / "cache")
        assert len(references) == 1
        return
    result = _qualify_ggml_kernels(str(model), str(model), tmp_path / "cache")
    assert len(references) == (2 if mode == "generated" else 1)
    if mode == "generated":
        generated, stock = result["kernel_fallback"]["attempts"]
        assert not generated["passes"] and generated["kernel_mode"] == "generated"
        assert generated["diagnostic"]["candidates"] == []
        assert stock["passes"] and stock["kernel_mode"] == "stock"
    assert result["ran"] and result["passes"] and result["settings"] == ggml_engine.DEFAULT_SETTINGS
    assert result["chunks"] == numeric_choice.CHUNKS and result["tried"][0]["basis"] == "identical_reference"
    assert result["tuning"] == [] and result["diagnostic"]["candidates"] == []
    assert result["diagnostic"]["code"] == "skipped_uninformative_reference"
    for category, row in result["diagnostic"]["reference_factual"].items():
        expected = 2 if category == "facts" else 0
        assert row == {"total": 6, "reference_correct": expected, "repeat_correct": expected}
    for role in ("shared", "streams"):
        assert result[role]["status"] == "skipped_uninformative_reference"
        assert result[role]["passes"] is False and "interval" not in result[role]
