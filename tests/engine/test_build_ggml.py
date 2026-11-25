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
    assert set(tool.PATCH_SETS) == {"fa1", "row1", "gr1", "kg1", "kg5", "pk2", "am1", "gr2", "pk3", "gr3", "fu1", "pk4", "pk5", "tk1", "em1", "gr4", "qk1", "pd1", "tl1", "ps1", "tl2", "tl2c", "ps1a", "pk6"}
    for name, (diffs, headers) in tool.PATCH_SETS.items():
        for target, diff in diffs:
            text = (tool.PATCHES / name / diff).read_text()
            assert target.startswith(("ggml/src/ggml-cpu/", "ggml/src/ggml-cuda/", "src/llama-context.cpp", "src/unicode.cpp",
                                      "src/llama-batch.h", "src/llama-kv-cache-unified.", "src/llama-graph.",
                                      "ggml/src/ggml-alloc.c"))
            assert re.match(r"^\d+(,\d+)?[acd]\d+(,\d+)?$", text.splitlines()[0])  # a normal diff hunk header
        for header, target in headers:
            assert (tool.PATCHES / name / header).is_file() and target.endswith(header)


def test_fa1_is_taken_only_where_it_is_exact_in_kind():
    text = (ROOT / "native" / "ggml_patches" / "fa1" / "fa1_kern.h").read_text()
    for guard in ("max_bias != 0.0f) return false", "sinks", "GGML_TYPE_F16", "FA1"):
        assert guard in text


def test_auto_builds_the_visible_cuda_architectures_and_exports_paths(tmp_path, monkeypatch):
    import shlex
    from ironmule import kernel_gen

    tool = load()
    monkeypatch.setattr(kernel_gen, "nvcc_path", lambda: "/cuda/bin/nvcc")
    monkeypatch.setattr(kernel_gen, "_visible_cuda_devices", lambda: [{"cc": "100"}, {"cc": "80"}, {"cc": "80"}])
    source = tmp_path / "source with spaces"
    monkeypatch.setattr(tool, "fetch", lambda dest: source)
    monkeypatch.setattr(tool, "patch", lambda *_: [])
    built = []
    def build(tree, cuda, jobs, rocm):
        built.append((cuda, rocm))
        return tree / "build/bin"
    monkeypatch.setattr(tool, "build", build)
    exports = tmp_path / "ggml.env"
    assert tool.main(["--auto", "--env-file", str(exports)]) == 0
    assert built == [("80,100", None)]
    assert shlex.split(exports.read_text().splitlines()[0]) == ["export", f"IRONMULE_GGML_BINDINGS={source}"]


def test_new_cuda_card_selects_compiler_that_can_target_it(tmp_path, monkeypatch):
    from ironmule import kernel_gen

    old, new = tmp_path / "old", tmp_path / "new"
    for directory, codes in ((old, "sm_75 sm_80"), (new, "sm_80 sm_120")):
        compiler = directory / "bin/nvcc"
        compiler.parent.mkdir(parents=True)
        compiler.write_text(f"#!/bin/sh\necho '{codes}'\n")
        compiler.chmod(0o700)
    monkeypatch.setenv("PATH", str(old / "bin"))
    monkeypatch.delenv("CUDACXX", raising=False)
    monkeypatch.delenv("CUDAToolkit_ROOT", raising=False)
    monkeypatch.setenv("CUDA_HOME", str(new))
    monkeypatch.setattr(kernel_gen.glob, "glob", lambda _: [])
    assert kernel_gen.cuda_compiler_for(["80", "120"]) == str(new / "bin/nvcc")
    assert kernel_gen.cuda_compiler_for(["75", "120"]) is None


def test_auto_falls_back_to_working_cpu_backend_if_accelerator_build_fails(tmp_path, monkeypatch):
    from ironmule import kernel_gen

    tool = load()
    monkeypatch.setattr(kernel_gen, "nvcc_path", lambda: "nvcc")
    monkeypatch.setattr(kernel_gen, "_visible_cuda_devices", lambda: [{"cc": "120"}])
    monkeypatch.setattr(tool, "fetch", lambda dest: dest)
    monkeypatch.setattr(tool, "patch", lambda *_: [])
    attempted = []
    def build(tree, cuda, jobs, rocm):
        attempted.append(cuda)
        if cuda:
            raise SystemExit("native toolchain rejected this device")
        return tree / "bin"
    monkeypatch.setattr(tool, "build", build)
    exported = tmp_path / "native.env"
    assert tool.main(["--auto", "--dest", str(tmp_path), "--env-file", str(exported)]) == 0
    assert attempted == ["120", None]
    assert "/cpu/" in exported.read_text()
