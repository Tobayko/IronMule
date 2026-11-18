import ctypes
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

import ironmule_devices
from ironmule import ggml_binary as binary
from ironmule import ggml_engine


def test_the_gate_rekeys_after_a_backend_change(monkeypatch, tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr(ggml_engine, "ggml_devices", lambda: ())
    monkeypatch.setattr(binary, "identity", lambda: {"sha256": {"backend/ROCm": "build1"}})
    first = ggml_engine.gate_key(str(gguf), str(gguf))
    monkeypatch.setattr(binary, "identity", lambda: {"sha256": {"backend/ROCm": "build2"}})
    assert first != ggml_engine.gate_key(str(gguf), str(gguf))


def test_unresolvable_backend_cannot_reuse_an_old_promotion(monkeypatch):
    fake = SimpleNamespace(llama_backend_init=lambda: None, llama_cpp=SimpleNamespace(_lib=None))
    monkeypatch.setattr(ironmule_devices, "bindings", lambda: fake)
    assert binary.identity() != binary.identity()


@pytest.mark.skipif(not sys.platform.startswith("linux") or not shutil.which("cc"), reason="requires ELF and cc")
def test_hash_the_mapped_backend_and_reject_a_replaced_loader_path(tmp_path):
    # Each native symbol is actually mapped in this fixture. Replacing the same pathname must not
    # launder the old in-memory library under the new hash.
    source = tmp_path / "backend.c"
    source.write_text("""
        #include <stddef.h>
        static int reg;
        void llama_backend_init(void) {}
        void ggml_init(void) {}
        size_t ggml_backend_reg_count(void) {return 1;}
        void *ggml_backend_reg_get(size_t i) {return &reg;}
        const char *ggml_backend_reg_name(void *p) {return "ROCm";}
    """)
    library = tmp_path / "ggml.so"
    subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    handle = ctypes.CDLL(str(library))
    identities = binary.loaded_objects(handle)
    expected = ggml_engine.file_sha256(str(library))
    assert len(identities) == 4 and identities["backend/ROCm"] == expected
    hits = binary._digest.cache_info().hits
    assert binary.loaded_objects(handle) == identities and binary._digest.cache_info().hits > hits
    replacement = tmp_path / "replacement"
    replacement.write_bytes(library.read_bytes() + b"next build")
    os.replace(replacement, library)
    with pytest.raises(OSError, match="replaced"):
        binary.loaded_objects(handle)
