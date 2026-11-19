"""Content identities of the native binaries participating in this process's GGML registry."""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path


class _Object(C.Structure):
    _fields_ = [("path", C.c_char_p), ("base", C.c_void_p), ("symbol", C.c_char_p), ("address", C.c_void_p)]


def _stat_id(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


@lru_cache(maxsize=64)
def _digest(path, stat_id):
    # A replacement between discovery and hashing must never identify an old mapping as the new build.
    with open(path, "rb") as handle:
        if _stat_id(os.fstat(handle.fileno())) != stat_id:
            raise OSError("GGML binary changed before hashing")
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        if _stat_id(os.fstat(handle.fileno())) != stat_id:
            raise OSError("GGML binary changed during hashing")
    return digest.hexdigest()


def _objects(lib):
    from ironmule_devices import _function

    p = C.c_void_p
    count = _function(lib, "ggml_backend_reg_count", C.c_size_t)
    entry = _function(lib, "ggml_backend_reg_get", p, C.c_size_t)
    name = _function(lib, "ggml_backend_reg_name", C.c_char_p, p)
    if not count():
        raise OSError("GGML has no loaded backend registry")
    objects = [(symbol, C.cast(getattr(lib, symbol), p).value)
               for symbol in ("llama_backend_init", "ggml_init", "ggml_backend_reg_count")]
    for i in range(count()):
        pointer = entry(i)
        objects.append((f"backend/{name(pointer).decode()}", pointer))
    return objects


def _mappings():
    # dladdr reports the loader's pathname. Linux can retain the old mapped inode after that pathname
    # was replaced. Match the file backing each image's base address before trusting its contents.
    if not sys.platform.startswith("linux"):
        return None
    result = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        bounds, _, offset, device, inode, *_ = line.split(maxsplit=5)
        if int(offset, 16) == 0 and int(inode):
            major, minor = device.split(":")
            result[int(bounds.split("-", 1)[0], 16)] = (os.makedev(int(major, 16), int(minor, 16)), int(inode))
    return result


def _mapped_images(lib, *, only=None):
    from ironmule_devices import _function

    dladdr = _function(C.CDLL(None), "dladdr", C.c_int, C.c_void_p, C.POINTER(_Object))
    maps = _mappings()
    for tag, pointer in _objects(lib):
        if only is not None and tag != only:
            continue
        obj = _Object()
        if not dladdr(pointer, C.byref(obj)) or not obj.path:
            raise OSError(f"unresolved native image for {tag}")
        path = Path(os.fsdecode(obj.path)).resolve(strict=True)
        st = path.stat()
        if maps is not None and maps.get(obj.base) != (st.st_dev, st.st_ino):
            raise OSError(f"native image for {tag} replaced after loading")
        yield tag, str(path), _stat_id(st)


def loaded_objects(lib):
    return {tag: _digest(path, stat_id) for tag, path, stat_id in _mapped_images(lib)}


@lru_cache(maxsize=16)
def _kg1_literals(path, stat_id):
    # Both NUL-terminated strings are part of the historical KG1 getenv/dlsym ABI, even in stripped
    # builds. Scan the mapped CUDA image, which can be a plugin or a monolithic build of llama.cpp.
    required = {b"IRONMULE_KG_LIB\0", b"ironmule_kg_mul_mat\0"}
    overlap = max(map(len, required)) - 1
    tail = b""
    with open(path, "rb") as handle:
        if _stat_id(os.fstat(handle.fileno())) != stat_id:
            raise OSError("CUDA backend changed before KG1 inspection")
        while required and (block := handle.read(2**20)):
            data = tail + block
            required = {literal for literal in required if literal not in data}
            tail = data[-overlap:]
        if _stat_id(os.fstat(handle.fileno())) != stat_id:
            raise OSError("CUDA backend changed during KG1 inspection")
    return not required


def cuda_kg1_hook(lib=None) -> bool | None:
    """Detect the KG1 plugin ABI on the loaded GGML CUDA image. An unresolved image is inconclusive."""
    from ironmule_devices import bindings

    try:
        if lib is None:
            binding = bindings()
            binding.llama_backend_init()
            lib = binding.llama_cpp._lib
        images = list(_mapped_images(lib, only="backend/CUDA"))
        if len(images) == 1:
            _, path, stat_id = images[0]
            return _kg1_literals(path, stat_id)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError):
        pass
    return None


# Missing bindings cannot serve in-process. Give simulations/stubs a process-local scope.
_PROCESS_SCOPE = os.urandom(16).hex()


def identity():
    from ironmule_devices import bindings

    try:
        L = bindings()
    except (ImportError, OSError, RuntimeError):
        return {"unavailable_in_process": _PROCESS_SCOPE}
    try:
        L.llama_backend_init()
        return {"sha256": loaded_objects(L.llama_cpp._lib)}
    except (OSError, ValueError, TypeError, AttributeError, UnicodeError):
        # Incomplete identity must not restore a prior measured choice (even within this process).
        return {"unresolved_measurement": os.urandom(16).hex()}
