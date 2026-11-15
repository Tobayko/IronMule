"""CPU6: MLX's float32 matrix products on an optimised BLAS, on Linux x86-64 CPUs.

The `mlx-cpu` wheel links a reference BLAS (`mlx_cpu.libs/libblas-*.so`, netlib 3.10): one thread, no vector
kernels. On a 4-core AVX-512 Xeon a 2,048-cube float32 product ran at 3.4 GFLOP/s there and at about 300 GFLOP/s on
the OpenBLAS that NumPy's wheel bundles (BACKLOG CPU6, development data). Every linear layer of a dense float32
model (the `dequantize` and `float32` plans) is such a product.

This module builds a small forwarding library (`SHIM_SOURCE`) with the system's C compiler, cached under the user's
cache directory. Preloaded into a worker process (`LD_PRELOAD`), its `cblas_sgemm` wins over the reference BLAS's
and forwards each call to an OpenBLAS found on the machine, NumPy's bundled one first. The target is resolved when
the library loads, from `IRONMULE_CPU_BLAS_LIBRARY` and `IRONMULE_CPU_BLAS_SYMBOL`; if it cannot be resolved, every
call goes to MLX's own reference BLAS (`IRONMULE_CPU_BLAS_FALLBACK`), so a missing or broken target never fails a
product, it is only not faster. `ironmule_cpu_blas_active()` says which one serves. Before a worker is given the
variables, `environment` checks them in a child process: the product must match a float64 reference, and the
shim must report the target active; otherwise the worker runs on the stock BLAS and the record says why.

A different BLAS sums in a different order, so answers can change: an opt-in plan (`--cpu-blas openblas`), reported
in health, never selected for a caller. Only `cblas_sgemm` is forwarded; MLX's other BLAS and LAPACK calls stay on
the libraries its wheel ships.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SHIM_VERSION = 1
CHOICES = ("stock", "openblas")

SHIM_SOURCE = r"""
/* IronMule CPU6: forward cblas_sgemm to an optimised BLAS named at load time; else to the next definition. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef void (*lp64_t)(int, int, int, int, int, int, float, const float *, int, const float *, int, float,
                       float *, int);
typedef void (*ilp64_t)(int, int, int, int64_t, int64_t, int64_t, float, const float *, int64_t, const float *,
                        int64_t, float, float *, int64_t);
static lp64_t target_lp64 = 0, next_sgemm = 0;
static ilp64_t target_ilp64 = 0;

__attribute__((constructor)) static void ironmule_resolve(void) {
  const char *path = getenv("IRONMULE_CPU_BLAS_LIBRARY"), *symbol = getenv("IRONMULE_CPU_BLAS_SYMBOL");
  const char *ilp64 = getenv("IRONMULE_CPU_BLAS_ILP64"), *fallback = getenv("IRONMULE_CPU_BLAS_FALLBACK");
  next_sgemm = (lp64_t)dlsym(RTLD_NEXT, "cblas_sgemm");
  if (!next_sgemm && fallback) { /* MLX's own BLAS is loaded locally, so RTLD_NEXT cannot see it */
    void *own = dlopen(fallback, RTLD_NOW | RTLD_LOCAL);
    if (own) next_sgemm = (lp64_t)dlsym(own, "cblas_sgemm");
  }
  if (!path || !symbol) return;
  void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!handle) return;
  void *found = dlsym(handle, symbol);
  if (!found) return;
  if (ilp64 && strcmp(ilp64, "1") == 0) target_ilp64 = (ilp64_t)found; else target_lp64 = (lp64_t)found;
}

int ironmule_cpu_blas_active(void) { return target_ilp64 ? 64 : target_lp64 ? 32 : 0; }
int ironmule_cpu_blas_version(void) { return SHIM_VERSION_VALUE; }

void cblas_sgemm(int order, int ta, int tb, int m, int n, int k, float alpha, const float *a, int lda,
                 const float *b, int ldb, float beta, float *c, int ldc) {
  if (target_ilp64) { target_ilp64(order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); return; }
  if (target_lp64) { target_lp64(order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); return; }
  if (next_sgemm) next_sgemm(order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  else abort(); /* no BLAS at all in the process: MLX itself could not have loaded */
}
""".replace("SHIM_VERSION_VALUE", str(SHIM_VERSION))

#: (symbol, ILP64) in the order they are tried: NumPy >= 2 (scipy-openblas64), NumPy 1.x (openblas64_), then a
#: plain LP64 OpenBLAS.
SYMBOLS = (("scipy_cblas_sgemm64_", True), ("cblas_sgemm64_", True), ("scipy_cblas_sgemm", False),
           ("cblas_sgemm", False))


def supported() -> str | None:
    """Why this machine cannot use the shim, or None."""
    if sys.platform != "linux":
        return "the CPU BLAS shim is for Linux (macOS's MLX uses Accelerate)"
    if platform.machine() not in ("x86_64", "AMD64"):
        return "the CPU BLAS shim is built for x86-64"
    return None


def candidate_libraries() -> list[str]:
    """OpenBLAS libraries on this machine: NumPy's bundled one, the scipy-openblas wheels', then the system's."""
    found: list[str] = []
    try:
        import numpy

        root = Path(numpy.__file__).resolve().parent
        for pattern in ("../numpy.libs/libscipy_openblas*.so*", "../numpy.libs/libopenblas*.so*",
                        ".libs/libopenblas*.so*"):
            found += sorted(glob.glob(str(root / pattern)))
    except ImportError:
        pass
    for package in ("scipy_openblas64", "scipy_openblas32"):
        try:
            module = __import__(package)
            found += sorted(glob.glob(os.path.join(module.get_lib_dir(), "lib*.so*")))
        except (ImportError, AttributeError):
            continue
    import ctypes.util

    system = ctypes.util.find_library("openblas")
    if system:
        found.append(system)
    unique: list[str] = []
    for path in found:
        path = str(Path(path).resolve()) if os.path.exists(path) else path
        if path not in unique:
            unique.append(path)
    return unique


def find_target() -> dict[str, Any] | None:
    """The first OpenBLAS with a usable single-precision CBLAS product: {library, symbol, ilp64}."""
    for path in candidate_libraries():
        try:
            library = ctypes.CDLL(path)
        except OSError:
            continue
        for symbol, ilp64 in SYMBOLS:
            if hasattr(library, symbol):
                return {"library": path, "symbol": symbol, "ilp64": ilp64}
    return None


def mlx_blas() -> str | None:
    """The reference BLAS the `mlx-cpu` wheel ships (the shim's fallback), or None."""
    try:
        import importlib.util

        found = importlib.util.find_spec("mlx")
        roots = list(found.submodule_search_locations or []) if found else []
    except (ImportError, ValueError):
        return None
    for root in roots:
        hits = sorted(glob.glob(os.path.join(os.path.dirname(root), "mlx_cpu.libs", "libblas*.so*")))
        if hits:
            return hits[0]
    return None


CHECK = r"""
import json, sys
import numpy as np
import mlx.core as mx
from ironmule import cpu_blas
mx.set_default_device(mx.cpu)
rng = np.random.default_rng(0)
worst = 0.0
for m, k, n in ((1, 576, 1536), (8, 1536, 576), (37, 300, 129)):
    a, b = rng.standard_normal((m, k), dtype=np.float32), rng.standard_normal((n, k), dtype=np.float32)
    out = np.array(mx.array(a) @ mx.array(b).T)
    ref = a.astype(np.float64) @ b.astype(np.float64).T
    worst = max(worst, float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-30)))
print(json.dumps({**cpu_blas.active(), "rel_error": worst}))
"""
CHECK_TOLERANCE = 1e-5


def check(variables: dict[str, str], timeout: float = 120) -> dict[str, Any]:
    """Run `CHECK` in a child process with `variables`: what serves `cblas_sgemm` there, and its error."""
    root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, **variables, "PYTHONPATH": os.pathsep.join(
        item for item in (root, os.environ.get("PYTHONPATH", "")) if item)}
    try:
        done = subprocess.run([sys.executable, "-c", CHECK], env=env, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "reason": f"the BLAS check did not run ({type(exc).__name__})"}
    if done.returncode != 0:
        return {"ok": False, "reason": f"the BLAS check failed (exit {done.returncode})"}
    try:
        result = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "reason": "the BLAS check printed no result"}
    ok = bool(result.get("active")) and float(result.get("rel_error", 1.0)) <= CHECK_TOLERANCE
    return {**result, "ok": ok, **({} if ok else {"reason": "the optimised BLAS did not serve, or disagreed with a "
                                                            "float64 reference"})}


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "ironmule" / "cpu_blas"


def build(compiler: str | None = None, root: Path | None = None) -> tuple[Path | None, str | None]:
    """The compiled shim (cached by its source's digest), or None and why."""
    reason = supported()
    if reason:
        return None, reason
    compiler = compiler or os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if not compiler:
        return None, "no C compiler (cc or gcc) to build the CPU BLAS shim"
    digest = hashlib.sha256(SHIM_SOURCE.encode()).hexdigest()[:16]
    folder = (root or _cache_root()) / digest
    target = folder / "libironmule_cpu_blas.so"
    if target.is_file():
        return target, None
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "shim.c"
    source.write_text(SHIM_SOURCE)
    partial = folder / f"libironmule_cpu_blas.{os.getpid()}.so"
    try:
        done = subprocess.run([compiler, "-O2", "-shared", "-fPIC", str(source), "-o", str(partial), "-ldl"],
                              capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"the CPU BLAS shim did not compile ({type(exc).__name__})"
    if done.returncode != 0 or not partial.is_file():
        return None, f"the CPU BLAS shim did not compile: {(done.stderr or '').strip()[-200:]}"
    os.replace(partial, target)  # atomic: concurrent builders end with one complete file
    return target, None


def environment(threads: int | None = None, compiler: str | None = None,
                root: Path | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    """The variables a worker process needs to run on the optimised BLAS, and a record of what was found.

    An empty mapping means the worker runs on the stock BLAS; the record says why. `threads` caps the BLAS's
    threads (several workers sharing one CPU each take a share of its cores)."""
    record: dict[str, Any] = {"requested": "openblas", "shim_version": SHIM_VERSION}
    shim, reason = build(compiler, root)
    if shim is None:
        return {}, {**record, "available": False, "reason": reason}
    target = find_target()
    if target is None:
        return {}, {**record, "available": False, "reason": "no OpenBLAS library found on this machine"}
    fallback = mlx_blas()
    if fallback is None:
        return {}, {**record, "available": False, "reason": "MLX's own CPU BLAS was not found (not the mlx-cpu wheel)"}
    preload = " ".join(item for item in (str(shim), os.environ.get("LD_PRELOAD", "")) if item)
    variables = {"LD_PRELOAD": preload, "IRONMULE_CPU_BLAS_LIBRARY": target["library"],
                 "IRONMULE_CPU_BLAS_SYMBOL": target["symbol"], "IRONMULE_CPU_BLAS_ILP64": "1" if target["ilp64"] else "0",
                 "IRONMULE_CPU_BLAS_FALLBACK": fallback}
    if threads is not None:
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive int")
        variables["OPENBLAS_NUM_THREADS"] = str(threads)
    checked = check(variables)
    if not checked["ok"]:
        return {}, {**record, "available": False, "reason": checked["reason"]}
    return variables, {**record, "available": True, "library": os.path.basename(target["library"]),
                       "symbol": target["symbol"], "threads": threads, "check_rel_error": checked["rel_error"]}


def active() -> dict[str, Any]:
    """In a running process: whether the shim is loaded, and which BLAS serves `cblas_sgemm`."""
    try:
        process = ctypes.CDLL(None)
        probe = process.ironmule_cpu_blas_active
    except (OSError, AttributeError):
        return {"loaded": False, "active": False, "serving": "stock"}
    width = int(probe())
    return {"loaded": True, "active": width > 0, "serving": "openblas" if width else "stock",
            "integer_bits": width or None,
            "library": (os.path.basename(os.environ.get("IRONMULE_CPU_BLAS_LIBRARY", "")) or None) if width else None}


__all__ = ["CHECK_TOLERANCE", "CHOICES", "SHIM_SOURCE", "SHIM_VERSION", "active", "build", "candidate_libraries",
           "check", "environment", "find_target", "mlx_blas", "supported"]
