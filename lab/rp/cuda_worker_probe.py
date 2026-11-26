"""Diagnose CUDA initialization using a fixed one-element SAXPY. Only the full worker runs candidate code."""
from __future__ import annotations

import argparse
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve()
UID = 60000


def worker_module():
    spec = importlib.util.spec_from_file_location("restricted_worker", HERE.with_name("restricted_worker.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add():
    if os.getuid() != UID:
        raise RuntimeError("expected the dedicated non-root worker")
    driver = C.CDLL("libcuda.so.1")

    def call(lib, name, *args):
        status = getattr(lib, name)(*args)
        if status:
            detail = C.c_char_p()
            if lib is driver:
                driver.cuGetErrorName(status, C.byref(detail))
            raise RuntimeError(f"{name}: {status} {detail.value or b''}")

    call(driver, "cuInit", 0)
    device, count = C.c_int(), C.c_int()
    call(driver, "cuDeviceGetCount", C.byref(count))
    call(driver, "cuDeviceGet", C.byref(device), 0)
    ctx = C.c_void_p()
    call(driver, "cuCtxCreate_v2", C.byref(ctx), 0, device)
    # Host reads only the one-element result. The multiplication and addition execute in cuBLAS.
    blas = C.CDLL("libcublas.so.12")
    handle = C.c_void_p()
    call(blas, "cublasCreate_v2", C.byref(handle))
    x, y = C.c_uint64(), C.c_uint64()
    for ptr, host in ((x, C.c_float(2)), (y, C.c_float(3))):
        call(driver, "cuMemAlloc_v2", C.byref(ptr), C.c_size_t(4))
        call(driver, "cuMemcpyHtoD_v2", ptr, C.byref(host), C.c_size_t(4))
    alpha, result = C.c_float(2), C.c_float()
    call(blas, "cublasSaxpy_v2", handle, 1, C.byref(alpha), x, 1, y, 1)
    call(driver, "cuMemcpyDtoH_v2", C.byref(result), y, C.c_size_t(4))
    if result.value != 7.0:
        raise RuntimeError("incorrect SAXPY")
    call(blas, "cublasDestroy_v2", handle)
    call(driver, "cuCtxDestroy_v2", ctx)
    print(json.dumps({"uid": os.getuid(), "cuda_devices": count.value, "saxpy": result.value}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("uid", "seccomp", "full"), required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    worker = worker_module()
    worker.immutable(HERE)
    worker.immutable(Path(worker.__file__).resolve())
    output = args.output.resolve(strict=True)
    env = {"PATH": "/usr/local/cuda/bin:/usr/bin:/bin", "LANG": "C"}
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib"
    cmd = [sys.executable, "-I", "-S", str(HERE), "_sum"]
    if args.mode == "full":
        result = worker.run(cmd, env=env, cwd=output, output=output, readonly=[HERE], timeout=45)
    else:
        env.update(HOME=str(output), TMPDIR=str(output), CUDA_CACHE_PATH=str(output / ".cuda"))
        if args.mode == "uid":
            result = subprocess.run(cmd, user=UID, group=UID, extra_groups=[], env=env, cwd=output, timeout=45,
                                    capture_output=True, text=True, check=False)
        else:
            cmd[-1] = "_seccomp_sum"
            result = subprocess.run(cmd, env=env, cwd=output.parent, timeout=45,
                                    capture_output=True, text=True, check=False)
    print(json.dumps({"mode": args.mode, "rc": result.returncode,
                      "diagnostic": (result.stdout + (result.stderr or ""))[-1500:]}), flush=True)
    return result.returncode


if __name__ == "__main__":
    if sys.argv[1:] == ["_seccomp_sum"]:
        worker_module().restrict(UID, UID, 45)
        os.chdir(os.environ["HOME"])
        add()
    elif sys.argv[1:] == ["_sum"]:
        add()
    else:
        raise SystemExit(main())
