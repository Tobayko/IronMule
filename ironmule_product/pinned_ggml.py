"""Qualify one --gpus worker after its launcher has selected the CUDA device."""
import json
import os
import sys
from argparse import Namespace
from contextlib import redirect_stdout

from ironmule.ggml_profiles import FLAGS, profile, select

from .state import ProductStore
from .types import ModelSpec


def qualify(options, spec, store, pinned, overrides):
    """Run in the pinned calibration child. Both gate and width children inherit the exact flags."""
    from ironmule import ggml_binary
    from ironmule import ggml_engine as engine
    from ironmule_product.cli import _local_ggml_options
    from ironmule_product.errors import InvalidRequest

    layers = engine.gpu_layers()
    gguf = str(options.gguf)
    model_sha = engine.file_sha256(gguf)
    native = ggml_binary.identity()
    name, flags = select(layers, engine.ggml_devices(), model_sha, native, overrides)
    os.environ.update(flags)
    if layers:
        os.environ["IRONMULE_GPU_PROFILE_GATE"] = "1"
    recovered = None
    try:
        result = _local_ggml_options(options, spec, store, pinned=pinned)
    except InvalidRequest as exc:
        if name != "ps1-ps1a-pk6" or "did not pass the quality gate" not in str(exc):
            raise
        recovered = str(exc)
        name, flags = "conservative", profile("conservative")
        os.environ.update(flags)
        result = _local_ggml_options(options, spec, store, pinned=pinned)
    # Re-establish the actual worker kernel library after the calibration helper restored its environment.
    os.environ["IRONMULE_KG_LIB"] = result["ggml_kernel_library"]
    receipt = {"profile": name, "flags": flags, "gguf_sha256": model_sha, "machine": engine.machine_key(),
               "native_sha256": native.get("sha256"), "kernel_sha256": engine._kernel_digest(), "layers": layers, "fallback_reason": recovered}
    result["ggml_gate"]["kernel_profile"] = receipt
    result["ggml_worker_env"] = {**result.get("ggml_worker_env", {}), **flags}
    return result


def verify(receipt, gguf):
    """The serving child validates its inherited bundle before creating the first model context."""
    from ironmule import ggml_binary
    from ironmule import ggml_engine as engine

    actual_flags = {k: os.environ.get(k) for k in FLAGS}
    native = ggml_binary.identity().get("sha256")
    kernel = engine._kernel_digest()
    if (receipt["flags"] != actual_flags or receipt["layers"] != engine.gpu_layers()
            or receipt["gguf_sha256"] != engine.file_sha256(gguf) or receipt["machine"] != engine.machine_key()
            or not native or receipt["native_sha256"] != native or kernel is None or receipt["kernel_sha256"] != kernel):
        raise ValueError("GGUF kernel profile differs from its local qualification; requalify this worker")


def main():
    payload = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        if bus := payload.get("rocm_bus"):
            from .placement_devices import verify_rocm_worker
            verify_rocm_worker(bus)
        options = qualify(Namespace(**payload["options"]), ModelSpec.from_dict(payload["spec"]),
                          ProductStore(payload["root"]), payload.get("pinned", True), payload["kernel_overrides"])
    print(json.dumps(options, allow_nan=False))


if __name__ == "__main__":
    main()
