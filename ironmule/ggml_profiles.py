"""Complete PS1/PK6 environment profiles for GGUF qualification and serving."""

from __future__ import annotations

import os

FLAGS = ("IRONMULE_PS1", "IRONMULE_PK6", "IRONMULE_PS1_WARM", "IRONMULE_PS1_CONCAT",
         "IRONMULE_GGML_DH2", "IRONMULE_SV5")
QWEN32 = "efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689"


def profile(name):
    if name not in ("cpu", "ps1-ps1a-pk6", "conservative"):
        raise ValueError(f"unknown GGUF kernel profile: {name}")
    values = dict.fromkeys(FLAGS, "0")
    if name != "conservative":
        values.update(IRONMULE_PS1="1", IRONMULE_PK6="1")
    return values


def select(layers, devices, model_sha256, native, overrides):
    retained = (layers >= 65 and len(devices) == 1 and devices[0].get("provider") == "cuda"
                and devices[0].get("description") == "NVIDIA A100-SXM4-80GB" and model_sha256 == QWEN32
                and bool(native.get("sha256", {}).get("backend/CUDA")))
    name = "cpu" if layers == 0 else "ps1-ps1a-pk6" if retained else "conservative"
    for key, value in overrides.items():
        allowed = ("0", "1", "2") if key == "IRONMULE_PS1_WARM" else ("0", "1")
        if key not in FLAGS or value not in allowed:
            raise ValueError(f"invalid explicit kernel flag: {key}")
    return ("explicit" if overrides else name), {**profile(name), **overrides}


def cublas_override():
    """The native default is zero. Unrecognized explicit values also require qualification."""
    value = os.environ.get("IRONMULE_CUBLAS_MIN_ROWS", "")
    return value if value not in ("", "0") else None


def cache_key():
    """Include the stock-reference rule as well as each candidate's effective flags."""
    key = ["ps1-pk6-v1", *(os.environ.get(key) for key in (*FLAGS, "IRONMULE_GPU_PROFILE_GATE"))]
    if os.environ.get("IRONMULE_KG_LIB") == "" and os.environ.get("IRONMULE_GGML_PACK") == "1":
        key.append("explicit-packed-stock-v1")
    if cublas_override() is not None:
        key.extend(["explicit-cublas-quality-v1", cublas_override()])
    return key
