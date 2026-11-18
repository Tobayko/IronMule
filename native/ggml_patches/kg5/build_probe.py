"""Build the KG5 measurement hook beside an existing Linux KG2 CUDA build, without recompiling llama.cpp.

--source is the vendor/llama.cpp directory of llama-cpp-python 0.3.16.
Use the resulting DSO via LD_PRELOAD in a new test process with the KG2
build/bin directory on LD_LIBRARY_PATH. Both arms should preload it.
"""

import argparse
import hashlib
from pathlib import Path
import subprocess


SYMBOL = "_Z24ggml_cuda_flash_attn_extR25ggml_backend_cuda_contextP11ggml_tensor"
# The probe uses the real internal context and pool layouts from the pinned backend.
PINNED_HEADERS = {
    "ggml/include/ggml.h": "1434a10c6b764e8983a5260ac2ad86c90ec6476594fe1c74ecb6364a7ec0b274",
    "ggml/src/ggml-cuda/common.cuh": "273d20bfdcb3db8df6c5a5a442e0635b34de9f12a77cd924f5775c717cdd029f",
    "ggml/include/ggml-cuda.h": "8f22e8e43e460db426bb94e14e10dc91e64789e8d216c5ae62a0d02a2eb07d7f",
    "ggml/src/ggml-cuda/fattn.cuh": "69fa53aa9573a6734a931f80f4ca6946cd847dce87cccdc0c7aa7b10a8d9066b",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--lib-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    for name, digest in PINNED_HEADERS.items():
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != digest:
            parser.error(f"unexpected upstream header {name}; use the pinned llama.cpp 4227c9b source")
    lib = args.lib_dir / "libggml-cuda.so"
    relocs = subprocess.check_output(["readelf", "--wide", "--relocs", lib], text=True)
    if not any(SYMBOL in line and ("JUMP_SLOT" in line or "GLOB_DAT" in line) for line in relocs.splitlines()):
        parser.error(f"{lib} does not call the dynamic flash-attention symbol; attach a KG5-enabled build")
    ggml = args.source / "ggml"
    source = Path(__file__).with_name("probe.cu")
    subprocess.run([
        args.nvcc, "-std=c++17", "-O3", "-arch=sm_75", "--cudart=shared", "-shared", "-Xcompiler=-fPIC",
        f"-I{ggml / 'src' / 'ggml-cuda'}", f"-I{ggml / 'src'}", f"-I{ggml / 'include'}",
        str(source), "-Xlinker=-z", "-Xlinker=defs", f"-L{args.lib_dir.resolve()}", "-lggml-cuda", "-lggml-base", "-ldl",
        "-o", str(args.output),
    ], check=True)


if __name__ == "__main__":
    main()
