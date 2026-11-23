"""Build the llama.cpp library IronMule's ggml engine serves with (`ironmule serve --engine ggml`), with IronMule's kernels.

Downloads the source of llama-cpp-python 0.3.16 (its ctypes bindings and the llama.cpp it vendors, 4227c9b), applies
the patches in `native/ggml_patches/`, builds the shared libraries and `llama-server`, and prints the two environment
variables the engine reads. Nothing is installed system-wide.

  fa1   AVX-512 flash attention (`fa1_kern.h`): grouped-query heads together, blocked online softmax, f32
        accumulation. Taken on CPUs with AVX-512 for f16 KV caches; `FA1=0` at run time turns it off. KV1/FA1 in
        docs/LAB.md: 1.76x a decode step at 16 sequences of 1,024 tokens on a 4-vCPU Xeon, perplexity ratio 1.00001
        against llama.cpp's default.
  row1  AMX int8 kernels for Q8_0 weights (`row1_kern.h`) on Sapphire Rapids and later; off unless `ROW1_GGML=B16`
        (or A, B) is set at run time. ROW1 in docs/LAB.md.
  gr1   CUDA builds only: CUDA graphs on GPUs older than Ampere (a T4) when IRONMULE_CUDA_GRAPHS=1 is set at run time;
        llama.cpp turns them off there, so a one-token step launches every kernel on its own. SG1 in docs/LAB.md.
  kg1   CUDA builds only: a hook that hands a matmul to a kernel library generated and tuned for this GPU
        (IRONMULE_KG_LIB, `ironmule/kernel_gen.py`) when it has a kernel for it; llama.cpp's own kernels otherwise.

  kg5   Linux CUDA builds: experimental grouped-query decode attention on Turing, opt-in for a measured shape.
        Other shapes, types and devices use the library's own attention.
  am1   a decode can return only each output row's argmax (ironmule_set_argmax_only), computed where the logits are,
        instead of copying every row of logits to the host.
  gr2   CUDA builds only: CUDA graphs for decode steps of up to IRONMULE_CUDA_GRAPHS_ROWS rows (default 1, llama.cpp's).
  pk3   CUDA builds only: a graph epoch; within one graph the cuBLAS matmuls that read the same src1 node (q, k, v;
        gate, up) share one fp16 copy, and the KG library learns each src1 node (IRONMULE_PK3=0 turns sharing off).
  gr3   CUDA builds only: CUDA graphs survive warm-up (small helper graphs stay out, too many updates pause instead
        of disabling for good) and a changed graph runs once uncaptured before it is captured (IRONMULE_GR3=0 off).
  fu1   CUDA builds only: a residual ADD, its RMS_NORM and the weight MUL in one kernel, bit-identical
        (IRONMULE_FU1=0 off).
  pk4   CUDA builds only: FU1 and SWIGLU write the fp16 copy a following cuBLAS prompt matmul needs into PK3's
        slot, so no conversion kernel runs (IRONMULE_PK4=0 off).
  pk5   CUDA builds only: Q4_K -> fp16 for PK2's prompt matmuls with 16-byte stores, bit-identical to llama.cpp's
        dequantization (IRONMULE_PK5=0 off).
  pk2   CUDA builds only: from IRONMULE_CUBLAS_MIN_ROWS prompt rows on, quantized matmuls go through dequantization
        and cuBLAS with fp32 accumulation instead of llama.cpp's MMQ kernels; unset keeps llama.cpp's choice.

    python tools/build_ggml.py                 # CPU build in ~/.cache/ironmule/ggml
    python tools/build_ggml.py --cuda 75,86    # with CUDA for these compute capabilities (T4 = 75)
    python tools/build_ggml.py --rocm          # detect AMD GPU targets via the local ROCm toolkit
    python tools/build_ggml.py --no-patches    # stock llama.cpp, for a fair comparison

Then, as printed: `export IRONMULE_GGML_BINDINGS=... LLAMA_CPP_LIB_PATH=...` and `ironmule serve --engine ggml --gguf
<file.gguf> --model <registered model with the same tokenizer>`.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

VERSION = "0.3.16"
ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "native" / "ggml_patches"
# patch set -> (file in the llama.cpp tree, diff), and headers copied beside them
PATCH_SETS = {
    "fa1": ([("ggml/src/ggml-cpu/ops.cpp", "ops.cpp.diff")], [("fa1_kern.h", "ggml/src/ggml-cpu/fa1_kern.h")]),
    "row1": ([("ggml/src/ggml-cpu/amx/mmq.cpp", "mmq.cpp.diff"), ("ggml/src/ggml-cpu/amx/amx.cpp", "amx.cpp.diff")],
             [("row1_kern.h", "ggml/src/ggml-cpu/amx/row1_kern.h")]),
    "kg5": ([("ggml/src/ggml-cuda/fattn.cu", "fattn.cu.diff")],
            [(h, f"ggml/src/ggml-cuda/{h}") for h in ("kg5_dispatch.cuh", "kg5_kernel.cuh")]),
    "gr1": ([("ggml/src/ggml-cuda/ggml-cuda.cu", "ggml-cuda.cu.diff")], []),
    "kg1": ([("ggml/src/ggml-cuda/ggml-cuda.cu", "ggml-cuda.cu.diff")], []),  # after gr1, on the file gr1 left
    "pk2": ([("ggml/src/ggml-cuda/ggml-cuda.cu", "ggml-cuda.cu.diff")], []),  # after kg1, on the file kg1 left
    "am1": ([("src/llama-context.cpp", "llama-context.cpp.diff")], []),
    "gr2": ([("ggml/src/ggml-cuda/ggml-cuda.cu", "ggml-cuda.cu.diff")], []),  # after pk2
    "pk3": ([("ggml/src/ggml-cuda/ggml-cuda.cu", "ggml-cuda.cu.diff")], []),  # after gr2
    "gr3": ([("ggml/src/ggml-cuda/ggml-cuda.cu", "ggml-cuda.cu.diff")], []),  # after pk3
    "fu1": ([("ggml/src/ggml-cuda/ggml-cuda.cu", "ggml-cuda.cu.diff")], []),  # after gr3
    "pk4": ([("ggml/src/ggml-cuda/ggml-cuda.cu", "ggml-cuda.cu.diff")], []),  # after fu1
    "pk5": ([("ggml/src/ggml-cuda/ggml-cuda.cu", "ggml-cuda.cu.diff")], []),  # after pk4
}


def run(command: list[str], cwd: Path | None = None, log: Path | None = None) -> None:
    print("$", " ".join(command), flush=True)
    if log is None:
        subprocess.run(command, cwd=cwd, check=True)
        return
    with log.open("a") as handle:
        done = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
    if done.returncode:
        sys.exit(f"failed ({done.returncode}); see {log}")


def fetch(dest: Path) -> Path:
    source = dest / f"llama_cpp_python-{VERSION}"
    if source.exists():
        return source
    dest.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "download", f"llama_cpp_python=={VERSION}", "--no-binary", "llama-cpp-python", "--no-deps",
         "-d", str(dest), "-q"])
    with tarfile.open(dest / f"llama_cpp_python-{VERSION}.tar.gz") as archive:
        archive.extractall(dest, filter="data")
    return source


def patch(tree: Path, names: list[str]) -> list[str]:
    applied = []
    for name in names:
        diffs, headers = PATCH_SETS[name]
        marker = tree / f".ironmule-{name}"
        if marker.exists():
            applied.append(name)
            continue
        for target, diff in diffs:
            run(["patch", str(tree / target), str(PATCHES / name / diff)])
        for header, target in headers:
            shutil.copy(PATCHES / name / header, tree / target)
        marker.write_text("applied\n")
        applied.append(name)
    return applied


def build(tree: Path, cuda: str | None, jobs: int, rocm: str | None = None) -> Path:
    log = tree / "ironmule-build.log"
    options = ["-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON", "-DLLAMA_CURL=OFF", "-DGGML_NATIVE=ON"]
    if rocm:
        options += ["-DGGML_HIP=ON", f"-DAMDGPU_TARGETS={rocm.replace(',', ';')}"]
    if cuda:
        options += ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda.replace(',', ';')}"]
        # Kaggle: a CUDA 12 toolkit where one is installed (its default CUDA 13 postdates llama.cpp 4227c9b), and its
        # driver stub for CUDA::cuda_driver
        toolkits = sorted(Path("/usr/local").glob("cuda-12*"), reverse=True) + [Path("/usr/local/cuda")]
        root = next((t for t in toolkits if (t / "bin" / "nvcc").exists()), None)
        if root is not None:
            options += [f"-DCUDAToolkit_ROOT={root}", f"-DCMAKE_CUDA_COMPILER={root}/bin/nvcc",
                        f"-DCMAKE_LIBRARY_PATH={root}/lib64/stubs"]
            # CUDA::cuda_driver: the toolkit's stub, else the driver itself (Kaggle's toolkit ships no stub; KG1 v3's
            # build failed without it)
            driver = next(iter(sorted(root.glob("lib64/stubs/libcuda.so")) + sorted(root.glob("targets/*/lib/stubs/libcuda.so"))
                               + sorted(Path("/usr/lib/x86_64-linux-gnu").glob("libcuda.so*"))
                               + sorted(Path("/usr/local/nvidia/lib64").glob("libcuda.so*"))), None)
            if driver is not None:
                options.append(f"-DCUDA_cuda_driver_LIBRARY={driver}")
    run(["cmake", "-B", "build", *options], cwd=tree, log=log)
    run(["cmake", "--build", "build", "--target", "llama", "llama-server", "-j", str(jobs)], cwd=tree, log=log)
    return tree / "build" / "bin"


def rocm_targets() -> str:
    """Use ROCm's own enumeration, as CMake does for its native HIP architectures."""
    root = Path(os.environ.get("ROCM_PATH") or "/opt/rocm")
    for tool, flags in (("rocm_agent_enumerator", ["-t", "GPU"]), ("amdgpu-arch", [])):
        path = shutil.which(tool) or next((str(p) for p in (root / "bin" / tool, root / "llvm/bin" / tool)
                                        if p.is_file()), None)
        if not path:
            continue
        try:
            done = subprocess.run([path, *flags], capture_output=True, text=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            continue
        detected = {token.split(":")[0] for token in done.stdout.split()}
        targets = sorted(t for t in detected if t != "gfx000" and re.fullmatch(r"gfx[0-9a-f]+", t))
        if targets:
            return ",".join(targets)
    raise ValueError("no native AMD GPU architecture detected; install the ROCm toolkit or specify --rocm gfxTARGET")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", type=Path, default=Path("~/.cache/ironmule/ggml").expanduser())
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--rocm", nargs="?", const="auto", metavar="ARCHS",
                         help="build for detected AMD GPUs or specify targets, e.g. gfx1100 or gfx90a,gfx942")
    backend.add_argument("--cuda", metavar="ARCHS", help="CUDA compute capabilities, e.g. 75,86")
    backend.add_argument("--auto", action="store_true", help="detect the visible CUDA/ROCm toolchain or build for CPU")
    parser.add_argument("--env-file", type=Path, help="write the two native library exports for a later shell")
    parser.add_argument("--no-patches", action="store_true", help="stock llama.cpp")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if args.auto:
        sys.path.insert(0, str(ROOT))
        from ironmule.kernel_gen import _visible_cuda_devices, nvcc_path
        archs = sorted({d["cc"] for d in _visible_cuda_devices()}, key=int) if nvcc_path() else []
        if archs:
            args.cuda = ",".join(archs)
        elif shutil.which("hipcc") or (Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin/hipcc").is_file():
            try:
                args.rocm = rocm_targets()
            except ValueError:
                pass
        print(f"Build target: CUDA {args.cuda}" if args.cuda else f"Build target: ROCm {args.rocm}"
              if args.rocm else "Build target: CPU", flush=True)
    if args.rocm == "auto":
        try:
            args.rocm = rocm_targets()
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Detected AMD GPU targets: {args.rocm}", flush=True)
    if args.rocm and not re.fullmatch(r"gfx[0-9a-f]+(?:,gfx[0-9a-f]+)*", args.rocm):
        parser.error("--rocm requires comma-separated gfx targets, e.g. gfx1100")
    dest = args.dest / ("stock" if args.no_patches else "ironmule") / ("rocm" if args.rocm else "cuda" if args.cuda else "cpu")
    if args.rocm:
        dest = dest / args.rocm
    source = fetch(dest)
    tree = source / "vendor" / "llama.cpp"
    applied = [] if args.no_patches else patch(tree, ["fa1", "row1", "am1"] + (["gr1", "kg1", "kg5", "pk2", "gr2", "pk3", "gr3", "fu1", "pk4", "pk5"] if args.cuda else []))
    lib = build(tree, args.cuda, args.jobs, args.rocm)
    print(f"\nbuilt with patches: {', '.join(applied) or 'none'}")
    print(f"export IRONMULE_GGML_BINDINGS={source}")
    print(f"export LLAMA_CPP_LIB_PATH={lib}")
    print(f"llama-server: {lib / 'llama-server'}")
    if args.env_file:
        args.env_file.write_text(f"export IRONMULE_GGML_BINDINGS={shlex.quote(str(source.resolve()))}\n"
                                 f"export LLAMA_CPP_LIB_PATH={shlex.quote(str(lib.resolve()))}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
