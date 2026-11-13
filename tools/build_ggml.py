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

    python tools/build_ggml.py                 # CPU build in ~/.cache/ironmule/ggml
    python tools/build_ggml.py --cuda 75,86    # with CUDA for these compute capabilities (T4 = 75)
    python tools/build_ggml.py --no-patches    # stock llama.cpp, for a fair comparison

Then, as printed: `export IRONMULE_GGML_BINDINGS=... LLAMA_CPP_LIB_PATH=...` and `ironmule serve --engine ggml --gguf
<file.gguf> --model <registered model with the same tokenizer>`.
"""

from __future__ import annotations

import argparse
import shutil
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
    run([sys.executable, "-m", "pip", "download", f"llama_cpp_python=={VERSION}", "--no-binary", ":all:", "--no-deps",
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


def build(tree: Path, cuda: str | None, jobs: int) -> Path:
    log = tree / "ironmule-build.log"
    options = ["-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON", "-DLLAMA_CURL=OFF", "-DGGML_NATIVE=ON"]
    if cuda:
        options += ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda.replace(',', ';')}"]
    run(["cmake", "-B", "build", *options], cwd=tree, log=log)
    run(["cmake", "--build", "build", "--target", "llama", "llama-server", "-j", str(jobs)], cwd=tree, log=log)
    return tree / "build" / "bin"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", type=Path, default=Path("~/.cache/ironmule/ggml").expanduser())
    parser.add_argument("--cuda", metavar="ARCHS", help="build with CUDA for these compute capabilities, e.g. 75,86")
    parser.add_argument("--no-patches", action="store_true", help="stock llama.cpp")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    dest = args.dest / ("stock" if args.no_patches else "ironmule") / ("cuda" if args.cuda else "cpu")
    source = fetch(dest)
    tree = source / "vendor" / "llama.cpp"
    applied = [] if args.no_patches else patch(tree, ["fa1", "row1"])
    lib = build(tree, args.cuda, args.jobs)
    print(f"\nbuilt with patches: {', '.join(applied) or 'none'}")
    print(f"export IRONMULE_GGML_BINDINGS={source}")
    print(f"export LLAMA_CPP_LIB_PATH={lib}")
    print(f"llama-server: {lib / 'llama-server'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
