"""Recompile one Q4_K template instance from a received order and relink a private copy of GGML CUDA."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from lab.pk6.native_dispatch.prepare import substitute
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import save
from training.code_author import read_request, read_response

NATIVE_MMQ = "537c64a520bc24ed4f874945dedac8a8f4a87dbd8adc3f4ee37ed81c9813e8fc"


def build(state, work):
    request_id, request = read_request(state / "author")
    if (
        request["order"]["target"]["ggml_type"] != 12
        or request["order"]["target"]["executed_tile"] != 128
    ):
        raise ValueError("unrecognized order target")
    # Input is the reviewed reply checked into this experiment. Loading a different reply needs code review.
    proposal = read_response(state / "author", request_id)
    base = Path(os.environ["IRONMULE_GGML_BINDINGS"]) / "vendor/llama.cpp"
    native = base / "ggml/src/ggml-cuda"
    header = native / "mmq.cuh"
    if checksum(header) != NATIVE_MMQ:
        raise ValueError("different upstream kernel source")
    work.mkdir(exist_ok=False, parents=True)
    work = work.resolve()
    (work / "bin").mkdir()
    body = header.read_text()
    needle = "template <ggml_type type>\nvoid mul_mat_q_case(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {"
    body = substitute(
        body,
        needle,
        "#include <cstring>\n#include <cstdlib>\n#include <atomic>\n"
        + 'static std::atomic<uint64_t> tile_calls{0};\nextern "C" uint64_t im_mmq_tile_calls() { return tile_calls.load(); }\n'
        + proposal["definition"]
        + "\n"
        + needle,
    )
    body = substitute(
        body,
        "    const int mmq_x_max = get_mmq_x_max_host(cc);",
        """    const int native_limit = get_mmq_x_max_host(cc);
    const int mmq_x_max = prompt_tile_limit(type,cc,args.nchannels_y*args.nsamples_y,args.ncols_y,native_limit);
    if (mmq_x_max < native_limit) ++tile_calls;""",
    )
    (work / "mmq.cuh").write_text(body)
    cpp = work / "instance.cu"
    cpp.write_text('#include "mmq.cuh"\nDECL_MMQ_CASE(GGML_TYPE_Q4_K);\n')
    tree = base / "build"
    cmds = json.loads((tree / "compile_commands.json").read_bytes())
    cmd = next(r for r in cmds if Path(r["file"]).name == "mmq-instance-q4_k.cu")
    directory = Path(cmd["directory"])
    args = shlex.split(cmd["command"])
    obj = args[args.index("-o") + 1]
    args[args.index("-o") + 1] = str(work / "instance.o")
    args[args.index(cmd["file"])] = str(cpp)
    args.insert(1, "-I" + str(native))
    subprocess.run(args, check=True, cwd=directory)
    link = shlex.split((directory / "CMakeFiles/ggml-cuda.dir/link.txt").read_text())
    link[link.index("-o") + 1] = str(work / "bin/libggml-cuda.so")
    resp = next(v for v in link if v.endswith("objects1.rsp"))
    objs = shlex.split((directory / resp[1:]).read_text())
    if objs.count(obj) != 1:
        raise ValueError("cannot isolate Q4_K object")
    objs[objs.index(obj)] = str(work / "instance.o")
    (work / "objects.rsp").write_text(shlex.join(objs))
    link[link.index(resp)] = "@" + str(work / "objects.rsp")
    subprocess.run(link, check=True, cwd=directory)
    for path in (tree / "bin").glob("*.so*"):
        if path.name != "libggml-cuda.so":
            shutil.copy2(path, work / "bin" / path.name, follow_symlinks=True)
    save(
        work / "build.json",
        {
            "request": request_id,
            "header_sha256": NATIVE_MMQ,
            "edited_header_sha256": checksum(work / "mmq.cuh"),
            "reference_cuda": checksum(tree / "bin/libggml-cuda.so"),
            "candidate_cuda": checksum(work / "bin/libggml-cuda.so"),
            "observed_source": checksum(cpp),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("order", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    build(a.order, a.output)
