"""Build native Q4_K grid alternatives from the same pinned backend with a measured or empty allow-list."""

import argparse
import json
import os
from pathlib import Path

from lab.pk6.mixed_progress.trial import board
from lab.pk6.native_dispatch.prepare import substitute
from lab.pk6.native_dispatch.tile_build import NATIVE_MMQ, relink_q4
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import save


def build(work, plan=None):
    work.mkdir(exist_ok=False, parents=True)
    work = work.resolve()
    (work / "bin").mkdir()
    base = Path(os.environ["IRONMULE_GGML_BINDINGS"]) / "vendor/llama.cpp"
    header = base / "ggml/src/ggml-cuda/mmq.cuh"
    if checksum(header) != NATIVE_MMQ:
        raise ValueError("foreign GGML kernel")
    hardware = board()
    native = checksum(base / "build/bin/libggml-cuda.so")
    ranges = []
    if plan:
        p = json.loads(plan.read_bytes())
        if p["hardware"] != hardware or p["reference_cuda"] != native:
            raise ValueError("foreign grid plan")
        ranges = p["ranges"]
    lines = []
    for n, k, lo, hi, mode in ranges or [[0, 0, 0, 0, 0]]:
        if (
            any(type(v) is not int or v < 0 for v in (n, k, lo, hi, mode))
            or hi < lo
            or mode > 3
        ):
            raise ValueError("invalid launch parameters")
        lines.append("{" + ",".join(map(str, (n, k, lo, hi, mode))) + "}")
    (work / "grid_ranges.h").write_text(
        "struct GridRange { int n,k,lo,hi,mode; };\nconstexpr GridRange measured_grid_ranges[]={"
        + ",".join(lines)
        + "};\n"
    )
    shim = Path(__file__).with_name("grid.cuh")
    original = "template <ggml_type type>\nvoid mul_mat_q_case(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {"
    body = substitute(
        header.read_text(),
        original,
        shim.read_text()
        + "\n"
        + original.replace(" & args,", " & supplied,")
        + "\n    mmq_args args = supplied;",
    )
    body = substitute(
        body,
        "    const int mmq_x_max = get_mmq_x_max_host(cc);",
        """    const int mode = grid_mode(args, cc);
    const int native_limit = get_mmq_x_max_host(cc);
    const int mmq_x_max = (mode == 2 || mode == 3) && native_limit > 64 ? 64 : native_limit;
    if (mode == 1 || mode == 2) args.use_stream_k = false;
    if (mode) ++grid_calls;""",
    )
    # Ada compiles out the upstream XY device branch. Match each host grid to
    # its own template specialization, including the dynamic shared-memory limit.
    body = substitute(
        body,
        "template <ggml_type type, int mmq_x, bool need_check>\n#if defined(GGML_USE_HIP)",
        "template <ggml_type type, int mmq_x, bool need_check, bool explicit_xy = false>\n#if defined(GGML_USE_HIP)",
    )
    body = substitute(
        body,
        "#if (defined(GGML_USE_HIP) && !defined(CDNA)) || __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA\n    {\n        const int wt",
        "#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA\n    if constexpr (explicit_xy)\n#endif\n    {\n        const int wt",
    )
    body = substitute(
        body,
        "    }\n#endif // (defined(GGML_USE_HIP) && !defined(CDNA3)) || __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA\n\n    const     int64_t blocks_per_ne00",
        "    }\n\n    const     int64_t blocks_per_ne00",
    )
    begin = body.index("    if (!args.use_stream_k) {")
    end = body.index("    const dim3 block_nums_stream_k", begin)
    xy = body[begin:end]
    if xy.count("mul_mat_q<type, mmq_x, need_check>") != 2:
        raise ValueError("unsupported XY kernel launch sites")
    xy = xy.replace(
        "mul_mat_q<type, mmq_x, need_check>", "mul_mat_q<type, mmq_x, need_check, true>"
    )
    xy = xy.replace(
        "    if (!args.use_stream_k) {",
        "    if (!args.use_stream_k) {\n        CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, mmq_x, false, true>), nbytes_shared);\n        CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, mmq_x, true, true>), nbytes_shared);",
    )
    body = body[:begin] + xy + body[end:]
    relink_q4(body, work, base)
    save(
        work / "build.json",
        {
            "hardware": hardware,
            "reference_cuda": native,
            "upstream": NATIVE_MMQ,
            "source": checksum(shim),
            "plan": checksum(plan) if plan else None,
            "backend": checksum(work / "bin/libggml-cuda.so"),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--plan", type=Path)
    a = p.parse_args()
    build(a.output, a.plan)
