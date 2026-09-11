"""The `B48` shared-weight kernel for the aligned projections, frozen as qualified.

`K=4096` and `K=15360` run MLX's `qmv_fast_impl`, which uses 16 values per thread and
512-value blocks and has no tail path, so this is a separate transcription rather than a
widened copy of the `K=3840` kernel. Copied unchanged from the study tooling;
`tests/test_paired_opt_in.py` asserts the two still match.
"""

from __future__ import annotations

import mlx.core as mx

from . import kernel_registry

SIMD_SIZE = 32
PACKS_PER_THREAD = 2
PACK_FACTOR = 8
BYTES_PER_PACK = 4
VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD  # 16
BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE  # 512
NUM_SIMDGROUPS = 2
RESULTS_PER_SIMDGROUP = 4
GROUP_SIZE = 64
SCALE_STEP_PER_THREAD = GROUP_SIZE // VALUES_PER_THREAD  # 4
COMPILE_OPTIONS = {"math_mode": "safe"}
SUPPORTED_K = (4096, 15360)

# Transcribed from mlx/backend/metal/kernels/quantized.h, v0.32.0, bits==4, with
# values_per_thread = 16.
HEADER = """
typedef bfloat T;
typedef float U;

inline U load_vector16(const device T* x, thread U* x_thread) {
  U sum = 0;
  for (int i = 0; i < 16; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}

inline U qdot16(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  U accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < (16 / 4); i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}
"""


def _source(k: int, width: int) -> str:
    blocks = k // BLOCK_SIZE
    declare = "\n".join(
        f"  thread U x_thread{v}[16];\n  thread U result{v}[4] = {{0}};"
        for v in range(width)
    )
    starts = "\n".join(f"  const device T* xs{v} = x{v} + simd_lid * 16;" for v in range(width))
    outs = "\n".join(f"  device T* ys{v} = out{v} + out_row;" for v in range(width))
    loads = "\n".join(
        f"    U sum{v} = load_vector16(xs{v} + block * {BLOCK_SIZE}, x_thread{v});"
        for v in range(width)
    )
    dots = "\n".join(
        f"      result{v}[row] += qdot16(wl, x_thread{v}, s, b, sum{v});"
        for v in range(width)
    )
    reduce = "\n".join(f"    U reduced{v} = simd_sum(result{v}[row]);" for v in range(width))
    stores = "\n".join(f"      ys{v}[row] = static_cast<T>(reduced{v});" for v in range(width))
    return f"""
  const int out_vec_size = shape[1];
  constexpr int in_vec_size = {k};
  constexpr int in_vec_size_w = in_vec_size * {BYTES_PER_PACK} / {PACK_FACTOR};
  constexpr int in_vec_size_g = in_vec_size / {GROUP_SIZE};

  uint3 tid = threadgroup_position_in_grid;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  const int out_row = tid.y * 8 + simd_gid * 4;
  if (out_row >= out_vec_size) {{
    return;
  }}

{declare}

  const device uint8_t* ws = (const device uint8_t*)w
      + out_row * in_vec_size_w + simd_lid * {PACKS_PER_THREAD} * {BYTES_PER_PACK};
  const device T* sc = scales + out_row * in_vec_size_g
      + simd_lid / {SCALE_STEP_PER_THREAD};
  const device T* bi = biases + out_row * in_vec_size_g
      + simd_lid / {SCALE_STEP_PER_THREAD};
{starts}
{outs}

  for (int block = 0; block < {blocks}; block++) {{
{loads}

    for (int row = 0; row < 4; row++) {{
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      U s = sc[row * in_vec_size_g];
      U b = bi[row * in_vec_size_g];
{dots}
    }}

    ws += {BLOCK_SIZE} * {BYTES_PER_PACK} / {PACK_FACTOR};
    sc += {BLOCK_SIZE} / {GROUP_SIZE};
    bi += {BLOCK_SIZE} / {GROUP_SIZE};
  }}

  for (int row = 0; row < 4; row++) {{
{reduce}
    if (simd_lid == 0) {{
{stores}
    }}
  }}
"""


def _kernel(k: int, width: int):
    inputs = ["w", "scales", "biases"] + [f"x{v}" for v in range(width)] + ["shape"]
    return kernel_registry.build(
        f"qmv_fast_k{k}_x{width}",
        input_names=inputs,
        output_names=[f"out{v}" for v in range(width)],
        source=_source(k, width),
        header=HEADER,
        ensure_row_contiguous=True,
        template={"width": width, "k": k},
    )


KERNELS = {(k, width): _kernel(k, width) for k in SUPPORTED_K for width in (1, 2)}
_SHAPES: dict[tuple[int, int], mx.array] = {}


def shape_array(in_features: int, out_features: int) -> mx.array:
    key = (in_features, out_features)
    cached = _SHAPES.get(key)
    if cached is None:
        cached = mx.array([in_features, out_features], dtype=mx.int32)
        mx.eval(cached)
        _SHAPES[key] = cached
    return cached


def run_fast(w, scales, biases, vectors, out_features: int, in_features: int):
    """One weight sweep for `len(vectors)` requests on an aligned reduction length."""

    width = len(vectors)
    groups = (out_features + 7) // 8
    return KERNELS[(in_features, width)](
        inputs=[w, scales, biases, *vectors, shape_array(in_features, out_features)],
        output_shapes=[(1, out_features)] * width,
        output_dtypes=[vectors[0].dtype] * width,
        grid=(SIMD_SIZE, NUM_SIMDGROUPS * groups, 1),
        threadgroup=(SIMD_SIZE, NUM_SIMDGROUPS, 1),
    )
