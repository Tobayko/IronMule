#!/usr/bin/env python3
"""Width four on the qualified K=3840 kernel: same source, one more activation slot.

The `B45` generator already takes the width; nothing about the per-request arithmetic
changes. Each request keeps its own `x_thread`, its own `result`, the same `qdot`, the
same 256-value blocks, the same partial-sum order and the same `simd_sum`. What grows is
register demand: four activations and four accumulators instead of two, which is the
thing to watch and the reason this is measured rather than assumed.

The shipped two-request path is untouched; this is a separate study module.
"""

from __future__ import annotations

import mlx.core as mx

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from ironmule import kernel_registry  # noqa: E402

SIMD_SIZE = 32
VALUES_PER_THREAD = 8
BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE  # 256
NUM_SIMDGROUPS = 2
RESULTS_PER_SIMDGROUP = 4
BYTES_PER_PACK = 4
PACK_FACTOR = 8
GROUP_SIZE = 64
SCALE_STEP_PER_THREAD = GROUP_SIZE // VALUES_PER_THREAD
COMPILE_OPTIONS = {"math_mode": "safe"}
TARGET_K = 3840
BLOCKS = TARGET_K // BLOCK_SIZE  # 15

# Identical to the qualified kernel's helpers; transcribed from quantized.h, bits==4.
HEADER = """
typedef bfloat T;
typedef float U;

inline U load_vector(const device T* x, thread U* x_thread) {
  U sum = 0;
  for (int i = 0; i < 8; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}

inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  U accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < (8 / 4); i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}
"""


def _source(width: int) -> str:
    """The qualified body with `width` activations sharing every weight load."""

    declare = "\n".join(
        f"  thread U x_thread{v}[8];\n  thread U result{v}[4] = {{0}};"
        for v in range(width)
    )
    advance = "\n".join(f"    const device T* xs{v} = x{v} + base;" for v in range(width))
    loads = "\n".join(
        f"      U sum{v} = load_vector(xs{v} + block * 256, x_thread{v});"
        for v in range(width)
    )
    dots = "\n".join(
        f"        result{v}[row] += qdot(wl, x_thread{v}, s, b, sum{v});"
        for v in range(width)
    )
    # simd_sum must run on every lane; only lane 0 writes. Splitting the two keeps the
    # reduction identical to the qualified kernel's.
    reduce = "\n".join(
        f"    U reduced{v} = simd_sum(result{v}[row]);" for v in range(width)
    )
    stores = "\n".join(f"      ys{v}[row] = static_cast<T>(reduced{v});" for v in range(width))
    outs = "\n".join(f"    device T* ys{v} = out{v} + used_out_row;" for v in range(width))
    return f"""
  const int out_vec_size = shape[1];
  constexpr int in_vec_size = {TARGET_K};
  constexpr int in_vec_size_w = in_vec_size * {BYTES_PER_PACK} / {PACK_FACTOR};
  constexpr int in_vec_size_g = in_vec_size / {GROUP_SIZE};

  uint3 tid = threadgroup_position_in_grid;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  const int out_row = tid.y * 8 + simd_gid * 4;
  if (out_row >= out_vec_size) {{
    return;
  }}
  const int used_out_row = min(out_vec_size - 4, out_row);

{declare}

  const device uint8_t* ws = (const device uint8_t*)w
      + used_out_row * in_vec_size_w + simd_lid * {BYTES_PER_PACK};
  const device T* sc = scales + used_out_row * in_vec_size_g
      + simd_lid / {SCALE_STEP_PER_THREAD};
  const device T* bi = biases + used_out_row * in_vec_size_g
      + simd_lid / {SCALE_STEP_PER_THREAD};
  const int base = simd_lid * 8;
{advance}
{outs}

  for (int block = 0; block < {BLOCKS}; block++) {{
{loads}

    for (int row = 0; row < 4; row++) {{
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      U s = sc[row * in_vec_size_g];
      U b = bi[row * in_vec_size_g];
{dots}
    }}

    ws += 256 * {BYTES_PER_PACK} / {PACK_FACTOR};
    sc += 256 / {GROUP_SIZE};
    bi += 256 / {GROUP_SIZE};
  }}

  for (int row = 0; row < 4; row++) {{
{reduce}
    if (simd_lid == 0) {{
{stores}
    }}
  }}
"""


def _kernel(width: int):
    inputs = ["w", "scales", "biases"] + [f"x{v}" for v in range(width)] + ["shape"]
    outputs = [f"out{v}" for v in range(width)]
    return kernel_registry.build(
        f"qmv_k3840_shared_x{width}",
        input_names=inputs,
        output_names=outputs,
        source=_source(width),
        header=HEADER,
        ensure_row_contiguous=True,
        template={"width": width, "k": TARGET_K},
    )


SHARED = {width: _kernel(width) for width in (1, 2, 4)}
_SHAPES: dict[tuple[int, int], mx.array] = {}


def shape_array(in_features: int, out_features: int) -> mx.array:
    key = (in_features, out_features)
    cached = _SHAPES.get(key)
    if cached is None:
        cached = mx.array([in_features, out_features], dtype=mx.int32)
        mx.eval(cached)
        _SHAPES[key] = cached
    return cached


def run_shared(w, scales, biases, vectors, out_features: int):
    """One weight sweep, `len(vectors)` independent results."""

    width = len(vectors)
    kernel = SHARED[width]
    groups = (out_features + 7) // 8
    return kernel(
        inputs=[w, scales, biases, *vectors, shape_array(TARGET_K, out_features)],
        output_shapes=[(1, out_features)] * width,
        output_dtypes=[vectors[0].dtype] * width,
        grid=(SIMD_SIZE, NUM_SIMDGROUPS * groups, 1),
        threadgroup=(SIMD_SIZE, NUM_SIMDGROUPS, 1),
    )
