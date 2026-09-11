"""The `B42`/`B43` quantised matrix-vector kernel for `K=3840`, behind an admission gate.

The kernel itself is the qualified candidate, copied unchanged from the study tooling:
MLX `0.32.0`'s `qmv_impl` for `bits=4`, `group_size=64`, with the reduction length fixed
at 3840. `tests/test_qmv_k3840_integration.py` asserts it is still character-identical to
the studied source, so integration cannot quietly alter it.

Admission is deliberately narrow. `K == 3840` is not sufficient: the hardware
fingerprint, the MLX and mlx_lm versions, the model identity digest, the architecture,
the quantisation and every projection's dtype, group size, bit width, output width and
row-contiguity are checked once, at load. Anything unrecognised leaves the model on the
library path. Prefill, multi-token inputs, other shapes, sampling, batching and grouped
execution are never routed here.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from . import kernel_registry


class K3840Unsupported(RuntimeError):
    """The candidate is not admitted here. The library path stays in charge."""


# The exact state the candidate was qualified on. B42 proved bit identity, B43 confirmed
# the model-level result over two preregistered sessions.
VERIFIED_HARDWARE_FINGERPRINT = "dc652d66f24ac207"
VERIFIED_MLX = "0.32.0"
VERIFIED_MLX_LM = "0.31.3"
VERIFIED_IDENTITY_SHA256 = (
    "2b5b13a3c53a96299b33d0385b13a4b54973b810540cf7a99d4aa3966ebf1474"
)
VERIFIED_ARCHITECTURE = "gemma3"
VERIFIED_BITS = 4
VERIFIED_GROUP_SIZE = 64
VERIFIED_DTYPE = "bfloat16"
TARGET_K = 3840

SIMD_SIZE = 32
VALUES_PER_THREAD = 8  # pack_factor(4 bit, 32) * packs_per_thread(1)
BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE  # 256
NUM_SIMDGROUPS = 2
RESULTS_PER_SIMDGROUP = 4
BYTES_PER_PACK = 4
PACK_FACTOR = 8
GROUP_SIZE = 64
SCALE_STEP_PER_THREAD = GROUP_SIZE // VALUES_PER_THREAD  # 8
COMPILE_OPTIONS = {"math_mode": "safe"}
# Recorded, not passed, and that is correct: MLX 0.32.0's own header declares
# `CompileOptions.math_mode = MathMode::Safe`, so passing nothing already compiles safe.
# Measured (`B53_compiler_contract_v2_20260910`): passing this explicitly is byte-identical
# to passing nothing, on real Gemma projections and on 96 constructed hard cases, while a
# fast arm differs everywhere. Do not start passing it: it would change the kernel
# identifier for no change in translation.

# Transcribed from mlx/backend/metal/kernels/quantized.h, v0.32.0, bits==4 paths only.
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

inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  U sum = 0;
  for (int i = 0; i < N; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  for (int i = N; i < 8; i++) {
    x_thread[i] = 0;
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

inline U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N) {
  U accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < (N / 4); i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}
"""

# The else-branch of qmv_impl: the tile is moved back rather than guarded, which is the
# branch every Gemma projection takes because out_vec_size is far above 8.
BODY = """
  constexpr int num_simdgroups = {num_simdgroups};
  constexpr int results_per_simdgroup = {results_per_simdgroup};
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = {pack_factor};
  constexpr int bytes_per_pack = {bytes_per_pack};
  constexpr int values_per_thread = {values_per_thread};
  constexpr int block_size = {block_size};
  constexpr int group_size = {group_size};
  constexpr int scale_step_per_thread = {scale_step_per_thread};

  {in_vec_size_decl}
  const int out_vec_size = shape[1];

  uint3 tid = threadgroup_position_in_grid;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {{0}};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {{
    return;
  }}

  const device T* xs = x;
  const device T* sc = scales;
  const device T* bi = biases;
  device T* ys = out;

  ws += used_out_row * in_vec_size_w +
      simd_lid * packs_per_thread * bytes_per_pack;
  sc += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  bi += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  xs += tid.x * in_vec_size + simd_lid * values_per_thread;
  ys += tid.x * out_vec_size + used_out_row;

  {main_loop}

  {tail}

  for (int row = 0; row < results_per_simdgroup; row++) {{
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {{
      ys[row] = static_cast<T>(result[row]);
    }}
  }}
"""

INNER = """
      U sum = load_vector<T, U, values_per_thread, 4>(xs, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {{
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bi + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot(wl, x_thread, s, b, sum);
      }}

      ws += block_size * bytes_per_pack / pack_factor;
      sc += block_size / group_size;
      bi += block_size / group_size;
      xs += block_size;
"""

RUNTIME_LOOP = """
  int k = 0;
  for (; k < in_vec_size - block_size; k += block_size) {{
""" + INNER.replace("load_vector<T, U, values_per_thread, 4>", "load_vector") + """  }}
"""

RUNTIME_TAIL = """
  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {{
    U sum = load_vector_safe(xs, x_thread, remaining);

    for (int row = 0; row < results_per_simdgroup; row++) {{
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = sc + row * in_vec_size_g;
      const device T* bl = bi + row * in_vec_size_g;

      U s = sl[0];
      U b = bl[0];
      result[row] += qdot_safe(wl, x_thread, s, b, sum, remaining);
    }}
  }}
"""

# Full unrolling was measured at 7.1x slower and removed; the loop bound stays a
# compile-time constant, which is the whole point.
# K = 3840 is 15 whole blocks of 256. The reference runs 14 through the unsafe path and
# the fifteenth through the safe path with remaining == values_per_thread, where the two
# are character-identical. Unrolling all 15 through qdot keeps every partial sum in the
# same order and drops the clamp, the branch and the bounds arithmetic.
FIXED_LOOP = """
  for (int block = 0; block < 15; block++) {{
""" + INNER.replace("load_vector<T, U, values_per_thread, 4>", "load_vector") + """  }}
"""


def _kernel(name: str, fixed_k: int | None):
    if fixed_k is None:
        in_vec_size_decl = "const int in_vec_size = shape[0];"
        loop, tail = RUNTIME_LOOP, RUNTIME_TAIL
    else:
        if fixed_k % BLOCK_SIZE:
            raise ValueError(f"{fixed_k} is not a whole number of {BLOCK_SIZE}-value blocks")
        in_vec_size_decl = f"constexpr int in_vec_size = {fixed_k};"
        loop, tail = FIXED_LOOP.replace("15", str(fixed_k // BLOCK_SIZE)), ""
    source = BODY.format(
        num_simdgroups=NUM_SIMDGROUPS,
        results_per_simdgroup=RESULTS_PER_SIMDGROUP,
        pack_factor=PACK_FACTOR,
        bytes_per_pack=BYTES_PER_PACK,
        values_per_thread=VALUES_PER_THREAD,
        block_size=BLOCK_SIZE,
        group_size=GROUP_SIZE,
        scale_step_per_thread=SCALE_STEP_PER_THREAD,
        in_vec_size_decl=in_vec_size_decl,
        main_loop=loop,
        tail=tail,
    )
    return kernel_registry.build(
        name,
        input_names=["w", "scales", "biases", "x", "shape"],
        output_names=["out"],
        source=source,
        header=HEADER,
        ensure_row_contiguous=True,
        template={"fixed_k": fixed_k},
    )


PORT = _kernel("qmv_port", None)
K3840 = _kernel("qmv_k3840", 3840)


_SHAPES: dict[tuple[int, int], mx.array] = {}


def shape_array(in_features: int, out_features: int) -> mx.array:
    """One shared constant per geometry; building it per call would be measured work."""

    key = (in_features, out_features)
    cached = _SHAPES.get(key)
    if cached is None:
        cached = mx.array([in_features, out_features], dtype=mx.int32)
        mx.eval(cached)
        _SHAPES[key] = cached
    return cached


def run(kernel, w, scales, biases, x, out_features: int, in_features: int):
    """Dispatch exactly as MLX does: (32, 2, 1) threads, ceil(N/8) groups in y."""

    groups = (out_features + 7) // 8
    shape = shape_array(in_features, out_features)
    return kernel(
        inputs=[w, scales, biases, x, shape],
        output_shapes=[(1, out_features)],
        output_dtypes=[x.dtype],
        grid=(SIMD_SIZE, NUM_SIMDGROUPS * groups, 1),
        threadgroup=(SIMD_SIZE, NUM_SIMDGROUPS, 1),
    )[0]


class K3840QuantizedLinear(nn.Module):
    """The admitted projection with its matmul replaced. Same buffers, same bytes.

    Decode reaches the kernel; anything else falls back to the library call on the very
    same arrays, so a shape this module was not qualified for costs one comparison.
    """

    def __init__(self, source: nn.QuantizedLinear) -> None:
        super().__init__()
        # The original module is kept so a fallback restores the known-good path rather
        # than rebuilding one from parts.
        self._source = source
        self.weight = source.weight
        self.scales = source.scales
        self.biases = source.biases
        self.group_size = source.group_size
        self.bits = source.bits
        self.out_features = int(source.weight.shape[0])

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        # Single-token decode only. Prefill and every wider input keep the library path.
        if len(shape) != 3 or shape[0] != 1 or shape[1] != 1:
            return mx.quantized_matmul(
                x, self.weight, self.scales, self.biases,
                transpose=True, group_size=self.group_size, bits=self.bits,
            )
        out = run(K3840, self.weight, self.scales, self.biases, x[0],
                  self.out_features, TARGET_K)
        return out[None]


def _admit_environment(identity: Any) -> None:
    """Everything expensive happens here, once, and never per token."""

    import mlx_lm  # noqa: PLC0415 - only needed while admitting

    from .hw import fingerprint  # noqa: PLC0415 - avoids a load-time cycle

    found = fingerprint()
    if found != VERIFIED_HARDWARE_FINGERPRINT:
        raise K3840Unsupported(
            f"qualified on hardware {VERIFIED_HARDWARE_FINGERPRINT}, found {found}"
        )
    if mx.__version__ != VERIFIED_MLX:
        raise K3840Unsupported(
            f"qualified on mlx {VERIFIED_MLX}, found {mx.__version__}"
        )
    if mlx_lm.__version__ != VERIFIED_MLX_LM:
        raise K3840Unsupported(
            f"qualified on mlx_lm {VERIFIED_MLX_LM}, found {mlx_lm.__version__}"
        )
    if identity is None:
        raise K3840Unsupported("no model identity available; the candidate fails closed")
    digest = getattr(identity, "identity_sha256", None)
    if digest != VERIFIED_IDENTITY_SHA256:
        raise K3840Unsupported(
            f"qualified on identity {VERIFIED_IDENTITY_SHA256[:16]}, found {str(digest)[:16]}"
        )
    architecture = getattr(identity, "architecture", None)
    if architecture != VERIFIED_ARCHITECTURE:
        raise K3840Unsupported(
            f"qualified on architecture {VERIFIED_ARCHITECTURE}, found {architecture}"
        )


def _admit_projection(module: nn.QuantizedLinear) -> bool:
    """K alone is not enough: bits, group size, dtype, width and layout all decide."""

    if int(module.bits) != VERIFIED_BITS or int(module.group_size) != VERIFIED_GROUP_SIZE:
        return False
    columns = int(module.weight.shape[1]) * 32 // int(module.bits)
    if columns != TARGET_K:
        return False
    out_features = int(module.weight.shape[0])
    if out_features % (NUM_SIMDGROUPS * RESULTS_PER_SIMDGROUP):
        return False  # the kernel writes eight rows per threadgroup
    for array in (module.scales, module.biases):
        if str(array.dtype).rsplit(".", 1)[-1] != VERIFIED_DTYPE:
            return False
    if str(module.weight.dtype).rsplit(".", 1)[-1] != "uint32":
        return False
    if int(module.scales.shape[1]) != TARGET_K // VERIFIED_GROUP_SIZE:
        return False
    if module.scales.shape != module.biases.shape:
        return False
    # MLX exposes neither strides nor flags, so row-contiguity cannot be read off an
    # array. The kernel is built with ensure_row_contiguous=True, which makes MLX supply
    # a contiguous copy when it has to. What is checkable is that each buffer is exactly
    # as large as its shape implies, which rules out a view over something larger.
    expected = (
        (module.weight, out_features * (TARGET_K * VERIFIED_BITS // 32) * 4),
        (module.scales, out_features * (TARGET_K // VERIFIED_GROUP_SIZE) * 2),
        (module.biases, out_features * (TARGET_K // VERIFIED_GROUP_SIZE) * 2),
    )
    return all(int(array.nbytes) == size for array, size in expected)


def _replace(module: nn.Module, admitted: list[str], skipped: list[str],
             prefix: str = "") -> None:
    for name, child in list(module.children().items()):
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, list):
            for index, item in enumerate(child):
                if isinstance(item, nn.Module):
                    _replace(item, admitted, skipped, f"{path}.{index}")
            continue
        if isinstance(child, nn.QuantizedLinear):
            columns = int(child.weight.shape[1]) * 32 // int(child.bits)
            if columns != TARGET_K:
                continue
            if _admit_projection(child):
                setattr(module, name, K3840QuantizedLinear(child))
                admitted.append(path)
            else:
                skipped.append(path)
        elif isinstance(child, nn.Module):
            _replace(child, admitted, skipped, path)


def _restore(module: nn.Module, restored: list[str], prefix: str = "") -> None:
    for name, child in list(module.children().items()):
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, list):
            for index, item in enumerate(child):
                if isinstance(item, nn.Module):
                    _restore(item, restored, f"{path}.{index}")
            continue
        if isinstance(child, K3840QuantizedLinear):
            setattr(module, name, child._source)
            restored.append(path)
        elif isinstance(child, nn.Module):
            _restore(child, restored, path)


def disable(model: nn.Module) -> int:
    """Put every admitted projection back on the library path. Returns how many.

    This is the fallback the contract requires: it restores the original modules, so the
    model returns to a state that was never touched rather than to a reconstruction.
    Callers must still discard any KV cache built while the candidate was active.
    """

    restored: list[str] = []
    _restore(model, restored)
    return len(restored)


def enable(model: nn.Module, identity: Any) -> dict[str, Any]:
    """Admit the candidate on this model, or raise and leave it untouched.

    Returns what was admitted and what was declined, so a caller can record it. The
    model is only mutated after the environment passes, so a rejected environment
    cannot leave a half-swapped model behind.
    """

    _admit_environment(identity)
    admitted: list[str] = []
    skipped: list[str] = []
    _replace(model, admitted, skipped)
    if not admitted:
        raise K3840Unsupported("no projection met the admission conditions")
    return {
        "admitted": len(admitted),
        "declined": len(skipped),
        "admitted_paths": admitted,
        "declined_paths": skipped,
        "hardware_fingerprint": VERIFIED_HARDWARE_FINGERPRINT,
        "identity_sha256": VERIFIED_IDENTITY_SHA256,
    }
