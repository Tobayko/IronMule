"""Frozen Metal source for the single Phase-1B candidate.

This module intentionally imports neither MLX nor any dynamic source.  The source
is bound to its SHA-256 in the preregistration and is compiled only by the fixed
worker entrypoint.
"""

from __future__ import annotations

import hashlib

ROWS = 1024
HIDDEN_SIZE = 2560
THREADGROUP_SIZE = 256
SIMDGROUP_SIZE = 32
SIMDGROUPS_PER_THREADGROUP = THREADGROUP_SIZE // SIMDGROUP_SIZE
EPSILON = 1.0e-6

KERNEL_SOURCE = r"""
uint row = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint simd_id = simdgroup_index_in_threadgroup;
uint row_base = row * 2560u;

threadgroup half z_cache[2560];
threadgroup float simd_sums[8];
threadgroup float inverse_rms[1];

float local_sum = 0.0f;
for (uint column = tid; column < 2560u; column += 256u) {
    uint index = row_base + column;
    half z = half(float(x[index]) + float(residual[index]));
    z_cache[column] = z;
    float z_float = float(z);
    local_sum += z_float * z_float;
}

local_sum = simd_sum(local_sum);
if (lane == 0u) {
    simd_sums[simd_id] = local_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (simd_id == 0u) {
    float total = lane < 8u ? simd_sums[lane] : 0.0f;
    total = simd_sum(total);
    if (lane == 0u) {
        inverse_rms[0] = metal::rsqrt(total * 0.000390625f + 0.000001f);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float scale = inverse_rms[0];
for (uint column = tid; column < 2560u; column += 256u) {
    uint index = row_base + column;
    out[index] = half(float(z_cache[column]) * scale * float(weight[column]));
}
""".strip()

KERNEL_SOURCE_SHA256 = hashlib.sha256(KERNEL_SOURCE.encode("utf-8")).hexdigest()
KERNEL_NAME = f"friday_rrms_f16_r1024_h2560_{KERNEL_SOURCE_SHA256[:12]}"


def validate_frozen_source() -> None:
    """Fail closed if the checked-in source or fixed geometry drifts."""

    if ROWS != 1024 or HIDDEN_SIZE != 2560 or THREADGROUP_SIZE != 256:
        raise RuntimeError("frozen Phase-1B geometry changed")
    if SIMDGROUPS_PER_THREADGROUP != 8:
        raise RuntimeError("frozen Phase-1B SIMD geometry changed")
    if KERNEL_SOURCE_SHA256 != hashlib.sha256(
        KERNEL_SOURCE.encode("utf-8")
    ).hexdigest():
        raise RuntimeError("frozen Phase-1B Metal source digest mismatch")
    if not KERNEL_NAME.endswith(KERNEL_SOURCE_SHA256[:12]):
        raise RuntimeError("kernel name is not bound to the source digest")
