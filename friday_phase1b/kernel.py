"""Closed MLX adapter for the single frozen Metal source."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .kernel_source import (
    HIDDEN_SIZE,
    KERNEL_NAME,
    KERNEL_SOURCE,
    ROWS,
    THREADGROUP_SIZE,
    validate_frozen_source,
)


class KernelContractError(RuntimeError):
    """Inputs or source differ from the frozen custom-kernel contract."""


def _validate_array(mx: Any, value: Any, shape: tuple[int, ...], name: str) -> None:
    if tuple(value.shape) != shape or value.dtype != mx.float16:
        raise KernelContractError(f"{name} shape or dtype differs from the frozen contract")


def construct_candidate(mx: Any) -> Callable[[Any, Any, Any], Any]:
    validate_frozen_source()
    forbidden = ("#include", "atomic_", "device ", "kernel ", "[[buffer")
    if any(token in KERNEL_SOURCE for token in forbidden):
        raise KernelContractError("frozen source contains a forbidden signature/header token")
    kernel = mx.fast.metal_kernel(
        name=KERNEL_NAME,
        input_names=["x", "residual", "weight"],
        output_names=["out"],
        source=KERNEL_SOURCE,
        header="",
        ensure_row_contiguous=True,
        atomic_outputs=False,
        compile_options={"math_mode": "safe"},
    )

    def candidate(x: Any, residual: Any, weight: Any) -> Any:
        _validate_array(mx, x, (ROWS, HIDDEN_SIZE), "x")
        _validate_array(mx, residual, (ROWS, HIDDEN_SIZE), "residual")
        _validate_array(mx, weight, (HIDDEN_SIZE,), "weight")
        outputs = kernel(
            inputs=[x, residual, weight],
            grid=(ROWS * THREADGROUP_SIZE, 1, 1),
            threadgroup=(THREADGROUP_SIZE, 1, 1),
            output_shapes=[(ROWS, HIDDEN_SIZE)],
            output_dtypes=[mx.float16],
            verbose=False,
        )
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 1:
            raise KernelContractError("custom kernel returned an unexpected output set")
        return outputs[0]

    return candidate
