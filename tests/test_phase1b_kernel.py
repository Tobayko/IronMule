from __future__ import annotations

import unittest

from friday_phase1b.kernel import KernelContractError, construct_candidate
from friday_phase1b.kernel_source import (
    HIDDEN_SIZE,
    KERNEL_NAME,
    KERNEL_SOURCE,
    KERNEL_SOURCE_SHA256,
    ROWS,
    THREADGROUP_SIZE,
    validate_frozen_source,
)


class _Array:
    def __init__(self, shape: tuple[int, ...], dtype: object, strides: tuple[int, ...]) -> None:
        self.shape = shape
        self.dtype = dtype
        self.strides = strides


class _Fast:
    def __init__(self, owner: "_MX") -> None:
        self.owner = owner

    def metal_kernel(self, **kwargs: object):
        self.owner.constructor = kwargs

        def kernel(**call: object):
            self.owner.call = call
            return [_Array((ROWS, HIDDEN_SIZE), self.owner.float16, (HIDDEN_SIZE, 1))]

        return kernel


class _MX:
    def __init__(self) -> None:
        self.float16 = object()
        self.fast = _Fast(self)
        self.constructor: dict[str, object] = {}
        self.call: dict[str, object] = {}


class Phase1BKernelTests(unittest.TestCase):
    def test_source_name_and_geometry_are_frozen(self) -> None:
        validate_frozen_source()
        self.assertEqual(
            KERNEL_SOURCE_SHA256,
            "33b626c16c79819d6995d6bb78745eb1fd81face648b59f505a924d3125da6f6",
        )
        self.assertTrue(KERNEL_NAME.endswith(KERNEL_SOURCE_SHA256[:12]))
        self.assertNotIn("#include", KERNEL_SOURCE)
        self.assertNotIn("atomic_", KERNEL_SOURCE)
        self.assertEqual(KERNEL_SOURCE.count("out[index] ="), 1)

    def test_adapter_passes_only_fixed_configuration(self) -> None:
        mx = _MX()
        candidate = construct_candidate(mx)
        x = _Array((ROWS, HIDDEN_SIZE), mx.float16, (HIDDEN_SIZE, 1))
        residual = _Array((ROWS, HIDDEN_SIZE), mx.float16, (HIDDEN_SIZE, 1))
        weight = _Array((HIDDEN_SIZE,), mx.float16, (1,))
        output = candidate(x, residual, weight)
        self.assertEqual(output.shape, (ROWS, HIDDEN_SIZE))
        self.assertEqual(mx.constructor["name"], KERNEL_NAME)
        self.assertEqual(mx.constructor["source"], KERNEL_SOURCE)
        self.assertEqual(mx.constructor["header"], "")
        self.assertEqual(mx.constructor["compile_options"], {"math_mode": "safe"})
        self.assertTrue(mx.constructor["ensure_row_contiguous"])
        self.assertEqual(
            mx.call["grid"], (ROWS * THREADGROUP_SIZE, 1, 1)
        )
        self.assertEqual(mx.call["threadgroup"], (THREADGROUP_SIZE, 1, 1))

    def test_adapter_rejects_wrong_shape_and_dtype(self) -> None:
        mx = _MX()
        candidate = construct_candidate(mx)
        good = _Array((ROWS, HIDDEN_SIZE), mx.float16, (HIDDEN_SIZE, 1))
        weight = _Array((HIDDEN_SIZE,), mx.float16, (1,))
        with self.assertRaises(KernelContractError):
            candidate(_Array((1, HIDDEN_SIZE), mx.float16, (HIDDEN_SIZE, 1)), good, weight)
        with self.assertRaises(KernelContractError):
            candidate(_Array((ROWS, HIDDEN_SIZE), object(), (HIDDEN_SIZE, 1)), good, weight)


if __name__ == "__main__":
    unittest.main()
