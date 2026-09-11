from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from b42_qmv_kernel import (  # noqa: E402
    BLOCK_SIZE,
    COMPILE_OPTIONS,
    K3840,
    PORT,
    run,
    shape_array,
)

# The real Gemma 12B output widths whose reduction dimension is 3840.
WIDTHS = (2048, 4096, 15360)


def _case(out_features: int, in_features: int = 3840):
    # Seeded: a bit-identity claim that fails must fail on inputs someone can rebuild.
    # A run of this test was once seen to differ under full-suite load and could not be
    # reproduced afterwards, which is exactly what unseeded inputs make impossible.
    mx.random.seed(20260909 + out_features)
    dense = mx.random.normal((out_features, in_features)).astype(mx.bfloat16)
    weight, scales, biases = mx.quantize(dense, group_size=64, bits=4)
    x = mx.random.normal((1, in_features)).astype(mx.bfloat16)
    mx.eval(weight, scales, biases, x)
    return weight, scales, biases, x


def _digest(*arrays) -> str:
    running = hashlib.sha256()
    for array in arrays:
        running.update(bytes(memoryview(array)))
    return running.hexdigest()[:16]


@pytest.mark.parametrize("out_features", WIDTHS)
def test_both_kernels_match_the_library_byte_for_byte(out_features: int) -> None:
    """The entry's whole claim: same bytes out, not merely a close answer."""

    _require_the_gpu_default()
    weight, scales, biases, x = _case(out_features)
    reference = mx.quantized_matmul(
        x, weight, scales, biases, transpose=True, group_size=64, bits=4
    )
    ported = run(PORT, weight, scales, biases, x, out_features, 3840)
    special = run(K3840, weight, scales, biases, x, out_features, 3840)
    mx.eval(reference, ported, special)

    inputs = _digest(weight, scales, biases, x)
    expected = bytes(memoryview(reference))
    assert bytes(memoryview(ported)) == expected, f"ported, inputs {inputs}"
    assert bytes(memoryview(special)) == expected, f"specialised, inputs {inputs}"


def _captured_source(module, builder_args) -> str:
    """The exact source a module hands to `mx.fast.metal_kernel`, without compiling it."""

    captured = {}
    real = mx.fast.metal_kernel

    def record(**kwargs):
        captured["body"] = kwargs["source"] + "\n--header--\n" + (kwargs.get("header") or "")
        captured["name"] = kwargs["name"]
        return object()

    try:
        mx.fast.metal_kernel = record
        module._kernel(*builder_args)
    finally:
        mx.fast.metal_kernel = real
    return captured["name"], captured["body"]


# Groups of modules that register the same Metal kernel name. `mx.fast.metal_kernel`
# caches by name, so the second registration in one process is silently ignored and
# whichever module imported first decides what the other one runs.
SHARED_KERNEL_NAMES = (
    (("b42_qmv_kernel", ("qmv_port", None)), ("ironmule.qmv_k3840", ("qmv_port", None))),
    (("b42_qmv_kernel", ("qmv_k3840", 3840)), ("ironmule.qmv_k3840", ("qmv_k3840", 3840))),
    (("b45_shared_weight_kernel", (2,)), ("ironmule.qmv_shared", (2,)),
     ("b49_quad_kernel", (2,))),
    (("b45_shared_weight_kernel", (4,)), ("ironmule.qmv_shared", (4,)),
     ("b49_quad_kernel", (4,))),
    (("b48_fast_shared_kernel", (4096, 2)), ("ironmule.qmv_fast_shared", (4096, 2))),
    (("b48_fast_shared_kernel", (15360, 2)), ("ironmule.qmv_fast_shared", (15360, 2))),
)


@pytest.mark.parametrize("group", SHARED_KERNEL_NAMES)
def test_one_metal_kernel_name_carries_one_source(group) -> None:
    """Sharing a kernel name is only safe while the sources are byte-identical."""

    seen = {}
    for module_name, args in group:
        module = importlib.import_module(module_name)
        name, body = _captured_source(module, args)
        seen.setdefault(name, {})[module_name] = hashlib.sha256(body.encode()).hexdigest()

    assert len(seen) == 1, f"the group does not share one name: {sorted(seen)}"
    (name, bodies), = seen.items()
    assert len(bodies) == len(group)
    assert len(set(bodies.values())) == 1, f"{name} has several sources: {bodies}"


def test_3840_is_a_whole_number_of_blocks() -> None:
    """The specialisation is only exact because there is no partial block."""

    assert 3840 % BLOCK_SIZE == 0
    assert 3840 // BLOCK_SIZE == 15


def test_the_shape_constant_is_built_once_per_geometry() -> None:
    first = shape_array(3840, 4096)
    again = shape_array(3840, 4096)
    other = shape_array(3840, 2048)

    assert first is again
    assert other is not first


ROOT = Path(__file__).resolve().parents[1]
DUMPS = ROOT / ".friday-data" / "b53-dumps"


def _first_difference(got: bytes, expected: bytes) -> dict:
    for index, (a, b) in enumerate(zip(got, expected)):
        if a != b:
            return {"index": index, "got": a, "expected": b}
    return {"index": min(len(got), len(expected)), "got": None, "expected": None,
            "note": "one output is shorter than the other"}


def _kernel_identities() -> dict:
    """What was actually registered under each kernel name in this process."""

    captured = {}
    real = mx.fast.metal_kernel

    def record(**kwargs):
        body = kwargs["source"] + "\n--header--\n" + (kwargs.get("header") or "")
        captured[kwargs["name"]] = {
            "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "input_names": list(kwargs.get("input_names") or ()),
            "compile_options": kwargs.get("compile_options"),
        }
        return object()

    module = importlib.import_module("b42_qmv_kernel")
    try:
        mx.fast.metal_kernel = record
        module._kernel("qmv_port", None)
        module._kernel("qmv_k3840", 3840)
    finally:
        mx.fast.metal_kernel = real
    return captured


def _dump(arm: str, out_features: int, arrays: dict, difference: dict) -> Path:
    """Keep everything a mismatch would need, before anything can be re-drawn.

    A rare disagreement has to survive the run that saw it, so this is written outside
    the published tree, once, under a name no other process can take.
    """

    import mlx_lm

    DUMPS.mkdir(parents=True, exist_ok=True)
    stamp = f"{time.time_ns()}-{os.getpid()}-{arm}-{out_features}"
    mx.savez(str(DUMPS / f"{stamp}.npz"), **arrays)
    meta = {
        "arm": arm,
        "out_features": out_features,
        "in_features": 3840,
        "first_difference": difference,
        "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__,
        "pid": os.getpid(),
        "worker": os.environ.get("PYTEST_XDIST_WORKER", "none"),
        "kernels": _kernel_identities(),
        "compile_options_passed": None,
        "compile_options_declared": COMPILE_OPTIONS,
    }
    path = DUMPS / f"{stamp}.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _require_the_gpu_default() -> None:
    """This file compares a library call against a Metal kernel, so it must say where.

    `mx.quantized_matmul` follows the process-global default device; a custom Metal
    kernel can only run on the GPU. If something earlier in the process left the default
    on the CPU, the two arms are computing on different devices and their outputs are not
    supposed to match. B53 was exactly that, and it looked like a kernel defect for days.
    Fail here, loudly, rather than let it look like one again.
    """

    device = str(mx.default_device())
    assert "gpu" in device, (
        f"the default device is {device}; something in this process left it there. "
        "These comparisons are only meaningful with the library and the kernel on the "
        "same device"
    )


def _library(weight, scales, biases, x):
    _require_the_gpu_default()
    return mx.quantized_matmul(x, weight, scales, biases, transpose=True,
                               group_size=64, bits=4)


@pytest.mark.parametrize("out_features", WIDTHS)
def test_fresh_inputs_agree_and_the_library_agrees_with_itself(out_features: int) -> None:
    """The B53 probe: fresh draws, a library-against-library control, exact bytes.

    Three draws per width per run. The control answers what the original failure left
    open, namely whether the reference itself was stable. A mismatch in any arm dumps its
    inputs and outputs before the test fails.
    """

    for draw in range(3):
        dense = mx.random.normal((out_features, 3840)).astype(mx.bfloat16)
        weight, scales, biases = mx.quantize(dense, group_size=64, bits=4)
        x = mx.random.normal((1, 3840)).astype(mx.bfloat16)
        mx.eval(weight, scales, biases, x)

        first = _library(weight, scales, biases, x)
        second = _library(weight, scales, biases, x)
        ported = run(PORT, weight, scales, biases, x, out_features, 3840)
        special = run(K3840, weight, scales, biases, x, out_features, 3840)
        mx.eval(first, second, ported, special)

        expected = bytes(memoryview(first))
        arms = {"library_against_library": bytes(memoryview(second)),
                "port_against_library": bytes(memoryview(ported)),
                "k3840_against_library": bytes(memoryview(special))}
        for arm, produced in arms.items():
            if produced == expected:
                continue
            path = _dump(arm, out_features,
                         {"weight": weight, "scales": scales, "biases": biases, "x": x,
                          "library_first": first, "library_second": second,
                          "ported": ported, "special": special},
                         _first_difference(produced, expected))
            raise AssertionError(f"{arm} differs on draw {draw}; dump at {path}")
