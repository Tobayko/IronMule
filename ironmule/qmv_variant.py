"""The locally qualified `K=3840` geometry, installable from shipped code and gated at run time.

`qmv_k3840` carries the geometry `B42` and `B43` qualified, `(2, 4)`, behind its own
activation hold. `B66`, `B69`, `B75` and `B76` measured a different one, `(4, 8)`, and every
one of those studies installed it from `tools/b66_stack_proof.py` because no shipped path
existed. `B79` is the first time a dispatch may actually use it, and a shipped dispatch cannot
depend on a study tool, so the installer lives here.

**It is the same kernel body.** `qmv_k3840.BODY` and `qmv_k3840.HEADER`, formatted at another
threadgroup geometry, through the same `kernel_registry`. Nothing is re-derived and no source
is copied; if the body changes, this changes with it.

**Every projection proves itself before it is swapped.** A candidate projection is run against
`mx.quantized_matmul` on its own weights and a probe, and is installed only if the two are
byte-identical. A projection that differs is left on the library path and recorded. That check
is the correctness contract `B69` ran in every candidate child, moved to where a product would
run it.

**A gate, not a rebuild.** Installation is expensive and is done once. Whether a given dispatch
may use the kernel is one boolean read on a shared gate, so a workload class the evidence does
not cover pays a comparison rather than a module swap, and a kill switch is immediate for every
request that has not started.

**A shut gate is the reference, not something equivalent to it.** The closed path calls the
original `nn.QuantizedLinear` it replaced. Calling `mx.quantized_matmul` directly would give
the same bytes and, measured here, would be roughly twice as fast as the module -- which is
precisely the reason not to do it. A fallback that is quicker than the thing it falls back to
is a third path, and nothing was ever qualified on a third path.

**Nothing here decides anything.** It installs what it is told to install. What may be
installed, and when, is `ironmule/activation.py`, and that is off by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from . import kernel_registry, qmv_k3840 as qmv

#: The geometry this machine qualified, and the identifier the evidence names it by.
QUALIFIED_GEOMETRY = (4, 8)
QUALIFIED_ACTION_ID = "k3840_geometry_sg4_r8"
CORRECTNESS_CONTRACT = (
    "every admitted projection is byte-identical to mx.quantized_matmul on its own buffers "
    "before it is installed, and a projection that differs is left on the library path")


class VariantUnsupported(RuntimeError):
    """The variant is not admitted here. The library path stays in charge."""


@dataclass
class VariantGate:
    """One mutable boolean the installed projections read, and the reason it is shut.

    Deliberately not frozen: shutting it is the fallback, and the fallback must be able to
    happen without rebuilding anything.
    """

    active: bool = False
    killed: bool = False
    reason: str = "not activated"

    def open(self, reason: str) -> None:
        if self.killed:
            return
        self.active, self.reason = True, reason

    def close(self, reason: str) -> None:
        self.active, self.reason = False, reason

    def kill(self, reason: str) -> None:
        """Permanent for the life of this process. There is no second experimental path."""
        self.active, self.killed, self.reason = False, True, reason


def kernel_for(geometry: tuple[int, int], fixed_k: int = qmv.TARGET_K):
    """The owned kernel body at this geometry, named by the registry from its whole spec."""
    simdgroups, results = geometry
    source = qmv.BODY.format(
        num_simdgroups=simdgroups, results_per_simdgroup=results,
        pack_factor=qmv.PACK_FACTOR, bytes_per_pack=qmv.BYTES_PER_PACK,
        values_per_thread=qmv.VALUES_PER_THREAD, block_size=qmv.BLOCK_SIZE,
        group_size=qmv.GROUP_SIZE, scale_step_per_thread=qmv.SCALE_STEP_PER_THREAD,
        in_vec_size_decl=f"constexpr int in_vec_size = {fixed_k};",
        main_loop=qmv.FIXED_LOOP.replace("15", str(fixed_k // qmv.BLOCK_SIZE)), tail="")
    return kernel_registry.build(
        f"qmv_k{fixed_k}_sg{simdgroups}_r{results}",
        input_names=["w", "scales", "biases", "x", "shape"], output_names=["out"],
        source=source, header=qmv.HEADER, ensure_row_contiguous=True,
        template={"fixed_k": fixed_k, "num_simdgroups": simdgroups,
                  "results_per_simdgroup": results})


class GatedVariantLinear(nn.Module):
    """An admitted projection that uses the variant only while its gate is open."""

    def __init__(self, source: nn.QuantizedLinear, kernel, geometry: tuple[int, int],
                 gate: VariantGate) -> None:
        super().__init__()
        # The original is kept, so a fallback restores a path that was never touched.
        self._source = source
        self._kernel = kernel
        self._gate = gate
        self.weight, self.scales, self.biases = source.weight, source.scales, source.biases
        self.group_size, self.bits = source.group_size, source.bits
        self.out_features = int(source.weight.shape[0])
        self._simdgroups = geometry[0]
        self._groups = self.out_features // (geometry[0] * geometry[1])
        self._shape = qmv.shape_array(qmv.TARGET_K, self.out_features)

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        # One boolean and one shape check, then the *original module*, not a reconstruction
        # of what it does. Calling `mx.quantized_matmul` directly here would be byte-identical
        # and measurably faster, which is exactly why it would be wrong: a shut gate has to
        # be the reference path a user would have had, not a third path nothing qualified.
        if not self._gate.active or len(shape) != 3 or shape[0] != 1 or shape[1] != 1:
            return self._source(x)
        out = self._kernel(
            inputs=[self.weight, self.scales, self.biases, x[0], self._shape],
            output_shapes=[(1, self.out_features)], output_dtypes=[x.dtype],
            grid=(qmv.SIMD_SIZE, self._simdgroups * self._groups, 1),
            threadgroup=(qmv.SIMD_SIZE, self._simdgroups, 1))[0]
        return out[None]


def _identical(module: nn.QuantizedLinear, kernel, geometry: tuple[int, int]) -> bool:
    """The contract, run on this projection's own weights before it is installed."""
    gate = VariantGate(active=True)
    probe = mx.random.normal((1, 1, qmv.TARGET_K)).astype(module.scales.dtype)
    mx.eval(probe)
    want = mx.quantized_matmul(probe, module.weight, module.scales, module.biases,
                               transpose=True, group_size=module.group_size,
                               bits=module.bits)
    got = GatedVariantLinear(module, kernel, geometry, gate)(probe)
    mx.eval(want, got)
    return bytes(memoryview(mx.array(want))) == bytes(memoryview(mx.array(got)))


def install(model: nn.Module, identity: Any,
            geometry: tuple[int, int] = QUALIFIED_GEOMETRY) -> tuple[VariantGate, dict[str, Any]]:
    """Admit the variant on this model, or raise and leave it untouched.

    The gate comes back shut. Opening it is the activation layer's decision and nothing
    here makes it.
    """
    # The same environment gate the qualified module uses: fingerprint, mlx, mlx_lm, model
    # identity and architecture, all checked once before a single module is replaced.
    qmv._admit_environment(identity)
    simdgroups, results = geometry
    rows_per_group = simdgroups * results
    kernel = kernel_for(geometry)
    gate = VariantGate()
    admitted: list[str] = []
    declined: list[str] = []
    mismatched: list[str] = []

    def walk(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.children().items()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, list):
                for index, item in enumerate(child):
                    if isinstance(item, nn.Module):
                        walk(item, f"{path}.{index}")
                continue
            if isinstance(child, nn.QuantizedLinear):
                columns = int(child.weight.shape[1]) * 32 // int(child.bits)
                if columns != qmv.TARGET_K:
                    continue
                if not qmv._admit_projection(child) \
                        or int(child.weight.shape[0]) % rows_per_group:
                    declined.append(path)
                    continue
                if not _identical(child, kernel, geometry):
                    mismatched.append(path)
                    continue
                setattr(parent, name, GatedVariantLinear(child, kernel, geometry, gate))
                admitted.append(path)
            elif isinstance(child, nn.Module):
                walk(child, path)

    walk(model)
    if mismatched:
        # A single byte difference anywhere means this build is not the qualified one.
        # Undo the whole installation rather than keep the projections that happened to
        # agree: a partly-installed model is a state nothing was qualified on.
        uninstall(model)
        raise VariantUnsupported(
            f"{len(mismatched)} projections were not byte-identical to the library; "
            f"nothing is installed")
    if not admitted:
        raise VariantUnsupported("no projection met the admission conditions")
    return gate, {"admitted": len(admitted), "declined": len(declined),
                  "mismatched": len(mismatched), "admitted_paths": admitted,
                  "declined_paths": declined,
                  "geometry": {"num_simdgroups": simdgroups,
                               "results_per_simdgroup": results},
                  "action_id": QUALIFIED_ACTION_ID,
                  "correctness_contract": CORRECTNESS_CONTRACT}


def uninstall(model: nn.Module) -> int:
    """Put every installed projection back. Returns how many were restored."""
    restored = 0

    def walk(parent: nn.Module) -> None:
        nonlocal restored
        for name, child in list(parent.children().items()):
            if isinstance(child, list):
                for item in child:
                    if isinstance(item, nn.Module):
                        walk(item)
                continue
            if isinstance(child, GatedVariantLinear):
                setattr(parent, name, child._source)
                restored += 1
            elif isinstance(child, nn.Module):
                walk(child)

    walk(model)
    return restored


def _self_check() -> None:
    """The gate's own smallest proof, without a model: shut means shut, killed means killed."""
    gate = VariantGate()
    assert not gate.active and not gate.killed
    gate.open("eligible")
    assert gate.active and gate.reason == "eligible"
    gate.close("not eligible for this workload class")
    assert not gate.active and not gate.killed
    gate.open("eligible again")
    assert gate.active
    gate.kill("a fallback was recorded in the candidate path")
    assert not gate.active and gate.killed
    gate.open("anything at all")
    assert not gate.active, "a killed gate never reopens in this process"
    print("qmv_variant self-check passed")


if __name__ == "__main__":
    _self_check()
