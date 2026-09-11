"""One Metal kernel name per specification, checked against the bug it exists for.

MLX caches compiled custom kernels by name and, inside a single `eval` batch, does not
notice that the source changed (ml-explore/mlx#3832, `0.31.1` to `0.32.0`). The first
test here reproduces that on this machine with hand-named kernels, so the rest of the
file is measuring against a demonstrated failure rather than a described one.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ironmule import kernel_registry as kr

ADD_ONE = "uint i = thread_position_in_grid.x; out[i] = a[i] + 1.0f;"
ADD_HUNDRED = "uint i = thread_position_in_grid.x; out[i] = a[i] + 100.0f;"


def _call(kernel):
    zeros = mx.zeros((4,), dtype=mx.float32)
    return kernel(inputs=[zeros], output_shapes=[(4,)], output_dtypes=[mx.float32],
                  grid=(4, 1, 1), threadgroup=(4, 1, 1))[0]


def _named(name: str, source: str):
    return mx.fast.metal_kernel(name=name, input_names=["a"], output_names=["out"],
                                source=source, ensure_row_contiguous=True)


def _built(source: str, **extra):
    return kr.build("registry_probe", source=source, input_names=["a"],
                    output_names=["out"], **extra)


def test_the_collision_this_guards_against_is_real_here() -> None:
    """Before: one hand-picked name, two sources, both run the first one."""

    first = _call(_named("registry_collision_probe", ADD_ONE))
    second = _call(_named("registry_collision_probe", ADD_HUNDRED))
    mx.eval(first, second)

    assert first.tolist() == [1.0] * 4
    assert second.tolist() == [1.0] * 4, (
        "MLX no longer collides on name; the registry can then be simplified away"
    )


def test_two_sources_in_one_evaluation_keep_their_own_results() -> None:
    """After: the same two sources, dispatched before one `eval`, stay separate."""

    first = _call(_built(ADD_ONE))
    second = _call(_built(ADD_HUNDRED))
    mx.eval(first, second)

    assert first.tolist() == [1.0] * 4
    assert second.tolist() == [100.0] * 4


def test_the_order_of_registration_does_not_decide_the_result() -> None:
    second = _call(_built(ADD_HUNDRED))
    first = _call(_built(ADD_ONE))
    mx.eval(second, first)

    assert first.tolist() == [1.0] * 4
    assert second.tolist() == [100.0] * 4


def test_the_same_source_reuses_one_identifier() -> None:
    """Rebuilding a specialisation must not invent a second kernel for it."""

    one = kr.identifier(kr.specification("probe", source=ADD_ONE, input_names=["a"],
                                         output_names=["out"]))
    again = kr.identifier(kr.specification("probe", source=ADD_ONE, input_names=["a"],
                                           output_names=["out"]))

    assert one == again
    assert _call(_built(ADD_ONE)).tolist() == _call(_built(ADD_ONE)).tolist()


@pytest.mark.parametrize("difference", [
    {"header": "// a header"},
    {"compile_options": {"math_mode": "safe"}},
    {"template": {"width": 2}},
    {"ensure_row_contiguous": False},
])
def test_anything_that_changes_what_is_compiled_changes_the_identifier(difference) -> None:
    """Equal source bytes are not equal specifications."""

    base = dict(source=ADD_ONE, input_names=["a"], output_names=["out"])
    plain = kr.identifier(kr.specification("probe", **base))
    altered = kr.identifier(kr.specification("probe", **base, **difference))

    assert plain != altered


def test_the_identifier_is_not_random_and_not_per_call() -> None:
    """The name has to be stable across processes, or nothing is cached at all."""

    spec = kr.specification("probe", source=ADD_ONE, input_names=["a"],
                            output_names=["out"])

    assert kr.identifier(spec) == kr.identifier(dict(spec))
    assert kr.identifier(spec).startswith("probe_")
    assert len(kr.identifier(spec)) == len("probe_") + kr.IDENTIFIER_LENGTH


def test_one_identifier_cannot_stand_for_two_specifications() -> None:
    """The registry refuses rather than handing back someone else's kernel."""

    spec = kr.specification("registry_probe", source=ADD_ONE, input_names=["a"],
                            output_names=["out"])
    name = kr.identifier(spec)
    kr._REGISTRY[name] = {**spec, "source": ADD_HUNDRED}
    try:
        with pytest.raises(kr.KernelSpecificationConflict):
            _built(ADD_ONE)
    finally:
        kr._REGISTRY.pop(name, None)


def test_every_kernel_this_repository_builds_has_a_derived_name() -> None:
    """No caller may go around the registry with a name of its own choosing."""

    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    modules = ("ironmule.qmv_k3840", "ironmule.qmv_shared", "ironmule.qmv_fast_shared",
               "b42_qmv_kernel", "b45_shared_weight_kernel", "b49_quad_kernel",
               "b48_fast_shared_kernel")
    for name in modules:
        importlib.import_module(name)

    registered = kr.registered()
    assert registered, "importing the kernel modules registered nothing"
    for name, spec in registered.items():
        assert name == kr.identifier(spec), name
