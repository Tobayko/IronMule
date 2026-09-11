"""One Metal kernel name per specification, and never two specifications per name.

MLX caches compiled custom kernels by name — `Device::get_library(name_, …)` in
`backend/metal/custom_kernel.cpp` — and its stale-source invalidation works across `eval`
boundaries but not inside one batch (ml-explore/mlx#3832, affecting `0.31.1` through
`0.32.0`; reproduced here on `0.32.0`). Two kernels registered under one name with
different sources therefore both run whichever compiled first, silently and without an
error, whenever both are dispatched before the same `eval`.

So the name is not chosen here, it is derived. The identifier is a digest over the whole
specification that decides what gets compiled: the base name, the source, the header, the
input and output names, the row-contiguity and atomic flags, the compile options and any
template values. Two specifications cannot then share an identifier, and one identifier
cannot mean two specifications.

The digest is computed once per specialisation, at the point the kernel is built, which
is module import for every caller in this repository. Nothing here runs per call, per
token or per request.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import mlx.core as mx

IDENTIFIER_LENGTH = 16          # 64 bits of the digest, enough to separate a few kernels


class KernelSpecificationConflict(RuntimeError):
    """One identifier was asked to stand for two different specifications."""


_REGISTRY: dict[str, dict[str, Any]] = {}


def specification(base: str, *, source: str, header: str = "",
                  input_names: Sequence[str], output_names: Sequence[str],
                  ensure_row_contiguous: bool = True, atomic_outputs: bool = False,
                  compile_options: Mapping[str, Any] | None = None,
                  template: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Everything that decides what MLX compiles, in one canonical mapping."""

    return {
        "base": base,
        "source": source,
        "header": header,
        "input_names": list(input_names),
        "output_names": list(output_names),
        "ensure_row_contiguous": bool(ensure_row_contiguous),
        "atomic_outputs": bool(atomic_outputs),
        "compile_options": dict(compile_options) if compile_options else None,
        "template": dict(template) if template else None,
    }


def identifier(spec: Mapping[str, Any]) -> str:
    """The kernel name for a specification. Same specification, same name; never else."""

    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:IDENTIFIER_LENGTH]
    return f"{spec['base']}_{digest}"


def build(base: str, *, source: str, header: str = "",
          input_names: Sequence[str], output_names: Sequence[str],
          ensure_row_contiguous: bool = True, atomic_outputs: bool = False,
          compile_options: Mapping[str, Any] | None = None,
          template: Mapping[str, Any] | None = None):
    """Register a kernel under a name derived from its whole specification.

    Building the same specification twice returns a kernel with the same name, which is
    the case MLX handles correctly. Building a different specification always yields a
    different name, which is the case MLX gets wrong.
    """

    spec = specification(base, source=source, header=header, input_names=input_names,
                         output_names=output_names,
                         ensure_row_contiguous=ensure_row_contiguous,
                         atomic_outputs=atomic_outputs, compile_options=compile_options,
                         template=template)
    name = identifier(spec)
    seen = _REGISTRY.get(name)
    if seen is not None and seen != spec:
        raise KernelSpecificationConflict(
            f"{name} already stands for a different specification"
        )
    _REGISTRY[name] = spec
    arguments: dict[str, Any] = {
        "name": name,
        "input_names": list(input_names),
        "output_names": list(output_names),
        "source": source,
        "header": header,
        "ensure_row_contiguous": ensure_row_contiguous,
    }
    if atomic_outputs:
        arguments["atomic_outputs"] = True
    if compile_options:
        arguments["compile_options"] = dict(compile_options)
    return mx.fast.metal_kernel(**arguments)


def registered() -> dict[str, dict[str, Any]]:
    """What this process has registered so far. A copy; the registry stays private."""

    return {name: dict(spec) for name, spec in _REGISTRY.items()}
