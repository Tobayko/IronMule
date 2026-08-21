#!/usr/bin/env python3
"""Validate and isolate model-generated execution plans.

A language model writes a small function body; this module decides whether it may
run at all, and then runs it in a separate process that cannot outlive its
timeout.  Two independent layers, because either one alone is a single point of
failure:

  1. **Static AST allowlist.** Only a fixed set of node types, names and MLX
     attributes survive. Everything else -- imports, attribute access to dunders,
     subscripting into internals, comprehension over arbitrary iterables, lambdas,
     any call to a name that is not explicitly allowed -- is rejected before a
     single byte is executed.
  2. **Process isolation.** What passes runs in a fresh subprocess with a hard
     timeout, a scrubbed environment and CPU/memory limits, so a plan that passes
     validation but hangs or allocates without bound still cannot take the machine
     with it.

The allowlist is deliberately small.  A plan that needs something not on it is
rejected, and that is the correct outcome: the point is measuring execution-plan
variants of one fixed computation, not running arbitrary code.
"""

from __future__ import annotations

import ast
import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Names the plan body may reference.  Nothing else resolves.
ALLOWED_NAMES = frozenset({"mx", "a", "operands", "out", "result", "chunk", "i", "n", "x", "b"})

# MLX operations a plan may use.  Chosen to span execution-plan variation
# (ordering, batching, layout) without opening general compute.
ALLOWED_MX_ATTRS = frozenset(
    {
        "matmul", "eval", "synchronize", "concatenate", "stack", "split",
        "transpose", "contiguous", "array", "zeros", "addmm", "einsum",
    }
)

ALLOWED_BUILTINS = frozenset({"range", "len", "enumerate", "list"})

# Names that may hold a local list, and the methods they may be sent.
# Needed because the obvious correct plan builds a list and evaluates it in one
# go -- rejecting `out.append(x)` rejected the very optimization being searched
# for.  Restricted to known local accumulator names and two mutators, so this
# cannot become a general attribute-access escape hatch.
ALLOWED_LIST_TARGETS = frozenset({"out", "result", "chunk"})
ALLOWED_LIST_METHODS = frozenset({"append", "extend"})

ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
    ast.Expr, ast.Assign, ast.AugAssign, ast.For, ast.Pass,
    ast.Name, ast.Load, ast.Store, ast.Attribute, ast.Call, ast.keyword,
    ast.Constant, ast.List, ast.Tuple, ast.Subscript, ast.Slice, ast.Index
    if hasattr(ast, "Index") else ast.Slice,
    ast.ListComp, ast.comprehension, ast.BinOp, ast.Add, ast.Sub, ast.Mult,
    ast.FloorDiv, ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq,
)

MAX_SOURCE_BYTES = 2000
MAX_AST_NODES = 200
PLAN_TIMEOUT_S = 30.0
# CPU-time ceiling, independent of wall clock: a plan that spins is killed by the
# kernel even if it never hits the subprocess timeout.
PLAN_CPU_SECONDS = 60
PLAN_MEMORY_LIMIT_BYTES = 8 * 1024**3


class PlanRejected(ValueError):
    """Raised when a generated plan may not run."""


def validate_plan_source(source: str) -> ast.Module:
    """Parse and allowlist-check a plan body, or refuse it.

    Fails closed on everything not explicitly permitted.  The checks are ordered
    cheapest-first so a hostile input costs as little as possible.
    """

    if not isinstance(source, str) or not source.strip():
        raise PlanRejected("empty plan")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise PlanRejected("plan exceeds the size limit")
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise PlanRejected(f"plan does not parse: {type(exc).__name__}") from exc

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise PlanRejected("plan is too complex")

    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(tree.body) != 1 or not functions:
        raise PlanRejected("plan must be exactly one function definition")
    function = functions[0]
    if function.name != "plan":
        raise PlanRejected("the function must be named plan")
    arg_names = [arg.arg for arg in function.args.args]
    if arg_names != ["mx", "a", "operands"]:
        raise PlanRejected("plan must take exactly (mx, a, operands)")
    if function.args.vararg or function.args.kwarg or function.args.defaults:
        raise PlanRejected("plan may not use varargs, kwargs or defaults")
    if function.decorator_list:
        raise PlanRejected("plan may not be decorated")

    for node in nodes:
        if not isinstance(node, ALLOWED_NODES):
            raise PlanRejected(f"forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            # Two shapes only: mx.<allowed op>, and <known list>.append/extend.
            # No chained access, no dunders, nothing else.
            if not isinstance(node.value, ast.Name):
                raise PlanRejected("attribute access is limited to mx.<operation>")
            owner = node.value.id
            if owner == "mx":
                if node.attr not in ALLOWED_MX_ATTRS:
                    raise PlanRejected(f"mx.{node.attr} is not on the allowlist")
            elif owner in ALLOWED_LIST_TARGETS:
                if node.attr not in ALLOWED_LIST_METHODS:
                    raise PlanRejected(f"{owner}.{node.attr} is not on the allowlist")
            else:
                raise PlanRejected("attribute access is limited to mx.<operation>")
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise PlanRejected("names starting with underscore are forbidden")
            if node.id not in ALLOWED_NAMES | ALLOWED_BUILTINS:
                raise PlanRejected(f"unknown name: {node.id}")
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                if target.id not in ALLOWED_BUILTINS:
                    raise PlanRejected(f"calling {target.id} is not allowed")
            elif not isinstance(target, ast.Attribute):
                raise PlanRejected("only direct calls are allowed")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Strings are the usual carrier for smuggled code; plans need none.
            raise PlanRejected("string literals are not allowed in a plan")
    return tree


_WORKER = '''
import json, sys, time, resource
sys.path.insert(0, {root!r})
# macOS refuses RLIMIT_AS even when the hard limit is unlimited, so the address
# space is not the lever here.  RLIMIT_CPU does work and bounds a runaway loop
# independently of the wall-clock timeout; MLX's own limit bounds GPU memory,
# which is the allocation that actually matters for a plan.
try:
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
except (ValueError, OSError):
    pass
import mlx.core as mx
try:
    mx.set_memory_limit({mem})
except Exception:
    pass
import numpy as np
from friday_h0.benchmark import _generate_fixture

exec(compile(open({plan_file!r}).read(), "<plan>", "exec"), globals())

fixture = _generate_fixture(np, {seed})
a = mx.array(fixture.a)
mx.eval(a)
rng = np.random.Generator(np.random.PCG64({pool_seed}))
operands = [mx.array(rng.uniform(-1.0, 1.0, (2048, 2048)).astype(np.float16))
            for _ in range({n})]
mx.eval(*operands); mx.synchronize()
references = [np.array(mx.matmul(a, o), copy=False).astype(np.float32) for o in operands]

def baseline():
    out = []
    for o in operands:
        v = mx.matmul(a, o); mx.eval(v); mx.synchronize(); out.append(v)
    return out

produced = plan(mx, a, operands)
mx.eval(produced); mx.synchronize()
if not isinstance(produced, (list, tuple)) or len(produced) != {n}:
    print(json.dumps({{"ok": False, "reason": "plan did not return one result per operand"}}))
    raise SystemExit(0)
worst = 0.0
for value, reference in zip(produced, references):
    worst = max(worst, float(np.abs(np.array(value, copy=False).astype(np.float32) - reference).max()))
if worst != 0.0:
    print(json.dumps({{"ok": False, "reason": "plan changed the result", "deviation": worst}}))
    raise SystemExit(0)

for _ in range(3):
    baseline(); plan(mx, a, operands); mx.synchronize()

import math, statistics
ratios = []
for _ in range({blocks}):
    t = time.perf_counter_ns(); baseline(); base_ns = time.perf_counter_ns() - t
    t = time.perf_counter_ns()
    v = plan(mx, a, operands); mx.eval(v); mx.synchronize()
    plan_ns = time.perf_counter_ns() - t
    ratios.append(math.log(plan_ns / base_ns))
print(json.dumps({{"ok": True, "log_ratios": ratios}}))
'''


def run_plan_isolated(
    source: str,
    *,
    n: int,
    blocks: int,
    fixture_seed: int,
    pool_seed: int,
    timeout: float = PLAN_TIMEOUT_S,
) -> dict:
    """Run an already validated plan in a separate, limited process.

    The caller must have validated the source first; this function assumes that
    and adds the second layer -- a process that cannot outlive its timeout and
    cannot allocate past its limit.
    """

    with tempfile.TemporaryDirectory() as directory:
        plan_file = Path(directory) / "plan.py"
        plan_file.write_text(source, encoding="utf-8")
        script = _WORKER.format(
            root=str(PROJECT_ROOT),
            mem=PLAN_MEMORY_LIMIT_BYTES,
            cpu=PLAN_CPU_SECONDS,
            plan_file=str(plan_file),
            seed=fixture_seed,
            pool_seed=pool_seed,
            n=n,
            blocks=blocks,
        )
        script_file = Path(directory) / "worker.py"
        script_file.write_text(script, encoding="utf-8")
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": directory,
            "TMPDIR": directory,
        }
        try:
            completed = subprocess.run(
                [sys.executable, str(script_file)],
                cwd=directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout,
                check=False,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "plan exceeded its time limit"}
        if completed.returncode != 0:
            tail = completed.stderr.decode("utf-8", errors="replace").strip()[-200:]
            return {"ok": False, "reason": "plan process failed", "detail": tail}
        try:
            return json.loads(completed.stdout.decode("utf-8", errors="replace").strip())
        except (ValueError, TypeError):
            return {"ok": False, "reason": "plan produced unreadable output"}


__all__ = [
    "ALLOWED_BUILTINS",
    "ALLOWED_LIST_METHODS",
    "ALLOWED_MX_ATTRS",
    "PlanRejected",
    "run_plan_isolated",
    "validate_plan_source",
]
