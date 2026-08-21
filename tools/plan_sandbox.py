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
     timeout, a scrubbed environment and a kernel CPU limit.  The accepted
     language itself bounds allocation shapes; MLX's memory setting is only an
     additional best-effort guideline, not an OS memory limit.

The allowlist is deliberately small.  A plan that needs something not on it is
rejected, and that is the correct outcome: the point is measuring execution-plan
variants of one fixed computation, not running arbitrary code.
"""

from __future__ import annotations

import ast
import json
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
        "matmul", "eval", "synchronize",
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
MAX_INTEGER_LITERAL = 16
MAX_WEIGHTED_MATMULS = 32
PLAN_TIMEOUT_S = 30.0
# CPU-time ceiling, independent of wall clock: a plan that spins is killed by the
# kernel even if it never hits the subprocess timeout.
PLAN_CPU_SECONDS = 25
PLAN_MEMORY_LIMIT_BYTES = 8 * 1024**3
PLAN_CONTINUOUS_GPU_SECONDS = 6.0


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
    if (
        function.args.posonlyargs
        or function.args.vararg
        or function.args.kwarg
        or function.args.kwonlyargs
        or function.args.defaults
        or function.args.kw_defaults
    ):
        raise PlanRejected(
            "plan may not use positional-only, keyword-only or default arguments"
        )
    if function.decorator_list or function.returns or any(
        argument.annotation for argument in function.args.args
    ):
        raise PlanRejected("plan may not use decorators or annotations")

    parents = {
        child: parent
        for parent in nodes
        for child in ast.iter_child_nodes(parent)
    }
    loop_types = (ast.For, ast.ListComp)
    for node in nodes:
        if isinstance(node, ast.ListComp) and len(node.generators) != 1:
            raise PlanRejected("plans may use only one generator per comprehension")
        if isinstance(node, loop_types):
            ancestor = parents.get(node)
            while ancestor is not None:
                if isinstance(ancestor, loop_types):
                    raise PlanRejected("nested iteration is not allowed")
                ancestor = parents.get(ancestor)

    def bounded_iterable(value: ast.AST) -> bool:
        if isinstance(value, ast.Name) and value.id == "operands":
            return True
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            return value.func.id in {"range", "enumerate"}
        return False

    for node in nodes:
        if isinstance(node, ast.For) and not bounded_iterable(node.iter):
            raise PlanRejected("for-loops must iterate over operands or a bounded range")
        if isinstance(node, ast.comprehension) and not bounded_iterable(node.iter):
            raise PlanRejected(
                "comprehensions must iterate over operands or a bounded range"
            )

    weighted_matmuls = 0
    for node in nodes:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "mx"
            and node.func.attr == "matmul"
        ):
            ancestor = parents.get(node)
            inside_iteration = False
            while ancestor is not None:
                if isinstance(ancestor, loop_types):
                    inside_iteration = True
                    break
                ancestor = parents.get(ancestor)
            weighted_matmuls += MAX_INTEGER_LITERAL if inside_iteration else 1
    if weighted_matmuls > MAX_WEIGHTED_MATMULS:
        raise PlanRejected("plan exceeds the bounded matmul allocation budget")

    def numeric_expression(value: ast.AST) -> bool:
        if isinstance(value, ast.Constant):
            return type(value.value) is int and 0 <= value.value <= MAX_INTEGER_LITERAL
        if isinstance(value, ast.Name):
            return value.id in {"i", "n"}
        if isinstance(value, ast.Call):
            return (
                isinstance(value.func, ast.Name)
                and value.func.id == "len"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.Name)
                and value.args[0].id == "operands"
                and not value.keywords
            )
        if isinstance(value, ast.BinOp):
            return numeric_expression(value.left) and numeric_expression(value.right)
        return False

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
            if isinstance(node.ctx, ast.Store) and node.id in {"mx", "a", "operands"}:
                raise PlanRejected("input bindings may not be reassigned")
        if isinstance(node, ast.AugAssign):
            raise PlanRejected("augmented assignment is not allowed")
        if isinstance(node, ast.BinOp) and not numeric_expression(node):
            raise PlanRejected("binary arithmetic is limited to bounded indices")
        if isinstance(node, ast.Compare) and not all(
            numeric_expression(value) for value in (node.left, *node.comparators)
        ):
            raise PlanRejected("comparisons are limited to bounded indices")
        if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) > MAX_INTEGER_LITERAL:
            raise PlanRejected("literal sequence exceeds the allocation bound")
        if isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name) or node.value.id != "operands":
                raise PlanRejected("subscripting is limited to operands")
            if isinstance(node.slice, ast.Slice):
                bounds = (node.slice.lower, node.slice.upper, node.slice.step)
                if any(
                    bound is not None and not numeric_expression(bound)
                    for bound in bounds
                ):
                    raise PlanRejected("slice bounds must use bounded indices")
            elif not numeric_expression(node.slice):
                raise PlanRejected("operand indices must be bounded")
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                if target.id not in ALLOWED_BUILTINS:
                    raise PlanRejected(f"calling {target.id} is not allowed")
                if target.id == "range":
                    if not 1 <= len(node.args) <= 3 or node.keywords:
                        raise PlanRejected("range must use one to three bounded arguments")
                    for argument in node.args:
                        is_small_integer = (
                            isinstance(argument, ast.Constant)
                            and type(argument.value) is int
                            and 0 <= argument.value <= MAX_INTEGER_LITERAL
                        )
                        is_operand_length = (
                            isinstance(argument, ast.Call)
                            and isinstance(argument.func, ast.Name)
                            and argument.func.id == "len"
                            and len(argument.args) == 1
                            and isinstance(argument.args[0], ast.Name)
                            and argument.args[0].id == "operands"
                            and not argument.keywords
                        )
                        if not (is_small_integer or is_operand_length):
                            raise PlanRejected("range arguments must be bounded by operands")
                elif target.id == "len":
                    if (
                        len(node.args) != 1
                        or node.keywords
                        or not isinstance(node.args[0], ast.Name)
                        or node.args[0].id != "operands"
                    ):
                        raise PlanRejected("len is limited to operands")
                elif target.id == "enumerate":
                    if (
                        len(node.args) != 1
                        or node.keywords
                        or not isinstance(node.args[0], ast.Name)
                        or node.args[0].id != "operands"
                    ):
                        raise PlanRejected("enumerate is limited to operands")
                elif target.id == "list" and (len(node.args) > 1 or node.keywords):
                    raise PlanRejected("list accepts at most one positional argument")
            elif not isinstance(target, ast.Attribute):
                raise PlanRejected("only direct calls are allowed")
            else:
                if node.keywords:
                    raise PlanRejected("keyword arguments are not allowed")
                if isinstance(target.value, ast.Name) and target.value.id == "mx":
                    if target.attr not in ALLOWED_MX_ATTRS:
                        raise PlanRejected(f"mx.{target.attr} is not on the allowlist")
                    arity = {"matmul": 2, "eval": 1, "synchronize": 0}[target.attr]
                    if len(node.args) != arity:
                        raise PlanRejected(f"mx.{target.attr} has a fixed arity of {arity}")
                elif len(node.args) != 1:
                    raise PlanRejected("list mutators take exactly one argument")
                elif target.attr == "extend" and not isinstance(
                    node.args[0], (ast.List, ast.Tuple)
                ):
                    raise PlanRejected("extend accepts only a bounded literal sequence")
        if isinstance(node, ast.Constant):
            if type(node.value) is not int or not 0 <= node.value <= MAX_INTEGER_LITERAL:
                raise PlanRejected("only small nonnegative integer literals are allowed")
    return tree


_WORKER = '''
import json, sys, time, resource
sys.path.insert(0, {root!r})
# macOS refuses RLIMIT_AS even when the hard limit is unlimited, so the address
# space is not the lever here.  RLIMIT_CPU does work and bounds a runaway loop
# independently of the wall-clock timeout. MLX documents set_memory_limit as a
# guideline, not a hard allocation ceiling; allocation shapes are therefore
# bounded by the validated language above and this setting is defense in depth.
try:
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
except (ValueError, OSError):
    print(json.dumps({{"ok": False, "fatal": True, "reason": "CPU limit unavailable"}}))
    raise SystemExit(0)
import mlx.core as mx
try:
    mx.set_memory_limit({mem})
except Exception:
        print(json.dumps({{"ok": False, "fatal": True, "reason": "MLX memory guideline unavailable"}}))
    raise SystemExit(0)
import numpy as np
from friday_h0.benchmark import _generate_fixture
from tools.plan_sandbox import validate_plan_source

gpu_work = [0.0]
def charge_gpu(start_ns):
    elapsed = (time.perf_counter_ns() - start_ns) / 1e9
    gpu_work[0] += elapsed
    if gpu_work[0] > {continuous_gpu}:
        print(json.dumps({{"ok": False, "fatal": True,
                          "reason": "continuous GPU budget exceeded",
                          "gpu_work_seconds": gpu_work[0]}}))
        raise SystemExit(0)
    return elapsed

with open({plan_file!r}, encoding="utf-8") as handle:
    plan_source = handle.read()
validate_plan_source(plan_source)
plan_globals = {{
    "__builtins__": {{
        "range": range, "len": len, "enumerate": enumerate, "list": list
    }}
}}
exec(compile(plan_source, "<plan>", "exec"), plan_globals)
plan = plan_globals["plan"]

fixture = _generate_fixture(np, {seed})
a = mx.array(fixture.a)
t = time.perf_counter_ns(); mx.eval(a); mx.synchronize(); charge_gpu(t)
rng = np.random.Generator(np.random.PCG64({pool_seed}))
operands = [mx.array(rng.uniform(-1.0, 1.0, (2048, 2048)).astype(np.float16))
            for _ in range({n})]
t = time.perf_counter_ns(); mx.eval(*operands); mx.synchronize(); charge_gpu(t)
t = time.perf_counter_ns()
references = [np.array(mx.matmul(a, o), copy=False).astype(np.float32) for o in operands]
charge_gpu(t)

def baseline():
    out = []
    for o in operands:
        v = mx.matmul(a, o); mx.eval(v); mx.synchronize(); out.append(v)
    return out

t = time.perf_counter_ns()
produced = plan(mx, a, operands)
mx.eval(produced); mx.synchronize(); charge_gpu(t)
if not isinstance(produced, (list, tuple)) or len(produced) != {n}:
    print(json.dumps({{"ok": False, "fatal": False,
                      "reason": "plan did not return one result per operand",
                      "gpu_work_seconds": gpu_work[0]}}))
    raise SystemExit(0)
worst = 0.0
for value, reference in zip(produced, references):
    worst = max(worst, float(np.abs(np.array(value, copy=False).astype(np.float32) - reference).max()))
if worst != 0.0:
    print(json.dumps({{"ok": False, "fatal": False, "reason": "plan changed the result",
                      "deviation": worst, "gpu_work_seconds": gpu_work[0]}}))
    raise SystemExit(0)

for _ in range(3):
    t = time.perf_counter_ns(); baseline(); plan(mx, a, operands); mx.synchronize(); charge_gpu(t)

import math, statistics
ratios = []
for _ in range({blocks}):
    t = time.perf_counter_ns(); baseline(); base_ns = time.perf_counter_ns() - t
    gpu_work[0] += base_ns / 1e9
    if gpu_work[0] > {continuous_gpu}:
        print(json.dumps({{"ok": False, "fatal": True,
                          "reason": "continuous GPU budget exceeded",
                          "gpu_work_seconds": gpu_work[0]}})); raise SystemExit(0)
    t = time.perf_counter_ns()
    v = plan(mx, a, operands); mx.eval(v); mx.synchronize()
    plan_ns = time.perf_counter_ns() - t
    gpu_work[0] += plan_ns / 1e9
    if gpu_work[0] > {continuous_gpu}:
        print(json.dumps({{"ok": False, "fatal": True,
                          "reason": "continuous GPU budget exceeded",
                          "gpu_work_seconds": gpu_work[0]}})); raise SystemExit(0)
    ratios.append(math.log(plan_ns / base_ns))
print(json.dumps({{"ok": True, "fatal": False, "log_ratios": ratios,
                  "gpu_work_seconds": gpu_work[0]}}))
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

    Validation is repeated here even when the caller already performed it.  The
    process layer then adds a worker that cannot outlive its timeout. Allocation
    shapes are bounded by the validated language; the MLX memory guideline is
    additional defense in depth, not hard containment.
    """

    validate_plan_source(source)
    if type(n) is not int or not 2 <= n <= 16:
        raise PlanRejected("operand count is outside the registered range")
    if type(blocks) is not int or not 2 <= blocks <= 30:
        raise PlanRejected("block count is outside the registered range")
    if type(timeout) not in (int, float) or not 0 < float(timeout) <= PLAN_TIMEOUT_S:
        raise PlanRejected("timeout is outside the registered range")
    for seed in (fixture_seed, pool_seed):
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise PlanRejected("seed is outside the registered range")

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
            continuous_gpu=PLAN_CONTINUOUS_GPU_SECONDS,
        )
        script_file = Path(directory) / "worker.py"
        script_file.write_text(script, encoding="utf-8")
        environment = {
            "PATH": "/usr/bin:/bin",
            "TMPDIR": directory,
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
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
            return {"ok": False, "fatal": True, "reason": "plan exceeded its time limit"}
        if completed.returncode != 0:
            tail = completed.stderr.decode("utf-8", errors="replace").strip()[-200:]
            return {"ok": False, "fatal": True, "reason": "plan process failed", "detail": tail}
        try:
            return json.loads(completed.stdout.decode("utf-8", errors="replace").strip())
        except (ValueError, TypeError):
            return {"ok": False, "fatal": True, "reason": "plan produced unreadable output"}


__all__ = [
    "ALLOWED_BUILTINS",
    "ALLOWED_LIST_METHODS",
    "ALLOWED_MX_ATTRS",
    "PlanRejected",
    "run_plan_isolated",
    "validate_plan_source",
]
