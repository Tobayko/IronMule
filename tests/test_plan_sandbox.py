"""Adversarial checks for the generated-plan validator.

No GPU, no subprocess, no model.  This is the layer that decides whether
model-written code may run at all, so it is tested by trying to break it.
Every payload below must be rejected; a single one getting through would mean
arbitrary code execution from an untrusted source.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "plan_sandbox", PROJECT_ROOT / "tools" / "plan_sandbox.py"
)
sandbox = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sandbox)

VALID = """def plan(mx, a, operands):
    out = [mx.matmul(a, b) for b in operands]
    mx.eval(out)
    mx.synchronize()
    return out
"""


class AcceptsLegitimatePlansTest(unittest.TestCase):
    def test_the_batched_plan_is_accepted(self) -> None:
        self.assertIsNotNone(sandbox.validate_plan_source(VALID))

    def test_a_serial_plan_is_accepted(self) -> None:
        source = """def plan(mx, a, operands):
    out = []
    for b in operands:
        x = mx.matmul(a, b)
        mx.eval(x)
        mx.synchronize()
        out.append(x)
    return out
"""
        self.assertIsNotNone(sandbox.validate_plan_source(source))

    def test_an_indexed_plan_is_accepted(self) -> None:
        source = """def plan(mx, a, operands):
    out = []
    for i in range(len(operands)):
        x = mx.matmul(a, operands[i])
        out.append(x)
    mx.eval(out)
    mx.synchronize()
    return out
"""
        self.assertIsNotNone(sandbox.validate_plan_source(source))


class RejectsCodeExecutionTest(unittest.TestCase):
    """Each payload is a way someone might try to get arbitrary code to run."""

    def reject(self, source: str) -> str:
        with self.assertRaises(sandbox.PlanRejected, msg=source) as caught:
            sandbox.validate_plan_source(source)
        return str(caught.exception)

    def test_imports_are_rejected(self) -> None:
        for payload in (
            "import os\ndef plan(mx, a, operands):\n    return operands\n",
            "def plan(mx, a, operands):\n    import os\n    return operands\n",
            "from os import system\ndef plan(mx, a, operands):\n    return operands\n",
        ):
            with self.subTest(payload=payload[:30]):
                self.reject(payload)

    def test_dunder_access_is_rejected(self) -> None:
        for payload in (
            "def plan(mx, a, operands):\n    return mx.__class__\n",
            "def plan(mx, a, operands):\n    return a.__reduce__()\n",
            "def plan(mx, a, operands):\n    return operands.__class__.__bases__\n",
        ):
            with self.subTest(payload=payload[:40]):
                self.reject(payload)

    def test_builtins_traversal_is_rejected(self) -> None:
        payload = "def plan(mx, a, operands):\n    return mx.matmul.__globals__\n"
        self.reject(payload)

    def test_eval_and_exec_are_rejected(self) -> None:
        for name in ("eval", "exec", "compile", "open", "__import__", "getattr"):
            with self.subTest(builtin=name):
                self.reject(f"def plan(mx, a, operands):\n    return {name}(a)\n")

    def test_string_literals_are_rejected(self) -> None:
        # Strings are the usual carrier for smuggled code; plans need none.
        self.reject('def plan(mx, a, operands):\n    return mx.array("x")\n')

    def test_allocation_primitives_are_not_available(self) -> None:
        for operation in ("zeros", "ones", "array", "empty"):
            with self.subTest(operation=operation):
                self.reject(
                    f"def plan(mx, a, operands):\n    return mx.{operation}((16, 16))\n"
                )

    def test_large_or_unbound_ranges_are_rejected(self) -> None:
        for expression in ("range(17)", "range(1000000000)", "range(n)"):
            with self.subTest(expression=expression):
                self.reject(
                    "def plan(mx, a, operands):\n"
                    "    out = []\n"
                    f"    for i in {expression}:\n"
                    "        out.append(a)\n"
                    "    return out\n"
                )

    def test_input_bindings_cannot_be_reassigned(self) -> None:
        for name in ("mx", "a", "operands"):
            with self.subTest(name=name):
                self.reject(
                    f"def plan(mx, a, operands):\n    {name} = []\n    return operands\n"
                )

    def test_self_growing_or_nested_iteration_is_rejected(self) -> None:
        payloads = (
            """def plan(mx, a, operands):
    out = [a]
    for b in out:
        out.append(b)
    return out
""",
            """def plan(mx, a, operands):
    out = []
    for b in operands:
        for i in range(16):
            out.append(mx.matmul(a, b))
    return out
""",
            """def plan(mx, a, operands):
    out = [a]
    out.extend(out)
    return out
""",
        )
        for payload in payloads:
            with self.subTest(payload=payload[:40]):
                self.reject(payload)

    def test_static_matmul_allocation_budget_is_enforced(self) -> None:
        self.reject(
            """def plan(mx, a, operands):
    out = []
    for b in operands:
        x = mx.matmul(a, b)
        result = mx.matmul(a, b)
        chunk = mx.matmul(a, b)
        out.extend([x, result, chunk])
    return out
"""
        )

    def test_lambda_is_rejected(self) -> None:
        self.reject("def plan(mx, a, operands):\n    return (lambda: 1)()\n")

    def test_unknown_mx_operation_is_rejected(self) -> None:
        message = self.reject(
            "def plan(mx, a, operands):\n    return mx.system(a)\n"
        )
        self.assertIn("allowlist", message)

    def test_attribute_access_on_other_objects_is_rejected(self) -> None:
        self.reject("def plan(mx, a, operands):\n    return a.shape\n")

    def test_unknown_names_are_rejected(self) -> None:
        self.reject("def plan(mx, a, operands):\n    return os\n")

    def test_underscore_names_are_rejected(self) -> None:
        self.reject("def plan(mx, a, operands):\n    _x = a\n    return _x\n")


class RejectsMalformedPlansTest(unittest.TestCase):
    def reject(self, source: str) -> None:
        with self.assertRaises(sandbox.PlanRejected, msg=source):
            sandbox.validate_plan_source(source)

    def test_wrong_function_name_is_rejected(self) -> None:
        self.reject("def run(mx, a, operands):\n    return operands\n")

    def test_wrong_signature_is_rejected(self) -> None:
        for signature in ("mx, a", "mx, a, operands, extra", "a, mx, operands"):
            with self.subTest(signature=signature):
                self.reject(f"def plan({signature}):\n    return operands\n")

    def test_varargs_are_rejected(self) -> None:
        self.reject("def plan(mx, a, operands, *args):\n    return operands\n")

    def test_keyword_only_positional_only_and_annotated_signatures_are_rejected(self) -> None:
        for signature in (
            "mx, a, operands, *, n=1",
            "mx, a, operands, /",
            "mx: list, a, operands",
        ):
            with self.subTest(signature=signature):
                self.reject(f"def plan({signature}):\n    return operands\n")

    def test_extra_top_level_statements_are_rejected(self) -> None:
        self.reject(VALID + "\nplan = None\n")

    def test_decorators_are_rejected(self) -> None:
        self.reject("@staticmethod\ndef plan(mx, a, operands):\n    return operands\n")

    def test_empty_and_nonsense_input_is_rejected(self) -> None:
        for payload in ("", "   ", "not python at all {{{", None, 42):
            with self.subTest(payload=repr(payload)[:20]):
                with self.assertRaises(sandbox.PlanRejected):
                    sandbox.validate_plan_source(payload)

    def test_oversized_source_is_rejected(self) -> None:
        self.reject(VALID + "\n".join("    out = out" for _ in range(500)))

    def test_overly_complex_plan_is_rejected(self) -> None:
        body = "\n".join(f"    out = mx.matmul(a, operands[{i}])" for i in range(80))
        self.reject(f"def plan(mx, a, operands):\n{body}\n    return operands\n")


class AllowlistIntegrityTest(unittest.TestCase):
    def test_no_dangerous_operation_sits_on_the_allowlist(self) -> None:
        # Note on mx.eval: it is MLX's "force the lazy graph" call and has nothing
        # to do with Python's eval.  Python's eval is a bare Name call and is
        # blocked by ALLOWED_BUILTINS instead -- see test_eval_and_exec_are_rejected.
        for forbidden in ("exec", "compile", "system", "load", "save", "import"):
            with self.subTest(op=forbidden):
                self.assertNotIn(forbidden, sandbox.ALLOWED_MX_ATTRS)

    def test_python_eval_is_blocked_even_though_mx_eval_is_allowed(self) -> None:
        # The two must not be confused: one is required, the other must never run.
        self.assertIn("eval", sandbox.ALLOWED_MX_ATTRS)
        self.assertNotIn("eval", sandbox.ALLOWED_BUILTINS)
        with self.assertRaises(sandbox.PlanRejected):
            sandbox.validate_plan_source(
                "def plan(mx, a, operands):\n    return eval(a)\n"
            )

    def test_allowlist_stays_small(self) -> None:
        # Growth here widens the attack surface; it should be a deliberate act.
        self.assertLessEqual(len(sandbox.ALLOWED_MX_ATTRS), 16)

    def test_worker_revalidates_and_uses_restricted_builtins(self) -> None:
        self.assertIn("validate_plan_source(plan_source)", sandbox._WORKER)
        self.assertIn('"__builtins__":', sandbox._WORKER)


if __name__ == "__main__":
    unittest.main()


class ListAccumulatorTest(unittest.TestCase):
    """A plan must be able to build a list and evaluate it in one call.

    Rejecting `out.append(x)` once rejected the exact optimization the search
    exists to find: collect the results, then hand the whole list to mx.eval.
    The allowance is narrow -- known accumulator names, two mutators.
    """

    ACCUMULATING = """def plan(mx, a, operands):
    out = []
    for b in operands:
        x = mx.matmul(a, b)
        out.append(x)
    mx.eval(out)
    mx.synchronize()
    return out
"""

    def test_the_accumulating_plan_is_accepted(self) -> None:
        self.assertIsNotNone(sandbox.validate_plan_source(self.ACCUMULATING))

    def test_extend_is_accepted(self) -> None:
        source = self.ACCUMULATING.replace("out.append(x)", "out.extend([x])")
        self.assertIsNotNone(sandbox.validate_plan_source(source))

    def test_other_list_methods_are_rejected(self) -> None:
        for method in ("pop", "clear", "sort", "__init__", "count"):
            with self.subTest(method=method):
                source = self.ACCUMULATING.replace("out.append(x)", f"out.{method}()")
                with self.assertRaises(sandbox.PlanRejected):
                    sandbox.validate_plan_source(source)

    def test_the_allowance_does_not_extend_to_other_names(self) -> None:
        # a and operands are MLX objects; they must stay untouchable.
        for owner in ("a", "operands", "mx"):
            with self.subTest(owner=owner):
                source = self.ACCUMULATING.replace("out.append(x)", f"{owner}.append(x)")
                with self.assertRaises(sandbox.PlanRejected):
                    sandbox.validate_plan_source(source)

    def test_dunder_on_an_allowed_list_is_still_rejected(self) -> None:
        source = self.ACCUMULATING.replace("out.append(x)", "out.__class__")
        with self.assertRaises(sandbox.PlanRejected):
            sandbox.validate_plan_source(source)
