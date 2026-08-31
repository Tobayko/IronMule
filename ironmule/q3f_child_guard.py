"""Bounded no-detach guard for the Q3f model child.

This module intentionally has no dependency on the model runtime.  It is
loaded as the first operation of :func:`ironmule.ab._child`, before any model
module is imported.  The guard is a *fail-closed* evidence device: an
operation that could create a process or move a process into a new session is
recorded and blocked, and an incomplete ledger is never reported as success.
"""

from __future__ import annotations

import ast
import ctypes
import inspect
import json
import os
import subprocess
import sys
import textwrap
import time
from types import FunctionType
from typing import Any, Callable, Mapping


VERSION = "ironmule.q3f_child_guard.v1"
MAX_EVENTS = 32
MAX_EVENT_BYTES = 512
MAX_SOURCE_BYTES = 512 * 1024
NOTE_FORK = 0x40000000
EVFILT_PROC = -5
EV_ADD = 0x0001
EV_ENABLE = 0x0004
EV_CLEAR = 0x0020

# This is the single source of truth shared by the audit hook, wrappers and
# the static checker.  Keep spellings exact: Q3f tests assert set equality.
OPERATION_SET = frozenset({
    "subprocess.Popen",
    "os.system",
    "os.fork",
    "os.forkpty",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.setsid",
    "os.setpgid",
})
KNOWN_INFERENCE_ACTIVITY = ("mlx", "llama", "ollama", "vllm", "gemma", "qwen")
BLOCKER_TOKENS = tuple(dict.fromkeys(
    token.casefold() for token in KNOWN_INFERENCE_ACTIVITY
    + ("q3c", "q3d", "ironmule", "mlx", "gemma", "huggingface")
))
BLOCKER_SET = frozenset(BLOCKER_TOKENS)

_AUDIT_EVENT_TO_OPERATION = {operation: operation for operation in OPERATION_SET}
_WRAPPED_OPERATIONS = {
    "subprocess.Popen": (subprocess, "Popen"),
    "os.system": (os, "system"),
    "os.fork": "fork",
    "os.forkpty": "forkpty",
    "os.posix_spawn": "posix_spawn",
    "os.posix_spawnp": "posix_spawnp",
    "os.setsid": "setsid",
    "os.setpgid": "setpgid",
}
REVIEWED_MODULES = frozenset({
    "tune", "ironmule.tune", "runtime", "ironmule.runtime", "mlx", "mlx.core",
    "os", "json", "math", "q3f_child_guard", "ironmule.q3f_child_guard",
})
REVIEWED_ATTRIBUTES = frozenset({
    "close", "eos_token_ids", "eos_token_id", "language_model", "tie_word_embeddings",
    "__version__", "bias", "layers", "self_attn", "mlp", "name",
})
NATIVE_BOUNDARY_MODULES = frozenset({"mlx", "mlx.core", "mlx_lm"})
REVIEWED_EXTERNAL_MODULES = frozenset({"huggingface_hub", "huggingface_hub.utils"})
REVIEWED_SOURCE_MODULES = frozenset({
    "ironmule.ab", "ironmule.tune", "ironmule.runtime", "ironmule.model_identity",
    "ironmule.fast", "ironmule.hw", "ironmule.bench",
    "ironmule.q3f_child_guard",
})
REVIEWED_STDLIB_MODULES = frozenset({
    "argparse", "ast", "collections", "dataclasses", "hashlib", "importlib",
    "json", "math", "os", "pathlib", "re", "resource", "secrets", "signal",
    "statistics", "subprocess", "sys", "threading", "time", "typing", "types",
})


class GuardViolation(RuntimeError):
    """A process/session operation was attempted by the model child."""


class GuardInstallationError(RuntimeError):
    """The bounded guard or its evidence ledger could not be installed."""


class _Kevent(ctypes.Structure):
    _fields_ = [
        ("ident", ctypes.c_uint64), ("filter", ctypes.c_int16),
        ("flags", ctypes.c_uint16), ("fflags", ctypes.c_uint32),
        ("data", ctypes.c_int64), ("udata", ctypes.c_uint64),
    ]


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_int64), ("tv_nsec", ctypes.c_int64)]


class _NativeForkMonitor:
    """Bounded macOS kqueue NOTE_FORK monitor, with no project imports."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._libc: Any = None
        self.events: list[dict[str, Any]] = []

    def install(self) -> None:
        if sys.platform != "darwin":
            return
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            kqueue = libc.kqueue
            kqueue.argtypes = []
            kqueue.restype = ctypes.c_int
            fd = int(kqueue())
            if fd < 0:
                raise OSError(ctypes.get_errno(), "kqueue")
            kevent = libc.kevent
            kevent.argtypes = [ctypes.c_int, ctypes.POINTER(_Kevent), ctypes.c_int,
                               ctypes.POINTER(_Kevent), ctypes.c_int,
                               ctypes.POINTER(_Timespec)]
            kevent.restype = ctypes.c_int
            change = _Kevent(os.getpid(), EVFILT_PROC, EV_ADD | EV_ENABLE | EV_CLEAR,
                             NOTE_FORK, 0, 0)
            result = int(kevent(fd, ctypes.byref(change), 1, None, 0, None))
            if result < 0:
                error = ctypes.get_errno()
                os.close(fd)
                raise OSError(error, "kevent")
            self._fd, self._libc = fd, libc
        except BaseException as exc:
            self._fd, self._libc = None, None
            if isinstance(exc, GuardInstallationError):
                raise
            raise GuardInstallationError("Q3f native fork monitor unavailable") from exc

    def poll(self) -> None:
        if sys.platform != "darwin":
            return
        if self._fd is None or self._libc is None:
            raise GuardInstallationError("Q3f native fork monitor is not installed")
        output = (_Kevent * 32)()
        timeout = _Timespec(0, 0)
        try:
            count = int(self._libc.kevent(self._fd, None, 0, output, 32,
                                          ctypes.byref(timeout)))
        except BaseException as exc:
            raise GuardInstallationError("Q3f native fork monitor poll failed") from exc
        if count < 0:
            raise GuardInstallationError("Q3f native fork monitor poll failed")
        for index in range(count):
            event = output[index]
            if event.fflags & NOTE_FORK:
                if len(self.events) >= MAX_EVENTS:
                    raise GuardInstallationError("Q3f native fork monitor ledger overflow")
                self.events.append({"pid": int(event.ident), "fflags": int(event.fflags),
                                    "monotonic": time.monotonic()})
        if self.events:
            raise GuardViolation("native process fork detected by Q3f monitor")

    def uninstall(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self._fd, self._libc = None, None


def _event_record(event: str, operation: str, blocked: bool) -> dict[str, Any]:
    record = {
        "event": event,
        "operation": operation,
        "monotonic": time.monotonic(),
        "blocked": blocked,
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode()
    if len(encoded) > MAX_EVENT_BYTES:
        raise GuardInstallationError("Q3f guard event exceeds its bound")
    return record


class _Guard:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._wrapped: list[str] = []
        self._originals: dict[str, Any] = {}
        self._installed = False
        self._enabled = False
        self._native_monitor = _NativeForkMonitor()

    @property
    def events(self) -> list[dict[str, Any]]:
        return self._events

    def _record(self, event: str, operation: str, *, blocked: bool = True) -> None:
        if operation not in OPERATION_SET:
            raise GuardInstallationError("Q3f guard received an unreviewed operation")
        if len(self._events) >= MAX_EVENTS:
            raise GuardInstallationError("Q3f guard ledger overflow")
        record = _event_record(event, operation, blocked)
        self._events.append(record)

    def block(self, operation: str, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._record(operation, operation)
        raise GuardViolation(f"blocked Q3f operation: {operation}")

    def audit(self, event: str, args: tuple[Any, ...]) -> None:
        del args
        if not self._enabled:
            return
        operation = _AUDIT_EVENT_TO_OPERATION.get(event)
        if operation is None:
            return
        self.block(operation)

    def snapshot(self) -> dict[str, Any]:
        if not self._installed:
            raise GuardInstallationError("Q3f guard was not installed")
        self._native_monitor.poll()
        value = {"version": VERSION, "installed": True,
                 "events": [dict(item) for item in self._events]}
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()
        if len(encoded) > MAX_EVENT_BYTES * (MAX_EVENTS + 1):
            raise GuardInstallationError("Q3f guard ledger is unbounded")
        return value

    def install(self) -> dict[str, Any]:
        if self._installed:
            raise GuardInstallationError("Q3f guard installed more than once")
        try:
            self._native_monitor.install()
            sys.addaudithook(self.audit)
        except BaseException as exc:
            self._native_monitor.uninstall()
            raise GuardInstallationError("Q3f audit hook installation failed") from exc
        try:
            for operation, target in _WRAPPED_OPERATIONS.items():
                module, attribute = target if isinstance(target, tuple) else (os, target)
                if not hasattr(module, attribute):
                    continue
                original = getattr(module, attribute)
                if not callable(original):
                    raise GuardInstallationError(f"Q3f os operation is not callable: {attribute}")

                def blocked(*args: Any, _operation=operation, **kwargs: Any) -> Any:
                    return self.block(_operation, *args, **kwargs)

                key = f"{module.__name__}.{attribute}"
                self._originals[key] = (module, attribute, original)
                setattr(module, attribute, blocked)
                self._wrapped.append(operation)
            # An operation present in the runtime must be wrapped.  This catches
            # incomplete/unknown platforms before the model is loaded.
            expected: set[str] = set()
            for operation, target in _WRAPPED_OPERATIONS.items():
                module, attribute = target if isinstance(target, tuple) else (os, target)
                if hasattr(module, attribute):
                    expected.add(operation)
            if set(self._wrapped) != expected:
                raise GuardInstallationError("Q3f available operation wrapping is incomplete")
            self._installed = True
            self._enabled = True
            return self.snapshot()
        except BaseException as exc:
            self._enabled = False
            for _, (module, attribute, original) in self._originals.items():
                try:
                    setattr(module, attribute, original)
                except BaseException:
                    pass
            self._originals.clear()
            self._wrapped.clear()
            self._installed = False
            self._native_monitor.uninstall()
            if isinstance(exc, GuardInstallationError):
                raise
            raise GuardInstallationError("Q3f guard installation failed") from exc

    def uninstall(self) -> None:
        """Restore wrapped APIs for direct/in-process test callers.

        A production child exits immediately after the result marker, but
        restoring here also makes a direct closure invocation exception-safe.
        The audit hook cannot be removed; it becomes inert via ``_enabled``.
        """
        self._enabled = False
        for _, (module, attribute, original) in self._originals.items():
            try:
                setattr(module, attribute, original)
            except BaseException:
                # The guard is still considered unavailable if restoration
                # fails; callers must treat the enclosing child as failed.
                raise GuardInstallationError(f"Q3f cannot restore os.{attribute}")
        self._originals.clear()
        self._wrapped.clear()
        self._installed = False
        self._native_monitor.uninstall()


def install() -> dict[str, Any]:
    """Install one guard and return its exact initial zero-event ledger."""
    global _ACTIVE_GUARD
    if _ACTIVE_GUARD is not None:
        raise GuardInstallationError("Q3f active guard already exists")
    guard = _Guard()
    value = guard.install()
    # Keep the state private but expose a narrow test/runtime handle.  There
    # is exactly one guard per child invocation.
    _ACTIVE_GUARD = guard
    return value


_ACTIVE_GUARD: _Guard | None = None


def ledger() -> dict[str, Any]:
    """Return the current bounded ledger, or fail closed if unavailable."""
    if _ACTIVE_GUARD is None:
        raise GuardInstallationError("Q3f active guard is unavailable")
    return _ACTIVE_GUARD.snapshot()


def failure_marker() -> dict[str, Any] | None:
    """Return bounded guard evidence without polling a failed native monitor."""
    if _ACTIVE_GUARD is None:
        return None
    events = [dict(item) for item in _ACTIVE_GUARD.events[:MAX_EVENTS]]
    value = {"version": VERSION, "installed": _ACTIVE_GUARD._installed,
             "events": events}
    native_events = [dict(item) for item in _ACTIVE_GUARD._native_monitor.events[:MAX_EVENTS]]
    if native_events:
        value["native_events"] = native_events
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError):
        return {"version": VERSION, "installed": False, "events": []}
    if len(encoded) > MAX_EVENT_BYTES * (MAX_EVENTS + 1):
        return {"version": VERSION, "installed": False, "events": []}
    return value


def uninstall() -> None:
    """Disable the active guard and restore wrapped process APIs."""
    global _ACTIVE_GUARD
    if _ACTIVE_GUARD is None:
        return
    guard = _ACTIVE_GUARD
    try:
        guard.uninstall()
    finally:
        _ACTIVE_GUARD = None


def is_installed() -> bool:
    """Whether the bootstrap-installed guard is active in this process."""
    return _ACTIVE_GUARD is not None and _ACTIVE_GUARD._installed and _ACTIVE_GUARD._enabled


def blocker_operation_set() -> frozenset[str]:
    """Expose the exact operation set for static/runtime equality tests."""
    return OPERATION_SET


def _scan_source_tree(tree: ast.AST) -> None:
    """Reject blocked operations and dynamic paths in an AST without imports."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted = _dotted_name(node.func, aliases)
            if dotted is None:
                if (isinstance(node.func, ast.Call)
                        and _dotted_name(node.func.func, aliases) == "_trunk"):
                    continue
                raise GuardInstallationError("Q3f source call target is not statically reviewable")
            if dotted in OPERATION_SET or dotted in {
                f"{name}.{operation}" for name in ("os", "subprocess")
                for operation in ("Popen", "system", "fork", "forkpty", "posix_spawn", "posix_spawnp", "setsid", "setpgid")
            }:
                raise GuardInstallationError(f"Q3f source reaches blocked operation: {dotted}")
            if dotted in {"eval", "exec", "compile", "__import__", "importlib.import_module"}:
                raise GuardInstallationError("Q3f source has a dynamic/unreviewed call path")
            if dotted and dotted.rsplit(".", 1)[-1] == "getattr":
                constant_attribute = (len(node.args) >= 2
                                      and isinstance(node.args[1], ast.Constant)
                                      and node.args[1].value in REVIEWED_ATTRIBUTES)
                model_identity_attribute = (len(node.args) >= 2
                                            and isinstance(node.args[0], ast.Name)
                                            and node.args[0].id == "self"
                                            and isinstance(node.args[1], ast.Name)
                                            and node.args[1].id == "name")
                if not constant_attribute and not model_identity_attribute:
                    raise GuardInstallationError("Q3f source has a dynamic attribute path")


def _normalise_local_module(module: str | None, current: str) -> str:
    if module is None or module == "":
        return "ironmule.q3f_child_guard"
    if module.startswith("ironmule."):
        return module
    if module in {"tune", "runtime", "model_identity", "fast", "hw", "bench"}:
        return "ironmule." + module
    return module


def _import_aliases(module: str, nodes: list[ast.AST]) -> dict[str, str]:
    """Resolve module-level imports without importing any project module."""
    aliases: dict[str, str] = {}
    for item in nodes:
        if isinstance(item, ast.Import):
            for alias in item.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(item, ast.ImportFrom):
            if item.level:
                package_parts = module.split(".")[:-item.level]
                imported_module = ".".join(package_parts + ([item.module] if item.module else []))
            else:
                imported_module = item.module or ""
            for alias in item.names:
                target = ".".join(part for part in (imported_module, alias.name) if part)
                aliases[alias.asname or alias.name] = target
    return aliases


def _recursive_source_surface(root: str) -> tuple[tuple[str, str], ...]:
    """Resolve and scan the bounded local call graph without importing it."""
    paths = {
        "ironmule.ab": os.path.join(root, "ironmule", "ab.py"),
        "ironmule.tune": os.path.join(root, "ironmule", "tune.py"),
        "ironmule.runtime": os.path.join(root, "ironmule", "runtime.py"),
        "ironmule.model_identity": os.path.join(root, "ironmule", "model_identity.py"),
        "ironmule.fast": os.path.join(root, "ironmule", "fast.py"),
        "ironmule.hw": os.path.join(root, "ironmule", "hw.py"),
        "ironmule.bench": os.path.join(root, "ironmule", "bench.py"),
    }
    trees: dict[str, ast.Module] = {}
    for module, filename in paths.items():
        try:
            with open(filename, "rb") as stream:
                raw = stream.read(MAX_SOURCE_BYTES + 1)
            if len(raw) > MAX_SOURCE_BYTES:
                raise GuardInstallationError("Q3f reviewed source exceeds bound")
            trees[module] = ast.parse(raw.decode("utf-8"), filename=filename)
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise GuardInstallationError("Q3f recursive source is unavailable") from exc
    definitions: dict[str, dict[str, ast.AST]] = {}
    module_aliases: dict[str, dict[str, str]] = {}
    methods: dict[str, list[tuple[str, str, ast.AST]]] = {}
    for module, tree in trees.items():
        module_aliases[module] = _import_aliases(module, list(tree.body))
        definitions[module] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions[module][node.name] = node
                if isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            key = f"{node.name}.{child.name}"
                            definitions[module][key] = child
                            methods.setdefault(child.name, []).append((module, key, child))
    queue: list[tuple[str, str, ast.AST]] = [
        ("ironmule.ab", "_child", definitions["ironmule.ab"].get("_child")),
        ("ironmule.ab", "_child_execution", definitions["ironmule.ab"].get("_child_execution")),
        ("ironmule.tune", "load_engine", definitions["ironmule.tune"].get("load_engine")),
        ("ironmule.tune", "resolve_local_model", definitions["ironmule.tune"].get("resolve_local_model")),
        ("ironmule.tune", "verify_resolved_model", definitions["ironmule.tune"].get("verify_resolved_model")),
        ("ironmule.tune", "_eos_ids", definitions["ironmule.tune"].get("_eos_ids")),
        ("ironmule.tune", "prompt_ids", definitions["ironmule.tune"].get("prompt_ids")),
        ("ironmule.runtime", "Engine", definitions["ironmule.runtime"].get("Engine")),
    ]
    if any(node is None for _, _, node in queue):
        raise GuardInstallationError("Q3f recursive source root is incomplete")
    visited: set[tuple[str, str]] = set()
    while queue:
        module, key, node = queue.pop()
        identity = (module, key)
        if identity in visited:
            continue
        visited.add(identity)
        _scan_source_tree(node)
        # Module-level imports (especially tune's model_identity imports) are
        # inherited by every selected function, then local imports overlay them.
        aliases: dict[str, str] = dict(module_aliases[module])
        for item in ast.walk(node):
            if isinstance(item, ast.Import):
                for alias in item.names:
                    aliases[alias.asname or alias.name.split(".")[0]] = alias.name
                    imported = alias.name
                    root_name = imported.split(".")[0]
                    native = any(imported == boundary or imported.startswith(boundary + ".")
                                 for boundary in NATIVE_BOUNDARY_MODULES)
                    external = any(imported == boundary or imported.startswith(boundary + ".")
                                   for boundary in REVIEWED_EXTERNAL_MODULES)
                    if not native and not external and root_name not in REVIEWED_STDLIB_MODULES:
                        raise GuardInstallationError(f"Q3f reachable import is outside the reviewed allowlist: {imported}")
            elif isinstance(item, ast.ImportFrom):
                if item.module is None:
                    for alias in item.names:
                        local_module = f"ironmule.{alias.name}"
                        aliases[alias.asname or alias.name] = local_module
                        if (local_module not in REVIEWED_SOURCE_MODULES
                                and local_module not in NATIVE_BOUNDARY_MODULES
                                and local_module not in REVIEWED_EXTERNAL_MODULES):
                            raise GuardInstallationError(
                                f"Q3f reachable import is outside the reviewed allowlist: {local_module}"
                            )
                    continue
                local_module = _normalise_local_module(item.module, module)
                for alias in item.names:
                    aliases[alias.asname or alias.name] = (
                        local_module if item.module is None else f"{local_module}.{alias.name}"
                    )
                native = any(local_module == boundary or local_module.startswith(boundary + ".")
                             for boundary in NATIVE_BOUNDARY_MODULES)
                external = any(local_module == boundary or local_module.startswith(boundary + ".")
                               for boundary in REVIEWED_EXTERNAL_MODULES)
                if (local_module not in REVIEWED_SOURCE_MODULES and not native
                        and not external and local_module.split(".")[0] not in REVIEWED_STDLIB_MODULES):
                    raise GuardInstallationError(f"Q3f reachable import is outside the reviewed allowlist: {local_module}")
        for item in ast.walk(node):
            if not isinstance(item, ast.Call):
                continue
            dotted = _dotted_name(item.func, aliases)
            if dotted is None:
                if isinstance(item.func, ast.Call) and _dotted_name(item.func.func, aliases) == "_trunk":
                    continue
                raise GuardInstallationError("Q3f reachable call target is not statically reviewable")
            target_module, _, target_name = dotted.rpartition(".")
            if dotted in definitions.get(module, {}):
                child = definitions[module][dotted]
                queue.append((module, dotted, child))
            elif target_module in definitions and target_name in definitions[target_module]:
                queue.append((target_module, target_name, definitions[target_module][target_name]))
            elif dotted in definitions.get(module, {}):
                queue.append((module, dotted, definitions[module][dotted]))
            elif isinstance(item.func, ast.Attribute) and item.func.attr in methods:
                queue.extend(methods[item.func.attr])
            elif target_module.startswith("ironmule.") and target_module not in REVIEWED_SOURCE_MODULES:
                raise GuardInstallationError(f"Q3f reachable local call is outside the reviewed allowlist: {dotted}")
    required = {
        ("ironmule.model_identity", "resolve_model_source"),
        ("ironmule.model_identity", "scan_local_cache"),
    }
    if not required.issubset(visited):
        raise GuardInstallationError("Q3f model identity closure is incomplete")
    return tuple(sorted(visited))[:128]

def assert_source_surface(ab_path: os.PathLike[str] | str) -> tuple[tuple[str, str], ...]:
    """Scan the frozen child/runtime source surface using bytes only.

    No project module is imported.  The explicit allowlist covers the child
    closures plus exactly the tune/runtime entrypoints they can reach.  The
    parent-side ``ab.run`` is intentionally not selected, so its legitimate
    ``subprocess.Popen`` is not mistaken for a child escape path.
    """
    path = os.path.realpath(os.fspath(ab_path))
    root = os.path.dirname(os.path.dirname(path))
    files = (
        (path, ("_child", "_child_execution")),
        (os.path.join(root, "ironmule", "tune.py"), ("load_engine", "_eos_ids", "prompt_ids")),
        (os.path.join(root, "ironmule", "runtime.py"), ("Engine",)),
    )
    for filename, names in files:
        try:
            with open(filename, "rb") as stream:
                source = stream.read(MAX_SOURCE_BYTES + 1)
        except (OSError, ValueError) as exc:
            raise GuardInstallationError("Q3f reviewed source is unavailable") from exc
        if len(source) > MAX_SOURCE_BYTES:
            raise GuardInstallationError("Q3f reviewed source exceeds bound")
        try:
            tree = ast.parse(source.decode("utf-8"), filename=filename)
        except (UnicodeError, SyntaxError) as exc:
            raise GuardInstallationError("Q3f reviewed source is not parseable") from exc
        selected: list[ast.AST] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
                selected.append(node)
            elif isinstance(node, ast.ClassDef) and node.name in names:
                selected.append(node)
        if len(selected) != len(names):
            raise GuardInstallationError("Q3f reviewed source function allowlist is incomplete")
        for node in selected:
            _scan_source_tree(node)
            if isinstance(node, ast.ClassDef) and node.name == "Engine":
                methods = {item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
                for method in ("generate", "close"):
                    if method not in methods:
                        raise GuardInstallationError("Q3f Engine child method allowlist is incomplete")
    return _recursive_source_surface(root)


def _dotted_name(node: ast.AST, aliases: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        left = _dotted_name(node.value, aliases)
        return f"{left}.{node.attr}" if left else node.attr
    return None


_REVIEWED_CALLS = frozenset({
    "_eos_ids", "load_engine", "prompt_ids", "Knobs", "range", "len", "list",
    "map", "int", "all", "zip", "max", "getattr", "close", "reset_peak_memory",
    "get_peak_memory", "generate", "snapshot", "install", "uninstall", "ledger", "failure_marker", "is_installed", "assert_child_surface", "_child_execution", "get", "values", "add_note", "type",
})


def assert_child_surface(function: Callable[..., Any]) -> None:
    """Statically check the actual child closure's reviewed Python surface.

    The scan is intentionally bounded and conservative.  It starts at the
    supplied closure, resolves only the explicit allowlist of module/callee
    spellings, rejects dynamic execution/import and every process operation,
    and does not scan the parent-side ``ab.run`` (which legitimately uses
    ``subprocess.Popen``).
    """
    if not isinstance(function, FunctionType) or function.__name__ != "_child":
        raise GuardInstallationError("Q3f static scan must start at ab._child")
    targets: list[FunctionType] = [function]
    execution = function.__globals__.get("_child_execution")
    if not isinstance(execution, FunctionType):
        raise GuardInstallationError("Q3f guarded execution closure is unavailable")
    targets.append(execution)
    for target in targets:
        try:
            source = textwrap.dedent(inspect.getsource(target))
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError) as exc:
            raise GuardInstallationError("Q3f child source is not statically reviewable") from exc
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    aliases[alias.asname or alias.name.split(".")[0]] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", None)
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                else:
                    names = [f"{module}.{alias.name}" for alias in node.names]
                imported_modules = ([name.split(".")[0] for name in names]
                                    if isinstance(node, ast.Import)
                                    else [module or "q3f_child_guard"])
                if any(module_name not in REVIEWED_MODULES
                       and not any(module_name == allowed or module_name.startswith(allowed + ".")
                                   for allowed in REVIEWED_MODULES)
                       for module_name in imported_modules):
                    raise GuardInstallationError("Q3f child imports a module outside the reviewed allowlist")
                if any(name in {"subprocess", "multiprocessing", "os"} for name in names):
                    # ``os`` is allowed only for the harmless getpid call in
                    # the closure; process operations are still rejected below.
                    if names != ["os"]:
                        raise GuardInstallationError("Q3f child imports an unreviewed module")
            if isinstance(node, ast.Call):
                dotted = _dotted_name(node.func, aliases)
                if dotted is None:
                    raise GuardInstallationError("Q3f child call target is not statically reviewable")
                if dotted in OPERATION_SET or dotted in {
                    f"{name}.{operation}" for name in ("os", "subprocess")
                    for operation in ("Popen", "system", "fork", "forkpty", "posix_spawn", "posix_spawnp", "setsid", "setpgid")
                }:
                    raise GuardInstallationError(f"Q3f child reaches blocked operation: {dotted}")
                if dotted in {"eval", "exec", "compile", "__import__", "importlib.import_module"}:
                    raise GuardInstallationError("Q3f child has a dynamic/unreviewed call path")
                terminal = dotted.rsplit(".", 1)[-1] if dotted else None
                if terminal == "getattr":
                    if (len(node.args) < 2 or not isinstance(node.args[1], ast.Constant)
                            or node.args[1].value != "close"):
                        raise GuardInstallationError("Q3f child has a dynamic attribute path")
                if terminal not in _REVIEWED_CALLS and dotted not in {
                    "os.getpid", "json.dumps", "json.loads", "math.isfinite",
                }:
                    # Attribute calls on explicitly reviewed runtime objects are
                    # bounded to their known terminal operation names.
                    raise GuardInstallationError(f"Q3f child call is not reviewed: {dotted}")
