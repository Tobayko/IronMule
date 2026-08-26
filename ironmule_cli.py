"""Dependency-light command-line interface for IronMule.

This module intentionally lives at the distribution top level. Python can load
the ``ironmule`` console script without executing ``ironmule.__init__`` first,
so ``ironmule doctor`` remains useful when MLX is missing or cannot initialize.
The benchmark subcommand lazily imports the established benchmark module and
does not alter its measurements or arithmetic.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import platform
import sys
from typing import Any, Iterable


MIN_PYTHON = (3, 10)


def _version(distribution: str, module: Any = None) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown")) if module else "unknown"


def _load_optional(module_name: str, distribution: str) -> tuple[bool, str, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # MLX may fail for architecture/Metal reasons.
        return False, _version(distribution), f"{type(exc).__name__}: {exc}"
    return True, _version(distribution, module), "importable"


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine() or "unknown"


def _doctor_checks() -> list[tuple[str, bool, str]]:
    machine = platform.machine().lower()
    checks: list[tuple[str, bool, str]] = [
        ("Apple Silicon architecture", machine in {"arm64", "aarch64"},
         f"{machine} ({_cpu_name()})"),
        ("macOS", platform.system() == "Darwin", platform.system()),
        ("Python", sys.version_info[:2] >= MIN_PYTHON,
         f"{platform.python_version()} (requires >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})"),
    ]
    mlx_ok, mlx_version, mlx_detail = _load_optional("mlx.core", "mlx")
    checks.append(("MLX", mlx_ok, f"{mlx_version}; {mlx_detail}"))
    mlx_lm_ok, mlx_lm_version, mlx_lm_detail = _load_optional("mlx_lm", "mlx-lm")
    checks.append(("MLX-LM", mlx_lm_ok, f"{mlx_lm_version}; {mlx_lm_detail}"))
    numpy_ok, numpy_version, numpy_detail = _load_optional("numpy", "numpy")
    checks.append(("NumPy", numpy_ok, f"{numpy_version}; {numpy_detail}"))

    metal_ok = False
    metal_detail = "MLX unavailable"
    if mlx_ok:
        try:
            mlx = importlib.import_module("mlx.core")
            metal_ok = bool(mlx.metal.is_available())
            metal_detail = "Metal available" if metal_ok else "Metal unavailable"
        except Exception as exc:
            metal_detail = f"Metal check failed: {type(exc).__name__}: {exc}"
    checks.append(("MLX Metal device", metal_ok, metal_detail))
    return checks


def doctor(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(
        prog="ironmule doctor", description="Check IronMule runtime prerequisites."
    )
    parser.parse_args(list(argv))
    print("IronMule doctor")
    checks = _doctor_checks()
    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print("\nMissing or unavailable prerequisites: " + ", ".join(failed))
        if any(name in failed for name in ("MLX", "MLX-LM")):
            print("Hint: on Apple Silicon, install IronMule with `pip install ironmule`, then rerun `ironmule doctor`.")
        return 1
    print("\nAll runtime prerequisites are available.")
    return 0


def info(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(
        prog="ironmule info", description="Show IronMule package information."
    )
    parser.parse_args(list(argv))
    try:
        version = metadata.version("ironmule")
    except metadata.PackageNotFoundError:
        version = "source checkout"
    print(f"IronMule {version}")
    print("Adaptive MLX inference runtime for local LLMs on Apple Silicon")
    print("Measured, not assumed.")
    return 0


def _is_runtime_dependency_error(exc: ImportError) -> bool:
    """Recognize dependency failures while leaving unrelated ImportErrors visible."""
    missing = getattr(exc, "name", "") or ""
    if missing in {"mlx", "mlx.core", "mlx_lm", "numpy"}:
        return True
    message = str(exc).lower()
    return any(token in message for token in ("no module named 'mlx", "no module named 'mlx_lm", "no module named 'numpy"))


def _load_benchmark():
    from ironmule.benchmark import main as benchmark_main

    return benchmark_main


def _run_benchmark(argv: list[str]) -> int:
    try:
        benchmark_main = _load_benchmark()
    except ImportError as exc:
        if not _is_runtime_dependency_error(exc):
            raise
        print(
            "IronMule benchmark requires MLX/MLX-LM runtime dependencies; run `ironmule doctor` "
            f"for details ({exc}).",
            file=sys.stderr,
        )
        return 1

    try:
        return int(benchmark_main(argv))
    except ImportError as exc:
        if not _is_runtime_dependency_error(exc):
            raise
        print(
            "IronMule benchmark requires MLX/MLX-LM runtime dependencies; run `ironmule doctor` "
            f"for details ({exc}).",
            file=sys.stderr,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: ironmule {doctor|benchmark|info} [options]")
        print("\ncommands:\n  doctor       Check Apple Silicon and MLX prerequisites")
        print("  benchmark   Run the existing reproducible local benchmark")
        print("  info        Show package information")
        return 0
    command, rest = args[0], args[1:]
    if command == "doctor":
        return doctor(rest)
    if command == "benchmark":
        return _run_benchmark(rest)
    if command == "info":
        return info(rest)
    print(f"ironmule: unknown command {command!r}; use --help", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
