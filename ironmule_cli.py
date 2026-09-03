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
import json
import platform
import subprocess
import sys
from typing import Any, Iterable


MIN_PYTHON = (3, 10)


def _version(distribution: str, module: Any = None) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown")) if module else "unknown"


def _load_optional(module_name: str, distribution: str) -> tuple[bool, str, str]:
    """Probe optional dependencies out of process so failed imports cannot leak state."""
    probe = (
        "import importlib,sys; "
        "module=importlib.import_module(sys.argv[1]); "
        "print(getattr(module, '__version__', 'unknown'))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe, module_name],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, _version(distribution), f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        detail = (result.stderr.strip().splitlines()[-1]
                  if result.stderr.strip() else f"probe exited {result.returncode}")
        return False, _version(distribution), f"isolated probe failed: {detail}"
    version = result.stdout.strip() or _version(distribution)
    return True, version, "importable (isolated probe)"


def _probe_metal() -> tuple[bool, str]:
    """Check Metal in a child process for the same isolation guarantee as imports."""
    probe = (
        "import importlib; "
        "module=importlib.import_module('mlx.core'); "
        "print('available' if module.metal.is_available() else 'unavailable')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        detail = (result.stderr.strip().splitlines()[-1]
                  if result.stderr.strip() else f"probe exited {result.returncode}")
        return False, f"isolated probe failed: {detail}"
    available = result.stdout.strip() == "available"
    return available, "Metal available" if available else "Metal unavailable"


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

    metal_ok, metal_detail = _probe_metal() if mlx_ok else (False, "MLX unavailable")
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


def _load_tune():
    """Load the existing tuner only after a tune/revalidate/status command."""
    # ``ironmule.__init__`` re-exports the tune function under this name; import the
    # submodule explicitly so command handlers receive its parser and API.
    return importlib.import_module("ironmule.tune")


def _dependency_error(command: str, exc: ImportError, *, dependency: str = "MLX/MLX-LM") -> int:
    print(
        f"IronMule {command} requires {dependency} runtime dependencies; run `ironmule doctor` "
        f"for details ({exc}).",
        file=sys.stderr,
    )
    return 1


def _run_tune(argv: list[str]) -> int:
    try:
        tune_module = _load_tune()
    except ImportError as exc:
        if not _is_runtime_dependency_error(exc):
            raise
        return _dependency_error("tune", exc)
    try:
        return int(tune_module.main(argv))
    except ImportError as exc:
        if not _is_runtime_dependency_error(exc):
            raise
        return _dependency_error("tune", exc)


def _run_revalidate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="ironmule revalidate", description="Canary-check the stored tuning profile."
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    try:
        tune_module = _load_tune()
    except ImportError as exc:
        if not _is_runtime_dependency_error(exc):
            raise
        return _dependency_error("revalidate", exc)
    try:
        model = args.model or tune_module.DEFAULT_MODEL
        result = tune_module.revalidate(model_id=model, max_tokens=args.max_tokens)
    except ImportError as exc:
        if not _is_runtime_dependency_error(exc):
            raise
        return _dependency_error("revalidate", exc)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _run_status(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="ironmule status", description="Show the local hardware and profile status."
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)
    try:
        tune_module = _load_tune()
    except ImportError as exc:
        if not _is_runtime_dependency_error(exc):
            raise
        return _dependency_error("status", exc)
    try:
        model = args.model or tune_module.DEFAULT_MODEL
        profile = tune_module.load_profile(model, require_compatible=False)
        compatible = tune_module.load_profile(model, require_compatible=True)
        status = "compatible" if compatible else "stale" if profile else "missing"
        result = {
            "model": model,
            "hardware_fingerprint": tune_module.fingerprint(),
            "profile_status": status,
            "profile_store": str(tune_module.PROFILES),
        }
    except ImportError as exc:
        if not _is_runtime_dependency_error(exc):
            raise
        return _dependency_error("status", exc)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _is_huggingface_dependency_error(exc: ImportError) -> bool:
    """`models` reaches the cache through `ironmule`, so MLX counts here too.

    Reading the cache needs nothing but `huggingface_hub`, but the shared helper lives
    in the `ironmule` package and importing that pulls in MLX. Either missing piece is
    an install problem with the same answer, so both get the same message rather than a
    traceback; `ironmule doctor` imports neither and still says which one broke.
    """
    missing = getattr(exc, "name", "") or ""
    if missing.split(".", 1)[0] in {"huggingface_hub", "mlx", "mlx_lm"}:
        return True
    message = str(exc).lower()
    return any(
        f"no module named '{name}" in message
        for name in ("huggingface_hub", "mlx", "mlx_lm")
    )


def _safe_string(value: Any) -> str:
    try:
        return str(value)
    except Exception:  # pragma: no cover - defensive for third-party warning objects
        return f"<{type(value).__name__} could not be rendered>"


def _cached_revision(revision: Any) -> dict[str, Any]:
    return {
        "commit_hash": _safe_string(getattr(revision, "commit_hash", "")),
        "snapshot_path": _safe_string(getattr(revision, "snapshot_path", "")),
        "size_on_disk": getattr(revision, "size_on_disk", None),
        "last_modified": getattr(revision, "last_modified", None),
    }


def _cached_model(repo: Any) -> dict[str, Any]:
    revisions = [
        _cached_revision(revision)
        for revision in (getattr(repo, "revisions", ()) or ())
    ]
    revisions.sort(key=lambda item: (item["commit_hash"], item["snapshot_path"]))
    return {
        "repo_id": _safe_string(getattr(repo, "repo_id", "")),
        "revisions": revisions,
        "size_on_disk": getattr(repo, "size_on_disk", None),
        "last_modified": getattr(repo, "last_modified", None),
    }


def _run_models(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="ironmule models",
        description="List locally cached Hugging Face model snapshots without downloading.",
    )
    parser.add_argument("--model", default=None, help="exact Hugging Face repo id filter")
    args = parser.parse_args(argv)
    try:
        from ironmule.model_identity import scan_local_cache

        cache = scan_local_cache()
        repos = []
        for repo in (getattr(cache, "repos", ()) or ()):
            if getattr(repo, "repo_type", None) != "model":
                continue
            if args.model is not None and getattr(repo, "repo_id", None) != args.model:
                continue
            repos.append(_cached_model(repo))
        repos.sort(key=lambda item: item["repo_id"])
        warnings = sorted(
            _safe_string(warning)
            for warning in (getattr(cache, "warnings", ()) or ())
        )
        result = {"models": repos, "warnings": warnings}
    except ImportError as exc:
        if not _is_huggingface_dependency_error(exc):
            raise
        return _dependency_error("models", exc, dependency="huggingface_hub")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _run_serve(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="ironmule serve",
        description="Serve a local model over an OpenAI-compatible HTTP endpoint "
                    "(POST /v1/chat/completions, GET /v1/models). One request at a time.",
    )
    parser.add_argument("--model", default=None, help="model repo id (default: the tuned default)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-tuned-profile", action="store_true",
                        help="ignore this machine's tuned profile and run the baseline knobs")
    args = parser.parse_args(argv)
    try:
        from ironmule import Runtime
        from ironmule.http import serve
        from ironmule.tune import DEFAULT_MODEL
    except ImportError as exc:
        if not _is_runtime_dependency_error(exc):
            raise
        return _dependency_error("serve", exc)
    try:
        runtime = Runtime.load(args.model or DEFAULT_MODEL,
                               use_tuned_profile=not args.no_tuned_profile)
    except ImportError as exc:
        if not _is_runtime_dependency_error(exc):
            raise
        return _dependency_error("serve", exc)
    with runtime:
        serve(runtime, host=args.host, port=args.port)
    return 0


def _run_benchmark(argv: list[str]) -> int:
    try:
        benchmark_main = _load_benchmark()
    except ImportError as exc:
        if not _is_runtime_dependency_error(exc):
            raise
        return _dependency_error("benchmark", exc)

    try:
        return int(benchmark_main(argv))
    except ImportError as exc:
        if not _is_runtime_dependency_error(exc):
            raise
        return _dependency_error("benchmark", exc)


def main(argv: list[str] | None = None) -> int:
    try:
        return _dispatch(argv)
    except Exception as exc:  # noqa: BLE001 - the CLI reports, the library raises
        # Matched by name, not by import: importing ModelIdentityError pulls in the
        # `ironmule` package, which imports MLX. `ironmule doctor` has to keep working
        # on exactly the machine where that import is the thing that is broken.
        if (type(exc).__name__ != "ModelIdentityError"
                or not type(exc).__module__.startswith("ironmule.")):
            raise
        # A model that is not cached is the normal first-run outcome, not a crash.
        # The exception already carries the exact command that fixes it.
        print(f"ironmule: {exc}", file=sys.stderr)
        return 1


def _dispatch(argv: list[str] | None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: ironmule {doctor|benchmark|serve|models|tune|revalidate|status|info} [options]")
        print("\ncommands:")
        print("  doctor       Check Apple Silicon and MLX prerequisites")
        print("  benchmark   Run the existing reproducible local benchmark")
        print("  serve        Serve a model over an OpenAI-compatible HTTP endpoint")
        print("  models      List locally cached Hugging Face model snapshots")
        print("  tune        Tune or inspect the existing local profile (--show)")
        print("  revalidate  Canary-check the stored profile")
        print("  status       Show local hardware/profile status")
        print("  info        Show package information")
        return 0
    command, rest = args[0], args[1:]
    if command == "doctor":
        return doctor(rest)
    if command == "benchmark":
        return _run_benchmark(rest)
    if command == "serve":
        return _run_serve(rest)
    if command == "tune":
        return _run_tune(rest)
    if command == "models":
        return _run_models(rest)
    if command == "revalidate":
        return _run_revalidate(rest)
    if command == "status":
        return _run_status(rest)
    if command == "info":
        return info(rest)
    print(f"ironmule: unknown command {command!r}; use --help", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
