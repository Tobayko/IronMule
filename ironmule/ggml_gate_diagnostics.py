"""A bounded, content-free view of a GGUF qualification failure for progress and readiness reports."""
from __future__ import annotations

import re

SCHEMA = "ironmule.gguf-gate-diagnostic.v1"
STAGES = ("tokenize", "tuning", "reference", "candidate", "acceptance", "model_changed")
CODES = ("out_of_memory", "missing_cuda_library", "missing_cpu_library", "missing_dependency", "missing_fixture",
         "invalid_argument", "cuda_kernel_error", "timeout", "child_signal", "child_failed", "invalid_result",
         "quality_rejected", "model_changed")
EXCEPTIONS = ("MemoryError", "OutOfMemoryError", "ModuleNotFoundError", "ImportError", "FileNotFoundError",
              "OSError", "RuntimeError", "ValueError", "TypeError", "OverflowError", "IndexError", "KeyError",
              "JSONDecodeError", "TimeoutExpired", "TimeoutError", "AssertionError", "UnicodeDecodeError")
SETTINGS = ("flash_attn", "kv_unified", "kv_f32", "kv_q8")
COUNTS = ("total", "reference_correct", "repeat_correct", "candidate_correct")


def _number(n):
    return n if type(n) in (int, float) and 0 < n <= 1e300 else None


def sanitize(value):
    """Project untrusted JSON onto whitelisted field names, enums and bounded numeric values."""
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("stage") not in STAGES
            or value.get("code") not in CODES):
        return None
    status, exc = value.get("exit_code"), value.get("exception_class")
    result = {"schema": SCHEMA, "stage": value["stage"], "code": value["code"],
              "exit_code": status if type(status) is int and -255 <= status <= 255 else None,
              "exception_class": exc if exc in EXCEPTIONS else None, "candidates": []}
    candidates = value.get("candidates")
    for row in candidates[:12] if isinstance(candidates, list) else ():
        if not isinstance(row, dict) or row.get("role") not in ("option", "shared", "streams"):
            continue
        settings, scores, facts = row.get("settings"), row.get("factual"), {}
        if not isinstance(settings, dict) or type(row.get("passes")) is not bool:
            continue
        for category in ("arithmetic", "extraction", "facts"):
            counts = scores.get(category) if isinstance(scores, dict) else None
            if (isinstance(counts, dict) and all(type(counts.get(k)) is int and
                    0 <= counts[k] <= 64 for k in COUNTS) and all(counts[k] <= counts["total"] for k in COUNTS[1:])):
                facts[category] = {k: counts[k] for k in COUNTS}
        launches = row.get("kg5_launches")
        result["candidates"].append({"role": row["role"], "passes": row["passes"],
            "settings": {k: settings[k] for k in SETTINGS if type(settings.get(k)) is bool},
            "nll_upper": _number(row.get("nll_upper")), "bound": _number(row.get("bound")),
            "kg5_launches": launches if type(launches) is int and 0 <= launches <= 2147483647 else None,
            "dispatch_required": row.get("dispatch_required") is True, "factual": facts})
    return result


def make(stage, code, *, status=None, exception=None, candidates=()):
    return sanitize({"schema": SCHEMA, "stage": stage, "code": code, "exit_code": status, "exception_class": exception,
                     "candidates": list(candidates)})


def failed_child(stage, stderr, status=None, exception=None):
    # Text is inspected privately. Never return or include a substring of it in a reason.
    raw = stderr[-65536:] if isinstance(stderr, str) else ""
    exc = type(exception).__name__ if exception is not None else None
    if exc is None:
        for line in reversed(raw.splitlines()):
            match = re.match(r"^(?:[\w.]+\.)?(\w+)(?=:|$)", line.strip())
            if match and match[1] in EXCEPTIONS:
                exc = match[1]
                break
    message = raw.lower()
    missing = any(k in message for k in ("cannot open", "not found", "no such file", "could not load",
                                         "failed to load", "cannot find", "undefined symbol"))
    if exc in ("TimeoutExpired", "TimeoutError"):
        code = "timeout"
    elif exc in ("OutOfMemoryError", "MemoryError") or any(k in message for k in
            ("out of memory", "cuda_error_out_of_memory", "cudaerrormemoryallocation", "std::bad_alloc",
             "cannot allocate memory")):
        code = "out_of_memory"
    elif missing and any(k in message for k in ("libcuda", "libcublas", "libggml-cuda", "libnvrtc")):
        code = "missing_cuda_library"
    elif missing and any(k in message for k in ("libllama", "libggml", "libgomp", "libomp")):
        code = "missing_cpu_library"
    elif exc in ("ImportError", "ModuleNotFoundError"):
        code = "missing_dependency"
    elif exc == "FileNotFoundError" or missing and any(k in message for k in ("fixture", "snapshot", ".gguf", "tokenizer")):
        code = "missing_fixture"
    elif "invalid argument" in message or "invalid configuration argument" in message:
        code = "invalid_argument"
    elif any(k in message for k in ("no kernel image", "cuda error", "cuda_error", "cublas_status",
                                     "invalid device function", "kernel launch", "hip error", "hiperror")):
        code = "cuda_kernel_error"
    elif exc in ("JSONDecodeError", "UnicodeDecodeError"):
        code = "invalid_result"
    elif exc in ("ValueError", "TypeError"):
        code = "invalid_argument"
    elif isinstance(status, int) and status < 0:
        code = "child_signal"
    else:
        code = "child_failed"
    return make(stage, code, status=status, exception=exc)


def candidate(row, bound, required, role="option"):
    interval = row.get("interval") or []
    return {"role": role, "settings": row.get("settings", {}), "passes": row.get("passes", False),
            "nll_upper": interval[1] if len(interval) == 2 else None, "bound": bound, "kg5_launches": row.get("kg5_launches"),
            "dispatch_required": required and row.get("settings", {}).get("flash_attn") is True,
            "factual": (row.get("factual") or {}).get("by_category", {})}
