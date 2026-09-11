"""Fixed stdlib worker entrypoint; only the allowlisted H0 engine is dispatched."""

from __future__ import annotations

import math
import hashlib
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .benchmark import (
    H0_BATCH_MIN_NS,
    H0_MAX_REPETITIONS,
    H0_MAX_WARMUPS,
    REGISTERED_BENCHMARK_ERROR_CODES,
)
from .canonical import canonical_json_bytes
from .decision import evaluate_analysis_fixture
from .manifest import ManifestError
from .protocol import (
    INTERNAL_CONTROL_SLEEP_ENV,
    MANIFEST_FILENAME,
    MANIFEST_SHA_ENV,
    PRODUCTION_MANIFEST_BYTES,
    PRODUCTION_JSON_DEPTH,
    PRODUCTION_RESULT_BYTES,
    RESULT_FILENAME,
    ProtocolError,
    close_manifest,
    read_capped_json,
    validate_result,
    write_json_atomic,
)


_ANALYSIS_KINDS = {
    "analysis_slow": "slow",
    "analysis_known_win": "known_win",
    "analysis_wrong_fixture": "wrong",
    "analysis_missing_data": "missing",
}

_BENCHMARK_MODES = frozenset({"eager_baseline", "compile_comparison", "aa_gpu"})
_DOMAIN_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "mode",
        "manifest_sha256",
        "status",
        "classification",
        "benchmark_classification",
        "action",
        "error",
        "evidence",
        "adapter_contract",
    }
)
_DOMAIN_SUCCESS_EVIDENCE_KEYS = frozenset(
    {
        "fixture",
        "correctness",
        "memory",
        "memory_limit",
        "memory_gate",
        "cache_state",
        "fresh_process_required",
        "aggregation_required",
        "compile_wrapper_setup_ns",
        "first_eval_compile_inclusive_ns",
        "total_elapsed_ns",
        "arms",
        "comparison",
        "raw_samples",
    }
)
_DOMAIN_CLASSES = frozenset(
    {"baseline_reference", "measurement_complete", "runtime_unavailable", "invalid", "invalid: correctness"}
)
_DOMAIN_ACTIONS = frozenset({"not_run", "aggregation_required", "baseline_fallback"})
_COMPLETED_MODE_BINDINGS = {
    "eager_baseline": ("baseline_reference", "not_run", False),
    "compile_comparison": ("measurement_complete", "aggregation_required", True),
    "aa_gpu": ("measurement_complete", "aggregation_required", True),
}
_CONTRACT_KEYS = frozenset({"common_result_ready", "reason", "mapping"})
_CONTRACT_MAPPING = {
    "runtime_unavailable": "invalid/baseline_fallback",
    "invalid*": "invalid/baseline_fallback",
    "measurement_complete": "aggregation_required",
    "baseline_reference": "not_run",
}
_DOMAIN_MAX_BYTES = PRODUCTION_RESULT_BYTES - 32 * 1024
_DOMAIN_AUDIT_MAX_BYTES = 64 * 1024
_DOMAIN_MAX_NODES = 50_000
_DOMAIN_MAX_SEQUENCE = 10_000
_DOMAIN_MAX_STRING = 64 * 1024


class _DomainBudget:
    __slots__ = ("bytes", "maximum_bytes", "nodes", "active")

    def __init__(self, maximum_bytes: int) -> None:
        self.bytes = 0
        self.maximum_bytes = maximum_bytes
        self.nodes = 0
        self.active: set[int] = set()

    def add(self, amount: int) -> None:
        self.bytes += amount
        if self.bytes > self.maximum_bytes:
            raise ProtocolError("domain result exceeds its bounded byte budget")

    def node(self) -> None:
        self.nodes += 1
        if self.nodes > _DOMAIN_MAX_NODES:
            raise ProtocolError("domain result exceeds its node budget")


def _bounded_domain_json(value: Any, *, maximum_bytes: int) -> Any:
    """Copy JSON values under depth, node, string, cycle, and byte budgets."""

    budget = _DomainBudget(maximum_bytes)

    def visit(current: Any, depth: int) -> Any:
        if depth > PRODUCTION_JSON_DEPTH:
            raise ProtocolError("domain result JSON depth exceeds the protocol limit")
        budget.node()
        if current is None:
            budget.add(len(canonical_json_bytes(current)))
            return None
        if isinstance(current, bool):
            budget.add(len(canonical_json_bytes(current)))
            return current
        if isinstance(current, int):
            if not -(1 << 63) <= current <= (1 << 63) - 1:
                raise ProtocolError("domain integer is not signed 64-bit")
            try:
                encoded_length = len(canonical_json_bytes(current))
            except (OverflowError, ValueError) as exc:
                raise ProtocolError("domain integer is not bounded") from exc
            if encoded_length > _DOMAIN_MAX_STRING:
                raise ProtocolError("domain integer is not bounded")
            budget.add(encoded_length)
            return current
        if isinstance(current, float):
            if not math.isfinite(current):
                raise ProtocolError("domain result contains a non-finite value")
            budget.add(len(canonical_json_bytes(current)))
            return current
        if isinstance(current, str):
            encoded = canonical_json_bytes(current)
            if len(encoded) > _DOMAIN_MAX_STRING:
                raise ProtocolError("domain string exceeds its bounded length")
            budget.add(len(encoded))
            return current
        if isinstance(current, dict):
            identity = id(current)
            if identity in budget.active:
                raise ProtocolError("domain result contains a cycle")
            budget.active.add(identity)
            try:
                if len(current) > _DOMAIN_MAX_SEQUENCE:
                    raise ProtocolError("domain object exceeds its key budget")
                result: dict[str, Any] = {}
                for key, child in current.items():
                    if not isinstance(key, str):
                        raise ProtocolError("domain result contains a non-string key")
                    key_bytes = canonical_json_bytes(key)
                    if len(key_bytes) > _DOMAIN_MAX_STRING:
                        raise ProtocolError("domain key exceeds its bounded length")
                    budget.add(len(key_bytes) + 3)
                    result[key] = visit(child, depth + 1)
                return result
            finally:
                budget.active.remove(identity)
        if isinstance(current, (list, tuple)):
            identity = id(current)
            if identity in budget.active:
                raise ProtocolError("domain result contains a cycle")
            budget.active.add(identity)
            try:
                if len(current) > _DOMAIN_MAX_SEQUENCE:
                    raise ProtocolError("domain sequence exceeds its item budget")
                budget.add(2)
                return [visit(child, depth + 1) for child in current]
            finally:
                budget.active.remove(identity)
        raise ProtocolError("domain result contains a non-JSON value")

    return visit(value, 0)


def _finite_json(value: Any) -> Any:
    """Backward-compatible name for the bounded domain copier."""

    return _bounded_domain_json(value, maximum_bytes=_DOMAIN_MAX_BYTES)


def _bounded_domain_text(value: Any, maximum: int = 256) -> str:
    try:
        text = str(value)
    except Exception:
        return "worker benchmark failure"
    if not text:
        return "worker benchmark failure"
    return text if len(text) <= maximum else text[: maximum - 1] + "…"


_DIAGNOSTIC_MAX_INT = (1 << 63) - 1
_DIAGNOSTIC_CODES = REGISTERED_BENCHMARK_ERROR_CODES


def _diagnostic_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= _DIAGNOSTIC_MAX_INT


def _validate_diagnostic_tree(value: Any) -> None:
    """Validate correctness details with the same fail-closed domain budgets."""

    budget = _DomainBudget(_DOMAIN_AUDIT_MAX_BYTES)

    def visit(current: Any, depth: int) -> None:
        if depth > PRODUCTION_JSON_DEPTH:
            raise ProtocolError("correctness diagnostic JSON depth exceeds the protocol limit")
        budget.node()
        if current is None or isinstance(current, bool):
            budget.add(len(canonical_json_bytes(current)))
            return
        if isinstance(current, int):
            if not -(1 << 63) <= current <= (1 << 63) - 1:
                raise ProtocolError("correctness diagnostic integer is not signed 64-bit")
            budget.add(len(canonical_json_bytes(current)))
            return
        if isinstance(current, float):
            if not math.isfinite(current):
                raise ProtocolError("correctness diagnostic contains a non-finite value")
            budget.add(len(canonical_json_bytes(current)))
            return
        if isinstance(current, str):
            encoded = canonical_json_bytes(current)
            if len(encoded) > _DOMAIN_MAX_STRING:
                raise ProtocolError("correctness diagnostic string exceeds its bounded length")
            budget.add(len(encoded))
            return
        if isinstance(current, dict):
            identity = id(current)
            if identity in budget.active:
                raise ProtocolError("correctness diagnostic contains a cycle")
            budget.active.add(identity)
            try:
                if len(current) > _DOMAIN_MAX_SEQUENCE:
                    raise ProtocolError("correctness diagnostic object exceeds its key budget")
                for key, child in current.items():
                    if not isinstance(key, str):
                        raise ProtocolError("correctness diagnostic key is not a string")
                    key_bytes = canonical_json_bytes(key)
                    if len(key_bytes) > _DOMAIN_MAX_STRING:
                        raise ProtocolError("correctness diagnostic key exceeds its bounded length")
                    budget.add(len(key_bytes) + 3)
                    visit(child, depth + 1)
            finally:
                budget.active.remove(identity)
            return
        if isinstance(current, list):
            identity = id(current)
            if identity in budget.active:
                raise ProtocolError("correctness diagnostic contains a cycle")
            budget.active.add(identity)
            try:
                if len(current) > _DOMAIN_MAX_SEQUENCE:
                    raise ProtocolError("correctness diagnostic sequence exceeds its item budget")
                budget.add(2)
                for child in current:
                    visit(child, depth + 1)
            finally:
                budget.active.remove(identity)
            return
        raise ProtocolError("correctness diagnostic contains a non-JSON value")

    visit(value, 0)


def _validate_failure_diagnostic(value: Any, error_code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema_version", "code", "details"}:
        raise ProtocolError("failure diagnostic envelope is not closed")
    schema_version = value["schema_version"]
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise ProtocolError("failure diagnostic schema version is not registered")
    if value["code"] != error_code:
        raise ProtocolError("failure diagnostic code does not match the domain error")
    if schema_version == 2 and error_code != "warmup_unstable":
        raise ProtocolError("schema version 2 is only registered for warmup diagnostics")
    if error_code not in _DIAGNOSTIC_CODES:
        raise ProtocolError("failure diagnostic code is not registered")
    details = value["details"]
    if not isinstance(details, dict):
        raise ProtocolError("failure diagnostic details are not an object")
    if error_code == "evaluation_timeout_observed":
        if set(details) != {"evaluation_ns"} or not _diagnostic_positive_int(details["evaluation_ns"]):
            raise ProtocolError("evaluation timeout diagnostic is invalid")
    elif error_code == "synchronization_timeout_observed":
        if set(details) != {"synchronize_ns"} or not _diagnostic_positive_int(details["synchronize_ns"]):
            raise ProtocolError("synchronization timeout diagnostic is invalid")
    elif error_code == "warmup_unstable":
        if schema_version == 1:
            warmups = details.get("warmups_ns")
            if set(details) != {"warmups_ns"} or not isinstance(warmups, list) or len(warmups) != H0_MAX_WARMUPS or not all(_diagnostic_positive_int(item) for item in warmups):
                raise ProtocolError("warmup diagnostic is invalid")
        else:
            expected_keys = {"warmup_block_per_eval_ns", "warmup_blocks"}
            values = details.get("warmup_block_per_eval_ns")
            blocks = details.get("warmup_blocks")
            if (
                set(details) != expected_keys
                or not isinstance(values, list)
                or len(values) != H0_MAX_WARMUPS
                or not all(_diagnostic_positive_int(item) for item in values)
                or not isinstance(blocks, list)
                or len(blocks) != H0_MAX_WARMUPS
            ):
                raise ProtocolError("warmup v2 diagnostic is invalid")
            block_keys = {
                "block_index", "evaluations", "block_ns", "per_eval_ns",
                "median_eval_ns", "min_eval_ns", "max_eval_ns",
            }
            for index, block in enumerate(blocks):
                if not isinstance(block, dict) or set(block) != block_keys:
                    raise ProtocolError("warmup v2 block is not closed")
                if (
                    type(block["block_index"]) is not int
                    or block["block_index"] != index
                    or not _diagnostic_positive_int(block["evaluations"])
                    or block["evaluations"] > H0_MAX_REPETITIONS
                    or not _diagnostic_positive_int(block["block_ns"])
                    or block["block_ns"] < H0_BATCH_MIN_NS
                    or not _diagnostic_positive_int(block["per_eval_ns"])
                    or any(
                        not isinstance(block[field], int)
                        or isinstance(block[field], bool)
                        or block[field] < 0
                        or block[field] > _DIAGNOSTIC_MAX_INT
                        for field in ("median_eval_ns", "min_eval_ns", "max_eval_ns")
                    )
                    or not (block["min_eval_ns"] <= block["median_eval_ns"] <= block["max_eval_ns"])
                    or block["per_eval_ns"] != max(1, int(round(block["block_ns"] / block["evaluations"])))
                    or values[index] != block["per_eval_ns"]
                ):
                    raise ProtocolError("warmup v2 block is inconsistent")
    elif error_code == "repetition_window_unreachable":
        if details:
            if set(details) != {"repetitions", "batch_ns"}:
                raise ProtocolError("repetition diagnostic is not closed")
            repetitions = details["repetitions"]
            if not _diagnostic_positive_int(repetitions) or repetitions > H0_MAX_REPETITIONS or repetitions & (repetitions - 1) != 0 or not _diagnostic_positive_int(details["batch_ns"]):
                raise ProtocolError("repetition diagnostic is invalid")
    elif error_code == "correctness_failed":
        if set(details) != {"correctness"}:
            raise ProtocolError("correctness diagnostic is invalid")
        _validate_diagnostic_tree(details["correctness"])
        if not isinstance(details["correctness"], dict):
            raise ProtocolError("correctness diagnostic is invalid")
    elif error_code == "total_timeout_observed":
        if set(details) != {"total_ns"} or not _diagnostic_positive_int(details["total_ns"]):
            raise ProtocolError("total timeout diagnostic is invalid")
    elif error_code == "result_too_large":
        if details != {"truncated": True, "missing_reason": "result_limit"}:
            raise ProtocolError("result-size diagnostic is invalid")
    elif details != {}:
        raise ProtocolError("unexpected diagnostic details for error code")
    return value


def _fallback_benchmark_result(
    manifest: Any,
    *,
    code: str,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "run_id": manifest.run_id,
        "mode": manifest.mode,
        "manifest_sha256": manifest.sha256,
        "status": "invalid",
        "classification": "invalid",
        "action": "baseline_fallback",
        "error": {"code": _bounded_domain_text(code), "message": _bounded_domain_text(message)},
        "evidence": dict(evidence or {}),
    }
    return validate_result(result, manifest=manifest)


def _benchmark_result(closed: Any, domain: Any, base_evidence: dict[str, Any]) -> dict[str, Any]:
    """Adapt one benchmark-domain result into the closed common worker result."""

    try:
        safe = _finite_json(domain)
        if set(safe) != _DOMAIN_RESULT_KEYS:
            raise ProtocolError("domain result has unknown or missing keys")
        if type(safe["schema_version"]) is not int or safe["schema_version"] != 1:
            raise ProtocolError("domain result schema version is not 1")
        if safe["run_id"] != closed.run_id or safe["mode"] != closed.mode:
            raise ProtocolError("domain result is not bound to the manifest")
        if safe["manifest_sha256"] != closed.sha256:
            raise ProtocolError("domain result manifest hash mismatch")
        domain_class = safe["benchmark_classification"]
        if safe["classification"] != domain_class or domain_class not in _DOMAIN_CLASSES:
            raise ProtocolError("domain classification is not allowlisted")
        domain_action = safe["action"]
        if domain_action not in _DOMAIN_ACTIONS:
            raise ProtocolError("domain action is not allowlisted")
        domain_status = safe["status"]
        if domain_status not in {"completed", "invalid"}:
            raise ProtocolError("domain status is not allowlisted")
        domain_error = safe["error"]
        if domain_error is not None:
            if not isinstance(domain_error, dict) or set(domain_error) != {"code", "message"}:
                raise ProtocolError("domain error is not closed")
            if any(
                not isinstance(domain_error[field], str) or not domain_error[field]
                for field in ("code", "message")
            ):
                raise ProtocolError("domain error fields are not bounded strings")
            if domain_error["code"] not in _DIAGNOSTIC_CODES:
                raise ProtocolError("domain error code is not registered")
            domain_error = {
                "code": _bounded_domain_text(domain_error["code"]),
                "message": _bounded_domain_text(domain_error["message"]),
            }
        contract = safe["adapter_contract"]
        if not isinstance(contract, dict) or set(contract) != _CONTRACT_KEYS:
            raise ProtocolError("domain adapter contract is not closed")
        if type(contract["common_result_ready"]) is not bool or contract["common_result_ready"] is not False:
            raise ProtocolError("domain adapter contract must not claim common-result readiness")
        reason = contract["reason"]
        if not isinstance(reason, str) or not reason or len(reason) > 256:
            raise ProtocolError("domain adapter contract reason is not bounded")
        mapping = contract["mapping"]
        if mapping != _CONTRACT_MAPPING:
            raise ProtocolError("domain adapter contract mapping is not registered")
        domain_evidence = safe["evidence"]
        if not isinstance(domain_evidence, dict):
            raise ProtocolError("domain evidence is not an object")
        if domain_status == "completed":
            if set(domain_evidence) != _DOMAIN_SUCCESS_EVIDENCE_KEYS:
                raise ProtocolError("completed domain evidence does not match the registered success schema")
            if type(domain_evidence["aggregation_required"]) is not bool:
                raise ProtocolError("completed aggregation_required must be a boolean")
            if domain_evidence["aggregation_required"] is not (domain_action == "aggregation_required"):
                raise ProtocolError("completed aggregation_required disagrees with the domain action")
            expected_binding = _COMPLETED_MODE_BINDINGS.get(closed.mode)
            if expected_binding is None or (
                domain_class,
                domain_action,
                domain_evidence["aggregation_required"],
            ) != expected_binding:
                raise ProtocolError("completed domain result mode binding is invalid")
        else:
            if set(domain_evidence) != {"failure_diagnostic"}:
                raise ProtocolError("invalid domain evidence must contain only failure diagnostic")
            _validate_failure_diagnostic(domain_evidence["failure_diagnostic"], domain_error["code"] if domain_error else "")

        projected = dict(base_evidence)
        projected.update(
            {
                "benchmark_classification": domain_class,
                "benchmark_action": domain_action,
                    "aggregation_required": (
                        domain_evidence["aggregation_required"]
                        if domain_status == "completed"
                        else domain_action == "aggregation_required"
                    ),
                "adapter_contract": contract,
                "benchmark_evidence": domain_evidence,
            }
        )

        if domain_class in {"measurement_complete", "baseline_reference"}:
            expected = (
                ("completed", "aggregation_required", None)
                if domain_class == "measurement_complete"
                else ("completed", "not_run", None)
            )
            if (domain_status, domain_action, domain_error) != expected:
                raise ProtocolError("domain neutral result has an invalid status/action/error")
            result = {
                "schema_version": 1,
                "run_id": closed.run_id,
                "mode": closed.mode,
                "manifest_sha256": closed.sha256,
                "status": "completed",
                "classification": "measurement_complete",
                "action": "baseline_fallback",
                "error": None,
                "evidence": projected,
            }
        else:
            if domain_status != "invalid" or domain_action != "baseline_fallback" or domain_error is None:
                raise ProtocolError("domain invalid result has an invalid status/action/error")
            classification = (
                "runtime_unavailable"
                if domain_class == "runtime_unavailable"
                else "invalid: correctness"
                if domain_class == "invalid: correctness" or domain_error["code"] == "correctness_failed"
                else "invalid"
            )
            result = {
                "schema_version": 1,
                "run_id": closed.run_id,
                "mode": closed.mode,
                "manifest_sha256": closed.sha256,
                "status": "invalid",
                "classification": classification,
                "action": "baseline_fallback",
                "error": domain_error,
                "evidence": projected,
            }

        encoded = canonical_json_bytes(result)
        if len(encoded) > PRODUCTION_RESULT_BYTES:
            raise ProtocolError("common result exceeds the production byte limit")
        return validate_result(result, manifest=closed)
    except (ProtocolError, RecursionError, TypeError, ValueError) as exc:
        # Never pass untrusted domain evidence through a fallback.  A small digest
        # is retained only when the raw value is itself finite/canonicalizable.
        summary: dict[str, Any] = dict(base_evidence)
        try:
            audited = _bounded_domain_json(domain, maximum_bytes=_DOMAIN_AUDIT_MAX_BYTES)
            raw = canonical_json_bytes(audited)
        except (ProtocolError, RecursionError, TypeError, ValueError):
            raw = b""
        if raw and len(raw) <= _DOMAIN_AUDIT_MAX_BYTES:
            summary.update(
                {
                    "domain_evidence_bytes": len(raw),
                    "domain_evidence_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        return _fallback_benchmark_result(
            closed,
            code="benchmark_domain_invalid",
            message=_bounded_domain_text(str(exc)),
            evidence=summary,
        )


def _rss_sample() -> tuple[int | None, str | None]:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=0.10,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            close_fds=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, "unavailable"
    if completed.returncode != 0:
        return None, "ps_exit"
    try:
        rss_kib = int(completed.stdout.decode("ascii", errors="strict").strip())
    except (UnicodeDecodeError, ValueError):
        return None, "ps_parse"
    if rss_kib < 0:
        return None, "ps_negative"
    return rss_kib * 1024, None


def _error_result(manifest: Any, *, code: str, message: str) -> dict[str, Any]:
    rss, reason = _rss_sample()
    result = {
        "schema_version": 1,
        "run_id": manifest.run_id,
        "mode": manifest.mode,
        "manifest_sha256": manifest.sha256,
        "status": "invalid",
        "classification": "invalid",
        "action": "baseline_fallback",
        "error": {"code": _bounded_domain_text(code), "message": _bounded_domain_text(message)},
        "evidence": {"rss_peak_bytes": rss, "rss_missing_reason": reason},
    }
    return validate_result(result, manifest=manifest)


def _write_result(manifest: Any, result: dict[str, Any]) -> None:
    validate_result(result, manifest=manifest)
    write_json_atomic(Path.cwd() / RESULT_FILENAME, result, limit=PRODUCTION_RESULT_BYTES)


def _load_closed_manifest() -> Any:
    value, raw = read_capped_json(Path.cwd() / MANIFEST_FILENAME, limit=PRODUCTION_MANIFEST_BYTES)
    closed = close_manifest(value)
    expected_sha = os.environ.get(MANIFEST_SHA_ENV, "")
    if expected_sha != closed.sha256 or raw != closed.canonical_bytes:
        raise ProtocolError("manifest bytes do not match the controlled parent hash")
    return closed


def _control_sleep_seconds() -> float:
    default = 121.0
    raw = os.environ.get(INTERNAL_CONTROL_SLEEP_ENV)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value) or value <= 0.0 or value > 120.0:
        return default
    return value


def _run(closed: Any) -> int:
    mode = closed.mode
    if mode == "control_timeout":
        time.sleep(_control_sleep_seconds())
        return 0
    if mode == "control_exit_70":
        os._exit(70)

    rss, reason = _rss_sample()
    evidence: dict[str, Any] = {"rss_peak_bytes": rss, "rss_missing_reason": reason}
    if mode in _BENCHMARK_MODES:
        try:
            benchmark = importlib.import_module(".benchmark", __package__)
            domain = benchmark.run_mlx_benchmark(closed.value)
            result = _benchmark_result(closed, domain, evidence)
        except (SystemExit, RecursionError) as exc:
            result = _fallback_benchmark_result(
                closed,
                code="benchmark_exception",
                message=f"{type(exc).__name__}: {_bounded_domain_text(exc)}",
                evidence=evidence,
            )
        except Exception as exc:
            result = _fallback_benchmark_result(
                closed,
                code="benchmark_exception",
                message=f"{type(exc).__name__}: {_bounded_domain_text(exc)}",
                evidence=evidence,
            )
        _write_result(closed, result)
        return 0
    if mode in _ANALYSIS_KINDS:
        decision = evaluate_analysis_fixture(_ANALYSIS_KINDS[mode])
        classification = decision["classification"]
        result = {
            "schema_version": 1,
            "run_id": closed.run_id,
            "mode": mode,
            "manifest_sha256": closed.sha256,
            "status": "invalid" if classification.startswith("invalid") else "completed",
            "classification": classification,
            "action": decision["action"],
            "error": (
                {"code": "analysis_invalid", "message": classification}
                if classification.startswith("invalid")
                else None
            ),
            "evidence": {**evidence, "decision": decision},
        }
        _write_result(closed, result)
        return 0
    raise ProtocolError(f"worker mode is not executable: {mode}")


def main() -> int:
    closed = None
    try:
        closed = _load_closed_manifest()
        return _run(closed)
    except (ProtocolError, ManifestError, RecursionError, ValueError, OSError) as exc:
        if closed is None:
            return 64
        try:
            _write_result(closed, _error_result(closed, code="worker_protocol", message=str(exc)))
            return 0
        except (ProtocolError, OSError):
            return 66


if __name__ == "__main__":
    sys.exit(main())
