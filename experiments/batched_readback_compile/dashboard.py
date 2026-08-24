#!/usr/bin/env python3
"""Read-only localhost dashboard for Cycle17 batched-readback evidence.

The current study/run/candidate identity is strict.  The HTTP projection never
contains prompts, decoded/raw text, token arrays, local paths, stderr, or
arbitrary model output; Cycle15/16 remain scalar historical comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path(__file__).with_name("results.json")
HISTORY_PATHS = {15: ROOT / "experiments/dual_model_planner/results.json",
                 16: ROOT / "experiments/matmul_compile_ab/results.json"}
STUDY_ID = "fixed-compiled-batched-readback-20260824-01"
RUN_ID = "fixed-compiled-batched-readback-validation-20260824-01"
CANDIDATE_ID = "fixed_compiled_batched_readback_n8_v1"
MAX_RESULT_BYTES = 8_000_000
MAX_RESPONSE_BYTES = 64_000
PRIVATE_KEYS = {"prompt", "prompt_tokens", "prompt_token_ids", "rendered_prompt", "text",
                "visible_text", "visible_output", "output_text", "raw_output", "decoded_output",
                "raw_text", "decoded_text", "model_text", "token_ids", "tokens",
                "physical_tokens", "logical_tokens", "visible_tokens", "stderr", "path",
                "snapshot_path", "weight_path", "outputs", "events", "worker_events",
                "stderr_tail", "error", "message", "traceback"}
SAFE_DECISIONS = {"runtime_readback8_wins_exact_scope", "readback8_regression_baseline_retained",
                  "no_clear_speedup_baseline_retained", "candidate_not_runnable",
                  "correctness_failed", "resource_or_budget_failed", "not_available"}
CURRENT_KEYS = {
    "schema_version", "study_id", "run_id", "candidate_id", "formal_claim", "runs",
    "worker_events", "error", "partial_result", "completed_at_unix_ns", "decision",
    "budget", "resources", "gates", "metrics", "provenance", "thresholds",
    "snapshot_postflight",
}
HISTORY_IDS = {
    15: ("dual-model-evidence-planner-20260824-01", "dual-model-evidence-planner-validation-20260824-01"),
    16: ("matmul-compile-ab-20260824-01", "matmul-compile-validation-20260824-01"),
}
HISTORY_KEYS = {
    15: {"budget", "completed_at_unix_ns", "decision", "error", "finalization_errors", "formal_claim",
         "gates", "metrics", "partial_result", "provenance", "resources", "run_id", "runs",
         "schema_version", "snapshot_postflight", "study_id", "thresholds"},
    16: {"budget", "completed_at_unix_ns", "decision", "error", "formal_claim", "gates", "metrics",
         "partial_result", "provenance", "resources", "run_id", "runs", "schema_version",
         "snapshot_postflight", "study_id", "thresholds", "worker_events"},
}


class DashboardError(RuntimeError):
    pass


def _parse_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise DashboardError("not_available")
    payload = path.read_bytes()
    if not payload or len(payload) > MAX_RESULT_BYTES:
        raise DashboardError("not_available")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise DashboardError("invalid_json") from exc
    if not isinstance(value, dict):
        raise DashboardError("invalid_schema")
    return value


def _finite(value: Any) -> int | float | None:
    if value is None or type(value) is bool or type(value) not in (int, float):
        return None
    return value if math.isfinite(float(value)) else None


DISPLAY_KEYS = {
    "arms", "paired", "derived", "runs_completed", "complete", "count", "median", "mad", "min", "max",
    "p50", "p95", "p99", "values", "decode_critical_path", "token_rate", "ttft", "prefill",
    "cache_conversion", "arm_wall", "observed_model_work", "readback", "readback_block", "block_latency",
    "host_boundary", "host_available_total", "host_boundary_gap", "worker_process_wall", "parent_process_wall",
    "rss_peak_bytes", "mlx_peak_bytes", "swap_delta_bytes", "primary", "ratios", "lower", "upper",
    "bootstrap_95_ci", "seed", "resamples", "blocks", "calculated_only", "warmed_decode_ratio_median",
    "cold_decode_ratio_median", "warmed_decode_ratio_values", "cold_decode_ratio_values",
    "break_even_decode_forwards", "measured_decode_saving_per_forward_seconds", "cold_setup_seconds",
    "runtime_readback8_wins_exact_scope", "readback8_regression_baseline_retained",
    "no_clear_speedup_baseline_retained", "candidate_not_runnable", "correctness_failed",
    "resource_or_budget_failed", "not_available", "model_work_seconds", "process_wall_seconds",
    "ttft_seconds", "arm_wall_seconds", "decode_total", "intertoken_p50", "intertoken_p95", "intertoken_p99",
    "model_work", "process_wall", "token_rate", "exact_text_equal_count", "runs", "priority_successes",
    "contract_successes", "correctness_pass", "deterministic", "identity_pass", "functional_pass",
    "peak_mlx_bytes", "peak_rss_bytes", "swap_deltas_bytes", "pairwise", "ratios_1b_div_4b",
    "fixed_compiled_div_fixed_eager", "fixed_compiled_div_standard_eager", "method", "statistic",
    "percentiles", "interpolation", "1b", "4b",
    "fixed_compiled", "fixed_eager", "standard_eager", "model_1b", "model_4b", "cross_model_text",
}


def _safe(value: Any, depth: int = 0) -> Any:
    """Project only allowlisted numeric dashboard fields; never pass strings through."""
    if depth > 3:
        return None
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if (str(key) not in DISPLAY_KEYS or lowered in PRIVATE_KEYS
                    or lowered.endswith("_path") or "/" in str(key) or "\\" in str(key)):
                continue
            projected = _safe(item, depth + 1)
            if projected is not None:
                out[str(key)] = projected
        return out
    if isinstance(value, list):
        if len(value) > 16 or any(isinstance(item, (dict, list, str)) for item in value):
            return None
        return [_safe(item, depth + 1) for item in value]
    return _finite(value)


def _sha256(path: Path) -> str | None:
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_RESULT_BYTES:
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _identity(raw: dict[str, Any], cycle: int) -> bool:
    if cycle != 17:
        return True
    return ((raw.get("study_id"), raw.get("run_id"), raw.get("candidate_id")) ==
            (STUDY_ID, RUN_ID, CANDIDATE_ID) and raw.get("formal_claim") is False)


def _valid_current(raw: dict[str, Any]) -> bool:
    if set(raw) != CURRENT_KEYS or raw.get("schema_version") != 1 or not _identity(raw, 17):
        return False
    if not isinstance(raw.get("runs"), list) or not isinstance(raw.get("budget"), dict):
        return False
    if not isinstance(raw.get("resources"), dict) or not isinstance(raw.get("gates"), dict):
        return False
    if not isinstance(raw.get("metrics"), dict) or not isinstance(raw.get("provenance"), dict):
        return False
    if not isinstance(raw.get("thresholds"), dict) or not isinstance(raw.get("partial_result"), bool):
        return False
    if raw.get("decision") not in SAFE_DECISIONS:
        return False
    metrics = raw["metrics"]
    if not isinstance(metrics.get("arms"), dict) or not isinstance(metrics.get("paired"), dict):
        return False
    for run in raw["runs"]:
        if not isinstance(run, dict) or type(run.get("block")) is not int:
            return False
        if run.get("status") not in {"complete", "correctness_failed", "candidate_not_runnable",
                                      "resource_or_budget_failed", "error"}:
            return False
    return True


def _valid_history(raw: dict[str, Any], cycle: int) -> bool:
    expected = HISTORY_IDS[cycle]
    return (set(raw) == HISTORY_KEYS[cycle] and raw.get("schema_version") == 1
            and (raw.get("study_id"), raw.get("run_id")) == expected
            and raw.get("formal_claim") is False
            and isinstance(raw.get("runs"), list)
            and isinstance(raw.get("metrics"), dict)
            and isinstance(raw.get("gates"), dict)
            and isinstance(raw.get("budget"), dict)
            and isinstance(raw.get("resources"), dict))


def _project_current(raw: dict[str, Any], digest: str | None) -> dict[str, Any]:
    if not _valid_current(raw):
        return {"cycle": 17, "available": False, "status": "identity_invalid", "sha256": digest}
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
    source_arms = metrics.get("arms") if isinstance(metrics.get("arms"), dict) else {}
    arms = {name: _safe(source_arms[name]) for name in
            ("fixed_compiled_readback_1", "fixed_compiled_readback_8") if name in source_arms}
    decision = raw.get("decision") if raw.get("decision") in SAFE_DECISIONS else "not_available"
    return {
        "cycle": 17, "available": True, "study_id": STUDY_ID, "run_id": RUN_ID,
        "candidate_id": CANDIDATE_ID, "formal_claim": False, "decision": decision,
        "partial_result": raw.get("partial_result") is True,
        "measured": {"arms": arms, "paired": _safe(metrics.get("paired")),
                     "runs_completed": len(raw.get("runs", [])) if isinstance(raw.get("runs"), list) else 0},
        "calculated": _safe(raw.get("thresholds")), "resources": _safe(raw.get("resources")),
        "budget": _safe(raw.get("budget")), "gates": _safe(raw.get("gates")), "sha256": digest,
    }


def _project_history(raw: dict[str, Any], cycle: int, digest: str | None) -> dict[str, Any]:
    if not _valid_history(raw, cycle):
        return {"cycle": cycle, "available": False, "status": "history_schema_invalid", "sha256": digest}
    safe = _safe(raw)
    return {"cycle": cycle, "available": True,
            "study_id": raw.get("study_id") if raw.get("study_id") in {HISTORY_IDS[cycle][0]} else "unavailable",
            "decision": raw.get("decision") if raw.get("decision") in SAFE_DECISIONS else "not_available",
            "formal_claim": raw.get("formal_claim") is True,
            "measured": safe.get("metrics", safe.get("measured", {})) if isinstance(safe, dict) else {},
            "calculated": safe.get("calculated", safe.get("derived", {})) if isinstance(safe, dict) else {},
            "sha256": digest}


def _project_file(path: Path, cycle: int) -> dict[str, Any]:
    digest = _sha256(path)
    try:
        raw = _parse_json(path)
    except DashboardError as exc:
        return {"cycle": cycle, "available": False, "status": str(exc), "sha256": digest}
    return _project_current(raw, digest) if cycle == 17 else _project_history(raw, cycle, digest)


def snapshot() -> dict[str, Any]:
    return {"schema_version": 1, "study_id": STUDY_ID, "candidate_id": CANDIDATE_ID,
            "formal_claim": False, "current": _project_file(RESULT_PATH, 17),
            "history": [_project_file(path, cycle) for cycle, path in HISTORY_PATHS.items()],
            "history_cycles": [15, 16, 17], "read_only": True}


def _html_snapshot(value: dict[str, Any]) -> bytes:
    rows = []
    for item in [*value["history"], value["current"]]:
        rows.append("<tr><td>Cycle %s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            html.escape(str(item.get("cycle", ""))), html.escape(str(item.get("study_id", "unavailable"))),
            html.escape(str(item.get("decision", item.get("status", "unavailable")))),
            "measured" if item.get("available") else "not available"))
    return ("<!doctype html><meta charset=utf-8><title>Project Friday Readback</title>"
            "<h1>Fixed-Compiled Batched Readback</h1>"
            "<p>Cycle17 A/B: fixed compiled, readback interval 1 versus 8. "
            "Only measured scalar projections are shown.</p><table>"
            "<thead><tr><th>Cycle</th><th>Study</th><th>Decision</th><th>State</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>").encode()


class Handler(BaseHTTPRequestHandler):
    server_version = "FridayCycle17Dashboard/1"

    def _host_ok(self) -> bool:
        raw = self.headers.get("Host")
        if not raw or raw != raw.strip() or any(char.isspace() for char in raw):
            return False
        port = int(self.server.server_address[1])
        match = re.fullmatch(r"(localhost|127\.0\.0\.1):(\d+)", raw.lower())
        if match:
            return int(match.group(2)) == port
        match = re.fullmatch(r"\[::1\]:(\d+)", raw.lower())
        return bool(match and int(match.group(1)) == port)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        if len(body) > MAX_RESPONSE_BYTES:
            status, body, content_type = 500, b'{"error":"response_too_large"}', "application/json"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _dispatch(self) -> None:
        if not self._host_ok():
            self._send(421, b'{"error":"misdirected_request"}', "application/json")
            return
        path = urlsplit(self.path).path
        if path == "/":
            self._send(200, _html_snapshot(snapshot()), "text/html; charset=utf-8")
        elif path == "/api/snapshot":
            body = json.dumps(snapshot(), separators=(",", ":"), allow_nan=False).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b'{"error":"not_found"}', "application/json")

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch()

    def _read_only(self) -> None:
        foreign = not self._host_ok()
        self._send(421 if foreign else 405,
                   b'{"error":"misdirected_request"}' if foreign else b'{"error":"read_only"}',
                   "application/json")

    def do_POST(self) -> None: self._read_only()  # noqa: N802, E704
    do_PUT = do_POST
    do_PATCH = do_POST
    do_DELETE = do_POST
    do_OPTIONS = do_POST
    do_TRACE = do_POST

    def log_message(self, *_args: Any) -> None:
        return


def make_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("loopback bind only")
    return ThreadingHTTPServer((host, port), Handler)


create_server = make_server


def _self_check() -> int:
    synthetic = {"study_id": STUDY_ID, "run_id": RUN_ID, "candidate_id": CANDIDATE_ID,
                 "schema_version": 1, "worker_events": [], "error": None,
                 "completed_at_unix_ns": 1, "budget": {}, "resources": {}, "gates": {},
                 "provenance": {}, "snapshot_postflight": {},
                 "formal_claim": False, "decision": "no_clear_speedup_baseline_retained",
                 "partial_result": False, "runs": [{"status": "complete", "block": 1}],
                 "metrics": {"arms": {"fixed_compiled_readback_1": {"median": 1.0, "/secret/path": "leak"},
                                        "fixed_compiled_readback_8": {"median": 0.9, "visible_text": "raw"}}, "paired": {"/secret": "leak"}},
                 "thresholds": {"bootstrap_seed": 20260824, "bootstrap_resamples": 10000}}
    projected = _project_current(synthetic, "0" * 64)
    assert projected["available"] and projected["formal_claim"] is False
    encoded = json.dumps(projected, allow_nan=False).lower()
    assert all(term not in encoded for term in ("physical_tokens", "logical_tokens", "visible_tokens", "prompt", "stderr", "secret", "leak", "raw"))
    for cycle in (15, 16):
        history = {key: None for key in HISTORY_KEYS[cycle]}
        history.update({"schema_version": 1, "study_id": HISTORY_IDS[cycle][0],
                        "run_id": HISTORY_IDS[cycle][1], "formal_claim": False,
                        "runs": [], "metrics": {"/secret/path": "leak"}, "gates": {},
                        "budget": {}, "resources": {}, "provenance": {}})
        projected_history = _project_history(history, cycle, "0" * 64)
        history_encoded = json.dumps(projected_history, allow_nan=False).lower()
        assert "/secret" not in history_encoded and "leak" not in history_encoded
    assert not _identity({"study_id": "wrong", "run_id": RUN_ID, "candidate_id": CANDIDATE_ID, "formal_claim": False}, 17)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="read-only Cycle17 dashboard")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    if args.show or not args.serve:
        print(json.dumps(snapshot(), sort_keys=True, allow_nan=False))
        return 0
    server = make_server(args.host, args.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
