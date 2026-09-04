#!/usr/bin/env python3
"""Read-only loopback dashboard for the cycle-16 matmul/compile study.

The dashboard is deliberately a small projection layer.  It never serves raw
model text, token arrays, prompts, paths, or stored error messages and it never
writes a result, marker, database, or cache.  Missing or malformed evidence is
represented by a stable ``not_available`` snapshot.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

RESULT_PATH = Path(__file__).with_name("results.json")
HISTORY_PATH = Path(__file__).parents[1] / "dual_model_planner" / "results.json"
EXPECTED_STUDY_ID = "matmul-compile-ab-20260824-01"
EXPECTED_RUN_ID = "matmul-compile-validation-20260824-01"
HISTORY_STUDY_ID = "dual-model-evidence-planner-20260824-01"
HISTORY_RUN_ID = "dual-model-evidence-planner-validation-20260824-01"
MAX_RESULT_BYTES = 8_000_000
MAX_HISTORY_BYTES = 8_000_000
ARM_NAMES = ("standard_eager", "fixed_eager", "fixed_compiled")
PAIR_METRICS = ("decode_total", "intertoken_p50", "intertoken_p95", "intertoken_p99")
ALLOWED_DECISIONS = {
    "runtime_compile_wins_exact_scope",
    "compile_gain_no_system_gain",
    "fixed_cache_gain_not_compile_gain",
    "no_clear_speedup_baseline_retained",
    "compile_regression_baseline_retained",
    "candidate_not_runnable",
    "correctness_failed",
    "resource_or_budget_failed",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_FINISH = {"complete", "correctness_failed", "candidate_not_runnable", "resource_or_budget_failed", "error"}


class DashboardError(RuntimeError):
    """Evidence is unavailable or does not match the sealed study schema."""


def _parse_json(path: Path, limit: int) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise DashboardError("result_unavailable")
    payload = path.read_bytes()
    if not payload or len(payload) > limit:
        raise DashboardError("result_unavailable")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate")
            value[key] = item
        return value

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise DashboardError("result_invalid") from exc
    if not isinstance(value, dict):
        raise DashboardError("result_invalid")
    return value


_RESULT_KEYS = {
    "schema_version", "study_id", "run_id", "formal_claim", "decision", "runs",
    "worker_events", "provenance", "partial_result", "error", "budget", "resources", "gates",
    "metrics", "snapshot_postflight", "thresholds", "completed_at_unix_ns",
}


def _load_result(path: Path = RESULT_PATH) -> dict[str, Any]:
    value = _parse_json(path, MAX_RESULT_BYTES)
    if set(value) - _RESULT_KEYS:
        raise DashboardError("result_schema_invalid")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or type(value.get("study_id")) is not str
        or value.get("study_id") != EXPECTED_STUDY_ID
        or type(value.get("run_id")) is not str
        or value.get("run_id") != EXPECTED_RUN_ID
        or value.get("formal_claim") is not False
        or type(value.get("decision")) is not str
        or value.get("decision") not in ALLOWED_DECISIONS
        or not isinstance(value.get("runs"), list)
        or len(value["runs"]) > 6
    ):
        raise DashboardError("result_identity_invalid")
    if not isinstance(value.get("metrics"), dict):
        raise DashboardError("result_schema_invalid")
    return value


def _finite(value: Any, *, allow_none: bool = True) -> float | None:
    if value is None and allow_none:
        return None
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise DashboardError("numeric_field_invalid")
    return float(value)


def _positive_int(value: Any, *, allow_none: bool = True) -> int | None:
    if value is None and allow_none:
        return None
    if type(value) is not int or value <= 0:
        raise DashboardError("integer_field_invalid")
    return value


def _hash(value: Any, *, allow_none: bool = True) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise DashboardError("hash_field_invalid")
    return value


def _hash_list(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) > 6:
        raise DashboardError("hash_list_invalid")
    return [_hash(item, allow_none=False) for item in value]  # type: ignore[list-item]


def _metric(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DashboardError("metric_invalid")
    return {
        "median": _finite(value.get("median")),
        "mad": _finite(value.get("mad")),
        "p50": _finite(value.get("p50")),
        "p95": _finite(value.get("p95")),
        "p99": _finite(value.get("p99")),
    }


def _arm_projection(source: Any, arm: str) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise DashboardError("arm_metrics_invalid")
    if source.get("arm") != arm or type(source.get("runs")) is not int:
        raise DashboardError("arm_identity_invalid")
    metric_source = source.get("metrics")
    if not isinstance(metric_source, dict):
        raise DashboardError("arm_metrics_invalid")
    metrics = {label: _metric(metric_source.get(label)) for label in PAIR_METRICS}
    token_hashes = _hash_list(source.get("token_sha256", []))
    text_hashes = _hash_list(source.get("text_sha256", []))
    token_identity = bool(token_hashes) and len(set(token_hashes)) == 1
    text_identity = bool(text_hashes) and len(set(text_hashes)) == 1
    return {
        "arm": arm,
        "runs": source["runs"],
        "metrics": metrics,
        "token_identity": token_identity,
        "token_identity_count": len(token_hashes) if token_identity else 0,
        "token_identity_total": len(token_hashes),
        "text_identity": text_identity,
        "text_identity_count": len(text_hashes) if text_identity else 0,
        "text_identity_total": len(text_hashes),
        # SHA-256 values are identifiers, not model output.  They let the UI
        # distinguish stable token streams without exposing the streams.
        "token_sha256": token_hashes[0] if token_identity else None,
        "peak_rss_bytes": _positive_int(source.get("peak_rss_bytes")),
        "peak_mlx_bytes": _positive_int(source.get("peak_mlx_bytes")),
        "swap_deltas_bytes": [
            _finite(item, allow_none=False) for item in source.get("swap_deltas_bytes", [])
        ] if isinstance(source.get("swap_deltas_bytes", []), list) else [],
    }


def _ci(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DashboardError("confidence_interval_invalid")
    return {
        "lower": _finite(value.get("lower")),
        "upper": _finite(value.get("upper")),
        "method": "paired bootstrap percentile" if isinstance(value.get("method"), str) else None,
        "resamples": value.get("resamples") if type(value.get("resamples")) is int else None,
        "seed": value.get("seed") if type(value.get("seed")) is int else None,
    }


def _paired_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DashboardError("paired_metrics_invalid")
    ratios = value.get("ratios")
    if not isinstance(ratios, dict):
        raise DashboardError("paired_metrics_invalid")
    projected: dict[str, Any] = {}
    for metric in PAIR_METRICS:
        item = ratios.get(metric)
        if not isinstance(item, dict):
            raise DashboardError("paired_metric_invalid")
        comparisons: dict[str, Any] = {}
        for name in ("fixed_compiled_div_standard_eager", "fixed_compiled_div_fixed_eager"):
            comparison = item.get(name)
            if not isinstance(comparison, dict):
                raise DashboardError("paired_metric_invalid")
            comparisons[name] = {
                "median": _finite(comparison.get("median")),
                "bootstrap_95_ci": _ci(comparison.get("bootstrap_95_ci")),
            }
        projected[metric] = comparisons
    return {"complete": value.get("complete") is True, "ratios": projected}


def _derived_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DashboardError("derived_metrics_invalid")
    breaks = value.get("break_even_decode_forwards", [])
    if not isinstance(breaks, list) or len(breaks) > 6:
        raise DashboardError("break_even_invalid")
    return {
        "complete": value.get("complete") is True,
        "calculated_only": value.get("calculated_only") is True,
        "warmed_decode_ratio_median": _finite(value.get("warmed_decode_ratio_median")),
        "cold_decode_ratio_median": _finite(value.get("cold_decode_ratio_median")),
        "break_even_decode_forwards": [_finite(item) for item in breaks],
    }


def _gates(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        raise DashboardError("gates_invalid")
    return {key: value.get(key) is True for key in (
        "all_blocks_completed", "resource_pass", "budget_pass",
        "block_correctness_pass", "determinism_pass", "candidate_runnable",
        "pairing_pass", "snapshot_content_pass",
    )}


def _runs_projection(runs: Any) -> list[dict[str, Any]]:
    if not isinstance(runs, list) or len(runs) > 6:
        raise DashboardError("runs_invalid")
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(runs, start=1):
        if not isinstance(value, dict):
            raise DashboardError("run_invalid")
        status = value.get("status", "complete")
        if status not in SAFE_FINISH:
            raise DashboardError("run_status_invalid")
        arms = value.get("arms")
        if not isinstance(arms, dict):
            raise DashboardError("run_arms_invalid")
        arm_names = [name for name in ARM_NAMES if name in arms]
        if any(name not in ARM_NAMES for name in arms):
            raise DashboardError("run_arms_invalid")
        rows.append({
            "block": value.get("block") if type(value.get("block")) is int else index,
            "status": status,
            "arm_order": arm_names,
            "token_identity": value.get("correctness", {}).get("all_arms_token_equal") is True
            if isinstance(value.get("correctness"), dict) else False,
            "text_identity": value.get("correctness", {}).get("all_arms_text_equal") is True
            if isinstance(value.get("correctness"), dict) else False,
        })
    return rows


def _worker_events_projection(events: Any) -> list[dict[str, Any]]:
    if not isinstance(events, list) or len(events) > 6:
        raise DashboardError("worker_events_invalid")
    projected: list[dict[str, Any]] = []
    for value in events:
        if not isinstance(value, dict) or set(value) != {"event", "error_type", "message", "model_key", "block"}:
            raise DashboardError("worker_event_invalid")
        if (value["event"] != "error" or not isinstance(value["error_type"], str) or
                not isinstance(value["message"], str) or len(value["message"]) > 500 or
                not isinstance(value["model_key"], str) or type(value["block"]) is not int or value["block"] < 1 or value["block"] > 6):
            raise DashboardError("worker_event_invalid")
        projected.append({key: value[key] for key in ("event", "error_type", "message", "model_key", "block")})
    return projected


def _project_result(value: dict[str, Any]) -> dict[str, Any]:
    metrics = value["metrics"]
    arms_source = metrics.get("arms")
    if not isinstance(arms_source, dict):
        raise DashboardError("arm_metrics_invalid")
    arms = {arm: _arm_projection(arms_source.get(arm), arm) for arm in ARM_NAMES}
    return {
        "status": "available",
        "study_id": EXPECTED_STUDY_ID,
        "run_id": EXPECTED_RUN_ID,
        "decision": value["decision"],
        "formal_claim": False,
        "read_only": True,
        "runs_completed": metrics.get("runs_completed") if type(metrics.get("runs_completed")) is int else len(value["runs"]),
        "partial_result": value.get("partial_result") is True,
        "worker_events": _worker_events_projection(value.get("worker_events", [])),
        "gates": _gates(value.get("gates")),
        "arms": arms,
        "paired": _paired_projection(metrics.get("paired")),
        "derived": _derived_projection(metrics.get("derived")),
        "runs": _runs_projection(value["runs"]),
    }


def _history_projection(path: Path = HISTORY_PATH) -> dict[str, Any] | None:
    """Project the previous dual-model study without importing its UI."""
    try:
        value = _parse_json(path, MAX_HISTORY_BYTES)
        if (
            value.get("study_id") != HISTORY_STUDY_ID
            or value.get("run_id") != HISTORY_RUN_ID
            or value.get("formal_claim") is not False
            or value.get("decision") not in {
                "no_planner_qualified", "planner_1b_qualified_exact_case",
                "planner_4b_qualified_exact_case", "resource_or_budget_failed",
            }
        ):
            return None
        metrics = value.get("metrics")
        if not isinstance(metrics, dict):
            return None
        models = metrics.get("model_1b"), metrics.get("model_4b")
        if any(not isinstance(model, dict) for model in models):
            return None
        projected_models: dict[str, Any] = {}
        for key, model in zip(("1b", "4b"), models):
            metric_source = model.get("metrics")
            if not isinstance(metric_source, dict):
                return None
            projected_models[key] = {
                "contract_successes": model.get("contract_successes") if type(model.get("contract_successes")) is int else 0,
                "runs_completed": model.get("runs_completed") if type(model.get("runs_completed")) is int else 0,
                "ttft_median": _finite(metric_source.get("ttft_seconds", {}).get("median")) if isinstance(metric_source.get("ttft_seconds"), dict) else None,
                "model_work_median": _finite(metric_source.get("model_work_seconds", {}).get("median")) if isinstance(metric_source.get("model_work_seconds"), dict) else None,
                "process_wall_median": _finite(metric_source.get("process_wall_seconds", {}).get("median")) if isinstance(metric_source.get("process_wall_seconds"), dict) else None,
                "peak_rss_bytes": _positive_int(model.get("peak_rss_bytes")),
            }
        return {
            "cycle": 15,
            "study_id": HISTORY_STUDY_ID,
            "decision": value["decision"],
            "formal_claim": False,
            "models": projected_models,
        }
    except (DashboardError, OSError, TypeError, ValueError):
        return None


def snapshot(path: Path = RESULT_PATH, history_path: Path = HISTORY_PATH) -> dict[str, Any]:
    """Return a safe JSON projection; this function has no write side effects."""
    try:
        result = _project_result(_load_result(path))
    except (DashboardError, OSError):
        result = {
            "status": "not_available",
            "study_id": EXPECTED_STUDY_ID,
            "run_id": EXPECTED_RUN_ID,
            "decision": "not_available",
            "formal_claim": False,
            "read_only": True,
            "message_code": "result_unavailable",
            "history": _history_projection(history_path),
        }
        return result
    result["history"] = _history_projection(history_path)
    return result


HTML = """<!doctype html><html lang='de'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width'><title>Friday Matmul-Compile-Vergleich</title>
<style>body{font:15px system-ui;max-width:1200px;margin:30px auto;padding:0 16px;background:#10151b;color:#e9eef5}.card{background:#17202a;border:1px solid #314052;border-radius:10px;padding:16px;margin:14px 0}table{border-collapse:collapse;width:100%}th,td{padding:7px;border-bottom:1px solid #314052;text-align:left}.muted{color:#9fb0c3}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>Matmul-/Compile-Vergleich</h1><p class='muted'>Nur lesbare lokale Historie · Matmul bleibt in allen drei Armen aktiv</p>
<div class='card' id='summary'>Wird geladen …</div><div class='grid' id='arms'></div>
<div class='card' id='paired'></div><div class='card' id='derived'></div>
<div class='card'><h2>Läufe</h2><table><thead><tr><th>Block</th><th>Reihenfolge</th><th>Tokenidentität</th><th>Textidentität</th><th>Status</th></tr></thead><tbody id='runs'></tbody></table></div>
<div class='card' id='history'></div>
<script>
const txt=(node,value)=>{node.textContent=String(value??'n/a')};
const sec=v=>typeof v==='number'?v.toFixed(6)+' s':'n/a';
const bytes=v=>typeof v==='number'?(v/1073741824).toFixed(2)+' GB':'n/a';
const summary=document.getElementById('summary'),arms=document.getElementById('arms'),paired=document.getElementById('paired'),derived=document.getElementById('derived'),runs=document.getElementById('runs'),history=document.getElementById('history');
const add=(parent,tag,value)=>{const n=document.createElement(tag);txt(n,value);parent.appendChild(n);return n};
const armCard=(name,data)=>{const card=document.createElement('div');card.className='card';add(card,'h2',name);add(card,'p','Tokenidentität: '+(data.token_identity?data.token_identity_count+'/'+data.token_identity_total:'nein')+' · Textidentität: '+(data.text_identity?data.text_identity_count+'/'+data.text_identity_total:'nein'));add(card,'p','Speicher RSS: '+bytes(data.peak_rss_bytes)+' · MLX: '+bytes(data.peak_mlx_bytes));for(const key of ['decode_total','intertoken_p50','intertoken_p95','intertoken_p99']){const m=data.metrics?.[key]||{};add(card,'p',key+': Median '+sec(m.median)+' · MAD '+sec(m.mad)+' · P50 '+sec(m.p50)+' · P95 '+sec(m.p95)+' · P99 '+sec(m.p99))}arms.appendChild(card)};
fetch('/api/snapshot').then(r=>{if(!r.ok)throw Error('http');return r.json()}).then(x=>{txt(summary,x.status==='available'?'Entscheidung: '+x.decision+' · Läufe: '+(x.runs_completed??0)+' · formal_claim=false':'Noch keine gültige Zyklus-16-Messung · Ergebnis nicht verfügbar');if(x.status!=='available'){txt(paired,'Keine Zyklus-16-Ratios verfügbar.');txt(derived,'Keine Zyklus-16-Ableitungen verfügbar.')}else{for(const name of ['standard_eager','fixed_eager','fixed_compiled'])armCard(name,x.arms?.[name]||{});const p=x.paired?.ratios||{};txt(paired,'Gepaarte Ratios fixed_compiled/reference und fixed_compiled/fixed_eager: '+JSON.stringify(p));const d=x.derived||{};txt(derived,'Berechnet (nicht separat gemessen): warm '+(d.warmed_decode_ratio_median??'n/a')+' · kalt '+(d.cold_decode_ratio_median??'n/a')+' · Break-even '+(d.break_even_decode_forwards??'n/a'));for(const row of x.runs||[]){const tr=document.createElement('tr');for(const v of [row.block,(row.arm_order||[]).join(' → '),row.token_identity,row.text_identity,row.status])add(tr,'td',v);runs.appendChild(tr)}}const h=x.history;if(h){txt(history,'Zyklus 15 · '+h.decision+' · 1B Prozessmedian '+sec(h.models?.['1b']?.process_wall_median)+' · 4B Prozessmedian '+sec(h.models?.['4b']?.process_wall_median))}else txt(history,'Keine sichere historische Zyklus-15-Projektion verfügbar.');}).catch(()=>txt(summary,'Daten nicht verfügbar.'));
</script></html>""".encode("utf-8")


def _host_header_allowed(value: str | None, bound_port: int) -> bool:
    return value in {"127.0.0.1", f"127.0.0.1:{bound_port}"}


def _handler(result_path: Path, history_path: Path = HISTORY_PATH) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _headers(self, status: int, content_type: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()

        def _write(self, status: int, content_type: str, payload: bytes) -> None:
            self._headers(status, content_type, len(payload))
            if self.command != "HEAD":
                self.wfile.write(payload)

        def _host_ok(self) -> bool:
            return _host_header_allowed(self.headers.get("Host"), int(self.server.server_address[1]))

        def _not_allowed(self) -> None:
            if not self._host_ok():
                self._write(421, "application/json; charset=utf-8", b'{"error":"host not allowed"}')
                return
            self.send_response(405)
            self.send_header("Allow", "GET, HEAD")
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def _read(self) -> None:
            if not self._host_ok():
                self._write(421, "application/json; charset=utf-8", b'{"error":"host not allowed"}')
                return
            route = urlsplit(self.path).path
            if route == "/":
                self._write(200, "text/html; charset=utf-8", HTML)
            elif route == "/api/snapshot":
                payload = json.dumps(snapshot(result_path, history_path), allow_nan=False, sort_keys=True).encode("utf-8")
                self._write(200, "application/json; charset=utf-8", payload)
            else:
                self._write(404, "application/json; charset=utf-8", b'{"error":"not found"}')

        def do_GET(self) -> None:  # noqa: N802
            self._read()

        def do_HEAD(self) -> None:  # noqa: N802
            self._read()

        def do_POST(self) -> None:  # noqa: N802
            self._not_allowed()

        def do_PUT(self) -> None:  # noqa: N802
            self._not_allowed()

        def do_PATCH(self) -> None:  # noqa: N802
            self._not_allowed()

        def do_DELETE(self) -> None:  # noqa: N802
            self._not_allowed()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._not_allowed()

        def do_TRACE(self) -> None:  # noqa: N802
            self._not_allowed()

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def _self_check() -> int:
    """Offline projection and protocol sanity checks; no file is written."""
    assert _host_header_allowed("127.0.0.1", 8786)
    assert _host_header_allowed("127.0.0.1:8786", 8786)
    assert not _host_header_allowed("localhost:8786", 8786)
    missing = snapshot(Path("/definitely/missing/cycle16-results.json"), Path("/definitely/missing/history.json"))
    assert missing["status"] == "not_available" and missing["formal_claim"] is False
    assert b"innerHTML" not in HTML and b"prompt" not in HTML.lower()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="matmul_compile_ab_dashboard", allow_abbrev=False)
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8786)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    parser.add_argument("--history", type=Path, default=HISTORY_PATH)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    server = ThreadingHTTPServer((args.host, args.port), _handler(args.result, args.history))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
