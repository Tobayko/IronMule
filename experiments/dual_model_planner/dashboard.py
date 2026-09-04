#!/usr/bin/env python3
"""Read-only loopback history UI for the dual-model planner study."""

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
EXPECTED_STUDY_ID = "dual-model-evidence-planner-20260824-01"
EXPECTED_RUN_ID = "dual-model-evidence-planner-validation-20260824-01"
MAX_RESULT_BYTES = 8_000_000
ALLOWED_DECISIONS = {
    "planner_1b_qualified_exact_case",
    "planner_4b_qualified_exact_case",
    "both_qualified_1b_preferred",
    "both_qualified_no_automatic_preference",
    "no_planner_qualified",
    "resource_or_budget_failed",
    "correctness_failed",
}
ALLOWED_MODELS = {"1b", "4b"}
ALLOWED_CANDIDATES = {
    "persistent_service_qualification",
    "batched_readback",
    "host_readback_upper_bound",
    "kv_cache_preallocation_ab",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DashboardError(RuntimeError):
    """Stored evidence is unavailable or has an unexpected identity."""


def _load_result(path: Path = RESULT_PATH) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise DashboardError("result is unavailable")
    payload = path.read_bytes()
    if not payload or len(payload) > MAX_RESULT_BYTES:
        raise DashboardError("result size is invalid")
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate result field")
            value[key] = item
        return value
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise DashboardError("result is invalid JSON") from exc
    if (
        not isinstance(value, dict)
        or value.get("study_id") != EXPECTED_STUDY_ID
        or type(value.get("study_id")) is not str
        or value.get("run_id") != EXPECTED_RUN_ID
        or type(value.get("run_id")) is not str
        or value.get("formal_claim") is not False
        or type(value.get("decision")) is not str
        or value.get("decision") not in ALLOWED_DECISIONS
        or not isinstance(value.get("runs"), list)
        or len(value["runs"]) > 12
    ):
        raise DashboardError("result identity is invalid")
    return value


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise DashboardError("numeric history field is invalid")
    return float(value)


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise DashboardError("integer history field is invalid")
    return value


def _sha256_value(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise DashboardError("hash history field is invalid")
    return value


def _candidate_value(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in ALLOWED_CANDIDATES:
        raise DashboardError("candidate history field is invalid")
    return value


def _safe_run_row(index: int, run: Any) -> dict[str, Any]:
    if not isinstance(run, dict):
        raise DashboardError("run history is invalid")
    parser = run.get("parser")
    if not isinstance(parser, dict):
        raise DashboardError("parser history is invalid")
    model_key = run.get("model_key")
    if model_key not in ALLOWED_MODELS:
        raise DashboardError("model history field is invalid")
    pair_id = run.get("pair_id")
    if pair_id is not None and (type(pair_id) is not int or not 1 <= pair_id <= 6):
        raise DashboardError("pair history field is invalid")
    finish_reason = run.get("finish_reason")
    if finish_reason is not None and finish_reason not in {"stop", "length"}:
        raise DashboardError("finish history field is invalid")
    return {
        "candidate_id": _candidate_value(run.get("candidate_id")),
        "contract_ok": parser.get("contract_ok") is True,
        "exact_text_sha256": _sha256_value(run.get("text_utf8_sha256")),
        "finish_reason": finish_reason,
        "model_key": model_key,
        "model_work_seconds": (
            run.get("model_work_ns") / 1_000_000_000
            if _positive_int(run.get("model_work_ns")) is not None
            else None
        ),
        "pair_id": pair_id,
        "process_wall_seconds": (
            run.get("process_wall_ns") / 1_000_000_000
            if _positive_int(run.get("process_wall_ns")) is not None
            else None
        ),
        "run": index + 1,
        "token_sha256": _sha256_value(run.get("token_sha256")),
        "ttft_seconds": (
            run.get("ttft_ns") / 1_000_000_000
            if _positive_int(run.get("ttft_ns")) is not None
            else None
        ),
    }


def _safe_metric_stats(value: Any) -> dict[str, float | None]:
    if not isinstance(value, dict):
        return {"median": None, "mad": None}
    return {
        "median": _finite_number(value.get("median")),
        "mad": _finite_number(value.get("mad")),
    }


def _safe_model_metrics(value: Any, model_key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    candidate_ids = value.get("candidate_ids", [])
    if not isinstance(candidate_ids, list):
        raise DashboardError("candidate history is invalid")
    metric_source = value.get("metrics")
    if not isinstance(metric_source, dict):
        metric_source = {}
    return {
        "candidate_ids": [_candidate_value(item) for item in candidate_ids],
        "contract_pass": value.get("contract_pass") is True,
        "contract_successes": value.get("contract_successes", 0)
        if type(value.get("contract_successes", 0)) is int
        else 0,
        "correctness_pass": value.get("correctness_pass") is True,
        "deterministic": value.get("deterministic") is True,
        "identity_pass": value.get("identity_pass") is True,
        "metrics": {
            label: _safe_metric_stats(metric_source.get(label))
            for label in (
                "ttft_seconds",
                "model_work_seconds",
                "process_wall_seconds",
            )
        },
        "model_key": model_key,
        "peak_mlx_bytes": _positive_int(value.get("peak_mlx_bytes")),
        "peak_rss_bytes": _positive_int(value.get("peak_rss_bytes")),
        "priority_pass": value.get("priority_pass") is True,
        "priority_successes": value.get("priority_successes", 0)
        if type(value.get("priority_successes", 0)) is int
        else 0,
        "runs_completed": value.get("runs_completed", 0)
        if type(value.get("runs_completed", 0)) is int
        else 0,
        "functional_pass": value.get("functional_pass") is True,
    }


def _safe_bootstrap(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise DashboardError("bootstrap history is invalid")
    return {
        "lower": _finite_number(value.get("lower")),
        "upper": _finite_number(value.get("upper")),
        "method": value.get("method") if isinstance(value.get("method"), str) else None,
        "resamples": value.get("resamples") if type(value.get("resamples")) is int else None,
        "seed": value.get("seed") if type(value.get("seed")) is int else None,
    }


def _safe_pairwise(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"complete": False, "error": "pairwise metrics unavailable", "ratios_1b_div_4b": {}}
    source = value.get("ratios_1b_div_4b")
    if not isinstance(source, dict):
        source = {}
    ratios: dict[str, Any] = {}
    for label in ("ttft", "model_work", "process_wall", "token_rate"):
        item = source.get(label)
        if not isinstance(item, dict):
            continue
        ratios[label] = {
            "median": _finite_number(item.get("median")),
            "bootstrap_95_ci": _safe_bootstrap(item.get("bootstrap_95_ci")),
        }
    return {
        "complete": value.get("complete") is True,
        "error": value.get("error") if isinstance(value.get("error"), str) else None,
        "ratios_1b_div_4b": ratios,
    }


def _safe_cross_model_text(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "complete": False,
            "exact_text_equal_count": 0,
            "exact_text_equal_total": "0/6",
            "informational_only": True,
            "pairs": [],
        }
    rows = value.get("pairs", [])
    if not isinstance(rows, list):
        raise DashboardError("cross-model history is invalid")
    projected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise DashboardError("cross-model pair history is invalid")
        pair_id = row.get("pair_id")
        if type(pair_id) is not int or not 1 <= pair_id <= 6:
            raise DashboardError("cross-model pair ID is invalid")
        projected.append(
            {
                "pair_id": pair_id,
                "exact_text_equal": row.get("exact_text_equal") is True,
                "1b_text_utf8_sha256": _sha256_value(row.get("1b_text_utf8_sha256")),
                "4b_text_utf8_sha256": _sha256_value(row.get("4b_text_utf8_sha256")),
                "1b_token_sha256": _sha256_value(row.get("1b_token_sha256")),
                "4b_token_sha256": _sha256_value(row.get("4b_token_sha256")),
            }
        )
    return {
        "complete": value.get("complete") is True,
        "exact_text_equal_count": value.get("exact_text_equal_count", 0)
        if type(value.get("exact_text_equal_count", 0)) is int
        else 0,
        "exact_text_equal_total": value.get("exact_text_equal_total")
        if isinstance(value.get("exact_text_equal_total"), str)
        else "0/6",
        "informational_only": value.get("informational_only") is True,
        "pairs": projected,
    }


def _safe_gates(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        value = {}
    return {
        key: value.get(key) is True
        for key in (
            "all_runs_completed",
            "resource_pass",
            "budget_pass",
            "correctness_failure",
            "fresh_process_pass",
            "prompt_identity_pass",
            "pairing_pass",
            "cross_model_text_complete",
            "model_1b",
            "model_4b",
        )
    }


def snapshot(path: Path = RESULT_PATH) -> dict[str, Any]:
    """Return only a structured projection; raw text and token arrays stay hidden."""

    result = _load_result(path)
    rows = [_safe_run_row(index, run) for index, run in enumerate(result["runs"])]
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "cross_model_text": _safe_cross_model_text(metrics.get("cross_model_text")),
        "decision": result["decision"],
        "formal_claim": False,
        "gates": _safe_gates(result.get("gates")),
        "metrics": {
            "model_1b": _safe_model_metrics(metrics.get("model_1b"), "1b"),
            "model_4b": _safe_model_metrics(metrics.get("model_4b"), "4b"),
            "pairwise": _safe_pairwise(metrics.get("pairwise")),
            "runs_completed": metrics.get("runs_completed")
            if type(metrics.get("runs_completed")) is int
            else len(rows),
        },
        "read_only": True,
        "rows": rows,
        "run_id": result["run_id"],
        "study_id": result["study_id"],
    }


HTML = """<!doctype html><html lang='de'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width'><title>Friday Plannervergleich</title>
<style>body{font:15px system-ui;max-width:1180px;margin:30px auto;padding:0 16px;background:#10151b;color:#e9eef5}
.card{background:#17202a;border:1px solid #314052;border-radius:10px;padding:16px;margin:14px 0}
table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #314052;text-align:left}
.muted{color:#9fb0c3}code{color:#8bd5ff}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}</style>
<h1>Gemma-Plannervergleich</h1><p class='muted'>Nur lesbare lokale Historie — 1B und 4B</p>
<div class='card' id='summary'>Wird geladen...</div><div class='grid'>
<div class='card' id='model-1b'></div><div class='card' id='model-4b'></div></div>
<div class='card' id='pairs'></div><div class='card'><table><thead><tr>
<th>Lauf</th><th>Paar</th><th>Modell</th><th>Vertrag</th><th>Auswahl</th><th>TTFT</th><th>Modellzeit</th><th>Gesamtzeit</th><th>Tokenhash</th>
</tr></thead><tbody id='rows'></tbody></table></div>
<script>
const text=(node,value)=>{node.textContent=String(value ?? 'n/a')};
const seconds=(value)=>typeof value==='number'?value.toFixed(6)+' s':'n/a';
const summary=document.getElementById('summary'),pairs=document.getElementById('pairs');
const rows=document.getElementById('rows');
const modelCard=(id,label,data)=>{const node=document.getElementById(id);const m=data||{};const stats=m.metrics||{};
node.textContent='';const h=document.createElement('h2');text(h,label);node.appendChild(h);
const p=document.createElement('p');text(p,'Vertrag '+(m.contract_successes??0)+'/6 | richtige Auswahl '+(m.priority_successes??0)+'/6');node.appendChild(p);
for(const key of ['ttft_seconds','model_work_seconds','process_wall_seconds']){const q=document.createElement('p');const x=stats[key]||{};text(q,key+': Median '+seconds(x.median)+' | MAD '+seconds(x.mad));node.appendChild(q)}
};
fetch('/api/snapshot').then(r=>{if(!r.ok)throw new Error('HTTP '+r.status);return r.json()}).then(x=>{
text(summary,'Entscheidung: '+x.decision+' | Läufe: '+(x.metrics?.runs_completed??0)+' | formal_claim=false');
modelCard('model-1b','Gemma 1B',x.metrics?.model_1b);modelCard('model-4b','Gemma 4B',x.metrics?.model_4b);
const c=x.cross_model_text||{},r=x.metrics?.pairwise?.ratios_1b_div_4b||{};
text(pairs,'Cross-Model: exakt gleiche Antwortbytes '+(c.exact_text_equal_total??'n/a')+' (nur Information) | TTFT 1B/4B Median '+(r.ttft?.median??'n/a')+' | Modellzeit '+(r.model_work?.median??'n/a')+' | Gesamtzeit '+(r.process_wall?.median??'n/a'));
for(const item of Array.isArray(x.rows)?x.rows:[]){const tr=document.createElement('tr');for(const value of [item.run,item.pair_id,item.model_key,item.contract_ok,item.candidate_id,seconds(item.ttft_seconds),seconds(item.model_work_seconds),seconds(item.process_wall_seconds),item.token_sha256]){const td=document.createElement('td');text(td,value);tr.appendChild(td)}rows.appendChild(tr)}
}).catch(error=>text(summary,'Nicht verfügbar: '+error));
</script></html>""".encode("utf-8")


def _host_header_allowed(value: str | None, bound_host: str, bound_port: int) -> bool:
    if bound_host == "127.0.0.1":
        return value in {"127.0.0.1", f"127.0.0.1:{bound_port}"}
    if bound_host == "::1":
        return value in {"[::1]", f"[::1]:{bound_port}"}
    return False


def _handler(result_path: Path) -> type[BaseHTTPRequestHandler]:
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

        def _host_ok(self) -> bool:
            address = self.server.server_address
            return _host_header_allowed(
                self.headers.get("Host"), str(address[0]), int(address[1])
            )

        def _write(self, status: int, content_type: str, payload: bytes) -> None:
            self._headers(status, content_type, len(payload))
            if self.command != "HEAD":
                self.wfile.write(payload)

        def _method_not_allowed(self) -> None:
            if not self._host_ok():
                self._write(421, "application/json; charset=utf-8", b'{"error":"host not allowed"}')
                return
            self.send_response(405)
            self.send_header("Allow", "GET, HEAD")
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def _get_or_head(self) -> None:
            if not self._host_ok():
                self._write(421, "application/json; charset=utf-8", b'{"error":"host not allowed"}')
                return
            path = urlsplit(self.path).path
            if path == "/":
                self._write(200, "text/html; charset=utf-8", HTML)
                return
            if path == "/api/snapshot":
                try:
                    payload = json.dumps(
                        snapshot(result_path), allow_nan=False, sort_keys=True
                    ).encode("utf-8")
                    status = 200
                except DashboardError as exc:
                    payload = json.dumps({"error": str(exc)}, sort_keys=True).encode("utf-8")
                    status = 503
                self._write(status, "application/json; charset=utf-8", payload)
                return
            self._write(404, "application/json; charset=utf-8", b'{"error":"not found"}')

        def do_GET(self) -> None:  # noqa: N802
            self._get_or_head()

        def do_HEAD(self) -> None:  # noqa: N802
            self._get_or_head()

        def do_DELETE(self) -> None:  # noqa: N802
            self._method_not_allowed()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._method_not_allowed()

        def do_PATCH(self) -> None:  # noqa: N802
            self._method_not_allowed()

        def do_POST(self) -> None:  # noqa: N802
            self._method_not_allowed()

        def do_PUT(self) -> None:  # noqa: N802
            self._method_not_allowed()

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dual_model_planner_dashboard", allow_abbrev=False)
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "::1"))
    parser.add_argument("--port", type=int, default=8784)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    server = ThreadingHTTPServer((args.host, args.port), _handler(args.result))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
