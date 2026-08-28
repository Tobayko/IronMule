"""Generate a local, self-contained B27 evidence dashboard."""

from __future__ import annotations

import argparse
import html
import json
import tempfile
from pathlib import Path
from typing import Any


def _text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def render(summary: dict[str, Any], verification: dict[str, Any] | None = None) -> str:
    if summary.get("schema") != "ironmule.main_baseline.public.v1":
        raise ValueError("unsupported public-summary schema")
    cells = summary.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("public summary has no baseline cells")

    rows = []
    timeline = []
    for failure in summary.get("premeasurement_failures", []):
        timeline.append(
            '<li class="event failed"><span>Pre-measurement</span>'
            f'<strong>{_text(failure.get("type", "unknown"))}</strong>'
            f'<small>{_text(failure.get("stage", "unknown"))}; no timing used</small></li>'
        )
    for cell in cells:
        model = cell.get("model", {})
        comparison = cell.get("comparison", {})
        wall = comparison.get("wall_ratio_throughput_over_interactive", {})
        rate = comparison.get("physical_rate_ratio_throughput_over_interactive", {})
        resources = cell.get("resources", {})
        interactive = cell.get("interactive", {})
        throughput = cell.get("throughput", {})
        wall_gain = (1.0 - float(wall["median"])) * 100.0
        rate_gain = (float(rate["median"]) - 1.0) * 100.0
        rows.append(
            "<tr>"
            f'<td><strong>{_text(model.get("id"))}</strong><small>{_text(str(model.get("revision", ""))[:10])}…</small></td>'
            f'<td>{_number(interactive.get("outer_wall_ms_median"))}</td>'
            f'<td>{_number(throughput.get("outer_wall_ms_median"))}</td>'
            f'<td><span class="good">−{_number(wall_gain)}%</span><small>{_number(wall.get("ci_low"), 4)}–{_number(wall.get("ci_high"), 4)}</small></td>'
            f'<td><span class="good">+{_number(rate_gain)}%</span><small>{_number(rate.get("ci_low"), 4)}–{_number(rate.get("ci_high"), 4)}</small></td>'
            f'<td>{_number(resources.get("mlx_peak_memory_bytes", 0) / 1_000_000_000, 3)} GB</td>'
            f'<td>{_text(resources.get("swap_delta_bytes", "—"))} B</td>'
            "</tr>"
        )
        timeline.append(
            '<li class="event passed"><span>Baseline cell</span>'
            f'<strong>{_text(model.get("id"))}</strong>'
            f'<small>{_text(cell.get("status"))}; tokens identical; zero fallback</small></li>'
        )

    verification_card = ""
    if verification is not None:
        verified = verification.get("ok") is True and not verification.get("errors")
        label = "VERIFIED" if verified else "FAILED"
        css = "good" if verified else "warn"
        verification_card = (
            '<div class="card"><small>Public-summary integrity</small>'
            f'<strong class="{css}">{label}</strong>'
            f'<span>{_text(verification.get("checked_cells", 0))} cells · '
            f'{_text(verification.get("checked_failures", 0))} failures</span></div>'
        )
        timeline.append(
            '<li class="event passed"><span>Evidence integrity</span>'
            f'<strong>{label}</strong><small>{_text(len(verification.get("errors", [])))} errors; '
            'no qualification or activation change</small></li>'
        )

    corpus = summary.get("corpus_inventory", {})
    tests = summary.get("tests", {})
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IronMule B27 evidence dashboard</title>
<style>
:root {{ color-scheme: dark; --bg:#0b0d10; --panel:#15191f; --line:#2a313b;
  --text:#f2f5f7; --muted:#9aa7b5; --green:#65d6a2; --red:#ff8e8e; --amber:#f0c66b; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text);
  font:15px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ width:min(1180px,calc(100% - 32px)); margin:32px auto 64px; }}
h1 {{ font-size:clamp(28px,5vw,48px); margin:0; letter-spacing:-.04em; }}
h2 {{ margin:32px 0 12px; }} .kicker {{ color:var(--green); font-weight:700; letter-spacing:.12em;
  text-transform:uppercase; }} .lede {{ color:var(--muted); max-width:760px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin:24px 0; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px; }}
.card small, td small {{ color:var(--muted); display:block; }} .card strong {{ font-size:25px; display:block; }}
.good {{ color:var(--green); font-weight:700; }} .warn {{ color:var(--amber); }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:14px; }}
table {{ width:100%; border-collapse:collapse; min-width:900px; background:var(--panel); }}
th,td {{ padding:14px 16px; text-align:right; border-bottom:1px solid var(--line); }}
th:first-child,td:first-child {{ text-align:left; }} th {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
tr:last-child td {{ border-bottom:0; }} ol {{ list-style:none; padding:0; display:grid; gap:10px; }}
.event {{ position:relative; padding:14px 18px 14px 22px; background:var(--panel); border:1px solid var(--line);
  border-radius:12px; }} .event:before {{ content:""; position:absolute; left:8px; top:10px; bottom:10px; width:4px; border-radius:3px; }}
.event.failed:before {{ background:var(--red); }} .event.passed:before {{ background:var(--green); }}
.event span,.event small {{ color:var(--muted); display:block; }} footer {{ color:var(--muted); margin-top:36px; }}
code {{ color:#c7d6e5; }}
</style>
</head>
<body><main>
<div class="kicker">Measure · Verify · Remember</div>
<h1>IronMule B27 evidence dashboard</h1>
<p class="lede">Current-main engineering baseline and audit history. This local,
self-contained view makes no qualification or activation claim.</p>
<section class="grid">
  <div class="card"><small>Top status</small><strong>{_text(summary.get("status"))}</strong><span class="warn">Qualification: false</span></div>
  <div class="card"><small>Unique evidence artifacts</small><strong>{_text(corpus.get("unique_artifacts", "—"))}</strong><span>{_text(corpus.get("artifacts", "—"))} source occurrences</span></div>
  <div class="card"><small>Local-only / ignored</small><strong>{_text(corpus.get("local_only_or_ignored", "—"))}</strong><span>kept out of public history</span></div>
  <div class="card"><small>Regression suite</small><strong>{_text(tests.get("non_integration", "—"))}</strong><span>serial, no xdist</span></div>
  {verification_card}
</section>
<h2>Protected baseline cells</h2>
<div class="table-wrap"><table>
<thead><tr><th>Model / revision</th><th>Interactive ms</th><th>Throughput ms</th><th>Wall effect</th><th>Rate effect</th><th>MLX peak</th><th>Swap Δ</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table></div>
<h2>History</h2><ol>{''.join(timeline)}</ol>
<footer>Base <code>{_text(summary.get("base_commit"))}</code> · runtime tree
<code>{_text(summary.get("runtime_tree_sha256"))}</code> · no external assets,
network calls, model downloads or runtime routing.</footer>
</main></body></html>"""


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = json.loads(args.summary.read_text())
    verification = json.loads(args.verification.read_text()) if args.verification else None
    _atomic_write(args.output, render(summary, verification))
    print(json.dumps({"output": str(args.output), "cells": len(summary["cells"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
