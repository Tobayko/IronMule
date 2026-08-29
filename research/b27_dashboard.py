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


def render(
    summary: dict[str, Any], verification: dict[str, Any] | None = None,
    post_change: dict[str, Any] | None = None,
    post_verification: dict[str, Any] | None = None,
    cross_commit: dict[str, Any] | None = None,
    cross_verification: dict[str, Any] | None = None,
    d2_pre: dict[str, Any] | None = None,
    d2_pre_verification: dict[str, Any] | None = None,
) -> str:
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

    post_card = ""
    post_section = ""
    if post_change is not None:
        classification = _text(post_change.get("classification", "unknown"))
        clean = post_change.get("classification") == "NO_REGRESSION_OBSERVED"
        css = "good" if clean else "warn"
        post_card = (
            '<div class="card"><small>D1 post-change screen</small>'
            f'<strong class="{css}">{classification}</strong>'
            f'<span>{_text(post_change.get("regression_kind", "unknown"))}</span></div>'
        )
        post_rows = []
        for cell in post_change.get("cells", []):
            comparisons = cell.get("comparisons", {})
            interactive = comparisons.get("interactive", {})
            throughput = comparisons.get("throughput", {})
            iw = interactive.get("outer_wall_post_over_pre", {})
            ir = interactive.get("physical_rate_post_over_pre", {})
            tw = throughput.get("outer_wall_post_over_pre", {})
            tr = throughput.get("physical_rate_post_over_pre", {})
            misses = cell.get("performance_misses", [])
            post_rows.append(
                "<tr>"
                f'<td><strong>{_text(cell.get("model_id"))}</strong></td>'
                f'<td>{_number(iw.get("median_ratio"), 4)}<small>{_number(iw.get("ci_low"), 4)}–{_number(iw.get("ci_high"), 4)}</small></td>'
                f'<td>{_number(ir.get("median_ratio"), 4)}<small>{_number(ir.get("ci_low"), 4)}–{_number(ir.get("ci_high"), 4)}</small></td>'
                f'<td>{_number(tw.get("median_ratio"), 4)}<small>{_number(tw.get("ci_low"), 4)}–{_number(tw.get("ci_high"), 4)}</small></td>'
                f'<td>{_number(tr.get("median_ratio"), 4)}<small>{_number(tr.get("ci_low"), 4)}–{_number(tr.get("ci_high"), 4)}</small></td>'
                f'<td class="{"warn" if misses else "good"}">{_text(", ".join(misses) if misses else "all pass")}</td>'
                "</tr>"
            )
        post_section = (
            '<h2>D1 post/pre regression screen</h2><div class="table-wrap"><table>'
            '<thead><tr><th>Model</th><th>Interactive wall</th><th>Interactive rate</th>'
            '<th>Throughput wall</th><th>Throughput rate</th><th>5% gates</th></tr></thead>'
            f'<tbody>{"".join(post_rows)}</tbody></table></div>'
        )
        timeline.append(
            '<li class="event failed"><span>D1 post-change screen</span>'
            f'<strong>{classification}</strong><small>frozen result; no retry or activation</small></li>'
        )

    corpus = summary.get("corpus_inventory", {})
    tests = summary.get("tests", {})
    current_test = (
        cross_verification.get("tests", {}).get("non_integration")
        if cross_verification is not None else None
    ) or (
        post_verification.get("tests", {}).get("non_integration")
        if post_verification is not None else None
    ) or tests.get("non_integration", "—")
    post_integrity = ""
    if post_verification is not None:
        verified = (post_verification.get("ok") is True
                    and post_verification.get("byte_identical_recomputation") is True)
        post_integrity = (
            '<div class="card"><small>D1 comparison integrity</small>'
            f'<strong class="{"good" if verified else "warn"}">'
            f'{"VERIFIED" if verified else "FAILED"}</strong>'
            '<span>byte-identical recomputation</span></div>'
        )
    cross_card = ""
    cross_section = ""
    if cross_commit is not None:
        classification = _text(cross_commit.get("classification", "unknown"))
        cross_card = (
            '<div class="card"><small>B27e mirrored control</small>'
            f'<strong class="warn">{classification}</strong>'
            f'<span>{_text(cross_commit.get("b27d_consequence", "unknown"))}</span></div>'
        )
        rows = []
        for block in cross_commit.get("blocks", []):
            interactive = block.get("ratios", {}).get("interactive", {})
            throughput = block.get("ratios", {}).get("throughput", {})
            rows.append(
                "<tr>"
                f'<td>Block {_text(block.get("block"))}<small>{_text(" → ".join(block.get("order", [])))}</small></td>'
                f'<td>{_number(interactive.get("d1_over_old_wall"), 4)}</td>'
                f'<td>{_number(interactive.get("d1_over_old_rate"), 4)}</td>'
                f'<td>{_number(throughput.get("d1_over_old_wall"), 4)}</td>'
                f'<td>{_number(throughput.get("d1_over_old_rate"), 4)}</td>'
                "</tr>"
            )
        cross_section = (
            '<h2>B27e mirrored OLD/D1 control</h2><div class="table-wrap"><table>'
            '<thead><tr><th>Block/order</th><th>Interactive wall</th><th>Interactive rate</th>'
            '<th>Throughput wall</th><th>Throughput rate</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
        )
        timeline.append(
            '<li class="event failed"><span>B27e mirrored control</span>'
            f'<strong>{classification}</strong><small>B27d remains inconclusive; no activation</small></li>'
        )
    cross_integrity = ""
    if cross_verification is not None:
        verified = (cross_verification.get("ok") is True
                    and cross_verification.get("byte_identical_recomputation") is True)
        cross_integrity = (
            '<div class="card"><small>B27e evidence integrity</small>'
            f'<strong class="{"good" if verified else "warn"}">'
            f'{"VERIFIED" if verified else "FAILED"}</strong>'
            '<span>4 children · mirrored order</span></div>'
        )
    d2_card = ""
    d2_section = ""
    if d2_pre is not None:
        classification = _text(d2_pre.get("classification", "unknown"))
        d2_card = (
            '<div class="card"><small>D2a exact-identity baseline</small>'
            f'<strong class="good">{classification}</strong>'
            f'<span>{_text(len(d2_pre.get("cells", [])))} same-day cells</span></div>'
        )
        d2_rows = []
        for cell in d2_pre.get("cells", []):
            model = cell.get("model", {})
            interactive = cell.get("interactive", {})
            throughput = cell.get("throughput", {})
            d2_rows.append(
                "<tr>"
                f'<td><strong>{_text(model.get("model_id"))}</strong><small>{_text(str(model.get("revision", ""))[:10])}…</small></td>'
                f'<td>{_number(interactive.get("outer_wall_ms", {}).get("median"))}</td>'
                f'<td>{_number(interactive.get("physical_tokens_per_second", {}).get("median"), 3)}</td>'
                f'<td>{_number(throughput.get("outer_wall_ms", {}).get("median"))}</td>'
                f'<td>{_number(throughput.get("physical_tokens_per_second", {}).get("median"), 3)}</td>'
                "</tr>"
            )
        d2_section = (
            '<h2>D2a same-day pre-change baseline</h2><div class="table-wrap"><table>'
            '<thead><tr><th>Model</th><th>Interactive ms</th><th>Interactive rate</th>'
            '<th>Throughput ms</th><th>Throughput rate</th></tr></thead>'
            f'<tbody>{"".join(d2_rows)}</tbody></table></div>'
        )
        timeline.append(
            '<li class="event passed"><span>D2a pre-change baseline</span>'
            f'<strong>{classification}</strong><small>exact local snapshots; no profile reuse</small></li>'
        )
    d2_integrity = ""
    if d2_pre_verification is not None:
        verified = (d2_pre_verification.get("ok") is True
                    and d2_pre_verification.get("byte_identical_recomputation") is True)
        d2_integrity = (
            '<div class="card"><small>D2a evidence integrity</small>'
            f'<strong class="{"good" if verified else "warn"}">'
            f'{"VERIFIED" if verified else "FAILED"}</strong>'
            '<span>path-free deterministic summary</span></div>'
        )
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
  <div class="card"><small>Regression suite</small><strong>{_text(current_test)}</strong><span>serial, no xdist</span></div>
  {verification_card}
  {post_card}
  {post_integrity}
  {cross_card}
  {cross_integrity}
  {d2_card}
  {d2_integrity}
</section>
<h2>Protected baseline cells</h2>
<div class="table-wrap"><table>
<thead><tr><th>Model / revision</th><th>Interactive ms</th><th>Throughput ms</th><th>Wall effect</th><th>Rate effect</th><th>MLX peak</th><th>Swap Δ</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table></div>
{post_section}
{cross_section}
{d2_section}
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
    parser.add_argument("--post-change", type=Path)
    parser.add_argument("--post-verification", type=Path)
    parser.add_argument("--cross-commit", type=Path)
    parser.add_argument("--cross-verification", type=Path)
    parser.add_argument("--d2-pre", type=Path)
    parser.add_argument("--d2-pre-verification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = json.loads(args.summary.read_text())
    verification = json.loads(args.verification.read_text()) if args.verification else None
    post_change = json.loads(args.post_change.read_text()) if args.post_change else None
    post_verification = (
        json.loads(args.post_verification.read_text()) if args.post_verification else None
    )
    cross_commit = json.loads(args.cross_commit.read_text()) if args.cross_commit else None
    cross_verification = (
        json.loads(args.cross_verification.read_text()) if args.cross_verification else None
    )
    d2_pre = json.loads(args.d2_pre.read_text()) if args.d2_pre else None
    d2_pre_verification = (
        json.loads(args.d2_pre_verification.read_text())
        if args.d2_pre_verification else None
    )
    _atomic_write(args.output, render(
        summary, verification, post_change, post_verification,
        cross_commit, cross_verification, d2_pre, d2_pre_verification,
    ))
    print(json.dumps({"output": str(args.output), "cells": len(summary["cells"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
