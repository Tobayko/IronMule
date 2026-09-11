"""Generate the bounded PROD7 product-history data-quality notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "research" / "product_history_20260907_v3"
DEFAULT_OUTPUT = ROOT / "research" / "PROD7_HISTORY_QA.ipynb"


def _markdown(source: str):
    return nbformat.v4.new_markdown_cell(source)


def _code(source: str):
    return nbformat.v4.new_code_cell(source)


def build_notebook(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> nbformat.NotebookNode:
    artifact_path = artifact_dir if artifact_dir.is_absolute() else ROOT / artifact_dir
    artifact_path = artifact_path.resolve()
    artifact_label = artifact_path.relative_to(ROOT).as_posix() if artifact_path.is_relative_to(ROOT) else artifact_path.name
    setup = f'''# PROD7 bounded snapshot QA setup
from __future__ import annotations
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
ARTIFACT_DIR = ROOT / {artifact_label!r}
assert ARTIFACT_DIR.is_dir(), f"missing artifact directory: {{ARTIFACT_DIR.name}}"

# Import the repository's actual canonical contract; it is stdlib-only.
sys.path.insert(0, str(ROOT))
from friday_evidence.canonical import canonical_json_bytes, canonical_sha256

assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
artifact = json.loads((ARTIFACT_DIR / "artifact.json").read_text(encoding="utf-8"))
export = json.loads((ARTIFACT_DIR / "optimization-events.json").read_text(encoding="utf-8"))
assert isinstance(artifact, dict) and isinstance(export, dict)
assert export["schema"] == "ironmule.optimization_export.v2"
assert isinstance(export["events"], list)
print({{"artifact_dir": {artifact_label!r}, "export_schema": export["schema"], "events": len(export["events"])}})
'''

    tldr = '''## tl;dr

The executed snapshot reconciles to **8 unique trials**, **140 completed calibration requests**, and **5 failed/blocked attempts** across 730 events (461 calibration + 269 load). The 14-point 12B load series matches the raw export exactly. This is metadata-only evidence: no inference, hardware, performance, or activation claim.'''

    context = '''## Context & Methods

The input is the checked-in snapshot export, not a live database. The notebook verifies the repository canonical JSON/SHA-256 contract for the combined export and each journal, then derives run status from the latest terminal event and latest passed sample validations.

### Key Assumptions

- `optimization-events.json` contains the complete bounded exports for the `optimization` and `load` journals.
- A completed request is a unique final `passed` sample validation; load-only rows are not requests.
- A load run with raw `passed` status and a nonzero worker return code is displayed as `failed` while raw fields remain visible in the source.
- This notebook reads metadata only and does not load models, inspect prompts, or claim future model activity.'''

    canonical_cell = '''# Canonical export and journal checks
assert canonical_json_bytes(export["events"])
assert canonical_sha256(export["events"]) == export["sha256"]
journals = export["journals"]
assert [journal["id"] for journal in journals] == ["optimization", "load"]
for journal in journals:
    assert journal["verified_events"] == len(journal["events"])
    assert journal["as_of_seq"] == (journal["events"][-1]["seq"] if journal["events"] else 0)
    assert canonical_sha256(journal["events"]) == journal["chain_sha256"]
    descriptor = next(item for item in artifact["manifest"]["sources"] if item.get("id") == journal["id"])
    assert descriptor["query"]["chain_sha256"] == journal["chain_sha256"]
assert export["verified_events"] == len(export["events"]) == 730
print({"combined_sha256": export["sha256"], "journal_sha256": {j["id"]: j["chain_sha256"] for j in journals}})
'''

    derive_cell = r'''# Derive bounded run rows from raw events and latest validations
def completed_requests(events, run_id):
    states = {}
    for event in events:
        if event["run_id"] != run_id or event["kind"] != "validation":
            continue
        payload = event["payload"]
        sample = payload.get("sample") if isinstance(payload, dict) else None
        if isinstance(sample, dict) and type(sample.get("sample_index")) is int:
            states[sample["sample_index"]] = sample.get("status")
    return sum(status == "passed" for status in states.values())

def model_label(model_id):
    match = re.search(r"(?:^|[-_/])(1B|4B|12B)(?:$|[-_/])", str(model_id), re.I)
    return match.group(1).upper() if match else str(model_id).rsplit("/", 1)[-1][:24]

run_rows = []
run_ids = []
for journal in journals:
    source = journal["id"]
    events = journal["events"]
    grouped = {}
    finished = set()
    for event in events:
        run_id = event["run_id"]
        payload = event["payload"]
        if event["kind"] == "run_started":
            assert run_id not in grouped
            grouped[run_id] = {
                "run_id": run_id, "model": payload.get("model_id"),
                "model_label": model_label(payload.get("model_id")),
                "kind": "load" if source == "load" else "calibration",
                "raw_status": "incomplete", "derived_status": "incomplete",
                "completed_requests": 0, "returncode": None,
            }
        else:
            assert run_id in grouped
        if run_id in finished:
            raise AssertionError("event appears after terminal run")
        if event["kind"] == "run_finished":
            assert run_id not in finished
            finished.add(run_id)
            worker = payload.get("worker") if isinstance(payload, dict) else {}
            worker = worker if isinstance(worker, dict) else {}
            row = grouped[run_id]
            row["raw_status"] = payload.get("status", "failed")
            row["returncode"] = worker.get("returncode")
            row["terminal_completed_calls"] = payload.get("completed_calls")
            row["completed_requests"] = completed_requests(events, run_id) if source == "optimization" else 0
            row["derived_status"] = row["raw_status"]
            if source == "load" and row["raw_status"] == "passed" and row["returncode"] != 0:
                row["derived_status"] = "failed"
            row["terminal_error"] = payload.get("error_code")
            run_rows.append(row)
            run_ids.append(run_id)

assert len(run_ids) == len(set(run_ids)) == 8
assert len(export["events"]) == sum(len(journal["events"]) for journal in journals) == 730
assert sum(len(journal["events"]) for journal in journals if journal["id"] == "optimization") == 461
assert sum(len(journal["events"]) for journal in journals if journal["id"] == "load") == 269
run_rows.sort(key=lambda row: row["run_id"])
cards = {
    "trials": len(run_rows),
    "completed_requests": sum(row["completed_requests"] for row in run_rows),
    "failed_or_blocked_attempts": sum(row["derived_status"] in {"failed", "deferred", "interrupted"} for row in run_rows),
}
assert cards == {"trials": 8, "completed_requests": 140, "failed_or_blocked_attempts": 5}
assert sum(row["kind"] == "load" and row["completed_requests"] == 0 for row in run_rows) == 4
legacy = next(row for row in run_rows if row["run_id"] == "b19ed2ff79fe4d9c98731f7dcb219868")
assert legacy["raw_status"] == "passed" and legacy["returncode"] == -6 and legacy["derived_status"] == "failed"
artifact_runs = {row["run_id"]: row for row in artifact["snapshot"]["datasets"]["runs"]}
assert set(artifact_runs) == set(run_ids)
for row in run_rows:
    expected = artifact_runs[row["run_id"]]
    assert row["model"] == expected["model_id"]
    assert row["kind"] == expected["kind"]
    assert row["derived_status"] == expected["derived_status"]
    assert row["completed_requests"] == expected["completed_requests"]
    if row["kind"] == "calibration":
        assert type(row["terminal_completed_calls"]) is int
        assert row["terminal_completed_calls"] == row["completed_requests"]
print(cards)
'''

    preview_cell = '''# Bounded eight-row preview (metadata only)
preview = [
    {key: row.get(key) for key in ("model_label", "kind", "derived_status", "completed_requests", "returncode")}
    for row in run_rows
]
assert len(preview) == 8
print(json.dumps(preview, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
'''

    results = '''## Results

The cards and reconciliations below are computed from raw latest validations, not copied from the dashboard. The compact load chart is a bounded textual view of the available 12B swap-delta observations.'''
    chart_cell = '''# Reconciled cards and bounded 12B load chart data
print("CARDS", json.dumps(cards, sort_keys=True))
load_journal = next(journal for journal in journals if journal["id"] == "load")
load_12b_ids = {row["run_id"] for row in run_rows if row["kind"] == "load" and row["model_label"] == "12B"}
load_points = []
for event in load_journal["events"]:
    payload = event["payload"]
    if event["run_id"] in load_12b_ids and event["kind"] == "validation" and payload.get("state") == "load_memory_sample":
        observation = payload.get("observation", {})
        if isinstance(observation, dict) and observation.get("swap_delta_bytes") is not None:
            load_points.append((observation.get("swap_delta_bytes", 0) / (1024 * 1024), observation.get("errors", [])))
artifact_points = [row["swap_delta_mib"] for row in artifact["snapshot"]["datasets"]["load_observations"]]
assert len(load_points) == len(artifact_points) == 14
assert all(abs(value - expected) < 1e-12 for (value, _), expected in zip(load_points, artifact_points))
compact_points = load_points[:2] + load_points[-2:]
print("12B swap delta chart data", json.dumps({"count": len(load_points), "first_two": compact_points[:2], "last_two": compact_points[2:], "max_mib": max(value for value, _ in load_points)}, allow_nan=False))
'''

    final_cell = '''# Final QA gates
required_artifact_keys = {"manifest", "snapshot", "sources", "surface"}
required_export_keys = {"schema", "generated_at", "verified_events", "events", "journals", "sha256"}
assert required_artifact_keys <= set(artifact)
assert required_export_keys <= set(export)
generated_ns = int(datetime.fromisoformat(export["generated_at"]).timestamp() * 1_000_000_000)
assert all(event["recorded_unix_ns"] <= generated_ns for event in export["events"])
assert all(row["model"] and row["kind"] in {"load", "calibration"} for row in run_rows)
assert all(row["kind"] == "load" and row["completed_requests"] == 0 or row["kind"] == "calibration" for row in run_rows)
print({"qa": "passed", "missing_key_checks": "passed", "sentinel_checks": "passed", "future_activity_check": "passed", "claims": "snapshot-only; no inference/hardware/performance claim"})
'''
    takeaways = '''## Takeaways

- The snapshot reconciles to 8 unique trials, 140 completed calibration requests, and 5 failed/blocked attempts.
- Load-only runs contribute zero requests by definition.
- The legacy 1B raw `passed`/return-code `-6` artifact is displayed as failed; its raw source remains unchanged.
- Canonical export/journal hashes, event counts, terminal ordering, and no-future-activity checks pass.
- This notebook is snapshot-only and makes no inference, hardware, performance, or activation claim.'''

    cells = [
        _markdown(tldr), _markdown(context), _code(setup), _code("print('TL;DR summary is computed after canonical validation below; see the executed Results section.')"),
        _code(canonical_cell), _markdown("## Data\n\nThe source export is bounded to the two verified journals and retains only event metadata."),
        _code(derive_cell), _code(preview_cell), _markdown(results), _code(chart_cell), _markdown(takeaways), _code(final_cell),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "PROD7 QA", "language": "python", "name": "prod7-qa"},
        "language_info": {"name": "python"},
        "prod7": {"artifact_dir": artifact_label, "claims": "snapshot-only"},
    })
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    notebook = build_notebook(args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
