"""Read-only inventory for IronMule evidence artifacts.

The auditor deliberately treats every input as untrusted data.  It never rewrites a
source artifact, does not import MLX, and records source aliases instead of absolute
local paths.  The resulting JSON is a reproducible inventory, not a training corpus
and not a performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "ironmule.evidence_inventory.v1"

_COVERAGE_TERMS: dict[str, tuple[str, ...]] = {
    "environment": ("environment", "hardware", "software", "fingerprint", "system_state"),
    "workload": ("workload", "prompt", "context", "requests", "max_tokens"),
    "baseline": ("baseline", "interactive", "control"),
    "candidate": ("candidate", "throughput", "knobs", "strategy"),
    "measurements": ("sample", "metric", "timing", "ratio", "endpoint", "result"),
    "correctness": ("correctness", "token_identity", "token_digest", "stop_reason", "canonical"),
    "resources": ("memory", "mlx_peak", "peak_memory", "rss", "swap", "crash"),
    "provenance": ("commit", "code_digest", "model_digest", "manifest", "revision", "sha256", "prereg"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _tracked_raw_paths(root: Path) -> set[str]:
    raw = _git(root, "ls-files", "research/raw")
    return {line for line in raw.splitlines() if line}


def _artifact_kind(path: Path) -> str:
    name = path.name.lower()
    if name.startswith(".") and name.endswith(".partial"):
        return "partial_result"
    if "preregistration" in name and path.suffix == ".md":
        return "preregistration"
    if name.endswith("preregistration.sha256"):
        return "preregistration_checksum"
    if name.endswith("_review.md"):
        return "review"
    if "public_summary" in name and path.suffix == ".json":
        return "public_summary"
    if name.endswith("_summary.json"):
        return "legacy_summary"
    if path.suffix == ".json":
        return "raw_result"
    if path.suffix == ".md":
        return "documentation"
    if path.suffix == ".sha256":
        return "checksum"
    return "other"


def _key_paths(value: Any, prefix: str = "", *, depth: int = 0) -> set[str]:
    """Collect schema paths without retaining any artifact values."""
    if depth > 12:
        return set()
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            name = str(key).strip().lower()
            path = f"{prefix}.{name}" if prefix else name
            paths.add(path)
            paths.update(_key_paths(child, path, depth=depth + 1))
    elif isinstance(value, list):
        # Repeated measurement rows normally share a schema.  Inspect a bounded,
        # deterministic head and tail so a huge raw run cannot exhaust the auditor.
        sample = value if len(value) <= 32 else value[:16] + value[-16:]
        for child in sample:
            paths.update(_key_paths(child, prefix + "[]", depth=depth + 1))
    return paths


def _first_scalar(value: Any, wanted: tuple[str, ...]) -> str | bool | int | float | None:
    if isinstance(value, dict):
        lowered = {str(key).lower(): child for key, child in value.items()}
        for key in wanted:
            candidate = lowered.get(key)
            if isinstance(candidate, (str, bool, int, float)) or candidate is None and key in lowered:
                return candidate
        for child in value.values():
            found = _first_scalar(child, wanted)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value[:8]:
            found = _first_scalar(child, wanted)
            if found is not None:
                return found
    return None


def _json_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"json_valid": False, "json_error": type(exc).__name__}

    keys = _key_paths(payload)
    normalized = "\n".join(sorted(keys))
    coverage = {
        group: any(term in normalized for term in terms)
        for group, terms in _COVERAGE_TERMS.items()
    }
    top_level = sorted(str(key) for key in payload) if isinstance(payload, dict) else []
    return {
        "json_valid": True,
        "top_level_keys": top_level,
        "schema": _first_scalar(payload, ("schema", "schema_version")),
        "status": _first_scalar(payload, ("status", "classification", "verdict")),
        "valid_for_performance": _first_scalar(payload, ("valid_for_performance",)),
        "activation_allowed": _first_scalar(payload, ("activation_allowed",)),
        "coverage": coverage,
    }


def _record(alias: str, root: Path, path: Path, tracked: set[str]) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    record: dict[str, Any] = {
        "source": alias,
        "path": relative,
        "kind": _artifact_kind(path),
        "tracked": relative in tracked,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".json" or path.name.endswith(".json.partial"):
        record.update(_json_metadata(path))
    return record


def inventory(sources: Iterable[tuple[str, Path]], *, generated_at: str | None = None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    seen_aliases: set[str] = set()

    for alias, supplied_root in sources:
        if not alias or alias in seen_aliases:
            raise ValueError(f"source aliases must be non-empty and unique: {alias!r}")
        seen_aliases.add(alias)
        root = supplied_root.resolve()
        raw = root / "research" / "raw"
        if not raw.is_dir():
            raise ValueError(f"source {alias!r} has no research/raw directory")
        tracked = _tracked_raw_paths(root)
        source_records = [
            _record(alias, root, path, tracked)
            for path in sorted(raw.iterdir(), key=lambda item: item.name)
            if path.is_file()
        ]
        records.extend(source_records)
        source_rows.append({
            "alias": alias,
            "git_head": _git(root, "rev-parse", "HEAD") or None,
            "git_branch": _git(root, "branch", "--show-current") or None,
            "artifact_count": len(source_records),
            "tracked_count": sum(bool(row["tracked"]) for row in source_records),
            "untracked_or_ignored_count": sum(not bool(row["tracked"]) for row in source_records),
        })

    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in records:
        by_hash[row["sha256"]].append(f"{row['source']}:{row['path']}")
    duplicate_groups = [
        {"sha256": digest, "artifacts": sorted(paths)}
        for digest, paths in sorted(by_hash.items())
        if len(paths) > 1
    ]

    json_rows = [row for row in records if "json_valid" in row]
    valid_json = [row for row in json_rows if row.get("json_valid")]
    coverage = {
        group: {
            "present": sum(bool(row.get("coverage", {}).get(group)) for row in valid_json),
            "eligible_json": len(valid_json),
        }
        for group in _COVERAGE_TERMS
    }
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": SCHEMA,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(canonical).hexdigest(),
        "sources": source_rows,
        "summary": {
            "artifacts": len(records),
            "unique_artifacts": len(by_hash),
            "duplicate_groups": len(duplicate_groups),
            "tracked": sum(bool(row["tracked"]) for row in records),
            "untracked_or_ignored": sum(not bool(row["tracked"]) for row in records),
            "by_kind": dict(sorted(Counter(row["kind"] for row in records).items())),
            "json_valid": sum(bool(row.get("json_valid")) for row in json_rows),
            "json_invalid": sum(row.get("json_valid") is False for row in json_rows),
        },
        "coverage": coverage,
        "duplicate_groups": duplicate_groups,
        "records": records,
        "limitations": [
            "Field coverage is structural presence, not semantic validation.",
            "Summaries and raw samples remain distinct artifact classes.",
            "The inventory does not mutate, promote, train on, or execute any artifact.",
            "Absolute source paths are intentionally omitted.",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# B27 evidence inventory",
        "",
        "This is a read-only corpus audit. It is not a performance result, a training",
        "dataset, or permission to route a strategy. Absolute local paths are omitted.",
        "",
        f"Dataset SHA-256: `{report['dataset_sha256']}`",
        "",
        "## Sources",
        "",
        "| Alias | Git head | Artifacts | Tracked | Local-only |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in report["sources"]:
        lines.append(
            f"| `{row['alias']}` | `{row['git_head'] or 'unknown'}` | {row['artifact_count']} | "
            f"{row['tracked_count']} | {row['untracked_or_ignored_count']} |"
        )
    lines.extend([
        "",
        "## Artifact classes",
        "",
        "| Kind | Count |",
        "| --- | ---: |",
    ])
    for kind, count in summary["by_kind"].items():
        lines.append(f"| `{kind}` | {count} |")
    lines.extend([
        "",
        f"Total rows: **{summary['artifacts']}**; unique contents: "
        f"**{summary['unique_artifacts']}**; duplicate-content groups: "
        f"**{summary['duplicate_groups']}**.",
        "",
        "## Structural JSON coverage",
        "",
        "Presence below is not a quality or trust judgment.",
        "",
        "| Group | Present | Eligible valid JSON |",
        "| --- | ---: | ---: |",
    ])
    for group, row in report["coverage"].items():
        lines.append(f"| `{group}` | {row['present']} | {row['eligible_json']} |")
    lines.extend([
        "",
        "## Gate",
        "",
        "Raw samples, formal preregistrations, reviews, partials, legacy summaries and",
        "public summaries remain separate quality classes. No learned or generalized",
        "claim is permitted until provenance, replayability, missingness, censoring and",
        "leakage are validated per experiment. Local-only artifacts must not be copied",
        "into the public repository merely to make the counts look complete.",
        "",
    ])
    return "\n".join(lines)


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    temporary.replace(path)


def _source(value: str) -> tuple[str, Path]:
    alias, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("source must be ALIAS=PATH")
    return alias, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=_source, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    default_root = Path(__file__).resolve().parents[1]
    report = inventory(args.source or [("branch", default_root)], generated_at=args.generated_at)
    _atomic_write(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown:
        _atomic_write(args.markdown, markdown(report))
    print(json.dumps({"output": str(args.output), **report["summary"],
                      "dataset_sha256": report["dataset_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
