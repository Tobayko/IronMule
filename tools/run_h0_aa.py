#!/usr/bin/env python3
"""Run the preregistered H0 A/A null-control sequence with hardware budgets.

The A/A design itself is frozen in ``docs/PHASE1_MATMUL_SPEC.md`` section 5.3.1:
exactly three characterization and three confirmation processes with fixed seeds.
This runner adds nothing to that contract.  It only sequences the six processes,
enforces the H1 hardware-protection budgets from
``docs/H1_VORREGISTRIERUNG_ENTWURF.md`` section 5, and fails closed.

It is a sequencer, not a measurement: it records no result of its own and lives
outside both the H0 and H0.1 closed code lists.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Frozen by docs/PHASE1_MATMUL_SPEC.md 5.3.1 -- not tunable here.
PROCESS_SETS = ("characterization", "confirmation")
PROCESS_INDICES = (0, 1, 2)

# Budgets from docs/H1_VORREGISTRIERUNG_ENTWURF.md section 5.
COOLDOWN_BETWEEN_PROCESSES_S = 60.0
MAX_WALL_SECONDS = 20 * 60


def _already_recorded() -> set[tuple[str, int]]:
    """Return the A/A process tuples already recorded for this exact provenance.

    The store is append-only and a second measurement of the same tuple carries
    different numbers, so it would be refused rather than replace the evidence.
    Resuming therefore means skipping what already exists, never re-running it.
    """

    import sqlite3

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from friday_h0.provenance import collect_provenance
    from friday_h0.runner import run_id_for

    provenance = collect_provenance()
    database = PROJECT_ROOT / ".friday-data" / "h0.sqlite3"
    if not database.exists():
        return set()
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        present = {row[0] for row in connection.execute("SELECT run_id FROM runs WHERE mode='aa_gpu'")}
    finally:
        connection.close()
    return {
        (process_set, index)
        for process_set in PROCESS_SETS
        for index in PROCESS_INDICES
        if run_id_for("aa_gpu", process_set, index, provenance) in present
    }


def _process_succeeded(exit_code: int, stdout: str) -> bool:
    """A single A/A process can never be 'promoted'; that needs the aggregate.

    The CLI therefore returns 10 for a perfectly valid process, so the exit code
    alone cannot tell success from failure.  The recorded status can.
    """

    if exit_code not in (0, 10):
        return False
    return "status=completed" in stdout and "classification=measurement_complete" in stdout


# tools/ is loaded as loose scripts, not a package, so make the directory
# importable before pulling in the shared preconditions.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench import release_gate, require_ac_power  # noqa: E402

_require_ac_power = require_ac_power


def _self_check() -> int:
    """Offline checks of the sequencer's decision rules; no GPU, no subprocess."""

    # A single A/A process cannot be "promoted" -- that needs the aggregate -- so
    # the CLI returns 10 for a perfectly valid process.  Judging by exit code
    # alone once aborted a healthy run after its first process.
    ok = "status=completed classification=measurement_complete"
    assert _process_succeeded(10, ok)
    assert _process_succeeded(0, ok)
    for code in (64, 65, 66, 70, 78):
        assert not _process_succeeded(code, ok), code
    # warmup_unstable really produced this: tolerated code, invalid status.
    assert not _process_succeeded(10, "status=invalid classification=invalid")
    assert not _process_succeeded(10, "")

    # The process plan is frozen by docs/PHASE1_MATMUL_SPEC.md 5.3.1.
    assert PROCESS_SETS == ("characterization", "confirmation")
    assert PROCESS_INDICES == (0, 1, 2)

    print(json.dumps({"self_check": "pass", "checks": 10}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_h0_aa", allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    # Same release gate as every other measuring tool.  Without it an accidental
    # invocation starts six GPU processes with 60 s cooldowns between them; only
    # the resume check stood between a stray call and a real run.
    gated = release_gate(args, _self_check)
    if gated is not None:
        return gated

    power = _require_ac_power()
    started = time.monotonic()
    log: list[dict[str, object]] = []
    print(json.dumps({"state": "aa_start", "power_source": power}), flush=True)

    recorded = _already_recorded()
    planned = [
        (s, i) for s in PROCESS_SETS for i in PROCESS_INDICES if (s, i) not in recorded
    ]
    if recorded:
        print(
            json.dumps(
                {
                    "state": "resuming",
                    "already_recorded": sorted(f"{s}[{i}]" for s, i in recorded),
                    "remaining": len(planned),
                }
            ),
            flush=True,
        )
    for position, (process_set, index) in enumerate(planned):
        elapsed = time.monotonic() - started
        if elapsed > MAX_WALL_SECONDS:
            print(json.dumps({"state": "wall_budget_exceeded", "elapsed_s": elapsed}), flush=True)
            return 3
        if position:
            # Cooldown between processes: thermal headroom, and it keeps the
            # replicates from sharing one continuous thermal state.
            time.sleep(COOLDOWN_BETWEEN_PROCESSES_S)

        argv = [
            sys.executable,
            "-m",
            "friday_h0.cli",
            "mlx-run",
            "--mode",
            "aa_gpu",
            "--process-set",
            process_set,
            "--process-index",
            str(index),
            "--execute",
        ]
        process_started = time.monotonic()
        completed = subprocess.run(argv, cwd=PROJECT_ROOT, capture_output=True, check=False)
        wall = time.monotonic() - process_started
        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        entry = {
            "process_set": process_set,
            "process_index": index,
            "exit_code": completed.returncode,
            "wall_seconds": wall,
            "stdout": stdout,
        }
        entry["succeeded"] = _process_succeeded(completed.returncode, stdout)
        log.append(entry)
        print(json.dumps(entry), flush=True)
        if not entry["succeeded"]:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            print(
                json.dumps(
                    {
                        "state": "aa_aborted",
                        "at": f"{process_set}[{index}]",
                        "exit_code": completed.returncode,
                        "stderr_tail": stderr[-400:],
                    }
                ),
                flush=True,
            )
            return completed.returncode

    print(
        json.dumps(
            {
                "state": "aa_sequence_complete",
                "processes": len(log),
                "total_wall_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
