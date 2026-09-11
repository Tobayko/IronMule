"""Is this machine steady enough right now to be worth measuring on?

`B81` spent twenty-six minutes on two comparative requalifications and threw both away. Neither
was wrong to throw away: the A/A controls ran at half widths of `0.20` and `0.23` against
`B76`'s `0.0069`, and the candidate looked up to twenty per cent faster precisely *because* the
reference arm was being disturbed. The gate did its job. It just did it after the expensive part.

This is the cheap part, run first.

**It predicts nothing about which action wins.** It cannot: it never runs the candidate. It runs
the reference against itself, in fresh processes, on the same workload and the same process
lifecycle a real requalification uses, and asks one question -- would an A/A control taken now
clear the gate a real run will be held to?

**It invents no threshold.** The limits are `requalification`'s own: the A/A offset and half
width, the drift gate over blocks, `B65`, token identity and zero fallbacks. Nothing here was
derived from `B81`'s failures, because a bar set from the runs it is meant to filter is not a
bar.

**A failure is an answer.** `NOT_READY` is a result, not an error: the reference keeps serving,
the requalification stays required, and nobody paid thirteen minutes to learn it. There is no
retry loop and no waiting: the caller is told, and a person decides when to ask again.

**It is not a guarantee.** It measures a few minutes of one machine. A disturbance that arrives
after it finishes will still be caught, later and more expensively, by the real run's own gates.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from .requalification import (AA_MAX_HALF_WIDTH, AA_MAX_OFFSET, CHILD_SOURCE, DRIFT_LIMIT,
                              MAX_NEW_TOKENS, PROBE_PROMPT, REPEATS, WARMUPS, WORKLOAD_CLASS,
                              _bootstrap)

SCHEMA = "ironmule.readiness.v1"

#: Fixed here, before it ran. Three blocks of two reference children, which is the same block
#: shape one requalification session uses, so the A/A it produces is the same statistic the
#: real run will be judged by -- and a sixth of the cost.
BLOCKS = 3
CHILDREN_PER_BLOCK = 2
CHILD_TIMEOUT_S = 1800.0

PROTOCOL = {
    "schema": SCHEMA,
    "question": ("is this machine steady enough right now for a comparative requalification "
                 "to be worth its cost"),
    "not_a_prediction": ("it never runs the candidate and says nothing about which action "
                         "would win. It measures the instrument, not the result"),
    "arms": "reference against reference, in fresh processes",
    "blocks": BLOCKS, "children_per_block": CHILDREN_PER_BLOCK,
    "repeats": REPEATS, "warmups": WARMUPS,
    "workload": f"one {WORKLOAD_CLASS} request, {MAX_NEW_TOKENS} new tokens, the same fixed "
                f"prompt the requalification uses",
    "lifecycle": "one model per process, one arm per process, exactly as the real run",
    "limits": {"aa_max_offset": AA_MAX_OFFSET, "aa_max_half_width": AA_MAX_HALF_WIDTH,
               "drift_limit": DRIFT_LIMIT,
               "source": ("requalification's own limits, unchanged. None of them was derived "
                          "from the runs this is meant to filter")},
    "on_failure": ("NOT_READY. No comparison is started, no retry happens in this execution, "
                   "nothing waits, and the requalification stays required"),
    "not_a_guarantee": ("a disturbance arriving after this finishes is still caught by the "
                        "real run's own gates, later and more expensively"),
}


@dataclass(frozen=True)
class Readiness:
    ready: bool
    reason: str
    record: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"ready": self.ready, "reason": self.reason, **self.record}


def probe(model_id: str, *, blocks: int = BLOCKS, project_root: Path | None = None,
          on_progress=None) -> Readiness:
    """Reference against reference, and whether its own control could be read."""
    from .hw import (installed_memory_bytes, memory_pressure_level, swap_used_bytes,
                     vm_counters)

    root = Path(project_root or Path(__file__).resolve().parents[1])
    total_pages = (installed_memory_bytes() or 0) // 16384
    memory_total = installed_memory_bytes() or 0

    def machine(label):
        counters = vm_counters() or {}
        free_pages = (counters.get("free_count", 0) + counters.get("speculative_count", 0)
                      + counters.get("inactive_count", 0))
        return {"label": label, "swap_used_bytes": swap_used_bytes(),
                "memory_pressure_level": memory_pressure_level(),
                "load_average": list(os.getloadavg()),
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
                "memory_free_percent": (100.0 * free_pages / total_pages) if total_pages else None}

    started = time.perf_counter()
    samples = [machine("start")]
    rows, failures = [], []
    children_run = 0
    for block in range(blocks):
        entry = {"block": block, "children": {}}
        for index in range(CHILDREN_PER_BLOCK):
            # Both arms are the reference. The names exist so the ratio has a direction.
            name = "reference" if index == 0 else "reference_aa"
            spec = {"model": model_id, "arm": "reference",
                    "geometry": [0, 0], "repeats": REPEATS, "warmups": WARMUPS,
                    "prompt": PROBE_PROMPT, "max_tokens": MAX_NEW_TOKENS}
            process = subprocess.Popen(
                [sys.executable, "-u", "-c",
                 CHILD_SOURCE.format(root=str(root)), json.dumps(spec)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(root),
                env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
            try:
                stdout, stderr = process.communicate(timeout=CHILD_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            children_run += 1
            marker = next((line[2:] for line in stdout.splitlines()
                           if line.startswith("@@")), None)
            if process.returncode != 0 or marker is None:
                failures.append({"block": block, "arm": name,
                                 "returncode": process.returncode,
                                 "stderr_tail": stderr[-2000:]})
                break
            entry["children"][name] = json.loads(marker)
            if on_progress:
                on_progress(block, name)
        rows.append(entry)
        samples.append(machine(f"after_block_{block}"))
        if failures:
            break
    samples.append(machine("end"))

    complete = [entry for entry in rows if len(entry["children"]) == CHILDREN_PER_BLOCK]
    ratios = [entry["children"]["reference_aa"]["wall_ns"]
              / entry["children"]["reference"]["wall_ns"] for entry in complete]
    aa = _bootstrap(ratios)
    half_width = ((aa["ci_high"] - aa["ci_low"]) / 2 if aa["ci_high"] is not None else None)

    walls = [sum(child["wall_ns"] / 1e6 for child in entry["children"].values())
             for entry in complete]
    wall_median = median(walls) if walls else None
    disturbed = ([{"block": complete[i]["block"], "wall_ms": w,
                   "deviation": w / wall_median - 1.0}
                  for i, w in enumerate(walls) if abs(w / wall_median - 1.0) > DRIFT_LIMIT]
                 if wall_median else [])

    tokens = {tuple(child["tokens"][0]) for entry in complete
              for child in entry["children"].values()}
    stops = {tuple(child["stop_reasons"]) for entry in complete
             for child in entry["children"].values()}
    fallbacks = sum(child["fallbacks"] for entry in complete
                    for child in entry["children"].values())
    correctness_errors = sum(child["correctness_errors"] for entry in complete
                             for child in entry["children"].values())

    swaps = [s["swap_used_bytes"] for s in samples if s["swap_used_bytes"] is not None]
    frees = [s["memory_free_percent"] for s in samples if s["memory_free_percent"] is not None]
    levels = [s["memory_pressure_level"] for s in samples]
    peak_rss = max((s["peak_rss_bytes"] for s in samples), default=0)
    gate = []
    if not swaps or max(swaps) > swaps[0]:
        gate.append("swap in use rose above its value at the start")
    if not frees or min(frees) < 10.0:
        gate.append("free memory below 10 per cent")
    if not all(level == 1 and level is not None for level in levels):
        gate.append("macOS memory pressure was not normal at every probe")
    if memory_total and peak_rss > memory_total * 0.60:
        gate.append("peak child RSS above 60 per cent of installed memory")

    reasons = []
    if failures:
        reasons.append("a child failed")
    if len(complete) < blocks:
        reasons.append(f"only {len(complete)} of {blocks} blocks completed")
    if len(tokens) > 1 or len(stops) > 1:
        reasons.append("two reference children disagreed on tokens or stop reasons")
    if fallbacks or correctness_errors:
        reasons.append("a fallback or correctness error was recorded on the reference path")
    if disturbed:
        reasons.append(f"{len(disturbed)} block(s) deviated past the {DRIFT_LIMIT:.0%} drift gate")
    if gate:
        reasons.extend(gate)
    if half_width is None or half_width > AA_MAX_HALF_WIDTH:
        reasons.append(f"the A/A half width is {half_width} against a limit of "
                       f"{AA_MAX_HALF_WIDTH}")
    elif abs(aa["median"] - 1.0) > AA_MAX_OFFSET:
        reasons.append(f"the A/A median is {aa['median']:.4f}, further than {AA_MAX_OFFSET} "
                       f"from 1.0")

    record = {
        "protocol": PROTOCOL,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "blocks": rows, "complete_blocks": len(complete), "child_failures": failures,
        "children_run": children_run,
        "aa": aa, "aa_half_width": half_width,
        "block_ratios": ratios,
        "block_wall_ms": walls, "disturbed_blocks": disturbed,
        "tokens_identical": len(tokens) <= 1 and len(stops) <= 1,
        "fallbacks": fallbacks, "correctness_errors": correctness_errors,
        "resource_gate_passed": not gate, "resource_gate_reasons": gate,
        "resource_samples": samples,
        "wall_seconds": time.perf_counter() - started,
        "limits": PROTOCOL["limits"],
    }
    if reasons:
        return Readiness(False, "; ".join(reasons), record)
    return Readiness(True, (f"the A/A control reads {aa['median']:.4f} with a half width of "
                            f"{half_width:.4f}, inside the limits the real run is held to"),
                     record)


def evaluate_recorded_aa(median_ratio: float, half_width: float) -> tuple[bool, str]:
    """The same judgement, applied to an A/A arm someone else already measured.

    This exists so a past run's own control can be read against the rule without pretending
    the rule was there at the time.
    """
    if half_width is None or half_width > AA_MAX_HALF_WIDTH:
        return False, (f"half width {half_width} exceeds {AA_MAX_HALF_WIDTH}")
    if abs(median_ratio - 1.0) > AA_MAX_OFFSET:
        return False, (f"median {median_ratio:.4f} is further than {AA_MAX_OFFSET} from 1.0")
    return True, f"median {median_ratio:.4f}, half width {half_width:.4f}, inside the limits"
