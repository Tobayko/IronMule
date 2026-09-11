#!/usr/bin/env python3
"""Does "any swapout blocks" describe memory pressure, or does it describe a busy machine?

Two `12B` stack `C` confirmations were blocked by one gate and nothing else. Both showed the
same shape: ten or so blocks with a flat swapout counter, one burst, then flat again, while
swap *in use* fell across the whole run and free memory never left the fifties. Neither
blocked run is reinterpreted here and neither ever will be: a rule changed after a run it
blocked cannot bless that run. The question is only whether the rule is the right instrument
for the next one.

`Swapouts` from `vm_stat` is a system-wide monotone counter. It counts pages this machine
wrote to swap for any reason, including compacting a swap file it already holds. A gate that
blocks on it attributes the whole machine's housekeeping to whatever happens to be measuring
at the time. Apple does not define memory pressure that way — `kern.memorystatus_vm_pressure_level`
is the system's own answer, and it is readable without starting a process.

This file reads the VM counters natively, harvests every resource trace this project already
recorded, and asks one question of both the old gate and a preregistered candidate: do they
separate the runs that were demonstrably starved from the runs that were not.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PAGE_BYTES = 16384
HOST_VM_INFO64 = 4
PRESSURE_NORMAL = 1
PRESSURE_NAMES = {1: "normal", 2: "warn", 4: "critical"}


class _VMStatistics64(ctypes.Structure):
    """`<mach/vm_statistics.h>`: what `vm_stat(1)` prints, without starting `vm_stat(1)`."""

    _fields_ = [
        ("free_count", ctypes.c_uint32), ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32), ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64), ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64), ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64), ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64), ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64), ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64), ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64), ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32), ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32), ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


_LIBC = None


def _libc():
    global _LIBC
    if _LIBC is None:
        _LIBC = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    return _LIBC


def vm_counters() -> dict | None:
    """Every VM figure the gate needs, in one Mach call and no subprocess."""
    try:
        stats = _VMStatistics64()
        count = ctypes.c_uint32(ctypes.sizeof(stats) // ctypes.sizeof(ctypes.c_int32))
        if _libc().host_statistics64(_libc().mach_host_self(), HOST_VM_INFO64,
                                     ctypes.byref(stats), ctypes.byref(count)) != 0:
            return None
        return {name: int(getattr(stats, name)) for name, _t in _VMStatistics64._fields_}
    except (OSError, AttributeError, ValueError, TypeError):
        return None


def pressure_level() -> int | None:
    """Apple's own answer: 1 normal, 2 warn, 4 critical. None means the machine did not say."""
    try:
        value = ctypes.c_int()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if _libc().sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(value),
                                ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
            return None
        return int(value.value)
    except (OSError, AttributeError, ValueError, TypeError):
        return None


def verify_native_readers() -> dict:
    """The native path has to agree with the tools it replaces, or it is not a replacement."""
    # A monotone counter on a live machine moves between two reads, so equality is the
    # wrong test: the shell reading is sandwiched between two native ones and has to land
    # inside them. A first attempt asserted equality and failed on `pageins` by exactly 1.
    before = vm_counters()
    text = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    after = vm_counters()
    native = after
    fields = {"swapouts": r"Swapouts:\s*([0-9]+)", "swapins": r"Swapins:\s*([0-9]+)",
              "pageouts": r"Pageouts:\s*([0-9]+)", "pageins": r"Pageins:\s*([0-9]+)",
              "compressor_page_count": r"Pages occupied by compressor:\s*([0-9]+)"}
    compared = {}
    for name, pattern in fields.items():
        found = re.search(pattern, text)
        shell = int(found.group(1)) if found else None
        low = before.get(name) if before else None
        high = after.get(name) if after else None
        compared[name] = {
            "native_before": low, "vm_stat": shell, "native_after": high,
            "equal": (low is not None and high is not None and shell is not None
                      and low <= shell <= high),
        }
    level = pressure_level()
    return {
        "monotone_counters": compared,
        "all_monotone_counters_bracket_the_shell_reading": all(row["equal"] for row in compared.values()),
        "instantaneous_note": (
            "free_count and wire_count are instantaneous and are not compared: the native "
            "read and the vm_stat read happen microseconds apart and legitimately differ"
        ),
        "pressure_level": level,
        "pressure_level_name": PRESSURE_NAMES.get(level),
        "no_subprocess_in_the_native_path": True,
    }


# ---------------------------------------------------------------------------
# The candidate gate, defined before any historical trace is scored.
# ---------------------------------------------------------------------------

GATE_OLD = {
    "id": "G_old",
    "rule": "swapout_counter_delta over the run must be 0; any swapout blocks",
    "provenance": "the gate as shipped in tools/b57_stack_composition.py",
}

GATE_CANDIDATE = {
    "id": "G_pressure",
    "rule": (
        "block if ANY of: (1) Apple's kern.memorystatus_vm_pressure_level is anything but "
        "normal at any probe, or is unreadable; (2) system free memory below 10 per cent at "
        "any probe; (3) combined peak RSS above 60 per cent of installed memory; (4) swap in "
        "use exceeds its value at run start at any probe; (5) any block deviates more than 25 "
        "per cent from the median block; (6) any fallback or any correctness difference within "
        "a plan kind; (7) any required probe value missing. Swapout, swapin and pageout "
        "counters are recorded in full and are not a blocking condition on their own"
    ),
    "where_each_threshold_comes_from": {
        "1": ("Apple's own definition of memory pressure, read through "
              "kern.memorystatus_vm_pressure_level. Not a number chosen here"),
        "2": "10 per cent, unchanged from the shipped gate",
        "3": "60 per cent of installed memory, unchanged from the shipped gate",
        "4": ("a zero-growth safety budget, not a fitted threshold: the run may not leave the "
              "machine holding more swap than it found. Starvation writes our pages out and "
              "swap in use RISES; compaction rewrites pages the machine already held and swap "
              "in use does not rise. The budget is zero precisely so that no number has to be "
              "chosen, and so that no burst size can be tuned around"),
        "5": "25 per cent, unchanged from the shipped gate. This is what catches thrashing: "
             "a run whose pages are being evicted and faulted back gets slower blocks",
        "6": "unchanged from the shipped gate",
        "7": "fail-closed: a gate that cannot see is a gate that blocks",
    },
    "what_it_deliberately_does_not_do": (
        "it does not pick a swapout burst size. No threshold in it can be satisfied by making "
        "a burst smaller, and none was derived from the size of any burst in B57 or B63"
    ),
    "why_the_swapout_counter_stops_being_a_blocker": (
        "vm_stat Swapouts is system-wide and monotone. It counts every page this machine wrote "
        "to swap for any reason, including compacting a swap file it already holds. A run long "
        "enough to matter will see it move on a machine that holds gigabytes of swap. Blocking "
        "on it attributes the machine's housekeeping to the measurement. The counter stays in "
        "the record because it is evidence; it stops being a verdict because it is not one"
    ),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _series(samples: list[dict], key: str) -> list:
    return [s[key] for s in samples if s.get(key) is not None]


def score_run(name: str, samples: list[dict], *, memory_total: int,
              disturbed_blocks: int | None, fallbacks: int | None,
              correctness_ok: bool | None, expected_class: str) -> dict:
    swapouts = _series(samples, "swapouts")
    swap_used = _series(samples, "swap_used_bytes")
    free = _series(samples, "memory_free_percent")
    rss = _series(samples, "peak_rss_bytes")
    mlx = _series(samples, "mlx_peak_bytes")

    swapout_delta = (max(swapouts) - min(swapouts)) if len(swapouts) >= 2 else None
    swap_start = swap_used[0] if swap_used else None
    swap_max = max(swap_used) if swap_used else None
    swap_growth_above_start = (swap_max - swap_start) if swap_used else None

    old_blocks = swapout_delta is None or swapout_delta != 0
    reasons = []
    if swap_used and swap_max > swap_start:
        reasons.append(f"swap in use rose {(swap_max - swap_start)/1e9:.2f} GB above its start")
    if free and min(free) < 10.0:
        reasons.append(f"free memory reached {min(free):.0f} per cent")
    if rss and memory_total and max(rss) > memory_total * 0.60:
        reasons.append("peak RSS above 60 per cent of installed memory")
    if disturbed_blocks:
        reasons.append(f"{disturbed_blocks} disturbed block(s)")
    if fallbacks:
        reasons.append(f"{fallbacks} fallback(s)")
    if correctness_ok is False:
        reasons.append("a correctness difference within a plan kind")
    if not swap_used or not free:
        reasons.append("a required probe value is missing (fail-closed)")

    return {
        "run": name,
        "expected_class": expected_class,
        "samples": len(samples),
        "swapout_counter_delta": swapout_delta,
        "swap_used_start_bytes": swap_start,
        "swap_used_max_bytes": swap_max,
        "swap_used_end_bytes": swap_used[-1] if swap_used else None,
        "swap_growth_above_start_bytes": swap_growth_above_start,
        "memory_free_percent_min": min(free) if free else None,
        "peak_rss_bytes": max(rss) if rss else None,
        "peak_rss_fraction_of_installed": (max(rss) / memory_total) if rss and memory_total else None,
        "mlx_peak_bytes": max(mlx) if mlx else None,
        "disturbed_blocks": disturbed_blocks,
        "fallbacks": fallbacks,
        "correctness_identical_within_plan": correctness_ok,
        "pressure_level_recorded": False,
        "G_old": "BLOCK" if old_blocks else "PASS",
        "G_pressure_without_criterion_1": "BLOCK" if reasons else "PASS",
        "G_pressure_reasons": reasons,
    }


def harvest(memory_total: int) -> list[dict]:
    raw = PROJECT_ROOT / "research" / "raw"
    scored = []

    composition = [
        ("B57 4B selection", "B57_stack_composition_20260910_selection.json", "quiet"),
        ("B57 4B confirmation", "B57_stack_composition_20260910_confirmation.json", "quiet"),
        ("B57 4B confirmation (parallel session)", "B57_stack_composition_20260910_4b_confirmation.json", "quiet"),
        ("B57 12B selection", "B57_stack_composition_20260910_12b_selection.json", "quiet"),
        ("B57 12B confirmation (blocked 1)", "B57_stack_composition_20260910_12b_confirmation.json", "disputed"),
        ("B63 12B confirmation (blocked 2)", "B57_stack_composition_20260910_12b_confirmation_b63.json", "disputed"),
    ]
    for name, filename, expected in composition:
        path = raw / filename
        if not path.is_file():
            continue
        record = _load(path)
        scored.append(score_run(
            name, record["resources"]["samples"],
            memory_total=record["resources"].get("memory_total_bytes") or memory_total,
            disturbed_blocks=len(record["stability"]["disturbed_blocks"]),
            fallbacks=record["resources"]["fallbacks"],
            correctness_ok=record["correctness"]["identical_within_plan"],
            expected_class=expected))

    tune = raw / "B61_12b_tune_without_wired_20260910_attempt2.json"
    if tune.is_file():
        record = _load(tune)
        scored.append(score_run(
            "B61 12B tune with confirmation (two resident models)", record["samples"],
            memory_total=memory_total, disturbed_blocks=None, fallbacks=None,
            correctness_ok=None, expected_class="loaded"))

    for label, filename in (("B60 12B child, wired_fraction 0.6", "B60_12b_child_diagnosis_20260910_conditionA.json"),
                            ("B60 12B child, wired_fraction off", "B60_12b_child_diagnosis_20260910_conditionB.json")):
        path = raw / filename
        if not path.is_file():
            continue
        record = _load(path)
        scored.append(score_run(label, record["runs"][0]["samples"], memory_total=memory_total,
                                disturbed_blocks=None, fallbacks=None, correctness_ok=None,
                                expected_class="quiet"))

    observation = raw / "B63_12b_stack_c_replacement_20260910.json"
    if observation.is_file():
        record = _load(observation)
        trace = record["observation_trace"]
        scored.append(score_run("B63 pre-start observation, NOTHING measuring", trace,
                                memory_total=memory_total, disturbed_blocks=None,
                                fallbacks=None, correctness_ok=None,
                                expected_class="idle_control"))
    return scored


def _sample_lists(node, path=""):
    """Every list of probes in a record, wherever a study happened to put it."""
    found = []
    if isinstance(node, list) and node and isinstance(node[0], dict) \
            and {"swapouts", "swap_used_bytes"} <= set(node[0]):
        found.append((path, node))
    elif isinstance(node, dict):
        for key, value in node.items():
            found += _sample_lists(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node[:8]):
            found += _sample_lists(value, f"{path}[{index}]")
    return found


def sweep_corpus(memory_total: int, already: set[str]) -> dict:
    """How often would the old gate have blocked a run that shows no sign of starvation?

    This does not classify anything as quiet or loaded. It counts, across every trace this
    project happens to have kept, how the two gates disagree — and a disagreement is always
    the same shape: the system-wide counter moved while the machine's own memory figures
    stayed where they were.
    """
    rows, unscoreable = [], []
    for path in sorted((PROJECT_ROOT / "research" / "raw").glob("*.json")):
        if path.name in already or path.stat().st_size > 20_000_000:
            continue
        try:
            record = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        for where, samples in _sample_lists(record):
            if len(samples) < 2:
                continue
            if not _series(samples, "memory_free_percent"):
                unscoreable.append({"file": path.name, "at": where,
                                    "reason": "no free-memory probe; fail-closed under the candidate"})
                continue
            scored = score_run(f"{path.name}{where}", samples, memory_total=memory_total,
                               disturbed_blocks=None, fallbacks=None, correctness_ok=None,
                               expected_class="unlabelled")
            rows.append(scored)
    disagree = [r for r in rows if r["G_old"] != r["G_pressure_without_criterion_1"]]
    return {
        "traces_scored": len(rows),
        "traces_not_scoreable_fail_closed": unscoreable,
        "agree": len(rows) - len(disagree),
        "disagree": len(disagree),
        "disagreements": [{"run": r["run"], "G_old": r["G_old"],
                           "G_pressure": r["G_pressure_without_criterion_1"],
                           "swapout_counter_delta": r["swapout_counter_delta"],
                           "swap_growth_above_start_bytes": r["swap_growth_above_start_bytes"],
                           "memory_free_percent_min": r["memory_free_percent_min"]}
                          for r in disagree],
        "every_disagreement_is_the_old_gate_blocking_alone": all(
            r["G_old"] == "BLOCK" and r["G_pressure_without_criterion_1"] == "PASS"
            for r in disagree),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preregister", type=Path)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once
    from ironmule.hw import installed_memory_bytes

    if args.preregister:
        write_once(args.preregister, json.dumps({
            "experiment": "B65_gate_semantics",
            "question": ("does the shipped resource gate describe memory pressure on macOS, "
                         "or does it describe a machine that holds a swap file"),
            "gates_compared": [GATE_OLD, GATE_CANDIDATE],
            "decision_rule": (
                "GATE_CONFIRMED only if the candidate passes every run this project recorded "
                "as quiet AND blocks every run with independent evidence of starvation, and "
                "is fail-closed on missing data. Anything else is OLD_GATE_RETAINED. The "
                "candidate is not adopted because C would pass under it: C's own runs are "
                "scored but are not evidence for or against the gate, and are labelled "
                "'disputed' for exactly that reason"
            ),
            "irreversible": (
                "the two blocked sessions stay BLOCKED whatever this finds. A rule changed "
                "after a run it blocked cannot bless that run. Any confirmation under a new "
                "gate is a new, separately preregistered first confirmation"
            ),
            "no_new_model_measurement": (
                "the historical traces answer the gate question. Only the native readers are "
                "exercised live, and they touch no model"
            ),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True))
        print(f"preregistration written to {args.preregister}")
        return 0

    memory_total = installed_memory_bytes() or 0
    readers = verify_native_readers()
    scored = harvest(memory_total)

    by_class: dict[str, list[dict]] = {}
    for row in scored:
        by_class.setdefault(row["expected_class"], []).append(row)

    quiet = by_class.get("quiet", [])
    loaded = by_class.get("loaded", [])
    idle = by_class.get("idle_control", [])
    disputed = by_class.get("disputed", [])

    separates = (bool(quiet) and bool(loaded)
                 and all(r["G_pressure_without_criterion_1"] == "PASS" for r in quiet)
                 and all(r["G_pressure_without_criterion_1"] == "BLOCK" for r in loaded))
    verdict = "GATE_CONFIRMED" if (separates and readers["all_monotone_counters_bracket_the_shell_reading"]) else "OLD_GATE_RETAINED"

    record = {
        "experiment": "B65_gate_semantics",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "native_readers": readers,
        "installed_memory_bytes": memory_total,
        "gates_compared": [GATE_OLD, GATE_CANDIDATE],
        "runs_scored": scored,
        "separation": {
            "quiet_runs_passing_candidate": sum(r["G_pressure_without_criterion_1"] == "PASS" for r in quiet),
            "quiet_runs_total": len(quiet),
            "loaded_runs_blocked_by_candidate": sum(r["G_pressure_without_criterion_1"] == "BLOCK" for r in loaded),
            "loaded_runs_total": len(loaded),
            "quiet_runs_blocked_by_old_gate": sum(r["G_old"] == "BLOCK" for r in quiet),
            "idle_control": [{"run": r["run"], "G_old": r["G_old"],
                              "G_pressure": r["G_pressure_without_criterion_1"],
                              "swapout_counter_delta": r["swapout_counter_delta"]} for r in idle],
            "disputed_not_used_as_evidence": [r["run"] for r in disputed],
        },
        "corpus_sweep": sweep_corpus(memory_total, already={
            "B57_stack_composition_20260910_selection.json",
            "B57_stack_composition_20260910_confirmation.json",
            "B57_stack_composition_20260910_4b_confirmation.json",
            "B57_stack_composition_20260910_12b_selection.json",
            "B57_stack_composition_20260910_12b_confirmation.json",
            "B57_stack_composition_20260910_12b_confirmation_b63.json",
            "B61_12b_tune_without_wired_20260910_attempt2.json",
            "B60_12b_child_diagnosis_20260910_conditionA.json",
            "B60_12b_child_diagnosis_20260910_conditionB.json",
            "B63_12b_stack_c_replacement_20260910.json",
        }),
        "criterion_1_caveat": (
            "Apple's pressure level was never recorded in any historical trace, so criterion "
            "1 cannot be validated retrospectively. It is included because it only ever adds "
            "blocking power and is fail-closed; the separation reported here rests entirely "
            "on criteria 2 to 7, which are all present in the traces"
        ),
        "verdict": verdict,
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "native_readers_agree": readers["all_monotone_counters_bracket_the_shell_reading"],
                      "separation": record["separation"]}, indent=2, default=str))
    for row in scored:
        print(f'  {row["expected_class"]:13} {row["run"]:48} G_old={row["G_old"]:5} '
              f'G_pressure={row["G_pressure_without_criterion_1"]:5} '
              f'swapout_delta={row["swapout_counter_delta"]} '
              f'swap_growth={row["swap_growth_above_start_bytes"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
