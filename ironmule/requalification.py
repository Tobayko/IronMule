"""The only way back out of `REQUALIFICATION_REQUIRED`, and it has to be asked for.

`B80` can take a qualified action away and deliberately has no way to give it back. An
observation says how long one dispatch took and nothing about what the alternative would have
cost, so no number of them may restore a preference. That property is what keeps users out of
experiments, and it also means a machine that drifts once stays on the reference for good
unless something else exists. This is that something else.

**It is a measurement, not a reset.** A requalification runs the same shape of comparison the
original qualification did: independent sessions, blocked and rotated, one resident model per
process, an A/A control in every block, byte identity for every admitted projection before a
token is timed, the `B65` resource gate, and a drift gate over blocks. It produces comparative
evidence and is the only thing in the system that can.

**A person starts it.** Nothing schedules it, no dispatch triggers it, and it refuses to run
unless the state actually requires it and the machine still matches what was qualified.

**Old evidence is never deleted.** A requalification opens a new epoch: the previous state file
is archived under its own name, the new qualification's rows carry their own ids, and a lineage
record makes the whole history readable -- qualified, drift detected, reference, requalified,
and what came of it.

**Every way it can fail lands on the reference.** A run that does not clear its own gates
leaves `REQUALIFICATION_REQUIRED` exactly where it was. A run that clears them and finds the
candidate no longer wins writes `REFERENCE_ONLY`. Only a complete, passing run restores the
candidate.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from .local_learner import (CANDIDATE_QUALIFIED, IntakeContext, LocalLearner, REFERENCE_ONLY,
                            VALID, evidence_from)
from .monitoring import (REQUALIFICATION_REQUIRED, default_action_code_digest,
                         requalification_path)
from .qmv_variant import QUALIFIED_ACTION_ID, QUALIFIED_GEOMETRY

SCHEMA = "ironmule.requalification.v1"

#: A requalification is a qualification. It is held to the shape that earned the original one,
#: not to a cheaper version of it: three independent sessions, each three blocks of three
#: children, one resident model per child.
SESSIONS = 3
BLOCKS = 3
REPEATS = 5
WARMUPS = 2
ARMS = ("reference", "candidate", "reference_aa")
CHILD_TIMEOUT_S = 1800.0
DRIFT_LIMIT = 0.25
AA_MAX_OFFSET = 0.05
AA_MAX_HALF_WIDTH = 0.05
WORKLOAD_CLASS = "single_short"
MAX_NEW_TOKENS = 32
#: The workload, fixed in shipped code so a requalification measures the same thing on every
#: machine and in every year. It is one `single_short` request, which is the class the action
#: was qualified for and the only class it may be restored for.
PROBE_PROMPT = ("In two sentences, explain what a quantised matrix-vector product is and why "
                "its speed depends on memory bandwidth.")

OUTCOMES = ("PASS", "NO_GAIN", "WORSE", "INVALID", "REFUSED", "NOT_READY")

PROTOCOL = {
    "schema": SCHEMA,
    "sessions": SESSIONS, "blocks_per_session": BLOCKS, "children_per_block": 3,
    "repeats": REPEATS, "warmups": WARMUPS,
    "arms": ("reference, candidate, and the reference again under another name as the A/A "
             "control"),
    "arm_order": "rotated by (session + block) modulo 3",
    "one_arm_per_process": "each child loads one model, runs one arm and exits",
    "statistic": ("per-block ratio against the reference child of the same block, median, "
                  "95 per cent bootstrap over 10000 resamples, per session"),
    "gates": ("byte identity for every admitted projection before a token is timed, token "
              "and stop-reason identity across arms, an A/A control, a 25 per cent drift "
              "gate over blocks, and B65 in full"),
    "no_historical_ratio_decides": ("the decision is made on this run's own intervals. B69, "
                                    "B75 and B76 are history and are not consulted"),
    "workload": f"one {WORKLOAD_CLASS} request, {MAX_NEW_TOKENS} new tokens, a fixed prompt",
}


class RequalificationRefused(RuntimeError):
    """The run may not start. The reference stays in charge and the reason is recorded."""


# --------------------------------------------------------------------------- preconditions


@dataclass(frozen=True)
class Preconditions:
    allowed: bool
    reason: str
    checked: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reason": self.reason, "checked": dict(self.checked)}


def check_preconditions(state_path: Path, context: IntakeContext,
                        learner: LocalLearner,
                        action_id: str = QUALIFIED_ACTION_ID) -> Preconditions:
    """Everything that has to be true before a single model is loaded."""
    from .activation import read_kill

    checked: dict[str, Any] = {"action_id": action_id}
    state_path = Path(state_path)

    required = requalification_path(state_path)
    checked["requalification_record"] = str(required)
    if not required.exists():
        return Preconditions(False, "nothing requires a requalification: this runs only after "
                                    "monitoring has taken the action away", checked)
    try:
        record = json.loads(required.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = {"reason": "the requalification record is unreadable"}
    checked["required_because"] = record.get("reason", "")

    killed = read_kill(state_path)
    checked["kill_record"] = killed
    if killed is not None:
        return Preconditions(False, "a kill record is present. A kill is a safety failure and "
                                    "is not cleared by a measurement", checked)

    if not context.identified:
        return Preconditions(False, "this installation does not identify its machine, model "
                                    "and libraries", checked)
    for name in ("hardware_fingerprint", "model_identity_sha256", "model_revision", "mlx",
                 "mlx_lm"):
        checked[name] = getattr(context, name)

    stored = learner.as_dict()
    checked["controller_digest"] = stored["digest"]
    for name in ("hardware_fingerprint", "model_identity_sha256", "model_revision", "mlx",
                 "mlx_lm"):
        if stored["context"].get(name) != getattr(context, name):
            return Preconditions(False, f"the stored controller was written for a different "
                                        f"{name}, so there is nothing here to requalify",
                                 checked)
    checked["action_code_digest"] = default_action_code_digest()
    known = {action for action, _workload in learner._recommendations}
    checked["known_actions"] = sorted(known)
    if action_id not in known:
        return Preconditions(False, f"the controller has never qualified {action_id!r}",
                             checked)
    return Preconditions(True, "the state requires a requalification and the machine still "
                               "matches what was qualified", checked)


def resources_available(minimum_free_percent: float = 25.0) -> tuple[bool, str, dict]:
    """Enough of the machine to run a comparison worth believing."""
    from .hw import installed_memory_bytes, memory_pressure_level, swap_used_bytes, vm_counters

    counters = vm_counters() or {}
    total_pages = (installed_memory_bytes() or 0) // 16384
    free_pages = (counters.get("free_count", 0) + counters.get("speculative_count", 0)
                  + counters.get("inactive_count", 0))
    free_percent = (100.0 * free_pages / total_pages) if total_pages else 0.0
    pressure = memory_pressure_level()
    probe = {"memory_free_percent": free_percent, "memory_pressure_level": pressure,
             "swap_used_bytes": swap_used_bytes(), "load_average": list(os.getloadavg())}
    if pressure != 1:
        return False, "macOS memory pressure is not normal", probe
    if free_percent < minimum_free_percent:
        return False, f"free memory is {free_percent:.1f} per cent, below {minimum_free_percent}", probe
    return True, "", probe


# --------------------------------------------------------------------------- the child


CHILD_SOURCE = r'''
import json, sys
sys.path.insert(0, {root!r})
from ironmule.requalification import child_main
print("@@" + json.dumps(child_main(json.loads(sys.argv[1])), sort_keys=True, allow_nan=False), flush=True)
'''


def child_main(spec: dict) -> dict:
    """One arm, one resident model, then exit. Shipped, because a product path must be."""
    import mlx.core as mx

    from .hw import memory_pressure_level, vm_counters
    from .plans import StrictOneShotPlan
    from .qmv_variant import install
    from .service import InteractiveMode, Request, Runtime
    from .tune import load_profile, resolve_local_model

    resolved = resolve_local_model(spec["model"])
    if load_profile(spec["model"], model_identity=resolved.identity) is None:
        raise RuntimeError("no confirmed profile for this model")

    mx.reset_peak_memory()
    load_started = time.perf_counter_ns()
    runtime = Runtime.load(spec["model"], mode=InteractiveMode())
    load_ns = time.perf_counter_ns() - load_started
    installed = None
    try:
        if spec["arm"] == "candidate":
            gate, installed = install(runtime.engine.model, resolved.identity,
                                      tuple(spec["geometry"]))
            gate.open("requalification")
            installed["admitted_paths"] = installed["admitted_paths"][:4]
            installed["declined_paths"] = installed["declined_paths"][:4]

        prompt = runtime.encode(spec["prompt"])

        def build():
            return [Request(prompt_ids=list(prompt), max_tokens=spec["max_tokens"],
                            plan=StrictOneShotPlan(), arrival_ms=0.0, objective="latency")]

        for _ in range(spec["warmups"]):
            runtime.serve(build())

        samples, tokens, stops = [], None, None
        for _ in range(spec["repeats"]):
            started = time.perf_counter_ns()
            results = runtime.serve(build())
            samples.append(time.perf_counter_ns() - started)
            seen = [list(map(int, result.tokens)) for result in results]
            stop = [result.stop_reason for result in results]
            if tokens is None:
                tokens, stops = seen, stop
            elif seen != tokens or stop != stops:
                raise RuntimeError("the arm is not deterministic within itself")
        snapshot = runtime.telemetry.snapshot()
        counters = vm_counters() or {}
        return {
            "pid": os.getpid(), "arm": spec["arm"],
            "wall_ns": statistics.median(samples), "samples_ns": samples,
            "tokens": tokens, "stop_reasons": stops,
            "fallbacks": snapshot["fallbacks"],
            "correctness_errors": snapshot.get("correctness_errors", 0),
            "installed": installed,
            "model_load_ns_outside_the_measurement": load_ns,
            "memory_pressure_level": memory_pressure_level(),
            "swapouts": counters.get("swapouts"),
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
    finally:
        runtime.close()


def _bootstrap(ratios, resamples: int = 10000, seed: int = 20260911) -> dict:
    generator = random.Random(seed)
    if not ratios:
        return {"median": None, "ci_low": None, "ci_high": None, "n": 0, "ratios": []}
    draws = sorted(median([generator.choice(ratios) for _ in ratios]) for _ in range(resamples))
    return {"median": median(ratios), "ci_low": draws[int(0.025 * resamples)],
            "ci_high": draws[int(0.975 * resamples) - 1], "n": len(ratios),
            "ratios": list(ratios)}


# --------------------------------------------------------------------------- the run


def run_comparison(model_id: str, *, sessions: int = SESSIONS, blocks: int = BLOCKS,
                   project_root: Path | None = None,
                   on_progress=None) -> dict:
    """The comparison itself. Fresh processes, rotated arms, and every gate that qualified it."""
    from .hw import installed_memory_bytes, memory_pressure_level, swap_used_bytes, vm_counters

    root = Path(project_root or Path(__file__).resolve().parents[1])
    total_pages = (installed_memory_bytes() or 0) // 16384
    memory_total = installed_memory_bytes() or 0

    def probe(label):
        counters = vm_counters() or {}
        free_pages = (counters.get("free_count", 0) + counters.get("speculative_count", 0)
                      + counters.get("inactive_count", 0))
        return {"label": label, "swap_used_bytes": swap_used_bytes(),
                "memory_pressure_level": memory_pressure_level(),
                "load_average": list(os.getloadavg()),
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
                "memory_free_percent": (100.0 * free_pages / total_pages) if total_pages else None}

    started = time.perf_counter()
    samples = [probe("start")]
    all_sessions, failures = [], []
    children_run = 0
    for session_index in range(sessions):
        session = {"session": session_index, "blocks": []}
        for block_index in range(blocks):
            shift = (session_index + block_index) % len(ARMS)
            order = ARMS[shift:] + ARMS[:shift]
            entry = {"block": block_index, "order": list(order), "children": {}}
            for arm in order:
                spec = {"model": model_id,
                        "arm": "reference" if arm == "reference_aa" else arm,
                        "geometry": list(QUALIFIED_GEOMETRY), "repeats": REPEATS,
                        "warmups": WARMUPS, "prompt": PROBE_PROMPT,
                        "max_tokens": MAX_NEW_TOKENS}
                process = subprocess.Popen(
                    [sys.executable, "-u", "-c",
                     CHILD_SOURCE.format(root=str(root)), json.dumps(spec)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    cwd=str(root),
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
                    failures.append({"session": session_index, "block": block_index,
                                     "arm": arm, "returncode": process.returncode,
                                     "stderr_tail": stderr[-3000:]})
                    break
                entry["children"][arm] = json.loads(marker)
                if on_progress:
                    on_progress(session_index, block_index, arm)
            session["blocks"].append(entry)
            if failures:
                break
        samples.append(probe(f"after_session_{session_index}"))
        all_sessions.append(session)
        if failures:
            break
    samples.append(probe("end"))

    per_session = []
    differences = []
    reference_tokens = None
    for session in all_sessions:
        complete = [b for b in session["blocks"] if set(b["children"]) == set(ARMS)]
        for entry in complete:
            base = entry["children"]["reference"]
            reference_tokens = reference_tokens or base["tokens"]
            for arm in ARMS:
                child = entry["children"][arm]
                if child["tokens"] != reference_tokens \
                        or child["stop_reasons"] != base["stop_reasons"]:
                    differences.append({"session": session["session"],
                                        "block": entry["block"], "arm": arm})
        ratios = {arm: [entry["children"][arm]["wall_ns"]
                        / entry["children"]["reference"]["wall_ns"] for entry in complete]
                  for arm in ("candidate", "reference_aa")}
        walls = [sum(entry["children"][arm]["wall_ns"] / 1e6 for arm in ARMS)
                 for entry in complete]
        wall_median = median(walls) if walls else None
        disturbed = ([i for i, w in enumerate(walls)
                      if abs(w / wall_median - 1.0) > DRIFT_LIMIT] if wall_median else [])
        candidate = _bootstrap(ratios["candidate"])
        aa = _bootstrap(ratios["reference_aa"])
        aa_half = ((aa["ci_high"] - aa["ci_low"]) / 2 if aa["ci_high"] is not None else None)
        per_session.append({
            "session": session["session"], "complete_blocks": len(complete),
            "candidate": candidate, "reference_aa": aa, "aa_half_width": aa_half,
            "aa_gate_passed": bool(aa_half is not None and aa_half <= AA_MAX_HALF_WIDTH
                                   and abs(aa["median"] - 1.0) <= AA_MAX_OFFSET),
            "disturbed_blocks": disturbed,
            "within_session_se": (statistics.stdev(ratios["candidate"])
                                  / (len(ratios["candidate"]) ** 0.5)
                                  if len(ratios["candidate"]) > 1 else None),
            "fallbacks": sum(entry["children"][arm]["fallbacks"]
                             for entry in complete for arm in ARMS),
            "correctness_errors": sum(entry["children"][arm]["correctness_errors"]
                                      for entry in complete for arm in ARMS),
        })

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

    return {
        "protocol": PROTOCOL,
        "sessions": per_session,
        "raw_sessions": all_sessions,
        "child_failures": failures,
        "children_run": children_run,
        "correctness_identical": not differences,
        "differences": differences,
        "resource_gate_passed": not gate,
        "resource_gate_reasons": gate,
        "resource_samples": samples,
        "peak_child_rss_bytes": peak_rss,
        "wall_seconds": time.perf_counter() - started,
    }


# --------------------------------------------------------------------------- the decision


def decide(comparison: dict) -> dict:
    """`PASS`, `NO_GAIN`, `WORSE` or `INVALID`, on this run's own numbers and no others."""
    blocked = []
    if comparison["child_failures"]:
        blocked.append("a child failed")
    if not comparison["correctness_identical"]:
        blocked.append("tokens or stop reasons differed across arms")
    if not comparison["resource_gate_passed"]:
        blocked.extend(comparison["resource_gate_reasons"])
    sessions = comparison["sessions"]
    if len(sessions) < SESSIONS or any(row["complete_blocks"] < BLOCKS for row in sessions):
        blocked.append("not every preregistered session and block completed")
    if any(row["fallbacks"] for row in sessions):
        blocked.append("a fallback was recorded")
    if any(row["correctness_errors"] for row in sessions):
        blocked.append("a correctness error was recorded")
    if any(row["disturbed_blocks"] for row in sessions):
        blocked.append("a block deviated more than the drift gate allows")
    if blocked:
        return {"outcome": "INVALID", "reason": "; ".join(blocked),
                "requalification_stays_required": True}

    readable = [row for row in sessions if row["aa_gate_passed"]]
    if len(readable) < SESSIONS:
        return {"outcome": "INVALID",
                "reason": (f"only {len(readable)} of {len(sessions)} sessions carried an A/A "
                           f"control narrow enough to read"),
                "requalification_stays_required": True}

    highs = [row["candidate"]["ci_high"] for row in readable]
    lows = [row["candidate"]["ci_low"] for row in readable]
    if all(high is not None and high < 1.0 for high in highs):
        return {"outcome": "PASS",
                "reason": ("every session's 95 per cent interval lies entirely below 1.0, "
                           "with an A/A control that cleared its gate"),
                "requalification_stays_required": False}
    if all(low is not None and low > 1.0 for low in lows):
        return {"outcome": "WORSE",
                "reason": "the candidate is measurably slower than the reference here now",
                "requalification_stays_required": False}
    return {"outcome": "NO_GAIN",
            "reason": "at least one interval contains 1.0, so nothing is decided in favour "
                      "of the candidate",
            "requalification_stays_required": False}


# --------------------------------------------------------------------------- the transition


def _epoch_name(state_path: Path, stamp: str) -> Path:
    return Path(state_path).with_name(f"{Path(state_path).stem}.{stamp}.epoch.json")


def lineage_path(state_path: Path) -> Path:
    return Path(state_path).with_name(f"{Path(state_path).stem}_lineage.json")


def append_lineage(state_path: Path, event: Mapping[str, Any]) -> Path:
    """History, appended and never rewritten, so the whole story stays readable."""
    path = lineage_path(state_path)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        events = existing.get("events") if isinstance(existing, Mapping) else None
    except (OSError, ValueError):
        events = None
    events = list(events or [])
    events.append({"at": datetime.now(timezone.utc).isoformat(), **dict(event)})
    payload = {"schema": SCHEMA, "events": events}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def apply_outcome(state_path: Path, context: IntakeContext, learner: LocalLearner,
                  comparison: dict, decision: dict,
                  action_id: str = QUALIFIED_ACTION_ID) -> dict:
    """Archive the old epoch, write the new one, and move the state exactly one way."""
    state_path = Path(state_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = None
    if state_path.exists():
        archived = _epoch_name(state_path, stamp)
        archived.write_bytes(state_path.read_bytes())

    outcome = decision["outcome"]
    if outcome == "INVALID":
        append_lineage(state_path, {"event": "requalification_invalid",
                                    "reason": decision["reason"],
                                    "archived_epoch": str(archived) if archived else None})
        return {"applied": False, "outcome": outcome, "state": REQUALIFICATION_REQUIRED,
                "archived_epoch": str(archived) if archived else None,
                "reason": "the run did not clear its own gates, so nothing it measured is "
                          "evidence and the requalification stays required"}

    fresh = LocalLearner(context)
    evidence_ids = []
    if outcome == "PASS":
        for row in comparison["sessions"]:
            evidence_id = f"B81_requalification_{stamp}_session_{row['session']:02d}"
            evidence_ids.append(evidence_id)
            fresh.observe(evidence_from({
                "evidence_id": evidence_id, "action_id": action_id,
                "workload_class": WORKLOAD_CLASS, "reference_stack": "A",
                "hardware_fingerprint": context.hardware_fingerprint,
                "gpu_architecture": context.gpu_architecture, "mlx": context.mlx,
                "mlx_lm": context.mlx_lm, "model_id": context.model_id,
                "model_identity_sha256": context.model_identity_sha256,
                "model_revision": context.model_revision,
                "ratio": row["candidate"]["median"],
                "ci_low": row["candidate"]["ci_low"],
                "ci_high": row["candidate"]["ci_high"],
                "within_session_se": row["within_session_se"],
                "aa_median": row["reference_aa"]["median"],
                "aa_half_width": row["aa_half_width"],
                "correctness_passed": True, "resource_gate_passed": True,
                "status": VALID,
                "measured_at": datetime.now(timezone.utc).isoformat()}))
        fresh.save(state_path)
        state = fresh.recommendation(action_id, WORKLOAD_CLASS).local_learning_state
    else:
        # The run was valid and the candidate did not win. The controller is left with no
        # evidence for it at all, which is REFERENCE_ONLY by construction rather than by
        # a flag someone has to remember to set.
        fresh.save(state_path)
        state = REFERENCE_ONLY

    required = requalification_path(state_path)
    cleared = required.exists()
    if cleared:
        required.unlink()
    # The monitor's own baselines described the machine before the run; they are archived
    # with the epoch so the next baseline is built from what the machine is now.
    from .monitoring import requalification_path as _requalification_path  # noqa: F401
    monitor_state = state_path.with_name(f"{state_path.stem}_monitoring.json")
    archived_monitor = None
    if monitor_state.exists():
        archived_monitor = monitor_state.with_name(
            f"{monitor_state.stem}.{stamp}.epoch.json")
        monitor_state.rename(archived_monitor)

    append_lineage(state_path, {
        "event": f"requalification_{outcome.lower()}",
        "reason": decision["reason"], "outcome": outcome,
        "resulting_state": state,
        "archived_epoch": str(archived) if archived else None,
        "archived_monitor_state": str(archived_monitor) if archived_monitor else None,
        "new_evidence_ids": evidence_ids,
        "requalification_record_cleared": cleared,
        "session_ratios": [row["candidate"]["median"] for row in comparison["sessions"]]})
    return {"applied": True, "outcome": outcome, "state": state,
            "archived_epoch": str(archived) if archived else None,
            "archived_monitor_state": str(archived_monitor) if archived_monitor else None,
            "new_evidence_ids": evidence_ids,
            "requalification_record_cleared": cleared,
            "reason": decision["reason"]}


# --------------------------------------------------------------------------- the command


def requalify(model_id: str | None = None, *, state_path: Path | None = None,
              sessions: int = SESSIONS, blocks: int = BLOCKS,
              readiness: "bool | Mapping[str, Any]" = True,
              on_progress=None) -> dict:
    """`ironmule requalify`. Refuses unless the state requires it and the machine matches.

    `readiness` (`B82`): `True` runs the cheap reference-against-reference probe first and
    stops at `NOT_READY` rather than spending a comparison on a machine whose own control
    cannot be read. A mapping is an already-completed probe, which is how a caller that ran
    one itself avoids paying for a second. `False` skips it, which is `B81`'s behaviour.
    """
    import mlx.core as mx
    import mlx_lm

    from .hw import fingerprint, static_facts
    from .local_learner import default_state_path
    from .tune import DEFAULT_MODEL, gpu_busy, resolve_local_model

    model_id = model_id or DEFAULT_MODEL
    state_path = Path(state_path or default_state_path())
    resolved = resolve_local_model(model_id)
    context = IntakeContext(
        hardware_fingerprint=fingerprint(),
        gpu_architecture=str(mx.device_info().get("architecture", "")),
        mlx=mx.__version__, mlx_lm=mlx_lm.__version__, model_id=model_id,
        model_identity_sha256=resolved.identity.identity_sha256,
        model_revision=getattr(resolved.identity, "revision", "") or "")
    learner = LocalLearner.restore(state_path, context)

    preconditions = check_preconditions(state_path, context, learner)
    started_at = datetime.now(timezone.utc).isoformat()
    if not preconditions.allowed:
        append_lineage(state_path, {"event": "requalification_refused",
                                    "reason": preconditions.reason})
        return {"schema": SCHEMA, "outcome": "REFUSED", "started_at": started_at,
                "preconditions": preconditions.as_dict(),
                "state": REQUALIFICATION_REQUIRED,
                "reason": preconditions.reason, "applied": False}

    busy = gpu_busy()
    if busy:
        reason = f"another model process is running ({busy})"
        append_lineage(state_path, {"event": "requalification_refused", "reason": reason})
        return {"schema": SCHEMA, "outcome": "REFUSED", "started_at": started_at,
                "preconditions": preconditions.as_dict(),
                "state": REQUALIFICATION_REQUIRED, "reason": reason, "applied": False}

    enough, why, probe = resources_available()
    if not enough:
        append_lineage(state_path, {"event": "requalification_refused", "reason": why})
        return {"schema": SCHEMA, "outcome": "REFUSED", "started_at": started_at,
                "preconditions": preconditions.as_dict(), "resources": probe,
                "state": REQUALIFICATION_REQUIRED, "reason": why, "applied": False}

    readiness_record = None
    if readiness is not False:
        if isinstance(readiness, Mapping):
            readiness_record = dict(readiness)
        else:
            from .readiness import probe as readiness_probe

            outcome = readiness_probe(model_id, on_progress=None)
            readiness_record = outcome.as_dict()
        if not readiness_record.get("ready"):
            reason = readiness_record.get("reason", "the machine is not steady enough to measure")
            append_lineage(state_path, {"event": "requalification_not_ready",
                                        "reason": reason})
            return {"schema": SCHEMA, "outcome": "NOT_READY", "started_at": started_at,
                    "preconditions": preconditions.as_dict(), "resources": probe,
                    "readiness": readiness_record,
                    "state": REQUALIFICATION_REQUIRED, "reason": reason, "applied": False,
                    "note": ("no comparison was started. The reference keeps serving, the "
                             "requalification stays required, and nothing retries here")}

    append_lineage(state_path, {"event": "requalification_started",
                                "model_id": model_id, "sessions": sessions,
                                "blocks_per_session": blocks,
                                "readiness_checked": readiness is not False})
    digest_before = default_action_code_digest()
    comparison = run_comparison(model_id, sessions=sessions, blocks=blocks,
                                on_progress=on_progress)
    digest_after = default_action_code_digest()
    comparison["action_code_digest_before"] = digest_before
    comparison["action_code_digest_after"] = digest_after
    if digest_before != digest_after:
        # The action's own source changed while it was being measured. Whatever the numbers
        # say, they do not describe one thing.
        decision = {"outcome": "INVALID",
                    "reason": ("the action's code digest changed during the run, so the "
                               "measurement does not describe a single action"),
                    "requalification_stays_required": True}
    else:
        decision = decide(comparison)
    applied = apply_outcome(state_path, context, learner, comparison, decision)
    return {
        "schema": SCHEMA, "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id, "state_path": str(state_path),
        "preconditions": preconditions.as_dict(), "resources": probe,
        "readiness": readiness_record,
        "protocol": PROTOCOL, "comparison": comparison, "decision": decision,
        "applied": applied, "outcome": decision["outcome"], "state": applied["state"],
        "cost": {"wall_seconds": comparison["wall_seconds"],
                 "model_loads": comparison["children_run"],
                 "comparative_requests": comparison["children_run"] * (REPEATS + WARMUPS),
                 "note": ("the goal is a requalification worth believing, not a cheap one. "
                          "Every child loads one model and runs one arm")},
        "lineage": str(lineage_path(state_path)),
    }
