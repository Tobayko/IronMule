"""Paired A/B across fresh processes.

One process per pair, arm order alternating, so a machine that drifts warmer or
busier during the run cannot hand the win to whichever arm ran second. Each child
loads its own model per arm, because some knobs mutate the model in place and a
reused model would carry one arm's surgery into the next.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import replace
from typing import Any, Mapping

from .bench import interleave, paired_ratio, summarise
from .runtime import Knobs

CHILD_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1"}
MAX_CHILD_OUTPUT = 512 * 1024
CHILD_BOOTSTRAP = (
    "import importlib.util,json,os,sys;"
    "guard_path=os.path.realpath(sys.argv[2]);"
    "ab_path=os.path.realpath(sys.argv[3]);"
    "guard_spec=importlib.util.spec_from_file_location('ironmule.q3f_child_guard',guard_path);"
    "assert guard_spec is not None and guard_spec.loader is not None;"
    "guard=importlib.util.module_from_spec(guard_spec);"
    "sys.modules['ironmule.q3f_child_guard']=guard;"
    "guard_spec.loader.exec_module(guard);"
    "guard.assert_source_surface(ab_path);"
    "guard.install();"
    "from ironmule.ab import _child\n"
    "def _emit_guard_failure(exc):\n"
    "    for note in getattr(exc,'__notes__',[]):\n"
    "        if isinstance(note,str) and note.startswith('@GUARD_FAILURE'):\n"
    "            print(note,flush=True); break\n"
    "try:\n"
    "    result=_child(json.loads(sys.argv[1]))\n"
    "except BaseException as exc:\n"
    "    _emit_guard_failure(exc); raise\n"
    "else:\n"
    "    print('@@'+json.dumps(result,sort_keys=True,allow_nan=False),flush=True)\n"
    "finally:\n"
    "    guard.uninstall()"
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


class ABRunError(RuntimeError):
    """A bounded child failure carrying only already-completed raw records."""

    def __init__(self, message: str, *, partial_children=None, child_index=None,
                 partial_evidence=None):
        super().__init__(message)
        self.partial_children = list(partial_children or [])
        self.child_index = child_index
        self.partial_evidence = partial_evidence


def _terminate_child(process: subprocess.Popen[str]) -> None:
    """Terminate and reap one direct child, failing if cleanup cannot be proven."""
    errors = []

    def kill_and_reap() -> None:
        """Escalate only while the child is still demonstrably alive."""
        if process.poll() is not None:
            return
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(f"kill: {type(exc).__name__}")
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            errors.append("child did not exit after kill")
        except OSError as exc:
            errors.append(f"wait: {type(exc).__name__}")

    try:
        process.terminate()
    except ProcessLookupError:
        pass
    except OSError as exc:
        errors.append(f"terminate: {type(exc).__name__}")
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        kill_and_reap()
    except OSError as exc:
        errors.append(f"wait: {type(exc).__name__}")
        kill_and_reap()
    try:
        process.communicate(timeout=2)
    except (subprocess.TimeoutExpired, OSError) as exc:
        errors.append(f"communicate: {type(exc).__name__}")
    if process.poll() is None:
        errors.append("child did not exit after termination")
    if errors:
        raise RuntimeError("child cleanup failed: " + "; ".join(errors))


def _child(spec: dict[str, Any]) -> dict[str, Any]:
    """Run the model child behind Q3f's bounded no-detach guard."""
    from . import q3f_child_guard

    owns_guard = not q3f_child_guard.is_installed()
    if owns_guard:
        q3f_child_guard.install()
    try:
        q3f_child_guard.assert_child_surface(_child)
        return _child_execution(spec)
    except BaseException as exc:
        try:
            marker = q3f_child_guard.failure_marker()
            if marker is not None:
                exc.add_note("@GUARD_FAILURE" + json.dumps(marker, sort_keys=True, allow_nan=False))
        except BaseException:
            pass
        raise
    finally:
        if owns_guard:
            q3f_child_guard.uninstall()


def _child_execution(spec: dict[str, Any]) -> dict[str, Any]:
    """Guarded execution closure: one fresh model per arm, in the given order."""
    # This must be the first child operation.  The guard is stdlib-only and is
    # installed before importing the model/runtime surface, so a Python-level
    # process/session escape cannot turn into an unattributed cleanup record.
    from .tune import DEFAULT_MODEL, DEFAULT_PROMPT, _eos_ids, load_engine, prompt_ids

    import mlx.core as mx

    out: dict[str, Any] = {"pid": os.getpid(), "arms": {}, "order": spec["order"]}
    for name in spec["order"]:
        knobs = Knobs(**spec["arms"][name])
        # MLX's peak is a high-water mark for the whole process. Every arm here loads
        # its own model into the same process, so without a reset the second arm
        # inherits the first one's peak and the number stops being about this arm.
        mx.reset_peak_memory()
        engine = None
        try:
            engine, tok = load_engine(spec.get("model", DEFAULT_MODEL), knobs)
            ids = prompt_ids(tok, spec.get("prompt", DEFAULT_PROMPT))
            eos = _eos_ids(tok)
            for _ in range(spec["warmup"]):
                engine.generate(ids, spec["max_tokens"], eos)
            runs = [engine.generate(ids, spec["max_tokens"], eos) for _ in range(spec["repeats"])]
            logical_per_repeat = [list(map(int, run["logical_tokens"])) for run in runs]
            physical_per_repeat = [list(map(int, run["physical_tokens"])) for run in runs]
            stop_reasons = [
                "eos" if logical and logical[-1] in eos else "length"
                for logical in logical_per_repeat
            ]
            token_counts = [
                {"logical": len(logical), "physical": len(physical)}
                for logical, physical in zip(logical_per_repeat, physical_per_repeat)
            ]
            capacities = [int(run["capacity"]) for run in runs]
            out["arms"][name] = {
                "total_ns": [r["total_ns"] for r in runs],
                "prefill_ns": [r["prefill_ns"] for r in runs],
                "decode_ns": [r["decode_ns"] for r in runs],
                "logical_tokens": logical_per_repeat[0],
                "logical_tokens_per_repeat": logical_per_repeat,
                "physical_tokens_per_repeat": physical_per_repeat,
                "token_counts": token_counts,
                "stop_reasons": stop_reasons,
                "capacities": capacities,
                "deterministic": all(
                    logical == logical_per_repeat[0]
                    and physical == physical_per_repeat[0]
                    and counts == token_counts[0]
                    and stop == stop_reasons[0]
                    and capacity == capacities[0]
                    for logical, physical, counts, stop, capacity in zip(
                        logical_per_repeat, physical_per_repeat, token_counts,
                        stop_reasons, capacities,
                    )
                ),
                "decode_steps": len(physical_per_repeat[0]) - 1,
                "prompt_tokens": len(ids),
                "mlx_peak_bytes": mx.get_peak_memory(),
            }
        except BaseException as primary:
            close = getattr(engine, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException as cleanup_error:
                    primary.add_note(
                        "A/B engine cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            engine = None
            raise
        finally:
            if engine is not None:
                close = getattr(engine, "close", None)
                if close is not None:
                    close()
                engine = None
    # Kept for existing readers, and still what it always was: the high-water mark
    # across every arm this process ran, not any single arm's peak.
    out["mlx_peak_bytes"] = max(arm["mlx_peak_bytes"] for arm in out["arms"].values())
    # Capture only after all work has completed.  Any guard event or malformed
    # ledger raises and therefore cannot be reported as a successful child.
    from . import q3f_child_guard
    out["guard"] = q3f_child_guard.ledger()
    return out


def _child_record_complete(child: Any, names: list[str], order: list[str]) -> bool:
    """Reject a partial marker before any aggregation/indexing can raise."""
    return _validate_child_record(child, names, order, repeats=None) is None


_CHILD_FIELDS = frozenset({
    "total_ns", "prefill_ns", "decode_ns", "logical_tokens",
    "logical_tokens_per_repeat", "physical_tokens_per_repeat", "token_counts",
    "stop_reasons", "capacities", "deterministic", "decode_steps",
    "prompt_tokens", "mlx_peak_bytes",
})
_GUARD_FIELDS = frozenset({"version", "installed", "events"})
_GUARD_EVENT_FIELDS = frozenset({"event", "operation", "monotonic", "blocked"})
_GUARD_OPERATIONS = frozenset({
    "subprocess.Popen", "os.system", "os.fork", "os.forkpty",
    "os.posix_spawn", "os.posix_spawnp", "os.setsid", "os.setpgid",
})


def _valid_child_guard(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != _GUARD_FIELDS:
        return False
    events = value["events"]
    if (value["version"] != "ironmule.q3f_child_guard.v1"
            or value["installed"] is not True
            or not isinstance(events, list) or len(events) > 32 or events != []):
        return False
    for event in events:
        if (not isinstance(event, dict) or set(event) != _GUARD_EVENT_FIELDS
                or event["event"] not in _GUARD_OPERATIONS
                or event["operation"] not in _GUARD_OPERATIONS
                or event["event"] != event["operation"]
                or not isinstance(event["monotonic"], (int, float))
                or isinstance(event["monotonic"], bool)
                or not math.isfinite(float(event["monotonic"]))
                or event["monotonic"] < 0 or event["blocked"] is not True):
            return False
        try:
            if len(json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) > 512:
                return False
        except (TypeError, ValueError, OverflowError):
            return False
    try:
        return len(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) <= 512 * 33
    except (TypeError, ValueError, OverflowError):
        return False
_SUMMARY_FIELDS = frozenset({"n", "median", "min", "max", "p95", "stdev"})
_RATIO_FIELDS = frozenset({"median_ratio", "ci_low", "ci_high", "pairs"})


def _positive_finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (OverflowError, ValueError):
        return False


def _integer_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in value
    )


def _validate_child_record(child: Any, names: list[str], order: list[str],
                           repeats: int | None) -> str | None:
    if not isinstance(child, dict):
        return "child_not_object"
    if set(child) != {"pid", "arms", "order", "mlx_peak_bytes", "guard"}:
        return "child_fields"
    if (not isinstance(child["pid"], int) or isinstance(child["pid"], bool)
            or child["pid"] <= 0 or child["order"] != order
            or not isinstance(child["arms"], dict)
            or set(child["arms"]) != set(names)
            or not isinstance(child["mlx_peak_bytes"], int)
            or isinstance(child["mlx_peak_bytes"], bool)
            or child["mlx_peak_bytes"] < 0
            or not _valid_child_guard(child["guard"])):
        return "child_identity"
    for name in names:
        arm = child["arms"][name]
        if not isinstance(arm, dict) or set(arm) != _CHILD_FIELDS:
            return f"arm_fields:{name}"
        if not all(isinstance(arm[field], list) for field in (
                "total_ns", "prefill_ns", "decode_ns",
                "logical_tokens_per_repeat", "physical_tokens_per_repeat",
                "token_counts", "stop_reasons", "capacities")):
            return f"arm_arrays:{name}"
        count = repeats if repeats is not None else len(arm["total_ns"])
        if count < 1:
            return f"arm_arrays:{name}"
        for field in ("total_ns", "prefill_ns", "decode_ns"):
            if len(arm[field]) != count or any(not _positive_finite(item) for item in arm[field]):
                return f"timing:{name}:{field}"
        logical = arm["logical_tokens_per_repeat"]
        physical = arm["physical_tokens_per_repeat"]
        if (len(logical) != count or len(physical) != count
                or any(not _integer_list(item) for item in logical)
                or any(not _integer_list(item) for item in physical)):
            return f"tokens:{name}"
        counts = arm["token_counts"]
        if (len(counts) != count
                or any(not isinstance(item, dict) or set(item) != {"logical", "physical"}
                       or not isinstance(item["logical"], int) or isinstance(item["logical"], bool)
                       or item["logical"] < 0 or not isinstance(item["physical"], int)
                       or isinstance(item["physical"], bool) or item["physical"] < 0
                       for item in counts)
                or any(item != {"logical": len(logical[index]), "physical": len(physical[index])}
                       for index, item in enumerate(counts))):
            return f"counts:{name}"
        stops = arm["stop_reasons"]
        if len(stops) != count or any(item not in {"eos", "length"} for item in stops):
            return f"stops:{name}"
        capacities = arm["capacities"]
        if (len(capacities) != count
                or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0
                       for item in capacities)):
            return f"capacities:{name}"
        if (not isinstance(arm["logical_tokens"], list)
                or not isinstance(arm["deterministic"], bool)
                or not isinstance(arm["decode_steps"], int)
                or isinstance(arm["decode_steps"], bool) or arm["decode_steps"] < 0
                or not isinstance(arm["prompt_tokens"], int)
                or isinstance(arm["prompt_tokens"], bool) or arm["prompt_tokens"] < 0
                or not isinstance(arm["mlx_peak_bytes"], int)
                or isinstance(arm["mlx_peak_bytes"], bool) or arm["mlx_peak_bytes"] < 0):
            return f"arm_scalars:{name}"
        if arm["logical_tokens"] != logical[0] or arm["decode_steps"] != len(physical[0]) - 1:
            return f"arm_reference:{name}"
        expected_deterministic = all(
            logical_item == logical[0] and physical_item == physical[0]
            and counts_item == counts[0] and stop_item == stops[0]
            and capacity_item == capacities[0]
            for logical_item, physical_item, counts_item, stop_item, capacity_item in zip(
                logical, physical, counts, stops, capacities
            )
        )
        if arm["deterministic"] != expected_deterministic:
            return f"determinism:{name}"
    return None


def validate_result(result: Any, *, processes: int, repeats: int, warmup: int,
                    expected_arms: Mapping[str, Knobs | Mapping[str, Any]],
                    baseline: str = "baseline",
                    candidate: str = "candidate") -> tuple[bool, str | None]:
    """Strictly validate one complete result and its exact measured knob arms."""
    expected_top = {"arms", "processes", "repeats", "warmup", "raw", "per_arm",
                    "token_identity", "token_count_identity", "stop_reason_identity",
                    "deterministic", "reference_tokens", "ratios"}
    if not isinstance(result, dict) or set(result) != expected_top:
        return False, "top_fields"
    if (result["processes"] != processes or result["repeats"] != repeats
            or result["warmup"] != warmup or not isinstance(result["arms"], dict)
            or not isinstance(expected_arms, Mapping)
            or set(expected_arms) != {baseline, candidate}
            or set(result["arms"]) != {baseline, candidate}):
        return False, "top_types"
    expected_serialized = {}
    for name, knobs in expected_arms.items():
        if isinstance(knobs, Knobs):
            expected_serialized[name] = knobs.as_dict()
        elif isinstance(knobs, Mapping):
            expected_serialized[name] = dict(knobs)
            if set(expected_serialized[name]) != set(Knobs().as_dict()):
                return False, "expected_arms"
        else:
            return False, "expected_arms"
    if result["arms"] != expected_serialized:
        return False, "arms"
    if (not isinstance(result["raw"], list) or len(result["raw"]) != processes
            or not isinstance(result["per_arm"], dict)
            or set(result["per_arm"]) != {baseline, candidate}
            or not isinstance(result["reference_tokens"], list)
            or any(not isinstance(item, int) or isinstance(item, bool) for item in result["reference_tokens"])
            or any(not isinstance(result[field], bool) for field in (
                "token_identity", "token_count_identity", "stop_reason_identity", "deterministic"))):
        return False, "top_types"
    pids = set()
    orders = []
    for index, child in enumerate(result["raw"]):
        expected_order = [baseline, candidate] if index % 2 == 0 else [candidate, baseline]
        reason = _validate_child_record(child, [baseline, candidate], expected_order, repeats)
        if reason is not None:
            return False, reason
        if child["pid"] in pids:
            return False, "duplicate_pid"
        pids.add(child["pid"])
        orders.append(expected_order)
    for name in (baseline, candidate):
        summary = result["per_arm"][name]
        if not isinstance(summary, dict) or set(summary) != {"total_ns", "prefill_ns", "decode_ns"}:
            return False, f"summary_metrics:{name}"
        for metric in ("total_ns", "prefill_ns", "decode_ns"):
            values = [statistics.median(child["arms"][name][metric]) for child in result["raw"]]
            if (not isinstance(summary[metric], dict)
                    or set(summary[metric]) != _SUMMARY_FIELDS
                    or summary[metric] != summarise(values)):
                return False, f"summary:{name}:{metric}"
    expected_counts = {name: [child["arms"][name]["token_counts"] for child in result["raw"]]
                       for name in (baseline, candidate)}
    expected_stops = {name: [child["arms"][name]["stop_reasons"] for child in result["raw"]]
                      for name in (baseline, candidate)}
    expected_tokens = {name: [child["arms"][name]["logical_tokens"] for child in result["raw"]]
                       for name in (baseline, candidate)}
    if result["reference_tokens"] != expected_tokens[baseline][0]:
        return False, "reference_tokens"
    if result["token_identity"] != all(seq == expected_tokens[baseline][0]
                                        for values in expected_tokens.values() for seq in values):
        return False, "token_identity"
    if result["token_count_identity"] != all(seq == expected_counts[baseline][0]
                                              for values in expected_counts.values() for seq in values):
        return False, "token_count_identity"
    if result["stop_reason_identity"] != all(seq == expected_stops[baseline][0]
                                               for values in expected_stops.values() for seq in values):
        return False, "stop_reason_identity"
    expected_deterministic = all(child["arms"][name]["deterministic"]
                                 for child in result["raw"] for name in (baseline, candidate))
    if result["deterministic"] != expected_deterministic:
        return False, "deterministic"
    if set(result["ratios"]) != {f"{candidate}/{baseline}"}:
        return False, "ratio_pairs"
    for metric in ("total_ns", "prefill_ns", "decode_ns"):
        expected = paired_ratio(
            [statistics.median(child["arms"][candidate][metric]) for child in result["raw"]],
            [statistics.median(child["arms"][baseline][metric]) for child in result["raw"]],
        )
        actual = result["ratios"][f"{candidate}/{baseline}"].get(metric)
        if actual != expected:
            return False, f"ratio:{metric}"
    return True, None


def run(arms: dict[str, Knobs], processes: int = 6, repeats: int = 7, warmup: int = 2,
        max_tokens: int = 32, model: str | None = None, prompt: str | None = None,
        *, child_timeout_seconds: float | None = None,
        before_child=None, on_child=None, on_child_start=None) -> dict[str, Any]:
    """Spawn children, collect raw samples, and expose bounded progress hooks.

    ``before_child(index, order)`` runs immediately before each child.  After a
    successful child, ``on_child(index, child_record)`` receives a defensive,
    JSON-safe copy of that record.  ``on_child_start(index, pid, order)`` runs
    immediately after the child is spawned so a caller can retain lifecycle
    evidence before any model work.  Timeout errors identify only the child
    index and arm order; command arguments are intentionally excluded.
    """
    if child_timeout_seconds is not None:
        try:
            finite_timeout = math.isfinite(float(child_timeout_seconds))
        except (TypeError, ValueError, OverflowError):
            finite_timeout = False
        if (isinstance(child_timeout_seconds, bool)
                or not isinstance(child_timeout_seconds, (int, float))
                or not finite_timeout
                or child_timeout_seconds <= 0):
            raise ValueError("child_timeout_seconds must be finite and positive")
    from .tune import gpu_busy
    busy = gpu_busy()
    if busy:
        raise RuntimeError(f"another model process is running, refusing to measure ({busy})")

    names = list(arms)
    orders = interleave(names, processes)
    spec_base = {"arms": {n: k.as_dict() for n, k in arms.items()}, "repeats": repeats,
                 "warmup": warmup, "max_tokens": max_tokens}
    if model:
        spec_base["model"] = model
    if prompt is not None:
        spec_base["prompt"] = prompt

    children = []
    for index, order in enumerate(orders):
        if before_child is not None:
            before_child(index, list(order))
        spec = dict(spec_base, order=order)
        timeout = child_timeout_seconds
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c",
                 CHILD_BOOTSTRAP, json.dumps(spec),
                 str(os.path.join(os.path.dirname(os.path.abspath(__file__)), "q3f_child_guard.py")),
                 os.path.abspath(__file__)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                env={**os.environ, **CHILD_ENV},
            )
        except OSError as exc:
            raise ABRunError(
                f"child {index} could not start", partial_children=children,
                child_index=index,
            ) from exc
        if on_child_start is not None:
            try:
                on_child_start(index, proc.pid, list(order))
            except BaseException as exc:
                try:
                    _terminate_child(proc)
                except BaseException as cleanup_exc:
                    exc.add_note(f"child-start callback cleanup failed: {cleanup_exc}")
                raise ABRunError(
                    f"child {index} start callback failed", partial_children=children,
                    child_index=index,
                ) from exc
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                _terminate_child(proc)
            except RuntimeError as cleanup_error:
                raise ABRunError(
                    f"child {index} timed out and cleanup failed",
                    partial_children=children, child_index=index,
                ) from cleanup_error
            raise ABRunError(
                f"child {index} timed out", partial_children=children,
                child_index=index,
            ) from None
        except OSError as exc:
            try:
                _terminate_child(proc)
            except RuntimeError as cleanup_error:
                raise ABRunError(
                    f"child {index} communication failed and cleanup failed",
                    partial_children=children, child_index=index,
                ) from cleanup_error
            raise ABRunError(
                f"child {index} communication failed",
                partial_children=children, child_index=index,
            ) from exc
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            raise ABRunError(
                f"child {index} produced malformed output",
                partial_children=children, child_index=index,
            )
        if len(stdout) > MAX_CHILD_OUTPUT or len(stderr) > MAX_CHILD_OUTPUT:
            try:
                _terminate_child(proc)
            except RuntimeError as cleanup_error:
                raise ABRunError(
                    f"child {index} output limit exceeded and cleanup failed",
                    partial_children=children, child_index=index,
                ) from cleanup_error
            raise ABRunError(
                f"child {index} output limit exceeded",
                partial_children=children, child_index=index,
            )
        if proc.returncode != 0:
            guard_failure = next((line[len("@GUARD_FAILURE"):].strip()
                                  for line in stdout.splitlines()
                                  if line.startswith("@GUARD_FAILURE")), None)
            evidence = None
            if isinstance(guard_failure, str):
                try:
                    evidence = {"guard_failure": json.loads(
                        guard_failure, parse_constant=_reject_json_constant)}
                except (TypeError, ValueError, json.JSONDecodeError):
                    evidence = {"guard_failure": {"malformed": True}}
            raise ABRunError(
                f"child {index} exited with status {proc.returncode}",
                partial_children=children, child_index=index,
                partial_evidence=evidence,
            )
        line = next((l for l in stdout.splitlines() if l.startswith("@@")), None)
        if line is None:
            raise ABRunError(
                f"child {index} produced no result marker",
                partial_children=children, child_index=index,
            )
        try:
            child = json.loads(line[2:], parse_constant=_reject_json_constant)
        except (TypeError, ValueError) as exc:
            raise ABRunError(
                f"child {index} produced invalid JSON",
                partial_children=children, child_index=index,
            ) from exc
        if not _child_record_complete(child, names, list(order)):
            raise ABRunError(
                f"child {index} produced an incomplete result",
                partial_children=children, child_index=index,
            )
        children.append(child)
        if on_child is not None:
            safe_copy = json.loads(json.dumps(child, sort_keys=True, allow_nan=False))
            on_child(index, safe_copy)

    per_arm: dict[str, dict[str, list[float]]] = {
        n: {"total_ns": [], "prefill_ns": [], "decode_ns": []} for n in names}
    tokens: dict[str, list[list[int]]] = {n: [] for n in names}
    token_counts: dict[str, list[list[dict[str, int]]]] = {n: [] for n in names}
    stop_reasons: dict[str, list[list[str]]] = {n: [] for n in names}
    for child in children:
        for name in names:
            arm = child["arms"][name]
            for metric in ("total_ns", "prefill_ns", "decode_ns"):
                per_arm[name][metric].append(statistics.median(arm[metric]))
            tokens[name].append(arm["logical_tokens"])
            token_counts[name].append(arm["token_counts"])
            stop_reasons[name].append(arm["stop_reasons"])

    reference = tokens[names[0]][0]
    identical = all(seq == reference for name in names for seq in tokens[name])
    deterministic = all(child["arms"][n]["deterministic"] for child in children for n in names)
    reference_counts = token_counts[names[0]][0]
    count_identity = all(counts == reference_counts for name in names for counts in token_counts[name])
    reference_stops = stop_reasons[names[0]][0]
    stop_identity = all(stops == reference_stops for name in names for stops in stop_reasons[name])

    result: dict[str, Any] = {
        "arms": {n: k.as_dict() for n, k in arms.items()},
        "processes": processes, "repeats": repeats, "warmup": warmup,
        "raw": children,
        "per_arm": {n: {m: summarise(v) for m, v in metrics.items()} for n, metrics in per_arm.items()},
        "token_identity": identical,
        "token_count_identity": count_identity,
        "stop_reason_identity": stop_identity,
        "deterministic": deterministic,
        "reference_tokens": reference,
        "ratios": {},
    }
    base = names[0]
    for name in names[1:]:
        result["ratios"][f"{name}/{base}"] = {
            metric: paired_ratio(per_arm[name][metric], per_arm[base][metric])
            for metric in ("total_ns", "prefill_ns", "decode_ns")
        }
    return result


def report(result: dict[str, Any]) -> str:
    lines = [f"token identity: {result['token_identity']}  deterministic: {result['deterministic']}"]
    for name, metrics in result["per_arm"].items():
        lines.append(f"  {name:34s} total {metrics['total_ns']['median']/1e6:8.2f} ms  "
                     f"prefill {metrics['prefill_ns']['median']/1e6:8.2f}  "
                     f"decode {metrics['decode_ns']['median']/1e6:8.2f}")
    for pair, metrics in result["ratios"].items():
        lines.append(f"  {pair}")
        for metric, r in metrics.items():
            lines.append(f"    {metric:11s} ratio {r['median_ratio']:.4f}  "
                         f"95% CI [{r['ci_low']:.4f}; {r['ci_high']:.4f}]  "
                         f"({(1-r['median_ratio'])*100:+.2f}%)")
    return "\n".join(lines)


def _self_check() -> None:
    assert interleave(["base", "cand"], 3) == [["base", "cand"], ["cand", "base"], ["base", "cand"]]
    k = Knobs(fuse_projections=True)
    assert Knobs(**k.as_dict()) == k
    assert replace(k, fuse_projections=False).fuse_projections is False
    print("ab self-check ok")


if __name__ == "__main__":
    _self_check()
