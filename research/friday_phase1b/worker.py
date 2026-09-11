"""Sacrificial fixed worker for Phase-1B qualification and measurement."""

from __future__ import annotations

import argparse
import os
import platform
import resource
import subprocess
import sys
import time
from typing import Any

from .canonical import canonical_json_bytes
from .constants import (
    AA_FIXTURE_SEEDS,
    AA_MODE,
    AA_ORDER_SEEDS,
    AB_FIXTURE_SEEDS,
    AB_MODE,
    AB_ORDER_SEEDS,
    BASELINE_NAMES,
    CHARACTERIZE_BLOCKS,
    CHARACTERIZE_FIXTURE_SEEDS,
    CHARACTERIZE_MODE,
    CHARACTERIZE_ORDER_SEEDS,
    CONFIRM_BLOCKS,
    CONTRACT_ID,
    CORE_LIMIT_BYTES,
    CPU_LIMIT_SECONDS,
    EXPECTED_DEVICE_NAME,
    EXPECTED_MLX_VERSION,
    FILE_LIMIT_BYTES,
    MEMORY_PROBE_OPERATIONS,
    MLX_CACHE_LIMIT_BYTES,
    MLX_MEMORY_LIMIT_BYTES,
    NOFILE_LIMIT,
    OPERATIONS_PER_BLOCK,
    QUALIFICATION_CASES,
    QUALIFICATION_MODE,
    RESULT_LIMIT_BYTES,
    SCHEMA_VERSION,
    WARMUPS,
    WORKER_MODES,
)
from .constants import CASE_SEEDS
from .kernel_source import KERNEL_NAME, KERNEL_SOURCE_SHA256


class WorkerError(RuntimeError):
    """The closed worker cannot produce valid evidence."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", choices=sorted(WORKER_MODES), required=True)
    parser.add_argument("--session-index", type=int, required=True)
    parser.add_argument("--baseline", choices=BASELINE_NAMES)
    return parser


def _install_limits() -> dict[str, Any]:
    limits = (
        (resource.RLIMIT_CPU, CPU_LIMIT_SECONDS, "cpu"),
        (resource.RLIMIT_CORE, CORE_LIMIT_BYTES, "core"),
        (resource.RLIMIT_FSIZE, FILE_LIMIT_BYTES, "fsize"),
        (resource.RLIMIT_NOFILE, NOFILE_LIMIT, "nofile"),
    )
    applied: dict[str, Any] = {}
    for key, value, name in limits:
        try:
            resource.setrlimit(key, (value, value))
        except (OSError, ValueError) as exc:
            raise WorkerError(f"required {name} resource limit is unavailable") from exc
        applied[name] = list(resource.getrlimit(key))
    address: dict[str, str] = {}
    for name in ("RLIMIT_AS", "RLIMIT_DATA"):
        key = getattr(resource, name, None)
        if key is None:
            address[name] = "unavailable"
            continue
        current = resource.getrlimit(key)
        target = 24 * 1024**3 if name == "RLIMIT_AS" else 4 * 1024**3
        try:
            resource.setrlimit(key, (target, current[1]))
        except (OSError, ValueError):
            address[name] = "unsupported"
        else:
            address[name] = "active"
    applied["address_space"] = address
    return applied


def _install_network_audit() -> None:
    blocked = {
        "socket.bind",
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.listen",
    }

    def audit(event: str, _args: tuple[Any, ...]) -> None:
        if event in blocked:
            raise PermissionError("network operation denied in Phase-1B worker")

    sys.addaudithook(audit)


def _power_source() -> str | None:
    try:
        completed = subprocess.run(
            ["/usr/bin/pmset", "-g", "batt"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    text = completed.stdout.decode("utf-8", errors="replace")
    if "AC Power" in text:
        return "ac"
    if "Battery Power" in text:
        return "battery"
    return "unknown"


def _device(mx: Any) -> dict[str, Any]:
    info = dict(mx.device_info())
    if mx.__version__ != EXPECTED_MLX_VERSION or not mx.metal.is_available():
        raise WorkerError("MLX version or Metal availability differs")
    if info.get("device_name") != EXPECTED_DEVICE_NAME:
        raise WorkerError("Metal device differs from the frozen target")
    return {
        "mlx_version": mx.__version__,
        "metal_available": True,
        "device_info": info,
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
    }


def _process_metrics(started_wall: int, started_cpu: int) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        rss *= 1024
    return {
        "wall_ns": time.perf_counter_ns() - started_wall,
        "cpu_ns": time.process_time_ns() - started_cpu,
        "rss_peak_bytes": rss,
        "pid": os.getpid(),
        "power_source": _power_source(),
    }


def _qualification(mx: Any, np: Any) -> dict[str, Any]:
    from .kernel import construct_candidate
    from .workload import (
        accuracy_metrics,
        evaluate,
        make_all_baselines,
        make_fixture,
        pair_metrics,
        to_mlx,
    )

    candidate = construct_candidate(mx)
    baselines = make_all_baselines(mx)
    first = make_fixture(np, "zeros", None)
    first_inputs = to_mlx(mx, first)
    compile_started = time.perf_counter_ns()
    first_output = candidate(*first_inputs)
    mx.eval(first_output)
    mx.synchronize()
    compile_first_eval_ns = time.perf_counter_ns() - compile_started
    cases: list[dict[str, Any]] = []
    all_passed = True
    for name in QUALIFICATION_CASES:
        fixture = first if name == "zeros" else make_fixture(np, name, CASE_SEEDS.get(name))
        inputs = first_inputs if name == "zeros" else to_mlx(mx, fixture)
        exact_zero = name in {"zeros", "cancellation"}
        candidate_output = evaluate(mx, np, candidate, inputs)
        candidate_metrics = accuracy_metrics(
            np, candidate_output, fixture.oracle, exact_zero=exact_zero
        )
        baseline_metrics: dict[str, Any] = {}
        pair_checks: dict[str, Any] = {}
        for baseline_name, baseline in baselines.items():
            output = evaluate(mx, np, baseline, inputs)
            baseline_metrics[baseline_name] = accuracy_metrics(
                np, output, fixture.oracle, exact_zero=exact_zero
            )
            pair_checks[baseline_name] = pair_metrics(np, candidate_output, output)
        case_passed = (
            candidate_metrics["passed"]
            and all(value["passed"] for value in baseline_metrics.values())
            and all(value["passed"] for value in pair_checks.values())
        )
        cases.append(
            {
                "name": name,
                "seed": fixture.seed,
                "fixture_sha256": fixture.digest,
                "candidate": candidate_metrics,
                "baselines": baseline_metrics,
                "candidate_pairs": pair_checks,
                "passed": case_passed,
            }
        )
        all_passed = all_passed and case_passed
    memory = {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }
    memory_passed = memory["peak_bytes"] <= MLX_MEMORY_LIMIT_BYTES
    return {
        "compile_first_eval_ns": compile_first_eval_ns,
        "cases": cases,
        "memory": memory,
        "gates": {"correctness": all_passed, "mlx_peak": memory_passed},
        "passed": all_passed and memory_passed,
    }


def _guard(
    mx: Any,
    np: Any,
    fixture: Any,
    arms: dict[str, Any],
) -> tuple[tuple[Any, Any, Any], dict[str, Any]]:
    from .workload import accuracy_metrics, evaluate, pair_metrics, to_mlx

    inputs = to_mlx(mx, fixture)
    outputs: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for name, arm in arms.items():
        outputs[name] = evaluate(mx, np, arm, inputs)
        metrics[name] = accuracy_metrics(np, outputs[name], fixture.oracle, exact_zero=False)
    pairs: dict[str, Any] = {}
    if "candidate" in outputs and "baseline" in outputs:
        pairs["candidate_baseline"] = pair_metrics(
            np, outputs["candidate"], outputs["baseline"]
        )
    passed = all(value["passed"] for value in metrics.values()) and all(
        value["passed"] for value in pairs.values()
    )
    return inputs, {"arms": metrics, "pairs": pairs, "passed": passed}


def _characterize(mx: Any, np: Any, index: int) -> dict[str, Any]:
    from .workload import make_all_baselines, make_fixture, measure_arms, warmup

    if not 0 <= index < 3:
        raise WorkerError("characterization session index differs")
    fixture = make_fixture(np, "performance", CHARACTERIZE_FIXTURE_SEEDS[index])
    arms = make_all_baselines(mx)
    inputs, guard = _guard(mx, np, fixture, arms)
    if not guard["passed"]:
        return {"fixture_sha256": fixture.digest, "correctness": guard, "passed": False}
    for arm in arms.values():
        warmup(mx, arm, inputs, WARMUPS)
    timing = measure_arms(
        mx,
        arms,
        inputs,
        blocks=CHARACTERIZE_BLOCKS,
        operations=OPERATIONS_PER_BLOCK,
        order_seed=CHARACTERIZE_ORDER_SEEDS[index],
    )
    return {
        "fixture_sha256": fixture.digest,
        "fixture_seed": CHARACTERIZE_FIXTURE_SEEDS[index],
        "order_seed": CHARACTERIZE_ORDER_SEEDS[index],
        "correctness": guard,
        "timing": timing,
        "passed": True,
    }


def _aa(mx: Any, np: Any, index: int, baseline_name: str) -> dict[str, Any]:
    from .workload import make_baseline, make_fixture, measure_arms, warmup

    if not 0 <= index < 3:
        raise WorkerError("A/A session index differs")
    fixture = make_fixture(np, "performance", AA_FIXTURE_SEEDS[index])
    arms = {
        "a": make_baseline(mx, baseline_name),
        "b": make_baseline(mx, baseline_name),
    }
    inputs, guard = _guard(mx, np, fixture, arms)
    if not guard["passed"]:
        return {"fixture_sha256": fixture.digest, "correctness": guard, "passed": False}
    for arm in arms.values():
        warmup(mx, arm, inputs, WARMUPS)
    timing = measure_arms(
        mx,
        arms,
        inputs,
        blocks=CONFIRM_BLOCKS,
        operations=OPERATIONS_PER_BLOCK,
        order_seed=AA_ORDER_SEEDS[index],
    )
    return {
        "baseline": baseline_name,
        "fixture_sha256": fixture.digest,
        "fixture_seed": AA_FIXTURE_SEEDS[index],
        "order_seed": AA_ORDER_SEEDS[index],
        "correctness": guard,
        "timing": timing,
        "passed": True,
    }


def _ab(mx: Any, np: Any, index: int, baseline_name: str) -> dict[str, Any]:
    from .kernel import construct_candidate
    from .workload import make_baseline, make_fixture, measure_arms, memory_probe, warmup

    if not 0 <= index < 3:
        raise WorkerError("A/B session index differs")
    fixture = make_fixture(np, "performance", AB_FIXTURE_SEEDS[index])
    arms = {
        "baseline": make_baseline(mx, baseline_name),
        "candidate": construct_candidate(mx),
    }
    inputs, guard = _guard(mx, np, fixture, arms)
    if not guard["passed"]:
        return {"fixture_sha256": fixture.digest, "correctness": guard, "passed": False}
    for arm in arms.values():
        warmup(mx, arm, inputs, WARMUPS)
    timing = measure_arms(
        mx,
        arms,
        inputs,
        blocks=CONFIRM_BLOCKS,
        operations=OPERATIONS_PER_BLOCK,
        order_seed=AB_ORDER_SEEDS[index],
    )
    memory = {
        name: memory_probe(mx, arm, inputs, operations=MEMORY_PROBE_OPERATIONS)
        for name, arm in arms.items()
    }
    return {
        "baseline": baseline_name,
        "fixture_sha256": fixture.digest,
        "fixture_seed": AB_FIXTURE_SEEDS[index],
        "order_seed": AB_ORDER_SEEDS[index],
        "correctness": guard,
        "timing": timing,
        "memory": memory,
        "passed": True,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.mode == QUALIFICATION_MODE:
        if args.session_index != 0 or args.baseline is not None:
            raise WorkerError("qualification arguments differ")
        return
    if not 0 <= args.session_index < 3:
        raise WorkerError("worker session index differs")
    if args.mode == CHARACTERIZE_MODE and args.baseline is not None:
        raise WorkerError("characterization cannot accept a baseline")
    if args.mode in {AA_MODE, AB_MODE} and args.baseline not in BASELINE_NAMES:
        raise WorkerError("confirmation baseline differs")


def main(argv: list[str] | None = None) -> int:
    started_wall = time.perf_counter_ns()
    started_cpu = time.process_time_ns()
    args = _parser().parse_args(argv)
    result: dict[str, Any]
    try:
        _validate_args(args)
        limits = _install_limits()
        _install_network_audit()
        import mlx.core as mx
        import numpy as np

        previous_memory = int(mx.set_memory_limit(MLX_MEMORY_LIMIT_BYTES))
        previous_cache = int(mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES))
        device = _device(mx)
        if args.mode == QUALIFICATION_MODE:
            evidence = _qualification(mx, np)
        elif args.mode == CHARACTERIZE_MODE:
            evidence = _characterize(mx, np, args.session_index)
        elif args.mode == AA_MODE:
            evidence = _aa(mx, np, args.session_index, args.baseline)
        else:
            evidence = _ab(mx, np, args.session_index, args.baseline)
        passed = bool(evidence.get("passed"))
        result = {
            "schema_version": SCHEMA_VERSION,
            "contract_id": CONTRACT_ID,
            "mode": args.mode,
            "session_index": args.session_index,
            "baseline": args.baseline,
            "status": "passed" if passed else "failed",
            "source_sha256": KERNEL_SOURCE_SHA256,
            "kernel_name": KERNEL_NAME,
            "limits": {
                "resource": limits,
                "mlx_memory_bytes": MLX_MEMORY_LIMIT_BYTES,
                "mlx_cache_bytes": MLX_CACHE_LIMIT_BYTES,
                "previous_mlx_memory_bytes": previous_memory,
                "previous_mlx_cache_bytes": previous_cache,
            },
            "device": device,
            "evidence": evidence,
            "error": None
            if passed
            else {"type": "GateFailure", "message": f"{args.mode} gate failed"},
        }
    except Exception as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "contract_id": CONTRACT_ID,
            "mode": getattr(args, "mode", "unknown"),
            "session_index": getattr(args, "session_index", -1),
            "baseline": getattr(args, "baseline", None),
            "status": "failed",
            "source_sha256": KERNEL_SOURCE_SHA256,
            "kernel_name": KERNEL_NAME,
            "limits": None,
            "device": None,
            "evidence": None,
            "error": {"type": type(exc).__name__, "message": str(exc)[:512]},
        }
    result["process"] = _process_metrics(started_wall, started_cpu)
    payload = canonical_json_bytes(result, maximum=RESULT_LIMIT_BYTES)
    sys.stdout.buffer.write(payload + b"\n")
    sys.stdout.buffer.flush()
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
