#!/usr/bin/env python3
"""Q3b: a dry-run-first residual-swap safety canary.

Q3b deliberately does not import IronMule or MLX in the parent.  The parent
binds the local 4B identity and starts two independent stage workers.  Each
worker reuses ``ironmule.ab.run`` with one arm only, so the model child is fresh
per arm and the second stage is never started after a failed first-stage gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import resource
import secrets
import shlex
import signal
import stat
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

EXPERIMENT_ID = "Q3b-residual-swap-safety-canary"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
EXPECTED_REVISION = "93724907d4ed1745d2fe50baadf3b0b01a65abf2"
STAGES = ("baseline", "candidate")
REPEATS = 3
WARMUP = 1
MAX_TOKENS = 32
CHILD_TIMEOUT_SECONDS = 35
WORKER_MAX_SECONDS = 120
PILOT_DEADLINE_SECONDS = 180
COMMAND_TIMEOUT_SECONDS = 1.0
START_SWAP_LIMIT_BYTES = 4 * 1024**3
SWAP_DELTA_LIMIT_BYTES = 128 * 1024**2
START_FREE_PERCENT = 35
AFTER_FREE_PERCENT = 20
LOAD_MAX = 8.0
LOAD_SPREAD_MAX = 2.0
PEAK_CEILING_FRACTION = 0.60
MAX_COMMAND_OUTPUT = 64 * 1024
MAX_PS_OUTPUT = 512 * 1024
MAX_WORKER_OUTPUT = 512 * 1024
CAPABILITY_MAX_BYTES = 16 * 1024
KNOWN_INFERENCE_ACTIVITY = ("mlx", "llama", "ollama", "vllm", "gemma", "qwen")
CLAUDE_DESKTOP_EXECUTABLE = "/Applications/Claude.app/Contents/MacOS/Claude"
CLAUDE_DESKTOP_BUNDLE = "/Applications/Claude.app"
CLAUDE_DESKTOP_CONTENTS = CLAUDE_DESKTOP_BUNDLE + "/Contents/"
CLAUDE_CODESIGN = "/usr/bin/codesign"
# The complete sealed-bundle verification is materially slower than the other
# short OS probes on this host; keep its longer bound local to these two calls.
CLAUDE_CODESIGN_TIMEOUT_SECONDS = 5.0
CLAUDE_BUNDLE_IDENTIFIER = "com.anthropic.claudefordesktop"
CLAUDE_BUNDLE_TEAM = "Q6L2SF6YDW"
CLAUDE_BUNDLE_AUTHORITY = "Developer ID Application: Anthropic PBC (Q6L2SF6YDW)"
STAGE_SAMPLE_INTERVAL_SECONDS = 0.25
# A sampler command may occupy the full one-second command timeout.  The
# interval plus that timeout and a 0.5 s scheduling margin bound gaps without
# pretending that the sampler is a real-time observer.
MAX_SWAP_SAMPLE_GAP_SECONDS = STAGE_SAMPLE_INTERVAL_SECONDS + COMMAND_TIMEOUT_SECONDS + 0.5
MAX_SWAP_SAMPLES = 512
POST_STAGE_RESERVE_SECONDS = 10
COMMANDS = {
    "pmset": "/usr/bin/pmset", "osascript": "/usr/bin/osascript",
    "sysctl": "/usr/sbin/sysctl", "memory_pressure": "/usr/bin/memory_pressure",
    "ps": "/bin/ps", "git": "/usr/bin/git",
}
CLEANUP_PS_COMMAND = ["/bin/ps", "-Ao", "pid=,ppid=,pgid=,sid=,uid=,stat=,start=,args="]
CLEANUP_EVIDENCE_SCHEMA = "ironmule.cleanup.v2"
CLEANUP_STAT_RE = re.compile(r"^[DRSITUWZNL](?:[s+<>-]*)$")
CLEANUP_START_RE = re.compile(r"^[^\s]+$")
MAX_BASELINE_IDENTITIES = 4096
PREREGISTRATION = Path(__file__).resolve().parent / "raw" / "Q3b_preregistration.md"
PREREGISTRATION_SHA = Path(__file__).resolve().parent / "raw" / "Q3b_preregistration.sha256"

Q2_INCUMBENT = {
    "fuse_projections": False, "compiled_fixed_cache": True, "fused_argmax": False,
    "head_skip_prefill": True, "prefill_into_fixed": False, "readback_every": 2,
    "speculate_k": 0, "speculate_ngram": 3, "capacity_slack": 0, "wired_fraction": 0.0,
}
Q3B_CANDIDATE = {**Q2_INCUMBENT, "fused_argmax": True}
ARMS = {"baseline": Q2_INCUMBENT, "candidate": Q3B_CANDIDATE}


class CanaryRefused(RuntimeError):
    """Raised when a safety fact is unknown or a stage cannot be proven safe."""


class _CommandText(str):
    def __new__(cls, value: str, ok: bool):
        result = super().__new__(cls, value)
        result.ok = ok
        return result


def _run_text(command: list[str], timeout: float = COMMAND_TIMEOUT_SECONDS) -> str:
    if (not command or not Path(command[0]).is_absolute()
            or not Path(command[0]).is_file() or not os.access(command[0], os.X_OK)):
        return _CommandText("", False)
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return _CommandText("", False)
    limit = MAX_PS_OUTPUT if command in (
        [COMMANDS["ps"], "-Ao", "pid=,ppid=,rss=,%cpu=,args="],
        [COMMANDS["ps"], "-Ao", "pid=,comm="],
        CLEANUP_PS_COMMAND,
    ) else MAX_COMMAND_OUTPUT
    if completed.returncode != 0 or not isinstance(completed.stdout, str) or len(completed.stdout) > limit:
        return _CommandText("", False)
    return _CommandText(completed.stdout, True)


def _deadline_runner(deadline: float, run: Callable[[list[str]], str] = _run_text):
    def bounded(command: list[str]) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _CommandText("", False)
        if run is _run_text:
            return run(command, timeout=min(COMMAND_TIMEOUT_SECONDS, remaining))
        return run(command)
    return bounded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_code_sha256(root: Path | None = None) -> str:
    root = Path(__file__).resolve().parents[1] if root is None else root.resolve()
    paths = sorted((root / "ironmule").rglob("*.py"))
    for extra in (root / "research" / "q3a_path_interaction.py", Path(__file__).resolve()):
        if extra.is_file() and extra not in paths:
            paths.append(extra)
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _load_q3a_helpers() -> Any:
    """Load unchanged Q3a helpers only after the capability gate in a worker."""
    path = Path(__file__).resolve().with_name("q3a_path_interaction.py")
    spec = importlib.util.spec_from_file_location("ironmule_q3b_q3a_helpers", path)
    if spec is None or spec.loader is None:
        raise CanaryRefused("unchanged Q3a helper surface is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_identity_module(root: Path) -> Any:
    path = root / "ironmule" / "model_identity.py"
    spec = importlib.util.spec_from_file_location("ironmule_q3b_model_identity", path)
    if spec is None or spec.loader is None:
        raise CanaryRefused("cannot load stdlib model identity contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ironmule_q3b_model_identity"] = module
    spec.loader.exec_module(module)
    return module


def _cache_roots() -> tuple[Path, ...]:
    roots = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]).expanduser())
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return tuple(dict.fromkeys(roots))


def resolve_exact_local_identity(root: Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1] if root is None else root.resolve()
    encoded = "models--" + MODEL_ID.replace("/", "--")
    snapshots = [cache / encoded / "snapshots" / EXPECTED_REVISION for cache in _cache_roots()]
    existing = [path for path in snapshots if path.is_dir()]
    if len(existing) != 1:
        raise CanaryRefused(f"expected exactly one local 4B snapshot at pinned revision, found {len(existing)}")
    try:
        identity = _load_identity_module(root).build_model_identity(MODEL_ID, existing[0], EXPECTED_REVISION)
        binding = {"model_id": identity.model_id, "model_revision": identity.revision,
                   "model_manifest_sha256": identity.model_manifest_sha256}
    except (AttributeError, ImportError, OSError, ValueError, TypeError, KeyError) as exc:
        raise CanaryRefused(f"local model identity failed: {exc}") from exc
    if (binding["model_id"] != MODEL_ID or binding["model_revision"] != EXPECTED_REVISION
            or not re.fullmatch(r"[0-9a-f]{64}", str(binding["model_manifest_sha256"]))):
        raise CanaryRefused("local model identity does not match pinned 4B binding")
    return binding


def _swap_bytes(text: str) -> int | None:
    if not isinstance(text, str):
        return None
    match = re.fullmatch(r"\s*.*?used\s*=\s*([0-9]+(?:[.,][0-9]+)?)\s*([KMG]?).*", text.rstrip("\r\n"), re.I)
    if not match:
        return None
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2).upper()]
    try:
        value = float(match.group(1).replace(",", ".")) * scale
        return int(value) if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _memory_free_percent(text: str) -> int | None:
    match = re.search(r"memory free percentage:\s*([0-9]+)%", text or "", re.I)
    if not match:
        return None
    value = int(match.group(1))
    return value if 0 <= value <= 100 else None


def _thermal_nominal(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("note: "):
            line = line[6:].strip()
        if line:
            lines.append(line.lower())
    values = set(lines)
    required = {"no thermal warning level has been recorded", "no performance warning level has been recorded"}
    if len(lines) != len(values) or not required.issubset(values):
        return False
    allowed = {"no cpu power status has been recorded", "no cpu power status is available", "no cpu power status available"}
    zero = re.compile(r"^(?:(?:cpu|gpu)[ _](?:speed|scheduler)[ _]limit|number of prochot entries|pressure)\s*[:=]\s*0(?:\.0+)?$")
    return all(line in allowed or zero.fullmatch(line) for line in values - required)


def system_environment(run: Callable[[list[str]], str] = _run_text) -> dict[str, Any]:
    try:
        battery = run([COMMANDS["pmset"], "-g", "batt"])
        low_power = run([COMMANDS["osascript"], "-l", "JavaScript", "-e",
                         'ObjC.import("Foundation"); JSON.stringify($.NSProcessInfo.processInfo.isLowPowerModeEnabled)'])
        thermal = run([COMMANDS["pmset"], "-g", "therm"])
        swap = run([COMMANDS["sysctl"], "-n", "vm.swapusage"])
        pressure = run([COMMANDS["memory_pressure"]])
    except Exception:
        battery = low_power = thermal = swap = pressure = ""
    def valid(value: Any) -> str:
        return value if isinstance(value, str) and getattr(value, "ok", True) else ""
    battery, low_power, thermal, swap, pressure = map(valid, (battery, low_power, thermal, swap, pressure))
    mode = low_power.strip()
    return {
        "power_source": "AC" if "ac power" in battery.lower() else "battery" if battery else "unknown",
        "low_power_mode": False if mode == "false" else True if mode == "true" else None,
        "thermal_state": "nominal" if _thermal_nominal(thermal) else "unknown",
        "swap_used_bytes": _swap_bytes(swap), "memory_free_percent": _memory_free_percent(pressure),
        "platform": platform.platform(), "python": platform.python_version(),
    }


def _read_swap_sample(run: Callable[[list[str]], str] = _run_text) -> int:
    """Read one strict swap sample; command and parse failures are fatal."""
    try:
        output = run([COMMANDS["sysctl"], "-n", "vm.swapusage"])
    except BaseException as exc:
        raise CanaryRefused(f"swap sampler command failed: {type(exc).__name__}") from exc
    if not isinstance(output, str) or not getattr(output, "ok", True):
        raise CanaryRefused("swap sampler command failed")
    value = _swap_bytes(output)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CanaryRefused("swap sampler output could not be parsed")
    return value


def loadavg_gate(sample: Callable[[], float] = lambda: os.getloadavg()[0], sleeper: Callable[[float], None] = time.sleep,
                 *, deadline: float | None = None) -> dict[str, Any]:
    samples = []
    for index in range(3):
        if deadline is not None and time.monotonic() >= deadline:
            samples.extend([math.nan] * (3 - index)); break
        if index:
            if deadline is not None and time.monotonic() + 1.0 >= deadline:
                samples.append(math.nan); continue
            try:
                sleeper(1.0)
            except (OSError, TypeError, ValueError):
                samples.append(math.nan); continue
        try:
            value = float(sample())
        except (TypeError, ValueError, OverflowError, OSError):
            value = math.nan
        samples.append(value)
    valid = len(samples) == 3 and all(math.isfinite(value) and value >= 0 for value in samples)
    spread = max(samples) - min(samples) if valid else None
    return {"samples": samples, "max": max(samples) if valid else None, "spread": spread,
            "passed": valid and max(samples) <= LOAD_MAX and spread <= LOAD_SPREAD_MAX}


def _trusted_claude_bundle(bundle: str = CLAUDE_DESKTOP_BUNDLE,
                           runner: Callable[..., Any] | None = None) -> bool:
    """Return true only for the exact, currently verified Claude app bundle.

    ``codesign --verify`` proves the complete sealed bundle.  ``codesign -dv``
    writes its metadata to stderr, so both the bounded command result and the
    exact identity/authority fields are checked here.  Any timeout, malformed
    output, non-zero status, or missing field is deliberately untrusted.
    """
    if bundle != CLAUDE_DESKTOP_BUNDLE or not Path(bundle).is_absolute():
        return False
    execute = subprocess.run if runner is None else runner
    try:
        verified = execute(
            [CLAUDE_CODESIGN, "--verify", "--deep", "--strict", bundle],
            capture_output=True, text=True, timeout=CLAUDE_CODESIGN_TIMEOUT_SECONDS, check=False,
        )
        if getattr(verified, "returncode", None) != 0:
            return False
        verify_stdout = getattr(verified, "stdout", "")
        verify_stderr = getattr(verified, "stderr", "")
        if (not isinstance(verify_stdout, str) or not isinstance(verify_stderr, str)
                or len(verify_stdout) + len(verify_stderr) > MAX_COMMAND_OUTPUT):
            return False
        metadata = execute(
            [CLAUDE_CODESIGN, "-dv", "--verbose=4", bundle],
            capture_output=True, text=True, timeout=CLAUDE_CODESIGN_TIMEOUT_SECONDS, check=False,
        )
        if getattr(metadata, "returncode", None) != 0:
            return False
        stdout = getattr(metadata, "stdout", "")
        stderr = getattr(metadata, "stderr", "")
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            return False
        if len(stdout) + len(stderr) > MAX_COMMAND_OUTPUT:
            return False
    except BaseException:
        return False

    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    identifiers = [line.split("=", 1)[1] for line in lines if line.startswith("Identifier=") and "=" in line]
    teams = [line.split("=", 1)[1] for line in lines if line.startswith("TeamIdentifier=") and "=" in line]
    authorities = [line.split("=", 1)[1] for line in lines if line.startswith("Authority=") and "=" in line]
    return (
        len(identifiers) == 1 and identifiers[0] == CLAUDE_BUNDLE_IDENTIFIER
        and len(teams) == 1 and teams[0] == CLAUDE_BUNDLE_TEAM
        and bool(authorities) and authorities[0] == CLAUDE_BUNDLE_AUTHORITY
    )


def _parse_process_args_inventory(output: Any) -> dict[int, tuple[int, int, float, str]] | str:
    if not isinstance(output, str) or not output or not getattr(output, "ok", True):
        return "process inventory unavailable or command failed"
    if len(output) > MAX_PS_OUTPUT:
        return "process inventory exceeded bounded limit"
    records: dict[int, tuple[int, int, float, str]] = {}
    for line in output.splitlines():
        parts = line.split(None, 4)
        if len(parts) != 5:
            return "process inventory malformed"
        try:
            pid, ppid, rss, cpu = (int(parts[0]), int(parts[1]), int(parts[2]),
                                   float(parts[3]))
        except (TypeError, ValueError, OverflowError):
            return "process inventory malformed"
        if (pid <= 0 or ppid < 0 or rss < 0 or not math.isfinite(cpu) or cpu < 0
                or pid in records):
            return "process inventory malformed"
        try:
            argv = shlex.split(parts[4], posix=True)
        except (TypeError, ValueError):
            return "process inventory malformed"
        if not argv:
            return "process inventory malformed"
        records[pid] = (ppid, rss, cpu, parts[4])
    return records


def _parse_process_comm_inventory(output: Any) -> dict[int, str] | str:
    if not isinstance(output, str) or not output or not getattr(output, "ok", True):
        return "process inventory unavailable or command failed"
    if len(output) > MAX_PS_OUTPUT:
        return "process inventory exceeded bounded limit"
    records: dict[int, str] = {}
    for line in output.splitlines():
        # Split only once: comm= is a path and may contain spaces.
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            return "process inventory malformed"
        try:
            pid = int(parts[0])
        except (TypeError, ValueError, OverflowError):
            return "process inventory malformed"
        comm = parts[1].strip()
        if pid <= 0 or pid in records or not comm:
            return "process inventory malformed"
        records[pid] = comm
    return records


def _pid_probe(pid: int, killer: Callable[[int, int], None] | None = None) -> str:
    """Classify a PID without sending a signal or changing process state.

    ``kill(pid, 0)`` only probes existence/permission.  A missing process is
    expected when the two ``ps`` snapshots straddle process exit; permission
    and every other error remain unknown and therefore fail closed at the
    caller.  ``killer`` is injectable for deterministic tests.
    """
    try:
        (os.kill if killer is None else killer)(pid, 0)
    except ProcessLookupError:
        return "gone"
    except BaseException:
        return "unknown"
    return "alive"


def competing_model_process(run: Callable[[list[str]], str] = _run_text,
                            *, absent_pids: tuple[int, ...] = (),
                            pid_probe: Callable[[int], str] | None = None) -> str | None:
    """Block inference activity using two bounded, PID-aware inventories.

    The args inventory is the relevance snapshot and must contain the current
    PID's complete same-snapshot ``ppid`` chain through a root (``ppid=0``).
    Only that PID and proven ancestors are ignored before Claude/model checks;
    descendants and siblings remain relevant.  Claude/model records outside
    that ancestry must have a same-PID comm record.  A missing comm record is
    tolerated only when an injected, read-only PID probe proves that the
    process exited between snapshots; alive, permission, and unknown states
    fail closed.  Extra comm records are ignored because they may be newer
    processes.
    """
    args_command = [COMMANDS["ps"], "-Ao", "pid=,ppid=,rss=,%cpu=,args="]
    comm_command = [COMMANDS["ps"], "-Ao", "pid=,comm="]
    try:
        args_output = run(args_command)
        comm_output = run(comm_command)
    except BaseException:
        return "process inventory unavailable or command failed"
    args_records = _parse_process_args_inventory(args_output)
    comm_records = _parse_process_comm_inventory(comm_output)
    if isinstance(args_records, str):
        return args_records
    if isinstance(comm_records, str):
        return comm_records
    probe = _pid_probe if pid_probe is None else pid_probe
    current_pid = os.getpid()
    if current_pid not in args_records:
        return "process inventory ancestry malformed"
    ancestry: set[int] = set()
    pid = current_pid
    while True:
        if pid in ancestry:
            return "process inventory ancestry malformed"
        ancestry.add(pid)
        record = args_records.get(pid)
        if record is None:
            return "process inventory ancestry malformed"
        ppid = record[0]
        if ppid == 0:
            break
        if ppid not in args_records:
            return "process inventory ancestry malformed"
        pid = ppid

    trusted_bundle: bool | None = None
    for pid, (ppid, rss, cpu, args_text) in args_records.items():
        if pid in absent_pids:
            return "prior model child was not reaped"
        if pid in ancestry:
            continue
        lowered_args = args_text.casefold()
        # Model/inference patterns are always blockers, including if a process
        # happens to use a path below the Claude bundle.  This check is before
        # comm lookup so a model process cannot disappear from the second
        # snapshot and become an allowed gap.
        if any(token in lowered_args for token in KNOWN_INFERENCE_ACTIVITY):
            return "competing model activity detected"
        comm_path = comm_records.get(pid)
        if comm_path is None:
            # Only an args record that is itself Claude-related needs a
            # same-PID comm proof.  Irrelevant args records may legitimately
            # lack a comm row, and extra comm rows are intentionally ignored.
            if "claude" not in lowered_args:
                continue
            try:
                status = probe(pid)
            except BaseException:
                status = "unknown"
            if status == "gone":
                continue
            return "process inventory pid map mismatch"
        claude_related = "claude" in comm_path.casefold() or "claude" in lowered_args
        if not claude_related:
            continue
        # The bundle path is an exact macOS path boundary; do not case-fold it
        # or permit a similarly named root to cross the boundary.
        inside_bundle = comm_path.startswith(CLAUDE_DESKTOP_CONTENTS)
        if inside_bundle:
            if trusted_bundle is None:
                trusted_bundle = _trusted_claude_bundle()
            if trusted_bundle:
                continue
        return "unverified Claude process activity detected"
    return None


def installed_memory_bytes(run: Callable[[list[str]], str] = _run_text) -> int | None:
    try:
        value = int(run([COMMANDS["sysctl"], "-n", "hw.memsize"]).strip())
        return value if value > 0 else None
    except (TypeError, ValueError, OSError):
        return None


def _git_binding(root: Path, run: Callable[[list[str]], str]) -> dict[str, Any]:
    try:
        commit_output = run([COMMANDS["git"], "-C", str(root), "rev-parse", "HEAD"])
        status = run([COMMANDS["git"], "-C", str(root), "status", "--porcelain", "--untracked-files=all"])
        commit = commit_output.strip() if isinstance(commit_output, str) else ""
    except Exception:
        return {"clean": False, "commit": None, "reason": "git command failed"}
    if (not isinstance(status, str) or not getattr(commit_output, "ok", True)
            or not getattr(status, "ok", True) or not re.fullmatch(r"[0-9a-f]{40}", commit)):
        return {"clean": False, "commit": None, "reason": "git commit unavailable"}
    dirty = []
    for line in status.splitlines():
        if len(line) < 4 or not line[3:].strip():
            return {"clean": False, "commit": commit, "dirty_paths": [], "reason": "git status malformed"}
        path = line[3:]
        if path != "research/data/squad-dev-v1.1.json":
            dirty.append(path)
    return {"clean": not dirty, "commit": commit, "dirty_paths": dirty}


def _preregistration_matches() -> bool:
    try:
        fields = PREREGISTRATION_SHA.read_text().strip().split()
        return (PREREGISTRATION.exists() and len(fields) == 2
                and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None
                and _sha256(PREREGISTRATION) == fields[0]
                and fields[1] == PREREGISTRATION.name)
    except (OSError, IndexError):
        return False


def preflight(*, root: Path | None = None, run: Callable[[list[str]], str] = _run_text,
              identity_resolver: Callable[[Path | None], dict[str, Any]] = resolve_exact_local_identity,
              load_sample: Callable[[], float] = lambda: os.getloadavg()[0],
              deadline: float | None = None, load_sleeper: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1] if root is None else root.resolve()
    bounded = _deadline_runner(deadline, run) if deadline is not None else run
    env = system_environment(bounded)
    memory = installed_memory_bytes(bounded)
    try:
        identity = identity_resolver(root)
        identity_error = None
    except Exception as exc:
        identity, identity_error = None, str(exc)
    load = loadavg_gate(load_sample, load_sleeper, deadline=deadline)
    git = _git_binding(root, bounded)
    checks = {
        "model_identity_exact": isinstance(identity, dict) and identity.get("model_id") == MODEL_ID and identity.get("model_revision") == EXPECTED_REVISION and bool(re.fullmatch(r"[0-9a-f]{64}", str(identity.get("model_manifest_sha256", "")))),
        "ac_power": env["power_source"] == "AC", "low_power_off": env["low_power_mode"] is False,
        "thermal_nominal": env["thermal_state"] == "nominal", "no_competing_model_process": competing_model_process(bounded) is None,
        "loadavg_gate": load["passed"], "start_swap_known": isinstance(env["swap_used_bytes"], int),
        "start_swap_within_4gib": isinstance(env["swap_used_bytes"], int) and env["swap_used_bytes"] <= START_SWAP_LIMIT_BYTES,
        "installed_memory_known": memory is not None,
        "start_memory_free_at_least_35_percent": isinstance(env["memory_free_percent"], int) and env["memory_free_percent"] >= START_FREE_PERCENT,
        "git_clean_and_bound": git["clean"] and bool(git["commit"]), "preregistration_matches": _preregistration_matches(),
        "runtime_code_hash_known": bool(re.fullmatch(r"[0-9a-f]{64}", runtime_code_sha256(root))),
    }
    return {"environment": env, "identity": identity, "identity_error": identity_error,
            "git": git, "loadavg": load, "installed_memory_bytes": memory,
            "peak_ceiling_bytes": int(memory * PEAK_CEILING_FRACTION) if memory else None,
            "checks": checks, "passed": all(checks.values())}


def _stage_gate(environment: dict[str, Any], initial_swap: int, *, peak: int | None = None,
                rss: int | None = None, installed: int | None = None,
                max_swap_used_bytes: int | None = None) -> dict[str, Any]:
    swap = environment.get("swap_used_bytes")
    observed_swap = max_swap_used_bytes if isinstance(max_swap_used_bytes, int) else swap
    delta = observed_swap - initial_swap if isinstance(observed_swap, int) and isinstance(initial_swap, int) else None
    ceiling = int(installed * PEAK_CEILING_FRACTION) if isinstance(installed, int) and installed > 0 else None
    checks = {
        "swap_endpoint_known": isinstance(swap, int) and not isinstance(swap, bool) and swap >= 0,
        "swap_delta_within_128mib": isinstance(delta, int) and delta <= SWAP_DELTA_LIMIT_BYTES,
        "memory_free_at_least_20_percent": isinstance(environment.get("memory_free_percent"), int) and environment["memory_free_percent"] >= AFTER_FREE_PERCENT,
        "mlx_peak_within_60_percent": isinstance(peak, int) and isinstance(ceiling, int) and peak <= ceiling,
        "child_rss_within_60_percent": isinstance(rss, int) and isinstance(ceiling, int) and rss <= ceiling,
        "ac_power": environment.get("power_source") == "AC", "low_power_off": environment.get("low_power_mode") is False,
        "thermal_nominal": environment.get("thermal_state") == "nominal",
        "loadavg_gate": isinstance(environment.get("loadavg"), dict) and environment["loadavg"].get("passed") is True,
        "no_competing_model_process": environment.get("competing_model_process") is None,
    }
    return {"swap_used_bytes": swap, "max_swap_used_bytes": observed_swap, "swap_delta_bytes": delta, "peak_ceiling_bytes": ceiling,
            "checks": checks, "passed": all(checks.values())}


def _read_capability() -> dict[str, Any]:
    fd_text = os.environ.get("IRONMULE_Q3B_CAP_FD")
    nonce, expected_text = os.environ.get("IRONMULE_Q3B_CAP_NONCE"), os.environ.get("IRONMULE_Q3B_EXPECTED")
    if not fd_text or not nonce or not expected_text:
        raise CanaryRefused("worker capability is absent")
    fd = -1
    try:
        fd = int(fd_text)
        if fd < 0:
            raise ValueError
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, min(4096, CAPABILITY_MAX_BYTES - total))
            if not chunk:
                break
            chunks.append(chunk); total += len(chunk)
            if total >= CAPABILITY_MAX_BYTES:
                raise ValueError
        payload, expected = json.loads(b"".join(chunks).decode()), json.loads(expected_text)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CanaryRefused("worker capability is malformed") from exc
    finally:
        if fd >= 0:
            try: os.close(fd)
            except OSError: pass
    if (not isinstance(payload, dict) or not isinstance(expected, dict) or payload.get("nonce") != nonce
            or payload.get("expected") != expected or set(expected) != {"identity", "runtime_code_sha256", "stage", "initial_swap", "installed_memory"}):
        raise CanaryRefused("worker capability or expected binding mismatch")
    return expected


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _activate_exact_repo_root() -> Path:
    """Make the worker's repository root the only injected import location."""
    root = Path(__file__).resolve().parents[1]
    package_init = root / "ironmule" / "__init__.py"
    if not package_init.is_file():
        raise CanaryRefused("exact IronMule package root is unavailable")

    for module_name, preloaded in tuple(sys.modules.items()):
        if preloaded is None or not (
                module_name == "ironmule" or module_name.startswith("ironmule.")):
            continue
        preloaded_file = getattr(preloaded, "__file__", None)
        preloaded_spec = getattr(preloaded, "__spec__", None)
        if not isinstance(preloaded_file, str):
            preloaded_file = getattr(preloaded_spec, "origin", None)
        preloaded_paths = getattr(preloaded, "__path__", ())
        try:
            preloaded_paths = tuple(preloaded_paths or ())
            preloaded_has_foreign_path = any(
                not _path_is_within(Path(item), root) for item in preloaded_paths
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise CanaryRefused("preloaded foreign IronMule module") from exc
        if (not isinstance(preloaded_file, str)
                or not _path_is_within(Path(preloaded_file), root)
                or preloaded_has_foreign_path):
            raise CanaryRefused("preloaded foreign IronMule module")

    root_text = str(root)
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    try:
        spec = importlib.util.find_spec("ironmule")
    except (ImportError, AttributeError, ValueError) as exc:
        raise CanaryRefused("exact IronMule module spec is unavailable") from exc
    origin = getattr(spec, "origin", None) if spec is not None else None
    if (spec is None or not isinstance(origin, str)
            or not _path_is_within(Path(origin), root)
            or Path(origin).resolve() != package_init.resolve()):
        raise CanaryRefused("IronMule module spec resolves outside the exact repository root")
    locations = getattr(spec, "submodule_search_locations", None)
    try:
        has_exact_location = locations is not None and any(
            Path(item).resolve() == (root / "ironmule").resolve() for item in locations
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        has_exact_location = False
    if not has_exact_location:
        raise CanaryRefused("IronMule package spec has an unexpected search path")
    return root


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _emit_safety_marker(event: dict[str, Any]) -> None:
    """Write only bounded safety evidence; never include command arguments."""
    print("@SAFETY " + json.dumps(event, sort_keys=True, allow_nan=False), flush=True)


def _capture_live_safety(*, reason: str, samples: list[int], sample_times: list[float],
                         sample_offsets: list[float], sampler_errors: list[str],
                         lock: threading.Lock, state: dict[str, Any],
                         marker_writer: Callable[[dict[str, Any]], None] | None = None,
                         kill_group: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Capture once, publish, and immediately terminate the worker process group."""
    with lock:
        existing = state.get("event")
        if isinstance(existing, dict):
            return existing
        event = {
            "reason": str(reason)[:256],
            "samples": list(samples[:MAX_SWAP_SAMPLES]),
            "times": list(sample_times[:MAX_SWAP_SAMPLES]),
            "offsets": list(sample_offsets[:MAX_SWAP_SAMPLES]),
            "errors": [str(item)[:512] for item in sampler_errors[:32]],
        }
        state["event"] = event
    writer = _emit_safety_marker if marker_writer is None else marker_writer
    killer = os.killpg if kill_group is None else kill_group
    marker_error: BaseException | None = None
    try:
        writer(event)
    except BaseException as exc:
        marker_error = exc
    pgid = os.getpgrp()
    try:
        killer(pgid, signal.SIGTERM)
    except BaseException as exc:
        # A failed TERM must not leave a live model child behind.  Escalate
        # immediately on the same process group; only a double failure is a
        # kill failure for the safety result.
        try:
            killer(pgid, signal.SIGKILL)
        except BaseException as kill_exc:
            failure = (f"safety group kill failed: TERM={type(exc).__name__}: {exc}; "
                       f"KILL={type(kill_exc).__name__}: {kill_exc}")
            with lock:
                sampler_errors.append(failure[:512])
                event["errors"] = [str(item)[:512] for item in sampler_errors[:32]]
                state["kill_error"] = failure[:512]
            try:
                print("@SAFETY_KILL_FAILURE " + json.dumps({"error": failure[:512]}, sort_keys=True),
                      file=sys.stderr, flush=True)
            except BaseException:
                pass
            if marker_error is not None:
                raise CanaryRefused("safety marker and group kill failed") from kill_exc
            raise CanaryRefused("safety group kill failed") from kill_exc
    if marker_error is not None:
        raise CanaryRefused("safety marker write failed") from marker_error
    return event


def _finalize_stage_safety(*, initial_swap: int, samples: list[int],
                           sample_times: list[float], sample_offsets: list[float],
                           sampler_errors: list[str], lock: threading.Lock,
                           state: dict[str, Any], final_error: BaseException | None,
                           marker_writer: Callable[[dict[str, Any]], None] | None = None,
                           kill_group: Callable[[int, int], None] | None = None) -> dict[str, Any] | None:
    """Turn final-read/highwater failures into a terminal safety event."""
    with lock:
        if isinstance(state.get("event"), dict):
            return state["event"]
        highwater = max(samples) if samples else None
        failed = bool(sampler_errors)
    if final_error is None and not failed and not (
            isinstance(highwater, int)
            and highwater - initial_swap > SWAP_DELTA_LIMIT_BYTES):
        return None
    reason = ("final_swap_sampler_error" if final_error is not None or failed
              else "final_swap_delta_exceeded")
    return _capture_live_safety(
        reason=reason, samples=samples, sample_times=sample_times,
        sample_offsets=sample_offsets, sampler_errors=sampler_errors,
        lock=lock, state=state, marker_writer=marker_writer, kill_group=kill_group,
    )


def _stage_worker(*, kill_group: Callable[[int, int], None] | None = None,
                  marker_writer: Callable[[dict[str, Any]], None] | None = None) -> int:
    samples: list[int] = []
    sample_times: list[float] = []
    sample_offsets: list[float] = []
    sampler_errors: list[str] = []
    sampler: threading.Thread | None = None
    stop_monitor: threading.Event | None = None
    safety_state: dict[str, Any] = {}
    try:
        expected = _read_capability()
        if runtime_code_sha256() != expected["runtime_code_sha256"]:
            raise CanaryRefused("runtime code changed after preflight")
        stage = expected["stage"]
        worker_deadline = float(os.environ.get("IRONMULE_Q3B_WORKER_DEADLINE", "nan"))
        if stage not in STAGES or not math.isfinite(worker_deadline):
            raise CanaryRefused("worker stage or deadline is malformed")
        _activate_exact_repo_root()
        worker_started = time.monotonic()
        sample_lock = threading.Lock()

        def record_error(label: str, exc: BaseException) -> None:
            with sample_lock:
                sampler_errors.append(f"{label}: {type(exc).__name__}: {exc}")

        def record_sample(value: int, timestamp: float) -> None:
            with sample_lock:
                if len(samples) >= MAX_SWAP_SAMPLES:
                    raise CanaryRefused("swap sampler sample limit exceeded")
                if not math.isfinite(timestamp) or timestamp < worker_started:
                    raise CanaryRefused("swap sampler monotonic timestamp is invalid")
                if sample_times and timestamp < sample_times[-1]:
                    raise CanaryRefused("swap sampler timestamps are not monotonic")
                samples.append(value)
                sample_times.append(timestamp)
                sample_offsets.append(timestamp - worker_started)

        def synchronous_sample(label: str, run: Callable[[list[str]], str]) -> None:
            try:
                value = _read_swap_sample(run)
                timestamp = time.monotonic()
                record_sample(value, timestamp)
            except BaseException as exc:
                record_error(label, exc)
                raise

        # This sample is synchronous and is deliberately not used as a new
        # comparison origin: the parent preflight value remains authoritative.
        synchronous_sample("worker-start", _deadline_runner(worker_deadline))
        if samples[-1] - expected["initial_swap"] > SWAP_DELTA_LIMIT_BYTES:
            raise CanaryRefused("worker-start swap delta exceeded 128 MiB")
        stop_monitor = threading.Event()

        def monitor_swap() -> None:
            while not stop_monitor.wait(STAGE_SAMPLE_INTERVAL_SECONDS):
                try:
                    synchronous_sample("periodic", _deadline_runner(worker_deadline))
                    with sample_lock:
                        highwater = max(samples) if samples else None
                        sampler_failed = bool(sampler_errors)
                    if sampler_failed:
                        raise CanaryRefused("swap sampler error")
                    if (isinstance(highwater, int)
                            and highwater - expected["initial_swap"] > SWAP_DELTA_LIMIT_BYTES):
                        _capture_live_safety(
                            reason="swap_delta_exceeded", samples=samples,
                            sample_times=sample_times, sample_offsets=sample_offsets,
                            sampler_errors=sampler_errors, lock=sample_lock,
                            state=safety_state, marker_writer=marker_writer,
                            kill_group=kill_group,
                        )
                        stop_monitor.set()
                except BaseException as exc:
                    with sample_lock:
                        live_event = safety_state.get("event")
                    if not isinstance(live_event, dict):
                        # Any sampler/read/parse/command failure while a model
                        # child may be alive is an immediate group abort.
                        _capture_live_safety(
                            reason="swap_sampler_error", samples=samples,
                            sample_times=sample_times, sample_offsets=sample_offsets,
                            sampler_errors=sampler_errors or [f"periodic: {type(exc).__name__}: {exc}"],
                            lock=sample_lock, state=safety_state,
                            marker_writer=marker_writer, kill_group=kill_group,
                        )
                    stop_monitor.set()
                    return

        sampler = threading.Thread(target=monitor_swap, name="q3b-swap-monitor", daemon=True)
        sampler.start()
        q3a_helpers = _load_q3a_helpers()
        from ironmule import ab
        from ironmule.model_identity import ModelIdentityError
        from ironmule.runtime import Knobs
        from ironmule.tune import resolve_local_model
        try:
            resolved = resolve_local_model(MODEL_ID, revision=EXPECTED_REVISION)
        except (OSError, ValueError, ModelIdentityError) as exc:
            raise CanaryRefused(f"worker model resolution failed: {exc}") from exc
        actual = {"model_id": resolved.identity.model_id, "model_revision": resolved.identity.revision,
                  "model_manifest_sha256": resolved.identity.model_manifest_sha256}
        if actual != expected["identity"]:
            raise CanaryRefused("worker model identity differs from parent")
        def before_child(_index: int, _order: list[str]) -> None:
            with sample_lock:
                if sampler_errors:
                    raise CanaryRefused("stage pre-child swap sampler error")
                highwater = max(samples) if samples else None
                live_event = safety_state.get("event")
            if isinstance(live_event, dict):
                raise CanaryRefused("stage pre-child safety abort")
            if (not isinstance(highwater, int)
                    or highwater - expected["initial_swap"] > SWAP_DELTA_LIMIT_BYTES):
                raise CanaryRefused("stage pre-child swap delta exceeded 128 MiB")
            env = system_environment(_deadline_runner(worker_deadline))
            env["loadavg"] = loadavg_gate(deadline=worker_deadline)
            env["competing_model_process"] = competing_model_process(_deadline_runner(worker_deadline))
            gate = _stage_gate(env, expected["initial_swap"], installed=expected["installed_memory"], peak=1, rss=1)
            if not all(gate["checks"][key] for key in ("swap_endpoint_known", "swap_delta_within_128mib", "memory_free_at_least_20_percent", "ac_power", "low_power_off", "thermal_nominal", "loadavg_gate", "no_competing_model_process")):
                raise CanaryRefused("stage pre-child resource gate failed")
        try:
            result = ab.run({stage: Knobs(**ARMS[stage])}, processes=1, repeats=REPEATS, warmup=WARMUP,
                             max_tokens=MAX_TOKENS, model=MODEL_ID, child_timeout_seconds=CHILD_TIMEOUT_SECONDS,
                             before_child=before_child)
        finally:
            stop_monitor.set()
            sampler.join(timeout=2)
            if sampler.is_alive():
                record_error("sampler-join", CanaryRefused("sampler thread did not stop"))
            final_error: BaseException | None = None
            try:
                synchronous_sample("worker-final", _deadline_runner(worker_deadline))
            except BaseException as final_exc:
                # synchronous_sample already recorded the command/read/parse
                # failure; retaining the exception here makes the failure
                # path explicit and keeps it hard-failing below.
                final_error = final_exc
                record_error("worker-final-unexpected", final_exc)
            _finalize_stage_safety(
                initial_swap=expected["initial_swap"], samples=samples,
                sample_times=sample_times, sample_offsets=sample_offsets,
                sampler_errors=sampler_errors, lock=sample_lock,
                state=safety_state, final_error=final_error,
                marker_writer=marker_writer, kill_group=kill_group,
            )
        if isinstance(safety_state.get("event"), dict):
            raise CanaryRefused("live safety abort: " + str(safety_state["event"].get("reason", "unknown")))
        if sampler_errors:
            raise CanaryRefused("swap sampler failed: " + "; ".join(sampler_errors))
        if len(samples) < 2:
            raise CanaryRefused("swap sampler produced fewer than two samples")
        result["stage"] = stage
        result["binding"] = {**actual, "runtime_code_sha256": expected["runtime_code_sha256"]}
        result["child_rss_peak_bytes"] = q3a_helpers._max_rss_bytes()
        result["swap_samples"] = samples[:MAX_SWAP_SAMPLES]
        result["swap_sample_times"] = sample_times[:MAX_SWAP_SAMPLES]
        result["swap_sample_offsets"] = sample_offsets[:MAX_SWAP_SAMPLES]
        result["sampler_errors"] = []
        result["max_swap_used_bytes"] = max(samples)
        print("@@" + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except BaseException as exc:
        if stop_monitor is not None:
            stop_monitor.set()
        if sampler is not None:
            sampler.join(timeout=2)
        print("@@" + json.dumps({"failure": f"{type(exc).__name__}: {exc}",
                                  "partial_children": getattr(exc, "partial_children", []),
                                  "swap_samples": samples[:MAX_SWAP_SAMPLES],
                                  "swap_sample_times": sample_times[:MAX_SWAP_SAMPLES],
                                  "swap_sample_offsets": sample_offsets[:MAX_SWAP_SAMPLES],
                                  "sampler_errors": sampler_errors,
                                  "safety_event": safety_state.get("event")}, sort_keys=True), flush=True)
        return 2


def _parse_cleanup_ps_snapshot(output: Any) -> dict[str, Any]:
    """Parse the bounded PID/PPID/PGID/UID/status cleanup process view."""
    if not isinstance(output, str) or not getattr(output, "ok", True):
        return {"valid": False, "records": [], "error": "ps snapshot unavailable"}
    if len(output) > MAX_PS_OUTPUT:
        return {"valid": False, "records": [], "error": "ps snapshot exceeded bounded limit"}
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line in output.splitlines():
        if not line.strip():
            return {"valid": False, "records": [], "error": "ps snapshot contains a blank row"}
        parts = line.split(None, 7)
        if len(parts) != 8 or "\x00" in parts[7] or not parts[7].strip():
            return {"valid": False, "records": [], "error": "ps snapshot row malformed"}
        if any(re.fullmatch(r"[0-9]+", part) is None for part in parts[:5]):
            return {"valid": False, "records": [], "error": "ps snapshot integer malformed"}
        try:
            pid, ppid, pgid, sid, uid = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        except (TypeError, ValueError, OverflowError):
            return {"valid": False, "records": [], "error": "ps snapshot integer malformed"}
        stat, start = parts[5].strip(), parts[6].strip()
        if (pid <= 0 or ppid < 0 or pgid < 0 or sid < 0 or uid < 0 or pid in seen
                or not stat or any(char.isspace() for char in stat)
                or CLEANUP_STAT_RE.fullmatch(stat) is None
                or CLEANUP_START_RE.fullmatch(start) is None
                or any(ord(char) < 32 for char in start)):
            return {"valid": False, "records": [], "error": "ps snapshot identity malformed"}
        seen.add(pid)
        records.append({"pid": pid, "ppid": ppid, "pgid": pgid, "sid": sid, "uid": uid,
                        "stat": stat, "start": start, "args": parts[7].strip()})
    # A complete snapshot must support ancestry reconstruction.  A missing
    # parent or a cycle is unknown process state, even if the target PGID is
    # absent; this is deliberately conservative for cleanup proof.
    by_pid = {row["pid"]: row for row in records}
    for row in records:
        seen_chain: set[int] = set()
        pid = row["pid"]
        while pid != 0:
            if pid in seen_chain:
                return {"valid": False, "records": [], "error": "ps snapshot ancestry cycle"}
            seen_chain.add(pid)
            parent = by_pid.get(pid)
            if parent is None:
                return {"valid": False, "records": [], "error": "ps snapshot parent link missing"}
            pid = parent["ppid"]
    return {"valid": True, "records": records, "error": None}


def _cleanup_ps_snapshot(run: Callable[[list[str]], str] = _run_text) -> dict[str, Any]:
    """Take one bounded cleanup snapshot, preserving command failure state."""
    stamp = time.monotonic()
    try:
        output = run(CLEANUP_PS_COMMAND)
    except BaseException as exc:
        return {"monotonic": stamp, "command_ok": False, "parse_ok": False,
                "records": [], "error": f"ps snapshot command: {type(exc).__name__}"}
    parsed = _parse_cleanup_ps_snapshot(output)
    return {"monotonic": stamp, "command_ok": bool(isinstance(output, str) and getattr(output, "ok", True)),
            "parse_ok": parsed["valid"], "records": parsed["records"], "error": parsed["error"]}


def _capture_process_baseline(run: Callable[[list[str]], str] = _run_text) -> dict[str, Any]:
    """Capture bounded stable PID/start/UID identities before a worker spawn."""
    snapshot = _cleanup_ps_snapshot(run)
    if snapshot.get("command_ok") is not True or snapshot.get("parse_ok") is not True:
        return {"valid": False, "digest": None, "identities": [], "error": "baseline snapshot unknown"}
    identities = [{"pid": row["pid"], "start": row["start"], "uid": row["uid"],
                   "sid": row["sid"], "pgid": row["pgid"]}
                  for row in snapshot["records"]]
    if len(identities) > MAX_BASELINE_IDENTITIES:
        return {"valid": False, "digest": None, "identities": [], "error": "baseline exceeded bounded limit"}
    encoded = json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    return {"valid": True, "digest": hashlib.sha256(encoded).hexdigest(), "identities": identities, "error": None}


def _capture_worker_identity(process: Any, *, run: Callable[[list[str]], str] = _run_text,
                             baseline: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Capture the worker PID/PGID/UID immediately after ``Popen``."""
    pid = getattr(process, "pid", None)
    if type(pid) is not int or pid <= 0:
        raise CanaryRefused("worker PID is malformed")
    owner_uid = os.getuid()
    if owner_uid <= 0:
        raise CanaryRefused("root execution is not permitted for worker cleanup")
    if (not isinstance(baseline, Mapping) or baseline.get("valid") is not True
            or not isinstance(baseline.get("identities"), list)
            or not isinstance(baseline.get("digest"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", baseline["digest"])):
        raise CanaryRefused("pre-spawn baseline is unavailable")
    snapshot = _cleanup_ps_snapshot(run)
    if not snapshot["parse_ok"] or not snapshot["command_ok"]:
        raise CanaryRefused("worker identity snapshot is unavailable")
    rows = [row for row in snapshot["records"] if row["pid"] == pid]
    if len(rows) != 1 or rows[0]["pgid"] != pid or rows[0]["sid"] != pid:
        raise CanaryRefused("worker identity or process-group leader mismatch")
    if rows[0]["uid"] != owner_uid:
        raise CanaryRefused("worker UID differs from current non-root UID")
    descendants = []
    by_pid = {row["pid"]: row for row in snapshot["records"]}
    for row in snapshot["records"]:
        cursor = row["pid"]
        chain: set[int] = set()
        while cursor:
            if cursor in chain or cursor not in by_pid:
                raise CanaryRefused("worker identity ancestry is malformed")
            chain.add(cursor)
            if cursor == pid:
                descendants.append(row["pid"])
                break
            cursor = by_pid[cursor]["ppid"]
    known = sorted(set(descendants))
    return {"worker_pid": pid, "parent_pid": rows[0]["ppid"], "pgid": rows[0]["pgid"],
            "sid": rows[0]["sid"], "uid": rows[0]["uid"], "known_descendant_pids": known,
            "known_process_starts": {str(row["pid"]): row["start"] for row in snapshot["records"]
                                     if row["pid"] in known},
            "known_process_sids": {str(row["pid"]): row["sid"] for row in snapshot["records"]
                                   if row["pid"] in known},
            "uid_invariant": {"owner_uid": owner_uid, "worker_uid": rows[0]["uid"],
                              "same_non_root": True},
            "spawn_baseline": {"valid": True, "digest": baseline["digest"],
                               "identities": baseline["identities"]}}


def _cleanup_worker_evidence(process: Any, identity: Mapping[str, Any] | None = None,
                             *, run: Callable[[list[str]], str] = _run_text,
                             global_inventory: bool = False) -> dict[str, Any]:
    """Terminate/reap a worker group and return fail-closed cleanup evidence v2."""
    pid = getattr(process, "pid", None)
    known_descendants = identity.get("known_descendant_pids", []) if isinstance(identity, Mapping) else []
    known_starts = identity.get("known_process_starts", {}) if isinstance(identity, Mapping) else {}
    spawn_baseline = identity.get("spawn_baseline", {}) if isinstance(identity, Mapping) else {}
    baseline_identities = spawn_baseline.get("identities", []) if isinstance(spawn_baseline, Mapping) else []
    valid_identity = (
        isinstance(identity, Mapping)
        and type(identity.get("worker_pid")) is int and identity["worker_pid"] > 0
        and type(identity.get("parent_pid")) is int and identity["parent_pid"] >= 0
        and type(identity.get("pgid")) is int and identity["pgid"] > 0
        and type(identity.get("sid")) is int and identity["sid"] > 0
        and type(identity.get("uid")) is int and identity["uid"] >= 0
        and identity["worker_pid"] == pid and identity["pgid"] == identity["worker_pid"]
        and identity["sid"] == identity["worker_pid"]
        and isinstance(known_descendants, list)
        and all(type(item) is int and item > 0 for item in known_descendants)
        and isinstance(known_starts, dict)
        and set(known_starts) == {str(item) for item in known_descendants}
        and all(isinstance(item, str) and CLEANUP_START_RE.fullmatch(item) is not None
                for item in known_starts.values())
        and isinstance(identity.get("known_process_sids"), dict)
        and set(identity["known_process_sids"]) == {str(item) for item in known_descendants}
        and all(type(item) is int and item == identity["sid"] for item in identity["known_process_sids"].values())
        and isinstance(identity.get("uid_invariant"), Mapping)
        and set(identity["uid_invariant"]) == {"owner_uid", "worker_uid", "same_non_root"}
        and type(identity["uid_invariant"]["owner_uid"]) is int and identity["uid_invariant"]["owner_uid"] > 0
        and identity["uid_invariant"]["owner_uid"] == os.getuid()
        and identity["uid_invariant"]["worker_uid"] == identity["uid"]
        and identity["uid_invariant"]["same_non_root"] is True
        and hashlib.sha256(json.dumps(baseline_identities, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == spawn_baseline["digest"]
        and isinstance(spawn_baseline, Mapping)
        and spawn_baseline.get("valid", True) is not False
        and isinstance(spawn_baseline.get("digest"), str)
        and re.fullmatch(r"[0-9a-f]{64}", spawn_baseline["digest"]) is not None
        and isinstance(baseline_identities, list) and len(baseline_identities) <= MAX_BASELINE_IDENTITIES
        and all(isinstance(item, Mapping) and set(item) == {"pid", "start", "uid", "sid", "pgid"}
                and type(item["pid"]) is int and item["pid"] > 0
                and isinstance(item["start"], str) and CLEANUP_START_RE.fullmatch(item["start"]) is not None
                and type(item["uid"]) is int and item["uid"] >= 0
                and type(item["sid"]) is int and item["sid"] >= 0
                and type(item["pgid"]) is int and item["pgid"] >= 0 for item in baseline_identities)
    )
    if valid_identity:
        exact_identity = {"worker_pid": identity["worker_pid"], "parent_pid": identity["parent_pid"],
                          "pgid": identity["pgid"], "sid": identity["sid"], "uid": identity["uid"],
                          "known_descendant_pids": sorted(set(known_descendants)),
                          "known_process_starts": {str(pid): known_starts[str(pid)]
                                                   for pid in sorted(set(known_descendants))},
                          "known_process_sids": {str(pid): identity["known_process_sids"][str(pid)]
                                                 for pid in sorted(set(known_descendants))},
                          "uid_invariant": dict(identity["uid_invariant"]),
                          "spawn_baseline": {"valid": True, "digest": spawn_baseline["digest"], "identities": baseline_identities}}
    else:
        exact_identity = {"worker_pid": pid if type(pid) is int and pid > 0 else None,
                          "parent_pid": identity.get("parent_pid") if isinstance(identity, Mapping) else None,
                          "pgid": identity.get("pgid") if isinstance(identity, Mapping) else None,
                          "sid": identity.get("sid") if isinstance(identity, Mapping) else None,
                          "uid": identity.get("uid") if isinstance(identity, Mapping) else None,
                          "known_descendant_pids": known_descendants if isinstance(known_descendants, list) else [],
                          "known_process_starts": known_starts if isinstance(known_starts, dict) else {},
                          "known_process_sids": identity.get("known_process_sids", {}) if isinstance(identity, Mapping) else {},
                          "uid_invariant": identity.get("uid_invariant", {}) if isinstance(identity, Mapping) else {},
                          "spawn_baseline": spawn_baseline if isinstance(spawn_baseline, Mapping) else {}}
    pgid = exact_identity["pgid"]
    signal_attempts: list[dict[str, Any]] = []

    def group_safe(snapshot: Mapping[str, Any], *, require_leader: bool = True,
                   reference_starts: Mapping[int, str] | None = None) -> bool:
        if not (valid_identity and snapshot.get("command_ok") is True and snapshot.get("parse_ok") is True):
            return False
        rows = snapshot.get("records", [])
        leader = [row for row in rows if row.get("pid") == pid]
        if require_leader and (len(leader) != 1 or leader[0].get("pgid") != pgid
                                or leader[0].get("sid") != exact_identity["sid"]
                                or leader[0].get("uid") != exact_identity["uid"]
                                or leader[0].get("start") != known_starts.get(str(pid))):
            return False
        members_now = [row for row in rows if row.get("pgid") == pgid]
        if any(row.get("uid") != exact_identity["uid"] for row in members_now):
            return False
        if reference_starts is not None:
            if any(row.get("pid") not in reference_starts for row in members_now):
                return False
            if any(row.get("start") != reference_starts.get(row.get("pid")) for row in members_now):
                return False
        for row in rows:
            row_pid = row.get("pid")
            if row_pid in known_descendants:
                if (row.get("start") != known_starts.get(str(row_pid))
                        or row.get("sid") != exact_identity["sid"]):
                    return False
        return True

    def attempt(name: str, sig: int) -> None:
        status = "sent"
        error = None
        if not valid_identity:
            status, error = "not_attempted", "worker identity is invalid"
        else:
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                status = "not_found"
            except PermissionError as exc:
                status, error = "permission_error", str(exc)[:256]
            except OSError as exc:
                status, error = "os_error", f"{type(exc).__name__}: {exc}"[:256]
        record: dict[str, Any] = {"signal": name, "status": status}
        if error is not None:
            record["error"] = error
        signal_attempts.append(record)

    pre_signal_snapshot = _cleanup_ps_snapshot(run)
    pre_signal_rows = pre_signal_snapshot.get("records", []) if pre_signal_snapshot.get("parse_ok") is True else []
    pre_signal_starts = {row["pid"]: row["start"] for row in pre_signal_rows if row.get("pgid") == pgid}
    if group_safe(pre_signal_snapshot):
        attempt("SIGTERM", signal.SIGTERM)
    elif (valid_identity and pre_signal_snapshot.get("command_ok") is True
          and pre_signal_snapshot.get("parse_ok") is True
          and not [row for row in pre_signal_snapshot.get("records", []) if row.get("pgid") == pgid]):
        signal_attempts.append({"signal": "SIGTERM", "status": "not_needed_group_already_gone"})
    else:
        signal_attempts.append({"signal": "SIGTERM", "status": "not_attempted",
                                "error": "fresh pre-signal group identity was not proven"})
    worker_reaped = False
    wait_errors: list[str] = []
    try:
        process.wait(timeout=2)
        worker_reaped = process.poll() is not None
    except subprocess.TimeoutExpired:
        wait_errors.append("wait:TimeoutExpired")
    except (OSError, ValueError) as exc:
        wait_errors.append(f"wait:{type(exc).__name__}")
    if not worker_reaped:
        attempt("SIGKILL", signal.SIGKILL)
        try:
            process.wait(timeout=2)
            worker_reaped = process.poll() is not None
        except subprocess.TimeoutExpired:
            wait_errors.append("reap:TimeoutExpired")
        except (OSError, ValueError) as exc:
            wait_errors.append(f"reap:{type(exc).__name__}")
    try:
        process.communicate(timeout=2)
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        wait_errors.append(f"communicate:{type(exc).__name__}")

    # The first post-TERM snapshot decides whether escalation is needed.  A
    # surviving member must be killed even when the group leader was already
    # reaped; otherwise an escaped child could be left behind.
    snapshots = [_cleanup_ps_snapshot(run)]
    pre_escalation_snapshot = snapshots[0]
    first_rows = snapshots[0].get("records", []) if snapshots[0].get("parse_ok") is True else []
    first_members = [row for row in first_rows if valid_identity and row.get("pgid") == pgid]
    first_descendants = [row for row in first_rows if valid_identity and row.get("pid") in known_descendants]
    start_mismatch = any(valid_identity and row.get("pid") in known_descendants
                         and row.get("start") != known_starts.get(str(row.get("pid")))
                         for row in first_rows)
    if valid_identity and snapshots[0].get("command_ok") is True and snapshots[0].get("parse_ok") is True \
            and (first_members or first_descendants) and not start_mismatch:
        descendant_kill_attempts: list[dict[str, Any]] = []
        if group_safe(snapshots[0], require_leader=False, reference_starts=pre_signal_starts):
            attempt("SIGKILL", signal.SIGKILL)
        for row in first_descendants:
            if row.get("pid") == pid or row.get("start") != known_starts.get(str(row.get("pid"))) \
                    or row.get("sid") != exact_identity["sid"]:
                continue
            attempt_record: dict[str, Any] = {"pid": row["pid"], "signal": "SIGKILL", "status": "sent"}
            try:
                os.kill(row["pid"], signal.SIGKILL)
            except ProcessLookupError:
                attempt_record["status"] = "not_found"
            except PermissionError as exc:
                attempt_record["status"] = "permission_error"; attempt_record["error"] = str(exc)[:256]
            except OSError as exc:
                attempt_record["status"] = "os_error"; attempt_record["error"] = f"{type(exc).__name__}: {exc}"[:256]
            descendant_kill_attempts.append(attempt_record)
        try:
            process.wait(timeout=2)
            worker_reaped = worker_reaped or process.poll() is not None
        except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
            wait_errors.append(f"reap-after-kill:{type(exc).__name__}")
        try:
            process.communicate(timeout=2)
        except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
            wait_errors.append(f"communicate-after-kill:{type(exc).__name__}")
        # The pre-escalation snapshot explains why KILL was needed; two fresh
        # snapshots are required for the final proof.
        snapshots = [_cleanup_ps_snapshot(run), _cleanup_ps_snapshot(run)]
    else:
        descendant_kill_attempts = []
        second_snapshot = _cleanup_ps_snapshot(run)
        # If the first post-TERM inventory is unknown, do not silently leave
        # known descendants behind.  A second valid snapshot can still prove
        # a captured PID/start identity before a direct PID kill.  The first
        # unknown snapshot remains in the final evidence, so the gate can never
        # become a false PASS.
        if (valid_identity and worker_reaped
                and snapshots[0].get("parse_ok") is not True
                and second_snapshot.get("parse_ok") is True):
            second_rows = second_snapshot.get("records", [])
            for row in second_rows:
                row_pid = row.get("pid")
                if (row_pid == pid or row_pid not in known_descendants
                        or row.get("start") != known_starts.get(str(row_pid))):
                    continue
                attempt_record = {"pid": row_pid, "signal": "SIGKILL", "status": "sent"}
                try:
                    os.kill(row_pid, signal.SIGKILL)
                except ProcessLookupError:
                    attempt_record["status"] = "not_found"
                except PermissionError as exc:
                    attempt_record["status"] = "permission_error"; attempt_record["error"] = str(exc)[:256]
                except OSError as exc:
                    attempt_record["status"] = "os_error"; attempt_record["error"] = f"{type(exc).__name__}: {exc}"[:256]
                descendant_kill_attempts.append(attempt_record)
            if descendant_kill_attempts:
                try:
                    process.communicate(timeout=2)
                except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
                    wait_errors.append(f"communicate-after-orphan-kill:{type(exc).__name__}")
                snapshots = [_cleanup_ps_snapshot(run), _cleanup_ps_snapshot(run)]
            else:
                snapshots.extend([second_snapshot])
        else:
            snapshots.append(second_snapshot)
    members: list[list[dict[str, Any]]] = []
    leaders: list[list[dict[str, Any]]] = []
    descendants: list[list[dict[str, Any]]] = []
    new_processes: list[list[dict[str, Any]]] = []
    baseline_pairs = {(item.get("pid"), item.get("start"), item.get("uid"), item.get("sid"), item.get("pgid")) for item in baseline_identities}
    for snapshot in snapshots:
        rows = snapshot.get("records", []) if snapshot.get("parse_ok") is True else []
        members.append([row for row in rows if valid_identity and row.get("pgid") == pgid])
        leaders.append([row for row in rows if valid_identity and row.get("pid") == pid])
        descendants.append([row for row in rows if valid_identity and row.get("pid") in known_descendants])
        new_processes.append([{"pid": row["pid"], "start": row["start"], "uid": row["uid"],
                              "sid": row["sid"], "pgid": row["pgid"]}
                             for row in rows if valid_identity and row.get("uid") == exact_identity["uid"]
                             and (row.get("pid"), row.get("start"), row.get("uid"), row.get("sid"), row.get("pgid")) not in baseline_pairs
                             and not (row.get("pid") in known_descendants
                                      and row.get("start") == known_starts.get(str(row.get("pid")))
                                      and row.get("sid") == exact_identity["sid"])])
    snapshots_valid = (len(snapshots) == 2
                       and all(snapshot.get("command_ok") is True and snapshot.get("parse_ok") is True
                               and isinstance(snapshot.get("monotonic"), (int, float))
                               and math.isfinite(float(snapshot["monotonic"])) for snapshot in snapshots)
                       and snapshots[1]["monotonic"] >= snapshots[0]["monotonic"])
    pre_escalation_known = pre_escalation_snapshot.get("command_ok") is True and pre_escalation_snapshot.get("parse_ok") is True
    members_empty = snapshots_valid and all(not group for group in members)
    descendants_empty = snapshots_valid and all(not group for group in descendants)
    new_processes_empty = snapshots_valid and all(not group for group in new_processes)
    leaders_absent = snapshots_valid and all(not leader for leader in leaders)
    non_zombie = snapshots_valid and all(not row.get("stat", "").startswith("Z") for row in sum(members + descendants + leaders, []))
    pre_signal_known = pre_signal_snapshot.get("command_ok") is True and pre_signal_snapshot.get("parse_ok") is True
    global_inventory_evidence = {"enabled": global_inventory, "known": True, "competing": None}
    if global_inventory:
        try:
            competing = competing_model_process(run)
            global_inventory_evidence["competing"] = competing
            if competing is not None:
                global_inventory_evidence["known"] = True
                unresolved_competing = f"global process inventory: {competing}"
            else:
                unresolved_competing = None
        except BaseException as exc:
            global_inventory_evidence = {"enabled": True, "known": False, "competing": None,
                                         "error": f"{type(exc).__name__}"}
            unresolved_competing = "global process inventory unknown"
    else:
        unresolved_competing = None
    group_gone = bool(valid_identity and worker_reaped and pre_signal_known and pre_escalation_known and snapshots_valid
                      and members_empty and descendants_empty and new_processes_empty and leaders_absent and non_zombie
                      and global_inventory_evidence["known"] and global_inventory_evidence["competing"] is None)
    unresolved: list[str] = []
    resolved: list[str] = []
    if not valid_identity:
        unresolved.append("worker identity invalid")
    if not worker_reaped:
        unresolved.append("worker leader was not reaped")
    if not snapshots_valid:
        unresolved.append("cleanup verification snapshot unknown or malformed")
    if pre_escalation_snapshot.get("parse_ok") is not True or pre_escalation_snapshot.get("command_ok") is not True:
        unresolved.append("pre-cleanup verification snapshot unknown or malformed")
    if not pre_signal_known:
        unresolved.append("pre-signal group verification snapshot unknown or malformed")
    if unresolved_competing is not None:
        unresolved.append(unresolved_competing)
    if snapshots_valid and not members_empty:
        unresolved.append("worker process group still has members")
    if snapshots_valid and not descendants_empty:
        unresolved.append("worker descendant still appears after cleanup")
    if snapshots_valid and not new_processes_empty:
        unresolved.append("new same-UID process appeared after worker spawn baseline")
    if snapshots_valid and not leaders_absent:
        unresolved.append("worker leader still appears in cleanup snapshot")
    if snapshots_valid and not non_zombie:
        unresolved.append("zombie worker or descendant remains")
    for snapshot in snapshots:
        rows = snapshot.get("records", []) if snapshot.get("parse_ok") is True else []
        if any(valid_identity and row.get("pid") in known_descendants
               and row.get("start") != known_starts.get(str(row.get("pid"))) for row in rows):
            unresolved.append("known PID start identity changed or was reused")
    if wait_errors:
        unresolved.extend(wait_errors)
    for attempt_record in descendant_kill_attempts:
        if attempt_record["status"] in {"permission_error", "os_error"}:
            (resolved if group_gone else unresolved).append(
                f"PID{attempt_record['pid']}:{attempt_record['status']}")
    for attempt_record in signal_attempts:
        if attempt_record["status"] in {"permission_error", "os_error"}:
            message = f"{attempt_record['signal']}:{attempt_record['status']}"
            (resolved if group_gone else unresolved).append(message)
    if group_gone and not unresolved:
        resolved.extend([f"{record['signal']}:{record['status']}" for record in signal_attempts if record["status"] in {"not_found", "not_attempted"}])
    verification = {
        "method": "two_independent_ps_snapshots",
        "pre_signal_snapshot": pre_signal_snapshot,
        "pre_escalation_snapshot": pre_escalation_snapshot,
        "snapshots": snapshots,
        "members": members,
        "leader": leaders,
        "descendants": descendants,
        "new_processes": new_processes,
        "snapshot_count": len(snapshots),
        "snapshot_gap_seconds": snapshots[1]["monotonic"] - snapshots[0]["monotonic"] if snapshots_valid else None,
        "independent": snapshots_valid and len(snapshots) == 2,
        "group_gone": group_gone,
        "global_process_inventory": global_inventory_evidence,
    }
    return {
        "schema": CLEANUP_EVIDENCE_SCHEMA,
        "identity": exact_identity,
        "worker_reaped": worker_reaped,
        "signal_attempts": signal_attempts,
        "descendant_kill_attempts": descendant_kill_attempts,
        "verification": verification,
        "resolved_errors": resolved,
        "unresolved_errors": unresolved,
    }


def _cleanup_worker(process: subprocess.Popen[str]) -> list[str]:
    errors = []
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    except OSError as exc: errors.append(f"SIGTERM:{type(exc).__name__}")
    try: process.wait(timeout=2)
    except subprocess.TimeoutExpired: pass
    except OSError as exc: errors.append(f"wait:{type(exc).__name__}")
    def group_gone() -> bool:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        return False
    if not group_gone():
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        except OSError as exc: errors.append(f"SIGKILL:{type(exc).__name__}")
    try: process.wait(timeout=2)
    except (subprocess.TimeoutExpired, OSError) as exc: errors.append(f"reap:{type(exc).__name__}")
    try: process.communicate(timeout=2)
    except (subprocess.TimeoutExpired, OSError) as exc: errors.append(f"communicate:{type(exc).__name__}")
    if process.poll() is None or not group_gone(): errors.append("worker process group still alive")
    return errors


def _start_stage(stage: str, identity: dict[str, Any], initial_swap: int, installed_memory: int, deadline: float) -> tuple[dict[str, Any], list[str]]:
    nonce = secrets.token_urlsafe(24)
    read_fd, write_fd = os.pipe()
    expected = {"identity": identity, "runtime_code_sha256": runtime_code_sha256(), "stage": stage,
                "initial_swap": initial_swap, "installed_memory": installed_memory}
    payload = json.dumps({"nonce": nonce, "expected": expected}, sort_keys=True, separators=(",", ":")).encode()
    if len(payload) > CAPABILITY_MAX_BYTES:
        os.close(read_fd); os.close(write_fd); raise CanaryRefused("capability payload exceeds bound")
    os.set_inheritable(read_fd, True)
    try:
        os.write(write_fd, payload); os.close(write_fd); write_fd = -1
        remaining = deadline - time.monotonic()
        timeout = min(float(WORKER_MAX_SECONDS), remaining - POST_STAGE_RESERVE_SECONDS)
        if timeout <= 0: raise CanaryRefused("global deadline exhausted")
        env = {**os.environ, "IRONMULE_Q3B_CAP_FD": str(read_fd), "IRONMULE_Q3B_CAP_NONCE": nonce,
               "IRONMULE_Q3B_EXPECTED": json.dumps(expected, sort_keys=True, separators=(",", ":")),
               "IRONMULE_Q3B_WORKER_DEADLINE": str(time.monotonic() + timeout)}
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--stage-worker"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                                   pass_fds=(read_fd,), start_new_session=True,
                                   cwd=str(Path(__file__).resolve().parents[1]))
    except (OSError, ValueError) as exc:
        raise CanaryRefused("stage worker could not be started") from exc
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        errors = _cleanup_worker(process)
        reason = "stage worker timeout" if isinstance(exc, subprocess.TimeoutExpired) else "stage worker communication failed"
        return {"failure": reason + ("; " + "; ".join(errors) if errors else ""),
                "cleanup_errors": errors, "group_gone": not errors}, []
    if len(stdout) > MAX_WORKER_OUTPUT or len(stderr) > MAX_WORKER_OUTPUT:
        errors = _cleanup_worker(process)
        return {"failure": "stage worker output exceeded bounded limit",
                "cleanup_errors": errors, "group_gone": not errors}, []

    safety_event = None
    safety_json_error = None
    marker = None
    for line in stdout.splitlines():
        if line.startswith("@SAFETY"):
            raw = line[len("@SAFETY"):].lstrip()
            try:
                candidate = json.loads(raw)
                if isinstance(candidate, dict) and safety_event is None:
                    safety_event = candidate
            except (TypeError, ValueError, json.JSONDecodeError):
                safety_json_error = "stage worker safety JSON is malformed"
        elif line.startswith("@@") and marker is None:
            marker = line[2:]

    result: dict[str, Any]
    if marker is None:
        result = {"failure": safety_json_error or
                  f"stage worker returned no result (status {process.returncode})"}
    else:
        try:
            parsed = json.loads(marker)
            result = parsed if isinstance(parsed, dict) else {"failure": "stage worker result JSON is malformed"}
        except (TypeError, ValueError, json.JSONDecodeError):
            result = {"failure": "stage worker result JSON is malformed"}
    if safety_event is not None:
        # A live safety marker is terminal even if a late @@ failure/success
        # marker was also flushed before the worker exited.
        partial = {key: safety_event.get(key) for key in ("samples", "times", "offsets", "errors")}
        result = {"failure": "stage worker live safety abort: " + str(safety_event.get("reason", "unknown")),
                  "safety_event": safety_event, "partial_evidence": partial}
    elif safety_json_error is not None:
        result = {"failure": safety_json_error}
    elif process.returncode != 0 and "failure" not in result:
        result = {"failure": f"stage worker exited with status {process.returncode}", **result}
    if process.returncode != 0 or marker is None or safety_event is not None or "failure" in result:
        errors = _cleanup_worker(process)
        result["cleanup_errors"] = errors
        result["group_gone"] = not errors
    return result, []


def _integer_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value)


def validate_stage_result(result: Any, stage: str) -> tuple[bool, str]:
    if not isinstance(result, dict) or set(result) != {"stage", "arms", "processes", "repeats", "warmup", "raw", "per_arm", "token_identity", "token_count_identity", "stop_reason_identity", "deterministic", "reference_tokens", "ratios", "binding", "child_rss_peak_bytes", "swap_samples", "swap_sample_times", "swap_sample_offsets", "sampler_errors", "max_swap_used_bytes"}:
        return False, "stage schema fields are not exact"
    if stage not in STAGES or result["stage"] != stage or result["arms"] != {stage: ARMS[stage]} or result["processes"] != 1 or result["repeats"] != REPEATS or result["warmup"] != WARMUP or result["ratios"] != {}:
        return False, "stage identity or counts are not exact"
    if not isinstance(result["raw"], list) or len(result["raw"]) != 1:
        return False, "stage raw record is incomplete"
    child = result["raw"][0]
    if not isinstance(child, dict) or set(child) != {"pid", "arms", "order", "mlx_peak_bytes"} or child["order"] != [stage] or set(child["arms"]) != {stage} or not isinstance(child["pid"], int) or child["pid"] <= 0:
        return False, "stage child identity is malformed"
    arm = child["arms"][stage]
    expected = {"total_ns", "prefill_ns", "decode_ns", "logical_tokens", "logical_tokens_per_repeat", "physical_tokens_per_repeat", "token_counts", "stop_reasons", "capacities", "deterministic", "decode_steps", "prompt_tokens", "mlx_peak_bytes"}
    if not isinstance(arm, dict) or set(arm) != expected:
        return False, "stage arm fields are incomplete"
    if any(not isinstance(arm[key], list) or len(arm[key]) != REPEATS for key in ("total_ns", "prefill_ns", "decode_ns", "logical_tokens_per_repeat", "physical_tokens_per_repeat", "token_counts", "stop_reasons", "capacities")):
        return False, "stage repeats are incomplete"
    summary = result["per_arm"]
    if (not isinstance(summary, dict) or set(summary) != {stage}
            or not isinstance(summary[stage], dict)
            or set(summary[stage]) != {"total_ns", "prefill_ns", "decode_ns"}):
        return False, "stage summary is malformed"
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0 for key in ("total_ns", "prefill_ns", "decode_ns") for value in arm[key]):
        return False, "stage timings are invalid"
    if not _integer_list(arm["logical_tokens"]) or any(not _integer_list(item) or not item for item in arm["logical_tokens_per_repeat"] + arm["physical_tokens_per_repeat"]):
        return False, "stage token arrays are invalid"
    for logical, physical, count in zip(arm["logical_tokens_per_repeat"], arm["physical_tokens_per_repeat"], arm["token_counts"]):
        if not isinstance(count, dict) or set(count) != {"logical", "physical"} or count != {"logical": len(logical), "physical": len(physical)}:
            return False, "stage token counts are invalid"
    if arm["logical_tokens"] != arm["logical_tokens_per_repeat"][0] or any(value not in {"eos", "length"} for value in arm["stop_reasons"]):
        return False, "stage logical tokens or stop reasons are invalid"
    if (not isinstance(arm["decode_steps"], int) or isinstance(arm["decode_steps"], bool)
            or arm["decode_steps"] != len(arm["physical_tokens_per_repeat"][0]) - 1
            or not isinstance(arm["prompt_tokens"], int) or isinstance(arm["prompt_tokens"], bool)
            or arm["prompt_tokens"] < 0 or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in arm["capacities"])):
        return False, "stage capacity metadata is invalid"
    if any(item != arm["logical_tokens_per_repeat"][0] for item in arm["logical_tokens_per_repeat"]) or any(item != arm["physical_tokens_per_repeat"][0] for item in arm["physical_tokens_per_repeat"]):
        return False, "stage output is nondeterministic"
    if any(item != arm["token_counts"][0] for item in arm["token_counts"]) or any(item != arm["stop_reasons"][0] for item in arm["stop_reasons"]) or any(item != arm["capacities"][0] for item in arm["capacities"]):
        return False, "stage identity is nondeterministic"
    for metric in ("total_ns", "prefill_ns", "decode_ns"):
        value = float(statistics.median(arm[metric]))
        expected_summary = {"n": 1, "median": value, "min": value, "max": value, "p95": value, "stdev": 0.0}
        if summary[stage][metric] != expected_summary:
            return False, "stage timing summary does not match raw"
    if arm["deterministic"] is not True or result["token_identity"] is not True or result["token_count_identity"] is not True or result["stop_reason_identity"] is not True or result["deterministic"] is not True:
        return False, "stage identity flags failed"
    sample_times = result["swap_sample_times"]
    sample_offsets = result["swap_sample_offsets"]
    samples = result["swap_samples"]
    if (result["reference_tokens"] != arm["logical_tokens"] or not isinstance(result["child_rss_peak_bytes"], int)
            or result["child_rss_peak_bytes"] <= 0 or not isinstance(child["mlx_peak_bytes"], int)
            or child["mlx_peak_bytes"] <= 0 or not isinstance(samples, list)
            or not isinstance(sample_times, list) or not isinstance(sample_offsets, list)
            or len(samples) < 2 or len(samples) > MAX_SWAP_SAMPLES
            or len(sample_times) != len(samples) or len(sample_offsets) != len(samples)
            or result["sampler_errors"] != []
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in samples)
            or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for value in sample_times + sample_offsets)
            or any(sample_times[index] > sample_times[index + 1] for index in range(len(sample_times) - 1))
            or any(sample_offsets[index] > sample_offsets[index + 1] for index in range(len(sample_offsets) - 1))
            or sample_offsets[0] < 0 or sample_times[-1] != max(sample_times)
            or sample_offsets[-1] != max(sample_offsets)
            or any(sample_times[index + 1] - sample_times[index] > MAX_SWAP_SAMPLE_GAP_SECONDS for index in range(len(sample_times) - 1))
            or any(sample_offsets[index + 1] - sample_offsets[index] > MAX_SWAP_SAMPLE_GAP_SECONDS for index in range(len(sample_offsets) - 1))
            or result["max_swap_used_bytes"] != max(samples)
            or not isinstance(result["max_swap_used_bytes"], int)
            or isinstance(result["max_swap_used_bytes"], bool)):
        return False, "stage resource or reference evidence is invalid"
    binding = result["binding"]
    if (not isinstance(binding, dict) or set(binding) != {"model_id", "model_revision", "model_manifest_sha256", "runtime_code_sha256"}
            or binding["model_id"] != MODEL_ID or binding["model_revision"] != EXPECTED_REVISION
            or not re.fullmatch(r"[0-9a-f]{64}", str(binding["model_manifest_sha256"])) or not re.fullmatch(r"[0-9a-f]{64}", str(binding["runtime_code_sha256"]))):
        return False, "stage model or runtime binding is invalid"
    return True, "ok"


def cross_stage_identity(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, bool]:
    """Recompute every workload/output identity fact from both validated arms."""
    base_arm = baseline["raw"][0]["arms"]["baseline"]
    cand_arm = candidate["raw"][0]["arms"]["candidate"]
    return {
        "logical_tokens": base_arm["logical_tokens"] == cand_arm["logical_tokens"],
        "logical_tokens_per_repeat": base_arm["logical_tokens_per_repeat"] == cand_arm["logical_tokens_per_repeat"],
        "physical_tokens_per_repeat": base_arm["physical_tokens_per_repeat"] == cand_arm["physical_tokens_per_repeat"],
        "token_counts": base_arm["token_counts"] == cand_arm["token_counts"],
        "stop_reasons": base_arm["stop_reasons"] == cand_arm["stop_reasons"],
        "capacities": base_arm["capacities"] == cand_arm["capacities"],
        "decode_steps": base_arm["decode_steps"] == cand_arm["decode_steps"],
        "prompt_tokens": base_arm["prompt_tokens"] == cand_arm["prompt_tokens"],
        "deterministic": (base_arm["deterministic"] is True and cand_arm["deterministic"] is True
                          and baseline["deterministic"] is True and candidate["deterministic"] is True),
    }


def descriptive_timing(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Build non-gating timing context from two complete, validated stages.

    This deliberately has no confidence interval, winner, or promotion field.
    It is descriptive only: the fixed baseline-first order is confounded, and
    the values must never affect the safety status or ``performance_valid``.
    """
    for stage, value in (("baseline", baseline), ("candidate", candidate)):
        valid, reason = validate_stage_result(value, stage)
        if not valid:
            raise ValueError(f"descriptive timing requires complete {stage}: {reason}")

    def metrics(result: dict[str, Any], stage: str) -> dict[str, float | int]:
        arm = result["raw"][0]["arms"][stage]
        median_total_ns = float(statistics.median(arm["total_ns"]))
        median_prefill_ns = float(statistics.median(arm["prefill_ns"]))
        median_decode_ns = float(statistics.median(arm["decode_ns"]))
        logical = len(arm["logical_tokens_per_repeat"][0])
        physical = len(arm["physical_tokens_per_repeat"][0])
        decode_steps = arm["decode_steps"]
        total_output_tokens_per_s = logical / (median_total_ns / 1e9)
        decode_steps_per_s = decode_steps / (median_decode_ns / 1e9)
        values = (median_total_ns / 1e6, median_prefill_ns / 1e6,
                  median_decode_ns / 1e6, total_output_tokens_per_s,
                  decode_steps_per_s)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("descriptive timing formula is not finite")
        return {
            "median_total_ms": values[0], "median_prefill_ms": values[1],
            "median_decode_ms": values[2], "logical_output_tokens": logical,
            "physical_output_tokens": physical,
            "total_output_tokens_per_s": values[3], "decode_steps_per_s": values[4],
        }

    stage_metrics = {"baseline": metrics(baseline, "baseline"),
                     "candidate": metrics(candidate, "candidate")}
    base = stage_metrics["baseline"]
    cand = stage_metrics["candidate"]
    comparisons: dict[str, dict[str, float]] = {}
    for key in ("total_ms", "prefill_ms", "decode_ms",
                "total_output_tokens_per_s", "decode_steps_per_s"):
        base_key = f"median_{key}" if key.endswith("_ms") else key
        candidate_value = float(cand[base_key])
        baseline_value = float(base[base_key])
        ratio = candidate_value / baseline_value
        higher_is_better = key in ("total_output_tokens_per_s", "decode_steps_per_s")
        percent_faster = 100.0 * (ratio - 1.0) if higher_is_better else 100.0 * (1.0 - ratio)
        if not math.isfinite(ratio) or not math.isfinite(percent_faster):
            raise ValueError("descriptive timing comparison is not finite")
        comparisons[key] = {
            "ratio": ratio, "percent_faster": percent_faster,
            "direction": "higher_is_better" if higher_is_better else "lower_is_better",
        }
    return {
        "descriptive_only": True,
        "performance_valid": False,
        "order_confounded": True,
        "statistical_confidence": "none",
        "stages": stage_metrics,
        "comparison": {"candidate_over_baseline": comparisons},
    }


def _failure(plan: dict[str, Any], preflight_result: dict[str, Any], reason: str, stages: list[Any], resources: list[Any]) -> dict[str, Any]:
    return {"schema": "ironmule.q3b_result.v1", "experiment": EXPERIMENT_ID, "status": "FAILED", "fallback": "BASE",
            "promotion_allowed": False, "performance_valid": False, "interpretation": "SAFETY_ONLY",
            "reason": reason, "plan": plan, "preflight": preflight_result, "stages": stages, "resource_history": resources}


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream: stream.write(payload)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        try: os.close(descriptor)
        except OSError: pass
        raise


def _post_environment(deadline: float, prior_pid: int | None = None) -> dict[str, Any]:
    bounded = _deadline_runner(deadline)
    environment = system_environment(bounded)
    environment["loadavg"] = loadavg_gate(deadline=deadline)
    environment["competing_model_process"] = competing_model_process(
        bounded, absent_pids=(prior_pid,) if isinstance(prior_pid, int) else ())
    return environment


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.stage_worker:
        return _stage_worker()
    plan = {"model": MODEL_ID, "revision": EXPECTED_REVISION, "baseline": Q2_INCUMBENT,
            "candidate": Q3B_CANDIDATE, "stages": list(STAGES), "repeats": REPEATS,
            "warmup": WARMUP, "max_tokens": MAX_TOKENS, "child_timeout_seconds": CHILD_TIMEOUT_SECONDS,
            "worker_max_seconds": WORKER_MAX_SECONDS, "deadline_seconds": PILOT_DEADLINE_SECONDS,
            "swap_start_limit_bytes": START_SWAP_LIMIT_BYTES, "swap_delta_limit_bytes": SWAP_DELTA_LIMIT_BYTES,
            "swap_sample_interval_seconds": STAGE_SAMPLE_INTERVAL_SECONDS,
            "swap_sample_max_gap_seconds": MAX_SWAP_SAMPLE_GAP_SECONDS,
            "swap_sample_limit": MAX_SWAP_SAMPLES,
            "performance_valid": False, "promotion_allowed": False, "runtime_code_sha256": runtime_code_sha256()}
    if not args.execute:
        print(json.dumps({"schema": "ironmule.q3b_plan.v1", "experiment": EXPERIMENT_ID,
                          "estimated_wall_seconds": 2 * CHILD_TIMEOUT_SECONDS + 2 * POST_STAGE_RESERVE_SECONDS + POST_STAGE_RESERVE_SECONDS, "plan": plan}, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required with --execute")
    if not args.output.parent.is_dir() or os.path.lexists(args.output):
        print("q3b: output path must have an existing parent and must not already exist", file=sys.stderr)
        return 2
    deadline = started + PILOT_DEADLINE_SECONDS
    try:
        pre = preflight(deadline=deadline)
    except BaseException as exc:
        pre = {"passed": False, "checks": {}, "error": f"preflight failed: {type(exc).__name__}"}
    stages: list[Any] = []
    resources: list[Any] = []
    if not pre.get("passed"):
        result = _failure(plan, pre, "preflight gate failed; no model child started", stages, resources)
    else:
        initial_swap = pre["environment"]["swap_used_bytes"]
        installed = pre["installed_memory_bytes"]
        for stage in STAGES:
            if time.monotonic() >= deadline:
                result = _failure(plan, pre, "global deadline exhausted before next stage", stages, resources)
                break
            try:
                worker, _markers = _start_stage(stage, pre["identity"], initial_swap, installed, deadline)
            except BaseException as exc:
                stages.append({"stage": stage, "failure": f"stage start failed: {type(exc).__name__}"})
                result = _failure(plan, pre, stages[-1]["failure"], stages, resources)
                break
            if "failure" in worker:
                stages.append({"stage": stage, "worker_result": worker, "failure": worker["failure"]})
                result = _failure(plan, pre, worker["failure"], stages, resources)
                break
            valid, reason = validate_stage_result(worker, stage)
            if not valid:
                stages.append({"stage": stage, "raw_result": worker, "failure": reason})
                result = _failure(plan, pre, reason, stages, resources)
                break
            try:
                after = _post_environment(deadline, worker["raw"][0]["pid"])
            except BaseException as exc:
                stages.append({"stage": stage, "raw_result": worker, "failure": f"post-stage snapshot failed: {type(exc).__name__}"})
                result = _failure(plan, pre, stages[-1]["failure"], stages, resources)
                break
            child = worker["raw"][0]
            resource_gate = _stage_gate(after, initial_swap, peak=child["mlx_peak_bytes"], rss=worker["child_rss_peak_bytes"], installed=installed, max_swap_used_bytes=worker["max_swap_used_bytes"])
            resources.append({"stage": stage, "environment": after, "gate": resource_gate})
            stages.append(worker)
            if not resource_gate["passed"]:
                result = _failure(plan, pre, f"{stage} post-stage resource gate failed", stages, resources)
                break
        else:
            baseline, candidate = stages
            identities = cross_stage_identity(baseline, candidate)
            if not all(identities.values()):
                result = _failure(plan, pre, "baseline/candidate token identity mismatch", stages, resources)
            else:
                result = {"schema": "ironmule.q3b_result.v1", "experiment": EXPERIMENT_ID,
                          "status": "SAFETY_CANARY_PASS", "fallback": None, "promotion_allowed": False,
                          "performance_valid": False, "interpretation": "SAFETY_ONLY", "token_identity": identities,
                          "descriptive_timing": descriptive_timing(baseline, candidate),
                          "plan": plan, "preflight": pre, "stages": stages, "resource_history": resources}
    _write_exclusive(args.output, (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": result["status"], "output": str(args.output), "promotion_allowed": False, "performance_valid": False}, sort_keys=True))
    return 0 if result["status"] == "SAFETY_CANARY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
