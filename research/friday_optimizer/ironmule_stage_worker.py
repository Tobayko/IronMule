"""Bounded worker for one offline IronMule calibration/test stage.

The worker is deliberately a small, standard-library-only protocol boundary.  It
must be copied into an adapter-owned stage directory and invoked with the exact
``--spec-file stage_spec.json`` argv produced by :mod:`ironmule_adapter`.  No
caller supplied model path, flag, source, network setting, or profile pointer is
accepted here.

The first (and normal) execution path imports the public ``ironmule.tune``
function only after all identity and safety checks have passed.  The returned
profile is treated as untrusted evidence: a screening gain is never promoted to
the confirmation fields, and missing evidence can only produce an inconclusive
envelope.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping


SCHEMA = "friday.ironmule.stage-spec.v1"
RESULT_SCHEMA = "friday.ironmule.result.v1"
WORKER_VERSION = "1"
WORKER_FILENAME = "friday_ironmule_stage_worker.py"
SPEC_FILENAME = "stage_spec.json"
MAX_SPEC_BYTES = 64 * 1024
MAX_RESULT_BYTES = 256 * 1024
MAX_DEPTH = 12
MAX_ITEMS = 4096
MAX_STRING = 4096
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REGISTRY = re.compile(r"^[0-9a-f]{64}$")
_HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TTFT_CONTRACT = "engine_prefill_to_first_token"
TUNE_SEARCH_CONTRACT = {
    "schema": "ironmule.tune.search-contract.v1",
    "source_commit": "03e884cb28a05d090d20844460fc3afc8e738a91",
    "search": [
        {"name": "compiled_fixed_cache", "values": [True]},
        {"name": "fused_argmax", "values": [True]},
        {"name": "head_skip_prefill", "values": [True]},
        {"name": "prefill_into_fixed", "values": [True]},
        {"name": "readback_every", "values": [2, 4, 8]},
        {"name": "speculate_k", "values": [4]},
        {"name": "capacity_slack", "values": [128]},
        {"name": "wired_fraction", "values": [0.6]},
        {"name": "fuse_projections", "values": [True]},
    ],
    "knobs_defaults": {
        "fuse_projections": False, "compiled_fixed_cache": False,
        "fused_argmax": False, "head_skip_prefill": False,
        "prefill_into_fixed": False, "readback_every": 1,
        "speculate_k": 0, "speculate_ngram": 3, "capacity_slack": 0,
        "wired_fraction": 0.0,
    },
    "keep_if_ratio_below": 0.995,
    "confirm_processes": 6,
    "confirm_repeats": 7,
    "confirm_warmup": 2,
}
TUNE_SEARCH_CONTRACT_SHA256 = hashlib.sha256(json.dumps(TUNE_SEARCH_CONTRACT, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
_ALLOWED_STAGES = frozenset({"calibrate", "test"})
_ALLOWED_CANDIDATES = frozenset({"combined_core_profile"})
_SPEC_KEYS = frozenset({
    "schema", "stage", "candidate", "model", "workload", "expected",
    "limits", "session", "source_manifest",
})
_MODEL_KEYS = frozenset({
    "model_id", "revision", "manifest", "architecture", "quant_bits",
    "quant_group_size", "tokenizer",
})
_WORKLOAD_KEYS = frozenset({
    "prompt_family", "tokenizer", "generator", "context_bucket", "batch",
    "concurrency", "max_tokens", "greedy", "prompt_logprobs", "power_mode",
    "mode",
})
_EXPECTED_KEYS = frozenset({
    "commit", "source_digest", "registry_hash", "fingerprint", "worker_sha256",
    "tune_search_contract_sha256", "pythonpath_sha256",
})
_LIMIT_KEYS = frozenset({
    "max_seconds", "max_output_bytes", "max_rss_bytes", "max_peak_memory_bytes",
    "max_swap_delta_bytes", "ac_connected", "low_power", "processes", "repeats",
    "warmup", "ttft_contract",
})
_SESSION_KEYS = frozenset({"session_id"})
_MANIFEST_KEYS = frozenset({"relative_path", "sha256", "size_bytes"})
OFFLINE_ENV = {
    "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
    "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HTTP_PROXY": "", "HTTPS_PROXY": "", "ALL_PROXY": "",
    "http_proxy": "", "https_proxy": "", "all_proxy": "", "NO_PROXY": "*",
}


class WorkerError(ValueError):
    """Malformed, stale, or unsafe worker input."""


class WorkerTimeout(TimeoutError):
    """The worker's own hard timeout elapsed."""


def validate_hub_model_id(value: Any) -> str:
    """Accept only one cached Hub ``org/name`` identifier; never a local path."""
    if not isinstance(value, str) or not _HUB_ID.fullmatch(value) or value.split("/", 1)[0] in {".", ".."} or value.split("/", 1)[1] in {".", ".."}:
        raise WorkerError("unsupported_model_source")
    return value


class _BoundedCapture(io.StringIO):
    """Keep noisy tuner diagnostics from becoming an unbounded result buffer."""

    def __init__(self, limit: int = 64 * 1024) -> None:
        super().__init__()
        self.limit = limit
        self.truncated = False

    def write(self, value: str) -> int:  # type: ignore[override]
        if not isinstance(value, str):
            value = str(value)
        remaining = max(0, self.limit - self.tell())
        if len(value) > remaining:
            super().write(value[:remaining])
            self.truncated = True
            return len(value)
        return super().write(value)


def _bounded(value: Any, *, depth: int = 0) -> Any:
    """Validate JSON-compatible data before any semantic interpretation."""

    if depth > MAX_DEPTH:
        raise WorkerError("JSON depth exceeds bound")
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and len(value) > MAX_STRING:
            raise WorkerError("JSON string exceeds bound")
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 10**18:
            raise WorkerError("JSON integer exceeds bound")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WorkerError("JSON number is not finite")
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_ITEMS or any(not isinstance(k, str) for k in value):
            raise WorkerError("JSON object is unbounded")
        return {k: _bounded(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_ITEMS:
            raise WorkerError("JSON array is unbounded")
        return [_bounded(v, depth=depth + 1) for v in value]
    raise WorkerError("JSON contains a non-canonical value")


def _canonical(value: Any, *, max_bytes: int = MAX_RESULT_BYTES) -> bytes:
    value = _bounded(value)
    try:
        result = json.dumps(value, ensure_ascii=False, allow_nan=False,
                            sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise WorkerError("JSON is not canonical") from exc
    if len(result) > max_bytes:
        raise WorkerError("canonical JSON exceeds bound")
    return result


def _strict_load(raw: bytes, *, maximum: int = MAX_SPEC_BYTES) -> dict[str, Any]:
    if len(raw) > maximum:
        raise WorkerError("spec exceeds byte bound")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeError, json.JSONDecodeError, WorkerError) as exc:
        raise WorkerError("spec is not strict JSON") from exc
    if not isinstance(value, dict):
        raise WorkerError("spec root must be an object")
    return _bounded(value)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise WorkerError("duplicate JSON key")
        value[key] = item
    return value


def _strict_object(value: Any, keys: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise WorkerError(f"{name} fields are missing or unknown")
    return value


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise WorkerError(f"{name} is invalid")
    return value


def _digest(value: Any, name: str) -> str:
    value = _text(value, name, maximum=64)
    if not _SHA256.fullmatch(value):
        raise WorkerError(f"{name} is not a SHA-256 digest")
    return value


def _registry_hash(paths: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def _relative(value: Any, name: str) -> str:
    value = _text(value, name, maximum=512)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\x00" in value:
        raise WorkerError(f"{name} must be a relative path")
    return path.as_posix()


def _safe_file(path: Path, root: Path, name: str) -> tuple[bytes, os.stat_result]:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        info = path.lstat()
    except (OSError, ValueError) as exc:
        raise WorkerError(f"{name} is outside the stage") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise WorkerError(f"{name} is not a regular file")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise WorkerError(f"{name} cannot be read") from exc
    if len(raw) > 16 * 1024 * 1024:
        raise WorkerError(f"{name} exceeds file bound")
    return raw, info


def _validate_spec(value: Mapping[str, Any], *, stage_root: Path | None = None) -> dict[str, Any]:
    spec = _strict_object(value, _SPEC_KEYS, "stage spec")
    if spec["schema"] != SCHEMA:
        raise WorkerError("unsupported stage spec schema")
    stage = _text(spec["stage"], "stage", maximum=32)
    if stage not in _ALLOWED_STAGES:
        raise WorkerError("stage is not allowlisted")
    if _text(spec["candidate"], "candidate", maximum=64) not in _ALLOWED_CANDIDATES:
        raise WorkerError("candidate is not allowlisted")
    model = _strict_object(spec["model"], _MODEL_KEYS, "model")
    model_id = validate_hub_model_id(model["model_id"])
    for name in ("revision", "manifest", "architecture", "tokenizer"):
        _text(model[name], "model." + name, maximum=512)
    for name in ("quant_bits", "quant_group_size"):
        if isinstance(model[name], bool) or not isinstance(model[name], int) or model[name] <= 0:
            raise WorkerError("model quantisation is invalid")
    workload = _strict_object(spec["workload"], _WORKLOAD_KEYS, "workload")
    for name in ("prompt_family", "tokenizer", "generator", "context_bucket", "power_mode", "mode"):
        _text(workload[name], "workload." + name, maximum=256)
    for name in ("batch", "concurrency", "max_tokens"):
        if isinstance(workload[name], bool) or not isinstance(workload[name], int) or not 1 <= workload[name] <= 1_000_000:
            raise WorkerError("workload integer is invalid")
    for name in ("greedy", "prompt_logprobs"):
        if not isinstance(workload[name], bool):
            raise WorkerError("workload boolean is invalid")
    if workload["greedy"] is not True or workload["prompt_logprobs"] is not False:
        raise WorkerError("only deterministic greedy workload is allowed")
    expected = _strict_object(spec["expected"], _EXPECTED_KEYS, "expected")
    if not _COMMIT.fullmatch(_text(expected["commit"], "expected.commit", maximum=40)):
        raise WorkerError("expected commit is invalid")
    _digest(expected["source_digest"], "expected.source_digest")
    if not _REGISTRY.fullmatch(_text(expected["registry_hash"], "expected.registry_hash", maximum=64)):
        raise WorkerError("expected registry hash is invalid")
    _digest(expected["fingerprint"], "expected.fingerprint")
    _digest(expected["worker_sha256"], "expected.worker_sha256")
    if expected["tune_search_contract_sha256"] != TUNE_SEARCH_CONTRACT_SHA256:
        raise WorkerError("tune search contract hash is invalid")
    if not isinstance(expected["pythonpath_sha256"], str) or not _SHA256.fullmatch(expected["pythonpath_sha256"]):
        raise WorkerError("pythonpath hash is invalid")
    limits = _strict_object(spec["limits"], _LIMIT_KEYS, "limits")
    seconds = limits["max_seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(float(seconds)) or not 0 < float(seconds) <= 1800:
        raise WorkerError("max_seconds is invalid")
    for name, minimum, maximum in (
        ("max_output_bytes", 1024, MAX_RESULT_BYTES),
        ("max_rss_bytes", 1, 2**63 - 1),
        ("max_peak_memory_bytes", 1, 2**63 - 1),
        ("max_swap_delta_bytes", 0, 2**63 - 1),
    ):
        item = limits[name]
        if isinstance(item, bool) or not isinstance(item, int) or not minimum <= item <= maximum:
            raise WorkerError(f"limits.{name} is invalid")
    for name, minimum, maximum in (("processes", 3, 8), ("repeats", 1, 32), ("warmup", 0, 8)):
        item = limits[name]
        if isinstance(item, bool) or not isinstance(item, int) or not minimum <= item <= maximum:
            raise WorkerError(f"limits.{name} is invalid")
    if limits["ttft_contract"] != TTFT_CONTRACT:
        raise WorkerError("unsupported TTFT measurement contract")
    if limits["ac_connected"] is not True or limits["low_power"] is not False:
        raise WorkerError("power safety evidence is not ready")
    session = _strict_object(spec["session"], _SESSION_KEYS, "session")
    _text(session["session_id"], "session.session_id", maximum=256)
    if session["session_id"] == "pending":
        raise WorkerError("stage authorization is incomplete")
    manifest_raw = spec["source_manifest"]
    if not isinstance(manifest_raw, list) or not manifest_raw or len(manifest_raw) > 128:
        raise WorkerError("source manifest is invalid")
    manifest: list[dict[str, Any]] = []
    previous = ""
    for item in manifest_raw:
        row = _strict_object(item, _MANIFEST_KEYS, "source manifest entry")
        relative = _relative(row["relative_path"], "manifest.relative_path")
        if not relative or relative <= previous:
            raise WorkerError("source manifest is not sorted and unique")
        previous = relative
        _digest(row["sha256"], "manifest.sha256")
        size = row["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= 16 * 1024 * 1024:
            raise WorkerError("manifest.size_bytes is invalid")
        manifest.append(row)
    if _registry_hash(tuple(row["relative_path"] for row in manifest)) != expected["registry_hash"]:
        raise WorkerError("source registry does not match manifest")
    if stage_root is not None:
        _validate_stage_files(spec, stage_root, manifest)
    return spec


def _validate_stage_files(spec: Mapping[str, Any], root: Path, manifest: list[dict[str, Any]]) -> None:
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise WorkerError("stage root is invalid")
    rows = {row["relative_path"]: row for row in manifest}
    expected = set(rows) | {WORKER_FILENAME, SPEC_FILENAME}
    observed: set[str] = set()
    scanned = 0
    for path in root.rglob("*"):
        scanned += 1
        if scanned > MAX_ITEMS:
            raise WorkerError("stage file count exceeds bound")
        if path.is_symlink():
            raise WorkerError("stage contains a symlink")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if len(relative) > 512:
                raise WorkerError("stage path exceeds bound")
            observed.add(relative)
    if observed != expected:
        raise WorkerError("stage contains an unbound file")
    for relative, row in rows.items():
        raw, info = _safe_file(root / relative, root, "source file")
        if len(raw) != row["size_bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise WorkerError("source manifest hash mismatch")
    worker_raw, _ = _safe_file(root / WORKER_FILENAME, root, "worker source")
    if hashlib.sha256(worker_raw).hexdigest() != spec["expected"]["worker_sha256"]:
        raise WorkerError("worker source hash mismatch")
    spec_raw, _ = _safe_file(root / SPEC_FILENAME, root, "stage spec")
    if _canonical(spec) != spec_raw:
        raise WorkerError("stage spec bytes are not canonical")


def _verify_pythonpath(expected_sha256: str) -> None:
    value = os.environ.get("PYTHONPATH", "")
    if hashlib.sha256(value.encode("utf-8")).hexdigest() != expected_sha256:
        raise WorkerError("PYTHONPATH identity mismatch")
    if not value:
        return
    if os.pathsep in value or not os.path.isabs(value) or sys.path.count(value) != 1:
        raise WorkerError("PYTHONPATH is not the single bound purelib")
    path = Path(value)
    if not path.is_dir() or path.is_symlink():
        raise WorkerError("bound purelib is unavailable")


def _read_spec(argv: list[str]) -> dict[str, Any]:
    # ``tune`` and ``--no-confirm`` are fixed protocol words retained for
    # compatibility with the old stage display; they are not forwarded to the
    # IronMule CLI and cannot be combined with any other free flag.
    if argv[:1] == ["tune"]:
        if argv[1:2] == ["--no-confirm"]:
            argv = argv[2:]
        else:
            argv = argv[1:]
    if len(argv) != 2 or argv[0] not in {"--spec-file", "--spec-fd"}:
        raise WorkerError("worker argv is not the fixed protocol")
    selector, value = argv
    if selector == "--spec-file":
        if value != SPEC_FILENAME:
            raise WorkerError("spec path is not the fixed stage file")
        path = Path.cwd() / SPEC_FILENAME
        if not path.is_file() or path.is_symlink():
            raise WorkerError("fixed spec file is missing")
        try:
            if path.stat().st_size > MAX_SPEC_BYTES:
                raise WorkerError("spec exceeds byte bound")
        except OSError as exc:
            raise WorkerError("fixed spec file is unreadable") from exc
        raw = path.read_bytes()
    else:
        if not value.isdecimal() or len(value) > 4:
            raise WorkerError("spec fd is invalid")
        fd = int(value)
        if fd < 3 or fd > 1024:
            raise WorkerError("spec fd is outside the fixed bound")
        try:
            raw = os.read(fd, MAX_SPEC_BYTES + 1)
        except OSError as exc:
            raise WorkerError("spec fd cannot be read") from exc
    return _strict_load(raw)


def _read_swap_bytes() -> int | None:
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            timeout=2, check=False, shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    text = (result.stdout or b"").decode("ascii", "ignore")[:2048]
    match = re.search(r"used\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTP])", text, re.I)
    if not match:
        return None
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
    return int(float(match.group(1)) * units[match.group(2).upper()])


def _read_power_state() -> tuple[bool | None, bool | None]:
    """Read only fixed macOS power commands; unknown is not made safe."""

    if platform.system() != "Darwin":
        return None, None
    try:
        battery_result = subprocess.run(
            ["/usr/bin/pmset", "-g", "batt"], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            timeout=2, check=False, shell=False,
        )
        custom_result = subprocess.run(
            ["/usr/bin/pmset", "-g", "custom"], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            timeout=2, check=False, shell=False,
        )
        if battery_result.returncode != 0 or custom_result.returncode != 0:
            return None, None
        battery = (battery_result.stdout or b"").decode("ascii", "ignore")[:4096]
        custom = (custom_result.stdout or b"").decode("ascii", "ignore")[:4096]
    except (OSError, subprocess.SubprocessError):
        return None, None
    ac = None if not battery else ("AC Power" in battery or "AC attached" in battery)
    low = None
    if custom:
        low = bool(re.search(r"lowpowermode\s+1", custom, re.I))
    return ac, low


def _rss_bytes() -> int | None:
    try:
        self_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        children_peak = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    except (AttributeError, OSError, ValueError, TypeError):
        # Model work is delegated to ab.run() children.  A self-only reading
        # would make a large child invisible and falsely pass the RSS gate.
        return None
    if self_peak < 0 or children_peak < 0:
        return None
    multiplier = 1 if sys.platform == "darwin" else 1024
    return max(self_peak, children_peak) * multiplier


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return default
    return float(value)


def _phase_ratio(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or not all(key in value for key in ("median_ratio", "ci_low", "ci_high")):
        return None
    if any(_number(value.get(key), -1.0) <= 0 for key in ("median_ratio", "ci_low", "ci_high")):
        return None
    if _number(value["ci_low"]) > _number(value["ci_high"]):
        return None
    result = {"median_ratio": _number(value["median_ratio"]),
              "ci_low": _number(value["ci_low"]), "ci_high": _number(value["ci_high"])}
    pairs = value.get("pairs")
    if pairs is not None:
        if not isinstance(pairs, (list, tuple)) or not pairs or len(pairs) > MAX_ITEMS:
            return None
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or float(item) <= 0 for item in pairs):
            return None
        result["pairs"] = [float(item) for item in pairs]
        result["pair_count"] = len(pairs)
    return result


def _confirmation_details(value: Any) -> tuple[dict[str, Any], dict[str, Any], int] | None:
    if not isinstance(value, Mapping):
        return None
    raw = value.get("ratio")
    if raw is None and isinstance(value.get("ratios"), Mapping):
        raw = value["ratios"].get("candidate/baseline")
    if not isinstance(raw, Mapping):
        return None
    phases: dict[str, Any] = {}
    if isinstance(raw.get("total_ns"), Mapping):
        for phase in ("total_ns", "prefill_ns", "decode_ns"):
            parsed = _phase_ratio(raw.get(phase))
            if parsed is None:
                return None
            phases[phase] = parsed
    else:
        parsed = _phase_ratio(raw)
        if parsed is None:
            return None
        phases["total_ns"] = parsed
    total = phases["total_ns"]
    pairs = total.get("pairs", [])
    if not pairs and isinstance(value.get("pairs"), (list, tuple)):
        pairs = list(value["pairs"])
        total = dict(total)
        total["pairs"] = pairs
    count = len(pairs) if pairs else int(value.get("pair_count", 0)) if isinstance(value.get("pair_count"), int) and not isinstance(value.get("pair_count"), bool) else 0
    return total, phases, count


def _valid_timing(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0


def _valid_calibration_trials(trials: Any) -> tuple[bool, list[Any]]:
    if not isinstance(trials, (list, tuple)) or not trials or len(trials) > MAX_ITEMS:
        return False, []
    try:
        copied = _bounded(list(trials), depth=2)
    except WorkerError:
        return False, []
    for trial in copied:
        if not isinstance(trial, Mapping):
            return False, []
        disposition = trial.get("disposition")
        verdict = trial.get("verdict")
        if disposition == "unsupported":
            if verdict != "unsupported":
                return False, []
            continue
        if disposition == "accepted":
            if verdict != "kept":
                return False, []
        elif disposition == "rejected":
            if not isinstance(verdict, str) or not verdict.startswith("rejected:"):
                return False, []
        else:
            return False, []
        if not _valid_timing(trial.get("total_ns")) or not _valid_timing(trial.get("prefill_ns")) or not _valid_timing(trial.get("decode_ns")):
            return False, []
        if not _valid_timing(trial.get("ratio")):
            return False, []
    return True, copied


def _verify_tune_contract(tune_module: Any, runtime_module: Any) -> None:
    """Fail closed if the staged IronMule tuner is not the bound 03e884 API."""
    search = getattr(tune_module, "SEARCH", None)
    if not isinstance(search, (list, tuple)):
        raise WorkerError("tune SEARCH is unavailable")
    normalized = [{"name": item[0], "values": list(item[1])}
                  for item in search
                  if isinstance(item, (list, tuple)) and len(item) == 2]
    if len(normalized) != len(search) or normalized != TUNE_SEARCH_CONTRACT["search"]:
        raise WorkerError("tune SEARCH contract mismatch")
    baseline = getattr(runtime_module, "BASELINE", None)
    knobs_type = getattr(runtime_module, "Knobs", None)
    if not callable(knobs_type) or not hasattr(baseline, "as_dict"):
        raise WorkerError("runtime Knobs contract is unavailable")
    if knobs_type().as_dict() != TUNE_SEARCH_CONTRACT["knobs_defaults"] or baseline.as_dict() != TUNE_SEARCH_CONTRACT["knobs_defaults"]:
        raise WorkerError("runtime Knobs defaults mismatch")
    for name, expected in (("KEEP_IF_RATIO_BELOW", "keep_if_ratio_below"),
                           ("CONFIRM_PROCESSES", "confirm_processes"),
                           ("CONFIRM_REPEATS", "confirm_repeats")):
        if getattr(tune_module, name, None) != TUNE_SEARCH_CONTRACT[expected]:
            raise WorkerError(f"tune {name} mismatch")
    warmup = getattr(tune_module, "CONFIRM_WARMUP", None)
    if warmup is None:
        try:
            source = inspect.getsource(tune_module.confirm)
        except (OSError, TypeError, AttributeError):
            raise WorkerError("tune confirmation warmup cannot be verified")
        warmup = 2 if "warmup=2" in source.replace(" ", "") else None
    if warmup != TUNE_SEARCH_CONTRACT["confirm_warmup"]:
        raise WorkerError("tune confirmation warmup mismatch")


def _verify_model_identity(tune_module: Any, spec: Mapping[str, Any]) -> Mapping[str, Any]:
    resolver = getattr(tune_module, "resolve_local_model", None)
    if not callable(resolver):
        raise WorkerError("local model resolver is unavailable")
    try:
        resolved = resolver(spec["model"]["model_id"], revision=spec["model"]["revision"])
    except Exception as exc:
        raise WorkerError("local model identity resolution failed") from exc
    identity = getattr(resolved, "identity", None)
    if identity is None:
        raise WorkerError("resolved model identity is missing")
    if hasattr(identity, "to_dict"):
        identity = identity.to_dict()
    if not isinstance(identity, Mapping):
        raise WorkerError("resolved model identity is invalid")
    expected = spec["model"]
    quant = identity.get("quantisation")
    actual = {
        "model_id": identity.get("model_id"), "revision": identity.get("revision"),
        "manifest": identity.get("model_manifest_sha256"), "architecture": identity.get("architecture"),
        "quant_bits": quant.get("bits") if isinstance(quant, Mapping) else None,
        "quant_group_size": quant.get("group_size") if isinstance(quant, Mapping) else None,
        "tokenizer": identity.get("tokenizer_sha256"),
    }
    if actual != expected:
        raise WorkerError("resolved model identity does not match stage spec")
    return identity


def _validate_tune_profile(profile: Any, *, stage: str, model_identity: Mapping[str, Any] | None = None) -> None:
    if not isinstance(profile, Mapping):
        raise WorkerError("tune profile is not an object")
    if model_identity is not None:
        stored = profile.get("model_identity")
        if not isinstance(stored, Mapping) or dict(stored) != dict(model_identity):
            raise WorkerError("tune profile model identity mismatch")
        conditions = profile.get("conditions")
        if not isinstance(conditions, Mapping):
            raise WorkerError("tune profile conditions are missing")
        expected_conditions = {
            "model_id": model_identity.get("model_id"), "model_revision": model_identity.get("revision"),
            "model_manifest_sha256": model_identity.get("model_manifest_sha256"),
            "model_architecture": model_identity.get("architecture"),
            "quantisation": model_identity.get("quantisation"),
            "quantisation_sha256": model_identity.get("quantisation_sha256"),
            "tokenizer_sha256": model_identity.get("tokenizer_sha256"),
            "model_identity_sha256": model_identity.get("identity_sha256"),
        }
        if any(conditions.get(key) != value for key, value in expected_conditions.items()):
            raise WorkerError("tune profile conditions identity mismatch")
    knobs = profile.get("knobs")
    defaults = TUNE_SEARCH_CONTRACT["knobs_defaults"]
    if not isinstance(knobs, Mapping) or set(knobs) != set(defaults):
        raise WorkerError("tune profile knobs are incomplete")
    allowed = {name: {json.dumps(defaults[name], sort_keys=True)} for name in defaults}
    for item in TUNE_SEARCH_CONTRACT["search"]:
        allowed[item["name"]].update(json.dumps(value, sort_keys=True) for value in item["values"])
    for name, value in knobs.items():
        if json.dumps(value, sort_keys=True) not in allowed[name]:
            raise WorkerError("tune profile knob is outside search contract")
    trials = profile.get("trials")
    if stage == "calibrate":
        if not isinstance(trials, (list, tuple)) or not trials:
            raise WorkerError("tune profile trials are missing")
    if trials is not None:
        if not isinstance(trials, (list, tuple)) or len(trials) > MAX_ITEMS:
            raise WorkerError("tune profile trials are invalid")
        names = set(allowed)
        for trial in trials:
            if not isinstance(trial, Mapping) or trial.get("knob") not in names:
                raise WorkerError("tune trial knob is outside search contract")
            if "value" in trial and json.dumps(trial["value"], sort_keys=True) not in allowed[trial["knob"]]:
                raise WorkerError("tune trial value is outside search contract")


def _finite_series(value: Any, *, length: int, name: str, allow_zero: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise WorkerError(f"raw {name} samples are incomplete")
    values = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or (float(item) < 0 if allow_zero else float(item) <= 0):
            raise WorkerError(f"raw {name} sample is invalid")
        values.append(float(item))
    return values


def _raw_sample(child: Mapping[str, Any], arm: str, *, pair_id: str,
                order: str, spec: Mapping[str, Any], arm_names: tuple[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    arms = child.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(arm_names):
        raise WorkerError("raw A/B arms are malformed")
    record = arms.get(arm)
    if not isinstance(record, Mapping):
        raise WorkerError("raw arm record is missing")
    repeats = spec["limits"]["repeats"]
    totals = _finite_series(record.get("total_ns"), length=repeats, name="total_ns")
    prefills = _finite_series(record.get("prefill_ns"), length=repeats, name="prefill_ns")
    decodes = _finite_series(record.get("decode_ns"), length=repeats, name="decode_ns", allow_zero=True)
    tokens = record.get("logical_tokens")
    if not isinstance(tokens, (list, tuple)) or not tokens or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in tokens):
        raise WorkerError("raw logical tokens are invalid")
    deterministic = record.get("deterministic") is True
    decode_steps = record.get("decode_steps")
    if isinstance(decode_steps, bool) or not isinstance(decode_steps, int) or decode_steps < 0:
        raise WorkerError("raw decode step count is invalid")
    pid = child.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise WorkerError("raw worker pid is invalid")
    decode_mean_ns = statistics.median(decodes)
    decode_tps = decode_steps / (decode_mean_ns / 1e9) if decode_steps > 0 and decode_mean_ns > 0 else None
    stop_reason = record.get("stop_reason") if isinstance(record.get("stop_reason"), str) and record.get("stop_reason") else None
    sample = {
        "session_id": spec["session"]["session_id"], "pair_id": pair_id, "arm": arm,
        "order": order, "fingerprint": spec["expected"]["fingerprint"],
        "workload": _sha_json(spec["workload"]), "workload_mode": spec["workload"]["mode"],
        "ttft_seconds": statistics.median(prefills) / 1e9,
        "ttft_ms": statistics.median(prefills) / 1e6,
        "engine_ttft_ns": statistics.median(prefills), "timer_scope": TTFT_CONTRACT,
        "decode_tps": decode_tps, "tokens": len(tokens),
        "token_hash": _sha_json(list(tokens)), "text_hash": None,
        "stop_reason": stop_reason, "status": "ok" if deterministic else "invalid",
        "error": "" if deterministic else "non_deterministic",
        "total_ns": totals, "prefill_ns": prefills, "decode_ns": decodes,
        "pid": pid, "logical_tokens": list(tokens), "decode_steps": decode_steps,
        "mlx_peak_bytes": record.get("mlx_peak_bytes") if isinstance(record.get("mlx_peak_bytes"), int) and not isinstance(record.get("mlx_peak_bytes"), bool) and record.get("mlx_peak_bytes") >= 0 else None,
    }
    return sample, {"total_ns": totals, "prefill_ns": prefills, "decode_ns": decodes,
                    "tokens": list(tokens), "pid": pid, "deterministic": deterministic,
                    "mlx_peak_bytes": sample["mlx_peak_bytes"],
                    "token_hash": sample["token_hash"],
                    "decode_steps": decode_steps,
        "count_hash": _sha_json(len(tokens)),
        "text_equivalence_hash": _sha_json({"tokenizer_sha256": spec["model"]["tokenizer"], "tokens": list(tokens)}),
        "stop_equivalence": "source_derived_stop_equivalence",
        "stop_equivalence_contract": {
            "max_tokens": spec["workload"]["max_tokens"],
            "capacity_source": "ironmule.runtime.Engine._capacity",
            "eos_source": "ironmule.tune._eos_ids",
        },
                    "stop_reason": sample["stop_reason"]}


def _normalise_ab(result: Any, *, spec: Mapping[str, Any], arm_names: tuple[str, str] = ("aa_left", "aa_right")) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise WorkerError("ab result is not an object")
    raw = result.get("raw")
    processes = spec["limits"]["processes"]
    if not isinstance(raw, (list, tuple)) or len(raw) != processes:
        raise WorkerError("ab raw process count is incomplete")
    seen_pids: set[int] = set()
    left: list[dict[str, Any]] = []
    right: list[dict[str, Any]] = []
    raw_pairs: list[dict[str, Any]] = []
    orders: list[str] = []
    for index, child in enumerate(raw):
        if not isinstance(child, Mapping):
            raise WorkerError("ab child record is malformed")
        order_value = child.get("order")
        order_tuple = tuple(order_value) if isinstance(order_value, (list, tuple)) else ()
        if order_tuple == arm_names:
            order = "AB"
        elif order_tuple == (arm_names[1], arm_names[0]):
            order = "BA"
        else:
            raise WorkerError("ab order is not balanced protocol")
        left_sample, left_raw = _raw_sample(child, arm_names[0], pair_id=f"pair-{index}", order=order, spec=spec, arm_names=arm_names)
        right_sample, right_raw = _raw_sample(child, arm_names[1], pair_id=f"pair-{index}", order=order, spec=spec, arm_names=arm_names)
        if left_raw["pid"] in seen_pids:
            raise WorkerError("ab worker pid is not unique")
        seen_pids.add(left_raw["pid"])
        left.append(left_sample); right.append(right_sample); orders.append(order)
        raw_pairs.append({"pair_id": f"pair-{index}", "order": order, "left": left_raw, "right": right_raw})
    if abs(orders.count("AB") - orders.count("BA")) > 1 or not orders.count("AB") or not orders.count("BA"):
        raise WorkerError("ab order balance is incomplete")
    reference = left[0]["logical_tokens"]
    token_identity = all(item["logical_tokens"] == reference for item in left + right)
    deterministic = all(item["status"] == "ok" for item in left + right)
    metric_complete = all(item.get("decode_tps") is not None and item.get("engine_ttft_ns") is not None for item in left + right)
    evidence = {"pairs": raw_pairs, "orders": orders, "token_identity": token_identity, "deterministic": deterministic, "metric_complete": metric_complete}
    # The source contract fixes max_tokens/capacity/eos derivation.  We do not
    # pretend `ab` observed a stop reason; this is a source-derived equivalence
    # gate based on identical logical token sequences and counts.
    stop_complete = token_identity and all(
        sample.get("tokens") == len(reference) for sample in left + right
    )
    result = {"baseline_samples": left, "candidate_samples": right,
              "raw_pairs": raw_pairs, "orders": orders,
              "pair_count": len(raw_pairs), "token_identity": token_identity,
              "deterministic": deterministic, "evidence_sha256": _sha_json(evidence),
              "stop_complete": stop_complete, "metric_complete": metric_complete,
              "complete": token_identity and deterministic and stop_complete and metric_complete}
    if arm_names == ("aa_left", "aa_right"):
        result["aa_baseline_samples"] = left
        result["aa_control_samples"] = right
    return result


def _metric_dict(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Return exactly the evaluator MetricSample wire shape.

    Engine-specific timing/hash details remain in ``raw_pairs``; putting them
    on this list would make RealSession's strict ``MetricSample(**item)``
    parser reject otherwise valid evidence.
    """
    return {key: sample.get(key) for key in (
        "session_id", "pair_id", "arm", "order", "fingerprint", "workload",
        "ttft_seconds", "decode_tps", "tokens", "status", "error")}


def _correctness_record(sample: Mapping[str, Any], *, passed: bool) -> dict[str, Any]:
    tokens = list(sample.get("logical_tokens", [])) if isinstance(sample.get("logical_tokens"), (list, tuple)) else []
    return {"token_ids": tokens, "text": "", "stop_reason": "source_derived_stop_equivalence",
            "physical_tokens": len(tokens), "visible_tokens": len(tokens),
            "response_hash": sample.get("text_equivalence_hash") or _sha_json(tokens),
            "passed": passed, "error": "" if passed else "token_identity_mismatch"}


def _profile_payload(spec: Mapping[str, Any], profile: Mapping[str, Any], *,
                     mlx_peak: int | None, rss_peak: int | None,
                     swap_delta: int | None, captured_output: str,
                     raw_ab: Mapping[str, Any] | None = None,
                     normalized_ab: Mapping[str, Any] | None = None) -> dict[str, Any]:
    expected = spec["expected"]
    stage = spec["stage"]
    tokens = profile.get("tokens")
    valid_tokens = isinstance(tokens, (list, tuple)) and bool(tokens) and all(
        isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 2**31 - 1 for item in tokens
    )
    if not valid_tokens:
        tokens = []
    token_count = profile.get("token_count")
    valid_token_count = isinstance(token_count, int) and not isinstance(token_count, bool) and token_count == len(tokens)
    if not valid_token_count:
        token_count = len(tokens)
    confirmation: dict[str, Any] | None = None
    confirmation_ok = False
    confirmation_source: Any = raw_ab if raw_ab is not None else profile.get("confirmation")
    details = _confirmation_details(confirmation_source)
    conf_token = None
    if isinstance(confirmation_source, Mapping):
        conf_token = confirmation_source.get("token_identity")
    if details is not None:
        total_ratio, phase_ratios, pair_count = details
        conf_token = conf_token if isinstance(conf_token, bool) else False
        confirmation_ok = conf_token is True and pair_count >= 3 and total_ratio["ci_high"] < 1.0
        confirmation = {
            "confirmed": confirmation_ok,
            "token_identity": conf_token is True,
            "pair_count": pair_count,
            "pairs": total_ratio.get("pairs", []),
            "ratio": total_ratio,
            "ratios": phase_ratios,
            "resources": {},
        }
    token_identity = profile.get("token_identity") if isinstance(profile.get("token_identity"), bool) else None
    if token_identity is None and isinstance(conf_token, bool):
        token_identity = conf_token
    evidence_missing: list[str] = []
    if not valid_tokens or not valid_token_count:
        evidence_missing.append("tokens")
    if not isinstance(mlx_peak, int):
        evidence_missing.append("mlx_peak")
    if not isinstance(rss_peak, int):
        evidence_missing.append("rss_peak")
    if not isinstance(swap_delta, int):
        evidence_missing.append("swap_delta")
    if stage == "calibrate":
        if valid_tokens:
            token_identity = True
        else:
            evidence_missing.append("token_gate")
    elif token_identity is not True:
        evidence_missing.append("token_identity")
    if stage == "test" and not confirmation_ok:
        evidence_missing.append("confirmation")
    if stage == "test" and normalized_ab is not None and normalized_ab.get("complete") is not True:
        evidence_missing.append("raw_evidence")
        if normalized_ab.get("token_identity") is not True:
            token_identity = False
    if not valid_tokens:
        token_identity = False
    normalized_ttft = None
    normalized_decode = None
    if normalized_ab is not None:
        ttft_values = [sample.get("ttft_ms") for sample in normalized_ab.get("candidate_samples", []) if isinstance(sample.get("ttft_ms"), (int, float))]
        decode_values = [sample.get("decode_tps") for sample in normalized_ab.get("candidate_samples", []) if isinstance(sample.get("decode_tps"), (int, float))]
        if ttft_values:
            normalized_ttft = statistics.median(ttft_values)
        if decode_values:
            normalized_decode = statistics.median(decode_values)
    ttft = normalized_ttft if normalized_ttft is not None else profile.get("ttft_ms")
    if stage == "test":
        if not _valid_timing(ttft):
            evidence_missing.append("ttft_ms")
    profile_json_hash = _sha_json(profile)
    required_times = ("baseline_ns", "baseline_prefill_ns", "baseline_decode_ns", "tuned_ns", "tuned_prefill_ns", "tuned_decode_ns")
    if stage == "calibrate" and any(not _valid_timing(profile.get(key)) for key in required_times):
        evidence_missing.append("screening_timing")
    valid_trials, trials_value = _valid_calibration_trials(profile.get("trials")) if stage == "calibrate" else (True, [])
    if stage == "calibrate" and not valid_trials:
        evidence_missing.append("screening")
    decode_ns = _number(profile.get("tuned_decode_ns"), 0.0)
    decode_tps = normalized_decode if normalized_decode is not None else ((float(token_count) / (decode_ns / 1e9)) if decode_ns > 0 and token_count > 0 else 0.0)
    limits = spec["limits"]
    if isinstance(mlx_peak, int) and mlx_peak > limits["max_peak_memory_bytes"]:
        evidence_missing.append("peak_memory_limit")
    if isinstance(rss_peak, int) and rss_peak > limits["max_rss_bytes"]:
        evidence_missing.append("rss_limit")
    if isinstance(swap_delta, int) and swap_delta > limits["max_swap_delta_bytes"]:
        evidence_missing.append("swap_limit")
    resources = {
        "ttft_ms": _number(ttft, 0.0),
        "decode_tokens_per_second": decode_tps,
        "peak_memory_bytes": mlx_peak if isinstance(mlx_peak, int) else 0,
        "peak_rss_bytes": rss_peak if isinstance(rss_peak, int) else 0,
        "swap_delta_bytes": swap_delta if isinstance(swap_delta, int) else 0,
        "resource_gate_passed": not evidence_missing and isinstance(mlx_peak, int) and isinstance(rss_peak, int) and isinstance(swap_delta, int),
    }
    if confirmation is not None:
        confirmation["resources"] = dict(resources)
    correctness = {
        "token_identity": bool(token_identity),
        "token_count": max(0, int(token_count)),
        "stop_reason": profile.get("stop_reason") if isinstance(profile.get("stop_reason"), str) and profile.get("stop_reason") else "unknown",
        "response_hash": _sha_json(tokens), "token_hash": _sha_json(tokens),
        "evidence_status": "token_gate_enforced_by_bound_tune_source" if stage == "calibrate" and valid_tokens else ("complete" if not evidence_missing else "missing:" + ",".join(evidence_missing)),
    }
    if normalized_ab is not None and normalized_ab.get("candidate_samples"):
        sample = normalized_ab["candidate_samples"][0]
        correctness.update({
            "count_hash": sample.get("count_hash"),
            "text_equivalence_hash": sample.get("text_equivalence_hash"),
            "text_hash": sample.get("text_equivalence_hash"),
            "stop_equivalence": sample.get("stop_equivalence"),
            "engine_ttft_ns": sample.get("engine_ttft_ns"),
            "timer_scope": sample.get("timer_scope"),
        })
    screening = {"status": "screening", "profile_artifact_sha256": profile_json_hash}
    if "gain" in profile:
        screening["reported_gain"] = profile["gain"]
    calibration: dict[str, Any] | None = None
    if stage == "calibrate" and valid_trials:
        diagnostics = {
            "baseline": {key: profile[key] for key in ("baseline_ns", "baseline_prefill_ns", "baseline_decode_ns")},
            "candidate": {key: profile[key] for key in ("tuned_ns", "tuned_prefill_ns", "tuned_decode_ns")},
            "trial_count": len(trials_value), "trials": trials_value,
        }
        calibration = {"complete": not evidence_missing, **diagnostics, "evidence_sha256": _sha_json(diagnostics)}
    payload = {
        "schema": RESULT_SCHEMA, "stage": stage, "commit": expected["commit"],
        "source_digest": expected["source_digest"], "fingerprint": expected["fingerprint"],
        "registry_hash": expected["registry_hash"], "worker_sha256": expected["worker_sha256"],
        "tune_search_contract_sha256": expected["tune_search_contract_sha256"],
        "candidate": spec["candidate"], "parameters": {}, "session_id": spec["session"]["session_id"],
        "correctness": correctness, "resources": resources, "screening": screening,
        "confirmation": None if stage == "calibrate" else confirmation, "calibration": calibration,
        # Confirmation is a correctness/statistics fact.  The outer outcome
        # still requires the independent worker resource gate, so a missing
        # TTFT or unsafe resource condition cannot become qualification.
        "confirmed": bool(stage == "test" and confirmation_ok and valid_tokens),
        "profile_id": profile.get("profile_id") if isinstance(profile.get("profile_id"), str) else None,
        "profile_version": profile.get("profile_version") if isinstance(profile.get("profile_version"), int) else None,
        "profile_artifact_sha256": profile_json_hash,
        "captured_output_bytes": len(captured_output.encode("utf-8", "replace")),
    }
    if normalized_ab is not None:
        payload.update({key: normalized_ab[key] for key in ("baseline_samples", "candidate_samples", "aa_baseline_samples", "aa_control_samples", "raw_pairs", "orders", "pair_count", "evidence_sha256") if key in normalized_ab})
        for key in ("baseline_samples", "candidate_samples", "aa_baseline_samples", "aa_control_samples"):
            if key in payload:
                payload[key] = [_metric_dict(sample) for sample in payload[key]]
        if normalized_ab.get("baseline_samples") and normalized_ab.get("candidate_samples"):
            passed = normalized_ab.get("complete") is True
            payload["baseline_correctness"] = _correctness_record(normalized_ab["baseline_samples"][0], passed=passed)
            payload["candidate_correctness"] = _correctness_record(normalized_ab["candidate_samples"][0], passed=passed)
    return payload


def _envelope(outcome: str, reason: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"outcome": outcome, "reason": reason[:512], "payload": dict(payload or {})}


def _resource_record(spec: Mapping[str, Any], *, mlx_peak: int | None,
                     rss_peak: int | None, swap_delta: int | None,
                     decode_tps: float = 0.0, ttft_ms: float = 0.0) -> dict[str, Any]:
    limits = spec["limits"]
    gate = (
        isinstance(mlx_peak, int) and isinstance(rss_peak, int) and isinstance(swap_delta, int)
        and mlx_peak <= limits["max_peak_memory_bytes"]
        and rss_peak <= limits["max_rss_bytes"]
        and swap_delta <= limits["max_swap_delta_bytes"]
    )
    return {"ttft_ms": ttft_ms, "decode_tokens_per_second": decode_tps,
            "peak_memory_bytes": mlx_peak if isinstance(mlx_peak, int) else 0,
            "peak_rss_bytes": rss_peak if isinstance(rss_peak, int) else 0,
            "swap_delta_bytes": swap_delta if isinstance(swap_delta, int) else 0,
            "resource_gate_passed": gate}


def _aa_payload(spec: Mapping[str, Any], evidence: Mapping[str, Any], *,
                mlx_peak: int | None, rss_peak: int | None,
                swap_delta: int | None, captured_output: str) -> dict[str, Any]:
    left_raw = evidence["aa_baseline_samples"]
    right_raw = evidence["aa_control_samples"]
    left = [_metric_dict(sample) for sample in left_raw]
    right = [_metric_dict(sample) for sample in right_raw]
    first_tokens = left_raw[0].get("logical_tokens", []) if left_raw else []
    decode_values = [item["decode_tps"] for item in left_raw if isinstance(item.get("decode_tps"), (int, float))] + [item["decode_tps"] for item in right_raw if isinstance(item.get("decode_tps"), (int, float))]
    ttft_values = [item["ttft_ms"] for item in left_raw if isinstance(item.get("ttft_ms"), (int, float))] + [item["ttft_ms"] for item in right_raw if isinstance(item.get("ttft_ms"), (int, float))]
    resources = _resource_record(spec, mlx_peak=mlx_peak, rss_peak=rss_peak,
                                 swap_delta=swap_delta,
                                 decode_tps=(statistics.median(decode_values) if decode_values else 0.0),
                                 ttft_ms=(statistics.median(ttft_values) if ttft_values else 0.0))
    # ``Engine.generate`` measures `_prefill` from immediately before the
    # prompt forward through argmax/eval/synchronize of the first token.  The
    # exact bound source digest makes this a stable engine-level TTFT contract;
    # it is not service/request-arrival TTFT.
    calibration = {
        "complete": bool(evidence["complete"] and resources["resource_gate_passed"] and spec["limits"]["ttft_contract"] != "unproven"),
        "trial_count": evidence["pair_count"], "trials": evidence["raw_pairs"],
        "baseline": {"samples": left}, "candidate": {"samples": right},
        "evidence_sha256": evidence["evidence_sha256"],
    }
    payload = {
        "schema": RESULT_SCHEMA, "stage": "calibrate", "commit": spec["expected"]["commit"],
        "source_digest": spec["expected"]["source_digest"], "fingerprint": spec["expected"]["fingerprint"],
        "registry_hash": spec["expected"]["registry_hash"], "worker_sha256": spec["expected"]["worker_sha256"],
        "tune_search_contract_sha256": spec["expected"]["tune_search_contract_sha256"],
        "candidate": spec["candidate"], "parameters": {}, "session_id": spec["session"]["session_id"],
        "correctness": {"token_identity": bool(evidence["token_identity"]), "token_count": len(first_tokens),
                         "stop_reason": "unknown", "response_hash": _sha_json(first_tokens),
                         "token_hash": _sha_json(first_tokens), "count_hash": _sha_json(len(first_tokens)),
                         "text_equivalence_hash": _sha_json({"tokenizer_sha256": spec["model"]["tokenizer"], "tokens": first_tokens}),
                         "stop_equivalence": "source_derived_stop_equivalence",
                         "timer_scope": TTFT_CONTRACT,
                         "stop_equivalence_contract": {"max_tokens": spec["workload"]["max_tokens"], "capacity_source": "ironmule.runtime.Engine._capacity", "eos_source": "ironmule.tune._eos_ids"},
                         "evidence_status": "source_derived_stop_equivalence"},
        "baseline_correctness": _correctness_record(left_raw[0], passed=bool(evidence["token_identity"])) if left_raw else None,
        "candidate_correctness": _correctness_record(right_raw[0], passed=bool(evidence["token_identity"])) if right_raw else None,
        "resources": resources, "screening": {"status": "aa_control", "evidence_sha256": evidence["evidence_sha256"]},
        "confirmation": None, "calibration": calibration, "confirmed": False,
        "profile_id": None, "profile_version": None,
        "profile_artifact_sha256": evidence["evidence_sha256"],
        "captured_output_bytes": len(captured_output.encode("utf-8", "replace")),
        "aa_baseline_samples": left, "aa_control_samples": right,
        "raw_pairs": evidence["raw_pairs"], "orders": evidence["orders"],
        "pair_count": evidence["pair_count"], "evidence_sha256": evidence["evidence_sha256"],
    }
    return payload


def _run(spec: Mapping[str, Any]) -> dict[str, Any]:
    root = Path.cwd().resolve()
    spec = _validate_spec(spec, stage_root=root)
    _verify_pythonpath(spec["expected"]["pythonpath_sha256"])
    ac, low = _read_power_state()
    if ac is not True:
        return _envelope("inconclusive", "ac_disconnected", {"schema": RESULT_SCHEMA, "stage": spec["stage"], "commit": spec["expected"]["commit"], "fingerprint": spec["expected"]["fingerprint"], "candidate": spec["candidate"], "status": "waiting_for_ac"})
    if low is not False:
        return _envelope("inconclusive", "low_power_enabled", {"schema": RESULT_SCHEMA, "stage": spec["stage"], "commit": spec["expected"]["commit"], "fingerprint": spec["expected"]["fingerprint"], "candidate": spec["candidate"], "status": "waiting_for_normal_power"})
    private_home = Path(tempfile.mkdtemp(prefix=".ironmule-session-", dir=root))
    transient_environment = ("IRONMULE_HOME", *OFFLINE_ENV, "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")
    old_environment = {key: os.environ.get(key) for key in transient_environment}
    try:
        os.chmod(private_home, 0o700)
        os.environ["IRONMULE_HOME"] = str(private_home)
        for key, value in OFFLINE_ENV.items():
            os.environ[key] = value
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
            os.environ.pop(key, None)
        before_swap = _read_swap_bytes()
        before_rss = _rss_bytes()
        mlx_peak: int | None = None
        captured = _BoundedCapture()
        raw_ab: Mapping[str, Any] | None = None
        normalized_ab: Mapping[str, Any] | None = None
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            try:
                mx = importlib.import_module("mlx.core")
            except (ImportError, ModuleNotFoundError) as exc:
                return _envelope("inconclusive", "mlx_unavailable", {"schema": RESULT_SCHEMA, "stage": spec["stage"], "commit": spec["expected"]["commit"], "fingerprint": spec["expected"]["fingerprint"], "candidate": spec["candidate"], "status": "runtime_unavailable", "error_type": type(exc).__name__})
            reset = getattr(mx, "reset_peak_memory", None)
            if callable(reset):
                reset()
            if spec["stage"] == "calibrate":
                try:
                    ab_module = importlib.import_module("ironmule.ab")
                    runtime_module = importlib.import_module("ironmule.runtime")
                    tune_module = importlib.import_module("ironmule.tune")
                    _verify_tune_contract(tune_module, runtime_module)
                    _verify_model_identity(tune_module, spec)
                    ab_result = ab_module.run(
                        {"aa_left": runtime_module.BASELINE, "aa_right": runtime_module.BASELINE},
                        processes=spec["limits"]["processes"], repeats=spec["limits"]["repeats"],
                        warmup=spec["limits"]["warmup"], max_tokens=spec["workload"]["max_tokens"],
                        model=spec["model"]["model_id"],
                    )
                    normalized_ab = _normalise_ab(ab_result, spec=spec)
                except (ImportError, ModuleNotFoundError) as exc:
                    return _envelope("inconclusive", "ironmule_runtime_unavailable", {"schema": RESULT_SCHEMA, "stage": spec["stage"], "commit": spec["expected"]["commit"], "fingerprint": spec["expected"]["fingerprint"], "candidate": spec["candidate"], "status": "runtime_unavailable", "error_type": type(exc).__name__})
                except WorkerError as exc:
                    message = str(exc).lower()
                    reason = "model_identity_mismatch" if "identity" in message or "model" in message else ("tune_contract_mismatch" if "tune" in message or "knobs" in message else "raw_aa_evidence_invalid")
                    return _envelope("inconclusive", reason, {"schema": RESULT_SCHEMA, "stage": spec["stage"], "commit": spec["expected"]["commit"], "fingerprint": spec["expected"]["fingerprint"], "candidate": spec["candidate"], "status": "tune_contract_mismatch" if reason == "tune_contract_mismatch" else "raw_evidence_invalid"})
            else:
                try:
                    tune_module = importlib.import_module("ironmule.tune")
                    runtime_module = importlib.import_module("ironmule.runtime")
                    _verify_tune_contract(tune_module, runtime_module)
                    model_identity = _verify_model_identity(tune_module, spec)
                    tune_fn = getattr(tune_module, "tune", None)
                    confirm_fn = getattr(tune_module, "confirm", None)
                    if not callable(tune_fn) or not callable(confirm_fn):
                        raise WorkerError("bound tune confirmation function is unavailable")
                    captured_confirm: dict[str, Any] = {}
                    def capture_confirm(*args: Any, **kwargs: Any) -> Any:
                        result = confirm_fn(*args, **kwargs)
                        captured_confirm["result"] = result
                        return result
                    tune_module.confirm = capture_confirm
                    try:
                        profile = tune_fn(spec["model"]["model_id"], max_tokens=spec["workload"]["max_tokens"], repeats=5, force=False, confirm_winner=True)
                        _validate_tune_profile(profile, stage="test", model_identity=model_identity)
                    finally:
                        tune_module.confirm = confirm_fn
                    raw_ab = captured_confirm.get("result")
                    if not isinstance(raw_ab, Mapping):
                        raise WorkerError("tune did not return its confirmation evidence")
                    normalized_ab = _normalise_ab(raw_ab, spec=spec, arm_names=("baseline", "candidate"))
                except (ImportError, ModuleNotFoundError) as exc:
                    return _envelope("inconclusive", "ironmule_runtime_unavailable", {"schema": RESULT_SCHEMA, "stage": spec["stage"], "commit": spec["expected"]["commit"], "fingerprint": spec["expected"]["fingerprint"], "candidate": spec["candidate"], "status": "runtime_unavailable", "error_type": type(exc).__name__})
                except WorkerError as exc:
                    message = str(exc).lower()
                    reason = "model_identity_mismatch" if "identity" in message or "model" in message else ("tune_contract_mismatch" if "tune" in message or "knobs" in message else "raw_confirmation_missing")
                    return _envelope("inconclusive", reason, {"schema": RESULT_SCHEMA, "stage": spec["stage"], "commit": spec["expected"]["commit"], "fingerprint": spec["expected"]["fingerprint"], "candidate": spec["candidate"], "status": "tune_contract_mismatch" if reason == "tune_contract_mismatch" else "raw_confirmation_missing"})
            peak = getattr(mx, "get_peak_memory", None)
            if callable(peak):
                value = peak()
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    mlx_peak = value
        after_swap = _read_swap_bytes()
        swap_delta = after_swap - before_swap if before_swap is not None and after_swap is not None else None
        rss_peak = _rss_bytes() or before_rss
        if spec["stage"] == "calibrate":
            child_peaks = [sample.get("mlx_peak_bytes") for sample in normalized_ab.get("aa_baseline_samples", []) + normalized_ab.get("aa_control_samples", []) if isinstance(sample.get("mlx_peak_bytes"), int)] if normalized_ab else []
            if (not isinstance(mlx_peak, int) or mlx_peak == 0) and child_peaks:
                mlx_peak = max(child_peaks)
            payload = _aa_payload(spec, normalized_ab, mlx_peak=mlx_peak, rss_peak=rss_peak, swap_delta=swap_delta, captured_output=captured.getvalue())
            return _envelope("ok" if payload["calibration"]["complete"] else "inconclusive", "calibration_complete" if payload["calibration"]["complete"] else "engine_ttft_unavailable", payload)
        payload = _profile_payload(spec, profile if isinstance(profile, Mapping) else {}, mlx_peak=mlx_peak, rss_peak=rss_peak, swap_delta=swap_delta, captured_output=captured.getvalue(), raw_ab=raw_ab, normalized_ab=normalized_ab)
        calibration = payload.get("calibration")
        if spec["stage"] == "calibrate" and isinstance(calibration, Mapping) and calibration.get("complete") is True and payload["resources"]["resource_gate_passed"]:
            return _envelope("ok", "calibration_complete", payload)
        if payload["confirmed"] and payload["resources"]["resource_gate_passed"]:
            return _envelope("qualified", "confirmed", payload)
        return _envelope("inconclusive", "missing_or_insufficient_evidence", payload)
    finally:
        for key, previous in old_environment.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        import shutil
        shutil.rmtree(private_home, ignore_errors=True)


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise WorkerTimeout("worker deadline")


def run_argv(argv: list[str] | None = None) -> tuple[int, dict[str, Any]]:
    """Run the fixed protocol without printing; useful for bounded tests."""

    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        spec = _read_spec(argv)
        _validate_spec(spec)
        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(spec["limits"]["max_seconds"]))
        try:
            result = _run(spec)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        encoded = _canonical(result)
        if len(encoded) > min(MAX_RESULT_BYTES, spec["limits"]["max_output_bytes"]):
            raise WorkerError("result exceeds output bound")
        return 0, result
    except (WorkerTimeout, TimeoutError) as exc:
        return 0, _envelope("timeout", "timeout", {"schema": RESULT_SCHEMA, "status": "timeout", "error_type": type(exc).__name__})
    except WorkerError as exc:
        return 2, _envelope("rejected", str(exc), {"schema": RESULT_SCHEMA, "status": "rejected"})
    except BaseException as exc:  # pragma: no cover - defensive process boundary
        return 0, _envelope("error", "worker_exception", {"schema": RESULT_SCHEMA, "status": "crash", "error_type": type(exc).__name__})


def main(argv: list[str] | None = None) -> int:
    code, result = run_argv(argv)
    encoded = _canonical(result)
    sys.stdout.buffer.write(encoded + b"\n")
    sys.stdout.buffer.flush()
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MAX_RESULT_BYTES", "MAX_SPEC_BYTES", "RESULT_SCHEMA", "SCHEMA", "TTFT_CONTRACT", "WorkerError",
    "validate_hub_model_id",
    "WorkerTimeout", "main", "run_argv",
]
