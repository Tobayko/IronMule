"""Versioned DATA1 contracts; no framework imports or implicit execution."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
import uuid

from friday_evidence.canonical import canonical_json_bytes, canonical_sha256

BACKENDS = ("mlx", "cuda", "tpu")
PARTITIONS = ("train", "validation", "holdout")
CANDIDATES = ("native", "compiled", "rows128", "rows512")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_ARRAY_BYTES = 256 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class ContractError(ValueError):
    """A bounded, provenance-bound collection input was not accepted."""


def identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ContractError(f"invalid_{field}")
    return value


def sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ContractError(f"invalid_{field}")
    return value


def safe_file(root: Path, name: str) -> Path:
    identifier(name, "file_name")
    path = root / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root.resolve():
        raise ContractError("input_is_not_a_regular_local_file")
    return path


def file_sha256(path: Path) -> str:
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_ARRAY_BYTES:
        raise ContractError("file_invalid_or_too_large")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    if path.stat().st_mtime_ns != info.st_mtime_ns or path.stat().st_size != info.st_size:
        raise ContractError("file_changed_while_hashing")
    return result.hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate_json_key")
        result[key] = value
    return result


def load_json(path: Path) -> dict:
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > MAX_JSON_BYTES:
        raise ContractError("json_invalid_or_too_large")
    def invalid(value: str) -> None:
        raise ContractError("nonfinite_json")
    result = json.loads(path.read_bytes(), object_pairs_hook=_unique, parse_constant=invalid)
    if not isinstance(result, dict):
        raise ContractError("json_object_required")
    canonical_json_bytes(result)
    return result


def write_json_new(path: Path, value: dict) -> None:
    """Never overwrite a sealed spec/report, including through a symlink."""
    raw = canonical_json_bytes(value)
    if len(raw) > MAX_JSON_BYTES:
        raise ContractError("json_too_large")
    path = Path(path)
    if any(parent.is_symlink() for parent in (path.parent, *path.parents)):
        raise ContractError("output_parent_symlink")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def validate_case(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ContractError("case_object_required")
    for field in ("case_id", "lineage_id", "prompt_family", "shape_family", "capture_session"):
        identifier(value.get(field), field)
    for field in ("a_sha256", "b_sha256", "weight_sha256", "model_sha256"):
        sha256(value.get(field), field)
    for field in ("a_file", "b_file"):
        name = identifier(value.get(field), field)
        if not name.endswith(".npy"):
            raise ContractError("only_npy_tensors_allowed")
    if value.get("source_kind") != "real_model_capture" or value.get("dtype") != "float32":
        raise ContractError("real_float32_capture_required")
    if value.get("partition") not in PARTITIONS:
        raise ContractError("invalid_partition")
    model_id = value.get("model_id")
    if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id):
        raise ContractError("invalid_model_id")
    shape = value.get("shape")
    if not isinstance(shape, list) or len(shape) != 3 or any(type(n) is not int or not 1 <= n <= 65536 for n in shape):
        raise ContractError("invalid_matmul_shape")
    m, k, n = shape
    if 4 * (m*k + k*n + m*n) > MAX_ARRAY_BYTES:
        raise ContractError("case_memory_limit")
    canonical_json_bytes(value)
    return value


def validate_spec(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("schema") != "ironmule.experiment.v1":
        raise ContractError("invalid_experiment_schema")
    if not isinstance(value.get("run_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", value["run_id"]):
        raise ContractError("invalid_run_id")
    if value.get("backend") not in BACKENDS or value.get("partition") not in PARTITIONS:
        raise ContractError("invalid_backend_or_partition")
    mode = value.get("mode")
    if mode not in ("smoke", "measure"):
        raise ContractError("invalid_mode")
    for field, minimum, maximum in (("seed", 0, 2**32-1), ("warmup", 5, 20), ("pairs", 12, 48),
                                    ("server_seconds", 1, 180 if mode == "smoke" else 900),
                                    ("work_seconds", 1, 60 if mode == "smoke" else 720)):
        number = value.get(field)
        if type(number) is not int or not minimum <= number <= maximum:
            raise ContractError(f"invalid_{field}")
    if value["work_seconds"] >= value["server_seconds"]:
        raise ContractError("cleanup_margin_required")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates or candidates[0] != "native" or any(c not in CANDIDATES for c in candidates) or len(candidates) != len(set(candidates)):
        raise ContractError("invalid_candidate_allowlist")
    cases = value.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 12:
        raise ContractError("one_to_twelve_cases_required")
    ids = set()
    for case in cases:
        validate_case(case)
        if case["case_id"] in ids or case["partition"] != value["partition"]:
            raise ContractError("duplicate_case_or_mixed_partition")
        ids.add(case["case_id"])
    if "code_sha256" in value:
        sha256(value["code_sha256"], "code_sha256")
    canonical_json_bytes(value)
    return value


def spec_digest(value: dict) -> str:
    return canonical_sha256(validate_spec(value))


def code_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    # Bind the actual shared dependencies, not unrelated concurrent studies.
    shared = ("__init__.py", "budget.py", "canonical.py", "events.py", "identity.py",
              "process_memory.py", "registry.py", "statistics.py", "storage.py")
    return [root / name for name in shared] + sorted((root / "portable").glob("*.py"))


def code_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = code_paths()
    return canonical_sha256({str(path.relative_to(root)): file_sha256(path) for path in paths})


def make_spec(cases: list[dict], backend: str, *, mode: str = "smoke", seed: int = 20260907,
              pairs: int = 12) -> dict:
    return validate_spec({"schema": "ironmule.experiment.v1", "run_id": uuid.uuid4().hex,
        "backend": backend, "partition": cases[0]["partition"] if cases else None,
        "mode": mode, "seed": seed, "warmup": 5, "pairs": pairs,
        "server_seconds": 180 if mode == "smoke" else 900,
        "work_seconds": 60 if mode == "smoke" else 720,
        "candidates": list(CANDIDATES), "cases": cases, "code_sha256": code_digest()})
