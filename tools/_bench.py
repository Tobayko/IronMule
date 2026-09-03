"""Shared preconditions for every measuring tool.

Kept in one place because a precondition that drifts between tools is worse than
no precondition: two runs would then be gated differently while both claim to
follow the same rules.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from friday_evidence.budget import BudgetError, BudgetGuard  # noqa: E402
from friday_evidence.run import run_persisted  # noqa: E402


class PowerError(SystemExit):
    """Raised when the machine is not in a state that allows measuring."""


class LocalModelError(RuntimeError):
    """Raised when a registered local model snapshot is absent or inconsistent."""


class LocalModelSnapshot:
    """Validated identity of one project-local Hugging Face snapshot."""

    __slots__ = ("model_id", "revision", "path", "weight_files", "weight_bytes")

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        path: Path,
        weight_files: tuple[str, ...],
        weight_bytes: int,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.path = path
        self.weight_files = weight_files
        self.weight_bytes = weight_bytes

    def report_identity(self) -> dict[str, object]:
        """Return reproducibility fields without exposing an absolute local path."""

        return {
            "model_id": self.model_id,
            "model_revision": self.revision,
            "model_snapshot_weight_files": list(self.weight_files),
            "model_snapshot_weight_bytes": self.weight_bytes,
            "model_source": "validated_project_local_snapshot",
        }


def resolve_local_model_snapshot(
    model_id: str, *, hub_root: Path | None = None
) -> LocalModelSnapshot:
    """Resolve a complete executable snapshot without any network fallback.

    Hugging Face may regard a cache as incomplete when repository documentation
    was intentionally omitted. MLX-LM does not need those files. This resolver
    validates the actual execution inputs -- config, tokenizer and the exact
    ``model*.safetensors`` set selected by the installed non-distributed MLX-LM
    loader -- and returns their immutable snapshot revision.
    """

    parts = model_id.split("/")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if len(parts) != 2 or any(
        not part or any(character not in allowed for character in part) for part in parts
    ):
        raise LocalModelError("invalid local model identifier")

    default_hub = PROJECT_ROOT / ".friday-data" / "models" / "hub"
    user_hub = Path.home() / ".cache" / "huggingface" / "hub"
    repo_name = f"models--{parts[0]}--{parts[1]}"
    if hub_root:
        selected_hub = hub_root
    elif (default_hub / repo_name).is_dir():
        selected_hub = default_hub
    elif (user_hub / repo_name).is_dir():
        selected_hub = user_hub
    else:
        selected_hub = default_hub
    repository = selected_hub / repo_name
    try:
        repository_real = repository.resolve(strict=True)
    except OSError as exc:
        raise LocalModelError(f"local model cache is unavailable: {model_id}") from exc
    if not repository_real.is_dir():
        raise LocalModelError(f"local model cache is not a directory: {model_id}")

    reference = repository_real / "refs" / "main"
    if reference.is_symlink() or not reference.is_file():
        raise LocalModelError(f"local model main reference is unavailable: {model_id}")
    try:
        revision = reference.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise LocalModelError(f"local model main reference is unreadable: {model_id}") from exc
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise LocalModelError(f"local model revision is invalid: {model_id}")

    try:
        snapshots_real = (repository_real / "snapshots").resolve(strict=True)
        snapshot = (snapshots_real / revision).resolve(strict=True)
        snapshot.relative_to(snapshots_real)
    except (OSError, ValueError) as exc:
        raise LocalModelError(f"local model snapshot is unavailable: {model_id}") from exc
    if not snapshot.is_dir():
        raise LocalModelError(f"local model snapshot is not a directory: {model_id}")

    def execution_file(relative: str) -> Path:
        candidate = snapshot / relative
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(repository_real)
        except (OSError, ValueError) as exc:
            raise LocalModelError(
                f"local model execution file is unavailable: {model_id}:{relative}"
            ) from exc
        if not resolved.is_file():
            raise LocalModelError(
                f"local model execution path is not a file: {model_id}:{relative}"
            )
        return resolved

    try:
        json.loads(execution_file("config.json").read_text(encoding="utf-8"))
        json.loads(execution_file("tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalModelError(f"local model metadata is invalid: {model_id}") from exc
    if not any((snapshot / name).exists() for name in ("tokenizer.json", "tokenizer.model")):
        raise LocalModelError(f"local model tokenizer is unavailable: {model_id}")
    for tokenizer_name in ("tokenizer.json", "tokenizer.model"):
        if (snapshot / tokenizer_name).exists():
            execution_file(tokenizer_name)

    # MLX-LM 0.31.3 non-distributed loading uses this exact direct-child glob.
    # A multimodal upstream index may still describe original shards that are
    # intentionally absent from a converted, monolithic MLX text checkpoint.
    names = sorted(path.name for path in snapshot.glob("model*.safetensors"))
    if not names or any(
        not isinstance(name, str)
        or not name.endswith(".safetensors")
        or Path(name).is_absolute()
        or ".." in Path(name).parts
        for name in names
    ):
        raise LocalModelError(f"local model weight set is invalid: {model_id}")
    weights = [execution_file(name) for name in names]

    return LocalModelSnapshot(
        model_id=model_id,
        revision=revision,
        path=snapshot,
        weight_files=tuple(names),
        weight_bytes=sum(path.stat().st_size for path in weights),
    )


def read_power_source() -> str:
    """Return 'ac_power', 'battery_power' or 'unknown' without ever raising."""

    try:
        completed = subprocess.run(
            ["/usr/bin/pmset", "-g", "ps"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    text = completed.stdout.decode("utf-8", errors="replace")
    if "AC Power" in text:
        return "ac_power"
    if "Battery Power" in text:
        return "battery_power"
    return "unknown"


def require_ac_power() -> str:
    """Refuse to measure unless the machine is on mains power.

    This is a measurement requirement, not a courtesy: on battery macOS caps the
    GPU power budget, so a run is neither comparable to a mains run nor gentle on
    the hardware.  Failing closed is the point -- a silently degraded measurement
    is worse than no measurement.
    """

    source = read_power_source()
    if source != "ac_power":
        raise PowerError(
            f"refused: mains power is a measurement requirement (found: {source})"
        )
    return source


#: What an offline measurement must never be able to reach. The established
#: workers set these on a child process they spawn; an in-process worker has
#: to set them on itself, before the model library is imported.
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "NO_PROXY": "*",
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
    "ALL_PROXY": "",
    "http_proxy": "",
    "https_proxy": "",
    "all_proxy": "",
}


def enforce_offline() -> dict:
    """Pin this process offline and return exactly what was applied.

    Call before importing the model library: the Hugging Face client reads
    these once at import time, so setting them afterwards changes nothing.
    Idempotent, and the returned mapping belongs in the provenance block so a
    result records the conditions it was measured under.
    """

    os.environ.update(OFFLINE_ENVIRONMENT)
    return dict(OFFLINE_ENVIRONMENT)


def check_prompt_length(ids, expected: int, *, tolerance: int = 0) -> int:
    """Fail before measuring if the prompt is not the one that was registered.

    A study whose prompt drifted answers a question nobody asked, and it does
    so using a whole approved measurement block. ``tolerance`` is zero where a
    prompt must reproduce a sealed one exactly, and a stated band where a
    comparison only needs the same workload class.
    """

    count = len(ids)
    if abs(count - expected) > tolerance:
        allowed = f"{expected}" if tolerance == 0 else f"{expected} +/- {tolerance}"
        raise SystemExit(
            f"prompt is {count} tokens, registered as {allowed}; "
            "the measurement would not be comparable"
        )
    return count


def study_provenance(
    code_paths: "list[str | Path]",
    *,
    preregistration: "str | Path | None" = None,
    extra: "dict | None" = None,
) -> dict:
    """Bind one standalone study result to the code and tree that produced it.

    The serious studies in ``experiments/`` (persistent process, matmul A/B)
    carry such a block; the exploratory scans do not.  A decision-grade result
    needs one: without it a later edit to a threshold file would silently make
    an old result look as if it had used the new threshold.

    Git and hashing come from :mod:`friday_evidence.provenance`.  Its
    ``collect_provenance`` is not reused because it is bound to the closed
    H1/H2 tool registry and to a fixed ``SOURCE_DIRS`` set that does not cover
    ``experiments/``; registering a study there would change the provenance
    hash of every unrelated tool.
    """

    from friday_evidence.provenance import _file_hashes, _run_git

    resolved = [Path(item).resolve() for item in code_paths]
    if preregistration is not None:
        resolved.append(Path(preregistration).resolve())
    revision = _run_git("rev-parse", "HEAD").decode("ascii").strip()
    status = _run_git("status", "--porcelain=v1", "--untracked-files=all").decode("utf-8").strip()
    files = _file_hashes(resolved)
    payload = {
        "git_revision": revision,
        "git_dirty": bool(status),
        "code_files_sha256": files,
        "code_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "power_source": read_power_source(),
    }
    if extra:
        payload.update(extra)
    return payload


def harness_preconditions(*, allow_long_gpu: bool = False) -> "BudgetGuard":
    """The gate every exploratory benchmark harness shares.

    Mains power + offline environment + a ``BudgetGuard``. ``allow_long_gpu``
    lifts only the 6 s continuity limit (for a deliberately-authorised long-run
    study); the duty cycle, wall limit and cooldown stay.
    """

    require_ac_power()
    enforce_offline()
    from friday_evidence.budget import BudgetPolicy

    policy = BudgetPolicy()
    if allow_long_gpu:
        policy = dataclasses.replace(policy, continuous_gpu_limit_s=1e9, gpu_work_limit_s=1e9)
    return BudgetGuard(policy)


def release_gate(args, self_check) -> int | None:
    """Return an exit code when the tool must not proceed, else None.

    Centralised on purpose: this exact five-line pattern was copied into each
    tool, and the one tool that lacked it (`run_h0_aa`) really did start a
    six-process GPU run from a stray invocation.  One implementation cannot
    diverge; five copies already had.
    """

    if getattr(args, "self_check", False):
        return self_check()
    if not getattr(args, "execute", False):
        print(json.dumps({"state": "not_released", "hint": "pass --execute"}))
        return 78
    return None


__all__ = [
    "BudgetError",
    "BudgetGuard",
    "LocalModelError",
    "LocalModelSnapshot",
    "PowerError",
    "read_power_source",
    "release_gate",
    "resolve_local_model_snapshot",
    "require_ac_power",
    "OFFLINE_ENVIRONMENT",
    "check_prompt_length",
    "enforce_offline",
    "harness_preconditions",
    "run_persisted",
    "study_provenance",
]
