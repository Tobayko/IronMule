"""Narrow subprocess adapter for the Kaggle CLI.

It never infers free-tier eligibility from a quota response.  Account/session
verification belongs to :class:`friday_evidence.portable.quota.AccountPreflight`.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import re
import stat
import subprocess
import time
from typing import Any

from .contracts import ContractError, file_sha256, load_json, safe_file
from .quota import AccountPreflight, QuotaError, QuotaSnapshot


class KaggleCliError(RuntimeError):
    """The provider CLI did not yield safe, usable evidence.

    Deliberately contains no provider output, which can contain credentials.
    """


_KERNEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,63}/[A-Za-z0-9][A-Za-z0-9-]{0,127}$")
_ACCELERATOR = re.compile(r"^[A-Za-z][A-Za-z0-9]{1,63}$")
_TERMINAL_STATUS = re.compile(r"KernelWorkerStatus\.([A-Z_]+)")
_DATASET_SUFFIXES = frozenset({".npy", ".json", ".zip"})
_DATASET_STATES = {
    "pending": "pending", "creating": "pending", "running": "pending",
    "ready": "ready", "complete": "ready", "completed": "ready",
    "error": "failed", "failed": "failed", "invalid": "failed",
}


def _local_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _strict_boolean(value: Any, expected: bool) -> bool:
    # Kaggle's official metadata example historically shows quoted booleans.
    # Treat only their exact conventional spelling as equivalent to a bool.
    return value is expected or value == str(expected).lower()


def validate_private_metadata(path: str | Path) -> dict[str, Any]:
    supplied = Path(path)
    root = supplied if supplied.is_dir() else supplied.parent
    candidate = root / "kernel-metadata.json" if supplied.is_dir() else supplied
    try:
        parsed = load_json(candidate)
    except (OSError, ContractError, ValueError) as exc:
        raise KaggleCliError("cannot read Kaggle kernel metadata") from exc
    private = parsed.get("is_private", parsed.get("private"))
    if not _strict_boolean(private, True):
        raise KaggleCliError("Kaggle kernel metadata must explicitly be private")
    slug = parsed.get("id")
    if not isinstance(slug, str) or not _KERNEL_ID.fullmatch(slug) or slug.startswith("-"):
        raise KaggleCliError("Kaggle kernel id must be an owner/kernel-slug")
    if "id_no" in parsed:
        raise KaggleCliError("Kaggle kernel metadata must not override its slug with id_no")
    if not _strict_boolean(parsed.get("enable_internet"), False):
        raise KaggleCliError("Kaggle kernel metadata must explicitly disable internet")
    if not root.is_dir():
        raise KaggleCliError("Kaggle kernel metadata must reside in a project directory")
    code_file = parsed.get("code_file")
    if not isinstance(code_file, str) or Path(code_file).is_absolute() or "/" in code_file or "\\" in code_file:
        raise KaggleCliError("Kaggle kernel metadata must specify one local code_file")
    try:
        code = safe_file(root, code_file)
    except ContractError as exc:
        raise KaggleCliError("Kaggle kernel code_file is not a regular local file") from exc
    if code.suffix not in {".py", ".ipynb", ".Rmd"}:
        raise KaggleCliError("Kaggle kernel code_file has an unsupported type")
    return parsed


def validate_private_dataset_metadata(project_dir: str | Path) -> dict[str, Any]:
    """Validate the complete, bounded local dataset upload surface.

    The official CLI uploads every ordinary file in the supplied directory.
    Rejecting any unapproved file here is therefore required; filtering argv
    cannot make an otherwise broad directory upload safe.
    """
    root = Path(project_dir)
    if root.is_symlink() or not root.is_dir():
        raise KaggleCliError("Kaggle dataset project must be a local directory")
    metadata_path = root / "dataset-metadata.json"
    try:
        metadata = load_json(metadata_path)
    except (OSError, ContractError, ValueError) as exc:
        raise KaggleCliError("cannot read Kaggle dataset metadata") from exc
    dataset_id = metadata.get("id")
    if not isinstance(dataset_id, str) or not _KERNEL_ID.fullmatch(dataset_id) or dataset_id.startswith("-"):
        raise KaggleCliError("Kaggle dataset id must be an owner/dataset-slug")
    owner, slug = dataset_id.split("/", 1)
    if not 6 <= len(slug) <= 50 or not owner:
        raise KaggleCliError("Kaggle dataset slug is outside official bounds")
    title = metadata.get("title")
    if not isinstance(title, str) or not 6 <= len(title) <= 50:
        raise KaggleCliError("Kaggle dataset title is outside official bounds")
    # DATA1 includes derived real-model captures, so it is always labelled
    # conservatively.  This also prevents an accidental publication/license
    # broadening at the CLI boundary.
    if metadata.get("licenses") != [{"name": "other"}]:
        raise KaggleCliError("DATA1 dataset metadata must use the other license")
    artifacts = 0
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise KaggleCliError("cannot enumerate Kaggle dataset project") from exc
    for entry in entries:
        try:
            info = entry.lstat()
        except OSError as exc:
            raise KaggleCliError("cannot inspect Kaggle dataset file") from exc
        if not stat.S_ISREG(info.st_mode) or entry.is_symlink() or entry.suffix not in _DATASET_SUFFIXES:
            raise KaggleCliError("Kaggle dataset contains an unapproved upload path")
        try:
            safe_file(root, entry.name)
            file_sha256(entry)
        except ContractError as exc:
            raise KaggleCliError("Kaggle dataset contains an unsafe or oversized file") from exc
        if entry.name != "dataset-metadata.json":
            artifacts += 1
    if artifacts == 0:
        raise KaggleCliError("Kaggle dataset has no approved artifacts")
    return metadata


def parse_job_state(payload: str) -> str:
    """Parse one documented textual ``kernels status`` state exactly."""
    if not isinstance(payload, str):
        raise KaggleCliError("Kaggle status is not usable")
    matches = _TERMINAL_STATUS.findall(payload)
    if len(matches) != 1:
        raise KaggleCliError("Kaggle status is not usable")
    normalized = matches[0].lower().replace("_", "-")
    state_map = {
        "queued": "queued",
        "running": "running",
        "complete": "complete",
        "completed": "complete",
        "error": "failed",
        "failed": "failed",
        "cancelled": "cancelled",
        "canceled": "cancelled",
    }
    try:
        return state_map[normalized]
    except KeyError as exc:
        raise KaggleCliError("Kaggle status is not usable") from exc


def parse_terminal_status(payload: str) -> str:
    """Return only a terminal state; queued/running are normal poll results."""
    state = parse_job_state(payload)
    if state not in {"complete", "failed", "cancelled"}:
        raise KaggleCliError("Kaggle job is not confirmed terminal")
    return state


def parse_dataset_state(payload: str) -> str:
    """Parse the official ``datasets status --format json`` response."""
    try:
        value = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise KaggleCliError("Kaggle dataset status is not usable") from exc
    status = value.get("status") if isinstance(value, dict) else None
    if not isinstance(status, str):
        raise KaggleCliError("Kaggle dataset status is not usable")
    try:
        return _DATASET_STATES[status.strip().lower()]
    except KeyError as exc:
        raise KaggleCliError("Kaggle dataset status is not usable") from exc


class KaggleCli:
    """Command-only adapter; every command is an argv list with ``shell=False``."""

    def __init__(
        self,
        executable: str = "kaggle",
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.executable = executable
        self._runner = runner
        self._clock = clock

    def _run(self, argv: list[str], *, timeout: float = 30.0) -> str:
        try:
            result = self._runner(
                argv,
                shell=False,
                check=False,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise KaggleCliError("Kaggle CLI command could not be executed") from exc
        if result.returncode != 0:
            raise KaggleCliError("Kaggle CLI command failed")
        if not isinstance(result.stdout, str):
            raise KaggleCliError("Kaggle CLI returned invalid output")
        return result.stdout

    def quota(self, *, resource: str | None = "gpu") -> QuotaSnapshot:
        output = self._run([self.executable, "quota", "--csv"])
        try:
            return QuotaSnapshot.from_csv(output, observed_at_unix_s=self._clock(), resource=resource)
        except QuotaError as exc:
            raise KaggleCliError("Kaggle quota output is unsafe") from exc

    @staticmethod
    def _kernel_id(kernel_id: str, metadata_path: str | Path) -> str:
        metadata = validate_private_metadata(metadata_path)
        if not isinstance(kernel_id, str) or not _KERNEL_ID.fullmatch(kernel_id) or kernel_id != metadata["id"]:
            raise KaggleCliError("Kaggle kernel identifier does not match private metadata")
        return kernel_id

    def push(
        self,
        project_dir: str | Path,
        *,
        accelerator: str,
        timeout_seconds: int,
        account: AccountPreflight,
    ) -> str:
        metadata = validate_private_metadata(project_dir)
        title = metadata.get("title")
        expected_slug = metadata["id"].split("/", 1)[1]
        if not isinstance(title, str) or _local_slug(title) != expected_slug:
            raise KaggleCliError("Kaggle kernel title must resolve exactly to its declared slug")
        try:
            account.validate(self._clock(), max_age_s=300)
        except QuotaError as exc:
            raise KaggleCliError("Kaggle account preflight is unsafe") from exc
        if (
            not isinstance(accelerator, str)
            or not _ACCELERATOR.fullmatch(accelerator)
            or accelerator not in account.supported_free_skus
        ):
            raise KaggleCliError("Kaggle accelerator is not an explicitly verified free SKU")
        if metadata.get("machine_shape") not in {None, "", accelerator}:
            raise KaggleCliError("Kaggle kernel metadata accelerator does not match the selected SKU")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 900:
            raise KaggleCliError("Kaggle job timeout is outside the approved bound")
        self._run([
            self.executable, "kernels", "push", "-p", str(project_dir),
            "--accelerator", accelerator, "--timeout", str(timeout_seconds),
        ], timeout=float(timeout_seconds) + 120.0)
        kernel_id = metadata.get("id")
        assert isinstance(kernel_id, str)  # validated above
        return kernel_id

    def create_dataset(self, project_dir: str | Path) -> str:
        """Create a verified private DATA1 dataset without a public CLI flag."""
        metadata = validate_private_dataset_metadata(project_dir)
        self._run([self.executable, "datasets", "create", "-p", str(project_dir)])
        dataset_id = metadata["id"]
        assert isinstance(dataset_id, str)  # validated above
        return dataset_id

    def dataset_state(self, dataset_id: str) -> str:
        if not isinstance(dataset_id, str) or not _KERNEL_ID.fullmatch(dataset_id):
            raise KaggleCliError("Kaggle dataset identifier is invalid")
        output = self._run([
            self.executable, "datasets", "status", dataset_id, "--format", "json"
        ])
        return parse_dataset_state(output)

    def job_state(self, kernel_id: str, *, metadata_path: str | Path) -> str:
        kernel = self._kernel_id(kernel_id, metadata_path)
        return parse_job_state(self._run([self.executable, "kernels", "status", kernel]))

    def status(self, kernel_id: str, *, metadata_path: str | Path) -> str:
        state = self.job_state(kernel_id, metadata_path=metadata_path)
        if state not in {"complete", "failed", "cancelled"}:
            raise KaggleCliError("Kaggle job is not confirmed terminal")
        return state

    def pull(self, kernel_id: str, destination: str | Path, *, metadata_path: str | Path) -> None:
        kernel = self._kernel_id(kernel_id, metadata_path)
        self._run([self.executable, "kernels", "pull", "-p", str(destination), kernel])

    def output(self, kernel_id: str, destination: str | Path, *, metadata_path: str | Path) -> None:
        kernel = self._kernel_id(kernel_id, metadata_path)
        self._run([self.executable, "kernels", "output", kernel, "-p", str(destination)])


__all__ = [
    "KaggleCli", "KaggleCliError", "parse_dataset_state", "parse_job_state", "parse_terminal_status",
    "validate_private_dataset_metadata", "validate_private_metadata",
]
