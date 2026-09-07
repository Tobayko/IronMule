"""Dependency-light inventory of locally cached Hugging Face model snapshots.

The inventory is deliberately metadata-only.  It does not import MLX or
``huggingface_hub``, download anything, read tensor contents, or claim that a
model is qualified to run on any particular device.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence


_MODEL_DIRECTORY = re.compile(r"^models--(?P<id>.+)$")
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt")
_TOKENIZER_NAMES = frozenset(
    {
        "added_tokens.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
)


def _path(value: Path | str) -> Path:
    return Path(value).expanduser()


def _default_cache_roots() -> list[Path]:
    """Return the documented Hugging Face cache locations in precedence order."""

    roots: list[Path] = []
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(name)
        if value:
            roots.append(_path(value))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(_path(hf_home) / "hub")
    # expanduser honours HOME on Unix, which also makes this deterministic in
    # callers that provide an isolated HOME for discovery.
    roots.append(_path("~/.cache/huggingface/hub"))
    return _unique_paths(roots)


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for value in paths:
        path = _path(value)
        key = os.path.normcase(str(path.absolute()))
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _resolved(path: Path) -> Path:
    """Resolve as much as possible without requiring a path to exist."""

    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError):
        return path.absolute()


def _read_json_metadata(path: Path) -> Any:
    with path.open("rb") as stream:
        raw = stream.read(_MAX_METADATA_BYTES + 1)
    if len(raw) > _MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds the inventory size limit")
    return json.loads(raw.decode("utf-8"))


def _row(
    *,
    model_id: str,
    revision: str,
    snapshot_path: Path,
    cache_root: Path,
    family: str,
    weight_bytes: int = 0,
    weight_files: int = 0,
    status: str,
    reasons: Sequence[str] = (),
    warnings: Sequence[str] = (),
    loader: str = "hf",
    weight_selection: str = "none",
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "revision": revision,
        "snapshot_path": str(_resolved(snapshot_path)),
        "cache_root": str(_resolved(cache_root)),
        "family": family,
        "weight_bytes": int(weight_bytes),
        "weight_files": int(weight_files),
        "status": status,
        "reasons": list(dict.fromkeys(str(reason) for reason in reasons)),
        "warnings": list(dict.fromkeys(str(warning) for warning in warnings)),
        "loader": loader,
        "weight_selection": weight_selection,
    }


def _error_root(cache_root: Path, reason: str, *, loader: str) -> dict[str, Any]:
    return _row(
        model_id="<cache-root>",
        revision="",
        snapshot_path=cache_root,
        cache_root=cache_root,
        family="",
        status="error",
        reasons=(reason,),
        loader=loader,
    )


def _model_id(repository: Path) -> tuple[str | None, str | None]:
    match = _MODEL_DIRECTORY.match(repository.name)
    if match is None:
        return None, "directory is not a Hugging Face model cache entry"
    encoded = match.group("id")
    if not encoded:
        return None, "model cache entry has an empty model id"
    # Hugging Face encodes the slash separator as ``--`` in cache directory
    # names.  Preserve the rest of the identifier exactly as it appears.
    return encoded.replace("--", "/"), None


def _safe_child(snapshot: Path, name: str) -> Path:
    """Resolve an index-provided name while forbidding traversal."""

    candidate = Path(name)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"weight index contains unsafe shard path: {name!r}")
    # Do not resolve symlinks here: HF snapshot entries commonly point out of
    # the snapshot directory into the repository's ``blobs`` directory.  The
    # lexical component check above is the traversal boundary; the eventual
    # symlink target is checked against the repository in ``_check_file``.
    return snapshot.joinpath(*candidate.parts)


def _check_file(path: Path, repository: Path, *, label: str) -> tuple[bool, str | None]:
    """Check existence and HF-style symlink containment without reading bytes."""

    try:
        path.resolve(strict=False).relative_to(repository.resolve(strict=False))
    except (OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, ValueError):
            return False, f"{label} path escapes model cache entry"
        return False, f"{label} path cannot be resolved"
    if not path.exists():
        if path.is_symlink():
            return False, f"{label} is a broken symlink"
        return False, f"missing {label}"
    if not path.is_file():
        return False, f"{label} is not a regular file"
    if path.is_symlink():
        try:
            target = path.resolve(strict=True)
            target.relative_to(repository.resolve(strict=False))
        except (OSError, ValueError, RuntimeError) as exc:
            if isinstance(exc, ValueError):
                return False, f"{label} symlink escapes model cache entry"
            return False, f"{label} symlink target cannot be resolved"
    return True, None


def _read_config(
    snapshot: Path, repository: Path
) -> tuple[dict[str, Any] | None, str | None, bool]:
    path = snapshot / "config.json"
    ok, reason = _check_file(path, repository, label="config.json")
    if not ok:
        return None, reason, False
    try:
        value = _read_json_metadata(path)
    except (OSError, UnicodeError, ValueError):
        return None, "config.json is invalid or exceeds the metadata size limit", True
    if not isinstance(value, dict):
        return None, "config.json root must be an object", True
    return value, None, False


def _family(config: dict[str, Any] | None, model_id: str) -> str:
    if config is not None:
        model_type = config.get("model_type")
        if isinstance(model_type, str) and model_type.strip():
            return model_type.strip()
        architectures = config.get("architectures")
        if isinstance(architectures, list):
            for architecture in architectures:
                if isinstance(architecture, str) and architecture.strip():
                    return architecture.strip()
    name = model_id.rsplit("/", 1)[-1]
    match = re.match(r"[A-Za-z][A-Za-z0-9]*", name)
    return match.group(0) if match else name


def _tokenizer_status(snapshot: Path, repository: Path) -> tuple[bool, list[str], bool]:
    reasons: list[str] = []
    try:
        entries = list(snapshot.iterdir())
    except OSError:
        return False, ["snapshot cannot be listed"], True
    by_name = {entry.name: entry for entry in entries}
    tiktoken = sorted(entry for entry in entries if entry.suffix == ".tiktoken")
    if "tokenizer.json" in by_name:
        selected = [by_name["tokenizer.json"]]
    elif "tokenizer.model" in by_name:
        selected = [by_name["tokenizer.model"]]
    elif "vocab.json" in by_name and "merges.txt" in by_name:
        selected = [by_name["vocab.json"], by_name["merges.txt"]]
    elif "vocab.txt" in by_name:
        selected = [by_name["vocab.txt"]]
    elif tiktoken:
        selected = [tiktoken[0]]
    else:
        return False, ["snapshot contains no complete tokenizer artifacts"], False
    had_error = False
    for entry in selected:
        ok, reason = _check_file(entry, repository, label=f"tokenizer artifact {entry.name}")
        if not ok and reason:
            reasons.append(reason)
            had_error = "broken symlink" not in reason and "missing" not in reason
    return not reasons, reasons, had_error


def _weight_status(
    snapshot: Path, repository: Path, *, loader: str
) -> tuple[int, int, list[str], bool, list[str], str]:
    """Return logical weight bytes/count and metadata problems.

    An index is authoritative when present.  A shard is counted once by its
    relative name; tensor payloads are only stat'ed, never opened.
    """

    reasons: list[str] = []
    had_error = False
    warnings: list[str] = []
    selection = "hf_index" if loader == "hf" else "mlx_lm_direct_glob"
    indexed: list[Path] = []
    try:
        entries = sorted(snapshot.iterdir(), key=lambda item: item.name)
    except OSError:
        return 0, 0, ["snapshot cannot be listed"], True, warnings, selection
    index_paths = [
        entry
        for entry in entries
        if entry.name.endswith((".safetensors.index.json", ".bin.index.json"))
    ]
    for index_path in index_paths:
        index_ok, index_reason = _check_file(
            index_path, repository, label=f"weight index {index_path.name}"
        )
        if not index_ok:
            message = index_reason or f"cannot read weight index {index_path.name}"
            (reasons if loader == "hf" else warnings).append(message)
            had_error = loader == "hf"
            continue
        try:
            value = _read_json_metadata(index_path)
        except (OSError, UnicodeError, ValueError):
            message = f"{index_path.name} is not valid UTF-8 JSON"
            (reasons if loader == "hf" else warnings).append(message)
            had_error = loader == "hf"
            continue
        weight_map = value.get("weight_map") if isinstance(value, dict) else None
        if not isinstance(weight_map, dict):
            message = f"{index_path.name} has no object weight_map"
            (reasons if loader == "hf" else warnings).append(message)
            had_error = loader == "hf"
            continue
        shard_names: list[Any] = []
        for candidate in weight_map.values():
            if candidate not in shard_names:
                shard_names.append(candidate)
        for shard_name in sorted(shard_names, key=lambda item: str(item)):
            if not isinstance(shard_name, str):
                message = f"{index_path.name} contains a non-string shard name"
                (reasons if loader == "hf" else warnings).append(message)
                had_error = loader == "hf"
                continue
            try:
                shard = _safe_child(snapshot, shard_name)
            except ValueError as exc:
                (reasons if loader == "hf" else warnings).append(str(exc))
                had_error = loader == "hf"
                continue
            if shard not in indexed:
                indexed.append(shard)
    direct = [
        entry
        for entry in entries
        if entry.name.startswith("model") and entry.name.endswith(".safetensors")
    ]
    if loader == "mlx_lm":
        candidates = direct
        indexed_names = {
            path.relative_to(snapshot).as_posix() for path in indexed
        }
        direct_names = {
            path.relative_to(snapshot).as_posix() for path in direct
        }
        if index_paths and indexed_names != direct_names:
            warnings.append(
                "weight index shard set differs from mlx_lm direct model*.safetensors "
                f"selection (index={sorted(indexed_names)!r}, selected={sorted(direct_names)!r})"
            )
    else:
        candidates = indexed
    if not candidates and not index_paths and loader == "hf":
        candidates = [entry for entry in entries if entry.name.endswith(_WEIGHT_SUFFIXES)]
    if not candidates and not reasons:
        reasons.append("snapshot contains no weight files")
    total = 0
    count = 0
    for weight in sorted(candidates, key=lambda item: item.as_posix()):
        ok, reason = _check_file(weight, repository, label=f"weight file {weight.name}")
        if not ok:
            if reason:
                reasons.append(reason)
            # A missing target is an incomplete cache, while malformed links
            # and other filesystem failures are operational errors.
            had_error = had_error or (
                reason is not None
                and "missing" not in reason
                and "broken symlink" not in reason
            )
            continue
        try:
            total += weight.stat().st_size
        except OSError:
            reasons.append(f"cannot stat weight file {weight.name}")
            had_error = True
            continue
        count += 1
    return total, count, reasons, had_error, warnings, selection


def _inspect_snapshot(
    *,
    model_id: str,
    revision: str,
    snapshot: Path,
    repository: Path,
    cache_root: Path,
    loader: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    operational_error = False
    if not _REVISION.fullmatch(revision):
        reasons.append("snapshot revision contains unsupported characters")
        operational_error = True
    if not snapshot.exists():
        reasons.append(
            "snapshot is a broken symlink"
            if snapshot.is_symlink()
            else "snapshot is missing"
        )
        operational_error = snapshot.is_symlink()
        return _row(
            model_id=model_id,
            revision=revision,
            snapshot_path=snapshot,
            cache_root=cache_root,
            family=_family(None, model_id),
            status="error" if operational_error else "incomplete",
            reasons=reasons,
            loader=loader,
        )
    if not snapshot.is_dir():
        reasons.append("snapshot is not a directory")
        operational_error = True
        return _row(
            model_id=model_id,
            revision=revision,
            snapshot_path=snapshot,
            cache_root=cache_root,
            family=_family(None, model_id),
            status="error",
            reasons=reasons,
            loader=loader,
        )
    config, config_reason, config_error = _read_config(snapshot, repository)
    if config_reason:
        reasons.append(config_reason)
        operational_error = operational_error or config_error
    family = _family(config, model_id)
    tokenizer_ok, tokenizer_reasons, tokenizer_error = _tokenizer_status(snapshot, repository)
    reasons.extend(tokenizer_reasons)
    operational_error = operational_error or tokenizer_error
    (
        weight_bytes,
        weight_files,
        weight_reasons,
        weight_error,
        weight_warnings,
        weight_selection,
    ) = _weight_status(snapshot, repository, loader=loader)
    reasons.extend(weight_reasons)
    operational_error = operational_error or weight_error
    if operational_error:
        status = "error"
    elif reasons or not tokenizer_ok or weight_files == 0:
        status = "incomplete"
    else:
        status = "available"
    return _row(
        model_id=model_id,
        revision=revision,
        snapshot_path=snapshot,
        cache_root=cache_root,
        family=family,
        weight_bytes=weight_bytes,
        weight_files=weight_files,
        status=status,
        reasons=reasons,
        warnings=weight_warnings,
        loader=loader,
        weight_selection=weight_selection,
    )


def _inspect_repository(
    repository: Path, cache_root: Path, *, loader: str
) -> list[dict[str, Any]]:
    model_id, model_reason = _model_id(repository)
    if model_id is None:
        return [
            _error_root(
                cache_root, model_reason or "invalid model cache entry", loader=loader
            )
        ]
    snapshots = repository / "snapshots"
    refs = repository / "refs"
    revisions: dict[str, Path] = {}
    reasons: list[str] = []
    try:
        if snapshots.is_dir():
            for entry in sorted(snapshots.iterdir(), key=lambda item: item.name):
                if entry.is_dir() or entry.is_symlink():
                    revisions.setdefault(entry.name, entry)
                else:
                    reasons.append(f"snapshot entry {entry.name} is not a directory")
        elif snapshots.exists():
            reasons.append("snapshots is not a directory")
        else:
            reasons.append("model cache entry has no snapshots directory")
    except OSError:
        reasons.append("snapshots cannot be listed")
    # Resolve every ref to its snapshot.  Snapshot enumeration above remains
    # authoritative, but ref-only entries are included so a cache with a
    # valid ref and a delayed snapshot is visible as incomplete.
    if refs.exists() and not refs.is_dir():
        reasons.append("refs is not a directory")
    elif refs.is_dir():
        try:
            for ref in sorted(refs.iterdir(), key=lambda item: item.name):
                if not ref.is_file():
                    reasons.append(f"ref {ref.name} is not a regular file")
                    continue
                try:
                    target = ref.read_text(encoding="ascii").strip()
                except (OSError, UnicodeError):
                    reasons.append(f"ref {ref.name} is not readable ASCII")
                    continue
                if not _REVISION.fullmatch(target):
                    reasons.append(f"ref {ref.name} contains an invalid revision")
                    continue
                revisions.setdefault(target, snapshots / target)
        except OSError:
            reasons.append("refs cannot be listed")
    if not revisions:
        return [
            _row(
                model_id=model_id,
                revision="",
                snapshot_path=snapshots,
                cache_root=cache_root,
                family=_family(None, model_id),
                status=(
                    "error"
                    if any(
                        "no snapshots" in reason or "not a directory" in reason
                        for reason in reasons
                    )
                    else "incomplete"
                ),
                reasons=reasons or ["model cache entry contains no snapshots"],
                loader=loader,
            )
        ]
    rows = [
        _inspect_snapshot(
            model_id=model_id,
            revision=revision,
            snapshot=snapshot,
            repository=repository,
            cache_root=cache_root,
            loader=loader,
        )
        for revision, snapshot in sorted(revisions.items())
    ]
    if reasons:
        # Repository-level ref/snapshot diagnostics belong on the affected
        # inventory rows and must not turn a valid snapshot into "available".
        for row in rows:
            row["reasons"] = list(dict.fromkeys([*row["reasons"], *reasons]))
            if row["status"] == "available":
                row["status"] = "incomplete"
    return rows


def _family_matches(actual: str, wanted: str | None) -> bool:
    if wanted is None:
        return True
    normalized_actual = actual.casefold()
    normalized_wanted = wanted.casefold()
    return (
        normalized_actual == normalized_wanted
        or normalized_actual.startswith(normalized_wanted + "_")
        or (normalized_wanted.isalpha() and normalized_actual.startswith(normalized_wanted))
    )


def discover_models(
    cache_roots: Sequence[Path] | None = None,
    family: str | None = None,
    *,
    loader: str = "hf",
) -> list[dict[str, Any]]:
    """Discover local model snapshots as deterministic, JSON-safe dictionaries.

    ``cache_roots`` is explicit by design: callers that want a project-local
    ``.friday-data/models/hub`` must pass it.  With no roots, only standard
    Hugging Face cache environment variables and the user's default cache are
    considered.  Missing roots produce no rows; malformed entries are returned
    with ``status`` ``error`` or ``incomplete`` and explanatory ``reasons``.
    ``loader="hf"`` treats a sharded index as authoritative.  The explicit
    ``loader="mlx_lm"`` mode mirrors MLX-LM's direct-child ``model*.safetensors``
    selection and reports index differences in ``warnings``; it remains a
    metadata inventory, not a tensor or hardware qualification.
    """

    if loader not in {"hf", "mlx_lm"}:
        raise ValueError("loader must be 'hf' or 'mlx_lm'")
    roots = _default_cache_roots() if cache_roots is None else _unique_paths(cache_roots)
    wanted_family = family.casefold() if isinstance(family, str) else None
    rows: list[dict[str, Any]] = []
    for root in roots:
        root = _path(root)
        if not root.exists():
            continue
        if not root.is_dir():
            rows.append(_error_root(root, "cache root is not a directory", loader=loader))
            continue
        try:
            repositories = [
                entry
                for entry in root.iterdir()
                if entry.name.startswith("models--")
            ]
        except OSError:
            rows.append(_error_root(root, "cache root cannot be listed", loader=loader))
            continue
        for repository in sorted(repositories, key=lambda item: item.name):
            inspected = _inspect_repository(repository, root, loader=loader)
            for row in inspected:
                if not _family_matches(row["family"], wanted_family):
                    continue
                rows.append(row)
    # A snapshot may be reachable through multiple cache roots (for example,
    # an explicit project root and a symlinked default cache).  Resolve first,
    # then retain the lexicographically stable representative so root argument
    # order cannot change the result.
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        snapshot_key = os.path.normcase(str(_resolved(Path(row["snapshot_path"]))))
        prior = unique.get(snapshot_key)
        if prior is None or (
            row["cache_root"], row["model_id"], row["revision"]
        ) < (
            prior["cache_root"], prior["model_id"], prior["revision"]
        ):
            unique[snapshot_key] = row
    rows = list(unique.values())
    rows.sort(
        key=lambda row: (
            row["model_id"], row["revision"], row["snapshot_path"], row["cache_root"]
        )
    )
    return rows


__all__ = ["discover_models"]
