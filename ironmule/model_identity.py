"""Exact, path-free identity for one resolved local model snapshot.

This module is stdlib-only.  Hugging Face cache discovery is supplied by the caller so
identity construction never imports a network-capable client or mutates global state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]*$")
_TOKENIZER_NAMES = {
    "added_tokens.json", "chat_template.json", "merges.txt", "special_tokens_map.json",
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json",
    "vocab.txt", "tiktoken.model",
}


class ModelIdentityError(ValueError):
    """A model source cannot be bound exactly under the D2 contract."""


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelIdentityError(f"{name} must be a non-empty string")
    return value.strip()


def _digest(name: str, value: Any) -> str:
    result = _text(name, value)
    if not _SHA256.fullmatch(result):
        raise ModelIdentityError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _revision(value: Any) -> str:
    result = _text("revision", value)
    if not _REVISION.fullmatch(result):
        raise ModelIdentityError("revision contains unsupported characters")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        keys = list(value)
        if any(not isinstance(key, str) for key in keys):
            raise ModelIdentityError("JSON object keys must be strings")
        return {key: _jsonable(value[key]) for key in sorted(keys)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelIdentityError("identity JSON forbids NaN and Infinity")
        return value
    raise ModelIdentityError(f"unsupported identity JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    allowed_root = root.parent.parent if root.parent.name == "snapshots" else root
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() and not path.is_file():
            raise ModelIdentityError(f"broken model symlink: {path.relative_to(root).as_posix()}")
        if not path.is_file():
            continue
        if path.is_symlink():
            try:
                path.resolve(strict=True).relative_to(allowed_root)
            except (OSError, ValueError) as exc:
                raise ModelIdentityError(
                    f"model symlink escapes its allowed root: {path.relative_to(root).as_posix()}"
                ) from exc
        relative = path.relative_to(root).as_posix()
        rows.append({"path": relative, "bytes": path.stat().st_size,
                     "sha256": _sha256_file(path)})
    if not rows:
        raise ModelIdentityError("model directory contains no files")
    return tuple(rows)


def _tokenizer_rows(rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    selected = tuple(
        row for row in rows
        if Path(row["path"]).name in _TOKENIZER_NAMES
        or Path(row["path"]).suffix == ".tiktoken"
    )
    if not selected:
        raise ModelIdentityError("model snapshot contains no tokenizer artifacts")
    return selected


def _config(root: Path) -> dict[str, Any]:
    path = root / "config.json"
    if not path.is_file():
        raise ModelIdentityError("model snapshot has no config.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelIdentityError("config.json is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ModelIdentityError("config.json root must be an object")
    return value


def _architecture(config: Mapping[str, Any]) -> str:
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type.strip():
        return model_type.strip()
    architectures = config.get("architectures")
    if (isinstance(architectures, list) and len(architectures) == 1
            and isinstance(architectures[0], str) and architectures[0].strip()):
        return architectures[0].strip()
    raise ModelIdentityError("model architecture is missing or ambiguous")


def _quantisation(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("quantization") or config.get("quantization_config")
    if not isinstance(value, dict):
        raise ModelIdentityError("model quantisation metadata is missing")
    value = _jsonable(value)
    for name in ("bits", "group_size"):
        field = value.get(name)
        if isinstance(field, bool) or not isinstance(field, int) or field <= 0:
            raise ModelIdentityError(f"quantisation.{name} must be a positive integer")
    return value


def _snapshot_revision(root: Path) -> str | None:
    return root.name if root.parent.name == "snapshots" and _REVISION.fullmatch(root.name) else None


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    SCHEMA: ClassVar[str] = "ironmule.model_identity.v1"

    model_id: str
    revision: str
    model_manifest_sha256: str
    architecture: str
    quantisation_json: str
    quantisation_sha256: str
    tokenizer_sha256: str
    manifest_file_count: int
    manifest_bytes: int
    tokenizer_file_count: int
    identity_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text("model_id", self.model_id))
        object.__setattr__(self, "revision", _revision(self.revision))
        object.__setattr__(self, "architecture", _text("architecture", self.architecture))
        for name in ("model_manifest_sha256", "quantisation_sha256", "tokenizer_sha256"):
            object.__setattr__(self, name, _digest(name, getattr(self, name)))
        for name in ("manifest_file_count", "manifest_bytes", "tokenizer_file_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ModelIdentityError(f"{name} must be a positive integer")
        try:
            quantisation = json.loads(self.quantisation_json)
        except json.JSONDecodeError as exc:
            raise ModelIdentityError("quantisation_json is invalid") from exc
        normalized = canonical_json(quantisation)
        if normalized != self.quantisation_json:
            raise ModelIdentityError("quantisation_json is not canonical")
        if canonical_sha256(quantisation) != self.quantisation_sha256:
            raise ModelIdentityError("quantisation digest does not match content")
        computed = canonical_sha256(self._semantic_dict())
        if self.identity_sha256 and self.identity_sha256 != computed:
            raise ModelIdentityError("identity digest does not match content")
        object.__setattr__(self, "identity_sha256", computed)

    @property
    def quantisation(self) -> dict[str, Any]:
        return json.loads(self.quantisation_json)

    def _semantic_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "model_id": self.model_id,
            "revision": self.revision,
            "model_manifest_sha256": self.model_manifest_sha256,
            "architecture": self.architecture,
            "quantisation": self.quantisation,
            "quantisation_sha256": self.quantisation_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "manifest_file_count": self.manifest_file_count,
            "manifest_bytes": self.manifest_bytes,
            "tokenizer_file_count": self.tokenizer_file_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "identity_sha256": self.identity_sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelIdentity":
        expected = {
            "schema", "model_id", "revision", "model_manifest_sha256", "architecture",
            "quantisation", "quantisation_sha256", "tokenizer_sha256",
            "manifest_file_count", "manifest_bytes", "tokenizer_file_count",
            "identity_sha256",
        }
        if not isinstance(data, Mapping) or set(data) != expected:
            raise ModelIdentityError("ModelIdentity fields are missing or unknown")
        if data["schema"] != cls.SCHEMA:
            raise ModelIdentityError("unsupported ModelIdentity schema")
        return cls(
            model_id=data["model_id"], revision=data["revision"],
            model_manifest_sha256=data["model_manifest_sha256"],
            architecture=data["architecture"],
            quantisation_json=canonical_json(data["quantisation"]),
            quantisation_sha256=data["quantisation_sha256"],
            tokenizer_sha256=data["tokenizer_sha256"],
            manifest_file_count=data["manifest_file_count"],
            manifest_bytes=data["manifest_bytes"],
            tokenizer_file_count=data["tokenizer_file_count"],
            identity_sha256=data["identity_sha256"],
        )


@dataclass(frozen=True, slots=True)
class ResolvedModelSource:
    path: Path
    identity: ModelIdentity

    def __post_init__(self) -> None:
        resolved = self.path.resolve()
        if not resolved.is_dir():
            raise ModelIdentityError("resolved model source is not a directory")
        object.__setattr__(self, "path", resolved)


def build_model_identity(model_id: str, source: Path, revision: str | None = None) -> ModelIdentity:
    root = source.resolve()
    if not root.is_dir():
        raise ModelIdentityError("model source is not a directory")
    rows = _files(root)
    manifest_sha256 = canonical_sha256(rows)
    tokenizer_rows = _tokenizer_rows(rows)
    config = _config(root)
    quantisation = _quantisation(config)
    snapshot_revision = _snapshot_revision(root)
    if revision is not None and snapshot_revision is not None and revision != snapshot_revision:
        raise ModelIdentityError("explicit revision does not match snapshot directory")
    resolved_revision = _revision(revision or snapshot_revision or f"local-{manifest_sha256[:16]}")
    public_id = (
        f"local:{root.name}" if Path(model_id).expanduser().is_dir() else _text("model_id", model_id)
    )
    return ModelIdentity(
        model_id=public_id,
        revision=resolved_revision,
        model_manifest_sha256=manifest_sha256,
        architecture=_architecture(config),
        quantisation_json=canonical_json(quantisation),
        quantisation_sha256=canonical_sha256(quantisation),
        tokenizer_sha256=canonical_sha256(tokenizer_rows),
        manifest_file_count=len(rows),
        manifest_bytes=sum(row["bytes"] for row in rows),
        tokenizer_file_count=len(tokenizer_rows),
    )


def scan_local_cache() -> Any:
    """Read the Hugging Face cache index read-only.

    A machine that has never downloaded a model has no cache directory at all, and
    `scan_cache_dir` raises for it. That is not an error here: no cache and an empty
    cache both mean "this model is not available locally", and the caller's own
    message says that far better than a traceback does.
    """
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.utils import CacheNotFound

    try:
        return scan_cache_dir()
    except CacheNotFound:
        return SimpleNamespace(repos=(), warnings=())


def select_cached_snapshot(cache: Any, model_id: str, revision: str | None = None) -> tuple[Path, str]:
    repositories = [repo for repo in cache.repos if repo.repo_id == model_id]
    candidates = [cached for repo in repositories for cached in repo.revisions]
    if revision is not None:
        candidates = [cached for cached in candidates if cached.commit_hash == revision]
    if len(candidates) != 1:
        detail = "requested revision" if revision is not None else "unique cached revision"
        # Three different situations reach this point and they need different advice.
        # Saying "not cached" about a model that is cached at another revision sends the
        # user to doubt their cache instead of their pin.
        if not candidates and not repositories:
            raise ModelIdentityError(
                f"model is not cached: no local snapshot of {model_id!r}\n\n"
                f"IronMule does not download models. Fetch it once yourself, then "
                f"re-run:\n"
                f"    hf download {model_id}"
                f"{f' --revision {revision}' if revision is not None else ''}\n"
                f"    ironmule models    # confirm it is cached"
            )
        if not candidates:
            raise ModelIdentityError(
                f"{model_id!r} is cached, but not at revision {revision!r}\n\n"
                f"Cached revisions:\n"
                f"    ironmule models --model {model_id}\n"
                f"Then either pin one of those, or fetch the one you asked for:\n"
                f"    hf download {model_id} --revision {revision}"
            )
        # More than one revision is cached and the caller pinned none. The CLI has no
        # --revision flag, so do not tell a CLI user to pass one they cannot pass.
        raise ModelIdentityError(
            f"expected exactly one {detail} for {model_id!r}, found {len(candidates)}"
            f"\n\nSeveral revisions are cached; `ironmule models --model {model_id}` "
            f"lists them. Pinning an exact revision is a Python API argument "
            f"(`Runtime.load(model_id=..., revision=...)`); the CLI has no --revision "
            f"flag yet."
        )
    selected = candidates[0]
    path = Path(selected.snapshot_path).resolve()
    if not path.is_dir():
        raise ModelIdentityError("cached snapshot directory is unavailable")
    return path, _revision(selected.commit_hash)


def resolve_model_source(
    model_id: str, *, revision: str | None = None, cache: Any | None = None,
) -> ResolvedModelSource:
    local = Path(model_id).expanduser()
    if local.is_dir():
        identity = build_model_identity(model_id, local, revision)
        return ResolvedModelSource(local, identity)
    if cache is None:
        raise ModelIdentityError("Hub model resolution requires a read-only cache index")
    path, selected_revision = select_cached_snapshot(cache, model_id, revision)
    return ResolvedModelSource(
        path, build_model_identity(model_id, path, selected_revision)
    )


__all__ = [
    "ModelIdentity", "ModelIdentityError", "ResolvedModelSource", "build_model_identity",
    "canonical_json", "canonical_sha256", "resolve_model_source",
    "scan_local_cache", "select_cached_snapshot",
]
