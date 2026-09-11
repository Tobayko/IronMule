"""Fail-closed worker-side routing for an already loaded product model.

The caller supplies the complete ``runtime_identity`` and immutable integration
evidence.  ``source_manifest_sha256`` identifies the execution package manifest;
it is deliberately not a harness/study-file hash.  When omitted, the current
runtime code digest is used.  Construction and selection perform no I/O, model
loading, benchmarking, exploration, or evidence persistence.  ``engine_for`` is
the deliberately lazy heavy setup boundary: it reads/hashes exact model identity
and constructs an engine around the already loaded model, so its real cost remains
part of the caller's cold/setup timing.
"""

from __future__ import annotations

from dataclasses import asdict
import re
import secrets
import threading
from typing import Any, Iterable, Mapping, Sequence

from friday_evidence.canonical import canonical_sha256

from .prefix_session import WorkerPrefixSession
from .selection import (
    IntegrationEvidence,
    RequestContext,
    SelectionContractError,
    SelectionDecision,
    select_candidate,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ENGINE_CANDIDATES = frozenset({
    "baseline_interactive", "core_interactive",
    "baseline_throughput", "core_throughput",
})
_RUNTIME_IDENTITY_FIELDS = frozenset({
    "model_id", "revision", "weight_bytes", "model_files", "model_sha256",
    "environment_sha256", "code_sha256", "hardware_sha256", "environment",
    "hardware", "code_files", "identity_sha256",
})


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _prompt_length(prompt_ids: Sequence[int]) -> int:
    ids = tuple(prompt_ids)
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("prompt_ids must contain non-negative integer token IDs")
    return len(ids)


class AutomaticRuntime:
    """Select only currently qualified candidates for one loaded worker model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        spec: Any,
        identity: Mapping[str, Any],
        evidence: Iterable[Mapping[str, Any]],
        *,
        prefix_cache_max_entries: int,
        prefix_cache_max_bytes: int,
        source_manifest_sha256: str | None = None,
    ) -> None:
        if not isinstance(identity, Mapping):
            raise ValueError("identity must be the complete runtime_identity mapping")
        if set(identity) != _RUNTIME_IDENTITY_FIELDS:
            raise ValueError("identity must be the complete runtime_identity mapping")
        required = ("model_id", "revision", "model_sha256", "hardware_sha256",
                    "environment_sha256", "code_sha256", "identity_sha256")
        if any(not isinstance(identity.get(key), str) or not identity[key] for key in required):
            raise ValueError("identity is missing a required runtime binding")
        for key in required[2:]:
            if not _SHA256.fullmatch(identity[key]):
                raise ValueError(f"identity {key} must be a lowercase SHA-256")
        if (type(identity["weight_bytes"]) is not int or identity["weight_bytes"] < 1
                or not isinstance(identity["model_files"], list)
                or not isinstance(identity["code_files"], Mapping)
                or not isinstance(identity["environment"], Mapping)
                or not isinstance(identity["hardware"], Mapping)):
            raise ValueError("identity contains an invalid runtime binding")
        if (getattr(spec, "model_id", None) != identity["model_id"]
                or getattr(spec, "revision", None) != identity["revision"]
                or getattr(spec, "weight_bytes", None) != identity["weight_bytes"]):
            raise ValueError("model specification and runtime identity differ")
        if (canonical_sha256(identity["code_files"]) != identity["code_sha256"]
                or canonical_sha256(identity["environment"]) != identity["environment_sha256"]
                or canonical_sha256(identity["hardware"]) != identity["hardware_sha256"]):
            raise ValueError("runtime identity component digest differs")
        identity_payload = dict(identity)
        del identity_payload["identity_sha256"]
        if canonical_sha256(identity_payload) != identity["identity_sha256"]:
            raise ValueError("runtime identity digest differs")
        manifest = source_manifest_sha256 or identity["code_sha256"]
        if not isinstance(manifest, str) or not _SHA256.fullmatch(manifest):
            raise ValueError("source_manifest_sha256 must be a lowercase SHA-256")

        parsed: list[IntegrationEvidence] = []
        rejected = 0
        try:
            rows = tuple(evidence)
        except Exception:
            rows = ()
            rejected = 1
        for row in rows:
            try:
                parsed.append(IntegrationEvidence.from_mapping(row))
            except (SelectionContractError, TypeError, AttributeError):
                # An unknown or malformed record can never activate a candidate.
                rejected += 1

        self._model = model
        self._tokenizer = tokenizer
        self._spec = spec
        self._identity = dict(identity)
        self._source_manifest_sha256 = manifest
        self._evidence = tuple(parsed)
        self._rejected_evidence_records = rejected
        self._engines: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._prefix_closed = False
        self._last_decision: dict[str, Any] | None = None
        self.prefix_session = WorkerPrefixSession(
            model, tokenizer,
            binding_sha256=identity["identity_sha256"],
            scope_nonce=secrets.token_hex(16),
            max_entries=_positive_int(prefix_cache_max_entries, "prefix_cache_max_entries"),
            max_bytes=_positive_int(prefix_cache_max_bytes, "prefix_cache_max_bytes"),
        )

    def _context(self, prompt_ids: Sequence[int], max_tokens: int, *, stream: bool,
                 session_requests: int, group_width: int) -> RequestContext:
        return RequestContext.from_mapping({
            "model_id": self._identity["model_id"],
            "model_revision": self._identity["revision"],
            "model_sha256": self._identity["model_sha256"],
            "hardware_sha256": self._identity["hardware_sha256"],
            "environment_sha256": self._identity["environment_sha256"],
            "code_sha256": self._identity["code_sha256"],
            "source_manifest_sha256": self._source_manifest_sha256,
            "profile": "throughput" if group_width > 1 else "interactive",
            "session_requests": session_requests,
            "context_tokens": _prompt_length(prompt_ids),
            "max_new_tokens": max_tokens,
            "stream": stream,
            "group_width": group_width,
            "prefix_hit": self.prefix_session.reuse_available(prompt_ids),
        })

    def _remember(self, decision: SelectionDecision) -> SelectionDecision:
        record = asdict(decision)
        record["rejected_evidence_records"] = self._rejected_evidence_records
        self._last_decision = record
        return decision

    @property
    def last_decision(self) -> dict[str, Any] | None:
        return None if self._last_decision is None else dict(self._last_decision)

    def choose(self, prompt_ids: Sequence[int], max_tokens: int, *, stream: bool = False,
               session_requests: int = 1, group_width: int = 1) -> SelectionDecision:
        if self._closed:
            raise RuntimeError("automatic runtime is closed")
        context = self._context(prompt_ids, max_tokens, stream=stream,
                                session_requests=session_requests, group_width=group_width)
        if self._rejected_evidence_records:
            return self._remember(SelectionDecision(
                "reference", "rejected_evidence_metadata_fails_closed", None))
        return self._remember(select_candidate(context, self._evidence))

    def choose_many(self, prompt_ids: Sequence[Sequence[int]],
                    max_tokens: int | Sequence[int], *, stream: bool = False,
                    session_requests: int | None = None,
                    group_width: int | None = None) -> SelectionDecision:
        if self._closed:
            raise RuntimeError("automatic runtime is closed")
        prompts = tuple(prompt_ids)
        if not prompts:
            return self._remember(SelectionDecision(
                "reference", "empty_batch_uses_qualified_reference", None))
        requests = len(prompts) if session_requests is None else session_requests
        if type(requests) is not int or requests < len(prompts):
            raise ValueError("session_requests cannot undercount the supplied batch")
        width = min(4, len(prompts)) if group_width is None else group_width
        limits = ([max_tokens] * len(prompts) if type(max_tokens) is int
                  else list(max_tokens))
        if len(limits) != len(prompts):
            raise ValueError("max_tokens must be one integer or match prompt_ids length")
        contexts = [self._context(ids, limit, stream=stream,
                                  session_requests=requests, group_width=width)
                    for ids, limit in zip(prompts, limits)]
        if self._rejected_evidence_records:
            return self._remember(SelectionDecision(
                "reference", "rejected_evidence_metadata_fails_closed", None))
        decisions = [select_candidate(context, self._evidence) for context in contexts]
        first = decisions[0]
        if all(item == first for item in decisions):
            return self._remember(first)
        return self._remember(SelectionDecision(
            "reference", "mixed_batch_has_no_common_eligible_candidate", None))

    def engine_for(self, candidate_id: str) -> Any:
        if self._closed:
            raise RuntimeError("automatic runtime is closed")
        if candidate_id not in _ENGINE_CANDIDATES:
            raise ValueError("candidate does not name an explicit B39 engine configuration")
        with self._lock:
            bridge = self._engines.get(candidate_id)
            if bridge is None:
                # This is the only heavy import/construction path.  It reuses the
                # product-owned loaded model and never projects current_profile.
                from .engine_bridge import CurrentEngineBridge
                bridge = CurrentEngineBridge.from_loaded(
                    self._model, self._tokenizer, self._spec,
                    configuration=candidate_id,
                )
                self._engines[candidate_id] = bridge
            return bridge

    def close(self) -> None:
        if self._closed:
            return
        failures: list[BaseException] = []
        if not self._prefix_closed:
            try:
                self.prefix_session.close()
                self._prefix_closed = True
            except BaseException as exc:
                failures.append(exc)
        with self._lock:
            for candidate_id, engine in tuple(self._engines.items()):
                try:
                    engine.close()
                except BaseException as exc:
                    failures.append(exc)
                else:
                    del self._engines[candidate_id]
            if self._prefix_closed and not self._engines:
                self._closed = True
        if failures:
            raise failures[0]


__all__ = ["AutomaticRuntime"]
