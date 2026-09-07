"""Dependency-free request and model contracts for the product service."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping
import uuid

from .errors import InvalidRequest


MAX_REQUEST_BYTES = 1024 * 1024
MAX_OUTPUT_TOKENS = 8192
MAX_MESSAGES = 256


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    snapshot_path: str
    weight_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "revision": self.revision,
                "snapshot_path": self.snapshot_path, "weight_bytes": self.weight_bytes}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelSpec":
        try:
            model_id, revision, path = (value[k] for k in ("model_id", "revision", "snapshot_path"))
            size = value["weight_bytes"]
        except (KeyError, TypeError) as exc:
            raise InvalidRequest("model registration is incomplete") from exc
        if any(not isinstance(v, str) or not v or len(v) > 4096 for v in (model_id, revision, path)):
            raise InvalidRequest("model identity fields must be bounded non-empty strings")
        if type(size) is not int or size <= 0:
            raise InvalidRequest("model weight_bytes must be a positive integer")
        if not Path(path).is_absolute():
            raise InvalidRequest("model snapshot path must be absolute")
        return cls(model_id, revision, path, size)


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    messages: tuple[tuple[str, str], ...]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    stop: tuple[str, ...] = ()
    stream: bool = False
    request_id: str = field(default_factory=lambda: "chatcmpl-" + uuid.uuid4().hex)

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.model, "messages": [{"role": r, "content": c} for r, c in self.messages],
                "max_tokens": self.max_tokens, "temperature": self.temperature,
                "top_p": self.top_p, "seed": self.seed, "stop": list(self.stop),
                "stream": self.stream, "request_id": self.request_id}

    @classmethod
    def from_payload(cls, payload: Any, *, exact: bool = True) -> "GenerationRequest":
        if not isinstance(payload, dict):
            raise InvalidRequest("request body must be a JSON object")
        known = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature",
                 "top_p", "seed", "stop", "stream", "n"}
        if set(payload) - known:
            raise InvalidRequest("unsupported request parameters: " + ", ".join(sorted(set(payload) - known)))
        model = payload.get("model")
        if not isinstance(model, str) or not model or len(model) > 512:
            raise InvalidRequest("model must identify a registered local model")
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not 1 <= len(raw_messages) <= MAX_MESSAGES:
            raise InvalidRequest("messages must be a non-empty list of at most 256 messages")
        messages = []
        for message in raw_messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise InvalidRequest("each message must contain only role and text content")
            role, content = message["role"], message["content"]
            if role not in ("system", "user", "assistant") or not isinstance(content, str):
                raise InvalidRequest("supported roles are system, user and assistant with text content")
            messages.append((role, content))
        try:
            content_bytes = sum(len(c.encode("utf-8")) for _, c in messages)
        except UnicodeError as exc:
            raise InvalidRequest("message content must contain valid Unicode") from exc
        if content_bytes > MAX_REQUEST_BYTES:
            raise InvalidRequest("message content exceeds the request limit")
        if "max_tokens" in payload and "max_completion_tokens" in payload:
            raise InvalidRequest("specify only one token limit")
        maximum = payload.get("max_tokens", payload.get("max_completion_tokens", 128))
        if type(maximum) is not int or not 1 <= maximum <= MAX_OUTPUT_TOKENS:
            raise InvalidRequest("token limit must be an integer from 1 through 8192")
        temperature, top_p = payload.get("temperature", 0.0), payload.get("top_p", 1.0)
        for name, value, low, high in (("temperature", temperature, 0, 2), ("top_p", top_p, 0, 1)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise InvalidRequest(f"{name} is outside its supported finite range")
        if top_p == 0:
            raise InvalidRequest("top_p must be greater than zero")
        if exact and (temperature != 0 or top_p != 1):
            raise InvalidRequest("Exact mode requires temperature=0 and top_p=1")
        seed = payload.get("seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise InvalidRequest("seed must be an unsigned 32-bit integer")
        stop = payload.get("stop", [])
        if isinstance(stop, str):
            stop = [stop]
        if not isinstance(stop, list) or len(stop) > 4 or any(not isinstance(s, str) or not 1 <= len(s) <= 256 for s in stop):
            raise InvalidRequest("stop must contain at most four non-empty strings of at most 256 characters")
        stream = payload.get("stream", False)
        if type(stream) is not bool:
            raise InvalidRequest("stream must be a boolean")
        if type(payload.get("n", 1)) is not int or payload.get("n", 1) != 1:
            raise InvalidRequest("only n=1 is supported")
        return cls(model, tuple(messages), maximum, float(temperature), float(top_p), seed, tuple(stop), stream)
