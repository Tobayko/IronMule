"""MLX implementation of the registered baseline and head-skip generation paths."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from friday_evidence.canonical import canonical_sha256
from tools._bench import resolve_local_model_snapshot

from .constants import MODEL_ID, MODEL_REVISION, PREFILL_CHUNK
from .executor import GenerationOutput, GenerationRequest


class MlxBackendError(RuntimeError):
    """The local model cannot satisfy the closed runtime contract."""


class MlxGenerationBackend:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        mx_module: Any,
        cache_factory: Any,
        sampler: Any,
        model_id: str,
        model_revision: str,
        prefill_chunk: int = PREFILL_CHUNK,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.mx = mx_module
        self.cache_factory = cache_factory
        self.sampler = sampler
        self.model_id = model_id
        self.model_revision = model_revision
        self.prefill_chunk = prefill_chunk
        inner = model.language_model if hasattr(model, "language_model") else model
        self.body = getattr(inner, "model", None)
        self.head = getattr(inner, "lm_head", None)
        if self.body is None or self.head is None:
            raise MlxBackendError("model does not expose the registered body/head boundary")

    @classmethod
    def load_local(cls) -> "MlxGenerationBackend":
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler
        import mlx.core as mx

        snapshot = resolve_local_model_snapshot(MODEL_ID)
        if snapshot.revision != MODEL_REVISION:
            raise MlxBackendError("local model revision differs")
        model, tokenizer = load(str(snapshot.path))
        return cls(
            model=model,
            tokenizer=tokenizer,
            mx_module=mx,
            cache_factory=make_prompt_cache,
            sampler=make_sampler(temp=0.0),
            model_id=MODEL_ID,
            model_revision=snapshot.revision,
        )

    def encode_prompt(self, prompt_content: str) -> tuple[int, ...]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_content}],
            add_generation_prompt=True,
        )
        values = rendered if isinstance(rendered, list) else self.tokenizer.encode(rendered)
        result = tuple(int(value) for value in values)
        if not result or any(value < 0 for value in result):
            raise MlxBackendError("tokenizer returned invalid prompt IDs")
        return result

    def reset_peak_memory(self) -> None:
        function = getattr(self.mx, "reset_peak_memory", None)
        if callable(function):
            function()

    def memory_snapshot(self) -> dict[str, int | None]:
        result: dict[str, int | None] = {}
        for key, name in (
            ("mlx_active_memory_bytes", "get_active_memory"),
            ("mlx_peak_memory_bytes", "get_peak_memory"),
            ("mlx_cache_memory_bytes", "get_cache_memory"),
        ):
            function = getattr(self.mx, name, None)
            try:
                value = function() if callable(function) else None
            except Exception:
                value = None
            result[key] = value if type(value) is int and value >= 0 else None
        return result

    def _generate(
        self,
        token_ids: Sequence[int],
        request: GenerationRequest,
        *,
        skip_head: bool,
    ) -> GenerationOutput:
        if (
            request.batch != 1
            or request.temperature != 0.0
            or request.prompt_logprobs is not False
            or request.fixed_horizon is not True
            or request.output_tokens <= 0
        ):
            raise MlxBackendError(
                "the qualification adapter cannot preserve these request semantics"
            )
        output_tokens = request.output_tokens
        mx = self.mx
        cache = self.cache_factory(self.model)
        logits = None
        blocks = 0
        head_calls = 0
        total_started = time.perf_counter_ns()
        prefill_started = total_started
        for offset in range(0, len(token_ids), self.prefill_chunk):
            piece = mx.array([token_ids[offset : offset + self.prefill_chunk]])
            blocks += 1
            if skip_head:
                hidden = self.body(piece, cache=cache)
                is_last = offset + self.prefill_chunk >= len(token_ids)
                if is_last:
                    logits = self.head(hidden[:, -1:, :])
                    head_calls += 1
                    mx.eval(logits)
                else:
                    mx.eval(hidden)
            else:
                logits = self.model(piece, cache=cache)
                head_calls += 1
                mx.eval(logits)
            mx.synchronize()
        prefill_ended = time.perf_counter_ns()
        if logits is None:
            raise MlxBackendError("prefill produced no logits")
        y = self.sampler(logits[:, -1, :].astype(mx.float32))[:, None]
        mx.eval(y)
        output = [int(y[0, 0])]
        for _ in range(output_tokens - 1):
            next_logits = self.model(y, cache=cache)
            y = self.sampler(next_logits[:, -1, :].astype(mx.float32))[:, None]
            mx.eval(y)
            output.append(int(y[0, 0]))
        mx.synchronize()
        total_ended = time.perf_counter_ns()
        text: str | None
        try:
            text = self.tokenizer.decode(output)
        except Exception:
            text = None
        return GenerationOutput(
            token_ids=tuple(output),
            token_sha256=canonical_sha256(output),
            text=text if isinstance(text, str) else None,
            prefill_ns=prefill_ended - prefill_started,
            total_ns=total_ended - total_started,
            prefill_blocks=blocks,
            head_calls=head_calls,
            memory=self.memory_snapshot(),
        )

    def generate_baseline(
        self, token_ids: Sequence[int], request: GenerationRequest
    ) -> GenerationOutput:
        return self._generate(token_ids, request, skip_head=False)

    def generate_head_skip(
        self, token_ids: Sequence[int], request: GenerationRequest
    ) -> GenerationOutput:
        return self._generate(token_ids, request, skip_head=True)


__all__ = ["MlxBackendError", "MlxGenerationBackend"]
