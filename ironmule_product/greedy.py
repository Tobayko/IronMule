"""Finite greedy generation candidate for the pinned stock ``mlx_lm``.

This module is deliberately inert at import time.  The implementation is a
small, finite-loop derivative of ``mlx_lm==0.31.3``'s ``generate_step``.  It
keeps the stock prompt-prefill segmentation and all prefetches needed before a
visible token, but does not submit the forward following the final visible
token when its private cache will be discarded.

Attribution and license: derived from mlx-lm 0.31.3, Copyright © 2023-2024
Apple Inc., MIT licensed.  The complete notice is in
``mlx_lm-0.31.3.dist-info/licenses/LICENSE`` in the pinned environment.
This candidate is not a replacement for the stock reference path.

MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any, Iterator


STOCK_MLX_LM_VERSION = "0.31.3"
MAX_FINITE_TOKENS = 8192
_ALLOWED_OPTIONS = {"prefill_step_size"}


class GreedyCompatibilityError(RuntimeError):
    """The installed stock mlx-lm is not the source this candidate derives from."""


def _check_stock_version() -> None:
    try:
        installed = metadata.version("mlx-lm")
    except metadata.PackageNotFoundError as exc:
        raise GreedyCompatibilityError("mlx-lm is not installed; bounded candidate is unavailable") from exc
    if installed != STOCK_MLX_LM_VERSION:
        raise GreedyCompatibilityError(
            f"bounded candidate requires mlx-lm=={STOCK_MLX_LM_VERSION}; installed {installed}"
        )


def _validate_contract(max_tokens: Any, options: dict[str, Any]) -> int:
    if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_FINITE_TOKENS:
        raise ValueError(f"max_tokens must be an integer from 1 through {MAX_FINITE_TOKENS}")
    unsupported = set(options) - _ALLOWED_OPTIONS
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(
            "bounded_generate_step supports only a fresh private cache, default "
            f"greedy decoding, and prefill_step_size; unsupported: {names}"
        )
    prefill_step_size = options.get("prefill_step_size", 2048)
    if type(prefill_step_size) is not int or not 1 <= prefill_step_size <= MAX_FINITE_TOKENS:
        raise ValueError(f"prefill_step_size must be an integer from 1 through {MAX_FINITE_TOKENS}")
    return prefill_step_size


def bounded_generate_step(
    prompt: Any,
    model: Any,
    *,
    max_tokens: int = 256,
    **kwargs: Any,
) -> Iterator[tuple[int, Any]]:
    """Yield a finite exact-greedy sequence without the final unused prefetch.

    Only ``prefill_step_size`` may be supplied in ``kwargs``.  In particular,
    custom samplers/processors, draft models, embeddings, quantised KV, and
    caller-provided caches are rejected before importing MLX or touching the
    model.  The cache is always created privately and is discarded when this
    generator ends.
    """
    prefill_step_size = _validate_contract(max_tokens, kwargs)
    if len(prompt) == 0:
        raise ValueError("prompt must contain at least one token")
    _check_stock_version()

    # These imports intentionally remain inside the generator body.  Importing
    # ironmule_product.greedy must not initialise MLX or a Metal device.
    import mlx.core as mx
    from mlx_lm.generate import generation_stream
    from mlx_lm.models import cache

    if not isinstance(prompt, mx.array) or getattr(prompt, "ndim", None) != 1:
        raise ValueError("prompt must be a one-dimensional MLX array")

    prompt_cache = cache.make_prompt_cache(model)

    def model_call(input_tokens: Any) -> Any:
        return model(input_tokens, cache=prompt_cache)

    def step(input_tokens: Any) -> tuple[Any, Any]:
        with mx.stream(generation_stream):
            logits = model_call(input_tokens[None])
            logits = logits[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled = mx.argmax(logprobs, axis=-1)
            return sampled, logprobs.squeeze(0)

    # This is the stock prefill loop from mlx-lm 0.31.3.  Keeping its
    # segmentation and synchronization is essential for a fair candidate.
    with mx.stream(generation_stream):
        total_prompt_tokens = len(prompt)
        prompt_processed_tokens = 0
        while total_prompt_tokens - prompt_processed_tokens > 1:
            remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
            n_to_process = min(prefill_step_size, remaining)
            model_call(prompt[:n_to_process][None])
            mx.eval([c.state for c in prompt_cache])
            prompt_processed_tokens += n_to_process
            prompt = prompt[n_to_process:]
            mx.clear_cache()

        y, logprobs = step(prompt)

    mx.async_eval(y, logprobs)
    n = 0
    while n < max_tokens:
        # Match stock ordering for every non-final visible token.  The stock
        # loop performs this prefetch before yielding y; the candidate simply
        # omits it when y is the final token the caller can consume.
        if n < max_tokens - 1:
            next_y, next_logprobs = step(y)
            mx.async_eval(next_y, next_logprobs)
        if n == 0:
            mx.eval(y)
        yield y.item(), logprobs
        if n % 256 == 0:
            mx.clear_cache()
        if n == max_tokens - 1:
            break
        y, logprobs = next_y, next_logprobs
        n += 1


__all__ = [
    "GreedyCompatibilityError",
    "MAX_FINITE_TOKENS",
    "STOCK_MLX_LM_VERSION",
    "bounded_generate_step",
]
