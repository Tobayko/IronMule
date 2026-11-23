"""A GGUF's own tokenizer and chat template, so that `ironmule serve --gguf` needs no registered Hugging Face snapshot.

A GGUF carries everything a chat server needs besides the weights: the vocabulary and its merges, the special tokens,
which tokens end a turn, and the chat template (`tokenizer.chat_template`). This loads the vocabulary alone through
llama.cpp (`vocab_only`: no weights, no context), tokenizes and detokenizes with llama.cpp's own code (the engine's ids
by construction), and renders the template with Jinja2 the way Hugging Face's `apply_chat_template` does (a sandboxed
environment, `trim_blocks` and `lstrip_blocks`, `raise_exception` and `strftime_now`). It offers the part of the
tokenizer interface the worker's ggml path uses: `apply_chat_template`, `encode`, `decode`, `eos_token_ids`,
`bos_token_id` and a streaming `detokenizer`. A group's prompts can be tokenized together (`tokenizer(texts)` and
`apply_chat_template` over a list of conversations, as Hugging Face's interface takes them): llama.cpp's tokenizer
releases the GIL, so a few threads tokenize them in parallel with the same ids.
"""

from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

GGUF_MODEL_PREFIX = "gguf/"
_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    """Threads for a group's tokenization (TK1, 2026-11-23: 32 benchmark prompts took 46 ms serially on four local
    cores and 12.5 ms on four threads, identical ids)."""
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1)), thread_name_prefix="gguf-tok")
    return _POOL


class GgufDetokenizer:
    """Streams text token by token; a character split over several tokens appears once it is complete."""

    def __init__(self, tokenizer: "GgufTokenizer"):
        self._tokenizer = tokenizer
        self.reset()

    def reset(self) -> None:
        self._pending = b""
        self.last_segment = ""
        self.tokens: list[int] = []

    def add_token(self, token: int) -> None:
        self.tokens.append(int(token))
        data = self._pending + self._tokenizer.piece(int(token))
        cut = _complete_prefix(data)
        self.last_segment, self._pending = data[:cut].decode("utf-8", "replace"), data[cut:]

    def finalize(self) -> None:
        self.last_segment, self._pending = self._pending.decode("utf-8", "replace"), b""


def _complete_prefix(data: bytes) -> int:
    """The length of `data` without a trailing incomplete UTF-8 sequence."""
    for back in range(1, min(4, len(data)) + 1):
        byte = data[-back]
        if byte & 0xC0 == 0x80:  # a continuation byte: keep looking for the lead byte
            continue
        need = 2 if byte & 0xE0 == 0xC0 else 3 if byte & 0xF0 == 0xE0 else 4 if byte & 0xF8 == 0xF0 else 1
        return len(data) - back if need > back else len(data)
    return len(data)


class GgufTokenizer:
    def __init__(self, gguf: str, bindings: Any = None):
        if bindings is None:
            from .ggml_engine import bindings as load_bindings

            bindings = load_bindings()
        L = self.L = bindings
        L.llama_backend_init()
        params = L.llama_model_default_params()
        params.vocab_only = True
        params.n_gpu_layers = 0
        self._model = L.llama_model_load_from_file(str(gguf).encode(), params)
        if not self._model:
            raise ValueError(f"llama.cpp could not read the vocabulary of {gguf}")
        self.vocab = L.llama_model_get_vocab(self._model)
        self.n_vocab = int(L.llama_vocab_n_tokens(self.vocab))
        raw = L.llama_model_chat_template(self._model, None)
        self.chat_template = raw.decode("utf-8") if raw else None
        bos = int(L.llama_vocab_bos(self.vocab))
        self.bos_token_id = bos if bos >= 0 else None
        self.add_bos = bool(L.llama_vocab_get_add_bos(self.vocab))
        eos = int(L.llama_vocab_eos(self.vocab))
        self.eos_token_id = eos if eos >= 0 else None
        # every token that ends generation (EOS, end of turn, ...), as llama-server stops on
        self.eos_token_ids = tuple(t for t in range(self.n_vocab) if L.llama_vocab_is_eog(self.vocab, t))
        self.bos_token = self.special_text(self.bos_token_id)
        self.eos_token = self.special_text(self.eos_token_id)

    def close(self) -> None:
        if getattr(self, "_model", None):
            self.L.llama_model_free(self._model)
            self._model = None

    # -- tokens and text -------------------------------------------------------------------------------------------
    def piece(self, token: int, special: bool = False) -> bytes:
        buf = ctypes.create_string_buffer(256)
        n = self.L.llama_token_to_piece(self.vocab, token, buf, 256, 0, special)
        if n < 0:  # a longer piece: ask again with room for it
            buf = ctypes.create_string_buffer(-n)
            n = self.L.llama_token_to_piece(self.vocab, token, buf, -n, 0, special)
        return buf.raw[:max(n, 0)]

    def special_text(self, token: int | None) -> str:
        return "" if token is None else self.piece(token, special=True).decode("utf-8", "replace")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Token ids of `text`; special tokens written in the text (`<|im_start|>`) are parsed as such, and BOS is
        added only with `add_special_tokens` where the GGUF asks for one."""
        raw = text.encode("utf-8")
        size = len(raw) + 16
        buf = (self.L.llama_token * size)()
        n = self.L.llama_tokenize(self.vocab, raw, len(raw), buf, size, bool(add_special_tokens and self.add_bos), True)
        if n < 0:
            size = -n
            buf = (self.L.llama_token * size)()
            n = self.L.llama_tokenize(self.vocab, raw, len(raw), buf, size, bool(add_special_tokens and self.add_bos),
                                      True)
        if n < 0:
            raise ValueError("llama.cpp could not tokenize the text")
        return list(buf[:n])

    def __call__(self, texts: str | Sequence[str], add_special_tokens: bool = False) -> dict[str, Any]:
        """Hugging Face's batch interface: {"input_ids": ids of each text}, tokenized in parallel threads."""
        if isinstance(texts, str):
            return {"input_ids": self.encode(texts, add_special_tokens)}
        texts = list(texts)
        if len(texts) < 2:
            return {"input_ids": [self.encode(text, add_special_tokens) for text in texts]}
        return {"input_ids": list(_pool().map(lambda text: self.encode(text, add_special_tokens), texts))}

    def decode(self, ids: Sequence[int]) -> str:
        return b"".join(self.piece(int(t)) for t in ids).decode("utf-8", "replace")

    @property
    def detokenizer(self) -> GgufDetokenizer:
        """A new streaming detokenizer (the worker keeps a pool of them, one per running stream)."""
        return GgufDetokenizer(self)

    # -- the chat template -----------------------------------------------------------------------------------------
    def apply_chat_template(self, messages: Sequence[dict[str, Any]], add_generation_prompt: bool = True,
                            tokenize: bool = True, **extra: Any) -> Any:
        if not self.chat_template:
            raise ValueError("this GGUF carries no chat template (tokenizer.chat_template); serve it with --model, "
                             "a registered model with the same tokenizer")
        if messages and all(isinstance(conversation, (list, tuple)) for conversation in messages):
            texts = [self.apply_chat_template(conversation, add_generation_prompt, tokenize=False, **extra)
                     for conversation in messages]  # a batch of conversations, as Hugging Face takes them
            return self(texts)["input_ids"] if tokenize else texts
        text = render(self.chat_template, messages, add_generation_prompt=add_generation_prompt,
                      bos_token=self.bos_token, eos_token=self.eos_token, **extra)
        return self.encode(text, add_special_tokens=False) if tokenize else text


@lru_cache(maxsize=8)
def _compile_template(template: str):
    """Compile the model's template once. Each render creates its own request context."""
    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message: str) -> None:
        raise TemplateError(message)

    def strftime_now(fmt: str) -> str:
        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals.update(raise_exception=raise_exception, strftime_now=strftime_now)
    return env.from_string(template)


def render(template: str, messages: Sequence[dict[str, Any]], **variables: Any) -> str:
    """A chat template rendered as Hugging Face's `apply_chat_template` renders it."""
    return _compile_template(template).render(messages=list(messages), **variables)


def gguf_model_id(gguf: str) -> str:
    """The model id a GGUF served on its own answers to: `gguf/` and the file name without `.gguf`."""
    name = Path(gguf).name
    return GGUF_MODEL_PREFIX + (name[:-5] if name.lower().endswith(".gguf") else name)


def is_gguf_spec(spec: Any) -> bool:
    """Whether a model spec names a GGUF file itself rather than a registered snapshot directory."""
    path = spec.get("snapshot_path") if isinstance(spec, dict) else getattr(spec, "snapshot_path", None)
    return isinstance(path, str) and path.lower().endswith(".gguf") and Path(path).is_file()


def gguf_revision(gguf: str) -> str:
    """The GGUF's content as a revision: its SHA-256."""
    from .ggml_engine import file_sha256

    return "sha256:" + file_sha256(gguf)


__all__ = ["GgufDetokenizer", "GgufTokenizer", "gguf_model_id", "gguf_revision", "is_gguf_spec", "render"]
