"""The one place that touches a model: IronMule's engine behind the serve protocol.

IronMule is the engine F1 measured through, and the only one in this repository
that implements all four calibrated knobs plus prompt-lookup speculation
(``ironmule/runtime.py:Knobs``). Serving through anything else would mean the
device profile authorises knobs that were verified somewhere they are not used.

The checkout is pinned: a different commit is a different engine, and a profile
says nothing about it.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import mlx.core as mx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IRONMULE = PROJECT_ROOT / ".worktrees" / "friday-optimizer-ironmule"
EXPECTED_IRONMULE_HEAD = "03e884cb28a05d090d20844460fc3afc8e738a91"


class BackendError(RuntimeError):
    """The local engine cannot serve under the profile's assumptions."""


def ironmule_head() -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(IRONMULE), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        raise BackendError("bound IronMule checkout is unreadable")
    return completed.stdout.strip()


class IronMuleBackend:
    """One loaded model; one engine per distinct knob setting, built on demand."""

    def __init__(self, model, tokenizer, *, model_id: str, model_revision: str) -> None:
        sys.path.insert(0, str(IRONMULE))
        from ironmule import BASELINE, Engine, Knobs, PrefixCache
        from ironmule.runtime import _leaves, _lookup_draft, _state_is_hybrid

        self._Engine = Engine
        self._Knobs = Knobs
        self._PrefixCache = PrefixCache
        self._baseline = BASELINE
        self._leaves = _leaves
        self._lookup_draft = _lookup_draft
        self._state_is_hybrid = _state_is_hybrid
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.model_revision = model_revision
        self._engines: dict[str, Any] = {}
        self.prefix_cache: Any | None = None
        self.eos_ids = tuple(
            sorted(
                {
                    int(value)
                    for value in (
                        getattr(tokenizer, "eos_token_id", None),
                        *(getattr(tokenizer, "eos_token_ids", None) or ()),
                    )
                    if isinstance(value, int)
                }
            )
        )
        if not self.eos_ids:
            raise BackendError("tokenizer exposes no end-of-sequence id")

    @classmethod
    def load(cls, model_id: str) -> "IronMuleBackend":
        head = ironmule_head()
        if head != EXPECTED_IRONMULE_HEAD:
            raise BackendError(
                f"IronMule checkout is at {head}, expected {EXPECTED_IRONMULE_HEAD}"
            )
        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        from _bench import enforce_offline, resolve_local_model_snapshot

        enforce_offline()
        snapshot = resolve_local_model_snapshot(model_id)
        from mlx_lm import load

        model, tokenizer = load(str(snapshot.path))
        return cls(
            model, tokenizer, model_id=model_id, model_revision=snapshot.revision
        )

    def encode(self, prompt: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )
        values = rendered if isinstance(rendered, list) else self.tokenizer.encode(rendered)
        ids = [int(value) for value in values]
        if not ids or any(value < 0 for value in ids):
            raise BackendError("tokenizer returned invalid prompt IDs")
        return ids

    def set_prefix_cache(self, prefix_ids: Sequence[int] | None) -> None:
        """Configure stateful prefix caching across all engines."""
        if prefix_ids is None:
            self.prefix_cache = None
        elif isinstance(prefix_ids, self._PrefixCache):
            self.prefix_cache = prefix_ids
        else:
            self.prefix_cache = self._PrefixCache(list(prefix_ids))
        for engine in self._engines.values():
            engine.prefix_cache = self.prefix_cache

    def _engine(self, knobs: Mapping[str, Any]):
        settings = self._baseline if not knobs else self._Knobs(**dict(knobs))
        key = settings.key()
        engine = self._engines.get(key)
        if engine is None:
            engine = self._engines[key] = self._Engine(self.model, self.tokenizer, settings)
            engine.prefix_cache = self.prefix_cache
        return engine

    def generate(
        self, token_ids: Sequence[int], max_tokens: int, knobs: Mapping[str, Any]
    ) -> dict[str, Any]:
        engine = self._engine(knobs)
        result = dict(engine.generate(list(token_ids), max_tokens, self.eos_ids))
        if "prefix_cache_hits" not in result:
            result["prefix_cache_hits"] = (
                getattr(engine.prefix_cache, "hits", 0)
                if getattr(engine, "prefix_cache", None) is not None
                else 0
            )
        visible = [value for value in result["logical_tokens"] if value not in self.eos_ids]
        try:
            result["text"] = self.tokenizer.decode(visible)
        except Exception:
            result["text"] = None
        return result

    def _decode_text(self, tokens: Sequence[int]) -> str:
        visible = [value for value in tokens if value not in self.eos_ids]
        if not visible:
            return ""
        try:
            rendered = self.tokenizer.decode(visible)
            return rendered if isinstance(rendered, str) else ""
        except Exception:
            return ""

    def stream_generate(
        self, token_ids: Sequence[int], max_tokens: int, knobs: Mapping[str, Any]
    ) -> Iterator[dict[str, Any]]:
        engine = self._engine(knobs)
        prompt_ids = [int(t) for t in token_ids]
        capacity = engine._capacity(len(prompt_ids), max_tokens)

        # 1. Prefill
        started = time.perf_counter_ns()
        state, token = engine._prefill(prompt_ids, capacity)
        prefill_ns = time.perf_counter_ns() - started

        first = int(token.reshape((-1,)).item())
        hits = (
            getattr(engine.prefix_cache, "hits", 0)
            if getattr(engine, "prefix_cache", None) is not None
            else 0
        )
        first_text = self._decode_text([first])

        yield {
            "type": "token",
            "token": first,
            "tokens": [first],
            "text": first_text,
            "is_first": True,
            "prefill_ns": prefill_ns,
            "prefix_cache_hits": hits,
        }

        if first in self.eos_ids:
            yield {
                "type": "done",
                "total_tokens": 1,
                "decode_ns": 0,
                "total_ns": prefill_ns,
                "knobs": engine.knobs.as_dict(),
                "prefix_cache_hits": hits,
                "logical_tokens": [first],
            }
            return

        if max_tokens <= 1:
            yield {
                "type": "done",
                "total_tokens": 1,
                "decode_ns": 0,
                "total_ns": prefill_ns,
                "knobs": engine.knobs.as_dict(),
                "prefix_cache_hits": hits,
                "logical_tokens": [first],
            }
            return

        # 2. Decode loop
        logical_tokens = [first]
        decode_started = time.perf_counter_ns()

        if engine.knobs.speculate_k > 0:
            if self._state_is_hybrid(state):
                raise ValueError("speculative decoding is unsupported for hybrid cache state")
            width = engine.knobs.speculate_k + 1
            body = engine._body(capacity, width)
            sequence = list(prompt_ids) + [first]
            offset = len(prompt_ids) + 1
            current = first

            while len(logical_tokens) < max_tokens:
                draft = self._lookup_draft(
                    sequence, engine.knobs.speculate_ngram, engine.knobs.speculate_k
                )
                padded = (draft + [current] * engine.knobs.speculate_k)[: engine.knobs.speculate_k]
                out = body(mx.array([[current] + padded]), state)
                picks = engine._picks(out)
                state = out[1]
                mx.eval(picks, *self._leaves(state))
                mx.synchronize()
                chosen = picks.reshape((-1,)).tolist()

                accepted = [chosen[0]]
                for i in range(1, width):
                    if i - 1 < len(draft) and draft[i - 1] == chosen[i - 1]:
                        accepted.append(chosen[i])
                    else:
                        break

                accepted = accepted[: max_tokens - len(logical_tokens)]

                chunk_tokens = []
                hit_eos = False
                for tok in accepted:
                    chunk_tokens.append(tok)
                    if tok in self.eos_ids:
                        hit_eos = True
                        break

                logical_tokens.extend(chunk_tokens)
                sequence.extend(accepted)
                offset += len(accepted)
                current = accepted[-1]
                state["position"]["offset"] = mx.array(offset - 1, dtype=mx.int32)

                chunk_text = self._decode_text(chunk_tokens)
                yield {
                    "type": "token",
                    "token": chunk_tokens[-1],
                    "tokens": chunk_tokens,
                    "text": chunk_text,
                    "is_first": False,
                }

                if hit_eos:
                    break
        else:
            body = engine._body(capacity, 1)
            every = max(1, engine.knobs.readback_every)
            pending: list[Any] = []
            curr_token = token
            steps_remaining = max_tokens - 1

            for step in range(steps_remaining):
                out = body(curr_token, state)
                picks = engine._picks(out)
                curr_token, state = picks[:, -1:], out[1]
                pending.append(curr_token)

                if len(pending) == every or step == steps_remaining - 1:
                    mx.eval(*pending, *self._leaves(state))
                    mx.synchronize()
                    raw_chunk = [int(item.reshape((-1,)).item()) for item in pending]
                    pending = []

                    chunk_tokens = []
                    hit_eos = False
                    for tok in raw_chunk:
                        chunk_tokens.append(tok)
                        if tok in self.eos_ids:
                            hit_eos = True
                            break

                    logical_tokens.extend(chunk_tokens)
                    chunk_text = self._decode_text(chunk_tokens)

                    yield {
                        "type": "token",
                        "token": chunk_tokens[-1],
                        "tokens": chunk_tokens,
                        "text": chunk_text,
                        "is_first": False,
                    }

                    if hit_eos:
                        break

        decode_ns = time.perf_counter_ns() - decode_started
        total_ns = prefill_ns + decode_ns

        yield {
            "type": "done",
            "total_tokens": len(logical_tokens),
            "decode_ns": decode_ns,
            "total_ns": total_ns,
            "knobs": engine.knobs.as_dict(),
            "prefix_cache_hits": hits,
            "logical_tokens": logical_tokens,
        }


__all__ = ["BackendError", "EXPECTED_IRONMULE_HEAD", "IronMuleBackend", "ironmule_head"]
