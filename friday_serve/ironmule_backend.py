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
from pathlib import Path
from typing import Any, Mapping, Sequence

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

        self._Engine = Engine
        self._Knobs = Knobs
        self._PrefixCache = PrefixCache
        self._baseline = BASELINE
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


__all__ = ["BackendError", "EXPECTED_IRONMULE_HEAD", "IronMuleBackend", "ironmule_head"]
