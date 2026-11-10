"""GG1: the worker's bridge to the ggml engine (`ironmule/ggml_engine.py`), with the interface of
`CurrentEngineBridge` that the worker's streamed batch path uses: `generate_many` with per-request token callbacks,
joins (`admit`) and cancellation, `free_slots`, `request_limit`, `eos_ids`, `metadata` and `counters`."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .engine_bridge import EngineBridgeResult


class GgmlBridge:
    configuration = "ggml"

    def __init__(self, gguf: str, tokenizer: Any, *, speculate: bool, gate: dict[str, Any] | None = None):
        from ironmule.ggml_engine import CHECK_TEXT, GgmlEngine, file_sha256

        self._engine = GgmlEngine(gguf, speculate=speculate)
        try:
            hf = getattr(tokenizer, "_tokenizer", tokenizer)
            if self._engine.tokenize(CHECK_TEXT) != list(hf.encode(CHECK_TEXT, add_special_tokens=False)):
                raise ValueError("the GGUF's tokenizer gives other token ids than the registered model's")
            stops = set(getattr(tokenizer, "eos_token_ids", ()) or ())
            if not stops or not all(self._engine.eog(t) for t in stops):
                raise ValueError("the GGUF's end tokens differ from the registered model's")
            self._eos = tuple(sorted(stops))
        except BaseException:
            self._engine.close()
            raise
        self._tokenizer = tokenizer
        self._gguf_sha256 = file_sha256(gguf)
        self._gate = dict(gate) if gate else None
        self._speculate = speculate

    def metadata(self) -> dict[str, Any]:
        """Safe startup metadata; no request data."""
        return {"configuration": "ggml", "plan": "strict_one_shot",
                "ggml": {"gguf_sha256": self._gguf_sha256, "gate": self._gate,
                         "speculation": "context" if self._speculate else "off",
                         "answers": "may differ from the MLX plans (a Q8_0 GGUF, and speculation's multi-position "
                                    "decode), within the gate"}}

    def counters(self) -> dict[str, Any]:
        summary = self._engine.summary()
        return {"ggml_steps": summary["steps"], **({"speculation": summary["speculation"]}
                                                   if isinstance(summary["speculation"], dict) else {})}

    def eos_ids(self) -> tuple[int, ...]:
        return self._eos

    def request_limit(self) -> int:
        return self._engine.seq_ctx

    def free_slots(self) -> int:
        return self._engine.free_slots()

    def generate(self, prompt_ids: Sequence[int], max_tokens: int, on_token=None,
                 boundaries: Sequence[int] | None = None) -> EngineBridgeResult:
        return self.generate_many([prompt_ids], max_tokens, on_tokens=[on_token],
                                  boundaries=None if boundaries is None else [boundaries])[0]

    def generate_many(self, prompt_ids: Sequence[Sequence[int]], max_tokens: int | Sequence[int], on_tokens=None,
                      boundaries=None, admit: Callable[[int], list[Any]] | None = None,
                      cancelled=None) -> list[EngineBridgeResult]:
        from ironmule.ggml_engine import Job

        if boundaries is not None and any(cuts is not None for cuts in boundaries):
            raise ValueError("the ggml engine serves one-shot prompts")
        limits = [max_tokens] * len(prompt_ids) if type(max_tokens) is int else list(max_tokens)
        emitters = list(on_tokens) if on_tokens is not None else [None] * len(prompt_ids)
        stops = list(cancelled) if cancelled is not None else [None] * len(prompt_ids)
        jobs = [Job(list(ids), int(limit), emit, stop)
                for ids, limit, emit, stop in zip(prompt_ids, limits, emitters, stops)]

        def take(free: int) -> list[Job]:
            rows = admit(free) if admit is not None else []
            fresh = []
            for row in rows:
                ids, limit, emit = row[:3]
                if len(row) > 3 and row[3] is not None:
                    raise ValueError("the ggml engine serves one-shot prompts")
                fresh.append(Job(list(ids), int(limit), emit, row[4] if len(row) > 4 else None))
            return fresh

        served = self._engine.run(jobs, admit=take if admit is not None else None)
        reasons = {"eos": "stop", "length": "length", "cancelled": "cancelled"}
        results = []
        for job in served:
            text = self._tokenizer.decode([t for t in job.tokens if t not in self._eos])
            results.append(EngineBridgeResult(tokens=list(job.tokens), text=text,
                                              finish_reason=reasons[job.reason or "cancelled"],
                                              token_count=len(job.tokens), metrics={},
                                              metadata={**self.metadata(), **self.counters()}))
        return results

    def close(self) -> None:
        self._engine.close()
