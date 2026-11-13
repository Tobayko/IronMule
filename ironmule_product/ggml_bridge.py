"""GG1: the worker's bridge to the ggml engine (`ironmule/ggml_engine.py`), with the interface of
`CurrentEngineBridge` that the worker's streamed batch path uses: `generate_many` with per-request token callbacks,
joins (`admit`) and cancellation, `free_slots`, `request_limit`, `eos_ids`, `metadata` and `counters`."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Sequence

from .engine_bridge import EngineBridgeResult


class GgmlBridge:
    configuration = "ggml"

    def __init__(self, gguf: str, tokenizer: Any, *, speculate: bool, gate: dict[str, Any] | None = None):
        from ironmule.ggml_engine import CHECK_TEXT, GgmlEngine, file_sha256

        settings = {k: bool(v) for k, v in ((gate or {}).get("settings") or {}).items()
                    if k in ("flash_attn", "kv_unified", "kv_f32", "kv_q8")}  # GG1: what the first start measured, and the gate passed
        plan = (gate or {}).get("agent") or {}
        start = (plan.get("configs") or {}).get(plan.get("incumbent")) if os.environ.get(
            "IRONMULE_GGML_AGENT", "1") != "0" else None
        # AG10: the service's capacity is the widest configuration the agent may switch to; the engine starts at the
        # incumbent's width
        self._engine = GgmlEngine(gguf, speculate=speculate, **settings,
                                  **({"seqs": int(start["seqs"])} if start else {}))
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
        self._run_lock = threading.Lock()  # one run or one reconfiguration at a time
        self._last_run = time.monotonic()
        self._agent = None
        self._stop = threading.Event()
        self._start_agent()

    def _start_agent(self) -> None:
        """AG10: the agent learns, dreams and switches the engine's configuration among those the cli measured and the
        gate allowed (`gate["agent"]`); IRONMULE_GGML_AGENT=0 turns it off."""
        plan = (self._gate or {}).get("agent")
        if not plan or os.environ.get("IRONMULE_GGML_AGENT", "1") == "0" or len(plan.get("configs", {})) < 2:
            return
        from ironmule.ggml_agent import GgmlAgent
        from ironmule.ggml_engine import ENGINE_VERSION, machine_key

        configs, incumbent = plan["configs"], plan["incumbent"]
        if incumbent not in configs:
            return
        binding = {"gguf": self._gguf_sha256, "machine": machine_key(), "engine": ENGINE_VERSION,
                   "gpu_layers": self._engine.gpu_layers}
        self._agent = GgmlAgent(plan.get("root"), binding, configs, incumbent)
        serving = configs[self._agent.current]  # resumed: what this binding learned last time
        wanted = {k: bool(v) for k, v in (serving.get("settings") or {}).items() if v}
        if self._engine.seqs != int(serving["seqs"]) or wanted != {k: v for k, v in self._engine.settings.items() if v}:
            self._switch(self._agent.current)
        self._engine.observer = self._agent.observe_step
        threading.Thread(target=self._idle_loop, name="ironmule-ggml-agent", daemon=True).start()

    def _switch(self, name: str) -> None:
        config = self._agent.configs[name]
        with self._run_lock:
            self._engine.reconfigure(int(config["seqs"]), config.get("settings") or {})
        self._agent.switched(name)

    def _idle_loop(self) -> None:
        dreamt_at = -1  # dream again only after new traffic: a quiet server spends no CPU on it
        while not self._stop.wait(1.0):
            if self._run_lock.locked() or time.monotonic() - self._last_run < 3.0:
                continue
            if self._agent.requests == dreamt_at:
                continue
            dreamt_at = self._agent.requests
            try:
                target = self._agent.idle()
                if target is not None and target != self._agent.current and not self._run_lock.locked():
                    self._switch(target)
            except Exception:  # noqa: BLE001 - the agent never stops the server; the incumbent keeps serving
                self._agent.pending_switch = None

    def metadata(self) -> dict[str, Any]:
        """Safe startup metadata; no request data."""
        return {"configuration": "ggml", "plan": "strict_one_shot",
                "ggml": {"gguf_sha256": self._gguf_sha256, "gate": self._gate, "settings": self._engine.settings,
                         "speculation": "context" if self._speculate else "off",
                         "answers": "may differ from the MLX plans (a Q8_0 GGUF, and speculation's multi-position "
                                    "decode), within the gate"}}

    def counters(self) -> dict[str, Any]:
        summary = self._engine.summary()
        return {"ggml_steps": summary["steps"], **({"speculation": summary["speculation"]}
                                                   if isinstance(summary["speculation"], dict) else {}),
                **({"ggml_agent": self._agent.status()} if self._agent is not None else {})}

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
        arrived = {id(job): time.monotonic() for job in jobs}

        def take(free: int) -> list[Job]:
            rows = admit(free) if admit is not None else []
            fresh = []
            for row in rows:
                ids, limit, emit = row[:3]
                if len(row) > 3 and row[3] is not None:
                    raise ValueError("the ggml engine serves one-shot prompts")
                fresh.append(Job(list(ids), int(limit), emit, row[4] if len(row) > 4 else None))
                arrived[id(fresh[-1])] = time.monotonic()
            return fresh

        with self._run_lock:
            try:
                served = self._engine.run(jobs, admit=take if admit is not None else None)
            finally:
                self._last_run = time.monotonic()
        if self._agent is not None:  # the traffic memory: arrival, prompt and answer lengths, no text
            for job in served:
                if job.reason != "cancelled":
                    self._agent.observe_request(arrived.get(id(job), time.monotonic()), len(job.ids),
                                                max(1, len(job.tokens)))
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
        self._stop.set()
        with self._run_lock:
            self._engine.close()
