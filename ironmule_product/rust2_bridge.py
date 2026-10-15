"""RUST2-SRV: rust2's serve mode as the worker's engine on a CUDA host (opt-in, `--engine rust2`).

rust2 (experiments/rust2) decodes Qwen3 dense 4-bit checkpoints on one NVIDIA card with its own kernels and
CUDA graphs, optionally speculating with DeepSeek's DSpark draft; every answer is plain greedy decoding. The
binary runs as a child process that loads the model once and serves token-ID requests as JSON lines; this
bridge gives it the engine bridge's `generate(prompt_ids, max_tokens, on_token=...)` interface, so the
worker's streaming path, tokenizer, chat template and detokenizer stay the same. The binary path comes from
IRONMULE_RUST2_BINARY and an optional draft directory from IRONMULE_RUST2_DRAFT; a request longer than the
binary's position cache is refused by the binary, never truncated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
from types import SimpleNamespace
from typing import Any, Callable, Sequence


READY_TIMEOUT_S = 900.0  # loading weights, the draft and capturing graphs (44 s for Qwen3 8B on a T4)
TOKEN_TIMEOUT_S = 300.0  # longest silence allowed while a request runs (the prefill of 1024 positions included)


class Rust2Bridge:
    def __init__(self, binary: str, model_dir: str, eos_ids: Sequence[int], draft_dir: str | None = None,
                 arm: str = "k4", ready_timeout_s: float = READY_TIMEOUT_S, token_timeout_s: float = TOKEN_TIMEOUT_S):
        if not Path(binary).is_file():
            raise RuntimeError("rust2 binary is unavailable")
        spec = {"model": model_dir, "draft": draft_dir, "arm": arm}
        handle, self._spec_path = tempfile.mkstemp(prefix="rust2-", suffix=".json")
        with os.fdopen(handle, "w") as stream:
            json.dump(spec, stream)
        self._process = subprocess.Popen([binary, "serve", self._spec_path], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._lines: queue.Queue[str] = queue.Queue()
        self._token_timeout_s = token_timeout_s
        threading.Thread(target=self._pump, daemon=True).start()
        line = self._next(ready_timeout_s)
        ready = json.loads(line) if line else {}
        if ready.get("ready") is not True:
            self.close()
            raise RuntimeError("rust2 did not become ready")
        self._ready, self._eos, self._count = ready, sorted(set(int(t) for t in eos_ids)), 0

    def _pump(self) -> None:
        for line in self._process.stdout:
            self._lines.put(line)
        self._lines.put("")  # end of output

    def _next(self, timeout_s: float) -> str:
        try:
            return self._lines.get(timeout=timeout_s)
        except queue.Empty:
            self.close()
            raise RuntimeError(f"rust2 sent nothing for {timeout_s:.0f} s; stopped") from None

    def metadata(self) -> dict[str, Any]:
        return {"engine": "rust2", "speculative": bool(self._ready.get("speculative")),
                "max_positions": self._ready.get("max_positions")}

    def generate(self, prompt_ids: Sequence[int], max_tokens: int, on_token: Callable[[int], None] | None = None,
                 boundaries: Any = None) -> SimpleNamespace:
        if boundaries is not None:
            raise ValueError("rust2 does not reuse chat turns")
        self._count += 1
        rid = f"r{self._count}"
        self._process.stdin.write(json.dumps({"id": rid, "prompt": [int(t) for t in prompt_ids],
                                              "max_tokens": int(max_tokens), "stop": self._eos}) + "\n")
        self._process.stdin.flush()
        tokens: list[int] = []
        while True:
            line = self._next(self._token_timeout_s)
            if not line:
                raise RuntimeError("rust2 exited during a request")
            message = json.loads(line)
            if message.get("id") != rid:
                raise RuntimeError("rust2 answered another request")
            if "error" in message:
                raise ValueError(f"rust2 refused the request: {message['error']}")
            if "token" in message:
                tokens.append(int(message["token"]))
                if on_token is not None:
                    on_token(tokens[-1])
            elif message.get("done"):
                finish = message.get("finish")
                if finish not in ("stop", "length") or message.get("tokens") != len(tokens):
                    raise RuntimeError("rust2 completion is invalid")
                return SimpleNamespace(tokens=tokens, token_count=len(tokens), finish_reason=finish,
                                       metadata={**self.metadata(), "ttft_ms": message.get("ttft_ms"),
                                                 "decode_tps": message.get("decode_tps")})

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                process.stdin.close()
                process.wait(timeout=30)
            except Exception:  # noqa: BLE001 - shutdown must not raise
                process.kill()
                process.wait()
        try:
            os.unlink(self._spec_path)
        except OSError:
            pass
