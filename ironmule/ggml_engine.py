"""GG1: llama.cpp's ggml kernels as a CPU engine, with IronMule's scheduling and learned speculation (BACKLOG GG).

The GGUF is served in process through `llama-cpp-python`'s ctypes bindings: one context of `SEQS` sequences of
`SEQ_CTX` tokens; up to `SEQS` requests in flight, each in its own sequence, and a request that arrives while others
run takes a free sequence at the next step. Each step is one decode call carrying every active request's last token
and the drafts `StepPolicy` (policy version 3) gives it, copied from the request's own prompt and answer and verified
in the same call; a request keeps its drafts up to the first that differs from the model's argmax, plus that argmax,
and the rejected positions are removed from its sequence. On ggml a decode of several positions takes another kernel
path than one position, so speculation can change answers slightly (GG0): it serves only behind `gate`, the owner's
condition (BACKLOG GG, 2026-11-10).

The bindings come from an installed `llama-cpp-python`, or from a directory named in `IRONMULE_GGML_BINDINGS` with
the built library in `LLAMA_CPP_LIB_PATH`; nothing here installs anything.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

SEQS, SEQ_CTX, MAX_ROWS, THREADS = 8, 2048, 16, 4
ENGINE_VERSION = 1
CHECK_TEXT = "IronMule checks that the GGUF's tokenizer gives the registered model's token ids: 0123 ÄÖÜ ß -> ok."


def bindings() -> Any:
    """`llama_cpp`, from an install or from `IRONMULE_GGML_BINDINGS` (an unpacked `llama-cpp-python` source)."""
    path = os.environ.get("IRONMULE_GGML_BINDINGS")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        import diskcache  # noqa: F401 - the bindings import it for a cache class this engine never uses
    except ImportError:
        sys.modules.setdefault("diskcache", types.SimpleNamespace(Cache=object))
    import llama_cpp

    return llama_cpp


def available() -> str | None:
    """Why this machine cannot serve the ggml engine, or None."""
    try:
        bindings()
    except Exception as exc:  # noqa: BLE001 - a missing or broken library is a reason, not a crash
        return f"llama.cpp's bindings are not available ({type(exc).__name__}: {exc})"[:300]
    return None


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Job:
    """One request: its prompt ids, limit, a callback for each token and an optional cancel predicate."""

    ids: list[int]
    limit: int
    emit: Callable[[int], None] | None = None
    cancelled: Callable[[], bool] | None = None
    tokens: list[int] = field(default_factory=list)
    reason: str | None = None
    first_ns: int = 0
    finished_ns: int = 0


class GgmlEngine:
    def __init__(self, gguf: str, *, threads: int = THREADS, seqs: int = SEQS, seq_ctx: int = SEQ_CTX,
                 speculate: bool = True, max_rows: int = MAX_ROWS):
        import numpy as np

        from .speculation import StepPolicy

        L = bindings()
        self.L, self.np = L, np
        L.llama_backend_init()
        self.model = L.llama_model_load_from_file(str(gguf).encode(), L.llama_model_default_params())
        if not self.model:
            raise ValueError(f"llama.cpp could not load {gguf}")
        params = L.llama_context_default_params()
        params.n_ctx, params.n_batch, params.n_ubatch = seqs * seq_ctx, seq_ctx, 512
        params.n_seq_max, params.n_threads, params.n_threads_batch = seqs, threads, threads
        self.ctx = L.llama_init_from_model(self.model, params)
        if not self.ctx:
            raise ValueError("llama.cpp could not create a context")
        self.vocab = L.llama_model_get_vocab(self.model)
        self.n_vocab = L.llama_vocab_n_tokens(self.vocab)
        self.memory = L.llama_get_memory(self.ctx)
        self.seqs, self.seq_ctx, self.threads = seqs, seq_ctx, threads
        self.batch = L.llama_batch_init(max(seq_ctx, max_rows), 0, seqs)
        self.policy = StepPolicy(max_rows) if speculate else None
        self._free = list(range(seqs))
        self._lock = threading.Lock()
        self.steps = 0

    # --- the model's vocabulary -------------------------------------------------------------------------------------
    def tokenize(self, text: str) -> list[int]:
        raw = text.encode()
        size = len(raw) + 16
        buf = (self.L.llama_token * size)()
        n = self.L.llama_tokenize(self.vocab, raw, len(raw), buf, size, False, True)
        if n < 0:
            raise ValueError("llama.cpp could not tokenize the text")
        return list(buf[:n])

    def eog(self, token: int) -> bool:
        return bool(self.L.llama_vocab_is_eog(self.vocab, token))

    def eos_ids(self, candidates: Sequence[int] = ()) -> tuple[int, ...]:
        found = {t for t in candidates if 0 <= t < self.n_vocab and self.eog(t)}
        eos = self.L.llama_vocab_eos(self.vocab)
        if eos >= 0:
            found.add(int(eos))
        return tuple(sorted(found))

    # --- decode calls -------------------------------------------------------------------------------------------------
    def _decode(self, rows: list[tuple[int, int, int, bool]]) -> list[int]:
        b = self.batch
        b.n_tokens = len(rows)
        for i, (token, pos, seq, logits) in enumerate(rows):
            b.token[i], b.pos[i], b.n_seq_id[i], b.logits[i] = token, pos, 1, int(logits)
            b.seq_id[i][0] = seq
        if self.L.llama_decode(self.ctx, b) != 0:
            raise RuntimeError("llama_decode failed")
        out = []
        for i, (_, _, _, logits) in enumerate(rows):
            if logits:
                ptr = self.L.llama_get_logits_ith(self.ctx, i)
                out.append(int(self.np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).argmax()))
        return out

    def _prefill(self, ids: list[int], seq: int) -> int:
        first = None
        for start in range(0, len(ids), self.seq_ctx):
            chunk = ids[start:start + self.seq_ctx]
            last = start + len(chunk) == len(ids)
            picks = self._decode([(t, start + j, seq, last and j == len(chunk) - 1) for j, t in enumerate(chunk)])
            first = picks[-1] if picks else first
        return first

    def free_slots(self) -> int:
        with self._lock:
            return len(self._free)

    # --- serving ------------------------------------------------------------------------------------------------------
    def run(self, jobs: list[Job], admit: Callable[[int], list[Job]] | None = None) -> list[Job]:
        """Serve `jobs` and every job `admit(free sequences)` hands over while they run; returns all, in order."""
        from .speculation import MAX_DRAFT, accept, draft

        served: list[Job] = []
        pending = list(jobs)
        active: dict[int, Job] = {}

        def finish(seq: int, job: Job, reason: str) -> None:
            job.reason, job.finished_ns = reason, time.monotonic_ns()
            self.L.llama_memory_seq_rm(self.memory, seq, -1, -1)
            active.pop(seq, None)
            with self._lock:
                self._free.append(seq)
                self._free.sort()

        def take(seq: int, job: Job, token: int) -> bool:
            """Accept one token; True when the job is done."""
            if job.cancelled is not None and job.cancelled():
                finish(seq, job, "cancelled")
                return True
            job.tokens.append(token)
            if job.first_ns == 0:
                job.first_ns = time.monotonic_ns()
            if job.emit is not None:
                job.emit(token)
            if self.eog(token):
                finish(seq, job, "eos")
                return True
            if len(job.tokens) >= job.limit:
                finish(seq, job, "length")
                return True
            return False

        try:
            while True:
                if admit is not None and self._free:
                    pending.extend(admit(len(self._free)))
                while pending and self._free:
                    job = pending.pop(0)
                    served.append(job)
                    if len(job.ids) + job.limit > self.seq_ctx:
                        raise ValueError(f"{len(job.ids) + job.limit} tokens do not fit a {self.seq_ctx}-token sequence")
                    if job.cancelled is not None and job.cancelled():
                        job.reason, job.finished_ns = "cancelled", time.monotonic_ns()
                        continue
                    with self._lock:
                        seq = self._free.pop(0)
                    active[seq] = job
                    take(seq, job, self._prefill(job.ids, seq))
                if not active:
                    if pending:
                        continue
                    if admit is not None:
                        fresh = admit(len(self._free))
                        if fresh:
                            pending.extend(fresh)
                            continue
                    break
                order = sorted(active)
                offers = []
                for seq in order:
                    job = active[seq]
                    pos = len(job.ids) + len(job.tokens) - 1
                    room = min(MAX_DRAFT, job.limit - len(job.tokens) - 1, self.seq_ctx - 1 - pos)
                    offers.append(draft(job.ids + job.tokens, room) if self.policy is not None and room > 0
                                  else ([], 0))
                ks = self.policy.choose(offers) if self.policy is not None else [0] * len(order)
                rows: list[tuple[int, int, int, bool]] = []
                for seq, (drafts, _), k in zip(order, offers, ks):
                    job = active[seq]
                    pos = len(job.ids) + len(job.tokens) - 1
                    rows += [(t, pos + j, seq, True) for j, t in enumerate([job.tokens[-1]] + list(drafts[:k]))]
                started = time.perf_counter()
                picks = self._decode(rows)
                if self.policy is not None:
                    self.policy.observe_step(len(rows), time.perf_counter() - started, any(ks))
                self.steps += 1
                at = 0
                for seq, (drafts, bucket), k in zip(order, offers, ks):
                    job = active[seq]
                    pos = len(job.ids) + len(job.tokens) - 1
                    new = accept(list(drafts[:k]), picks[at:at + k + 1])
                    at += k + 1
                    if k and self.policy is not None:
                        self.policy.observe_drafts(bucket, k, len(new) - 1)
                    if k:
                        self.L.llama_memory_seq_rm(self.memory, seq, pos + len(new), -1)  # rejected positions
                    for token in new:
                        if take(seq, job, token):
                            break
        finally:
            for seq in list(active):
                finish(seq, active[seq], active[seq].reason or "cancelled")
        return served

    # --- the quality gate ---------------------------------------------------------------------------------------------
    def chunk_nll(self, ids: list[int], lead: list[int]) -> float:
        """Mean next-token negative log-likelihood of `ids[1:]` after `lead + ids[:-1]`, in one multi-position decode
        (the verify path), as `numeric_choice.nll_per_chunk` scores a chunk."""
        np = self.np
        self.L.llama_memory_clear(self.memory, True)
        tokens = lead + ids[:-1]
        b = self.batch
        b.n_tokens = len(tokens)
        for i, token in enumerate(tokens):
            b.token[i], b.pos[i], b.n_seq_id[i], b.logits[i] = token, i, 1, int(i >= len(lead))
            b.seq_id[i][0] = 0
        if self.L.llama_decode(self.ctx, b) != 0:
            raise RuntimeError("llama_decode failed")
        total = 0.0
        for j, target in enumerate(ids[1:]):
            row = np.ctypeslib.as_array(self.L.llama_get_logits_ith(self.ctx, len(lead) + j),
                                        shape=(self.n_vocab,)).astype(np.float64)
            top = row.max()
            total += float(top + math.log(np.exp(row - top).sum()) - row[target])
        self.L.llama_memory_clear(self.memory, True)
        return total / (len(ids) - 1)

    def summary(self) -> dict[str, Any]:
        return {"engine": "ggml", "engine_version": ENGINE_VERSION, "threads": self.threads, "sequences": self.seqs,
                "sequence_tokens": self.seq_ctx, "steps": self.steps,
                **({"speculation": self.policy.summary()} if self.policy is not None else {"speculation": "off"})}

    def close(self) -> None:
        if getattr(self, "batch", None) is not None:
            self.L.llama_batch_free(self.batch)
            self.batch = None
        if getattr(self, "ctx", None):
            self.L.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            self.L.llama_model_free(self.model)
            self.model = None


GATE_REFERENCE = r"""
import json, sys
from types import SimpleNamespace
import mlx.core as mx
mx.set_default_device(mx.cpu)
from mlx_lm import load
from ironmule import cpu_blas
from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS, _text_ids, gate_bos, nll_per_chunk
from ironmule.numeric_plans import dequantize_model
model, tokenizer = load(sys.argv[1])
dequantize_model(model)
ids = _text_ids(tokenizer)
print(json.dumps({"serving": cpu_blas.active()["serving"], "ids": ids[:CHUNKS * CHUNK_TOKENS + 1],
                  "bos": gate_bos(tokenizer), "nll": nll_per_chunk(SimpleNamespace(model=model), ids, CHUNKS,
                                                                   CHUNK_TOKENS, gate_bos(tokenizer))}))
"""

GATE_CANDIDATE = r"""
import json, sys
from ironmule.ggml_engine import GgmlEngine
from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS, gate_lead
data = json.loads(sys.stdin.read())
engine = GgmlEngine(sys.argv[1], speculate=False)
ids, bos, rows = data["ids"], data["bos"], []
for index in range(CHUNKS):
    chunk = ids[index * CHUNK_TOKENS:(index + 1) * CHUNK_TOKENS + 1]
    if len(chunk) < CHUNK_TOKENS + 1:
        break
    rows.append(engine.chunk_nll(chunk, gate_lead(chunk, bos)))
engine.close()
print(json.dumps({"nll": rows}))
"""


def gate(snapshot: str, gguf: str, *, cache: Path | None = None, root: Path | None = None) -> dict[str, Any]:
    """The owner's condition: the GGUF's perplexity on the numeric gate's chunks, through the verify path, against
    the registered model widened to float32 on MLX (on OpenBLAS where this machine has it); passes when the ratio's
    95 % interval ends below `numeric_choice.QUALITY_BOUND`. Cached by the GGUF, snapshot and gate version."""
    import subprocess

    from . import cpu_blas
    from .numeric_choice import GATE_VERSION, ratio_interval

    root = root or Path(__file__).resolve().parents[1]
    key = hashlib.sha256(json.dumps([file_sha256(gguf), str(Path(snapshot).resolve()), GATE_VERSION, ENGINE_VERSION],
                                    sort_keys=True).encode()).hexdigest()[:24]
    cache = cache or Path(os.path.expanduser("~/.cache/ironmule/ggml_gate"))
    stored = cache / f"{key}.json"
    if stored.exists():
        try:
            return json.loads(stored.read_text())
        except (OSError, ValueError):
            pass
    variables, _ = cpu_blas.environment()
    env = {**os.environ, **variables, "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get(
        "PYTHONPATH")])), "HF_HUB_OFFLINE": "1"}
    started = time.perf_counter()
    reference = subprocess.run([sys.executable, "-c", GATE_REFERENCE, snapshot], env=env, capture_output=True,
                               text=True, timeout=7200, check=False)
    if reference.returncode != 0:
        return {"ran": False, "passes": False, "reason": f"the reference child failed: {reference.stderr[-300:]}"}
    ref = json.loads(reference.stdout.strip().splitlines()[-1])
    candidate = subprocess.run([sys.executable, "-c", GATE_CANDIDATE, gguf], env={**os.environ, "PYTHONPATH": env[
        "PYTHONPATH"]}, input=json.dumps({"ids": ref["ids"], "bos": ref["bos"]}), capture_output=True, text=True,
                               timeout=7200, check=False)
    if candidate.returncode != 0:
        return {"ran": False, "passes": False, "reason": f"the ggml child failed: {candidate.stderr[-300:]}"}
    cand = json.loads(candidate.stdout.strip().splitlines()[-1])
    result = {"ran": True, **ratio_interval(ref["nll"], cand["nll"]), "reference": "dequantize float32 on MLX",
              "reference_blas": ref["serving"], "seconds": round(time.perf_counter() - started, 1),
              "gate_version": GATE_VERSION, "key": key}
    try:
        cache.mkdir(parents=True, exist_ok=True)
        stored.write_text(json.dumps(result))
    except OSError:
        pass
    return result


__all__ = ["ENGINE_VERSION", "GgmlEngine", "Job", "available", "bindings", "file_sha256", "gate"]
