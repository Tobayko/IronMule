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
POLL = 50  # llama-server's default busy-wait level
ENGINE_VERSION = 2  # 2 (GG2): warm-up of 8 prompts, staggered prompts, the gate against llama.cpp's default
PROMPT_BUDGET = 96  # prompt tokens a step may carry beside the running requests' rows (whole prompts, at least one)
PASSED_OVER = 8  # a waiting request passed over this often by shorter prompts goes first
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
    passed_over: int = 0


class GgmlEngine:
    def __init__(self, gguf: str, *, threads: int = THREADS, seqs: int = SEQS, seq_ctx: int = SEQ_CTX,
                 speculate: bool = True, max_rows: int = MAX_ROWS, flash_attn: bool = False,
                 kv_unified: bool = False, kv_f32: bool = False):
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
        params.flash_attn, params.kv_unified = bool(flash_attn), bool(kv_unified)  # GG1: measured at the first start
        if kv_f32:  # GG1, second amendment: the KV cache in full precision (GGML_TYPE_F32) instead of f16
            params.type_k = params.type_v = 0
        self.settings = {"flash_attn": bool(flash_attn), "kv_unified": bool(kv_unified), "kv_f32": bool(kv_f32)}
        self.ctx = L.llama_init_from_model(self.model, params)
        if not self.ctx:
            raise ValueError("llama.cpp could not create a context")
        self._threadpool = self._attach_threadpool(threads, POLL)
        self.vocab = L.llama_model_get_vocab(self.model)
        self.n_vocab = L.llama_vocab_n_tokens(self.vocab)
        self.memory = L.llama_get_memory(self.ctx)
        self.seqs, self.seq_ctx, self.threads = seqs, seq_ctx, threads
        self.batch = L.llama_batch_init(max(seq_ctx, max_rows), 0, seqs)
        self.policy = StepPolicy(max_rows) if speculate else None
        self._free = list(range(seqs))
        self._lock = threading.Lock()
        self.steps = 0
        self._warm_up()

    def _warm_up(self) -> None:
        """As `llama-server` does at its start: one call of a prompt's size and one of a step's, so a request never pays
        for llama.cpp's first graphs and weight repacking; the cache is cleared after."""
        bos = max(0, int(self.L.llama_vocab_bos(self.vocab)))
        self._decode([(bos, j, 0, j == 63) for j in range(64)])
        self.L.llama_memory_clear(self.memory, True)
        # GG2: the first call that carries several sequences' prompts cost about 0.3 s once; pay it here
        self._decode([(bos, j, seq, j == 63) for seq in range(self.seqs) for j in range(64)])
        self._decode([(bos, 64, seq, True) for seq in range(self.seqs)])
        self._decode([(bos, 65, 0, True)])
        self.L.llama_memory_clear(self.memory, True)

    def _attach_threadpool(self, threads: int, poll: int) -> Any:
        """llama-server's own thread pool: `threads` threads that busy-wait up to `poll` (its default 50) between a
        graph's operations instead of sleeping; without it a context uses a pool that does not poll. Returns the pool,
        or None where the library does not offer it (the context's default pool then serves)."""
        import ctypes

        class Params(ctypes.Structure):
            _fields_ = [("cpumask", ctypes.c_bool * 512), ("n_threads", ctypes.c_int), ("prio", ctypes.c_int),
                        ("poll", ctypes.c_uint32), ("strict_cpu", ctypes.c_bool), ("paused", ctypes.c_bool)]

        try:
            folder = Path(self.L.llama_cpp._lib._name).parent if hasattr(self.L, "llama_cpp") else None
            folder = folder or Path(os.environ.get("LLAMA_CPP_LIB_PATH", ""))
            base = ctypes.CDLL(str(next(folder.glob("libggml-base.so*"))))
            cpu = ctypes.CDLL(str(next(folder.glob("libggml-cpu.so*"))))
            llama = ctypes.CDLL(str(next(folder.glob("libllama.so*"))))
            base.ggml_threadpool_params_init.argtypes = [ctypes.POINTER(Params), ctypes.c_int]
            cpu.ggml_threadpool_new.argtypes = [ctypes.POINTER(Params)]
            cpu.ggml_threadpool_new.restype = ctypes.c_void_p
            llama.llama_attach_threadpool.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            params = Params()
            base.ggml_threadpool_params_init(ctypes.byref(params), threads)
            params.poll = poll
            pool = cpu.ggml_threadpool_new(ctypes.byref(params))
            if not pool:
                return None
            llama.llama_attach_threadpool(ctypes.c_void_p(self.ctx), ctypes.c_void_p(pool), ctypes.c_void_p(pool))
            self._free_pool = lambda: (llama.llama_detach_threadpool(ctypes.c_void_p(self.ctx)),
                                       setattr(cpu.ggml_threadpool_free, "argtypes", [ctypes.c_void_p]),
                                       cpu.ggml_threadpool_free(ctypes.c_void_p(pool)))
            return pool
        except Exception:  # noqa: BLE001 - the default pool still serves
            return None

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

        served: list[Job] = list(jobs)
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
                    more = admit(len(self._free))
                    served.extend(more)
                    pending.extend(more)
                # GG2: the requests that start in this step: the shortest prompts first, within the step's prompt
                # budget (whole prompts, at least one), and one passed over too often first
                fresh: list[tuple[int, Job]] = []
                budget = PROMPT_BUDGET
                for job in sorted(pending, key=lambda j: (j.passed_over < PASSED_OVER, len(j.ids))):
                    if job.cancelled is not None and job.cancelled():
                        job.reason, job.finished_ns = "cancelled", time.monotonic_ns()
                        pending.remove(job)
                        continue
                    if not self._free or (fresh and len(job.ids) > budget):
                        job.passed_over += 1
                        continue
                    if len(job.ids) + job.limit > self.seq_ctx:
                        raise ValueError(f"{len(job.ids) + job.limit} tokens do not fit a {self.seq_ctx}-token sequence")
                    with self._lock:
                        seq = self._free.pop(0)
                    pending.remove(job)
                    budget -= len(job.ids)
                    fresh.append((seq, job))
                if not active and not fresh:
                    if pending:
                        continue
                    if admit is not None:
                        more = admit(len(self._free))
                        if more:
                            served.extend(more)
                            pending.extend(more)
                            continue
                    break
                # One call per step, as llama-server batches: every active request's last token and its drafts, and
                # the prompts of the requests that start now (beyond one call's tokens, the rest go first on their own).
                order = sorted(active)
                offers = []
                for seq in order:
                    job = active[seq]
                    pos = len(job.ids) + len(job.tokens) - 1
                    room = min(MAX_DRAFT, job.limit - len(job.tokens) - 1, self.seq_ctx - 1 - pos)
                    offers.append(draft(job.ids + job.tokens, room) if self.policy is not None and room > 0
                                  else ([], 0))
                ks = self.policy.choose(offers) if self.policy is not None and order else [0] * len(order)
                rows: list[tuple[int, int, int, bool]] = []
                for seq, (drafts, _), k in zip(order, offers, ks):
                    job = active[seq]
                    pos = len(job.ids) + len(job.tokens) - 1
                    rows += [(t, pos + j, seq, True) for j, t in enumerate([job.tokens[-1]] + list(drafts[:k]))]
                decode_rows = len(rows)
                room = self.seq_ctx - decode_rows  # one call's tokens
                batched: list[tuple[int, Job]] = []
                for seq, job in fresh:
                    if len(job.ids) <= room:
                        rows += [(t, j, seq, j == len(job.ids) - 1) for j, t in enumerate(job.ids)]
                        room -= len(job.ids)
                        batched.append((seq, job))
                    else:  # too long to share this call: its own prefill first
                        active[seq] = job
                        take(seq, job, self._prefill(job.ids, seq))
                started = time.perf_counter()
                picks = self._decode(rows) if rows else []
                if self.policy is not None and decode_rows and not batched:  # a prompt in the call is not a step's price
                    self.policy.observe_step(decode_rows, time.perf_counter() - started, any(ks))
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
                for seq, job in batched:  # each prompt asked for logits at its last position only
                    active[seq] = job
                    take(seq, job, picks[at])
                    at += 1
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
        return {"engine": "ggml", "engine_version": ENGINE_VERSION, "settings": dict(self.settings),
                "threadpool_poll": POLL if self._threadpool else None,
                "threads": self.threads, "sequences": self.seqs,
                "sequence_tokens": self.seq_ctx, "steps": self.steps,
                **({"speculation": self.policy.summary()} if self.policy is not None else {"speculation": "off"})}

    def close(self) -> None:
        if getattr(self, "batch", None) is not None:
            self.L.llama_batch_free(self.batch)
            self.batch = None
        if getattr(self, "_threadpool", None) and getattr(self, "ctx", None):
            try:
                self._free_pool()
            except Exception:  # noqa: BLE001 - freeing the context frees what is left
                pass
            self._threadpool = None
        if getattr(self, "ctx", None):
            self.L.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            self.L.llama_model_free(self.model)
            self.model = None


SETTINGS = tuple({"flash_attn": fa, "kv_unified": unified, "kv_f32": f32}
                 for fa in (False, True) for unified in (False, True) for f32 in (False, True))


def autotune(gguf: str, steps: int = 24, repeats: int = 3) -> dict[str, Any]:
    """GG1 (amendment of 2026-11-10): llama.cpp's flash attention and unified KV cache, each on and off, timed on a fixed
    synthetic workload (8 sequences of a fixed 64-token prompt and one sequence alone, `steps` decode steps each,
    median of `repeats`); the setting with the highest geometric mean of the two rates."""
    rows = []
    for setting in SETTINGS:
        engine = GgmlEngine(gguf, speculate=False, **setting)
        try:
            prompt = (engine.tokenize(CHECK_TEXT) * 8)[:64]
            rates = {}
            for width in (8, 1):
                times = []
                for _ in range(repeats):
                    engine.L.llama_memory_clear(engine.memory, True)
                    engine._decode([(t, j, seq, j == len(prompt) - 1) for seq in range(width)
                                    for j, t in enumerate(prompt)])
                    started = time.perf_counter()
                    for k in range(steps):
                        engine._decode([(prompt[k % len(prompt)], len(prompt) + k, seq, True) for seq in range(width)])
                    times.append(time.perf_counter() - started)
                rates[str(width)] = width * steps / statistics.median(times)
        finally:
            engine.close()
        rows.append({**setting, "tokens_per_s": rates, "score": math.sqrt(rates["8"] * rates["1"])})
    rows.sort(key=lambda row: -row["score"])
    return {"order": [{k: row[k] for k in ("flash_attn", "kv_unified", "kv_f32")} for row in rows], "measured": rows}


def machine_key() -> str:
    """What makes a measured setting this machine's: architecture, CPU count and the CPU's model name."""
    import platform

    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{platform.machine()}|{os.cpu_count()}|{model}"


TUNE_CHILD = r"""
import json, sys
from ironmule.ggml_engine import autotune
print(json.dumps(autotune(sys.argv[1])))
"""

GATE_CANDIDATE = r"""
import json, sys
from ironmule.ggml_engine import GgmlEngine
from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS, gate_lead
data = json.loads(sys.stdin.read())
engine = GgmlEngine(sys.argv[1], speculate=False, **data["settings"])
ids, bos, rows = data["ids"], data["bos"], []
for index in range(CHUNKS):
    chunk = ids[index * CHUNK_TOKENS:(index + 1) * CHUNK_TOKENS + 1]
    if len(chunk) < CHUNK_TOKENS + 1:
        break
    rows.append(engine.chunk_nll(chunk, gate_lead(chunk, bos)))
engine.close()
print(json.dumps({"nll": rows}))
"""


GATE_IDS = r"""
import json, sys
from mlx_lm.tokenizer_utils import load
from pathlib import Path
from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS, _text_ids, gate_bos
tokenizer = load(Path(sys.argv[1]))
print(json.dumps({"ids": _text_ids(tokenizer)[:CHUNKS * CHUNK_TOKENS + 1], "bos": gate_bos(tokenizer)}))
"""
DEFAULT_SETTINGS = {"flash_attn": False, "kv_unified": False, "kv_f32": False}


def gate(snapshot: str, gguf: str, *, cache: Path | None = None, root: Path | None = None) -> dict[str, Any]:
    """The owner's bar (2026-11-10): answer quality at least llama.cpp's. The engine's settings are timed
    (`autotune`), and the fastest one whose perplexity on the numeric gate's chunks, through the multi-position
    decode, is not higher than that of llama.cpp's default setting on the same GGUF serves; the default passes by
    definition. Cached by the GGUF, snapshot, gate and engine version and machine."""
    import subprocess

    from .numeric_choice import GATE_VERSION, ratio_interval

    root = root or Path(__file__).resolve().parents[1]
    key = hashlib.sha256(json.dumps([file_sha256(gguf), str(Path(snapshot).resolve()), GATE_VERSION, ENGINE_VERSION,
                                     machine_key()], sort_keys=True).encode()).hexdigest()[:24]
    cache = cache or Path(os.path.expanduser("~/.cache/ironmule/ggml_gate"))
    stored = cache / f"{key}.json"
    if stored.exists():
        try:
            return json.loads(stored.read_text())
        except (OSError, ValueError):
            pass
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")])),
           "HF_HUB_OFFLINE": "1"}
    started = time.perf_counter()

    def child(code: str, argument: str, payload: str | None = None) -> tuple[dict[str, Any] | None, str]:
        done = subprocess.run([sys.executable, "-c", code, argument], env=env, input=payload, capture_output=True,
                              text=True, timeout=7200, check=False)
        if done.returncode != 0:
            return None, done.stderr[-300:]
        return json.loads(done.stdout.strip().splitlines()[-1]), ""

    tune, error = child(TUNE_CHILD, gguf)
    if tune is None:
        return {"ran": False, "passes": False, "reason": f"the tuning child failed: {error}"}
    text, error = child(GATE_IDS, snapshot)
    if text is None:
        return {"ran": False, "passes": False, "reason": f"the gate's text could not be tokenized: {error}"}

    def nll(settings: dict[str, bool]) -> tuple[list[float] | None, str]:
        out, why = child(GATE_CANDIDATE, gguf, json.dumps({**text, "settings": settings}))
        return (None, why) if out is None else (out["nll"], "")

    reference, error = nll(DEFAULT_SETTINGS)
    if reference is None:
        return {"ran": False, "passes": False, "reason": f"the ggml child failed: {error}"}
    tried = []
    for settings in tune["order"]:  # the fastest setting that is not worse than llama.cpp's default serves
        if settings == DEFAULT_SETTINGS:
            tried.append({"settings": settings, "ratio": 1.0, "interval": [1.0, 1.0], "chunks": len(reference),
                          "passes": True})
            break
        rows, error = nll(settings)
        if rows is None:
            return {"ran": False, "passes": False, "reason": f"the ggml child failed: {error}", "tried": tried}
        interval = ratio_interval(reference, rows)
        tried.append({"settings": settings, **interval, "passes": interval["ratio"] <= 1.0})
        if tried[-1]["passes"]:
            break
    chosen = tried[-1]
    result = {"ran": True, **{k: chosen[k] for k in ("ratio", "interval", "chunks", "passes")},
              "settings": chosen["settings"], "tried": tried, "tuning": tune["measured"], "machine": machine_key(),
              "reference": "llama.cpp's default setting on the same GGUF",
              "seconds": round(time.perf_counter() - started, 1), "gate_version": GATE_VERSION, "key": key}
    try:
        cache.mkdir(parents=True, exist_ok=True)
        stored.write_text(json.dumps(result))
    except OSError:
        pass
    return result


__all__ = ["ENGINE_VERSION", "GgmlEngine", "Job", "autotune", "available", "bindings", "file_sha256", "gate",
           "machine_key"]
