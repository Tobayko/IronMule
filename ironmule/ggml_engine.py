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

SEQS, SEQ_CTX, MAX_ROWS = 8, 2048, 16


def _usable_cores() -> int:
    from .pilot import usable_cores

    return usable_cores()


# PI1 (2026-11-14): every core this process may use (affinity and cgroup quota), not a fixed 4; IRONMULE_GGML_THREADS
# sets a number. The Pilot then chooses how many of them a step uses.
THREADS = int(os.environ.get("IRONMULE_GGML_THREADS") or _usable_cores())
# ROW1 (2026-11-12): sequences and per-sequence context can be raised for a run, e.g. 32 requests in one batch
SEQS = int(os.environ.get("IRONMULE_GGML_SEQS", SEQS))
SEQ_CTX = int(os.environ.get("IRONMULE_GGML_SEQ_CTX", SEQ_CTX))
POLL = 50  # llama-server's default busy-wait level
ENGINE_VERSION = 4  # 2 (GG2): warm-up of 8 prompts, staggered prompts; 3 (GG3): llama.cpp's options within 1.005
OPTION_BOUND = 1.005  # GG3 (owner, 2026-11-11): a llama.cpp option serves within NUM1's bound of its default
# Prompt tokens a step may carry beside the running requests' rows (whole prompts, at least one). GG2 used 96, so early
# requests streamed while later prompts waited; ROW1 round 5 (2026-11-12, Qwen3-4B, 32 sequences) measured 7-9 % more
# tokens per second at 16-32 requests when every waiting prompt that fits one call goes in, as llama-server does.
# 0 (the default) means one call's size (the sequence length); a number sets the budget.
PROMPT_BUDGET = int(os.environ.get("IRONMULE_GGML_PROMPT_BUDGET", 0))
PASSED_OVER = 8
COLLECTIVE = os.environ.get("IRONMULE_GGML_COLLECTIVE", "1") != "0"  # SP4
PILOT = os.environ.get("IRONMULE_PILOT", "1") != "0"  # PI1: the decision network, when its weights are present
DRAFT_MAX_RUNNING = int(os.environ.get("IRONMULE_GGML_DRAFT_MAX_RUNNING", 2))  # DM1: it drafts up to this load
DRAFT_SLOTS = int(os.environ.get("IRONMULE_GGML_DRAFT_SLOTS", DRAFT_MAX_RUNNING))  # DM1: its sequences
LLAMA_MAX_SEQ = 64  # llama.cpp's limit on sequence ids in one context (4227c9b)
SHARE_WAIT = 64  # SP1: a prompt waits a step for a fresh one when that one would give it this many more shared tokens
DRAFT_BUCKET = 200  # DM1: drafts of the draft model (their lengths come from its own stopping rule)
MAX_DRAFT_SLACK = 8  # SP1: cells a running request may take beyond its answer for draft rows (speculation.MAX_DRAFT)  # a waiting request passed over this often by shorter prompts goes first
# ROW1 round 5 (2026-11-12): a finished request's KV stays in its sequence, and a new request takes the free sequence
# whose kept tokens share the longest prefix with its prompt and prefills only the rest, as llama-server's
# `cache_prompt` (its default). The reused positions were computed in another batch, so tokens can differ as on
# llama-server; "0" turns it off.
PREFIX_REUSE = os.environ.get("IRONMULE_GGML_PREFIX_REUSE", "1") != "0"
# MG2 (2026-11-13): a llama.cpp build with a GPU backend (CUDA, Metal, Vulkan, ...) gets every layer offloaded by default;
# IRONMULE_GGML_GPU_LAYERS sets a number (0 keeps the CPU). Several visible GPUs share one model by layers (llama.cpp's
# default); IRONMULE_GGML_SPLIT = layer | row | none chooses llama.cpp's split mode. Serving one model per GPU instead
# (data parallel) is `ironmule serve --gpus all`: a worker per GPU, each seeing only its own.
GPU_LAYERS = os.environ.get("IRONMULE_GGML_GPU_LAYERS")
SPLIT_MODES = {"none": 0, "layer": 1, "row": 2}
SPLIT = os.environ.get("IRONMULE_GGML_SPLIT", "layer")
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


def fa1_version() -> int:
    """FA1's version in the loaded ggml CPU library (native/ggml_patches/fa1; 3 = FA1u for several tokens), 0 without it.
    A shared KV pool (SP1) serves only with FA1u: llama.cpp's own CPU attention scans every cell of a unified pool for
    every query (KV1: 1.39 s a step instead of 0.43 s)."""
    import ctypes

    try:
        L = bindings()
        folder = Path(L.llama_cpp._lib._name).parent if hasattr(L, "llama_cpp") else Path(os.environ.get("LLAMA_CPP_LIB_PATH", ""))
        lib = ctypes.CDLL(str(next(folder.glob("libggml-cpu.so*"))))
        return int(lib.ironmule_fa1_version())
    except Exception:  # noqa: BLE001 - no library, no symbol: a stock build
        return 0


def gpu_offload(L: Any | None = None) -> bool:
    """Whether the loaded llama.cpp library has a GPU backend to offload layers to."""
    try:
        L = L or bindings()
        return bool(L.llama_supports_gpu_offload())
    except Exception:  # noqa: BLE001 - no library, no GPU
        return False


def gpu_layers(L: Any | None = None) -> int:
    """Layers to offload: IRONMULE_GGML_GPU_LAYERS, else all (-1 means every layer to llama.cpp) where a GPU backend is."""
    if GPU_LAYERS is not None:
        return int(GPU_LAYERS)
    return 999 if gpu_offload(L) else 0


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
    reused: int = 0  # prompt tokens the sequence's kept KV already held (prefix reuse)
    first_ns: int = 0
    finished_ns: int = 0
    passed_over: int = 0


class SharedPool:
    """SP1 (2026-11-13): one KV pool for every sequence (llama.cpp's unified cache) whose cells a new request shares with
    any running or finished sequence that holds the same leading tokens, as SGLang's RadixAttention does on a GPU.

    llama.cpp gives every sequence of a split cache its own `seq_ctx` cells, and `llama-server` reuses a slot's own
    prompt only. Here a request copies the longest matching prefix from whichever sequence holds it with
    `llama_memory_seq_cp`, which in a unified cache adds the request to those cells instead of copying them, and
    computes only the rest. Each sequence reserves only its prompt and its answer limit, not `seq_ctx`, so more requests
    fit the same memory. FA1u (native/ggml_patches/fa1) reads a shared block once per step for every sequence that sees
    it. The pool counts cells itself: every position of every sequence is a node with a reference count, shared nodes
    once; a request is admitted when the pool's used cells, every running request's remaining answer limit and its own
    need fit, after finished sequences (kept as a cache) have been evicted least recently used first."""

    def __init__(self, engine: "GgmlEngine", cells: int, cap: int, ids: int, slack: int):
        from collections import OrderedDict

        self.e, self.cells, self.cap, self.slack = engine, cells, cap, slack
        self.free_ids = list(range(ids))
        self.tokens: dict[int, list[int]] = {}  # sequence -> the tokens its cells hold, from position 0
        self.nodes: dict[int, list[int]] = {}  # sequence -> one node per held position
        self.ref: dict[int, int] = {}  # node -> sequences that hold it; a node is one physical cell
        self.next_node = 0
        self.active: set[int] = set()
        self.reserve: dict[int, int] = {}  # running sequence -> cells it may still take
        self.kept: OrderedDict[int, None] = OrderedDict()  # finished sequences kept as a cache, oldest first
        self.shared_tokens = 0

    @property
    def used(self) -> int:
        return len(self.ref)

    def slots(self) -> int:
        return max(0, self.cap - len(self.active))

    def reset(self) -> None:
        self.free_ids = sorted(set(self.free_ids) | set(self.tokens) | set(self.active))
        self.tokens.clear(); self.nodes.clear(); self.ref.clear(); self.active.clear(); self.reserve.clear()
        self.kept.clear()

    def _unref(self, nodes: list[int]) -> None:
        for node in nodes:
            count = self.ref[node] - 1
            if count:
                self.ref[node] = count
            else:
                del self.ref[node]

    def _evict(self, spare: int | None) -> bool:
        """Drop the least recently used finished sequence other than `spare`; False when there is none."""
        for seq in self.kept:
            if seq != spare:
                del self.kept[seq]
                self.e.L.llama_memory_seq_rm(self.e.memory, seq, -1, -1)
                self._unref(self.nodes.pop(seq, []))
                self.tokens.pop(seq, None)
                self.free_ids.append(seq)
                return True
        return False

    def place(self, ids: list[int], limit: int) -> tuple[int, int] | None:
        """A sequence for a new request and how many of its prompt tokens it already holds, or None while it does
        not fit (it waits)."""
        best, match = None, 0
        if self.e.prefix_reuse:
            for seq, held in self.tokens.items():
                n = GgmlEngine._common(held, ids)
                if n > match:
                    best, match = seq, n
        keep = min(match, len(ids) - 1)
        if keep <= 0:
            best, keep = None, 0
        need = len(ids) - keep + limit + self.slack
        while self.used + sum(self.reserve.values()) + need > self.cells:
            if not self._evict(best):
                return None
        if best is not None and best in self.kept:  # a finished sequence: continue in it, past the shared prefix
            del self.kept[best]
            seq = best
            if len(self.tokens[seq]) > keep:
                self.e.L.llama_memory_seq_rm(self.e.memory, seq, keep, -1)
                self._unref(self.nodes[seq][keep:])
                del self.nodes[seq][keep:]
                del self.tokens[seq][keep:]
        else:
            while not self.free_ids:
                if not self._evict(best):
                    return None
            seq = self.free_ids.pop(0)
            self.tokens[seq], self.nodes[seq] = [], []
            if keep:
                self.e.L.llama_memory_seq_cp(self.e.memory, best, seq, 0, keep)
                shared = self.nodes[best][:keep]
                for node in shared:
                    self.ref[node] += 1
                self.nodes[seq], self.tokens[seq] = list(shared), list(ids[:keep])
        self.shared_tokens += keep
        self.active.add(seq)
        self.reserve[seq] = len(ids) - keep + limit + self.slack
        return seq, keep

    def sync(self, seq: int, held: list[int], remaining: int) -> None:
        """After a step: `seq`'s cells hold `held`; it may still take `remaining` cells."""
        nodes, tokens = self.nodes[seq], self.tokens[seq]
        if len(nodes) > len(held):
            self._unref(nodes[len(held):])
            del nodes[len(held):]
            del tokens[len(held):]
        for token in held[len(nodes):]:
            self.ref[self.next_node] = 1
            nodes.append(self.next_node)
            tokens.append(token)
            self.next_node += 1
        self.reserve[seq] = max(0, remaining) + self.slack

    def release(self, seq: int, held: list[int] | None) -> None:
        """A request ends: keep its cells as a cache (`held`, the tokens they hold) or free them (None)."""
        self.active.discard(seq)
        self.reserve.pop(seq, None)
        if held is not None and held:
            self.sync(seq, held, 0)
            self.reserve.pop(seq, None)
            self.kept[seq] = None
        else:
            self.e.L.llama_memory_seq_rm(self.e.memory, seq, -1, -1)
            self._unref(self.nodes.pop(seq, []))
            self.tokens.pop(seq, None)
            self.free_ids.append(seq)

    def summary(self) -> dict[str, Any]:
        return {"cells": self.cells, "used": self.used, "running": len(self.active), "cap": self.cap,
                "kept": len(self.kept), "shared_tokens": self.shared_tokens}


class GgmlEngine:
    observer: Callable[[int, int, int, float], None] | None = None  # AG10: (rows, prompt, kv, seconds) of each step
    gpu_layers = 0
    pool: "SharedPool | None" = None
    pilot: Any = None
    pilot_refused: str | None = None
    collective: Any = None
    drafter: Any = None
    draft_refused: str | None = None

    def __init__(self, gguf: str, *, threads: int = THREADS, seqs: int = SEQS, seq_ctx: int = SEQ_CTX,
                 speculate: bool = True, max_rows: int = MAX_ROWS, flash_attn: bool = False,
                 kv_unified: bool = False, kv_f32: bool = False, kv_q8: bool = False, model: Any = None,
                 shared: bool = False, cap: int | None = None, draft_gguf: str | None = None):
        import numpy as np

        L = bindings()
        self.L, self.np = L, np
        L.llama_backend_init()
        self._owns_model = model is None  # SEQ1: a width measurement opens several contexts on one loaded model
        self.gpu_layers = gpu_layers(L)  # the same for a context opened on an already loaded model (SEQ1)
        if model is None:
            mparams = L.llama_model_default_params()
            mparams.n_gpu_layers = self.gpu_layers
            if self.gpu_layers:
                mparams.split_mode = SPLIT_MODES.get(SPLIT, 1)
        self.model = model or L.llama_model_load_from_file(str(gguf).encode(), mparams)
        if not self.model:
            raise ValueError(f"llama.cpp could not load {gguf}")
        self.threads, self.max_rows, self.speculate = threads, max_rows, speculate
        self.prefix_reuse = PREFIX_REUSE
        self.prompt_tokens = self.reused_tokens = 0
        self._lock = threading.Lock()
        self.shared = bool(shared)  # SP1: one pool, prefixes shared; `seqs * seq_ctx` cells, up to `cap` running
        self.cap = int(cap or min(LLAMA_MAX_SEQ, 2 * seqs)) if shared else seqs
        self._open(seqs, seq_ctx, flash_attn=flash_attn, kv_unified=kv_unified or self.shared, kv_f32=kv_f32,
                   kv_q8=kv_q8)
        self.steps = 0
        self._warm_up()
        self.drafter = None
        if draft_gguf and speculate:  # DM1: a small model of the same family drafts, this one verifies
            self.drafter = self._open_drafter(draft_gguf, threads)
        self.pilot = None
        if PILOT and model is None:  # PI1: the decision network, with this machine's measured step costs
            from .pilot import Pilot

            pilot = Pilot(self, self.measure_costs(), max_cap=self.cap if self.pool is not None else self.seqs,
                          gpu=bool(self.gpu_layers))
            self.pilot = pilot if pilot.active else None
            self.pilot_refused = None if pilot.active else pilot.reason

    def measure_costs(self) -> dict[str, float]:
        """PI1: what a step costs on this machine, for the Pilot's features (seconds): one row at a 64-token context,
        each further row, 1,000 prompt tokens, 1,000 attended tokens. About a second for a 4B model on a 4-core CPU."""
        bos = max(0, int(self.L.llama_vocab_bos(self.vocab)))
        rows8 = min(8, self.seqs if self.pool is None else LLAMA_MAX_SEQ)

        def median(fn, n=4):
            times = []
            for _ in range(n):
                started = time.perf_counter()
                fn()
                times.append(time.perf_counter() - started)
            return sorted(times)[len(times) // 2]

        self._clear()
        started = time.perf_counter()
        self._decode([(bos, j, 0, j == 255) for j in range(256)])
        prefill_1k = (time.perf_counter() - started) * 1000 / 256
        pos = [256]

        def one_long():
            self._decode([(bos, pos[0], 0, True)])
            pos[0] += 1
        t_long = median(one_long)
        self._clear()
        self._decode([(bos, j, s, j == 63) for s in range(rows8) for j in range(64)])
        pos = [64]

        def one():
            self._decode([(bos, pos[0], 0, True)])
            pos[0] += 1
        t_one = median(one)
        pos8 = [pos[0]]

        def eight():
            self._decode([(bos, pos8[0], s, True) for s in range(rows8)])
            pos8[0] += 1
        for s in range(1, rows8):  # the other sequences to the same position
            for p in range(64, pos[0]):
                self._decode([(bos, p, s, False)])
        t_eight = median(eight)
        self._clear()
        return {"step_one": t_one, "row": max(1e-5, (t_eight - t_one) / max(1, rows8 - 1)),
                "prefill_1k": prefill_1k, "kv_1k": max(1e-5, (t_long - t_one) / 0.192)}

    def _open_drafter(self, gguf: str, threads: int) -> Any:
        """The draft model with `DRAFT_SLOTS` sequences, or None (with the reason in `draft_refused`) where its
        tokenizer is not this model's."""
        from .draft_model import Drafter

        engine = GgmlEngine(gguf, threads=threads, seqs=DRAFT_SLOTS, seq_ctx=self.seq_ctx, speculate=False,
                            flash_attn=True)
        if engine.n_vocab != self.n_vocab or engine.tokenize(CHECK_TEXT) != self.tokenize(CHECK_TEXT):
            engine.close()
            self.draft_refused = "the draft GGUF's tokenizer is not the model's"
            return None
        return Drafter(engine, max_running=DRAFT_MAX_RUNNING)

    def _decode_conf(self, rows: list[tuple[int, int, int, bool]]) -> list[tuple[int, float]]:
        """As `_decode`, but per row that asks for logits its top token and that token's probability."""
        b, np = self.batch, self.np
        b.n_tokens = len(rows)
        for i, (token, pos, seq, logits) in enumerate(rows):
            b.token[i], b.pos[i], b.n_seq_id[i], b.logits[i] = token, pos, 1, int(logits)
            b.seq_id[i][0] = seq
        if self.L.llama_decode(self.ctx, b) != 0:
            raise RuntimeError("llama_decode failed")
        out = []
        for i, (_, _, _, logits) in enumerate(rows):
            if logits:
                row = np.ctypeslib.as_array(self.L.llama_get_logits_ith(self.ctx, i), shape=(self.n_vocab,))
                top = int(row.argmax())
                out.append((top, float(1.0 / np.exp(row - row[top]).sum())))
        return out

    def _open(self, seqs: int, seq_ctx: int, *, flash_attn: bool = False, kv_unified: bool = False,
              kv_f32: bool = False, kv_q8: bool = False) -> None:
        """A context of `seqs` sequences of `seq_ctx` tokens on the loaded model, with its batch and thread pool."""
        from .speculation import StepPolicy

        L, threads = self.L, self.threads
        params = L.llama_context_default_params()
        shared = getattr(self, "shared", False)
        ids = LLAMA_MAX_SEQ if shared else seqs  # SP1: running sequences and finished ones kept as a cache
        params.n_ctx, params.n_batch, params.n_ubatch = seqs * seq_ctx, seq_ctx, 512
        params.n_seq_max, params.n_threads, params.n_threads_batch = ids, threads, threads
        params.flash_attn, params.kv_unified = bool(flash_attn), bool(kv_unified)  # GG1: measured at the first start
        if kv_f32:  # GG1, second amendment: the KV cache in full precision (GGML_TYPE_F32) instead of f16
            params.type_k = params.type_v = 0
        if kv_q8:  # KV1 (2026-11-13): llama.cpp's 8-bit KV cache (GGML_TYPE_Q8_0, `-ctk q8_0 -ctv q8_0`), half the bytes
            if kv_f32 or not flash_attn:
                raise ValueError("an 8-bit KV cache needs flash attention and excludes kv_f32 (llama.cpp's rule)")
            params.type_k = params.type_v = 8
        if self.gpu_layers:
            params.offload_kqv = True  # the KV cache lives beside the offloaded layers
        self.settings = {"flash_attn": bool(flash_attn), "kv_unified": bool(kv_unified), "kv_f32": bool(kv_f32)}
        if kv_q8:
            self.settings["kv_q8"] = True
        self.ctx = L.llama_init_from_model(self.model, params)
        if not self.ctx:
            raise ValueError("llama.cpp could not create a context")
        self._threadpool = self._attach_threadpool(threads, POLL)
        self.vocab = L.llama_model_get_vocab(self.model)
        self.n_vocab = L.llama_vocab_n_tokens(self.vocab)
        self.memory = L.llama_get_memory(self.ctx)
        self.seqs, self.seq_ctx = seqs, seq_ctx
        self.batch = L.llama_batch_init(max(seq_ctx, self.max_rows), 0, seqs)
        self.policy = StepPolicy(self.max_rows) if self.speculate else None
        if self.speculate and COLLECTIVE and getattr(self, "collective", None) is None:
            from .speculation import CollectiveIndex

            self.collective = CollectiveIndex()  # SP4: drafts from other requests' answers
        self._free = list(range(seqs))
        self._kept: dict[int, list[int]] = {}  # free sequence -> the tokens its KV still holds (prefix reuse)
        self.pool = SharedPool(self, seqs * seq_ctx, self.cap, ids, MAX_DRAFT_SLACK) if shared else None
        if shared:
            self.settings["shared_prefix"] = True

    def reconfigure(self, seqs: int, settings: dict[str, bool]) -> None:
        """AG10: serve with another width or KV cache on the same loaded model; only between runs (the caller holds the
        run lock). The old context and its KV go; the new one is warmed up as at the start."""
        self.close_context()
        self._open(seqs, self.seq_ctx, **{k: bool(v) for k, v in settings.items() if k in SETTING_KEYS})
        self._warm_up()

    def _warm_up(self) -> None:
        """As `llama-server` does at its start: one call of a prompt's size and one of a step's, so a request never pays
        for llama.cpp's first graphs and weight repacking; the cache is cleared after."""
        bos = max(0, int(self.L.llama_vocab_bos(self.vocab)))
        self._decode([(bos, j, 0, j == 63) for j in range(64)])
        self._clear()
        # GG2: the first call that carries several sequences' prompts cost about 0.3 s once; pay it here
        rows = [(bos, j, seq, j == 63) for seq in range(self.seqs) for j in range(64)]
        for start in range(0, len(rows), self.seq_ctx):  # in pieces of the batch size when many sequences
            self._decode(rows[start:start + self.seq_ctx])
        self._clear()  # the warm-up's KV is not a prompt
        self._decode([(bos, 64, seq, True) for seq in range(self.seqs)])
        self._decode([(bos, 65, 0, True)])
        self.L.llama_memory_clear(self.memory, True)

    def set_threads(self, threads: int, poll: int | None = None) -> None:
        """PI1: change how many CPU threads the next steps use (and how long they busy-wait between a graph's
        operations), on the live context; between steps only. A few milliseconds: a new thread pool is attached."""
        threads = max(1, int(threads))
        poll = POLL if poll is None else int(poll)
        if threads == self.threads and poll == getattr(self, "poll", POLL):
            return
        if getattr(self, "_threadpool", None):
            try:
                self._free_pool()
            except Exception:  # noqa: BLE001 - the context's own pool serves meanwhile
                pass
            self._threadpool = None
        self.L.llama_set_n_threads(self.ctx, threads, threads)
        self.threads, self.poll = threads, poll
        self._threadpool = self._attach_threadpool(threads, poll)

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
        status = self.L.llama_decode(self.ctx, b)
        if status == 1 and getattr(self, "pool", None) is not None:  # SP1: no free cells; drop the cached sequences
            with self._lock:
                while self.pool._evict(None):
                    pass
            status = self.L.llama_decode(self.ctx, b)
        if status != 0:
            raise RuntimeError("llama_decode failed")
        out = []
        for i, (_, _, _, logits) in enumerate(rows):
            if logits:
                ptr = self.L.llama_get_logits_ith(self.ctx, i)
                out.append(int(self.np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).argmax()))
        return out

    def _clear(self) -> None:
        self.L.llama_memory_clear(self.memory, True)
        self._kept.clear()
        if getattr(self, "pool", None) is not None:
            self.pool.reset()

    @staticmethod
    def _common(a: Sequence[int], b: Sequence[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _take_sequence(self, ids: list[int]) -> tuple[int, int]:
        """A free sequence for a prompt and how many of its first tokens the sequence's KV already holds (at least one
        prompt token is always computed, for its logits). Call with the lock held."""
        best, keep = self._free[0], 0
        if self.prefix_reuse and self._kept:
            for seq in self._free:
                held = self._kept.get(seq)
                if held:
                    n = self._common(held, ids)
                    if n > keep:
                        best, keep = seq, n
        return best, min(keep, len(ids) - 1)

    def _claim(self, seq: int, keep: int) -> int:
        """Take `seq` off the free list with its first `keep` positions; returns the positions really kept."""
        self._free.remove(seq)
        held = self._kept.pop(seq, None)
        if held is not None and len(held) > keep:
            if keep == 0 or not self.L.llama_memory_seq_rm(self.memory, seq, keep, -1):
                self.L.llama_memory_seq_rm(self.memory, seq, -1, -1)
                keep = 0
        return keep

    def _prefill(self, ids: list[int], seq: int, start: int = 0) -> int:
        first = None
        for start in range(start, len(ids), self.seq_ctx):
            chunk = ids[start:start + self.seq_ctx]
            last = start + len(chunk) == len(ids)
            picks = self._decode([(t, start + j, seq, last and j == len(chunk) - 1) for j, t in enumerate(chunk)])
            first = picks[-1] if picks else first
        return first

    def free_slots(self) -> int:
        with self._lock:
            return self.pool.slots() if self.pool is not None else len(self._free)

    def _slots(self) -> int:
        return self.pool.slots() if self.pool is not None else len(self._free)

    # --- serving ------------------------------------------------------------------------------------------------------
    def run(self, jobs: list[Job], admit: Callable[[int], list[Job]] | None = None) -> list[Job]:
        """Serve `jobs` and every job `admit(free sequences)` hands over while they run; returns all, in order."""
        from .speculation import MAX_DRAFT, accept, draft

        served: list[Job] = list(jobs)
        pending = list(jobs)
        active: dict[int, Job] = {}

        def finish(seq: int, job: Job, reason: str, keep: bool = True) -> None:
            job.reason, job.finished_ns = reason, time.monotonic_ns()
            if self.collective is not None and reason in ("eos", "length") and job.tokens:
                self.collective.add(job.tokens)  # SP4: this answer becomes a draft source for later requests
            if self.drafter is not None:
                self.drafter.release(seq)
            held = len(job.ids) + len(job.tokens) - 1  # every token but the last went through the model
            if self.pool is not None:  # SP1: the pool keeps its cells as a cache, or frees them
                ok = (keep and self.prefix_reuse and held > 0
                      and self.L.llama_memory_seq_rm(self.memory, seq, held, -1))
                active.pop(seq, None)
                with self._lock:
                    self.pool.release(seq, (job.ids + job.tokens)[:held] if ok else None)
                return
            if (keep and self.prefix_reuse and held > 0
                    and self.L.llama_memory_seq_rm(self.memory, seq, held, -1)):
                kept = (job.ids + job.tokens)[:held]
            else:
                self.L.llama_memory_seq_rm(self.memory, seq, -1, -1)
                kept = None
            active.pop(seq, None)
            with self._lock:
                if kept is not None:
                    self._kept[seq] = kept
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

        decision = {"cap": 1 << 30, "budget": 1 << 30, "draft": True}
        arrived = 0
        try:
            while True:
                if self.pilot is not None:  # PI1: threads, running cap, prompt budget and drafting for this step
                    ctx = (sum(len(j.ids) + len(j.tokens) for j in active.values()) / len(active)) if active else 0.0
                    decision = self.pilot.decide(len(active), len(pending), sum(len(j.ids) for j in pending), ctx)
                    if decision["threads"] != self.threads:
                        self.set_threads(decision["threads"])

                def room() -> int:
                    return max(0, min(self._slots(), decision["cap"] - len(active)))

                if admit is not None and room():
                    more = admit(room())
                    served.extend(more)
                    pending.extend(more)
                    arrived += len(more)
                # GG2: the requests that start in this step: the shortest prompts first, within the step's prompt
                # budget (whole prompts, at least one), and one passed over too often first
                fresh: list[tuple[int, Job, int]] = []
                budget = min(PROMPT_BUDGET or self.seq_ctx, decision["budget"])
                for job in sorted(pending, key=lambda j: (j.passed_over < PASSED_OVER, len(j.ids))):
                    if job.cancelled is not None and job.cancelled():
                        job.reason, job.finished_ns = "cancelled", time.monotonic_ns()
                        pending.remove(job)
                        continue
                    if not self._slots() or len(active) + len(fresh) >= decision["cap"]:
                        job.passed_over += 1
                        continue
                    if len(job.ids) + job.limit > self.seq_ctx:
                        raise ValueError(f"{len(job.ids) + job.limit} tokens do not fit a {self.seq_ctx}-token sequence")
                    if self.pool is not None:  # SP1: a sequence that shares the longest prefix any sequence holds
                        # a prompt that shares a long prefix with one starting in this step waits one step and then
                        # shares its cells, instead of computing the same prefix again (a burst with one system prompt)
                        with self._lock:
                            pooled = max((self._common(t, job.ids) for t in self.pool.tokens.values()), default=0)
                        if any(self._common(other.ids, job.ids) >= pooled + SHARE_WAIT for _, other, _ in fresh):
                            continue
                        with self._lock:
                            placed = self.pool.place(job.ids, job.limit)
                        if placed is None:  # the pool is full until a running request ends
                            job.passed_over += 1
                            continue
                        seq, keep = placed
                    else:
                        with self._lock:
                            seq, keep = self._take_sequence(job.ids)
                        if fresh and len(job.ids) - keep > budget:
                            job.passed_over += 1
                            continue
                        with self._lock:
                            keep = self._claim(seq, keep)
                    pending.remove(job)
                    job.reused = keep
                    budget -= len(job.ids) - keep
                    self.prompt_tokens += len(job.ids)
                    self.reused_tokens += keep
                    fresh.append((seq, job, keep))
                if not active and not fresh:
                    if pending:
                        continue
                    if admit is not None:
                        more = admit(room())
                        if more:
                            served.extend(more)
                            pending.extend(more)
                            continue
                    break
                # One call per step, as llama-server batches: every active request's last token and its drafts, and
                # the prompts of the requests that start now (beyond one call's tokens, the rest go first on their own).
                order = sorted(active)
                offers = []
                modelled: dict[int, list[int]] = {}
                if (self.drafter is not None and order and len(order) < self.max_rows
                        and decision["draft"]):  # DM1: the draft model's drafts, where the Pilot allows them
                    wants = {}
                    # the step's free rows, shared by the requests the draft model has slots for: a draft beyond a
                    # request's share would be computed and then not verified
                    share = (self.max_rows - len(order)) // max(1, min(len(order), len(self.drafter.held)))
                    for seq in order:
                        job = active[seq]
                        pos = len(job.ids) + len(job.tokens) - 1
                        room = min(MAX_DRAFT, job.limit - len(job.tokens) - 1, self.seq_ctx - 1 - pos, share)
                        if room > 0:
                            wants[seq] = (job.ids + job.tokens, room)
                    modelled = self.drafter.draft(wants, len(order)) if wants else {}
                for seq in order:
                    job = active[seq]
                    pos = len(job.ids) + len(job.tokens) - 1
                    room = min(MAX_DRAFT, job.limit - len(job.tokens) - 1, self.seq_ctx - 1 - pos)
                    if seq in modelled:
                        offers.append((modelled[seq], DRAFT_BUCKET))
                        continue
                    offer = draft(job.ids + job.tokens, room) if self.policy is not None and room > 0 else ([], 0)
                    if self.collective is not None and room > 0 and len(offer[0]) < 2:  # SP4: others' answers
                        other = self.collective.draft(job.ids + job.tokens, room)
                        offer = other if len(other[0]) > len(offer[0]) else offer
                    offers.append(offer)
                if modelled:  # the stopping rule chose these lengths; the others share what the step's rows leave
                    rest = [([], 0) if seq in modelled else offer for seq, offer in zip(order, offers)]
                    ks = self.policy.choose(rest) if self.policy is not None else [0] * len(order)
                    budget = self.max_rows - len(order) - sum(ks)
                    for i, seq in enumerate(order):
                        if seq in modelled:
                            ks[i] = max(0, min(len(modelled[seq]), budget))
                            budget -= ks[i]
                else:
                    ks = self.policy.choose(offers) if self.policy is not None and order else [0] * len(order)
                rows: list[tuple[int, int, int, bool]] = []
                for seq, (drafts, _), k in zip(order, offers, ks):
                    job = active[seq]
                    pos = len(job.ids) + len(job.tokens) - 1
                    rows += [(t, pos + j, seq, True) for j, t in enumerate([job.tokens[-1]] + list(drafts[:k]))]
                decode_rows = len(rows)
                room = self.seq_ctx - decode_rows  # one call's tokens
                batched: list[tuple[int, Job]] = []
                for seq, job, keep in fresh:
                    if len(job.ids) - keep <= room:
                        rows += [(t, j, seq, j == len(job.ids) - 1) for j, t in enumerate(job.ids) if j >= keep]
                        room -= len(job.ids) - keep
                        batched.append((seq, job))
                    else:  # too long to share this call: its own prefill first
                        active[seq] = job
                        take(seq, job, self._prefill(job.ids, seq, keep))
                started = time.perf_counter()
                picks = self._decode(rows) if rows else []
                seconds = time.perf_counter() - started
                if self.policy is not None and decode_rows and not batched:  # a prompt in the call is not a step's price
                    self.policy.observe_step(decode_rows, seconds, any(ks))
                if self.observer is not None and rows:  # AG10: the agent's world model learns every step's price
                    self.observer(decode_rows, len(rows) - decode_rows,
                                  sum(len(j.ids) + len(j.tokens) for j in active.values()), seconds)
                self.steps += 1
                at = produced = 0
                for seq, (drafts, bucket), k in zip(order, offers, ks):
                    job = active[seq]
                    pos = len(job.ids) + len(job.tokens) - 1
                    new = accept(list(drafts[:k]), picks[at:at + k + 1])
                    at += k + 1
                    produced += len(new)
                    if seq in modelled:
                        self.drafter.settle(seq, k, len(new) - 1)
                    elif k and self.policy is not None:
                        self.policy.observe_drafts(bucket, k, len(new) - 1)
                    if k:
                        self.L.llama_memory_seq_rm(self.memory, seq, pos + len(new), -1)  # rejected positions
                    for token in new:
                        if take(seq, job, token):
                            break
                if self.drafter is not None and order and not batched:  # DM1: what this step cost and made
                    self.drafter.end_step(seconds, len(order), decode_rows - len(order), produced)
                if self.pilot is not None and rows:
                    acc = (self.drafter.stop.accepted / self.drafter.stop.drafted
                           if self.drafter is not None and self.drafter.stop.drafted else None)
                    self.pilot.observe(seconds, produced + len(batched), arrived, acc)
                    arrived = 0
                for seq, job in batched:  # each prompt asked for logits at its last position only
                    active[seq] = job
                    take(seq, job, picks[at])
                    at += 1
                if self.pool is not None:  # SP1: what every running sequence's cells hold now
                    with self._lock:
                        for seq, job in active.items():
                            self.pool.sync(seq, (job.ids + job.tokens)[:len(job.ids) + len(job.tokens) - 1],
                                           job.limit - len(job.tokens))
        finally:
            for seq in list(active):  # a run that ends here leaves no KV it cannot vouch for
                finish(seq, active[seq], active[seq].reason or "cancelled", keep=False)
        return served

    # --- the quality gate ---------------------------------------------------------------------------------------------
    def chunk_nll(self, ids: list[int], lead: list[int]) -> float:
        """Mean next-token negative log-likelihood of `ids[1:]` after `lead + ids[:-1]`, in one multi-position decode
        (the verify path), as `numeric_choice.nll_per_chunk` scores a chunk."""
        np = self.np
        self._clear()
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
        self._clear()
        return total / (len(ids) - 1)

    def summary(self) -> dict[str, Any]:
        return {"engine": "ggml", "engine_version": ENGINE_VERSION, "settings": dict(self.settings),
                "gpu": {"layers": self.gpu_layers, "split": SPLIT} if self.gpu_layers else None,
                "threadpool_poll": POLL if self._threadpool else None,
                "threads": self.threads, "sequences": self.seqs,
                "sequence_tokens": self.seq_ctx, "steps": self.steps,
                "prefix_reuse": {"on": self.prefix_reuse, "prompt_tokens": self.prompt_tokens,
                                 "reused_tokens": self.reused_tokens},
                **({"shared_pool": self.pool.summary()} if self.pool is not None else {}),
                **({"pilot": self.pilot.status()} if self.pilot is not None else
                   {"pilot": {"active": False, "reason": self.pilot_refused}} if self.pilot_refused else {}),
                **({"draft_model": self.drafter.summary()} if self.drafter is not None else
                   {"draft_model": {"refused": self.draft_refused}} if self.draft_refused else {}),
                **({"speculation": self.policy.summary()} if self.policy is not None else {"speculation": "off"})}

    def close(self) -> None:
        if getattr(self, "drafter", None) is not None:
            self.drafter.close()
            self.drafter = None
        self.close_context()
        if getattr(self, "model", None):
            if getattr(self, "_owns_model", True):
                self.L.llama_model_free(self.model)
            self.model = None

    def close_context(self) -> None:
        """Free the context, its KV cache, batch and thread pool; the loaded model stays."""
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

    def kv_cell_bytes(self) -> int:
        """Bytes one KV cell (one token of one sequence) takes in every layer: K and V of every KV head, f16 or f32."""
        def meta(key: str) -> str | None:
            buf = __import__("ctypes").create_string_buffer(256)
            n = self.L.llama_model_meta_val_str(self.model, key.encode(), buf, len(buf))
            return buf.value.decode() if n > 0 else None

        layers, kv_heads = self.L.llama_model_n_layer(self.model), self.L.llama_model_n_head_kv(self.model)
        arch = meta("general.architecture") or ""
        fallback = self.L.llama_model_n_embd(self.model) // max(1, self.L.llama_model_n_head(self.model))
        k_len = int(meta(f"{arch}.attention.key_length") or fallback)
        v_len = int(meta(f"{arch}.attention.value_length") or k_len)
        if self.settings.get("kv_q8"):  # Q8_0: 34 bytes per 32 values
            return layers * kv_heads * ((k_len + 31) // 32 + (v_len + 31) // 32) * 34
        return layers * kv_heads * (k_len + v_len) * (4 if self.settings.get("kv_f32") else 2)


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


# --- SEQ1: the engine chooses how many sequences it serves at once (owner, 2026-11-13: "like an agent") ----------------
WIDTHS = (8, 16, 32, 64)  # the sequence counts it may choose; fewer only where 8 does not fit the memory
WIDTH_GAIN = 1.10  # a doubling must buy at least 10 % more tokens per second, else the memory is not spent
WIDTH_VERSION = 3  # 2 (AG10): also the default setting and a longer context; 3: and the prefill's price
AGENT_LONG_CTX = 512


def _cgroup_memory() -> tuple[int, int] | None:
    """(limit, usage) of this process's memory cgroup (v2 `memory.max`, v1 `memory.limit_in_bytes`), or None."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":  # v2
            folder, names = Path("/sys/fs/cgroup") / path.lstrip("/"), ("memory.max", "memory.current")
        elif "memory" in controllers.split(","):  # v1
            folder, names = Path("/sys/fs/cgroup/memory") / path.lstrip("/"), ("memory.limit_in_bytes",
                                                                              "memory.usage_in_bytes")
        else:
            continue
        try:
            limit, usage = ((folder / name).read_text().strip() for name in names)
            if not (limit.isdigit() and int(limit) < 2**60):
                continue
            # page cache of files on disk can be reclaimed, and the cgroup counts it as used: a GGUF just read stays
            # cached (and is counted once more as the model's own bytes); shared memory (tmpfs) cannot be reclaimed
            stat = dict(line.split() for line in (folder / "memory.stat").read_text().splitlines() if " " in line)
            cache = int(stat.get("total_cache", stat.get("file", 0)))
            shmem = int(stat.get("total_shmem", stat.get("shmem", 0)))
            return int(limit), max(0, int(usage) - max(0, cache - shmem))
        except (OSError, ValueError):
            continue
    return None


def memory_bytes() -> tuple[int, int]:
    """(available, total) bytes this process may use: /proc/meminfo, narrowed by the memory cgroup's limit where one is
    set (a container's limit can be below the host's memory), else half of physical memory."""
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, _, rest = line.partition(":")
            values[name] = int(rest.split()[0]) * 1024
        available, total = values["MemAvailable"], values["MemTotal"]
    except (OSError, KeyError, ValueError, IndexError):
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        available = total // 2
    group = _cgroup_memory()
    if group is not None:
        limit, usage = group
        available, total = min(available, max(0, limit - usage)), min(total, limit)
    return available, total


def gpu_memory_bytes(devices: str | None = None) -> tuple[int, int] | None:
    """(free, total) bytes of the GPUs a context will use, from `nvidia-smi` (MG2): the visible devices' sum when a model
    is split over them, or the one device named in `devices`. None where no NVIDIA GPU answers."""
    import subprocess

    command = ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    rows = {}
    for line in done.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            rows[parts[0]] = (int(parts[1]) * 2**20, int(parts[2]) * 2**20)
    visible = devices or os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        rows = {k: v for k, v in rows.items() if k in [d.strip() for d in visible.split(",")]}
    if not rows:
        return None
    return sum(v[0] for v in rows.values()), sum(v[1] for v in rows.values())


def fitting_widths(cell_bytes: int, seq_ctx: int, model_bytes: int, available: int, total: int) -> list[int]:
    """The widths whose KV cache (every sequence `seq_ctx` tokens long) fits beside the model and a reserve of a tenth of
    the machine's memory (at least 1.5 GiB); 8 or the largest of 4, 2, 1 that fits where none of WIDTHS does."""
    budget = available - model_bytes - max(int(1.5 * 2**30), total // 10)
    fits = [w for w in WIDTHS if w * seq_ctx * cell_bytes <= budget]
    if fits:
        return fits
    small = [w for w in (4, 2, 1) if w * seq_ctx * cell_bytes <= budget]
    return [small[0] if small else 1]


def decide_width(rates: dict[int, float]) -> int:
    """From the measured tokens per second per width: start at the narrowest, double while that buys WIDTH_GAIN."""
    widths = sorted(rates)
    chosen = widths[0]
    for wider in widths[1:]:
        if rates[wider] >= WIDTH_GAIN * rates[chosen]:
            chosen = wider
        else:
            break
    return chosen


def measure_widths(gguf: str, settings: dict[str, bool], seq_ctx: int, widths: Sequence[int], steps: int = 16,
                   repeats: int = 3, long_ctx: int = 0) -> dict[str, Any]:
    """Tokens per second of every width at full load on this machine: per width one context on the same loaded model,
    each sequence a 64-token prompt, then `steps` decode steps of every sequence, median of `repeats`."""
    first = GgmlEngine(gguf, seqs=min(widths), seq_ctx=seq_ctx, speculate=False, **settings)
    out: dict[str, Any] = {"cell_bytes": first.kv_cell_bytes(), "rates": {}, "step_s": {}, "step_one_s": {},
                           "settings": dict(settings)}
    model = first.model
    try:
        prompt = (first.tokenize(CHECK_TEXT) * 8)[:64]
        for width in sorted(widths):
            # one context at a time: the previous one's KV cache is freed before the next is allocated
            engine = first if width == first.seqs else GgmlEngine(gguf, seqs=width, seq_ctx=seq_ctx, speculate=False,
                                                                   model=model, **settings)
            try:
                times = []
                for _ in range(repeats):
                    engine._clear()
                    rows = [(t, j, seq, j == len(prompt) - 1) for seq in range(width) for j, t in enumerate(prompt)]
                    for start in range(0, len(rows), seq_ctx):
                        engine._decode(rows[start:start + seq_ctx])
                    started = time.perf_counter()
                    for k in range(steps):
                        engine._decode([(prompt[k % len(prompt)], len(prompt) + k, seq, True) for seq in range(width)])
                    times.append(time.perf_counter() - started)
                out["rates"][str(width)] = width * steps / statistics.median(times)
                out["step_s"][str(width)] = statistics.median(times) / steps
                ones = []  # AG10: one row on the same context, the agent's world model's other end
                for _ in range(repeats):
                    engine.L.llama_memory_seq_rm(engine.memory, 0, len(prompt) + steps, -1)
                    started = time.perf_counter()
                    for k in range(steps):
                        engine._decode([(prompt[k % len(prompt)], len(prompt) + steps + k, 0, True)])
                    ones.append(time.perf_counter() - started)
                out["step_one_s"][str(width)] = statistics.median(ones) / steps
                # AG10: the same full-load step at a longer context, so the agent's world model knows what a step
                # pays per attended KV token under these settings (flash attention or not differ most here)
                if long_ctx and long_ctx + steps <= seq_ctx:
                    engine._clear()
                    longer = (prompt * (long_ctx // len(prompt) + 1))[:long_ctx]
                    rows = [(t, j, seq, j == long_ctx - 1) for seq in range(width) for j, t in enumerate(longer)]
                    started = time.perf_counter()
                    for start in range(0, len(rows), seq_ctx):
                        engine._decode(rows[start:start + seq_ctx])
                    out.setdefault("prefill_per_1k", {})[str(width)] = (time.perf_counter() - started) / (len(rows) / 1000)
                    started = time.perf_counter()
                    for k in range(steps):
                        engine._decode([(longer[k % long_ctx], long_ctx + k, seq, True) for seq in range(width)])
                    out.setdefault("step_long_s", {})[str(width)] = (time.perf_counter() - started) / steps
                    out["long_ctx"], out["short_ctx"] = long_ctx, len(prompt)
            finally:
                engine.close_context()
    finally:
        first.close_context()
        first.L.llama_model_free(model)
    return out


WIDTH_CHILD = r"""
import json, sys
from ironmule.ggml_engine import measure_widths
data = json.loads(sys.stdin.read())
print(json.dumps(measure_widths(sys.argv[1], data["settings"], data["seq_ctx"], data["widths"],
                                long_ctx=data.get("long_ctx", 0))))
"""

CELL_CHILD = r"""
import json, sys
from ironmule.ggml_engine import GgmlEngine
data = json.loads(sys.stdin.read())
engine = GgmlEngine(sys.argv[1], seqs=1, seq_ctx=256, speculate=False, **data["settings"])
print(json.dumps({"cell_bytes": engine.kv_cell_bytes()}))
engine.close()
"""


def choose_sequences(gguf: str, settings: dict[str, bool] | None = None, *, seq_ctx: int | None = None,
                     cache: Path | None = None, root: Path | None = None, device: str | None = None,
                     extra_bytes: int = 0) -> dict[str, Any]:
    """SEQ1: how many sequences the engine serves, chosen on this machine. It sees the memory the KV cache may take at
    the request length it promises (`seq_ctx`), measures the widths that fit (`measure_widths`), and keeps doubling
    while a doubling buys `WIDTH_GAIN` (`decide_width`). Remembered per GGUF, settings, request length, engine and
    machine, so a changed machine or model is measured again. Never raises: a failed measurement keeps 8."""
    import subprocess

    settings = {k: bool(v) for k, v in (settings or {}).items() if k in SETTING_KEYS}
    seq_ctx = seq_ctx or SEQ_CTX
    available, total = memory_bytes()
    on_gpu = gpu_layers() > 0
    if on_gpu:  # MG2: the KV cache and the weights live in the GPU's memory
        found = gpu_memory_bytes(device)
        if found is not None:
            available, total = found
    key = hashlib.sha256(json.dumps([file_sha256(gguf), settings, seq_ctx, ENGINE_VERSION, WIDTH_VERSION, WIDTHS,
                                     int(extra_bytes) >> 28,
                                     WIDTH_GAIN, machine_key(), round(total / 2**30)], sort_keys=True).encode()
                         ).hexdigest()[:24]
    cache = cache or Path(os.path.expanduser("~/.cache/ironmule/ggml_gate"))
    stored = cache / f"width-{key}.json"
    if stored.exists():
        try:
            return json.loads(stored.read_text())
        except (OSError, ValueError):
            pass
    root = root or Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")])),
           "HF_HUB_OFFLINE": "1"}
    env.pop("IRONMULE_GGML_SEQS", None)
    if device is not None:  # MG2: a worker per GPU; measure on one of them
        env["CUDA_VISIBLE_DEVICES"] = device
    started = time.perf_counter()

    def child(code: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        try:
            done = subprocess.run([sys.executable, "-c", code, gguf], env=env, input=json.dumps(payload),
                                  capture_output=True, text=True, timeout=3600, check=False)
            return json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else None
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    model_bytes = os.path.getsize(gguf) + int(extra_bytes)  # DM1: a draft model's weights and KV take memory too
    cell = child(CELL_CHILD, {"settings": settings})
    record: dict[str, Any] = {"seq_ctx": seq_ctx, "memory_available": available, "memory_total": total,
                              "memory": "gpu" if on_gpu else "host",
                              "model_bytes": model_bytes, "gain_rule": WIDTH_GAIN, "machine": machine_key(),
                              "version": WIDTH_VERSION}
    if cell is None:
        return {**record, "sequences": SEQS, "reason": "the KV size could not be read; 8 sequences as before"}
    widths = fitting_widths(cell["cell_bytes"], seq_ctx, model_bytes, available, total)
    record.update(cell_bytes=cell["cell_bytes"], fitting=widths)
    if len(widths) == 1:
        record.update(sequences=widths[0], reason="only this width fits the memory")
    else:
        measured = child(WIDTH_CHILD, {"settings": settings, "seq_ctx": seq_ctx, "widths": widths,
                                       "long_ctx": AGENT_LONG_CTX})
        if measured is None:
            return {**record, "sequences": SEQS, "reason": "the width measurement failed; 8 sequences as before"}
        rates = {int(w): r for w, r in measured["rates"].items()}
        record.update(sequences=decide_width(rates), rates=measured["rates"], step_s=measured["step_s"],
                      step_one_s=measured.get("step_one_s", {}), settings=settings,
                      reason="the widest width whose doubling still bought the gain rule")
        # AG10: the agent's alternatives -- the same widths under llama.cpp's default setting, which passes the gate
        # by definition and differs from flash attention in what a step pays per attended token
        alternatives = [measured]
        default = {k: False for k in DEFAULT_SETTINGS}
        if {k: v for k, v in settings.items() if v} and os.environ.get("IRONMULE_GGML_AGENT", "1") != "0":
            other = child(WIDTH_CHILD, {"settings": default, "seq_ctx": seq_ctx, "widths": widths,
                                        "long_ctx": AGENT_LONG_CTX})
            if other is not None:
                alternatives.append(other)
        record["measured"] = alternatives
    record["seconds"] = round(time.perf_counter() - started, 1)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        stored.write_text(json.dumps(record))
    except OSError:
        pass
    return record


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
SETTING_KEYS = ("flash_attn", "kv_unified", "kv_f32", "kv_q8")


def gate(snapshot: str, gguf: str, *, cache: Path | None = None, root: Path | None = None) -> dict[str, Any]:
    """The owner's bar (2026-11-10, 2026-11-11): answer quality at least llama.cpp's, its own options included. The
    engine's settings (all llama.cpp options) are timed (`autotune`), and the fastest one whose perplexity on the
    numeric gate's chunks, through the multi-position decode, is within `OPTION_BOUND` of llama.cpp's default setting
    on the same GGUF (the ratio's 95 % upper end) serves; the default passes by definition. Cached by the GGUF,
    snapshot, gate rule, engine version and machine."""
    import subprocess

    from .numeric_choice import GATE_VERSION, ratio_interval

    root = root or Path(__file__).resolve().parents[1]
    key = hashlib.sha256(json.dumps([file_sha256(gguf), str(Path(snapshot).resolve()), GATE_VERSION, ENGINE_VERSION,
                                     OPTION_BOUND, machine_key()], sort_keys=True).encode()).hexdigest()[:24]
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
        tried.append({"settings": settings, **interval, "passes": interval["interval"][1] <= OPTION_BOUND})
        if tried[-1]["passes"]:
            break
    chosen = tried[-1]
    shared = None
    if fa1_version() >= 3:  # SP1: the shared pool (unified KV, flash attention through FA1u) is gated on its own
        settings = {**chosen["settings"], "flash_attn": True, "kv_unified": True}
        rows, error = nll(settings)
        shared = ({"settings": settings, **ratio_interval(reference, rows)} if rows is not None
                  else {"settings": settings, "error": error})
        shared["passes"] = rows is not None and shared["interval"][1] <= OPTION_BOUND
    result = {"ran": True, "shared": shared, **{k: chosen[k] for k in ("ratio", "interval", "chunks", "passes")},
              "settings": chosen["settings"], "tried": tried, "tuning": tune["measured"], "machine": machine_key(),
              "reference": "llama.cpp's default setting on the same GGUF", "bound": OPTION_BOUND,
              "seconds": round(time.perf_counter() - started, 1), "gate_version": GATE_VERSION, "key": key}
    try:
        cache.mkdir(parents=True, exist_ok=True)
        stored.write_text(json.dumps(result))
    except OSError:
        pass
    return result


__all__ = ["ENGINE_VERSION", "OPTION_BOUND", "GgmlEngine", "Job", "autotune", "available", "bindings", "choose_sequences",
           "decide_width", "file_sha256", "fitting_widths", "gate", "machine_key", "measure_widths", "memory_bytes"]
