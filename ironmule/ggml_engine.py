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
import stat
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

from ironmule_devices import bindings, ggml_devices

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
# llama-server's default busy-wait level; IRONMULE_GGML_POLL sets another (HC1, 2026-11-23: with every layer on the GPU the
# CPU threads only run the small input split of each step and then spin, competing with the server's Python threads)
POLL = int(os.environ.get("IRONMULE_GGML_POLL", 50))
ENGINE_VERSION = 5  # 2 (GG2): warm-up of 8 prompts, staggered prompts; 3 (GG3): llama.cpp's options within 1.005;
# 5 (ST1): a float32 KV cache ranks first in the autotune only 8 % ahead (rank_settings)
OPTION_BOUND = 1.005  # GG3 (owner, 2026-11-11): a llama.cpp option serves within NUM1's bound of its default
# Prompt tokens a step may carry beside the running requests' rows (whole prompts, at least one). GG2 used 96, so early
# requests streamed while later prompts waited; ROW1 round 5 (2026-11-12, Qwen3-4B, 32 sequences) measured 7-9 % more
# tokens per second at 16-32 requests when every waiting prompt that fits one call goes in, as llama-server does.
# 0 (the default) means one call's size (the sequence length); a number sets the budget.
PROMPT_BUDGET = int(os.environ.get("IRONMULE_GGML_PROMPT_BUDGET", 0))
PASSED_OVER = 8
COLLECTIVE = os.environ.get("IRONMULE_GGML_COLLECTIVE", "1") != "0"  # SP4
# PAD1: measure a step's price by its token count at the start (on GPUs; "1" also on CPUs, "0" never) and fill a step
# with guessed tokens up to a size measured cheaper: llama.cpp's CUDA matmuls switch kernels above 8 tokens, and on a
# T4 a step of 8 rows cost 52 ms against 41 ms for 32 (MG2 v4). Guesses are verified like drafts: answers do not change.
# Only with one unified KV cache: per-sequence caches split a step whose sequences carry different token counts into
# several passes (MG2 v5, 8 requests on a T4: 271 tok/s with the shared pool, 95 per-sequence, 150 per-sequence unpadded).
PAD = os.environ.get("IRONMULE_GGML_PAD", "")
# KVD1 (experiment): llama.cpp's KV defragmentation threshold (0 = off, its default). Every step attends over the cells
# up to the highest one in use, so a pool whose kept and running sequences drift up the cache pays for its holes.
DEFRAG = float(os.environ.get("IRONMULE_GGML_DEFRAG", 0) or 0)
# KVD1 (experiment): the share of the pool's cells that finished sequences may keep as a prefix cache (1 = all, as
# before); beyond it the least recently used go when a request ends, so fewer cells sit in every step's attended range.
KEEP = float(os.environ.get("IRONMULE_GGML_KEEP", 1) or 1)
# KG1 (2026-11-15): a kernel library generated and tuned for this GPU (ironmule/kernel_gen.py), loaded by llama.cpp's
# kg1 hook from IRONMULE_KG_LIB. The engine measures its own step for 1-8 tokens (KG4: and 16 / 24 / 32 for 9-32) with
# the library on and off at startup and keeps it only for the token counts where the step is faster.
KG_LIB = os.environ.get("IRONMULE_KG_LIB", "")
PILOT = os.environ.get("IRONMULE_PILOT", "1") != "0"  # PI1: the decision network, when its weights are present
DRAFT_MAX_RUNNING = int(os.environ.get("IRONMULE_GGML_DRAFT_MAX_RUNNING", 2))  # DM1: it drafts up to this load
DRAFT_SLOTS = int(os.environ.get("IRONMULE_GGML_DRAFT_SLOTS", DRAFT_MAX_RUNNING))  # DM1: its sequences
LLAMA_MAX_SEQ = 64  # llama.cpp's limit on sequence ids in one context (4227c9b)
SHARE_WAIT = 64  # SP1: a prompt waits a step for a fresh one when that one would give it this many more shared tokens
# SD1 (2026-11-23, experiment): on a GPU, IRONMULE_GGML_SD=k gives every running request k drafts of the draft model
# (BatchDrafter) while at most IRONMULE_GGML_SD_MAX_RUNNING requests run; 0 (default) keeps DM1's drafter
SD_K = int(os.environ.get("IRONMULE_GGML_SD", 0) or 0)
SD_MAX_RUNNING = int(os.environ.get("IRONMULE_GGML_SD_MAX_RUNNING", 8))
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
# PK1 (2026-11-19, experiment): packed prefill on GPUs with one KV stream per sequence. Every prompt that starts in a step
# goes into one native call, in ascending sequence order, so llama.cpp splits it into few rectangular micro-batches
# (all sequences at the shortest remaining length, then the rest) instead of one call per prompt; the step's prompt
# budget becomes PACK_TOKENS. IRONMULE_GGML_UBATCH sets llama.cpp's physical micro-batch (512 by default): larger ones
# give the prompt matmuls more rows per weight read. "1" turns packing on; the agent may choose it per machine.
PACK = os.environ.get("IRONMULE_GGML_PACK", "") == "1"
PACK_TOKENS = int(os.environ.get("IRONMULE_GGML_PACK_TOKENS", 8192))
# PK1: from idle, a packed engine gathers arrivals for up to GATHER_MS (until GATHER_QUIET_MS pass without one), so a
# burst that reaches the server over a few milliseconds is prefilled in one call instead of one prompt first
GATHER_S = float(os.environ.get("IRONMULE_GGML_GATHER_MS", 20)) / 1000
GATHER_QUIET_S = float(os.environ.get("IRONMULE_GGML_GATHER_QUIET_MS", 4)) / 1000
# PK1-B (2026-11-20): once a second request arrives in that window, the arrivals are a burst; the window then waits for
# a longer quiet gap, up to a longer limit. A served A100 burst of 32 HTTP requests reached the engine over about 50 ms:
# with the 4 ms gap two prompts were prefilled first (0.36 s) and the other 30 waited behind them (about 12 % of the
# burst's wall time). A single request still starts after the short gap.
GATHER_BURST_S = float(os.environ.get("IRONMULE_GGML_GATHER_BURST_MS", 150)) / 1000
GATHER_BURST_QUIET_S = float(os.environ.get("IRONMULE_GGML_GATHER_BURST_QUIET_MS", 25)) / 1000
# GB1 (2026-11-23): after a burst, the next lone arrival is likely the first of another one. A100 pods 20 and 22: a
# 32-client burst that reached the server over about 50 ms started its first request alone after the 4 ms gap (a 441-row
# prefill of 217 ms) and the other 31 behind it, 2 of 3 waves on pod 22. For GATHER_BURST_MEMORY_S after a group of
# several prompts, a lone arrival waits with the burst's window, once: if no other arrives, the memory is cleared.
GATHER_BURST_MEMORY_S = float(os.environ.get("IRONMULE_GGML_GATHER_BURST_MEMORY_S", 30))
# PK1-C (2026-11-24, opt-in IRONMULE_GGML_PK1C=1): the service hands a burst over in two parts, the first one to
# three requests as a new group and the rest as joins 3 to over 60 ms later (A100 pod 32: its 32 HTTP requests admitted
# over about 100 ms); with the 25 ms quiet gap the window closed between them in about half the 32-client waves and
# the rest waited for a second prefill. Within GB1's memory the window then expects as many requests as the last run
# held at its peak: while fewer have come it waits for GATHER_EXPECT_QUIET_S of quiet (within the burst limit), and it
# ends at once when they are all there. Measured: one prefill call in 10 of 10 later 32-client waves against 6 of 10
# without it, but no wave-time gain resolved against the packed call's own spread, and a lone request within the
# memory waits up to the burst limit; so it stays off by default.
PK1C = os.environ.get("IRONMULE_GGML_PK1C", "0") == "1"
GATHER_EXPECT_QUIET_S = float(os.environ.get("IRONMULE_GGML_GATHER_EXPECT_QUIET_MS", 120)) / 1000
UBATCH = int(os.environ.get("IRONMULE_GGML_UBATCH", 0) or 0)
# PX1 (2026-11-24): with one KV stream per sequence llama.cpp splits a packed call sequentially: a pass takes the run of
# consecutive sequence ids that all still have tokens and advances each by the fewest any of them has left. Prefix
# reuse keeps a different number of template tokens in each stream (0-12 in the benchmark's later waves), so prompts of
# the same length become unequal and a 32-request prefill ran as 8-14 passes instead of 2, each a full trip through
# the model (A100 pod 32: 21.5 ms a pass, 0.335 ms a row, fitted over 60 calls). A packed group then keeps the same
# prefix in every stream (the smallest), when that saves passes for at most PX1_ROWS rows recomputed per pass saved.
# The recomputed tokens are computed as in any prefill. A100 pod 33 (one process, 12 ABBA pairs): 7 passes to 2 for 99
# rows, prefill -95 ms in every pair; HTTP over two starts per arm: 32-client wave median -96 ms, first token p50 -105
# ms, facts 18/18. On by default; IRONMULE_GGML_PX1=0 turns it off.
# PX2 (2026-11-25): the fixed row limit becomes a cost decision. A packed engine on one KV stream per sequence times
# native passes of its own sizes at its start (one sequence, logits on the last row only: 1, 8, 16, 32 rows, both sides
# of PK2's cuBLAS threshold and two large sizes), and prices a split as the sum of its passes' interpolated times. Every
# kept-prefix level of the group (cut every longer prefix to it) is priced with the passes it would really make, and
# the cheapest is taken when it saves at least PX2_MARGIN_MS. Pods 32/33's calls fit 16.5 ms + 0.98 ms a row for passes
# under 128 rows and 91 ms + 0.287 ms a row above (186 calls): repeat 1's 99 rows for 3 passes saved were refused by
# the fixed limit and are worth about 45 ms. Without the measurement (or IRONMULE_GGML_PX2=0) the fixed limit decides.
PX1 = os.environ.get("IRONMULE_GGML_PX1", "1") != "0"
PX1_ROWS = int(os.environ.get("IRONMULE_GGML_PX1_ROWS", 32))
PX2 = os.environ.get("IRONMULE_GGML_PX2", "1") != "0"
PX2_MARGIN_S = float(os.environ.get("IRONMULE_GGML_PX2_MARGIN_MS", 2)) / 1000


def split_shapes(seq_rows: dict[int, int], n_ubatch: int) -> list[int]:
    """PX1: the rows of every pass llama.cpp's sequential split_equal makes of a packed call ({sequence id: rows})."""
    left = {s: r for s, r in seq_rows.items() if r > 0}
    shapes = []
    while left:
        ids = sorted(left)
        run = [ids[0]]
        for s in ids[1:]:
            if s != run[-1] + 1:
                break
            run.append(s)
        step = max(1, min(min(left[s] for s in run), max(1, n_ubatch // len(run))))
        for s in run:
            left[s] -= step
            if left[s] <= 0:
                del left[s]
        shapes.append(step * len(run))
    return shapes


def split_passes(seq_rows: dict[int, int], n_ubatch: int) -> int:
    """PX1: how many passes llama.cpp's sequential split_equal makes of a packed call ({sequence id: rows})."""
    return len(split_shapes(seq_rows, n_ubatch))


class PassCosts:
    """PX2: seconds of one native pass of `rows` rows on this machine, interpolated between measured sizes ({rows:
    seconds}); beyond the largest one, extended with the slope of the last two."""

    def __init__(self, points: dict[int, float]):
        self.points = sorted((int(r), float(t)) for r, t in points.items() if r > 0 and t > 0)
        if len(self.points) < 2:
            raise ValueError("pass costs need at least two measured sizes")

    def __call__(self, rows: int) -> float:
        pts = self.points
        if rows <= pts[0][0]:
            return pts[0][1]
        for (r0, t0), (r1, t1) in zip(pts, pts[1:]):
            if rows <= r1:
                return t0 + (t1 - t0) * (rows - r0) / (r1 - r0)
        (r0, t0), (r1, t1) = pts[-2], pts[-1]
        return t1 + max(0.0, (t1 - t0) / (r1 - r0)) * (rows - r1)

    def split(self, shapes: Sequence[int]) -> float:
        return sum(self(r) for r in shapes)


# BF1 (2026-11-25): a native call of BF1_ROWS rows or more fills llama.cpp's batch with numpy array copies instead of a
# Python loop over its ctypes arrays (A100 pods 32/33: 13-16 ms of host batch build before every 32-client prefill of
# about 7,500 rows); the rows point into one contiguous sequence-id array of the engine's own. "0" keeps the loop.
BF1 = os.environ.get("IRONMULE_GGML_BF1", "1") != "0"
BF1_ROWS = 64
# DH1 (diagnostics, with the trace only): IRONMULE_GGML_STEPPROF=1 notes the median host time of each part of a run's
# full-width decode steps (pilot and admission, selection, drafts and rows, the native call, observers, accepting and
# streaming the tokens, the rest) as one "# stepprof" line per run
STEPPROF = os.environ.get("IRONMULE_GGML_STEPPROF") == "1"
# Diagnostics: IRONMULE_GGML_TRACE names a file that gets one line per native decode call (monotonic start, seconds,
# rows, sequences, output rows); off by default
TRACE = os.environ.get("IRONMULE_GGML_TRACE", "")
# PF3 (diagnostics, with the trace only): IRONMULE_GGML_PAUSE names a file for ironmule.ggml_pause's samples of native
# calls of IRONMULE_GGML_PAUSE_MIN_MS (1000) or more (a py-spy dump once a call passes IRONMULE_GGML_PAUSE_SPY_MS), and
# adds per-call CPU, fault, run-delay and throttle deltas to every trace line
PAUSE = os.environ.get("IRONMULE_GGML_PAUSE", "") if TRACE else ""


def trace_note(label: str) -> None:
    """A marker line ("# label monotonic") in the IRONMULE_GGML_TRACE file; nothing without it."""
    if TRACE:
        with open(TRACE, "a") as fh:
            fh.write(f"# {label} {time.monotonic():.6f}\n")
# PK2 (2026-11-19): with packing on a CUDA build that has the PK2 patch, the engine measures at its start whether prompt
# matmuls run faster through dequantization + cuBLAS than through llama.cpp's MMQ kernels, and from which row count;
# IRONMULE_CUBLAS_MIN_ROWS fixes the threshold instead (0 = never), IRONMULE_PK2=0 skips the measurement.
PK2_AUTO = os.environ.get("IRONMULE_CUBLAS_MIN_ROWS") is None and os.environ.get("IRONMULE_PK2", "1") != "0"
# PK2-M (2026-11-24, default): "mix" times the thresholds on a burst of half short (33-token) and half long prompts, as
# the benchmark's eight clients send, and chooses by it. On one KV stream per sequence such a burst runs as two passes
# (llama.cpp's split_equal), the short one of about 130 rows below every threshold "equal" measured. A100 pod 26,
# Qwen3-32B, 8 clients: "mix" chose 128 rows ("equal" 256); engine waves 1.392 against 1.439 s, first token 746 against
# 792 ms; over HTTP the first token came earlier in 3 of 3 repeats (median 824 against 869 ms), 170.7 against
# 166.6 tok/s (ranges overlap), facts 18/18. "equal": the earlier calibration on eight equal prompts.
PK2_CALIBRATION = os.environ.get("IRONMULE_PK2_CALIBRATION", "mix")
# W4 (2026-11-19): on a CUDA GPU of compute capability 8.0 or newer, packed engines generate and tune IronMule's W4R
# kernels for this GPU and GGUF (ironmule/w4_kernels.py, cached under IRONMULE_W4_CACHE) and hand them to the kg1 hook,
# unless IRONMULE_KG_LIB names a library; measure_kg then keeps them only for the token counts where the step got
# faster. "0" turns it off. The library repacks a matrix on its first call outside a CUDA graph capture.
W4_AUTO = os.environ.get("IRONMULE_W4", "1") != "0"
W4_CACHE = os.environ.get("IRONMULE_W4_CACHE") or str(Path.home() / ".cache" / "ironmule" / "w4")
CHECK_TEXT = "IronMule checks that the GGUF's tokenizer gives the registered model's token ids: 0123 ÄÖÜ ß -> ok."


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


def am1_function(L: Any) -> Any:
    """AM1: llama.cpp's `ironmule_set_argmax_only` toggle in the loaded library (native/ggml_patches/am1), or None."""
    import ctypes

    if os.environ.get("IRONMULE_ARGMAX", "1") == "0":
        return None
    try:
        folder = Path(L.llama_cpp._lib._name).parent if hasattr(L, "llama_cpp") else Path(os.environ.get("LLAMA_CPP_LIB_PATH", ""))
        fn = ctypes.CDLL(str(next(folder.glob("libllama.so*")))).ironmule_set_argmax_only
    except Exception:  # noqa: BLE001 - a build without AM1
        return None
    fn.argtypes, fn.restype = [ctypes.c_int], None
    return fn


# EM1 (2026-11-23): llama.cpp keeps a model's token embedding on the CPU ("very little benefit to offloading the input
# layer"), so with every layer on the GPU a decode step still wakes the CPU threadpool for the rows of its tokens
# (dequantized there and copied to the GPU), and the threads spin afterwards beside the server's Python threads. Builds
# with the em1 patch take Q4_K / Q6_K rows on the GPU; the engine then places the embedding there and a step needs no
# CPU work. Opt-in (IRONMULE_GGML_EM1=1): on an A100 with Qwen3-32B it gave the same tokens and saved only 10-15 ms of a
# 2.4 s 32-prompt prefill (decode steps unchanged), and its extra GPU memory (the table, the step's input rows) made the
# quality gate's KV-setting tuning run out of memory (2026-11-23, pod 22).
EM1 = os.environ.get("IRONMULE_GGML_EM1", "0") == "1"
EM1_TYPES = frozenset({"F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q4_K", "Q6_K"})


def _gguf_tensor_type(gguf: str, name: str) -> str | None:
    """The type name of one tensor in a GGUF (llama.cpp's own reader), or None."""
    root = Path(os.environ.get("IRONMULE_GGML_BINDINGS", "")) / "vendor" / "llama.cpp" / "gguf-py"
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import gguf as gguf_py

    for tensor in gguf_py.GGUFReader(gguf).tensors:
        if tensor.name == name:
            return tensor.tensor_type.name
    return None


def em1_overrides(L: Any, gguf: str) -> Any:
    """EM1: a NULL-terminated `llama_model_tensor_buft_override` array that puts `token_embd.weight` in the CUDA
    device's buffer, or None (switched off, no em1 patch in the build, not exactly one CUDA device, or an embedding
    type whose rows the build cannot take on the GPU)."""
    import ctypes

    if not EM1:
        return None
    try:
        folder = Path(L.llama_cpp._lib._name).parent if hasattr(L, "llama_cpp") else Path(os.environ.get("LLAMA_CPP_LIB_PATH", ""))
        cuda = ctypes.CDLL(str(next(folder.glob("libggml-cuda.so*"))))
        cuda.ironmule_em1.restype = ctypes.c_int
        if cuda.ironmule_em1() != 1 or not cuda_kernel_device(L):
            return None
        if _gguf_tensor_type(gguf, "token_embd.weight") not in EM1_TYPES:
            return None
        cuda.ggml_backend_cuda_buffer_type.argtypes, cuda.ggml_backend_cuda_buffer_type.restype = [ctypes.c_int], ctypes.c_void_p
        buft = cuda.ggml_backend_cuda_buffer_type(0)
    except Exception:  # noqa: BLE001 - no CUDA library, a build without EM1, an unreadable header: llama.cpp's placement
        return None
    if not buft:
        return None

    class Override(ctypes.Structure):
        _fields_ = [("pattern", ctypes.c_char_p), ("buft", ctypes.c_void_p)]

    overrides = (Override * 2)()
    overrides[0].pattern, overrides[0].buft = b"^token_embd\\.weight$", buft
    overrides[1].pattern, overrides[1].buft = None, None
    return overrides


def pk2_set_min_rows(rows: int) -> int | None:
    """PK2: route quantized matmuls with at least `rows` prompt rows through dequantization + cuBLAS (0 = llama.cpp's
    own kernels). Returns the previous threshold, or None where the loaded CUDA library lacks the PK2 patch."""
    import ctypes

    try:
        L = bindings()
        folder = Path(L.llama_cpp._lib._name).parent if hasattr(L, "llama_cpp") else Path(os.environ.get("LLAMA_CPP_LIB_PATH", ""))
        fn = ctypes.CDLL(str(next(folder.glob("libggml-cuda.so*")))).ironmule_pk2_set_min_rows
    except Exception:  # noqa: BLE001 - no CUDA library or a build without PK2
        return None
    fn.argtypes, fn.restype = [ctypes.c_long], ctypes.c_long
    return int(fn(int(rows)))


def gpu_offload(L: Any | None = None) -> bool:
    """Whether the loaded llama.cpp library has a GPU backend to offload layers to."""
    try:
        L = L or bindings()
        return bool(ggml_devices(L))
    except Exception:  # noqa: BLE001 - no library, no GPU
        return False


def cuda_kernel_device(L: Any | None = None) -> bool:
    """KG1/KG5 target a single CUDA GPU. Other registered providers retain their own kernels."""
    devices = ggml_devices(L)
    return len(devices) == 1 and devices[0]["provider"] == "cuda"


def pad_table(prices: dict[int, float], margin: float = 0.95) -> dict[int, int]:
    """PAD1: for each measured token count, the larger one to fill a step to when it was measured at least 5 % cheaper
    (the cheapest; the smaller of equally cheap ones)."""
    pad = {}
    for t in prices:
        best = min((m for m in prices if m >= t), key=lambda m: (prices[m], m))
        if best > t and prices[best] < margin * prices[t]:
            pad[t] = best
    return pad


def gpu_layers(L: Any | None = None) -> int:
    """Layers to offload: IRONMULE_GGML_GPU_LAYERS, else all (-1 means every layer to llama.cpp) where a GPU backend is."""
    if GPU_LAYERS is not None:
        requested = int(GPU_LAYERS)
        if requested < -1:
            raise ValueError("IRONMULE_GGML_GPU_LAYERS must be -1 (all layers) or a nonnegative count")
        if requested and not gpu_offload(L):
            raise ValueError("IRONMULE_GGML_GPU_LAYERS requested offload but GGML registered no GPU")
        return 999 if requested == -1 else requested
    return 999 if gpu_offload(L) else 0


def available() -> str | None:
    """Why this machine cannot serve the ggml engine, or None."""
    try:
        bindings()
    except Exception as exc:  # noqa: BLE001 - a missing or broken library is a reason, not a crash
        return f"llama.cpp's bindings are not available ({type(exc).__name__}: {exc})"[:300]
    return None


def _file_identity(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _hash_contents(path: str, identity=None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        if identity is not None and _file_identity(os.fstat(handle.fileno())) != identity:
            raise OSError("file changed before SHA-256 verification")
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
        if identity is not None and _file_identity(os.fstat(handle.fileno())) != identity:
            raise OSError("file changed during SHA-256 verification")
    return digest.hexdigest()


_cached_file_hash = lru_cache(maxsize=32)(_hash_contents)


def file_sha256(path: str) -> str:
    """Reuse a verified large regular file within this process while its identity is stable."""
    path = os.path.abspath(path)
    st = os.stat(path)
    if st.st_size < 1 << 20 or not stat.S_ISREG(st.st_mode):
        return _hash_contents(path)
    identity = _file_identity(st)
    digest = _cached_file_hash(path, identity)
    if _file_identity(os.stat(path)) != identity:
        raise OSError("file changed during SHA-256 verification")
    return digest


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
    context: Any = None  # speculation.NgramIndex over ids + tokens, kept up to date by the engine
    tag: str | None = None  # PF1: the serving request's id, for the trace only
    prefill_start_ns: int = 0  # PF1 (trace only): the start of the first native call with this prompt's rows
    prefill_end_ns: int = 0  # PF1 (trace only): the end of the call with its last prompt row
    seen_ns: int = field(default_factory=time.monotonic_ns)  # PF2: when the engine side made the job

    def full_context(self) -> list[int]:
        """ids + tokens, grown by the new tokens only (the engine's per-step view of a request's context)."""
        if self.context is None:
            from .speculation import NgramIndex

            self.context = NgramIndex(self.ids)
        self.context.follow(self.tokens)
        return self.context.tokens


class _PrefixTrie:
    """The sequences of a SharedPool by the tokens their cells hold, one node per token: the longest prefix any sequence
    shares with a prompt is a walk down the prompt (one dict lookup per token), not a scan of every sequence. A node
    knows the sequences that pass through it; ties go to the sequence that entered the pool first, as the scan's did."""

    __slots__ = ("root", "paths", "order", "_next")

    def __init__(self) -> None:
        self.root: list = [{}, set(), None, None]  # children, sequences, parent, token
        self.paths: dict[int, list[list]] = {}
        self.order: dict[int, int] = {}
        self._next = 0

    def add(self, seq: int) -> None:
        self.paths[seq] = []
        self.order[seq] = self._next
        self._next += 1

    def extend(self, seq: int, tokens: Sequence[int]) -> None:
        path = self.paths[seq]
        node = path[-1] if path else self.root
        for token in tokens:
            child = node[0].get(token)
            if child is None:
                child = [{}, set(), node, token]
                node[0][token] = child
            child[1].add(seq)
            path.append(child)
            node = child

    def truncate(self, seq: int, n: int) -> None:
        path = self.paths[seq]
        for node in reversed(path[n:]):
            node[1].discard(seq)
            if not node[1]:  # no sequence passes here, so none below either
                del node[2][0][node[3]]
        del path[n:]

    def remove(self, seq: int) -> None:
        if seq in self.paths:
            self.truncate(seq, 0)
            del self.paths[seq]
            del self.order[seq]

    def clear(self) -> None:
        self.root[0].clear()
        self.paths.clear()
        self.order.clear()

    def longest(self, ids: Sequence[int]) -> tuple[int | None, int]:
        """The sequence holding the longest prefix of `ids` (the earliest among equals) and its length."""
        node, depth = self.root, 0
        for token in ids:
            child = node[0].get(token)
            if child is None:
                break
            node, depth = child, depth + 1
        if depth == 0:
            return None, 0
        return min(node[1], key=self.order.__getitem__), depth


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
        self.keep_cells = int(KEEP * cells) if KEEP < 1 else None  # KVD1: the cells finished sequences may hold
        self.free_ids = list(range(ids))
        self.tokens: dict[int, list[int]] = {}  # sequence -> the tokens its cells hold, from position 0
        self.nodes: dict[int, list[int]] = {}  # sequence -> one node per held position
        self.ref: dict[int, int] = {}  # node -> sequences that hold it; a node is one physical cell
        self.next_node = 0
        self.active: set[int] = set()
        self.reserve: dict[int, int] = {}  # running sequence -> cells it may still take
        self.kept: OrderedDict[int, None] = OrderedDict()  # finished sequences kept as a cache, oldest first
        self.shared_tokens = 0
        self.trie = _PrefixTrie()  # self.tokens by prefix: the longest shared prefix in one walk down a prompt

    @property
    def used(self) -> int:
        return len(self.ref)

    def kept_cells(self) -> int:
        """Cells that only finished sequences hold (the prefix cache)."""
        running = {node for seq in self.active for node in self.nodes.get(seq, [])}
        return self.used - len(running)

    def slots(self) -> int:
        return max(0, self.cap - len(self.active))

    def reset(self) -> None:
        self.free_ids = sorted(set(self.free_ids) | set(self.tokens) | set(self.active))
        self.tokens.clear(); self.nodes.clear(); self.ref.clear(); self.active.clear(); self.reserve.clear()
        self.kept.clear()
        self.trie.clear()

    def longest(self, ids: Sequence[int]) -> tuple[int | None, int]:
        """The sequence holding the longest prefix of `ids` and its length (None, 0 when none shares a token)."""
        return self.trie.longest(ids)

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
                self.trie.remove(seq)
                self.free_ids.append(seq)
                return True
        return False

    def place(self, ids: list[int], limit: int) -> tuple[int, int] | None:
        """A sequence for a new request and how many of its prompt tokens it already holds, or None while it does
        not fit (it waits)."""
        best, match = self.trie.longest(ids) if self.e.prefix_reuse else (None, 0)
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
                self.trie.truncate(seq, keep)
        else:
            while not self.free_ids:
                if not self._evict(best):
                    return None
            seq = self.free_ids.pop(0)
            self.tokens[seq], self.nodes[seq] = [], []
            self.trie.add(seq)
            if keep:
                self.e.L.llama_memory_seq_cp(self.e.memory, best, seq, 0, keep)
                shared = self.nodes[best][:keep]
                for node in shared:
                    self.ref[node] += 1
                self.nodes[seq], self.tokens[seq] = list(shared), list(ids[:keep])
                self.trie.extend(seq, ids[:keep])
        self.shared_tokens += keep
        self.active.add(seq)
        self.reserve[seq] = len(ids) - keep + limit + self.slack
        return seq, keep

    def sync(self, seq: int, held: Sequence[int], remaining: int, length: int | None = None) -> None:
        """After a step: `seq`'s cells hold the first `length` tokens of `held` (all of them by default); it may still
        take `remaining` cells. Costs the tokens that changed, not the context."""
        n = len(held) if length is None else length
        nodes, tokens = self.nodes[seq], self.tokens[seq]
        if len(nodes) > n:
            self._unref(nodes[n:])
            del nodes[n:]
            del tokens[n:]
            self.trie.truncate(seq, n)
        new = held[len(nodes):n]
        for token in new:
            self.ref[self.next_node] = 1
            nodes.append(self.next_node)
            tokens.append(token)
            self.next_node += 1
        if new:
            self.trie.extend(seq, new)
        self.reserve[seq] = max(0, remaining) + self.slack

    def release(self, seq: int, held: list[int] | None) -> None:
        """A request ends: keep its cells as a cache (`held`, the tokens they hold) or free them (None)."""
        self.active.discard(seq)
        self.reserve.pop(seq, None)
        if held is not None and held:
            self.sync(seq, held, 0)
            self.reserve.pop(seq, None)
            self.kept[seq] = None
            if self.keep_cells is not None:  # KVD1: the oldest finished sequences go beyond the budget
                while self.kept_cells() > self.keep_cells and self._evict(None):
                    pass
        else:
            self.e.L.llama_memory_seq_rm(self.e.memory, seq, -1, -1)
            self._unref(self.nodes.pop(seq, []))
            self.tokens.pop(seq, None)
            self.trie.remove(seq)
            self.free_ids.append(seq)

    def summary(self) -> dict[str, Any]:
        return {"cells": self.cells, "used": self.used, "running": len(self.active), "cap": self.cap,
                "kept": len(self.kept), "shared_tokens": self.shared_tokens}


class GgmlEngine:
    observer: Callable[[int, int, int, float], None] | None = None  # AG10: (rows, prompt, kv, seconds) of each step
    prefill_batch = False
    prefill_batch_long = False
    prefill_long_shapes: dict[int, tuple[int, ...]] | None = None
    prefill_long_pairs = prefill_long_quads = 0
    prefill_batch_calls = prefill_batch_prompts = prefill_batch_rows = 0
    pack = False
    pack_calls = pack_prompts = pack_rows = 0
    gpu_layers = 0
    kg4_scope = None
    pool: "SharedPool | None" = None
    pilot: Any = None
    pilot_refused: str | None = None
    collective: Any = None
    drafter: Any = None
    draft_refused: str | None = None
    _burst_at = float("-inf")  # GB1: when the engine last gathered several prompts at once
    _burst_peak = 0  # PK1-C: how many requests the last run held at once at most
    _em1: Any = None  # EM1: the override array the model was loaded with (None: llama.cpp's own placement)

    def __init__(self, gguf: str, *, threads: int = THREADS, seqs: int = SEQS, seq_ctx: int = SEQ_CTX,
                 speculate: bool = True, max_rows: int = MAX_ROWS, flash_attn: bool = False,
                 kv_unified: bool = False, kv_f32: bool = False, kv_q8: bool = False, model: Any = None,
                 shared: bool = False, cap: int | None = None, draft_gguf: str | None = None,
                 kernel_mask: bool = True):
        import numpy as np

        L = bindings()
        self.L, self.np = L, np
        L.llama_backend_init()
        self._owns_model = model is None  # SEQ1: a width measurement opens several contexts on one loaded model
        self.gpu_layers = gpu_layers(L)  # the same for a context opened on an already loaded model (SEQ1)
        self.w4 = self._attach_w4(gguf) if model is None else {}
        self.prefill_batch = self.gpu_layers > 0 and os.environ.get("IRONMULE_GGML_PREFILL_BATCH") == "1"
        # The long-only mode is dormant until this worker verifies its own measured receipt and holdout.
        self.prefill_batch_long = (bool(self.gpu_layers) and os.environ.get("IRONMULE_GGML_PREFILL_BATCH") == "long-444"
                                  and cuda_kernel_device(L))
        if self.gpu_layers and KG_LIB and not cuda_kernel_device(L):
            raise ValueError("a generated CUDA kernel library needs exactly one GGML CUDA device")
        if model is None:
            mparams = L.llama_model_default_params()
            mparams.n_gpu_layers = self.gpu_layers
            if self.gpu_layers:
                mparams.split_mode = SPLIT_MODES.get(SPLIT, 1)
            # EM1: with every layer on the GPU, the token embedding too (kept referenced while llama.cpp loads)
            self._em1 = em1_overrides(L, gguf) if self.gpu_layers >= 999 else None
            if self._em1 is not None:
                import ctypes

                mparams.tensor_buft_overrides = ctypes.cast(self._em1, ctypes.c_void_p)
        self.model = model or L.llama_model_load_from_file(str(gguf).encode(), mparams)
        if not self.model:
            raise ValueError(f"llama.cpp could not load {gguf}")
        self.threads, self.max_rows, self.speculate = threads, max_rows, speculate
        self.prefix_reuse = PREFIX_REUSE
        self.prompt_tokens = self.reused_tokens = 0
        self._lock = threading.Lock()
        self.shared = bool(shared)  # SP1: one pool, prefixes shared; `seqs * seq_ctx` cells, up to `cap` running
        self.cap = int(cap or min(LLAMA_MAX_SEQ, 2 * seqs)) if shared else seqs
        context_started = time.perf_counter()
        self._open(seqs, seq_ctx, flash_attn=flash_attn, kv_unified=kv_unified or self.shared, kv_f32=kv_f32,
                   kv_q8=kv_q8)
        self.steps = 0
        trace_note("init warm_up begin")  # PF2: what the start runs before serving, with times
        self._warm_up()
        trace_note("init check_argmax begin")
        self._argmax_only = self._check_argmax() if self.gpu_layers else None
        self.context_build_s = time.perf_counter() - context_started
        self.kg: dict[str, Any] = {}
        if KG_LIB and os.environ.get("IRONMULE_KG4_SCOPE") == "low-width8":
            from .kg4_scope import Kg4Scope

            try:
                self.kg4_scope, self.kg = Kg4Scope.calibrate(self, gguf, KG_LIB)
            except BaseException:
                self.close()
                raise
        elif KG_LIB and self.gpu_layers and kernel_mask:
            # SD1 (2026-11-23): the library's token-count mask is one per process; a second engine in the same process
            # (a draft model, kernel_mask=False) must not measure and overwrite the served model's mask
            trace_note("init measure_kg begin")
            self.kg = self.measure_kg(KG_LIB)
        trace_note("init measure_pk2 begin")
        self.pk2: dict[str, Any] = self.measure_pk2() if PK2_AUTO and getattr(self, "pack", False) else {}
        self.pass_costs: PassCosts | None = None
        if (PX1 and PX2 and getattr(self, "pack", False) and self.gpu_layers and model is None
                and not getattr(self, "settings", {}).get("kv_unified", True)):
            trace_note("init measure_passes begin")
            self.pass_costs = self.measure_pass_costs()
        trace_note("init measure_padding begin")
        self.pad_to: dict[int, int] = {}
        self.pad_prices: dict[int, float] = {}
        self.padded = 0
        if self.speculate and PAD != "0" and (self.gpu_layers or PAD == "1") and (
                self.settings.get("kv_unified") or self.gpu_layers):  # PAD3: KV streams on a GPU pad evenly
            self.pad_to = self.measure_padding()
        self.drafter = None
        if draft_gguf and speculate:  # DM1: a small model of the same family drafts, this one verifies
            self.drafter = self._open_drafter(draft_gguf, threads)
        self.pilot = None
        # PI1: the decision network, with this machine's measured step costs; its cost measurement decodes past 256
        # tokens of one sequence, so a context too small for that (SEQ1's KV-size probe opens 256 cells) runs without it
        trace_note("init pilot begin")
        if PILOT and model is None and seq_ctx >= 512:
            from .pilot import Pilot

            pilot = Pilot(self, self.measure_costs(), max_cap=self.cap if self.pool is not None else self.seqs,
                          gpu=bool(self.gpu_layers))
            self.pilot = pilot if pilot.active else None
            self.pilot_refused = None if pilot.active else pilot.reason
        if TRACE:  # PF2: the start's choices that steer later calls
            trace_note("init end")
            import json as _json

            with open(TRACE, "a") as fh:
                fh.write("# init result " + _json.dumps({
                    "pk2": self.pk2, "kg_tokens": (self.kg or {}).get("tokens_served"), "pad_to": self.pad_to,
                    "pilot": self.pilot is not None, "argmax_only": self._argmax_only is not None,
                    "context_build_s": round(self.context_build_s, 3)}, default=str) + "\n")

    def _step_prices(self, counts: Any, repeats: int = 5) -> dict[int, float]:
        """Seconds of a step of t tokens of one sequence at positions 64+ (median of `repeats`), for each t in `counts`."""
        bos = max(0, int(self.L.llama_vocab_bos(self.vocab)))
        self._clear()
        self._decode([(bos, j, 0, j == 63) for j in range(64)])
        prices: dict[int, float] = {}
        for t in counts:
            times = []
            for _ in range(repeats + 1):
                started = time.perf_counter()
                self._decode([(bos, 64 + j, 0, True) for j in range(t)])
                times.append(time.perf_counter() - started)
                self.L.llama_memory_seq_rm(self.memory, 0, 64, -1)
            prices[t] = sorted(times[1:])[len(times[1:]) // 2]
        self._clear()
        return prices

    def _check_argmax(self) -> Any:
        """AM1 serves only where it returns the same tokens as the host argmax of the full logits."""
        fn = am1_function(self.L)
        if fn is None:
            return None
        bos = max(0, int(self.L.llama_vocab_bos(self.vocab)))
        rows = [(bos + j % 7, j, 0, j % 9 == 8) for j in range(48)]
        try:
            self._clear()
            full = self._decode(rows)
            self._clear()
            self._argmax_only = fn
            fast = self._decode(rows)
        except Exception:  # noqa: BLE001 - keep the full logits
            fast = None
        finally:
            self._argmax_only = None
            self._clear()
        return fn if fast == full else None

    def _attach_w4(self, gguf: str) -> dict[str, Any]:
        """W4: this GPU's tuned W4R library for this GGUF (built and measured once, then cached), set as the kg1 hook's
        library before the first native call. {} where it does not apply."""
        global KG_LIB
        if not (W4_AUTO and PACK and self.gpu_layers and not KG_LIB and "IRONMULE_KG_LIB" not in os.environ
                and cuda_kernel_device(self.L)):
            return {}
        try:
            from .w4_kernels import ensure

            got = ensure(str(gguf), Path(W4_CACHE), log=lambda *a, **k: None)
        except Exception as exc:  # noqa: BLE001 - a failed build or measurement leaves llama.cpp's kernels serving
            return {"error": f"{type(exc).__name__}: {exc}"[:200]}
        if got is None:
            return {}
        lib, table = got
        os.environ["IRONMULE_KG_LIB"], os.environ["IRONMULE_W4_TABLE"] = str(lib), str(table)
        KG_LIB = str(lib)
        return {"lib": lib.name, "table": table.name}

    def measure_pk2(self, thresholds: Sequence[int] = (256, 512, 1024), repeats: int = 3) -> dict[str, Any]:
        """PK2: time one packed prompt burst (eight 444-token prompts on eight KV streams, as a busy step carries) with
        llama.cpp's own matmul kernels and with the cuBLAS path from each row threshold on, alternating arms; keep the
        fastest threshold when it is at least 5 % faster and its teacher-forced likelihood on a 512-token chunk stays
        within OPTION_BOUND of the incumbent's. {} where the loaded CUDA library lacks PK2 or there is no GPU."""
        if not self.gpu_layers or not getattr(self, "pack", False) or pk2_set_min_rows(0) is None:
            return {}
        bos = max(0, int(self.L.llama_vocab_bos(self.vocab)))
        ids = self.tokenize(" ".join(f"Site {1000 + j} measured {20 + j % 60} units." for j in range(400)))
        width = min(8, self.seqs)
        length = min(444, self.seq_ctx - 8, max(64, (len(ids) - 1) // width))
        group = [(seq, Job([bos] + ids[seq * length:(seq + 1) * length - 1], 1), 0) for seq in range(width)]
        mix = PK2_CALIBRATION == "mix"
        if mix:  # PK2-M: half short prompts (the lowest sequence ids, as the server sorts them), half long
            short = min(33, length)
            group = [(seq, Job([bos] + ids[seq * length:seq * length + (short if seq < width // 2 else length) - 1], 1), 0)
                     for seq in range(width)]
            thresholds = tuple(sorted({128, *thresholds}))

        def burst() -> float:
            self._clear()
            started = time.perf_counter()
            self._pack_prefill(group)
            return time.perf_counter() - started

        arms = (0, *thresholds)
        times: dict[int, list[float]] = {t: [] for t in arms}
        try:
            for t in arms:  # warm each path once (cuBLAS handles, dequantization buffers)
                pk2_set_min_rows(t)
                burst()
            for _ in range(repeats):
                for t in arms:
                    pk2_set_min_rows(t)
                    times[t].append(burst())
            med = {t: sorted(v)[len(v) // 2] for t, v in times.items()}
            # the smallest threshold within 2 % of the fastest: a single prompt of a few hundred rows gains too
            fastest = min(med[t] for t in thresholds)
            best = min(t for t in thresholds if med[t] <= 1.02 * fastest)
            chosen, reason = 0, "llama.cpp's kernels were as fast"
            if med[best] <= 0.95 * med[0]:
                chunk = ids[:min(len(ids), max(513, best + 65), self.seq_ctx - 1)]
                pk2_set_min_rows(0)
                ref = self.chunk_nll(chunk, [bos])
                pk2_set_min_rows(best)
                cand = self.chunk_nll(chunk, [bos])
                ratio = math.exp(cand - ref)
                if ratio <= OPTION_BOUND:
                    chosen, reason = best, f"{med[0] / med[best]:.2f}x a packed burst, likelihood ratio {ratio:.5f}"
                else:
                    reason = f"likelihood ratio {ratio:.5f} above {OPTION_BOUND}"
        finally:
            self._clear()
        pk2_set_min_rows(chosen)
        return {"min_rows": chosen, "reason": reason, "rows": sum(len(job.ids) for _, job, _ in group),
                "calibration": "mix" if mix else "equal",
                "burst_ms": {str(t): round(1000 * v, 1) for t, v in med.items()}}

    def measure_pass_costs(self, repeats: int = 3) -> PassCosts | None:
        """PX2: the seconds of one native pass of t prompt rows (one sequence after a 64-token prefix, logits on its
        last row only, as a prefill pass asks), median of `repeats` after one warm call, for small passes (1-32 rows),
        both sides of PK2's cuBLAS threshold and two large passes. None where it cannot be measured; PX1's fixed limit
        then decides."""
        threshold = int((getattr(self, "pk2", None) or {}).get("min_rows") or 0)
        room = min(self.seq_ctx, getattr(self, "batch_cap", self.seq_ctx)) - 64 - 8
        large = max(256, 2 * threshold)
        counts = {1, 8, 16, 32, large, 4 * large}
        if threshold > 33:
            counts |= {threshold - 1, threshold}
        counts = sorted(t for t in counts if t <= room)
        if len(counts) < 4:
            return None
        bos = max(0, int(self.L.llama_vocab_bos(self.vocab)))
        points: dict[int, float] = {}
        try:
            self._clear()
            self._decode([(bos, j, 0, j == 63) for j in range(64)])
            for t in counts:
                times = []
                for _ in range(repeats + 1):
                    started = time.perf_counter()
                    self._decode([(bos, 64 + j, 0, j == t - 1) for j in range(t)])
                    times.append(time.perf_counter() - started)
                    self.L.llama_memory_seq_rm(self.memory, 0, 64, -1)
                points[t] = sorted(times[1:])[len(times[1:]) // 2]
            costs = PassCosts(points)
        except Exception:  # noqa: BLE001 - PX1's fixed limit decides without it
            return None
        finally:
            self._clear()
        if TRACE:
            trace_note("pass_costs " + json.dumps({str(r): round(1000 * t, 2) for r, t in points.items()}))
        return costs

    def measure_kg(self, path: str, repeats: int = 7) -> dict[str, Any]:
        """KG1: the step for 1-8 tokens (KG4: and 16, 24, 32) with the generated kernels on and off (alternating, median
        of `repeats`); the library keeps the token counts where it was at least 3 % faster. {} where it is not loaded."""
        import ctypes

        try:
            lib = ctypes.CDLL(path)  # the handle llama.cpp's hook opened: the same mask
            lib.ironmule_kg_set_mask.argtypes = [ctypes.c_uint32]
            lib.ironmule_kg_table.restype = ctypes.c_char_p
        except (OSError, AttributeError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"[:200]}
        # KG4: 9-32 tokens run the tensor-core kernels of one tile count (8 tokens each) at about one price, so 16, 24
        # and 32 stand for 9-16, 17-24 and 25-32. Q6_K is chosen on its own where the library has a mask for it: the
        # step with Q4_K alone, and with Q4_K and Q6_K, each against the better of the arms without it
        counts = list(range(1, 9)) + [16, 24, 32]
        six = hasattr(lib, "ironmule_kg_set_mask6")
        if six:
            lib.ironmule_kg_set_mask6.argtypes = [ctypes.c_uint32]
        arms = ((0, 0), (0xFFFFFFFF, 0), (0xFFFFFFFF, 0xFFFFFFFF)) if six else ((0, None), (0xFFFFFFFF, None))
        prices: list[dict[int, list[float]]] = [{t: [] for t in counts} for _ in arms]
        for _ in range(repeats):
            for (mask4, mask6), store in zip(arms, prices):
                lib.ironmule_kg_set_mask(mask4)
                if mask6 is not None:
                    lib.ironmule_kg_set_mask6(mask6)
                for t, v in self._step_prices(counts, repeats=1).items():
                    store[t].append(v)
        med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
        off, on = prices[0], prices[1]
        keep = [t for t in counts if med(on[t]) <= 0.97 * med(off[t])]
        keep6 = [t for t in counts if six and med(prices[2][t]) <= 0.97 * min(med(off[t]), med(on[t]))]
        bits = lambda ts: sum(1 << (u - 1) for t in ts for u in ((t,) if t <= 8 else range(t - 7, t + 1)))  # noqa: E731
        mask = bits(keep)
        if six:
            lib.ironmule_kg_set_mask6(bits(keep6))
        lib.ironmule_kg_set_mask(mask)
        self._kg_lib = lib
        try:
            entries = len(json.loads(lib.ironmule_kg_table().decode()).get("entries", []))
        except (ValueError, AttributeError):
            entries = None
        out = {"lib": os.path.basename(path), "entries": entries, "tokens_served": keep,
               "step_ms_on": {str(t): round(med(on[t]) * 1000, 2) for t in on},
               "step_ms_off": {str(t): round(med(off[t]) * 1000, 2) for t in off}}
        if six:
            out["q6k_tokens_served"] = keep6
            out["step_ms_q6k"] = {str(t): round(med(prices[2][t]) * 1000, 2) for t in counts}
        return out

    def measure_padding(self, repeats: int = 5) -> dict[int, int]:
        """PAD1: a verify step's seconds for 1 to `max_rows` tokens of one sequence (median of `repeats`), and for each
        token count the larger count to fill it to when that one is measured at least 5 % cheaper."""
        bos = max(0, int(self.L.llama_vocab_bos(self.vocab)))
        self._clear()
        self._decode([(bos, j, 0, j == 63) for j in range(64)])
        prices: dict[int, float] = {}
        for t in range(1, self.max_rows + 1):
            times = []
            for _ in range(repeats + 1):
                started = time.perf_counter()
                self._decode([(bos, 64 + j, 0, True) for j in range(t)])
                times.append(time.perf_counter() - started)
                self.L.llama_memory_seq_rm(self.memory, 0, 64, -1)
            prices[t] = sorted(times[1:])[len(times[1:]) // 2]
        self._clear()
        self.pad_prices = prices
        return pad_table(prices)

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
        from .draft_model import BatchDrafter, Drafter

        sd = SD_K > 0 and bool(self.gpu_layers)  # SD1: one draft per running request, all in one call
        engine = GgmlEngine(gguf, threads=threads, seqs=self.seqs if sd else DRAFT_SLOTS, seq_ctx=self.seq_ctx,
                            speculate=False, flash_attn=True, kernel_mask=False)
        if engine.n_vocab != self.n_vocab or engine.tokenize(CHECK_TEXT) != self.tokenize(CHECK_TEXT):
            engine.close()
            self.draft_refused = "the draft GGUF's tokenizer is not the model's"
            return None
        if sd:
            return BatchDrafter(engine, k=SD_K, max_running=SD_MAX_RUNNING)
        return Drafter(engine, max_running=DRAFT_MAX_RUNNING)

    def _batch_for(self, count: int) -> Any:
        # ctypes' row pointers have no Python bounds. n_batch is the usable
        # limit even when extra native slots were reserved for speculative rows.
        cap = getattr(self, "batch_cap", self.seq_ctx)
        if count > cap:
            raise ValueError(f"{count} decode rows exceed the context batch limit of {cap}")
        self.batch.n_tokens = count
        return self.batch

    def _decode_conf(self, rows: list[tuple[int, int, int, bool]]) -> list[tuple[int, float]]:
        """As `_decode`, but per row that asks for logits its top token and that token's probability."""
        b, np = self._batch_for(len(rows)), self.np
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
        physical = getattr(self, "_study_ubatch", 512)
        if physical != 512 and not (physical in (1024, 2048) and self.prefill_batch_long and self.gpu_layers
                                    and seqs == 32 and seq_ctx == 2048 and not shared and not kv_unified and flash_attn):
            raise ValueError("larger physical batches require the isolated split-KV long-prompt study")
        self.pack = PACK and bool(self.gpu_layers) and not shared and not kv_unified
        logical = seq_ctx
        if self.pack:  # PK1: one call carries a step's prompts; its micro-batch as set (or llama.cpp's 512)
            logical = max(seq_ctx, PACK_TOKENS)
            physical = min(logical, UBATCH or physical)
        params.n_ctx, params.n_batch, params.n_ubatch = seqs * seq_ctx, logical, physical
        params.n_seq_max, params.n_threads, params.n_threads_batch = ids, threads, threads
        params.flash_attn, params.kv_unified = bool(flash_attn), bool(kv_unified)  # GG1: measured at the first start
        if DEFRAG > 0:  # KVD1: llama.cpp compacts the cache when holes reach this share of its attended range
            params.defrag_thold = DEFRAG
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
        self.batch_cap = max(logical, self.max_rows)
        self.batch = L.llama_batch_init(self.batch_cap, 0, seqs)
        self._bf1 = self._bf1_views(self.batch, self.batch_cap)
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
        if self.prefill_batch_long:
            self.prefill_batch = False  # the previous receipt and holdout apply only to their original context
            self._study_ubatch = 512
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
    def _decode(self, rows: list[tuple[int, int, int, bool]], *, evict: bool = True) -> list[int]:
        if TRACE:
            probe = getattr(self, "_pause_probe", None)
            if PAUSE and probe is None:
                from ironmule.ggml_pause import PauseProbe

                probe = self._pause_probe = PauseProbe(
                    PAUSE, float(os.environ.get("IRONMULE_GGML_PAUSE_MIN_MS", 1000)),
                    spy_ms=float(os.environ.get("IRONMULE_GGML_PAUSE_SPY_MS", 0)))
            if probe is not None:
                probe.begin()
            started_ns = time.monotonic_ns()
            try:
                out = self._decode_once(rows, evict=evict)
            finally:
                ended_ns = time.monotonic_ns()
                pause_fields = probe.end(f"rows={len(rows)}") if probe is not None else ""
            # PF1: a row inside its request's prompt is a prompt row, one past it decodes, one of a sequence without a
            # request is a KG3 placeholder. Times around the call are taken where llama.cpp returns anyway: no added
            # synchronisation.
            jobs = getattr(self, "_tl_jobs", None) or {}
            seqs = sorted({r[2] for r in rows})
            prompt_rows = filler_rows = 0
            for r in rows:
                job = jobs.get(r[2])
                if job is None:
                    filler_rows += 1
                elif r[1] < len(job.ids):
                    prompt_rows += 1
            for seq in seqs:
                job = jobs.get(seq)
                if job is not None and not job.prefill_end_ns and any(r[2] == seq and r[1] < len(job.ids) for r in rows):
                    job.prefill_start_ns = job.prefill_start_ns or started_ns
                    if any(r[2] == seq and r[1] + 1 == len(job.ids) for r in rows):
                        job.prefill_end_ns = ended_ns
            with open(TRACE, "a") as fh:
                # PF2: the batch built on the host, llama_decode, then the readback of the sampled rows (ms)
                built, decoded = getattr(self, "_tl_marks", (started_ns, ended_ns))
                fh.write(f"{started_ns / 1e9:.6f} {(ended_ns - started_ns) / 1e9:.6f} {len(rows)} {len(seqs)} "
                         f"{sum(1 for r in rows if r[3])} {prompt_rows} {len(rows) - prompt_rows - filler_rows} "
                         f"{filler_rows} seqs={','.join(map(str, seqs))} b={(built - started_ns) / 1e6:.3f} "
                         f"l={(decoded - built) / 1e6:.3f} r={(ended_ns - decoded) / 1e6:.3f}{pause_fields}\n")
            return out
        return self._decode_once(rows, evict=evict)

    def _bf1_views(self, batch: Any, cap: int) -> dict[str, Any] | None:
        """BF1: numpy views of the batch's arrays, and one contiguous sequence-id array the rows point into."""
        import ctypes

        np = self.np
        try:
            # llama.cpp's own per-row arrays (their addresses: an indexed pointer is a view of its slot), given back
            # before its free
            own = [ctypes.cast(batch.seq_id[i], ctypes.c_void_p).value for i in range(cap)]
            seq = np.zeros(cap, dtype=np.int32)
            base = seq.ctypes.data
            kind = type(batch.seq_id[0])
            for i in range(cap):
                batch.seq_id[i] = ctypes.cast(base + 4 * i, kind)
            return {"own": own, "seq": seq,
                    "token": np.ctypeslib.as_array(batch.token, shape=(cap,)),
                    "pos": np.ctypeslib.as_array(batch.pos, shape=(cap,)),
                    "n_seq": np.ctypeslib.as_array(batch.n_seq_id, shape=(cap,)),
                    "logits": np.ctypeslib.as_array(batch.logits, shape=(cap,))}
        except Exception:  # noqa: BLE001 - the per-row loop fills it
            return None

    def _decode_once(self, rows: list[tuple[int, int, int, bool]], *, evict: bool = True) -> list[int]:
        b = self._batch_for(len(rows))
        views = getattr(self, "_bf1", None)
        if views is not None and BF1 and len(rows) >= BF1_ROWS:  # BF1: a large call's rows in four array copies
            n = len(rows)
            table = self.np.array(rows, dtype=self.np.int64).reshape(n, 4)
            views["token"][:n], views["pos"][:n], views["seq"][:n] = table[:, 0], table[:, 1], table[:, 2]
            views["n_seq"][:n], views["logits"][:n] = 1, table[:, 3]
        else:
            for i, (token, pos, seq, logits) in enumerate(rows):
                b.token[i], b.pos[i], b.n_seq_id[i], b.logits[i] = token, pos, 1, int(logits)
                b.seq_id[i][0] = seq
        built = time.monotonic_ns() if TRACE else 0  # PF2 (trace only)
        am = getattr(self, "_argmax_only", None)
        if am is not None:  # AM1: only each output row's argmax leaves the GPU
            am(1)
            try:
                status = self.L.llama_decode(self.ctx, b)
                if status == 1 and evict and getattr(self, "pool", None) is not None:
                    with self._lock:
                        while self.pool._evict(None):
                            pass
                    status = self.L.llama_decode(self.ctx, b)
            finally:
                am(0)
            if TRACE:
                self._tl_marks = (built, time.monotonic_ns())
            if status != 0:
                raise RuntimeError("llama_decode failed")
            out = []
            for i, row in enumerate(rows):
                if not row[3]:
                    continue
                ptr = self.L.llama_get_logits_ith(self.ctx, i)
                value = ptr[0]
                if value != int(value) or not 0 <= value < self.n_vocab:
                    # llama.cpp fell back to copying the whole row (AM1 could not run): its first element is a
                    # logit, not an index, so the argmax is taken here
                    value = self.np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).argmax()
                out.append(int(value))
            return out
        status = self.L.llama_decode(self.ctx, b)
        if status == 1 and evict and getattr(self, "pool", None) is not None:  # SP1: no free cells; drop the cached sequences
            with self._lock:
                while self.pool._evict(None):
                    pass
            status = self.L.llama_decode(self.ctx, b)
        if TRACE:
            self._tl_marks = (built, time.monotonic_ns())
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
        """The length of the longest common prefix: compared 64 tokens per slice in C, then token by token."""
        if type(a) is not type(b):
            a, b = list(a), list(b)
        n = min(len(a), len(b))
        i = 0
        while i + 64 <= n and a[i:i + 64] == b[i:i + 64]:
            i += 64
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _take_sequence(self, ids: list[int]) -> tuple[int, int]:
        """A free sequence for a prompt and how many of its first tokens the sequence's KV already holds (at least one
        prompt token is always computed, for its logits). Call with the lock held."""
        best, keep = self._free[0], 0
        settings = getattr(self, "settings", None)
        if settings is not None and not settings.get("kv_unified"):
            # KG3: with one KV stream per sequence llama.cpp keeps a call in one pass only for consecutive sequence
            # ids, so a new request takes the lowest free id unless a kept prefix of 64 tokens or more is elsewhere
            best = min(self._free)
            if self.prefix_reuse and self._kept:
                lowest = best
                for seq in self._free:
                    n = self._common(self._kept.get(seq) or [], ids)
                    # Reuse even a short match when the preferred stream already holds it.
                    if (seq == lowest and n >= keep) or (n >= 64 and n > keep):
                        best, keep = seq, n
            return best, min(keep, len(ids) - 1)
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
            rows = [(t, start + j, seq, last and j == len(chunk) - 1) for j, t in enumerate(chunk)]
            observer = self.observer
            if observer is not None:
                started = time.perf_counter()
            picks = self._decode(rows)
            if observer is not None:
                observer(0, len(chunk), start, time.perf_counter() - started)
            first = picks[-1] if picks else first
        return first

    def _prefill_groups(self, fresh: list[tuple[int, Job, int]]):
        """Adjacent KV stream IDs with equally long uncached suffixes can share rectangular native batches."""
        start = 0
        while start < len(fresh):
            seq, job, keep = fresh[start]
            length = len(job.ids) - keep
            end = start + 1
            widths = (self.prefill_long_shapes or {}).get(length, ()) if self.prefill_batch_long else ()
            if self.prefill_batch_long:
                # The machine's receipt names the cold rectangles its holdout and measured workload exercised.
                if keep or not widths:
                    yield fresh[start:end]
                    start = end
                    continue
            while length > 1 and end < len(fresh) and (end - start + 1) * length <= self.seq_ctx:
                next_seq, other, kept = fresh[end]
                if self.prefill_batch_long and (kept or end - start >= max(widths)):
                    break
                if next_seq != seq + end - start or len(other.ids) - kept != length:
                    break
                end += 1
            if self.prefill_batch_long:
                end = start + max((width for width in widths if width <= end - start), default=1)
            yield fresh[start:end]
            start = end

    def _group_prefill(self, group: list[tuple[int, Job, int]]) -> list[int]:
        rows = [(job.ids[pos], pos, seq, pos + 1 == len(job.ids))
                for seq, job, keep in group for pos in range(keep, len(job.ids))]
        observer = self.observer
        if observer is not None:
            started = time.perf_counter()
        picks = self._decode(rows)
        if observer is not None:
            observer(0, len(rows), sum(keep for _, _, keep in group), time.perf_counter() - started)
        if self.prefill_batch_long:
            if len(group) == 2:
                self.prefill_long_pairs += 1
            else:
                self.prefill_long_quads += 1
        self.prefill_batch_calls += 1
        self.prefill_batch_prompts += len(group)
        self.prefill_batch_rows += len(rows)
        return picks

    def _even_reuse(self, group: list[tuple[int, Job, int]]) -> list[tuple[int, Job, int]]:
        """PX1: cut the kept prefixes of a packed group to one level, when that saves llama.cpp passes; PX2: the level
        whose passes the measured pass costs price lowest (recomputed rows included)."""
        on = getattr(self, "px1", PX1)
        if (not on and not TRACE) or len(group) < 2:
            return group
        keeps = [keep for _, _, keep in group]
        low = min(keeps)
        if low == max(keeps) or sum(len(job.ids) - keep for _, job, keep in group) > self.batch_cap:
            return group  # nothing to even out, or the group spans several calls (each split on its own)
        ubatch = UBATCH or 512

        def shapes_at(level: int) -> list[int]:
            return split_shapes({seq: len(job.ids) - min(keep, level) for seq, job, keep in group}, ubatch)

        now_shapes = shapes_at(max(keeps))
        now = len(now_shapes)
        costs = getattr(self, "pass_costs", None)
        if costs is not None and getattr(self, "px2", PX2):
            now_s = costs.split(now_shapes)
            level, best_s = max(keeps), now_s
            for candidate in sorted(set(keeps))[:-1]:  # every lower level the group's own prefixes offer
                cost = costs.split(shapes_at(candidate))
                if cost < best_s:
                    level, best_s = candidate, cost
            even = len(shapes_at(level))
            extra = sum(max(0, keep - level) for keep in keeps)
            take = level < max(keeps) and now_s - best_s >= PX2_MARGIN_S
            trace_note(f"px1 passes {now} even {even} extra {extra} on {int(bool(on))} level {level} "
                       f"cost_ms {1000 * now_s:.1f} {1000 * best_s:.1f} model px2")
        else:
            level = low
            even = split_passes({seq: len(job.ids) - low for seq, job, _ in group}, ubatch)
            extra = sum(keep - low for keep in keeps)
            take = even < now and extra <= PX1_ROWS * (now - even)
            trace_note(f"px1 passes {now} even {even} extra {extra} on {int(bool(on))}")
        if not on or not take:
            return group
        out = []
        for seq, job, keep in group:
            if keep > level:
                if level == 0 or not self.L.llama_memory_seq_rm(self.memory, seq, level, -1):
                    self.L.llama_memory_seq_rm(self.memory, seq, -1, -1)
                    if level > 0:  # the stream could not be cut at `level`: it starts empty, the rest stay even
                        self.reused_tokens -= keep
                        job.reused = 0
                        out.append((seq, job, 0))
                        continue
                self.reused_tokens -= keep - level
                job.reused = level
            out.append((seq, job, min(keep, level)))
        self.px1_evened = getattr(self, "px1_evened", 0) + 1
        return out

    def _pack_prefill(self, group: list[tuple[int, Job, int]]) -> list[int]:
        """PK1: the prompts of `group` (ascending sequence ids) in as few native calls as the logical batch allows."""
        picks: list[int] = []
        rows: list[tuple[int, int, int, bool]] = []
        observer = self.observer

        def flush():
            if not rows:
                return
            started = time.perf_counter() if observer is not None else 0.0
            picks.extend(self._decode(rows))
            if observer is not None:
                observer(0, len(rows), 0, time.perf_counter() - started)
            self.pack_calls += 1
            self.pack_rows += len(rows)
            rows.clear()

        for seq, job, keep in group:
            if rows and len(rows) + len(job.ids) - keep > self.batch_cap:
                flush()
            rows.extend((job.ids[pos], pos, seq, pos + 1 == len(job.ids)) for pos in range(keep, len(job.ids)))
        flush()
        self.pack_prompts += len(group)
        return picks

    def check_prefill_batch(self, ids: list[int]) -> dict[str, Any]:
        """Compare the same real greedy requests with grouping disabled and enabled on this context."""
        limit = 8
        width = min(self.seqs, 8)
        length = min(33, self.seq_ctx // 2, self.seq_ctx - limit)
        if width < 2 or length < 2:
            return {"matches": False, "calls": 0}
        prompts = [ids[i * length:(i + 1) * length] for i in range(width)]
        if any(len(p) != length for p in prompts):
            raise ValueError("insufficient tokens for the grouped prefill gate")
        enabled = self.prefill_batch
        outputs = []
        calls = self.prefill_batch_calls
        try:
            for mode in (False, True):
                self._clear()
                self.prefill_batch = mode
                jobs = [Job(prompt, limit) for prompt in prompts]
                self.run(jobs)
                outputs.append([(job.tokens, job.reason) for job in jobs])
            valid = all(tokens and stop in ("length", "eos") for tokens, stop in outputs[0])
            return {"matches": valid and outputs[0] == outputs[1], "calls": self.prefill_batch_calls - calls}
        finally:
            self.prefill_batch = enabled
            self._clear()

    def free_slots(self) -> int:
        with self._lock:
            return self.pool.slots() if self.pool is not None else len(self._free)

    def _slots(self) -> int:
        return self.pool.slots() if self.pool is not None else len(self._free)

    # --- serving ------------------------------------------------------------------------------------------------------
    def run(self, jobs: list[Job], admit: Callable[[int], list[Job]] | None = None) -> list[Job]:
        """Serve `jobs` and every job `admit(free sequences)` hands over while they run; returns all, in order."""
        from .speculation import MAX_DRAFT, accept

        served: list[Job] = list(jobs)
        pending = list(jobs)
        admission_observer = getattr(self, "admission_observer", None)
        cohort = None
        active: dict[int, Job] = {}

        def finish(seq: int, job: Job, reason: str, keep: bool = True) -> None:
            job.reason, job.finished_ns = reason, time.monotonic_ns()
            if TRACE:  # PF1: the request's own timeline, monotonic seconds (0: not reached)
                self._tl_jobs.pop(seq, None)
                with open(TRACE, "a") as fh:
                    fh.write(f"# job seq={seq} tag={job.tag} prompt={len(job.ids)} reused={job.reused} "
                             f"tokens={len(job.tokens)} reason={reason} seen={job.seen_ns / 1e9:.6f} "
                             f"prefill_start={job.prefill_start_ns / 1e9:.6f} "
                             f"prefill_end={job.prefill_end_ns / 1e9:.6f} first={job.first_ns / 1e9:.6f} "
                             f"end={job.finished_ns / 1e9:.6f}\n")
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

        trace_note("run")
        if TRACE:
            self._tl_jobs: dict[int, Job] = getattr(self, "_tl_jobs", {})
        learning = getattr(self.pilot, "learning", None)
        decision = {"cap": 1 << 30, "budget": 1 << 30, "draft": True}
        arrived = 0
        scope = self.kg4_scope
        if scope is not None:
            scope.begin()
        peak = 0  # PK1-C
        prof = {} if STEPPROF and TRACE else None  # DH1 (trace only): host time of each part of a step
        marks = [time.perf_counter()]
        steps_prof: list[dict[str, float]] = []

        def mark(name: str) -> None:
            if prof is not None:
                now = time.perf_counter()
                prof[name] = prof.get(name, 0.0) + now - marks[0]
                marks[0] = now
        try:
            while True:
                if prof is not None:  # DH1: the step that ended here (decode-only steps of full width are kept)
                    mark("tail")
                    if prof.pop("_keep", 0):
                        steps_prof.append(dict(prof))
                    prof.clear()
                if self.pilot is not None:  # PI1: threads, running cap, prompt budget and drafting for this step
                    ctx = (sum(len(j.ids) + len(j.tokens) for j in active.values()) / len(active)) if active else 0.0
                    decision = self.pilot.decide(len(active), len(pending), sum(len(j.ids) for j in pending), ctx)

                def room() -> int:
                    return max(0, min(self._slots(), decision["cap"] - len(active)))

                if admit is not None and room():
                    more = admit(room())
                    served.extend(more)
                    pending.extend(more)
                    arrived += len(more)
                    if (getattr(self, "pack", False) and not active and pending and GATHER_S > 0
                            and (more or arrived == 0)):
                        # PK1: gather the rest of a burst before the first packed prefill
                        started_wait = last = time.monotonic()
                        trace_note(f"gather start {len(pending)}")  # PF2
                        limit, quiet = GATHER_S, GATHER_QUIET_S
                        # PK1-B: the window opened on a burst; GB1: or a burst ended recently
                        if len(pending) > 1 or started_wait - self._burst_at < GATHER_BURST_MEMORY_S:
                            limit, quiet = max(limit, GATHER_BURST_S), max(quiet, GATHER_BURST_QUIET_S)
                        # PK1-C: within GB1's memory, the size the last wave reached
                        expect = (min(self._burst_peak, self._slots())
                                  if PK1C and started_wait - self._burst_at < GATHER_BURST_MEMORY_S else 0)
                        # SV2 (2026-11-23): a burst that already fills every free sequence has nothing to wait for
                        while (time.monotonic() - started_wait < limit and room() and len(pending) < self._slots()
                               and not (expect and len(pending) >= expect)):
                            got = admit(room())
                            now = time.monotonic()
                            if got:
                                served.extend(got)
                                pending.extend(got)
                                arrived += len(got)
                                last = now
                                if len(pending) > 1:  # PK1-B: a burst; wait for the rest of it
                                    limit, quiet = max(limit, GATHER_BURST_S), max(quiet, GATHER_BURST_QUIET_S)
                            elif now - last >= (max(quiet, GATHER_EXPECT_QUIET_S) if expect else quiet):
                                break
                            else:
                                time.sleep(0.0005)
                        trace_note(f"gathered {len(pending)} expect {expect}")
                        # GB1: a group of several renews the memory; a lone arrival that waited in vain clears it, so
                        # a single user pays the burst's gap once
                        self._burst_at = time.monotonic() if len(pending) > 1 else float("-inf")
                peak = max(peak, len(active) + len(pending))  # PK1-C
                if scope is not None:
                    scope.update(len(active) + len(pending))
                if self.pilot is not None and learning is None and decision["threads"] != self.threads:
                    self.set_threads(decision["threads"])
                # GG2: the requests that start in this step: the shortest prompts first, within the step's prompt
                # budget (whole prompts, at least one), and one passed over too often first
                fresh: list[tuple[int, Job, int]] = []
                budget = min(PROMPT_BUDGET or self.seq_ctx, decision["budget"])
                if self.pool is not None and not PROMPT_BUDGET and decision["budget"] >= 1 << 30:
                    budget = 1 << 62  # the shared pool takes every waiting prompt that fits, unless a budget is set
                if (not PROMPT_BUDGET and self.pool is None and self.gpu_layers
                        and not getattr(self, "settings", {}).get("kv_unified", True)):
                    # PB1: one KV stream per request on a GPU: the Pilot's prompt budget cost 2-9 % on a T4 and gained
                    # nothing there, so the step takes what the Pilot-off engine takes
                    budget = self.seq_ctx
                    if getattr(self, "pack", False):  # PK1: a step's prompts share one packed call
                        budget = self.batch_cap
                if (admission_observer is not None and cohort is None and not active and 2 <= len(pending) <= 64
                        and all(j.limit == pending[0].limit for j in pending)):
                    # The same actual admission boundary the runtime rule can influence.
                    cohort = (time.monotonic_ns(), tuple(pending))
                if not active and len(pending) > 1 and getattr(self, "admission_budgets", None):
                    from .compare_learning import shape

                    first_limit = pending[0].limit
                    if all(j.limit == first_limit for j in pending):
                        local = self.admission_budgets.get(shape([len(j.ids) for j in pending], first_limit))
                        if local is not None and local < budget and sum(len(j.ids) for j in pending) > local:
                            budget = local
                            self.admission_budget_hits += 1
                mark("pilot_admit")
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
                            pooled = self.pool.longest(job.ids)[1]
                        if any(self._common(other.ids, job.ids) >= pooled + SHARE_WAIT for _, other, _ in fresh):
                            continue
                        if fresh and len(job.ids) - pooled > budget:  # PI2: the step's prompt budget (the Pilot's)
                            job.passed_over += 1
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
                    if TRACE:  # PF1
                        self._tl_jobs[seq] = job
                if fresh and learning is not None and decision["threads"] != self.threads:
                    self.set_threads(decision["threads"])  # includes separate prefills issued before the main decode
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
                mark("select")
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
                            wants[seq] = (job.full_context(), room)
                    modelled = self.drafter.draft(wants, len(order)) if wants else {}
                # SV3 (2026-11-23): with no row left for a draft and no PAD size to fill, the policy chooses none, so
                # the per-request context drafts are not looked up (about 1 ms a step at 32 requests)
                policy_rows = getattr(self.policy, "rows", None)
                no_room = (not modelled and isinstance(policy_rows, int) and policy_rows - len(order) <= 0
                           and not (getattr(self, "pad_to", None) or {}).get(len(order)))
                for seq in order:
                    job = active[seq]
                    pos = len(job.ids) + len(job.tokens) - 1
                    room = min(MAX_DRAFT, job.limit - len(job.tokens) - 1, self.seq_ctx - 1 - pos)
                    if no_room:
                        offers.append(([], 0))
                        continue
                    if seq in modelled:
                        offers.append((modelled[seq], DRAFT_BUCKET))
                        continue
                    context = job.full_context()
                    offer = job.context.draft(room) if self.policy is not None and room > 0 else ([], 0)
                    if self.collective is not None and room > 0 and len(offer[0]) < 2:  # SP4: others' answers
                        other = self.collective.draft(context, room)
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
                streams = getattr(self, "settings", None) is not None and not self.settings.get("kv_unified")
                if ks and streams and len(set(ks)) > 1:
                    # KG3: with one KV stream per sequence llama.cpp splits a call whose sequences carry different token
                    # counts into a pass per count; every sequence verifies as many drafts as the one with the fewest
                    ks = [min(ks)] * len(ks)
                real = list(ks)
                pad_to = getattr(self, "pad_to", None)
                want = pad_to.get(len(order) + sum(ks)) if pad_to else None
                if want and streams:
                    # PAD3: with one KV stream per sequence every stream must carry the same number of tokens, or the
                    # call splits; each verifies `per` (its own drafts first, then its last token again), so the step
                    # reaches the size measured cheaper, or the step is not padded
                    per = -(-want // len(order))
                    caps = [min(MAX_DRAFT, active[sq].limit - len(active[sq].tokens) - 1,
                                self.seq_ctx - len(active[sq].ids) - len(active[sq].tokens)) for sq in order]
                    if not modelled and all(per - 1 <= cap for cap in caps):
                        for i, sq in enumerate(order):
                            job = active[sq]
                            drafts = list(offers[i][0][:per - 1])
                            drafts += [job.tokens[-1]] * (per - 1 - len(drafts))
                            offers[i] = (drafts, offers[i][1])
                            ks[i] = per - 1
                        self.padded += sum(ks) - sum(real)
                    want = None
                if want:  # PAD1: guesses (the last token again) fill the step to the size measured cheaper
                    extra = want - len(order) - sum(ks)
                    for i, seq in enumerate(order * MAX_DRAFT):
                        if extra <= 0:
                            break
                        i %= len(order)
                        job = active[order[i]]
                        pos = len(job.ids) + len(job.tokens) - 1
                        cap = min(MAX_DRAFT, job.limit - len(job.tokens) - 1, self.seq_ctx - 1 - pos)
                        if ks[i] < cap and order[i] not in modelled:  # the draft model's rows keep its own lengths
                            if ks[i] >= len(offers[i][0]):
                                offers[i] = (list(offers[i][0][:ks[i]]) + [job.tokens[-1]], offers[i][1])
                            ks[i] += 1
                            extra -= 1
                    self.padded += sum(ks) - sum(real)
                rows: list[tuple[int, int, int, bool]] = []
                for seq, (drafts, _), k in zip(order, offers, ks):
                    job = active[seq]
                    pos = len(job.ids) + len(job.tokens) - 1
                    rows += [(t, pos + j, seq, True) for j, t in enumerate([job.tokens[-1]] + list(drafts[:k]))]
                decode_rows = len(rows)
                room = self.seq_ctx - decode_rows  # one call's tokens
                batched: list[tuple[int, Job]] = []
                singles = fresh
                delayed_groups = None
                if getattr(self, "pack", False) and streams and fresh:
                    group = self._even_reuse(sorted(fresh, key=lambda item: item[0]))  # PX1
                    for sq, other, _ in group:
                        active[sq] = other
                    for (sq, other, _), pick in zip(group, self._pack_prefill(group), strict=True):
                        take(sq, other, pick)
                    singles = []
                elif self.prefill_batch and streams and len(fresh) > 1:
                    def deliver_group(group):
                        for sq, other, _ in group:
                            active[sq] = other
                        try:
                            for (sq, other, _), pick in zip(group, self._group_prefill(group), strict=True):
                                take(sq, other, pick)
                        except BaseException:
                            active.update((sq, other) for sq, other, _ in fresh if other.reason is None)
                            raise

                    singles = []
                    delayed_groups = {} if self.prefill_batch_long else None
                    for group in self._prefill_groups(fresh):
                        if len(group) == 1:
                            singles.extend(group)
                        elif delayed_groups is None:
                            deliver_group(group)
                        else:
                            # Keep short and partially cached requests in the incumbent's arrival/admission order.
                            # Dispatch the rectangle when the incumbent would have reached its first member.
                            singles.append(group[0])
                            delayed_groups[group[0][0]] = group
                for seq, job, keep in singles:
                    if delayed_groups and seq in delayed_groups:
                        deliver_group(delayed_groups[seq])
                        continue
                    # KG3: with one KV stream per sequence a prompt beside others' rows would split the call into passes
                    if len(job.ids) - keep <= room and not (streams and (rows or batched) and len(job.ids) - keep > 1):
                        rows += [(t, j, seq, j == len(job.ids) - 1) for j, t in enumerate(job.ids) if j >= keep]
                        room -= len(job.ids) - keep
                        batched.append((seq, job))
                    else:  # too long to share this call: its own prefill first
                        active[seq] = job
                        take(seq, job, self._prefill(job.ids, seq, keep))
                prompt_rows = len(rows) - decode_rows
                fillers: list[tuple[int, int]] = []
                if streams and order and not batched and max(order) - min(order) + 1 > len(order):
                    # KG3: llama.cpp splits a call over KV streams into a pass per run of consecutive sequence ids; a
                    # free id between running ones gets placeholder rows (the BOS token past what its stream holds,
                    # no logits), removed again after the step, so the step stays one pass. Computed after this
                    # step's own prompt calls, from llama.cpp's own record of each stream.
                    per = len(rows) // len(order)
                    by_seq: dict[int, list[tuple[int, int, int, bool]]] = {}
                    for row in rows:
                        by_seq.setdefault(row[2], []).append(row)
                    bos = max(0, int(self.L.llama_vocab_bos(self.vocab)))
                    rows = []
                    for sq in range(min(order), max(order) + 1):
                        if sq in by_seq:
                            rows += by_seq[sq]
                            continue
                        held = int(self.L.llama_memory_seq_pos_max(self.memory, sq)) + 1
                        if held + per > self.seq_ctx:  # a full stream gives up its kept prefix for the placeholder
                            self.L.llama_memory_seq_rm(self.memory, sq, -1, -1)
                            self._kept.pop(sq, None)
                            held = 0
                        rows += [(bos, held + j, sq, False) for j in range(per)]
                        fillers.append((sq, held))
                if rows and learning is not None:
                    threads = decision["threads"]
                    # The learned trials priced only one real decode row per sequence. Draft selection,
                    # padding and late admissions above finish before this test of the actual call.
                    if (not fresh and not pending and not fillers and not modelled and len(rows) == len(order)
                            and len(order) <= learning.max_rows and not self.pilot.last_fallback):
                        band = int(ctx) // learning.bin_size
                        if all((len(active[s].ids) + len(active[s].tokens)) // learning.bin_size == band
                               for s in order):
                            threads = learning.choose(ctx, self.pilot.load.foreign, threads, len(order))
                    if threads != self.threads:
                        self.set_threads(threads)
                mark("offers_rows")
                started = time.perf_counter()
                picks = self._decode(rows) if rows else []
                seconds = time.perf_counter() - started
                mark("native")
                if prof is not None and not fresh and not batched and len(rows) == len(order) >= 8:
                    prof["_keep"] = 1
                for sq, held in fillers:  # KG3: the placeholders leave nothing behind
                    self.L.llama_memory_seq_rm(self.memory, sq, held, -1)
                if self.policy is not None and decode_rows and not batched:  # a prompt in the call is not a step's price
                    self.policy.observe_step(decode_rows, seconds, any(ks))
                if self.observer is not None and rows:  # AG10: the agent's world model learns every step's price
                    # Gap fillers execute in the decode batch and carry no request prompt.
                    self.observer(len(rows) - prompt_rows, prompt_rows,
                                  sum(len(j.ids) + len(j.tokens) for j in active.values()), seconds)
                mark("observe")
                self.steps += 1
                at = produced = 0
                for seq, (drafts, bucket), k, r in zip(order, offers, ks, real):
                    job = active[seq]
                    pos = len(job.ids) + len(job.tokens) - 1
                    new = accept(list(drafts[:k]), picks[at:at + k + 1])
                    at += k + 1
                    produced += len(new)
                    if seq in modelled:
                        self.drafter.settle(seq, r, min(r, len(new) - 1))
                    elif r and self.policy is not None:  # PAD1's guesses are not the drafts' acceptance
                        self.policy.observe_drafts(bucket, r, min(r, len(new) - 1))
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
                mark("accept_take")
                if self.pool is not None:  # SP1: what every running sequence's cells hold now
                    with self._lock:
                        for seq, job in active.items():
                            self.pool.sync(seq, job.full_context(), job.limit - len(job.tokens),
                                           length=len(job.ids) + len(job.tokens) - 1)
        finally:
            if steps_prof and TRACE:  # DH1: medians over this run's full-width decode steps, ms
                parts = sorted({k for d in steps_prof for k in d})
                trace_note("stepprof " + json.dumps({"steps": len(steps_prof), **{
                    k: round(1000 * statistics.median(d.get(k, 0.0) for d in steps_prof), 3) for k in parts}}))
            if peak > 1:  # PK1-C: a burst's size; a lone request between bursts leaves the memory as it was
                self._burst_peak = peak
                # and the burst memory itself: a window that caught only the burst's first request cleared it (GB1's
                # "lone arrival that waited in vain"), although the rest joined this run right after (A100 pod 32)
                self._burst_at = time.monotonic()
            if scope is not None:
                scope.end()
            for seq in list(active):  # a run that ends here leaves no KV it cannot vouch for
                finish(seq, active[seq], active[seq].reason or "cancelled", keep=False)
        if cohort is not None:
            started, group = cohort
            if all(j.reason in ("eos", "length") and started < j.first_ns <= j.finished_ns for j in group):
                seconds = (max(j.finished_ns for j in group) - started) / 1e9
                firsts = sorted((j.first_ns - started) / 1e6 for j in group)
                admission_observer(started, tuple(len(j.ids) for j in group), group[0].limit,
                                   sum(len(j.tokens) for j in group) / seconds,
                                   firsts[(95 * len(group) + 99) // 100 - 1])
        return served

    # --- the quality gate ---------------------------------------------------------------------------------------------
    def chunk_nll(self, ids: list[int], lead: list[int]) -> float:
        """Mean next-token negative log-likelihood of `ids[1:]` after `lead + ids[:-1]`, in one multi-position decode
        (the verify path), as `numeric_choice.nll_per_chunk` scores a chunk."""
        np = self.np
        self._clear()
        tokens = lead + ids[:-1]
        b = self._batch_for(len(tokens))
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
                "prefill_batch": {"enabled": self.prefill_batch, "calls": self.prefill_batch_calls,
                                  "mode": ("long-444" if self.prefill_batch_long else "all") if self.prefill_batch else "off",
                                  **({"long_shape_hits": {"2": self.prefill_long_pairs, "4": self.prefill_long_quads}}
                                     if self.prefill_batch_long else {}),
                                  "prompts": self.prefill_batch_prompts, "rows": self.prefill_batch_rows},
                **({"pack": {"calls": self.pack_calls, "prompts": self.pack_prompts, "rows": self.pack_rows,
                             "ubatch": UBATCH or 512, "batch": self.batch_cap,
                             "pk2": getattr(self, "pk2", {}),
                             "px1_evened": getattr(self, "px1_evened", 0),
                             **({"pass_ms": {str(r): round(1000 * t, 2) for r, t in self.pass_costs.points}}
                                if getattr(self, "pass_costs", None) is not None else {})}} if self.pack else {}),
                "prefix_reuse": {"on": self.prefix_reuse, "prompt_tokens": self.prompt_tokens,
                                 "reused_tokens": self.reused_tokens},
                **({"shared_pool": self.pool.summary()} if self.pool is not None else {}),
                **({"kernels": self.kg} if getattr(self, "kg", None) else {}),
                **({"w4": self.w4} if getattr(self, "w4", None) else {}),
                "argmax_on_device": getattr(self, "_argmax_only", None) is not None,
                **({"kg4_scope": self.kg4_scope.status()} if self.kg4_scope is not None else {}),
                **({"padding": {"to": {str(k): v for k, v in self.pad_to.items()}, "guesses": self.padded,
                                "step_ms_by_tokens": {str(k): round(v * 1000, 2) for k, v in self.pad_prices.items()}}}
                   if self.pad_prices else {}),
                **({"pilot": self.pilot.status()} if self.pilot is not None else
                   {"pilot": {"active": False, "reason": self.pilot_refused}} if self.pilot_refused else {}),
                **({"draft_model": self.drafter.summary()} if self.drafter is not None else
                   {"draft_model": {"refused": self.draft_refused}} if self.draft_refused else {}),
                **({"speculation": self.policy.summary()} if self.policy is not None else {"speculation": "off"})}

    def close(self) -> None:
        if getattr(self, "pilot", None) is not None:
            self.pilot.close()
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
            views = getattr(self, "_bf1", None)
            if views is not None:  # BF1: llama.cpp frees the per-row sequence arrays it allocated
                import ctypes

                kind = type(self.batch.seq_id[0])
                for i, address in enumerate(views["own"]):
                    self.batch.seq_id[i] = ctypes.cast(address, kind)
                self._bf1 = None
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


KV_F32_MARGIN = 1.08


def rank_settings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The autotune's order: highest score first, a float32 KV cache's score divided by KV_F32_MARGIN."""
    return sorted(rows, key=lambda row: -row["score"] / (KV_F32_MARGIN if row.get("kv_f32") else 1.0))


def autotune(gguf: str, steps: int = 24, repeats: int = 3) -> dict[str, Any]:
    """GG1 (amendment of 2026-11-10): llama.cpp's flash attention and unified KV cache, each on and off, timed on a fixed
    synthetic workload (up to 8 configured sequences of a fixed 64-token prompt and one sequence alone, `steps` decode steps each,
    median of `repeats`); the setting with the highest geometric mean of the two rates."""
    rows = []
    for setting in SETTINGS:
        engine = GgmlEngine(gguf, speculate=False, **setting)
        try:
            prompt = (engine.tokenize(CHECK_TEXT) * 8)[:64]
            rates = {}
            wide = min(8, engine.seqs)
            for width in dict.fromkeys((wide, 1)):
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
        rows.append({**setting, "tokens_per_s": rates, "score": math.sqrt(rates[str(wide)] * rates["1"])})
    # ST1 (2026-11-15): a float32 KV cache moves twice the bytes per attended token, which a 64-token tuning context
    # does not show: it ranks first only where it beats the rest by KV_F32_MARGIN (two fresh starts on the 4-vCPU Xeon
    # ranked it first and third on 1-6 % noise, and it then served 30 % slower at the context of real requests)
    rows = rank_settings(rows)
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
    """Free and total memory from the same GGML registry that runs the model.

    A selected CUDA worker may remap a physical index or UUID. Query its registry in a
    child before opening the model, never sum cards from another provider by host index.
    """
    rows = ggml_devices()
    if devices is not None:
        if not rows or any(row["provider"] != "cuda" for row in rows):
            return None
        if devices != os.environ.get("CUDA_VISIBLE_DEVICES"):
            import subprocess
            query = "import json;from ironmule_devices import ggml_devices;print(json.dumps(ggml_devices()))"
            try:
                proc = subprocess.run([sys.executable, "-c", query], capture_output=True, text=True, check=True,
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": devices}, timeout=15)
                rows = json.loads(proc.stdout)
            except (OSError, ValueError, subprocess.SubprocessError):
                return None
            if any(row["provider"] != "cuda" for row in rows):
                return None
    if SPLIT == "none":
        rows = rows[:1]  # llama.cpp's main GPU is device 0
    if not rows or any(not (0 <= row["free_memory"] <= row["total_memory"] and row["total_memory"] > 0) for row in rows):
        return None
    return sum(row["free_memory"] for row in rows), sum(row["total_memory"] for row in rows)


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
                   repeats: int = 3, long_ctx: int = 0, experience: dict | None = None) -> dict[str, Any]:
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
                if experience:
                    # Before adding a new width to AG10, compare its real greedy outputs to the
                    # local incumbent's at this workload. Different arrival limits can change matmul paths.
                    engine._clear()
                    length = experience["prompt_tokens"]
                    jobs = []
                    for index in range(experience["concurrency"]):
                        ids = engine.tokenize(f"Check {index}. List the next numbers after {index + 2}. ")
                        jobs.append(Job((ids * ((length + len(ids) - 1) // len(ids)))[:length], 6))
                    engine.run(jobs)
                    outputs = [(job.reason, job.tokens) for job in jobs]
                    if all(reason in ("length", "eos") and tokens for reason, tokens in outputs):
                        out.setdefault("output_digests", {})[str(width)] = hashlib.sha256(
                            json.dumps(outputs).encode()).hexdigest()
                if experience and width in experience["widths"]:
                    # Imported predictions decide which additional point to measure. Only the local
                    # decode time enters the agent. Benchmark throughput is never a per-step cost.
                    length = experience["prompt_tokens"]
                    rows_now = min(experience["concurrency"], width)
                    context = (prompt * ((length + len(prompt) - 1) // len(prompt)))[:length]
                    prices = []
                    for _ in range(repeats):
                        engine._clear()
                        for seq in range(rows_now):
                            engine._decode([(t, j, seq, j == length - 1) for j, t in enumerate(context)])
                        began = time.perf_counter()
                        for k in range(steps):
                            engine._decode([(prompt[k % len(prompt)], length + k, s, True) for s in range(rows_now)])
                        prices.append((time.perf_counter() - began) / steps)
                    out.setdefault("workload_steps", {})[str(width)] = {
                        "rows": rows_now, "kv": rows_now * (length + (steps - 1) // 2),
                        "seconds": statistics.median(prices)}
            finally:
                releasing = time.perf_counter()
                engine.close_context()
                out.setdefault("context_release_s", {})[str(width)] = time.perf_counter() - releasing
                out.setdefault("context_build_s", {})[str(width)] = engine.context_build_s
    finally:
        first.close_context()
        first.L.llama_model_free(model)
    return out


WIDTH_CHILD = r"""
import json, sys
from ironmule.ggml_engine import measure_widths
data = json.loads(sys.stdin.read())
print(json.dumps(measure_widths(sys.argv[1], data["settings"], data["seq_ctx"], data["widths"],
                                long_ctx=data.get("long_ctx", 0), experience=data.get("experience"))))
"""

CELL_CHILD = r"""
import json, sys
from ironmule.ggml_engine import GgmlEngine
data = json.loads(sys.stdin.read())
engine = GgmlEngine(sys.argv[1], seqs=1, seq_ctx=256, speculate=False, **data["settings"])
print(json.dumps({"cell_bytes": engine.kv_cell_bytes()}))
engine.close()
"""


def _kernel_digest() -> str | None:
    """Empty for stock kernels, a content hash for a pinned library, or None for an unresolved loader path."""
    name = os.environ.get("IRONMULE_KG_LIB")
    if not name:
        return ""
    path = Path(name)
    if path.is_absolute() and path.is_file():
        try:
            return file_sha256(str(path))
        except OSError:
            pass
    return None


def choose_sequences(gguf: str, settings: dict[str, bool] | None = None, *, seq_ctx: int | None = None,
                     cache: Path | None = None, root: Path | None = None, device: str | None = None,
                     extra_bytes: int = 0, experience_root: Path | None = None) -> dict[str, Any]:
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
        if found is None:
            return {"sequences": 1, "memory": "gpu-unknown", "seq_ctx": seq_ctx,
                    "reason": "visible GGML device memory unavailable; width measurement skipped"}
        available, total = found
    # Under generated matmuls even the default attention setting can change answers. AG10 may then
    # vary width only within the settings actually chosen by the gate.
    kernel = _kernel_digest()
    default_alternative = kernel == ""
    model_digest = file_sha256(gguf)
    machine = machine_key()
    experience = None
    if experience_root is not None and on_gpu:
        from .gpu_experience import suggest

        experience = suggest(experience_root, model_digest, ggml_devices(), machine, seq_ctx)
    key_parts = [model_digest, settings, seq_ctx, ENGINE_VERSION, WIDTH_VERSION, WIDTHS, default_alternative,
                 int(extra_bytes) >> 28, WIDTH_GAIN, machine, device, round(total / 2**30)]
    if not default_alternative:
        key_parts.append(kernel)
    if experience:
        key_parts.append(experience)
    key = hashlib.sha256(json.dumps(key_parts, sort_keys=True).encode()).hexdigest()[:24]
    cache = cache or Path(os.path.expanduser("~/.cache/ironmule/ggml_gate"))
    stored = cache / f"width-{key}.json"
    model_bytes = os.path.getsize(gguf) + int(extra_bytes)
    if kernel is not None and stored.exists():
        try:
            cached = json.loads(stored.read_text())
            if fitting_widths(cached["cell_bytes"], seq_ctx, model_bytes, available, total) == cached["fitting"]:
                return cached
        except (OSError, ValueError, KeyError, TypeError):
            pass
    root = root or Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")])),
           "HF_HUB_OFFLINE": "1",
           "IRONMULE_PILOT": "0"}  # SEQ1: its engines measure widths; the Pilot is not part of that
    env.pop("IRONMULE_GGML_SEQS", None)
    env["IRONMULE_CUDA_GRAPHS_ROWS"] = "1"  # GR3: one-row graphs in the measuring children (see the gate's children)
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

    cell = child(CELL_CHILD, {"settings": settings})
    record: dict[str, Any] = {"seq_ctx": seq_ctx, "memory_available": available, "memory_total": total,
                              "memory": "gpu" if on_gpu else "host",
                              "model_bytes": model_bytes, "gain_rule": WIDTH_GAIN, "machine": machine_key(),
                              "version": WIDTH_VERSION}
    if cell is None:
        return {**record, "sequences": SEQS, "reason": "the KV size could not be read; 8 sequences as before"}
    widths = fitting_widths(cell["cell_bytes"], seq_ctx, model_bytes, available, total)
    record.update(cell_bytes=cell["cell_bytes"], fitting=widths)
    trial_widths = widths
    if experience:
        # Every width is bounded by the local KV budget. The default single-width
        # emergency fallback has no headroom for exploratory allocations.
        budget = available - model_bytes - max(int(1.5 * 2**30), total // 10)
        fitting = [w for w in experience["widths"] if w <= max(widths) and w * seq_ctx * cell["cell_bytes"] <= budget]
        if len(fitting) == 2:
            trial_widths = sorted(set(widths) | set(fitting))
        else:
            experience = None
    if len(trial_widths) == 1:
        record.update(sequences=widths[0], reason="only this width fits the memory")
    else:
        payload = {"settings": settings, "seq_ctx": seq_ctx, "widths": trial_widths, "long_ctx": AGENT_LONG_CTX}
        if experience:
            payload["experience"] = experience
        measured = child(WIDTH_CHILD, payload)
        if measured is None and experience:
            # An optional workload probe must not prevent the usual width qualification.
            experience = None
            payload = {"settings": settings, "seq_ctx": seq_ctx, "widths": widths, "long_ctx": AGENT_LONG_CTX}
            measured = child(WIDTH_CHILD, payload)
        if measured is None:
            return {**record, "sequences": SEQS, "reason": "the width measurement failed; 8 sequences as before"}
        rates = {int(w): r for w, r in measured["rates"].items() if int(w) in widths}
        incumbent = decide_width(rates)
        if experience:
            digests = measured.get("output_digests", {})
            reference = digests.get(str(incumbent))
            checked = {int(w) for w, digest in digests.items() if reference and digest == reference}
            # Existing settings/widths keep their own gate and baseline; only matched new
            # widths may join the agent's choices. Unmatched anchors never train its model.
            for w in set(trial_widths) - checked:
                measured.get("workload_steps", {}).pop(str(w), None)
                if w not in widths:
                    measured["step_s"].pop(str(w), None)
                    measured["rates"].pop(str(w), None)
            record["experience_probe"] = {**experience, "locally_matched_widths": sorted(checked)}
        record.update(sequences=incumbent, rates=measured["rates"], step_s=measured["step_s"],
                      step_one_s=measured.get("step_one_s", {}), settings=settings,
                      reason="the widest width whose doubling still bought the gain rule")
        # AG10: the agent's alternatives -- the same widths under llama.cpp's default setting, which passes the gate
        # by definition and differs from flash attention in what a step pays per attended token
        alternatives = [measured]
        default = {k: False for k in DEFAULT_SETTINGS}
        if default_alternative and any(settings.values()) and os.environ.get("IRONMULE_GGML_AGENT", "1") != "0":
            other = child(WIDTH_CHILD, {"settings": default, "seq_ctx": seq_ctx, "widths": widths,
                                        "long_ctx": AGENT_LONG_CTX})
            if other is not None:
                alternatives.append(other)
        record["measured"] = alternatives
    record["seconds"] = round(time.perf_counter() - started, 1)
    if kernel is None or _kernel_digest() != kernel:
        return record
    try:
        cache.mkdir(parents=True, exist_ok=True)
        stored.write_text(json.dumps(record))
    except OSError:
        pass
    return record


def machine_key() -> str:
    """Bind GGML measurements to the CPU, visible backend devices and driver context."""
    import platform

    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    from ironmule_devices import runtime_binding

    from .ggml_binary import identity as native_identity

    binding = {**runtime_binding(ggml_devices()), "native": native_identity()}
    if os.environ.get("IRONMULE_KG_LIB") and os.environ.get("IRONMULE_KG4_SCOPE") == "low-width8":
        binding["kg4_scope"] = {"mode": "low-width8", "graphs": os.environ.get("GGML_CUDA_DISABLE_GRAPHS"),
                                "source": file_sha256(str(Path(__file__).with_name("kg4_scope.py")))}
    if os.environ.get("IRONMULE_GGML_PREFILL_BATCH") == "1":
        binding["grouped_prefill"] = 1
    if os.environ.get("IRONMULE_GGML_PREFILL_BATCH") == "long-444":
        binding["grouped_prefill"] = "cold-pairs-quads-444-v1"
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    return f"{platform.machine()}|{os.cpu_count()}|{model}|{digest}"


TUNE_CHILD = r"""
import json, sys
from ironmule.ggml_engine import autotune
print(json.dumps(autotune(sys.argv[1])))
"""

GATE_CANDIDATE = r"""
import json, sys
from ironmule.ggml_engine import GgmlEngine, SEQ_CTX
from ironmule.ggml_gate_checkpoint import fixture_signature
from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS, gate_lead
from ironmule.factual_gate import ggml_scores
from pathlib import Path
data = json.loads(sys.stdin.read())
fixture = {"seqs": 32, "seq_ctx": 1024, "max_rows": 32} if data["kg5_shape"] else {}
ids, bos, rows = data["ids"], data["bos"], []
serving_ctx = fixture.get("seq_ctx", SEQ_CTX)
# Gate2 scores 512 positions plus an optional BOS, in one native call.
fixture["seq_ctx"] = max(serving_ctx, CHUNK_TOKENS + int(bos is not None))
engine = GgmlEngine(sys.argv[1], speculate=False, shared=data["shared"], **fixture, **data["settings"])
for index in range(CHUNKS):
    chunk = ids[index * CHUNK_TOKENS:(index + 1) * CHUNK_TOKENS + 1]
    if len(chunk) < CHUNK_TOKENS + 1:
        break
    rows.append(engine.chunk_nll(chunk, gate_lead(chunk, bos)))
if engine.seq_ctx != serving_ctx:
    engine.close_context()
    cache_settings = {k: v for k, v in engine.settings.items() if k != "shared_prefix"}
    engine._open(engine.seqs, serving_ctx, **cache_settings)
    engine._warm_up()
snapshot = Path(data["snapshot"])
if snapshot.suffix.lower() == ".gguf":
    from ironmule.gguf_tokenizer import GgufTokenizer
    tokenizer = GgufTokenizer(snapshot)
else:
    from mlx_lm.tokenizer_utils import load
    tokenizer = load(snapshot)
if fixture_signature(tokenizer, data["kg5_shape"]) != data["facts_sha256"]:
    raise ValueError("the factual qualification fixture changed between children")
import os
require_dispatch = os.environ.get("IRONMULE_KG5") == "1" and data["settings"].get("flash_attn")
if require_dispatch:
    from ironmule.factual_gate import kg5_counter
    launches = kg5_counter(engine)
    before = launches()
facts = ggml_scores(engine, tokenizer, kg5=data["kg5_shape"])
dispatched = launches() - before if require_dispatch else None
repeat = ggml_scores(engine, tokenizer, kg5=data["kg5_shape"]) if data["reference"] else None
grouped = None
if data.get("check_prefill_batch"):
    if engine.prefill_batch_long:
        from ironmule.long_prefill_gate import check as check_long
        engine.prefill_batch = True
        grouped = check_long(engine, tokenizer)
    else:
        grouped = engine.check_prefill_batch(ids)
engine.close()
print(json.dumps({"nll": rows, "facts": facts, "repeat": repeat, "kg5_launches": dispatched, "prefill_batch": grouped}))
"""


GATE_IDS = r"""
import json, os, sys
from pathlib import Path
from ironmule.ggml_gate_checkpoint import fixture_signature
from ironmule.factual_gate import KG5_SHAPE
from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS, _text_ids, gate_bos
if sys.argv[1].lower().endswith(".gguf"):  # GG4: a GGUF served on its own tokenizes the gate's text itself
    from ironmule.gguf_tokenizer import GgufTokenizer
    tokenizer = GgufTokenizer(sys.argv[1])
else:
    from mlx_lm.tokenizer_utils import load
    tokenizer = load(Path(sys.argv[1]))
shape = os.environ.get("IRONMULE_KG5_SHAPE") == KG5_SHAPE
print(json.dumps({"ids": _text_ids(tokenizer)[:CHUNKS * CHUNK_TOKENS + 1], "bos": gate_bos(tokenizer),
                  "facts_sha256": fixture_signature(tokenizer, shape)}))
"""
DEFAULT_SETTINGS = {"flash_attn": False, "kv_unified": False, "kv_f32": False}
SETTING_KEYS = ("flash_attn", "kv_unified", "kv_f32", "kv_q8")


def _kg5_files() -> tuple[Path, ...]:
    """Identify the CUDA build and optional interposers. An ambiguous loader path disables cache reuse."""
    folder = Path(os.environ.get("LLAMA_CPP_LIB_PATH", ""))
    if not folder.is_absolute():
        return ()
    files = []
    for stem in ("libllama.so", "libggml-cuda.so"):
        linked = folder / stem
        if linked.is_file():
            files.append(linked.resolve())
        else:
            found = {p.resolve() for p in folder.glob(stem + ".*") if p.is_file()}
            if len(found) != 1:
                return ()
            files.extend(found)
    for entry in os.environ.get("LD_PRELOAD", "").replace(":", " ").split():
        path = Path(entry)
        if not path.is_absolute() or not path.is_file():
            return ()
        files.append(path.resolve())
    return tuple(files)


def gate_key(snapshot: str, gguf: str) -> str:
    """The gate's cache key: the GGUF, the snapshot (or the GGUF itself, GG4), the gate rule, the engine, the machine,
    and the generated kernel library the gate ran with (KG1: IRONMULE_KG_LIB), so that a new library is gated anew."""
    from .factual_gate import VERSION as FACT_VERSION
    from .numeric_choice import GATE_VERSION

    kg5 = None
    if os.environ.get("IRONMULE_KG5") == "1":
        kg5 = {"libraries": [file_sha256(str(p)) for p in _kg5_files()],
               "device": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_layers": gpu_layers(),
               "width": SEQS, "context": SEQ_CTX}
    shape_key = [file_sha256(gguf), str(Path(snapshot).resolve()), GATE_VERSION, FACT_VERSION,
                                     ENGINE_VERSION, OPTION_BOUND, machine_key(), SEQS, SEQ_CTX, kg5, os.environ.get("IRONMULE_KG5_SHAPE"),
                                     _kernel_digest()]
    if os.environ.get("IRONMULE_GGML_PREFILL_BATCH") == "1":
        shape_key.append("grouped-split-kv-prefill-v1")
    if os.environ.get("IRONMULE_GGML_PREFILL_BATCH") == "long-444":
        shape_key.append(["long-prefill.v1", file_sha256(str(Path(__file__).with_name("long_prefill_gate.py"))),
                          file_sha256(str(Path(__file__).with_name("long_prefill_study.py")))])
    return hashlib.sha256(json.dumps(shape_key, sort_keys=True).encode()).hexdigest()[:24]


def gate(snapshot: str, gguf: str, *, cache: Path | None = None, root: Path | None = None) -> dict[str, Any]:
    """The owner's bar (2026-11-10, 2026-11-11): answer quality at least llama.cpp's, its own options included. The
    engine's settings (all llama.cpp options) are timed (`autotune`), and the fastest one whose perplexity on the
    numeric gate's chunks, through the multi-position decode, is within `OPTION_BOUND` of llama.cpp's default setting
    on the same GGUF (the ratio's 95 % upper end) serves when it also preserves short-answer accuracy under QG1.
    Reference and candidate see the same questions. Cached by the GGUF, snapshot, gate rule, engine version and machine."""
    import subprocess

    from .factual_gate import KG5_SHAPE, VERSION as FACT_VERSION, _protocol, compare, reference_tallies
    from . import ggml_gate_checkpoint as checkpoint
    from . import ggml_gate_diagnostics as diag
    from .numeric_choice import CHUNKS, CHUNK_TOKENS, GATE_VERSION, ratio_interval

    root = root or Path(__file__).resolve().parents[1]
    def model_identity():
        try:
            s = os.stat(gguf)
            return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
        except OSError:
            return None

    original = model_identity()
    key = gate_key(snapshot, gguf)
    cache = cache or Path(os.path.expanduser("~/.cache/ironmule/ggml_gate"))
    stored = cache / f"{key}.json"
    # A fresh worker in the opposite benchmark order may reuse its own content-matched gate.
    experimental = os.environ.get("IRONMULE_KG5") == "1"
    batch_mode = os.environ.get("IRONMULE_GGML_PREFILL_BATCH")
    prefill_batch = (batch_mode == "1" or (batch_mode == "long-444" and cuda_kernel_device())) and gpu_layers() > 0
    shape = os.environ.get("IRONMULE_KG5_SHAPE") == KG5_SHAPE
    cacheable = _kernel_digest() is not None and (not experimental or bool(_kg5_files()))
    if cacheable and stored.exists():
        try:
            kept = json.loads(stored.read_text())
            optional_failed = any("error" in (kept.get(k) or {}) for k in ("streams", "shared"))
            if (model_identity() == original and not optional_failed
                    and not (gpu_layers() > 0 and kept.get("ran") and "streams" not in kept)):
                return kept
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")])),
           "HF_HUB_OFFLINE": "1"}
    started = time.perf_counter()

    def failure(diagnostic, *, tried=None, ran=False):
        result = {"ran": ran, "passes": False, "diagnostic": diagnostic,
                  "reason": f"GGUF qualification {diagnostic['stage']} returned {diagnostic['code']}"}
        if tried is not None:
            result["tried"] = tried
        return result

    def child(code: str, argument: str, payload: str | None = None, *, stock: bool = False):
        stage = "tokenize" if code == GATE_IDS else "tuning" if code == TUNE_CHILD else "reference" if stock else "candidate"
        if model_identity() != original:
            return None, diag.make("model_changed", "model_changed")
        child_env = ({**env, "IRONMULE_KG_LIB": "", "IRONMULE_KG5": "0", "IRONMULE_GGML_PREFILL_BATCH": "0",
                      "IRONMULE_W4": "0", "IRONMULE_GGML_PACK": "0", "IRONMULE_GGML_UBATCH": "0",
                      "IRONMULE_CUBLAS_MIN_ROWS": "0"} if stock else env)
        # GR3 (2026-11-22): the gate's children run with llama.cpp's one-row CUDA graphs. Multi-row graphs
        # (IRONMULE_CUDA_GRAPHS_ROWS) leave the tokens unchanged (A100, Qwen3-32B: the same greedy tokens with and
        # without) but ran the tuning child, which opens several contexts in one process, out of an A100's 80 GB.
        child_env = {**child_env, "IRONMULE_CUDA_GRAPHS_ROWS": "1"}
        if shape and code == GATE_CANDIDATE:
            # Admit all 32 near-1k prompts on the first step. Width and context are fixed in the measuring child.
            child_env = {**child_env, "IRONMULE_PILOT": "0", "IRONMULE_GGML_PROMPT_BUDGET": str(32 * 1024)}
        try:
            done = subprocess.run([sys.executable, "-c", code, argument], env=child_env, input=payload, capture_output=True,
                                  text=True, timeout=7200, check=False)
        except (subprocess.TimeoutExpired, OSError, UnicodeDecodeError) as exc:
            return None, diag.failed_child(stage, "", exception=exc)
        if model_identity() != original:
            return None, diag.make("model_changed", "model_changed")
        if done.returncode != 0:
            return None, diag.failed_child(stage, done.stderr, done.returncode)
        try:
            output = json.loads(done.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            return None, diag.make(stage, "invalid_result", status=0, exception=type(exc).__name__)
        if not isinstance(output, dict):
            return None, diag.make(stage, "invalid_result", status=0)
        return output, ""

    text, error = child(GATE_IDS, snapshot)
    if text is None:
        return failure(error)
    if (not isinstance(text.get("ids"), list) or not all(type(v) is int and v >= 0 for v in text["ids"])
            or len(text["ids"]) < 4 * CHUNK_TOKENS + 1 or "bos" not in text
            or text["bos"] is not None and type(text["bos"]) is not int
            or not isinstance(text.get("facts_sha256"), str) or len(text["facts_sha256"]) != 64):
        return failure(diag.make("tokenize", "invalid_result", status=0))
    folder = checkpoint.directory(cache, key=key, snapshot=snapshot, gguf=gguf, text=text,
                                  flags=[shape, experimental, SEQS, SEQ_CTX, THREADS, MAX_ROWS]) if cacheable else None
    completed = {"measured": 0, "reused": 0}
    def valid_tune(output):
        return (isinstance(output, dict) and isinstance(output.get("measured"), list)
                and isinstance(output.get("order"), list) and bool(output["order"])
                and all(settings in SETTINGS for settings in output["order"]))

    fact_keys = {case.key for case in _protocol(0, shape)}
    chunks = min(CHUNKS, (len(text["ids"]) - 1) // CHUNK_TOKENS)

    def measure(settings: dict[str, bool], *, reference: bool = False, shared: bool = False):
        check_batch = prefill_batch and not reference and not shared and not settings.get("kv_unified")
        def complete(output):
            if not isinstance(output, dict):
                return False
            nll = output.get("nll")
            batch = output.get("prefill_batch")
            return ((not check_batch or (isinstance(batch, dict) and type(batch.get("matches")) is bool
                                         and type(batch.get("calls")) is int and batch["calls"] >= 0))
                    and isinstance(nll, list) and len(nll) == chunks >= 4
                    and all(type(v) in (int, float) and 0 <= v < math.inf for v in nll)
                    and all(isinstance(output.get(k), dict) and output[k].keys() == fact_keys
                            and all(type(v) is bool for v in output[k].values())
                            for k in (("facts", "repeat") if reference else ("facts",))))

        payload = json.dumps({**text, "snapshot": str(snapshot), "settings": settings, "reference": reference,
                              "shared": shared, "kg5_shape": shape, "check_prefill_batch": check_batch})
        label = {"role": "reference" if reference else "candidate", "settings": settings, "shared": shared}
        output, error = checkpoint.stage(folder, label, lambda: child(GATE_CANDIDATE, gguf, payload, stock=reference),
                                         completed, complete)
        if output is not None and not complete(output):
            return None, diag.make("reference" if reference else "candidate", "invalid_result", status=0)
        return output, error

    reference, error = measure(DEFAULT_SETTINGS, reference=True)
    if reference is None:
        return failure(error)
    if model_identity() != original:
        return failure(diag.make("model_changed", "model_changed"))
    tallies = reference_tallies(reference["facts"], reference["repeat"], kg5=shape)
    skipped = None
    if any(min(row["reference_correct"], row["repeat_correct"]) == 0 for row in tallies.values()):
        # No B score can satisfy QG1 in this category. Only the identical, already executed
        # reference can serve. A generated or explicit experimental kernel still needs its own gate.
        skipped = diag.sanitize({**diag.make("acceptance", "skipped_uninformative_reference"),
                                 "reference_factual": tallies})
        if experimental or prefill_batch or os.environ.get("IRONMULE_KG_LIB"):
            return failure(skipped, ran=True, tried=[])
        tune = {"order": [DEFAULT_SETTINGS], "measured": []}
    else:
        tune, error = checkpoint.stage(folder, {"role": "tuning"}, lambda: child(TUNE_CHILD, gguf), completed, valid_tune)
        if tune is None:
            return failure(error)
        if not valid_tune(tune):
            return failure(diag.make("tuning", "invalid_result", status=0))

    measured: dict[tuple, dict[str, Any]] = {}

    def check(settings: dict[str, bool], *, shared: bool = False) -> dict[str, Any]:
        identity = (shared, tuple(sorted(settings.items())))
        if identity in measured:
            return measured[identity]
        output, error = measure(settings, shared=shared)
        if output is None:
            return {"settings": settings, "passes": False, "error": error["code"], "diagnostic": error}
        try:
            interval = ratio_interval(reference["nll"], output["nll"])
            factual = compare(reference["facts"], reference["repeat"], output["facts"], kg5=shape)
            if not all(math.isfinite(n) for n in interval["interval"]):
                raise ValueError("invalid interval")
        except (ValueError, ArithmeticError) as exc:
            return {"settings": settings, "passes": False, "error": "invalid_result",
                    "diagnostic": diag.make("candidate", "invalid_result", status=0, exception=type(exc).__name__)}
        launches = output.get("kg5_launches")
        hit = not (experimental and settings.get("flash_attn")) or isinstance(launches, int) and launches > 0
        grouped = output.get("prefill_batch")
        if prefill_batch and not shared and not settings.get("kv_unified"):
            hit = hit and grouped["matches"] and grouped["calls"] > 0
        record = {"settings": settings, **interval, "factual": factual, "kg5_launches": launches,
                  "passes": interval["interval"][1] <= OPTION_BOUND and factual["passes"] and hit}
        if grouped is not None:
            record["prefill_batch"] = grouped
        measured[identity] = record
        return record

    tried = []
    for settings in tune["order"]:  # the fastest option passing likelihood and factual correctness serves
        if settings == DEFAULT_SETTINGS and not os.environ.get("IRONMULE_KG_LIB") and not prefill_batch:
            tried.append({"settings": settings, "ratio": 1.0, "interval": [1.0, 1.0], "chunks": len(reference["nll"]),
                          "passes": True, "basis": "identical_reference"})
            break
        row = check(settings)
        if "error" in row:
            details = {**row["diagnostic"], "candidates": [diag.candidate(t, OPTION_BOUND, experimental) for t in [*tried, row]]}
            return failure(diag.sanitize(details), tried=tried)
        tried.append(row)
        if row["passes"]:
            break
    chosen = tried[-1]
    if not chosen["passes"]:
        return failure(diag.make("acceptance", "quality_rejected",
                                 candidates=[diag.candidate(t, OPTION_BOUND, experimental) for t in tried]),
                       ran=True, tried=tried)

    def probe(settings, role):
        if skipped:
            return {"settings": settings, "passes": False, "status": skipped["code"]}
        row = check(settings, shared=role == "shared")
        if row["passes"]:
            return row
        diagnostic = row.get("diagnostic") or diag.make("acceptance", "quality_rejected")
        return {**row, "diagnostic": diag.sanitize({**diagnostic, "candidates": [
            diag.candidate(row, OPTION_BOUND, experimental, role)]})}

    shared = None
    if fa1_version() >= 3 or gpu_layers() > 0:
        shared = probe({**chosen["settings"], "flash_attn": True, "kv_unified": True}, "shared")
    streams = None
    if gpu_layers() > 0:
        streams = probe({**chosen["settings"], "flash_attn": True, "kv_unified": False}, "streams")
    if model_identity() != original:
        return failure(diag.make("model_changed", "model_changed"))
    result = {"ran": True, "shared": shared, "streams": streams,
              **{k: chosen[k] for k in ("ratio", "interval", "chunks", "passes")},
              "settings": chosen["settings"], "tried": tried, "tuning": tune["measured"], "machine": machine_key(),
              "reference": "llama.cpp's default setting on the same GGUF", "bound": OPTION_BOUND,
              "seconds": round(time.perf_counter() - started, 1), "gate_version": GATE_VERSION,
              "factual_version": FACT_VERSION, "key": key, "checkpoint_stages": completed}
    if skipped:
        result.update(diagnostic=skipped, reason=skipped["code"])
    if not cacheable or any("error" in (row or {}) for row in (shared, streams)):
        return result
    try:
        cache.mkdir(parents=True, exist_ok=True)
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", dir=cache, delete=False) as handle:
            try:
                json.dump(result, handle, allow_nan=False)
                handle.close()
                os.replace(handle.name, stored)
            finally:
                Path(handle.name).unlink(missing_ok=True)
    except OSError:
        pass
    return result


__all__ = ["ENGINE_VERSION", "OPTION_BOUND", "GgmlEngine", "Job", "autotune", "available", "bindings", "choose_sequences",
           "decide_width", "file_sha256", "fitting_widths", "gate", "machine_key", "measure_widths", "memory_bytes"]
