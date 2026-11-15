"""GG0: llama.cpp's ggml kernels in process, against `llama-server`, before an engine is built (BACKLOG GG, GG0).

(a) parity of one request decoded one token per call; (b) the cost of a verify window of `r` positions of one
sequence, and of 8 sequences one position each; (c) whether prompt-lookup speculation verified as windows changes
any token; (d) its speed; (e) 8 requests batched one position each per call. The library and its ctypes bindings come
from an unpacked `llama-cpp-python` source (`--bindings`, with the built library in `--library`), not an install.
`--analyse REPORT` applies the rule fixed in the entry before the data.

    python tools/gg0_probe.py --bindings SRC --library SRC/vendor/llama.cpp/build/bin --server PATH/llama-server \\
        --gguf MODEL.gguf --dolly DIR --out gg0.json
    python -I tools/gg0_probe.py --analyse gg0.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROMPTS, FIRST, LIMIT, LOAD_LIMIT = 16, 14_600, 128, 64
KS = (2, 4, 7)
SEQS, SEQ_CTX, THREADS = 8, 2048, 4
PARITY, GOAL, WINDOW_ROWS, WINDOW_COST = 0.95, 1.2, 5, 1.5


class Ggml:
    """The few llama.cpp calls GG0 needs, through `llama-cpp-python`'s own ctypes bindings."""

    def __init__(self, gguf: str):
        import numpy as np

        import llama_cpp as L

        self.L, self.np = L, np
        L.llama_backend_init()
        self.model = L.llama_model_load_from_file(gguf.encode(), L.llama_model_default_params())
        params = L.llama_context_default_params()
        params.n_ctx, params.n_batch, params.n_ubatch = SEQS * SEQ_CTX, SEQ_CTX, 512
        params.n_seq_max, params.n_threads, params.n_threads_batch = SEQS, THREADS, THREADS
        self.ctx = L.llama_init_from_model(self.model, params)
        self.vocab = L.llama_model_get_vocab(self.model)
        self.n_vocab = L.llama_vocab_n_tokens(self.vocab)
        self.memory = L.llama_get_memory(self.ctx)
        self.batch = L.llama_batch_init(SEQ_CTX, 0, SEQS)

    def tokenize(self, text: str) -> list[int]:
        raw = text.encode()
        buf = (self.L.llama_token * 8192)()
        n = self.L.llama_tokenize(self.vocab, raw, len(raw), buf, 8192, False, True)
        return list(buf[:n])

    def eog(self, token: int) -> bool:
        return bool(self.L.llama_vocab_is_eog(self.vocab, token))

    def clear(self) -> None:
        self.L.llama_memory_clear(self.memory, True)

    def remove(self, seq: int, start: int) -> None:
        self.L.llama_memory_seq_rm(self.memory, seq, start, -1)

    def copy(self, src: int, dst: int) -> None:
        self.L.llama_memory_seq_cp(self.memory, src, dst, -1, -1)

    def decode(self, rows: list[tuple[int, int, int, bool]]) -> list[int]:
        """Rows of (token, position, sequence, logits?); the argmax of each row that asked for logits."""
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

    def prefill(self, ids: list[int], seq: int = 0) -> int:
        first = None
        for start in range(0, len(ids), SEQ_CTX):
            chunk = ids[start:start + SEQ_CTX]
            last = start + len(chunk) == len(ids)
            picks = self.decode([(t, start + j, seq, last and j == len(chunk) - 1) for j, t in enumerate(chunk)])
            first = picks[-1] if picks else first
        return first


def sequential(g: Ggml, ids: list[int], limit: int) -> dict[str, Any]:
    g.clear()
    offered = time.perf_counter()
    out = [g.prefill(ids)]
    first = time.perf_counter() - offered
    while len(out) < limit and not g.eog(out[-1]):
        out += g.decode([(out[-1], len(ids) + len(out) - 1, 0, True)])
    return {"tokens": out, "first_s": first, "end_s": time.perf_counter() - offered}


def speculative(g: Ggml, ids: list[int], limit: int, k: int) -> dict[str, Any]:
    from ironmule.speculation import accept, draft

    g.clear()
    offered = time.perf_counter()
    out = [g.prefill(ids)]
    first = time.perf_counter() - offered
    drafted = taken = 0
    while len(out) < limit and not g.eog(out[-1]):
        pos = len(ids) + len(out) - 1
        drafts, _ = draft(ids + out, min(k, limit - len(out) - 1))
        rows = [out[-1]] + drafts
        picks = g.decode([(t, pos + j, 0, True) for j, t in enumerate(rows)])
        new = accept(drafts, picks)
        drafted, taken = drafted + len(drafts), taken + len(new) - 1
        g.remove(0, pos + len(new))  # the rows after the last accepted one
        for token in new:
            out.append(token)
            if g.eog(token) or len(out) >= limit:
                break
    return {"tokens": out, "first_s": first, "end_s": time.perf_counter() - offered, "drafted": drafted,
            "accepted": taken}


def verify_cost(g: Ggml, ids: list[int], repeats: int = 20) -> dict[str, float]:
    g.clear()
    g.prefill(ids)
    pos = len(ids)
    out = {}
    for r in range(1, 9):
        times = []
        for _ in range(repeats):
            started = time.perf_counter()
            g.decode([(ids[j % len(ids)], pos + j, 0, True) for j in range(r)])
            times.append(time.perf_counter() - started)
            g.remove(0, pos)
        out[str(r)] = statistics.median(times)
    for seq in range(1, SEQS):
        g.copy(0, seq)
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        g.decode([(ids[0], pos, seq, True) for seq in range(SEQS)])
        times.append(time.perf_counter() - started)
        for seq in range(SEQS):
            g.remove(seq, pos)
    out["8_sequences"] = statistics.median(times)
    return out


def load(g: Ggml, prompts: list[list[int]], limit: int) -> dict[str, Any]:
    g.clear()
    offered = time.perf_counter()
    outs, firsts = [], []
    for seq, ids in enumerate(prompts):
        outs.append([g.prefill(ids, seq)])
        firsts.append(time.perf_counter() - offered)
    while True:
        active = [s for s in range(len(prompts)) if len(outs[s]) < limit and not g.eog(outs[s][-1])]
        if not active:
            break
        picks = g.decode([(outs[s][-1], len(prompts[s]) + len(outs[s]) - 1, s, True) for s in active])
        for s, token in zip(active, picks):
            outs[s].append(token)
    span = time.perf_counter() - offered
    return {"tokens": sum(len(o) for o in outs), "span_s": span, "mean_first_s": statistics.fmean(firsts)}


def _rate(row: dict[str, Any]) -> float | None:
    n = row["tokens"] if isinstance(row["tokens"], int) else len(row["tokens"])
    span = row["end_s"] - row["first_s"]
    return (n - 1) / span if n > 1 and span > 0 else None


def analyse(report: dict[str, Any]) -> dict[str, Any]:
    """The rule fixed in BACKLOG GG0 before the data."""
    server = [(_rate(a) + _rate(b)) / 2 for a, b in zip(report["server_quiet"][0], report["server_quiet"][1])
              if _rate(a) and _rate(b)]
    seq = [_rate(r) for r in report["probe"]["sequential"]]
    parity = statistics.fmean(s / v for s, v in zip(seq, server))
    spec = {}
    for k in KS:
        rows = report["probe"][f"k{k}"]
        spec[f"k{k}"] = {"over_server": statistics.fmean(_rate(r) / v for r, v in zip(rows, server)),
                         "over_sequential": statistics.fmean(_rate(r) / s for r, s in zip(rows, seq)),
                         "differing_prompts": sum(r["tokens"] != s["tokens"] for r, s in
                                                  zip(rows, report["probe"]["sequential"])),
                         "acceptance": (sum(r["accepted"] for r in rows) / max(1, sum(r["drafted"] for r in rows)))}
    cost = report["verify_cost"]
    window_room = any(cost[str(r)] <= WINDOW_COST * cost["1"] for r in range(WINDOW_ROWS, 9))
    best = max(spec, key=lambda key: spec[key]["over_server"])
    exact_best = spec[best]["differing_prompts"] == 0
    load_server = statistics.fmean(c["tokens"] / c["span_s"] for c in report["server_load"])
    load_probe = report["probe"]["load"]["tokens"] / report["probe"]["load"]["span_s"]
    any_differ = any(v["differing_prompts"] for v in spec.values())
    go = parity >= PARITY and ((spec[best]["over_server"] >= GOAL and exact_best) or window_room)
    decision = ("go: build GG1" if go and not any_differ else
                "owner: speculation on ggml changes tokens" if go and any_differ else "stop: Tier 0")
    return {"parity": parity, "speculation": spec, "best_k": best, "verify_cost_ratio":
            {r: cost[r] / cost["1"] for r in cost}, "window_room": window_room,
            "load_over_server": load_probe / load_server, "decision": decision}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--bindings", "--library", "--server", "--gguf"):
        parser.add_argument(name)
    parser.add_argument("--dolly", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--analyse", type=Path)
    parser.add_argument("--smoke", action="store_true", help="2 prompts the protocol never uses (index 14,700), 16 tokens")
    args = parser.parse_args(argv)
    if args.analyse:
        print(json.dumps(analyse(json.loads(args.analyse.read_text())), indent=1))
        return 0
    os.environ["LLAMA_CPP_LIB_PATH"] = args.library
    sys.path[:0] = [args.bindings, str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
    sys.modules.setdefault("diskcache", types.SimpleNamespace(Cache=object))  # the bindings import it, GG0 never uses it
    import cpu6_measure
    import lc1_measure
    import rl2_headroom
    import sp2_measure
    from transformers import AutoTokenizer

    texts, digest = rl2_headroom.dolly_prompts(args.dolly)
    first, count, limit, load_limit = (FIRST + 100, 2, 16, 16) if args.smoke else (FIRST, PROMPTS, LIMIT, LOAD_LIMIT)
    quiet = texts[first:first + count]
    tokenizer = AutoTokenizer.from_pretrained(sp2_measure._spec().snapshot_path)

    def render(text: str) -> str:
        return tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                             tokenize=False)

    report: dict[str, Any] = {"protocol": "gg0" + ("-smoke" if args.smoke else ""), "dolly_sha256": digest, "probe": {}, "server_quiet": [],
                              "server_load": []}

    def save() -> None:
        if args.out:
            args.out.write_text(json.dumps(report))

    def server_load() -> None:
        cell = lc1_measure.run_llama(args.server, args.gguf, cpu6_measure.prompts(), True, load_limit)
        rows = cell["requests"]
        report["server_load"].append({"tokens": sum(r["tokens"] for r in rows), "span_s": max(r["end_s"] for r in rows),
                                      "mean_first_s": statistics.fmean(r["first_s"] for r in rows)})

    # the order fixed in the entry: server, probe, server (quiet and load alike)
    report["server_quiet"].append(lc1_measure.run_llama(args.server, args.gguf, quiet, False, limit)["requests"])
    server_load()
    save()
    g = Ggml(args.gguf)
    ids = [g.tokenize(render(t)) for t in quiet]
    report["probe"]["sequential"] = [sequential(g, i, limit) for i in ids]
    for k in KS:
        report["probe"][f"k{k}"] = [speculative(g, i, limit, k) for i in ids]
    report["verify_cost"] = verify_cost(g, ids[0])
    load_ids = [g.tokenize(render(t)) for t in cpu6_measure.prompts()]
    report["probe"]["load"] = load(g, load_ids, load_limit)
    save()
    del g
    report["server_quiet"].append(lc1_measure.run_llama(args.server, args.gguf, quiet, False, limit)["requests"])
    server_load()
    report["analysis"] = analyse(report)
    save()
    print(json.dumps(report["analysis"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
