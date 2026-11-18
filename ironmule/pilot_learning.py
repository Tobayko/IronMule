"""A local residual thread head for the CPU Pilot, qualified on real, idle decode steps.

Only plain, single-request decode can use these additional weights. A weight is a
thread-count offset for one context-length bin and base decision at low foreign load.
Search compares two neighbouring actions. Qualification uses fresh filler prompts,
balanced A/B/A and B/A/A order, identical tokens and gains beyond the A/A spread.
Persisted weights are proposals until requalified in the current process.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from .ggml_engine import ENGINE_VERSION, machine_key

VERSION = 1
BIN = 64
MIN_STEPS = 64
MAX_CONTEXT = 512
DECODE = 8
REPETITIONS = 4


class ThreadLearning:
    def __init__(self, engine, root: Path, gguf_digest: str, base_digest: str):
        self.engine = engine
        self.pilot = engine.pilot
        binding = [VERSION, ENGINE_VERSION, gguf_digest, base_digest, machine_key(), self.pilot.cores,
                   engine.settings, engine.seqs, engine.seq_ctx, engine.shared, engine.cap]
        self.binding = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        self.path = root / "pilot" / (self.binding + ".json")
        self.weights: dict[str, int] = {}
        self.proposals = {}
        try:
            saved = json.loads(self.path.read_text())
            if saved.get("binding") == self.binding and isinstance(saved.get("weights"), dict):
                self.proposals = {k: v for k, v in saved["weights"].items() if type(v) is int and abs(v) == 1}
        except (OSError, ValueError, AttributeError):
            pass
        self.visits: Counter[str] = Counter()
        self.generation = 0
        self.last = None

    def choose(self, context: float, foreign: float, base: int) -> int:
        if not (BIN <= context < MAX_CONTEXT and math.isfinite(foreign)):
            return base
        # Only the low-contention regime can be reproduced in an idle spell.
        if foreign >= 0.5:
            return base
        key = f"{int(context) // BIN}:{base}"
        self.visits[key] += 1
        return base + self.weights.get(key, 0)

    def status(self):
        return {"version": VERSION, "binding": self.binding, "generation": self.generation,
                "weights": dict(self.weights), "last": self.last}

    def _save(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # The checkpoint is just a search hint. Different workers can checkpoint the
        # same binding; a reader still qualifies every hint on its own live engine.
        with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as output:
            try:
                json.dump(self.status(), output)
                output.close()
                os.replace(output.name, self.path)
            finally:
                Path(output.name).unlink(missing_ok=True)

    def idle(self, interrupted: Callable[[], bool]) -> None:
        if not self.visits or interrupted():
            return
        key, visits = self.visits.most_common(1)[0]
        if visits < MIN_STEPS:
            return
        self.visits[key] = 0
        bin_index, base = map(int, key.split(":"))
        context = bin_index * BIN
        trial = self._probe
        offset = self.weights.get(key) or self.proposals.pop(key, 0)
        if not 1 <= base + offset <= self.pilot.cores:
            offset = 0
        if not offset:
            # Search uses a distinct fixture from every qualification window.
            reference = trial(context + 16, base, base, 0, interrupted)
            if reference is None:
                return
            best = reference[0]
            for candidate in (base - 1, base + 1):
                if 1 <= candidate <= self.pilot.cores:
                    run = trial(context + 16, base, candidate, 0, interrupted)
                    if run is None:
                        return
                    if run[1] == reference[1] and run[0] < best:
                        best, offset = run[0], candidate - base
        if not offset:
            self.last = {"key": key, "result": "no faster candidate"}
            return
        candidate = base + offset
        ratios, noise = [], []
        passed = True
        for index in range(REPETITIONS):
            arms = (base, candidate, base) if index % 2 == 0 else (candidate, base, base)
            observations = []
            for threads in arms:
                observed = trial(context + 4 + 13 * index, base, threads, index + 1, interrupted)
                if observed is None:
                    return
                observations.append(observed)
            control, changed, repeat = observations if index % 2 == 0 else (
                observations[1], observations[0], observations[2])
            if control[1] != changed[1] or control[1] != repeat[1]:
                passed = False
                break
            midpoint = (control[0] + repeat[0]) / 2
            ratios.append(changed[0] / midpoint)
            noise.append(abs(control[0] - repeat[0]) / midpoint)
        passed = passed and max(ratios, default=1.0) < 0.97 and 1 - statistics.median(ratios) > max(noise)
        previous = self.weights.get(key)
        if passed:
            self.weights[key] = offset
        else:
            self.weights.pop(key, None)
        self.last = {"key": key, "candidate_threads": candidate, "ratios": ratios, "aa_spread": noise,
                     "result": "qualified" if passed else "reverted" if previous else "refused"}
        if self.weights.get(key) != previous:
            self.generation += 1
            self._save()

    def _probe(self, context: int, base: int, threads: int, seed: int, interrupted):
        """Replay a short greedy decode, using a free sequence without evicting a cached prefix."""
        e = self.engine
        if interrupted() or self.pilot.load.update() >= 0.5 or context + DECODE >= e.seq_ctx:
            return None
        if e.pool is not None:
            if e.pool.cells - e.pool.used < context + DECODE:
                return None
            slots = e.pool.free_ids
        else:
            slots = [s for s in e._free if s not in e._kept]
        if not slots:
            return None
        seq = slots[0]
        ids = e.tokenize(f"Local timing fixture {seed}. Continue counting from {seed + 31}: ")
        if not ids:
            return None
        prompt = (ids * ((context + len(ids) - 1) // len(ids)))[:context]
        original_threads = e.threads
        tokens = []
        try:
            e.set_threads(base)
            for pos in range(0, context, 32):
                if interrupted():
                    return None
                rows = [(token, pos + j, seq, pos + j + 1 == context) for j, token in enumerate(prompt[pos:pos + 32])]
                picks = e._decode(rows, evict=False)
            token = picks[-1]
            started = time.perf_counter()
            e.set_threads(threads)
            for pos in range(context, context + DECODE):
                if interrupted():
                    return None
                token = e._decode([(token, pos, seq, True)], evict=False)[0]
                tokens.append(token)
            e.set_threads(base)  # charge both switches in the candidate's measured time
            seconds = time.perf_counter() - started
            if not math.isfinite(seconds) or seconds <= 0 or self.pilot.load.update() >= 0.5:
                return None
            return seconds, tuple(tokens)
        finally:
            try:
                if not e.L.llama_memory_seq_rm(e.memory, seq, -1, -1):
                    raise RuntimeError("could not release pilot replay sequence")
            finally:
                e.set_threads(original_threads)
