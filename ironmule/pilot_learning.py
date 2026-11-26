"""A local residual thread head for the CPU Pilot, qualified on real, idle decode steps.

Only plain decode can use these additional weights. Each weight is a thread-count
offset for one context bin, batch width, base decision and foreign-load regime.
Search compares two neighbouring actions. Qualification uses fresh filler prompts,
balanced A/B/A and B/A/A order, identical tokens and gains beyond the A/A spread.
Persisted weights are proposals until requalified in the current process.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import statistics
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from zipfile import BadZipFile

from .ggml_engine import ENGINE_VERSION, machine_key
from .hardware_policy import SCHEMA, HardwarePolicy
from .pilot import OBS, THREAD_FRACTIONS, predict

VERSION = 2
BIN = 64
MIN_STEPS = 64
MAX_CONTEXT = 512
DECODE = 8
REPETITIONS = 4
MAX_ROWS = 4


class ThreadPolicy:
    """Apply the qualified per-regime thread offsets; an admission holdout replays a frozen copy."""

    bin_size = BIN
    max_rows = MAX_ROWS

    def __init__(self, pilot, weights):
        self.pilot = pilot
        self.weights = dict(weights)
        self.visits: Counter[str] = Counter()
        self.last_regime = None

    def choose(self, context: float, foreign: float, base: int, rows: int = 1) -> int:
        self.last_regime = None
        if not (BIN <= context < MAX_CONTEXT and math.isfinite(foreign)):
            return base
        if not 1 <= rows <= MAX_ROWS:
            return base
        load = min(self.pilot.cores, int(max(0.0, foreign) + 0.5))
        key = f"{int(context) // BIN}:{base}:{rows}:{load}"
        self.last_regime = key
        self.visits[key] += 1
        return base + self.weights.get(key, 0)

class ThreadLearning(ThreadPolicy):

    def __init__(self, engine, root: Path, gguf_digest: str, base_digest: str):
        self.engine = engine
        super().__init__(engine.pilot, {})
        machine = machine_key()
        binding = [VERSION, ENGINE_VERSION, gguf_digest, base_digest, machine, self.pilot.cores,
                   engine.settings, engine.seqs, engine.seq_ctx, engine.shared, engine.cap]
        self.binding = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        self.path = root / "pilot" / (self.binding + ".json")
        hardware = hashlib.sha256(f"{machine}|{self.pilot.cores}".encode()).hexdigest()[:20]
        costs = getattr(self.pilot, "costs", {})
        profile = {"machine": machine, "cores": self.pilot.cores, "gguf_sha256": gguf_digest,
                   "pilot_sha256": base_digest, "engine_version": ENGINE_VERSION, "sequences": engine.seqs,
                   "context_limit": engine.seq_ctx, "shared_pool": engine.shared, "capacity": engine.cap,
                   "settings": engine.settings, "step_costs": costs}
        self.decode_geometry = {"backend": "ggml-cpu", "hardware_runtime": machine, "model_sha256": gguf_digest,
                                "sequences": engine.seqs, "sequence_context": engine.seq_ctx, "capacity": engine.cap,
                                "shared_cache": engine.shared, "settings": dict(engine.settings)}
        self.policy = HardwarePolicy(self.path.parent / f"thread-model-v{SCHEMA}.sqlite3", self.binding,
                                     hardware, self.pilot.cores, costs, profile)
        self.replay_rows, self.replay_load = 1, 0
        self.proposals = {}
        legacy = [1, *binding[1:]]
        legacy_key = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
        source = self.path if self.path.exists() else self.path.with_name(legacy_key + ".json")
        try:
            saved = json.loads(source.read_text())
            if saved.get("binding") == source.stem and isinstance(saved.get("weights"), dict):
                for key, offset in saved["weights"].items():
                    if type(offset) is int and 0 < abs(offset) < self.pilot.cores:
                        self.proposals[key if source == self.path else key + ":1:0"] = offset
        except (OSError, ValueError, AttributeError):
            pass
        self.thread_candidate = None
        self.margins = {}
        self.last_thread_choice = None
        self.candidate_status = None
        path = os.environ.get("IRONMULE_PILOT_THREAD_CANDIDATE")
        if path:
            self._load_candidate(Path(path))
        self.generation = 0
        self.last = None

    def _load_candidate(self, path: Path) -> None:
        import numpy as np

        self.thread_candidate = None
        try:
            data = path.read_bytes()
            with np.load(io.BytesIO(data), allow_pickle=False) as checkpoint:
                if int(checkpoint["version"]) != 1 or str(checkpoint["binding"]) != self.binding:
                    raise ValueError("foreign binding or version")
                weights = [checkpoint[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")]
            shapes = [(128, OBS), (128,), (128, 128), (128,), (len(THREAD_FRACTIONS), 128), (len(THREAD_FRACTIONS),)]
            if any(w.shape != shape or w.dtype != np.float32 or not np.isfinite(w).all()
                   for w, shape in zip(weights, shapes)):
                raise ValueError("invalid network")
            self.thread_candidate = weights
            self.candidate_status = {"state": "available for local trial", "sha256": hashlib.sha256(data).hexdigest(),
                                     "trials": 0, "qualified_decisions": 0,
                                     "reference_decisions": 0, "inference_us": 0.0}
        except (OSError, KeyError, ValueError, EOFError, BadZipFile) as exc:
            self.candidate_status = {"state": "unavailable", "reason": type(exc).__name__}

    def choose(self, context: float, foreign: float, base: int, rows: int = 1) -> int:
        if self.thread_candidate is None:
            return super().choose(context, foreign, base, rows)
        started = time.perf_counter_ns()
        allowed = super().choose(context, foreign, base, rows)
        state = self.candidate_status
        regime = self.last_regime
        receipt = self.margins.get(regime) if allowed != base else None
        cost_us = None
        selected = base
        reason = "no local qualification" if regime else "outside measured workload"
        if allowed != base and receipt and receipt["candidate_threads"] == allowed:
            # The lower measured gain deducts the entire control-to-control difference in every window.
            # Only run the network for a profile with positive headroom after that allowance.
            headroom_us = receipt["min_gain_after_noise_us_per_step"]
            reason = "gain within observed timing noise"
            if headroom_us > 0:
                scores = predict(self.thread_candidate, self.pilot.features(rows, 0, 0, context))
                picked = self.pilot.threads_for(int(scores.argmax())) if all(map(math.isfinite, scores)) else None
                cost_us = (time.perf_counter_ns() - started) / 1000
                state["inference_us"] += cost_us
                reason = "model chose a different profile"
                if picked == allowed:
                    reason = "decision cost exceeds measured headroom"
                    if cost_us < headroom_us:
                        selected, reason = allowed, "locally qualified"
        state["qualified_decisions" if selected != base else "reference_decisions"] += 1
        self.last_thread_choice = (regime, base, selected, rows, reason, cost_us, receipt)
        return selected

    def thread_decision(self):
        if self.last_thread_choice is None:
            return None
        regime, base, selected, rows, reason, cost_us, receipt = self.last_thread_choice
        return {"choice": "qualified" if selected != base else "reference", "reason": reason,
                "profile": {**self.decode_geometry, "binding": self.binding, "operation": "plain_decode", "threads": selected,
                            "sequence_rows": rows, "prompt_rows": 0, "draft_rows": 0, "padding_rows": 0,
                            "pending_requests": 0},
                "regime": regime, "reference_threads": base, "selection_us": cost_us,
                "eligible_threads": [base, receipt["candidate_threads"]] if receipt else [base],
                "qualification": dict(receipt) if receipt else None, "uncertainty": "local paired windows" if receipt else "unknown"}

    def _trial_actions(self, key, context, base, rows, actions):
        ranked = self.policy.rank(key, actions)
        if self.thread_candidate is None:
            return ranked[:2]
        # One learned proposal competes with the strongest existing search action.
        # In plain decode all remaining scheduling knobs are already fixed for this call.
        scores = predict(self.thread_candidate, self.pilot.features(rows, 0, 0, context))
        threads = self.pilot.threads_for(int(scores.argmax()))
        self.candidate_status.update(trials=self.candidate_status["trials"] + 1, regime=key, threads=threads)
        if threads == base:
            return ranked[:2]
        return list(dict.fromkeys([threads] + ranked))[:2]

    def status(self):
        return {"version": VERSION, "binding": self.binding, "generation": self.generation,
                "weights": dict(self.weights), "model": self.policy.status(), "last": self.last,
                **({"scratch_candidate": self.candidate_status, "thread_decision": self.thread_decision()}
                   if self.candidate_status else {})}

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
        bin_index, base, self.replay_rows, self.replay_load = map(int, key.split(":"))
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
            cores = self.pilot.cores
            options = [base - 1, base + 1, max(1, base // 2), max(1, cores - self.replay_load), cores]
            actions = list(dict.fromkeys(n for n in options if 1 <= n <= cores and n != base))
            for candidate in self._trial_actions(key, context + 16, base, self.replay_rows, actions):
                run = trial(context + 16, base, candidate, 0, interrupted)
                if run is None:
                    return
                matched = run[1] == reference[1]
                faster = matched and run[0] < reference[0]
                self.policy.record(key, candidate, "search", run[0] / reference[0], matched, faster)
                if matched and run[0] < best:
                    best, offset = run[0], candidate - base
        if not offset:
            self.last = {"key": key, "result": "no faster candidate"}
            return
        candidate = base + offset
        ratios, noise, gain_after_noise = [], [], []
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
            midpoint = (control[0] + repeat[0]) / 2
            ratios.append(changed[0] / midpoint)
            aa_difference = abs(control[0] - repeat[0])
            noise.append(aa_difference / midpoint)
            gain_after_noise.append((midpoint - changed[0] - aa_difference) * 1e6 / DECODE)
            if control[1] != changed[1] or control[1] != repeat[1]:
                passed = False
                break
        matched = passed
        passed = passed and max(ratios, default=1.0) < 0.97 and 1 - statistics.median(ratios) > max(noise)
        self.policy.record(key, candidate, "qualification", statistics.median(ratios), matched, passed, max(noise))
        previous = self.weights.get(key)
        if passed:
            self.weights[key] = offset
            self.margins[key] = {"windows": len(ratios), "candidate_threads": candidate, "max_ratio": max(ratios),
                                 "max_aa_fraction": max(noise),
                                 "min_gain_after_noise_us_per_step": min(gain_after_noise)}
        else:
            self.weights.pop(key, None)
            self.margins.pop(key, None)
        self.last = {"key": key, "candidate_threads": candidate, "ratios": ratios, "aa_spread": noise,
                     "result": "qualified" if passed else "reverted" if previous else "refused"}
        if self.weights.get(key) != previous:
            self.generation += 1
            self._save()

    def _probe(self, context: int, base: int, threads: int, seed: int, interrupted):
        """Replay a short greedy decode, using a free sequence without evicting a cached prefix."""
        e = self.engine

        def same_load():
            return min(self.pilot.cores, int(self.pilot.load.update() + 0.5)) == self.replay_load

        if interrupted() or not same_load() or context + DECODE >= e.seq_ctx:
            return None
        if e.pool is not None:
            if e.pool.cells - e.pool.used < self.replay_rows * (context + DECODE):
                return None
            slots = e.pool.free_ids
        else:
            slots = [s for s in e._free if s not in e._kept]
        if len(slots) < self.replay_rows:
            return None
        sequences = slots[:self.replay_rows]
        original_threads = e.threads
        tokens = []
        try:
            e.set_threads(base)
            first = []
            for row, seq in enumerate(sequences):
                ids = e.tokenize(f"Local timing fixture {seed}.{row}. Continue counting from {seed + 31 + row}: ")
                if not ids:
                    return None
                prompt = (ids * ((context + len(ids) - 1) // len(ids)))[:context]
                for pos in range(0, context, 32):
                    if interrupted():
                        return None
                    rows = [(token, pos + j, seq, pos + j + 1 == context)
                            for j, token in enumerate(prompt[pos:pos + 32])]
                    picks = e._decode(rows, evict=False)
                first.append(picks[-1])
            picks = first
            started = time.perf_counter()
            e.set_threads(threads)
            for pos in range(context, context + DECODE):
                if interrupted():
                    return None
                picks = e._decode([(token, pos, seq, True) for token, seq in zip(picks, sequences)], evict=False)
                tokens.extend(picks)
            e.set_threads(base)  # charge both switches in the candidate's measured time
            seconds = time.perf_counter() - started
            if not math.isfinite(seconds) or seconds <= 0 or not same_load():
                return None
            return seconds, tuple(tokens)
        finally:
            try:
                released = [e.L.llama_memory_seq_rm(e.memory, seq, -1, -1) for seq in sequences]
                if not all(released):
                    raise RuntimeError("could not release pilot replay sequences")
            finally:
                e.set_threads(original_threads)
