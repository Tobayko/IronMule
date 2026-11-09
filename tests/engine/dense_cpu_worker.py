"""STAB4: the real product worker (`ironmule_product.worker.main`) serving KB2's Gemma stacked plan on MLX's CPU.

Started through `MLXWorkerClient.start()` in place of `worker.py` (tests/engine/test_dense_slot_flow.py), so the
service, the client, the JSON-lines transport, the worker's command loop, its admission (`_run_engine_batch`),
the bridge, the Runtime and the plan's slot pool (`DenseBackend`) are the product's own. Replaced, and only
these, because they need CUDA or a downloaded checkpoint:

* the model: a tiny random 4-bit Gemma 3 (`tests/engine/test_dense_gemma.py`'s shape, weights unscaled so that
  greedy answers vary token by token);
* the plan's kernels: `GemmaDensePlan(attention=reference_attention, native=False)`, the CPU reference the KB2
  tests check against mlx-lm, built here in place of `install_dense`'s CUDA install and probe;
* the snapshot identity: a stand-in, since there is no snapshot. Nothing is written to any evidence store, and
  the plan's record says `cpu_reference`, so no CUDA path can be qualified from this worker.

Environment: `DENSE_CPU_LEDGER` names a JSON-lines file that gets every slot taken and freed;
`DENSE_CPU_SLOTS` overrides the pool's slot count (a misconfiguration only a test makes: the client then
believes in another width); `DENSE_CPU_CAPACITY` is a slot's context (default 96 tokens);
`DENSE_CPU_FAULTS` is JSON: {"prefill_fail": "<text>"} fails the prefill of the request whose prompt is that
text; {"step_fail": "<text>"} fails every stacked step that holds that request's slot, and
{"step_fail_once": "<text>"} only the first such step; {"pause": [[step, "<file>"], ...]} holds the worker
before its stacked step `step` (1-based, counted over the process's life) until `<file>` exists, with a
`paused` row in the ledger first, so a test can act at a known point of a run; {"pause_rounds": [[round,
"<file>"], ...]} holds the worker at the start of its executor round `round` (1-based, over the process's life),
after finished requests have returned their slots and before newcomers are admitted (`paused_round` row), so a
test can let every join the service sent arrive first. The ledger also records each
join the worker receives (`join_received`), each done it sends (`done_sent`) and, on a step, the slots it reads that the pool holds as free
(`released_slots`: the Runtime's safety net replaying a request after its executor returned every slot).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class ChatTokenizer:
    """Characters as tokens inside the tiny model's vocabulary; a chat turn renders as its text alone."""

    eos_token_id = 255
    eos_token_ids = {255}

    def encode(self, text):
        return [ord(c) % 200 + 1 for c in text]

    def decode(self, ids):
        return "".join(chr(32 + i % 90) for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        return self.encode(messages[-1]["content"])

    @property
    def detokenizer(self):
        return _Detokenizer()


class _Detokenizer:
    def __init__(self):
        self.pending = ""

    def reset(self):
        self.pending = ""

    def add_token(self, token):
        self.pending += chr(32 + token % 90)

    def finalize(self):
        pass

    @property
    def last_segment(self):
        segment, self.pending = self.pending, ""
        return segment


def tiny_gemma(seed: int = 3):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import gemma3_text

    from tests.engine.test_dense_gemma import TINY

    mx.random.seed(seed)
    model = gemma3_text.Model(gemma3_text.ModelArgs.from_dict(TINY))
    model.sanitize({})
    model.update(nn.utils.tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    return model


def cpu_plan(model, slots: int, capacity: int):
    from ironmule import dense_gemma as dg

    return dg.GemmaDensePlan(model, slots=slots, capacity=capacity, attention=dg.reference_attention, native=False)


def instrument(ledger: str | None, faults: dict, encode) -> None:
    """Record every slot the pool hands out and takes back; inject the faults a test asks for."""
    from ironmule.dense_plan import DenseBackend

    lock = threading.Lock()
    occupied: dict[int, str] = {}
    steps = {"n": 0}
    def key_of(prompt_ids) -> str:
        return hashlib.sha256(json.dumps(list(prompt_ids)).encode()).hexdigest()[:12]

    fail_prompt = tuple(encode(faults["prefill_fail"])) if faults.get("prefill_fail") else None
    poison = {key_of(encode(faults[name])): name for name in ("step_fail", "step_fail_once") if faults.get(name)}
    poisoned: dict[int, str] = {}  # slot -> fault: a fault stays with the request's state, also when the
    # Runtime's safety net replays it in the slot the executor already returned (until another request takes it)
    pauses = {int(step): path for step, path in faults.get("pause", ())}

    def write(row: dict) -> None:
        if ledger:
            with open(ledger, "a") as stream:
                stream.write(json.dumps(row) + "\n")

    original_prefill, original_release, original_steps = (DenseBackend.prefill, DenseBackend.release,
                                                          DenseBackend.steps_group)

    def prefill(self, prompt_ids, plan, capacity):
        if fail_prompt is not None and tuple(prompt_ids) == fail_prompt:
            with self._lock:  # the slot it would take, taken and handed back as `prefill` does on a failure
                free = len(self._free)
            write({"op": "prefill_fault", "free": free})
            raise RuntimeError("injected prefill fault")
        state, first = original_prefill(self, prompt_ids, plan, capacity)
        key = key_of(prompt_ids)
        with lock:
            row = {"op": "take", "slot": state.slot, "prompt": key, "free": self.free_slots()}
            if state.slot in occupied:
                row["violation"] = f"slot {state.slot} taken while {occupied[state.slot]} holds it"
            occupied[state.slot] = key
            if key in poison:
                poisoned[state.slot] = poison[key]
            else:
                poisoned.pop(state.slot, None)
            write(row)
        return state, first

    def release(self, state):
        slot = getattr(state, "slot", None)
        with self._lock:
            held = slot is not None and slot not in self._free
        original_release(self, state)
        if held:
            with lock:
                write({"op": "free", "slot": slot, "prompt": occupied.pop(slot, None), "free": self.free_slots()})

    def steps_group(self, states, tokens, count):
        steps["n"] += 1
        step = steps["n"]
        if step in pauses:
            write({"op": "paused", "step": step, "free": self.free_slots(), "rows": len(states)})
            while not os.path.exists(pauses[step]):
                time.sleep(0.01)
        with lock:
            hit = [s.slot for s in states if s.slot in poisoned]
            for slot in hit:
                if poisoned[slot] == "step_fail_once":
                    poisoned.pop(slot)
                    poison.clear()  # once in the process's life
            released = [s.slot for s in states if s.slot not in occupied]
        if hit:
            write({"op": "step_fault", "step": step, "rows": len(states), "slots": hit})
            raise RuntimeError("injected stacked step fault")
        write({"op": "step", "step": step, "rows": len(states), "slots": [s.slot for s in states], "count": count,
               **({"released_slots": released} if released else {})})
        return original_steps(self, states, tokens, count)

    DenseBackend.prefill, DenseBackend.release, DenseBackend.steps_group = prefill, release, steps_group

    from ironmule.dense_plan import DenseStackedExecutor

    round_pauses = {int(number): path for number, path in faults.get("pause_rounds", ())}
    rounds = {"n": 0}
    original_sweep = DenseStackedExecutor._sweep

    def sweep(self):  # every round of a run starts here, then admits newcomers
        original_sweep(self)
        rounds["n"] += 1
        if rounds["n"] in round_pauses:
            write({"op": "paused_round", "round": rounds["n"], "free": self.backend.free_slots()})
            while not os.path.exists(round_pauses[rounds["n"]]):
                time.sleep(0.01)

    DenseStackedExecutor._sweep = sweep

    from ironmule_product import worker

    class Joins(type(worker.JOINS)):
        def put(self, item, *args, **kwargs):
            write({"op": "join_received", "ids": [r.get("request_id") for r in item.get("requests", ())]})
            return super().put(item, *args, **kwargs)

    worker.JOINS = Joins()
    emit = worker._emit

    def emitted(event):  # after the event is on the pipe: a test waits until its caller has it
        emit(event)
        if isinstance(event, dict) and event.get("type") == "done":
            write({"op": "done_sent", "id": event.get("request_id"), "finish": event.get("finish_reason")})

    worker._emit = emitted


def main(argv: list[str]) -> int:
    import mlx.core as mx
    from mlx_lm import stream_generate

    import friday_evidence.identity as identity_module
    import ironmule.model_identity as model_identity
    from ironmule_product import engine_bridge, worker

    tokenizer = ChatTokenizer()
    model = tiny_gemma()
    capacity = int(os.environ.get("DENSE_CPU_CAPACITY", "96"))
    forced = os.environ.get("DENSE_CPU_SLOTS")

    def install_dense(loaded, device_info, slots=engine_bridge.DENSE_WIDTH):
        slots = int(forced) if forced else slots
        return cpu_plan(loaded, slots, capacity), {"capacity": capacity, "slots": slots, "model_family": "gemma3",
                                                   "kernels": "cpu_reference", "native": None}

    stand_in = {"identity_sha256": "c" * 64, "model_id": "local/test", "revision": "revision"}
    identity_module.runtime_identity = lambda spec, host=None: dict(stand_in)
    identity_module.assert_model_unchanged = lambda spec, identity: None
    model_identity.build_model_identity = lambda model_id, source, revision=None: model_identity.ModelIdentity(
        model_id=model_id, revision=revision or "revision", model_manifest_sha256="c" * 64,
        architecture="gemma3_text (tiny, random, CPU test stand-in)", quantisation_json='{"bits":4}',
        quantisation_sha256=model_identity.canonical_sha256({"bits": 4}), tokenizer_sha256="e" * 64, manifest_file_count=1, manifest_bytes=1,
        tokenizer_file_count=1)
    engine_bridge.install_dense = install_dense
    worker._load = lambda spec, lazy=False: (model, tokenizer, stream_generate, mx.cpu)
    instrument(os.environ.get("DENSE_CPU_LEDGER"), json.loads(os.environ.get("DENSE_CPU_FAULTS") or "{}"),
               tokenizer.encode)
    return worker.main(argv)


if __name__ == "__main__":
    code = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
