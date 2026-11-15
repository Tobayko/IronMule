"""DM1: the draft model's stopping rule and its KV bookkeeping, with a stand-in engine (no llama.cpp)."""

import types

from ironmule.draft_model import Drafter, DraftStop


def test_the_rule_drafts_alone_and_stops_when_drafts_are_unlikely_or_calls_dear():
    stop = DraftStop()
    stop.by_rows = {1: 0.080, 4: 0.088}  # a step, and what three more rows add
    stop.call = {1: 0.016}
    assert stop.start(1) and abs(stop.c_row - 0.008 / 3) < 1e-9
    assert stop.more(q=0.9, expected=0.9, spent=0.02, running=1, sharers=1)  # a confident draft: go on
    assert not stop.more(q=0.05, expected=1.2, spent=0.05, running=1, sharers=1)  # the chain is likely broken
    stop.call = {1: 0.200}  # a draft model as slow as the step itself
    assert not stop.start(1)


def test_calibration_is_learned_from_what_the_target_accepted():
    stop = DraftStop()
    for _ in range(200):
        stop.learn([0.95, 0.95, 0.35], 2)  # two confident drafts accepted, the unsure third not
    assert stop.p(0.95) > 0.95 and stop.p(0.35) < 0.05 and stop.drafted == 600 and stop.accepted == 400


def test_drafting_that_measured_slower_than_plain_steps_stops_and_is_probed_again():
    stop = DraftStop()
    stop.by_rows, stop.call = {1: 0.080}, {1: 0.016}
    for _ in range(stop.TRUST):
        stop.observe_step(0.090, 1, 3, tokens=1, draft_seconds=0.050)  # one token in 140 ms: slower than 80 ms
    asked = [stop.start(1) for _ in range(stop.PROBE)]
    assert asked.count(True) == 1  # only the probe


class Engine:
    """Draft model stand-in: the next token is always the last token + 1, with a given confidence."""

    def __init__(self, seqs=2, confidence=0.9):
        self.seqs, self.confidence, self.memory, self.calls, self.removed = seqs, confidence, None, [], []
        self.L = types.SimpleNamespace(llama_memory_seq_rm=lambda m, s, a, b: self.removed.append((s, a, b)) or True)

    def _decode_conf(self, rows):
        self.calls.append(list(rows))
        return [(token + 1, self.confidence) for token, _, _, logits in rows if logits]

    def eog(self, token):
        return False


def prepared(confidence=0.9):
    drafter = Drafter(Engine(confidence=confidence))
    drafter.stop.by_rows, drafter.stop.call = {1: 0.080, 4: 0.088}, {1: 0.016, 2: 0.018}
    for _ in range(300):  # what this machine's target has taught: confident drafts hold, unsure ones do not
        drafter.stop.learn([0.9, 0.9, 0.9], 3)
        drafter.stop.learn([0.15], 0)
    return drafter


def test_the_drafter_keeps_its_cells_in_step_with_the_text():
    drafter = prepared()
    out = drafter.draft({7: ([10, 11, 12], 4)}, running=1)
    assert out[7] == [13, 14, 15, 16]  # four confident drafts, as much as there was room for
    slot = drafter.slot_of[7]
    assert drafter.held[slot] == [10, 11, 12, 13, 14, 15]  # the last draft was not fed to the draft model
    drafter.settle(7, used=4, accepted=1)  # the target took 13 and then wrote 99
    assert drafter.held[slot] == [10, 11, 12, 13] and (slot, 4, -1) in drafter.e.removed
    drafter.draft({7: ([10, 11, 12, 13, 99], 2)}, running=1)
    assert drafter.e.calls[-2][0][:2] == (99, 4)  # only the new token is caught up, at its position
    drafter.release(7)
    again = drafter.draft({8: ([10, 11, 12, 13, 50], 1)}, running=1)
    assert drafter.slot_of[8] == slot and again[8] == [51]  # a new request reuses the slot's shared prefix


def test_no_drafts_under_load_or_without_a_free_slot():
    drafter = prepared()
    assert drafter.draft({1: ([5, 6], 3)}, running=3) == {}  # beyond max_running the draft model rests
    drafter.draft({1: ([5, 6], 1), 2: ([7, 8], 1)}, running=2)
    assert drafter.draft({3: ([9, 9], 1)}, running=1) == {}  # both slots are taken


def test_unsure_drafts_end_the_chain_early():
    drafter = prepared(confidence=0.15)
    out = drafter.draft({1: ([5, 6], 6)}, running=1)
    assert len(out[1]) == 1  # the first draft is computed and offered (a row is cheap); no second call
