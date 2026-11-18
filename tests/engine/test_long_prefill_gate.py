import re

import pytest

from ironmule import long_prefill_gate as gate


class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.answers = {}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and not enable_thinking
        text = "<user>" + messages[0]["content"] + "</user><assistant>"
        return [self.vocab.setdefault(word, len(self.vocab) + 1) for word in re.findall(r"\w+|[^\w\s]", text)]

    def decode(self, ids):
        return self.answers[ids[0]] if ids else ""


class Engine:
    seqs = 4
    seq_ctx = 2048
    prefill_batch = True
    prefill_batch_long = True
    prefill_batch_calls = prefill_batch_prompts = prefill_batch_rows = 0

    def __init__(self, tokenizer, fault=None):
        self.tokenizer = tokenizer
        self.by_ids = {}
        for width in gate.WIDTHS:
            for case, ids in gate.fixtures(tokenizer, width):
                output = 1000 + len(self.by_ids)
                self.by_ids[tuple(ids)] = (output, case)
                tokenizer.answers[output] = case.expected
        tokenizer.answers[-1] = "wrong"
        self.cohorts = []
        self.cold = False
        self.fault = fault

    @staticmethod
    def eog(token):
        return token == 0

    def _clear(self):
        self.cold = True

    def run(self, jobs):
        assert self.cold
        self.cold = False
        mode, width = self.prefill_batch, len(jobs)
        self.cohorts.append((width, mode))
        if mode:
            if self.fault == "exception":
                raise RuntimeError("native failure")
            self.prefill_batch_calls += 2 if self.fault == "split" else 1
            self.prefill_batch_prompts += width
            self.prefill_batch_rows += width * 444
        for job in jobs:
            token, case = self.by_ids[tuple(job.ids)]
            if (self.fault == "uninformative" and case.category == "extraction"
                    or self.fault == "unstable" and not mode and self.cohorts.count((width, False)) == 8 // width + 1):
                token = -1
            job.tokens = [token, 0]
            job.reason = "length" if self.fault == "stop" and mode else "eos"


def test_render_complete_444_token_chats_and_aa_b_witnesses():
    tokenizer = Tokenizer()
    engine = Engine(tokenizer)
    assert len(engine.by_ids) == 16
    assert all(len(ids) == 444 for ids in engine.by_ids)
    report = gate.check(engine, tokenizer)
    assert report["matches"] and report["calls"] == 6
    assert engine.prefill_batch and engine.cold
    for width in gate.WIDTHS:
        record = report["shapes"][f"{width}x444"]
        assert [mode for w, mode in engine.cohorts if w == width] == [False] * (16 // width) + [True] * (8 // width)
        assert record["witnesses"]["candidate"] == [{"calls": 1, "prompts": width, "rows": width * 444}] * (8 // width)
        assert len(record["cases"]) == 8 and all(r["candidate"]["stop"] == "eos" for r in record["cases"])
        assert all(0 < c["reference"] == c["repeat"] == c["candidate"] for c in record["by_category"].values())


@pytest.mark.parametrize(("fault", "reason"), [("unstable", "unstable_reference"), ("split", "grouped_shape_not_proven"),
                                              ("stop", "candidate_differs"), ("uninformative", "uninformative_extraction")])
def test_fail_closed_on_weak_reference_wrong_shape_or_changed_generation(fault, reason):
    tokenizer = Tokenizer()
    engine = Engine(tokenizer, fault)
    report = gate.check(engine, tokenizer)
    assert not report["matches"] and engine.prefill_batch and engine.cold
    assert all(reason in s["reasons"] for s in report["shapes"].values())


def test_restore_after_native_failure_and_reject_unsupported_context():
    tokenizer = Tokenizer()
    engine = Engine(tokenizer, fault="exception")
    with pytest.raises(RuntimeError, match="native failure"):
        gate.check(engine, tokenizer)
    assert engine.prefill_batch and engine.cold
    engine.seq_ctx = 512
    assert gate.check(engine, tokenizer)["calls"] == 0
    engine.prefill_batch_long = False
    assert not gate.check(engine, tokenizer)["matches"]
