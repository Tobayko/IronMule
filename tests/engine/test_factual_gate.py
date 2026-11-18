import pytest

from ironmule import factual_gate as gate


def test_seeded_holdout_is_reproducible_but_fresh_and_grading_requires_the_final_answer():
    first, fresh = gate.cases(), gate.cases(1)
    assert first == gate.cases() and len(first) == 18
    assert first[1].prompt != fresh[1].prompt and first[2].prompt == fresh[2].prompt
    assert gate.correct(first[0], f"<think>something else</think>{first[0].expected}.")
    assert not gate.correct(first[0], f"<think>{first[0].expected}")
    assert not gate.correct(first[0], first[0].expected + "9")
    assert not gate.correct(first[0], first[0].expected + " or 0")
    assert not gate.correct(first[0], "")


def test_incomplete_foreign_or_uninformative_arms_cannot_qualify():
    answers = {c.prompt: c.expected for c in gate.cases()}
    good = gate.evaluate(answers.__getitem__)
    with pytest.raises(ValueError):
        gate.compare(good, good, {})
    with pytest.raises(ValueError):
        gate.compare(good, good, {case.key: True for case in gate.cases(1)})
    bad = dict.fromkeys(good, False)
    assert not gate.compare(bad, bad, bad)["passes"]
    assert gate.compare(good, good, good)["passes"]


def test_losses_cannot_be_hidden_by_gains_in_another_category():
    scores = {case.key: True for case in gate.cases()}
    arithmetic, extraction, _ = gate.cases()[:3]
    baseline = {**scores, arithmetic.key: False}
    candidate = {**scores, extraction.key: False}
    result = gate.compare(baseline, baseline, candidate)
    assert result["reference_correct"] == result["candidate_correct"]
    assert not result["passes"] and not result["by_category"]["extraction"]["passes"]
    # If the repeat itself lost this extraction case, the candidate lies within the reference variation.
    assert gate.compare(baseline, {**baseline, extraction.key: False}, candidate)["passes"]


def test_kg5_shape_uses_32_graded_long_requests_in_one_batch():
    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            return messages[0]["content"].split()

        @staticmethod
        def decode(tokens):
            return tokens[0]

    class Engine:
        def _clear(self):
            pass

        def eog(self, token):
            return False

        def run(self, jobs):
            assert len(jobs) == 32 and len({j.ids[1] for j in jobs}) == 32
            answers = {c.prompt: c.expected for c in gate._protocol(0, True)}
            for job in jobs:
                assert 960 <= len(job.ids) <= 970 and job.limit == gate.MAX_TOKENS
                prompt = " ".join(job.ids[job.ids.index("Question:") + 1:])
                job.tokens = [answers[prompt]]
                job.reason = "eos"
    scores = gate.ggml_scores(Engine(), Tokenizer(), kg5=True)
    assert len(scores) == 32 and gate.compare(scores, scores, scores, kg5=True)["passes"]
    with pytest.raises(ValueError):
        gate.compare(scores, scores, scores)


def test_only_complete_correct_equations_slot_assertions_and_facts_pass():
    arithmetic, extraction, fact = gate.cases()[:3]
    equation = arithmetic.accepted_forms[0]
    wrong_equation = equation.replace(arithmetic.prompt.split()[2], "987", 1)
    assert gate.correct(arithmetic, equation + ".")
    assert gate.correct(arithmetic, equation.replace(" ", ""))
    assert not gate.correct(arithmetic, wrong_equation)
    assert not gate.correct(arithmetic, equation + " or 9")
    slot_answer = extraction.accepted_forms[0]
    assert gate.correct(extraction, slot_answer + ".")
    wrong_slot = slot_answer.split()[3]
    assert not gate.correct(extraction, slot_answer.replace(f" {wrong_slot} ", " X "))
    assert gate.correct(fact, "The Red Planet is known as Mars.")
    assert not gate.correct(fact, "The Red Planet is not known as Mars.")
    assert not gate.correct(fact, "The Red Planet is Mars. The Red Planet is Venus.")
    assert not gate.correct(fact, "Mars ≠ the Red Planet.")
