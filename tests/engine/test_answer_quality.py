import json
import re

import pytest

from lab.answer_quality.evaluate import (
    Turn,
    answer_kind,
    cases,
    evaluate,
    execute,
    tool_call,
)


def test_fixed_split_and_independent_oracles():
    a = cases(20261118, "holdout", 8)
    b = cases(20261118, "development", 8)
    assert a == cases(20261118, "holdout", 8)
    assert not {r["id"] for r in a} & {r["id"] for r in b}
    assert not {r["tool"]["shipment"] for r in a} & {r["tool"]["shipment"] for r in b}
    for row in a:
        for entry in row["cases"]:
            ledger, q = entry["prompt"].removeprefix("Ledger:\n").split("\n\n")
            field, shipment = re.fullmatch(r"What is the (dock|seal) code for shipment (S-\d+)\?", q).groups()
            evidence = [p for line in ledger.splitlines() if line.split(" | ")[0] == shipment for p in line.split(" | ")[1:]
                        if p.split("=")[0] == field]
            assert entry["expected"] == (evidence[0].split("=")[1] if evidence else None)


def test_tool_interaction_requires_actual_path():
    fixtures = cases(1, "holdout", 2)
    datasets = {r["tool"]["shipment"]: r["tool"] for r in fixtures}
    def generate(conversations, tools):
        turns = []
        for messages in conversations:
            if tools is None:
                text = "UNKNOWN"
            else:
                shipment = messages[1]["content"].split()[-1][:-1]
                table = datasets[shipment]
                if shipment == fixtures[0]["tool"]["shipment"]:
                    text = table["docks"][table["depots"][shipment]]  # Guess with no tool returns.
                elif len(messages) == 2:
                    text = json.dumps({"name": "shipment_depot", "arguments": {"shipment": shipment}})
                elif len(messages) == 4:
                    depot = json.loads(messages[-1]["content"])["depot"]
                    text = '<tool_call>' + json.dumps({"name": "depot_dock", "arguments": {"depot": depot}}) + '</tool_call>'
                else:
                    text = json.loads(messages[-1]["content"])["dock"]
            turns.append(Turn(text, "eos", 1))
        return turns
    result = evaluate(generate, fixtures)
    assert result["tools"] == {"total": 2, "first_call_correct": 1, "both_calls_correct": 1, "correct_final": 2,
                               "end_to_end_correct": 1, "false_abstentions": 0, "model_turns": 4, "model_tokens": 4,
                               "tool_executions": 2, "withheld_after_budget": 0, "wrong_final_codes": 0}
    assert result["unsupported"]["correct_abstentions"] == result["supported"]["false_abstentions"] == 2
    with pytest.raises(ValueError, match="missing model responses"):
        evaluate(lambda messages, tools: [], fixtures)


def test_strict_grading_and_bad_tool_calls():
    assert answer_kind("D-84010.", "D-84010") == "correct"
    assert answer_kind("D-84010", None) == "unsupported_value"
    assert answer_kind("UNKNOWN but D-84010", None) == "other_text"
    assert answer_kind("D-84010 is wrong", "D-84010") == "other_text"
    assert answer_kind("Not provided.", None) == "abstained"
    assert tool_call("{} trailing") is None
    table = cases(1, "development", 1)[0]["tool"]
    for call in ({"name": "shipment_depot", "arguments": {"shipment": "other"}},
                 {"name": "shipment_depot", "arguments": {"depot": "wrong"}},
                 {"name": "unlisted", "arguments": {}}):
        assert "error" in execute(call, table)


def test_semantic_grader_rejects_misattribution_and_contradictory_wrappers():
    kw = {"shipment": "S-12345", "field": "dock"}
    assert answer_kind("The dock code for shipment **S-12345** is **D-00421**.", "D-00421", **kw) == "correct"
    assert answer_kind("UNKNOWN. There is no entry for this shipment in the provided ledger.", None, **kw) == "abstained"
    assert answer_kind("The dock code for shipment S-12345 is D-00321.", None, **kw) == "unsupported_value"
    assert answer_kind("The seal code for shipment S-12345 is D-00421.", "D-00421", **kw) == "wrong_reference"
    assert answer_kind("The dock code for shipment S-11111 is D-00421.", "D-00421", **kw) == "wrong_reference"
    assert answer_kind("UNKNOWN. But D-00421 is likely.", None, **kw) == "other_text"
    assert answer_kind("The dock code for shipment S-12345 is not D-00421.", "D-00421", **kw) == "other_text"


def test_evidence_condition_withholds_even_a_lucky_oracle_guess_without_a_return():
    fixture = cases(83, "development", 1)
    table = fixture[0]["tool"]
    answer = table["docks"][table["depots"][table["shipment"]]]
    def guess(conversations, tools):
        return [Turn(answer, "eos", 3) for _ in conversations]
    ref = evaluate(guess, fixture, "reference")
    guarded = evaluate(guess, fixture, "evidence_first")
    assert ref["tools"]["correct_final"] == 1 and ref["tools"]["end_to_end_correct"] == 0
    assert guarded["tools"]["correct_final"] == 0 and guarded["tools"]["false_abstentions"] == 1
    assert guarded["tools"]["withheld_after_budget"] == 1 and guarded["tools"]["model_turns"] == 4
    assert guarded["tool_traces"][0]["turns"][0]["response"] == answer
