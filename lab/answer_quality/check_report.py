"""Reconstruct both answer-quality scorecards from recorded model turns, without loading the model."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from . import evaluate, structured


def verify(report: dict) -> None:
    if report["protocol"] != evaluate.PROTOCOL:
        raise ValueError("unsupported rubric")
    fixtures = evaluate.cases(report["seed"], report["split"], report["groups"])
    if hashlib.sha256(json.dumps(fixtures, sort_keys=True).encode()).hexdigest() != report["cases_sha256"]:
        raise ValueError("fixture digest differs")
    study = report.get("study")
    if study not in (None, structured.STUDY):
        raise ValueError("unsupported study")
    conditions = {"reference", "structured"} if study else {"reference", "evidence_first"}
    if set(report["conditions"]) != conditions:
        raise ValueError("missing paired condition")
    for label, record in report["conditions"].items():
        _verify_condition(fixtures, label, record)


def _verify_condition(fixtures, label, record):
    by_prompt = {r["prompt"]: r for r in record["ledger_cases"]}
    by_question = {f"Which dock is assigned to shipment {r['shipment']}?": r for r in record["tool_traces"]}
    positions = Counter()

    def replay(conversations, tools):
        turns = []
        for messages in conversations:
            question = messages[1]["content"]
            if tools is None:
                row = by_prompt[question]
            else:
                row = by_question[question]["turns"][positions[question]]
                positions[question] += 1
            turns.append(evaluate.Turn(row["response"], row["finish"], row["tokens"]))
        return turns

    scores = structured.assess(replay, fixtures) if label == "structured" else evaluate.evaluate(replay, fixtures, label)
    if scores != record:
        raise ValueError(f"recorded grades or tool returns differ in {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    verify(json.loads(args.report.read_bytes()))
    print("Both scorecards and tool trajectories reproduced from the saved model responses.")
